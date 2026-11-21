# CA3 Report
## Faisal Riyaz Sarang

> All numbers below were measured in the Unity project (`My project/`, scene `Assets/CA3/CA3_UR5e.unity`): 70,000
> transitions recorded by `RolloutRecorder` in Play mode, models trained with `train.py` / `train_controller.py`, and
> results measured in Play mode by `CA3 > Run: Verify Controller + IK` (log: `Assets/CA3/Data/verify_log.txt`).

**Name:**
**Unity version:** 6000.3.24f1
**MuJoCo plugin version:** 3.17.0
**Chosen arm (Menagerie):** Universal Robots UR5e (`universal_robots_ur5e`)
**Gallery / repo link:** https://github.com/google-deepmind/mujoco_menagerie/tree/main/universal_robots_ur5e

## Why this arm
The UR5e is a widely used industrial 6-DOF arm with clean MJCF: six hinge joints, six position-servo actuators whose
`ctrl` is directly the target angle, and a `home` keyframe. That makes the joint-target action space natural. It also
differs from the lecture's 7-DOF Panda: it has no redundant joint and no gripper, and its joint ranges are ±2π.

## Pipeline summary
`ArmJointController` samples random joint goals around home, and `RolloutRecorder` logs 70k control-step transitions
`(q, qd, q*) → (q', qd')`. `train.py` fits an MLP world model that predicts the state change and exports it to ONNX for
`WorldModelDriver`, which draws a ghost arm and a 100-step forecast. `train_controller.py` trains a linear
policy entirely inside that world model with CMA-ES. In Unity, `IkGoalProvider` turns a draggable sphere into joint goals
with damped-least-squares IK, and the learned controller tracks them.

## World model results
- DOF (`ACTION`): 6 (STATE = 12)
- Architecture: input normalized `[q, qd, q*]` (18) plus the servo error `q* − q` computed inside the network. Then
  3 × 256 SiLU MLP → normalized Δ`[q, qd]` (12). Loss = one-step MSE + 0.5 × 10-step open-loop rollout MSE.
- Data: 70,000 control-step transitions from Unity (23 min of sim time), last 10% held out.
- One-step joint RMSE: **0.0025 rad** (copy-last-state baseline 0.030 rad, 12× worse)
- Open-loop 10 / 50 / 100-step RMSE: **0.032 / 0.079 / 0.089 rad**. Drift grows over the rollout but levels off; the
  held-out segment includes a stretch where even the real arm tracked poorly (probably a goal touching the floor).
- In Play mode the ghost arm stays within **0.005 rad** RMS of the real arm (mean; median 0.001 rad).

## Controller results
- Policy: `raw = clip(goal + W[q − goal, qd] + b)`, then low-pass (α = 0.3) and rate limit (0.1 rad/step).
  CMA-ES searches per-joint gains (one position gain and one velocity gain per joint, 12 params, `b` fixed at 0) that are
  expanded into the full `W` (6×12). 60 generations × 64 imagined episodes of 200 steps.
- Imagined return (CMA-ES): open-loop (θ = 0) **−0.165 → learned −0.156**
- In Unity, the same 120 s sequence of random goals (fixed seed), mean RMS joint error to goal:
  smoothed open-loop baseline (θ = 0) **0.082 rad → learned controller 0.073 rad** (11% lower; median unchanged at
  0.006 rad, so there is no steady-state offset).
- Learned gains: velocity gains of +0.05 to +0.11 on every joint make the command lead in the direction of motion,
  which compensates for the overdamped UR5e servos, so the arm arrives sooner.

## WorldModelDriver
- **Predict:** normalize `[s, a]` with `norm.json`, run `dynamics.onnx` on the CPU worker, de-normalize Δ, and add it to `s`.
- **Reanchor:** the ghost snaps back to the real state every `ghostInterval` steps or when its RMS joint error exceeds
  0.2 rad. In between, `softCorrection` (5%) blends it gently toward reality.
- **Forecast:** every 10 control steps, imagine 100 steps ahead. With the controller active, the policy is re-queried at
  each imagined state, so the forecast is a closed-loop dream. The end-effector path comes from MuJoCo FK on the imagined
  joints, and the live state is restored afterwards.

## IK + live target
- IK method: damped least squares (Levenberg–Marquardt), position only, with a numeric Jacobian from MuJoCo
  `mj_kinematics` on a private scratch `mjData`, a null-space pull toward home, warm-starting from last frame's solution,
  and 15 iterations per control step.
- Gravity-sag correction: the servos settle about 0.008 rad short under gravity, which put the hand 1.4 cm off every
  target. An outer Cartesian integral loop moves the IK aim point by 5% of the remaining hand error per control step
  (clamped to 5 cm, and paused when the target is unreachable).
- Results in Unity (learned controller tracking the IK goals):
  - 3 stationary targets (settled): hand-to-target **0.8 mm** mean (1.4 cm without sag correction). The IK residual is 0.8 mm.
  - Target moving on a loop at about 0.1 m/s: **1.0 mm median**, 1.2 cm mean, 5.8 cm p95. The large errors are lag
    while the target moves fastest.
- Workspace / failure cases: goals are clamped to the joint box the controller was trained on (home ± 0.15 × range), so
  targets far from home are only approached, and the gizmo turns yellow.

## Things that went wrong and how they were fixed
1. **Fixed Timestep 0.02 makes the servos ring.** Joint velocities flipped sign every step (−5.0, +1.3, −3.0, …), and the
   world model could only reach 2× better than baseline. Fix: run physics at 0.002 s and control every 10 physics steps
   (`ArmAgent.controlEvery`). One-step RMSE fell from 0.045 to 0.0016 rad.
2. **`randomScale = 0.35` puts about 36% of UR5e goals in floor or self-contact**, because the ±2π ranges are huge. Contacts
   make the dynamics discontinuous. 0.15 gives about 4% contact.
3. **CMA-ES could not beat open-loop** with the original penalties (velocity 0.05, rate limit 0.05). The rate limit
   decided the speed, and hand-tuned gains confirmed that open-loop was already optimal there. Searching all 78 weights from
   zero also failed, because random cross-joint terms dominate. A lighter velocity penalty and the per-joint
   parameterization fixed it.
4. **The controller exploited the world model.** With a free bias, CMA-ES learned a constant −0.03 to −0.04 rad offset
   on the shoulder and elbow. It looked better in imagination, but in Unity it made tracking worse (0.094 vs 0.080 rad)
   and left the hand 3 cm off every IK target. Fixing `b = 0` removed the exploit, and the learned controller then beat
   the baseline in Unity.
5. **A fair baseline matters.** My first comparison used raw step commands as "open-loop", which isn't what the policy
   competes against. The fair baseline is the same smoothed controller with θ = 0 (`WorldModelController.baselineOpenLoop`).
6. **Unity recompiled mid-Play** ("define symbols changed" right after startup). The domain reload stopped data
   collection at 46k rows. The batch harness now waits for compilation to settle before entering Play mode.

## Design question
*Why can rollout collection stay in joint space while deployment tracks a Cartesian target?*
The world model and the controller only ever deal with joint states and joint-target actions. Dynamics live in joint
space: torques, inertia and the servos all act per joint. Cartesian position is just a deterministic, known function of
the joints (forward kinematics), so it never needs to be learned. At deployment, IK converts the Cartesian target into a
joint goal inside the same region the data covered, and from then on it is the same joint-goal tracking problem the
controller was trained on. The only requirement is that the IK goals stay inside the training distribution, which is why
`IkGoalProvider` clamps to the training range.

---

## Reflection

**1. Hardest and easiest parts**
The main difficulty was how to make the world model to accurately learn the dynamics. The initial model worked two times better than just copying the previous state. However, it turned out that the problem lay not in the architecture of the network, but in the quality of the training data. At the default timestep of 0.02s in Unity, the UR5e position servos underwent numerical oscillations due to stiffness of the joints. This made the velocities of the joints switch directions on each step. I found the problem after I checked the autocorrelation of the `qd` at lag 1 and found out that it was negative. I solved this problem by reducing the timestep in MuJoCo to 0.002s and applying actions every 10 physics steps. This increased the accuracy of one-step predictions 12 times.

The most trivial part of the implementation is to write ArmAgent since the plugin had access to such things as `QposAddress`, `DofAddress`, and `MjActuator.Control`.

**2. How the world model works**
The data are gathered once per 0.02 seconds control loop. Each row includes values of current joint positions and velocities, target joint actions, and resulting joint positions and velocities, amounting to 30-dimensional space for each of six joints.

The target goals are randomly chosen near the home position, and the goals are achieved by applying steps/ramps with some randomness in order to gather diverse commands that the controller will have to deal with in future.

The used model is a three-layer MLP utilizing SiLU activation, with 256 units per layer. The normalized state/action, as well as the servo error `q* - q`, are used as input features of the model. I compute the servo error within the model since it correlates with the torque of the servos. I do not use state prediction from my model but the normalized state change prediction instead. The predicted value is summed with the input state to get the actual prediction. One-step MSE loss function and a 10-step open-loop loss function are employed to train the model.

**3. How the controller works**
Target position for each joint is computed according to the following linear formula that depends on the distance from the goal and the speed of the joint:

`goal + W·[q − goal, qd] + b`

This value is subsequently smoothed and rate-limited to be transmitted to actuators. Training process is conducted only in imagination mode, when CMA-ES produces different sets of parameters that are evaluated during 64 episodes of 200 steps each using the learned dynamics model. The objective function in this case measures errors in terms of joint deviation, velocity, and jerky commands.

In a Unity environment, `WorldModelController` works with similar formulas to compute the goals based on the real state of each joint at each control period. After that, these values are transferred to the `ArmAgent`.

**4. What the IK solver does**
`IkGoalProvider` uses world coordinates of the sphere and returns joint angles which can generate the corresponding end-effector location using the damped least square approach. The calculation of the Jacobian happens through finite differences, where the forward kinematics of MuJoCo operates on the scratch `mjData`.

The null space is needed for keeping the robot's posture near the home state, whereas the warm starting ensures the elbow joint won't be flipping between the possible solutions.

It is crucial to mention that the method of goal assignment (`SetTarget` in `ArmJointController`) happens exactly at this stage. Therefore, there is nothing special that could distinguish these two cases for the controller.

**5. How my understanding of world models changed**
At first, the question about the proper choice of the network architecture and its tuning seemed to be crucial for the work with world models. In practice, however, I invested a lot of efforts into solving the issue with the data, which appeared to play a much more significant role for the results than the neural net architecture. I learned that the quality of my data largely depended upon parameters like the timestep, contact dynamics, action distribution etc.

Additionally, I discovered that even a very high level of accuracy in one-step predictions doesn't guarantee that model is going to do well in a long rollout. The most important aspect while training a controller is the pace of accumulation of error during the prediction process, which explains why I implemented the multi-step loss.

The last thing I learned from my experience is that a controller trying to optimize its behavior in imagination takes advantage of penalties and constraints rather than learning the real dynamics. At first, I had troubles competing with open-loop controller using CMA-ES until I understood that the speed constraint is more important than the policy itself.

**6. Term project**
My term project has been combined with one another term project. Thus, we now have a simulation of an apartment which contains only one virtual roommate which is a humanoid robot. That humanoid can make gestures of pointing on objects and performing actions like washing of dishes and cleaning the room.

In the majority of cases, the above-described tasks can be reduced to a relatively simple command which instructs the robot to perform some manipulations with its hand which can be expressed through such action like "Move your hand to such point". Therefore, the components we've developed for this assignment may be used in the project. The action will specify Cartesian targets –sink, table or whatever pointed by a human. Then the `IkGoalProvider` will determine joint targets using the damped least squares solver and the controller will follow those targets thus making required movement.

Also it's possible to debug our project by using of the recorder component. We'll be able to log each chore trial and then play the failed ones in order to understand the reason of the failure: whether it's caused by IK solver, controller or the physics simulation.