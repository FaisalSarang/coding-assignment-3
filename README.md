# CA3 Report


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