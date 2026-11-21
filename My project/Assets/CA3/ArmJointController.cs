using UnityEngine;

// Owns the joint-space GOAL for the arm.
//  - sampleRandomGoals = true : picks a new random goal every ~holdSeconds
//    (data collection, and Step 3 "feed it the random targets" verification).
//  - sampleRandomGoals = false: the goal is written from outside (IkGoalProvider.SetTarget).
//  - emitActions = true       : open-loop tracking, the goal is sent straight to the
//    position actuators. WorldModelController turns this off and sends its own actions.
[DefaultExecutionOrder(-10)]
public class ArmJointController : MonoBehaviour
{
    public ArmAgent agent;
    public float holdSeconds = 3f;

    // From ur5e.xml: every joint is +-2pi except the elbow (size3_limited) which is +-pi.
    public Vector2[] jointRanges =
    {
        new Vector2(-6.28319f, 6.28319f), // shoulder_pan_joint
        new Vector2(-6.28319f, 6.28319f), // shoulder_lift_joint
        new Vector2(-3.1415f, 3.1415f),   // elbow_joint
        new Vector2(-6.28319f, 6.28319f), // wrist_1_joint
        new Vector2(-6.28319f, 6.28319f), // wrist_2_joint
        new Vector2(-6.28319f, 6.28319f), // wrist_3_joint
    };

    // ur5e.xml <key name="home">
    public float[] homeTargets = { -1.5708f, -1.5708f, 1.5708f, -1.5708f, -1.5708f, 0f };

    [Range(0f, 1f)]
    public float randomScale = 0.15f;   // 0.35 puts ~36% of UR5e goals in floor/self contact

    public bool emitActions = true;
    public bool sampleRandomGoals = true;
    public bool startAtHome = true;

    [Header("Data-collection diversity (only used while emitActions)")]
    [Tooltip("Chance that a new goal is approached with a rate-limited ramp instead of a step.")]
    [Range(0f, 1f)] public float rampProbability = 0.5f;
    [Tooltip("Max change of the commanded angle per step while ramping (rad).")]
    public float rampMaxDelta = 0.05f;
    [Tooltip("Gaussian noise added to every emitted command (rad).")]
    public float actionNoise = 0.02f;

    float[] target;
    float[] command;
    float[] noisy;
    float hold;
    bool ramping;
    bool ready;
    bool needsTeleport;

    public float[] CurrentTarget => target;

    void Start()
    {
        if (agent == null)
            agent = GetComponent<ArmAgent>();
        if (agent == null || !agent.Ready)
        {
            Debug.LogError("ArmJointController needs a ready ArmAgent.");
            enabled = false;
            return;
        }

        int n = agent.ActionSize;
        if (homeTargets == null || homeTargets.Length != n)
            homeTargets = new float[n];
        if (jointRanges == null || jointRanges.Length != n)
        {
            jointRanges = new Vector2[n];
            for (int i = 0; i < n; i++)
                jointRanges[i] = new Vector2(-Mathf.PI, Mathf.PI);
        }

        target = new float[n];
        command = new float[n];
        PlaceTarget(true);
        System.Array.Copy(target, command, n);
        // MjScene builds the model in its own Start (after this one), so jump to the
        // home pose on the first FixedUpdate, when qpos addresses are valid.
        needsTeleport = startAtHome;
        hold = holdSeconds;
        ready = true;
        if (emitActions)
            agent.SetAction(command);
    }

    void FixedUpdate()
    {
        if (!ready)
            return;

        if (needsTeleport && TeleportToHome())
            needsTeleport = false;

        // Goals and commands only change on control steps (50 Hz); the command is held
        // by the position servos for the physics sub-steps in between.
        if (!agent.IsControlStep)
            return;

        if (sampleRandomGoals)
        {
            hold -= agent.ControlDt;
            if (hold <= 0f)
            {
                PlaceTarget(false);
                // Vary the hold a bit so the data contains both settling and mid-motion switches.
                hold = holdSeconds * Random.Range(0.5f, 1.5f);
                ramping = Random.value < rampProbability;
                if (!ramping)
                    System.Array.Copy(target, command, target.Length);
            }
        }

        if (!emitActions)
            return;

        for (int i = 0; i < command.Length; i++)
        {
            if (ramping)
                command[i] += Mathf.Clamp(target[i] - command[i], -rampMaxDelta, rampMaxDelta);
            else
                command[i] = target[i];
        }

        if (noisy == null || noisy.Length != command.Length)
            noisy = new float[command.Length];
        for (int i = 0; i < command.Length; i++)
        {
            float noise = actionNoise > 0f ? actionNoise * Gaussian() : 0f;
            noisy[i] = Mathf.Clamp(command[i] + noise, jointRanges[i].x, jointRanges[i].y);
        }
        agent.SetAction(noisy);
    }

    // Writes the current joint GOAL (not the noisy command) into destination.
    public void ActionFrom(float[] destination)
    {
        if (target == null)
            return;
        System.Array.Copy(target, destination, Mathf.Min(destination.Length, target.Length));
    }

    // Same sampling rule as sample_goal() in train_controller.py:
    // goal = home + U(-span, span), span = 0.5 * (hi - lo) * randomScale, clipped to range.
    public void PlaceTarget(bool home)
    {
        if (target == null)
            return;
        for (int i = 0; i < target.Length; i++)
        {
            float value = homeTargets[i];
            if (!home)
            {
                float span = 0.5f * (jointRanges[i].y - jointRanges[i].x) * randomScale;
                value += Random.Range(-span, span);
            }
            target[i] = Mathf.Clamp(value, jointRanges[i].x, jointRanges[i].y);
        }
    }

    // Restart the random goal sequence from a fixed seed (used to compare controllers on
    // identical goals). A new goal is drawn on the next control step.
    public void RestartRandomGoals(int seed)
    {
        Random.InitState(seed);
        hold = 0f;
    }

    // External goal source (IkGoalProvider).
    public void SetTarget(float[] goal)
    {
        if (target == null || goal == null)
            return;
        for (int i = 0; i < target.Length && i < goal.Length; i++)
            target[i] = Mathf.Clamp(goal[i], jointRanges[i].x, jointRanges[i].y);
        if (emitActions)
            System.Array.Copy(target, command, target.Length);
    }

    unsafe bool TeleportToHome()
    {
        // The importer starts every joint at qpos = 0; jump to the MJCF home keyframe.
        if (!Mujoco.MjScene.InstanceExists || Mujoco.MjScene.Instance.Data == null)
            return false;
        var scene = Mujoco.MjScene.Instance;
        for (int i = 0; i < agent.JointCount; i++)
        {
            scene.Data->qpos[agent.joints[i].QposAddress] = homeTargets[i];
            scene.Data->qvel[agent.joints[i].DofAddress] = 0;
        }
        Mujoco.MujocoLib.mj_forward(scene.Model, scene.Data);
        return true;
    }

    static float Gaussian()
    {
        float u1 = 1f - Random.value;
        float u2 = Random.value;
        return Mathf.Sqrt(-2f * Mathf.Log(u1)) * Mathf.Cos(2f * Mathf.PI * u2);
    }
}
