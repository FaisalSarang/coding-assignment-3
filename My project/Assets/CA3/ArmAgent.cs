using System.Collections.Generic;
using Mujoco;
using UnityEngine;

// Thin wrapper around the imported MuJoCo arm.
//   state  = [q (JointCount), qd (JointCount)]   -> StateSize = 2 * JointCount
//   action = joint-position targets, one per actuator (the UR5e actuators are
//            position servos, so ctrl is literally "the angle I want").
//
// Execution order: controllers (-10 / -5) write an action with SetAction, this
// script (-2) pushes it into MuJoCo, RolloutRecorder (-1) logs it, and MjScene (0)
// steps the physics last. So every script in a FixedUpdate sees the same pre-step state.
[DefaultExecutionOrder(-2)]
public class ArmAgent : MonoBehaviour
{
    public int JointCount;
    public int StateSize;
    public int ActionSize;

    public MjActuator gripper;          // UR5e has no gripper; leave empty
    public MjBody endEffectorBody;      // wrist_3_link
    public Transform endEffector;       // attachment_site (or wrist_3_link) transform
    public MjHingeJoint[] joints;
    public MjActuator[] actuators;

    [Header("Auto-wiring (used when the arrays above are empty)")]
    public string[] jointNames =
    {
        "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
        "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
    };
    public string[] actuatorNames =
    {
        "shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3",
    };
    public string endEffectorBodyName = "wrist_3_link";
    public string endEffectorSiteName = "attachment_site";

    [Header("Control rate")]
    [Tooltip("Physics steps per control step. Physics runs at Time.fixedDeltaTime (set it to 0.002: " +
             "the UR5e servos ring numerically at larger steps); everything learned runs at " +
             "fixedDeltaTime * controlEvery = 0.02 s (50 Hz).")]
    public int controlEvery = 10;

    float[] action;
    float[] state;
    bool ready;

    public float[] CurrentAction => action;
    public bool Ready => ready;

    // True on the physics steps where controllers act, the recorder logs and the world
    // model advances. Time.fixedTime is constant during one FixedUpdate pass, so every
    // script agrees on it regardless of execution order.
    public bool IsControlStep =>
        controlEvery <= 1 ||
        Mathf.RoundToInt(Time.fixedTime / Time.fixedDeltaTime) % controlEvery == 0;

    public float ControlDt => Time.fixedDeltaTime * Mathf.Max(1, controlEvery);

    void Awake()
    {
        InitializeJointsAndActuators();
    }

    void FixedUpdate()
    {
        if (!ready)
            return;
        ApplyAction();
    }

    public void InitializeJointsAndActuators()
    {
        if (joints == null || joints.Length == 0)
        {
            var found = new List<MjHingeJoint>();
            foreach (var jointName in jointNames)
            {
                var joint = FindComponent<MjHingeJoint>(jointName);
                if (joint != null)
                    found.Add(joint);
            }
            // Fallback: every hinge under this object, in hierarchy order.
            joints = found.Count == jointNames.Length
                ? found.ToArray()
                : GetComponentsInChildren<MjHingeJoint>();
        }

        if (actuators == null || actuators.Length == 0)
        {
            var found = new List<MjActuator>();
            foreach (var actuatorName in actuatorNames)
            {
                var actuator = FindComponent<MjActuator>(actuatorName);
                if (actuator != null)
                    found.Add(actuator);
            }
            if (found.Count != actuatorNames.Length)
            {
                // Actuators are usually created under a sibling "actuators" object, so
                // search the whole scene and drop the gripper if one is assigned.
                found.Clear();
                foreach (var actuator in FindObjectsByType<MjActuator>(FindObjectsSortMode.None))
                    if (actuator != gripper)
                        found.Add(actuator);
                found.Sort((a, b) => a.transform.GetSiblingIndex().CompareTo(b.transform.GetSiblingIndex()));
            }
            actuators = found.ToArray();
        }

        if (endEffectorBody == null)
            endEffectorBody = FindComponent<MjBody>(endEffectorBodyName);
        if (endEffector == null)
        {
            var site = FindComponent<MjSite>(endEffectorSiteName);
            endEffector = site != null ? site.transform
                : endEffectorBody != null ? endEffectorBody.transform : null;
        }

        JointCount = joints.Length;
        ActionSize = actuators.Length;
        StateSize = 2 * JointCount;

        if (JointCount == 0 || ActionSize == 0 || ActionSize != JointCount)
        {
            Debug.LogError(
                $"ArmAgent: found {JointCount} joints and {ActionSize} actuators. " +
                "Wire them in the Inspector (they must match one-to-one).");
            ready = false;
            return;
        }

        state = new float[StateSize];
        action = new float[ActionSize];
        // Until someone sets an action, hold the configuration the scene starts in.
        for (int i = 0; i < ActionSize; i++)
            action[i] = actuators[i].Control;
        ready = true;
    }

    public void SetAction(float[] targets)
    {
        if (!ready || targets == null)
            return;
        int n = Mathf.Min(targets.Length, ActionSize);
        for (int i = 0; i < n; i++)
            action[i] = targets[i];
        ApplyAction();
    }

    public unsafe void ApplyAction()
    {
        if (!ready)
            return;
        // MjActuator.OnSyncState copies Control into data->ctrl after every step,
        // so Control is the source of truth. We also write ctrl directly so the
        // command takes effect on the very next mj_step.
        bool live = MjScene.InstanceExists && MjScene.Instance.Data != null;
        for (int i = 0; i < ActionSize; i++)
        {
            actuators[i].Control = action[i];
            if (live && actuators[i].MujocoId >= 0)
                MjScene.Instance.Data->ctrl[actuators[i].MujocoId] = action[i];
        }
    }

    public unsafe float[] Observe()
    {
        if (!ready)
            return state;
        if (!MjScene.InstanceExists || MjScene.Instance.Data == null)
            return state;

        var data = MjScene.Instance.Data;
        for (int i = 0; i < JointCount; i++)
        {
            state[i] = (float)data->qpos[joints[i].QposAddress];
            state[JointCount + i] = (float)data->qvel[joints[i].DofAddress];
        }
        return state;
    }

    public void CopyState(float[] destination)
    {
        var s = Observe();
        int n = Mathf.Min(destination.Length, s.Length);
        System.Array.Copy(s, destination, n);
    }

    private T FindComponent<T>(string name) where T : Component
    {
        if (string.IsNullOrEmpty(name))
            return null;
        // Prefer children of this arm, then anything in the scene (actuators live elsewhere).
        foreach (var component in GetComponentsInChildren<T>(true))
            if (component.name == name)
                return component;
        foreach (var component in FindObjectsByType<T>(FindObjectsInactive.Include, FindObjectsSortMode.None))
            if (component.name == name)
                return component;
        return null;
    }
}
