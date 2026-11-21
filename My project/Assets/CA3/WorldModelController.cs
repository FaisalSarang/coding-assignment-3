using System;
using UnityEngine;


[DefaultExecutionOrder(-5)]
public class WorldModelController : MonoBehaviour
{
    public ArmAgent agent;
    public ArmJointController goals;
    public TextAsset controllerJson;

    [Range(0.0f, 1f)]
    [Tooltip("Weight on the new raw action. Overwritten by controller.json so it matches training.")]
    public float actionSmoothing = 0.2f;

    public float maxDeltaPerStep = 0.05f;

    [Tooltip("Ablation: ignore the learned W and b (raw action = goal), keeping the same smoothing. " +
             "This is the theta = 0 baseline CMA-ES started from.")]
    public bool baselineOpenLoop = false;

    float[] W;
    float[] b;
    float[] jointLo;
    float[] jointHi;

    float[] features;
    float[] goalBuffer;
    float[] rawAction;
    float[] action;
    bool ready;
    bool hasAction;
    bool tookOverEmitActions;
    bool previousEmitActions;

    void Start()
    {
        if (agent == null)
            agent = GetComponent<ArmAgent>();
        if (goals == null)
            goals = GetComponent<ArmJointController>();
        if (agent == null || goals == null || controllerJson == null)
        {
            Debug.LogError("WorldModelController needs agent, goals, and controller.json.");
            enabled = false;
            return;
        }

        features = new float[agent.StateSize];
        rawAction = new float[agent.ActionSize];
        action = new float[agent.ActionSize];

        if (!Load(controllerJson.text))
        {
            enabled = false;
            return;
        }

        previousEmitActions = goals.emitActions;
        goals.emitActions = false;
        tookOverEmitActions = true;
        ready = true;
        Debug.Log(
            "WorldModelController ready (CMA-ES Controller C). " +
            "Goals come from ArmJointController; actions come from the learned linear policy.");
    }

    void OnEnable()
    {
        if (!ready || goals == null)
            return;

        previousEmitActions = goals.emitActions;
        goals.emitActions = false;
        tookOverEmitActions = true;
        hasAction = false;
    }

    void OnDisable()
    {
        if (tookOverEmitActions && goals != null)
        {
            goals.emitActions = previousEmitActions;
            tookOverEmitActions = false;
        }
        hasAction = false;
    }

    unsafe void FixedUpdate()
    {
        if (!ready || !agent.Ready || goals.CurrentTarget == null)
            return;
        if (!Mujoco.MjScene.InstanceExists || Mujoco.MjScene.Instance.Data == null)
            return;

        if (!agent.IsControlStep)
            return;   // the servos hold the last command between control steps

        var state = agent.Observe();

        // First step after enabling: start the smoothed command at the current joint
        // angles so the arm does not jump (same as rollout() in train_controller.py).
        if (!hasAction)
        {
            for (int i = 0; i < agent.ActionSize; i++)
                action[i] = state[i];
            hasAction = true;
        }

        ActionFrom(state, rawAction);       // raw = linear policy(q - goal, qd)
        ApplySmoothing(rawAction, action);  // low-pass + rate limit, in place on `action`
        agent.SetAction(action);
    }

    // Policy with the goal taken from ArmJointController (random goals or IK goals).
    public void ActionFrom(float[] state, float[] destination)
    {
        if (goalBuffer == null || goalBuffer.Length != agent.ActionSize)
            goalBuffer = new float[agent.ActionSize];
        goals.ActionFrom(goalBuffer);
        ActionFrom(state, goalBuffer, destination);
    }

    // Controller C (residual linear policy, identical to policy() in train_controller.py):
    //   f    = [q - goal, qd]                       (StateSize features)
    //   raw  = clip(goal + W f + b, joint_lo, joint_hi)
    // With W = 0, b = 0 this is plain open-loop tracking; CMA-ES learns the gains that
    // add damping / anticipation and the bias that cancels gravity sag.
    public void ActionFrom(float[] state, float[] goal, float[] destination)
    {
        int A = agent.ActionSize, S = agent.StateSize, J = agent.JointCount;
        for (int j = 0; j < J; j++)
        {
            features[j] = state[j] - goal[j];
            features[J + j] = state[J + j];
        }

        for (int i = 0; i < A; i++)
        {
            if (baselineOpenLoop)
            {
                destination[i] = Mathf.Clamp(goal[i], jointLo[i], jointHi[i]);
                continue;
            }
            float sum = b[i];
            int row = i * S;
            for (int k = 0; k < S; k++)
                sum += W[row + k] * features[k];
            destination[i] = Mathf.Clamp(goal[i] + sum, jointLo[i], jointHi[i]);
        }
    }

    // smoothed <- prev + clamp(alpha * (raw - prev), +-maxDelta)   (== smooth_action()).
    // `smoothed` holds the previous command on entry and the new command on exit.
    void ApplySmoothing(float[] raw, float[] smoothed)
    {
        for (int i = 0; i < raw.Length; i++)
        {
            float prev = smoothed[i];
            float blended = (1f - actionSmoothing) * prev + actionSmoothing * raw[i];
            float delta = Mathf.Clamp(blended - prev, -maxDeltaPerStep, maxDeltaPerStep);
            smoothed[i] = Mathf.Clamp(prev + delta, jointLo[i], jointHi[i]);
        }
    }

    bool Load(string json)
    {
        try
        {
            var data = JsonUtility.FromJson<ControllerFile>(json);
            if (data == null || data.W == null || data.b == null ||
                data.W.Length != agent.ActionSize * agent.StateSize ||
                data.b.Length != agent.ActionSize)
            {
                Debug.LogError(
                    "controller.json has unexpected W/b shapes for ActionSize=" +
                    agent.ActionSize + " StateSize=" + agent.StateSize + ".");
                return false;
            }

            W = data.W;
            b = data.b;
            jointLo = data.joint_lo != null && data.joint_lo.Length == agent.ActionSize
                ? data.joint_lo
                : CopyRangesLo();
            jointHi = data.joint_hi != null && data.joint_hi.Length == agent.ActionSize
                ? data.joint_hi
                : CopyRangesHi();
            // Use exactly the smoothing the policy was trained with in imagination.
            if (data.action_smooth > 0f)
                actionSmoothing = data.action_smooth;
            if (data.max_delta > 0f)
                maxDeltaPerStep = data.max_delta;
            return true;
        }
        catch (Exception exception)
        {
            Debug.LogException(exception);
            return false;
        }
    }

    float[] CopyRangesLo()
    {
        var result = new float[agent.ActionSize];
        for (int i = 0; i < agent.ActionSize; i++)
            result[i] = goals.jointRanges != null && i < goals.jointRanges.Length
                ? goals.jointRanges[i].x
                : -Mathf.PI;
        return result;
    }

    float[] CopyRangesHi()
    {
        var result = new float[agent.ActionSize];
        for (int i = 0; i < agent.ActionSize; i++)
            result[i] = goals.jointRanges != null && i < goals.jointRanges.Length
                ? goals.jointRanges[i].y
                : Mathf.PI;
        return result;
    }

    [Serializable]
    class ControllerFile
    {
        public string feature;
        public int state_size;
        public int action_size;
        public float[] W;
        public float[] b;
        public float[] joint_lo;
        public float[] joint_hi;
        public float[] home;
        public float action_smooth;
        public float max_delta;
    }
}
