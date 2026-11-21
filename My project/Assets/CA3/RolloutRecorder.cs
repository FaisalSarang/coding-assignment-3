using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using UnityEngine;

// Logs one transition per physics step:  q, qd, q*, q', qd'
// (STATE + ACTION + STATE = 12 + 6 + 12 = 30 columns for the UR5e).
//
// Runs at order -1: after the controllers chose an action and ArmAgent pushed it,
// but before MjScene (order 0) steps. So in step t we see the PRE-step state s_t,
// and the transition (s_{t-1}, a_{t-1}) -> s_t is completed here.
// Rows are recorded back-to-back with no gaps, so row i's next state == row i+1's
// state; train.py relies on that for its multi-step (open-loop) loss.
[DefaultExecutionOrder(-1)]
public class RolloutRecorder : MonoBehaviour
{
    public ArmAgent agent;
    public ArmJointController controller;
    public int steps = 70000;
    [Tooltip("Relative to the project's Assets folder. train.py reads <script folder>/Data/rollout.csv.")]
    public string outputPath = "CA3/Data/rollout.csv";
    public int warmupSteps = 5;
    public int logEvery = 5000;

    readonly List<float[]> states = new();
    readonly List<float[]> actions = new();
    readonly List<float[]> nextStates = new();

    int n;
    float[] prevS;
    float[] prevA;
    bool saved;
    int warmup;

    public bool Saved => saved;
    public int Count => n;
    float errorSum;

    void Start()
    {
        if (agent == null)
            agent = GetComponent<ArmAgent>();
        if (controller == null)
            controller = GetComponent<ArmJointController>();
        if (agent == null || !agent.Ready)
        {
            Debug.LogError("RolloutRecorder needs a ready ArmAgent.");
            enabled = false;
            return;
        }
        states.Capacity = steps;
        actions.Capacity = steps;
        nextStates.Capacity = steps;
        Debug.Log($"RolloutRecorder: collecting {steps} transitions " +
                  $"({steps * agent.ControlDt / 60f:F1} min of sim time; control dt={agent.ControlDt}, physics dt={Time.fixedDeltaTime}).");
        if (Time.fixedDeltaTime > 0.0025f)
            Debug.LogWarning("RolloutRecorder: set Project Settings > Time > Fixed Timestep to 0.002. " +
                             "At larger steps the UR5e servos ring and the data is mostly integrator noise.");
    }

    unsafe void FixedUpdate()
    {
        if (saved || !agent.Ready || !Mujoco.MjScene.InstanceExists || Mujoco.MjScene.Instance.Data == null)
            return;

        // Skip the first few steps (scene creation + the teleport to home).
        if (warmup < warmupSteps)
        {
            warmup++;
            prevS = null;
            return;
        }

        // Log one transition per CONTROL step (every agent.controlEvery physics steps).
        if (!agent.IsControlStep)
            return;

        float[] s = (float[])agent.Observe().Clone();

        if (prevS != null)
        {
            states.Add(prevS);
            actions.Add(prevA);
            nextStates.Add(s);
            n++;
            errorSum += TrackingError();

            if (logEvery > 0 && n % logEvery == 0)
            {
                Debug.Log($"RolloutRecorder: {n}/{steps}  mean |q - goal| = {errorSum / logEvery:F3} rad");
                errorSum = 0f;
            }

            if (n >= steps)
            {
                Save();
                return;
            }
        }

        prevS = s;
        prevA = (float[])agent.CurrentAction.Clone();
    }

    // RMS distance between the current joints and the controller's goal.
    float TrackingError()
    {
        var goal = controller != null ? controller.CurrentTarget : agent.CurrentAction;
        if (goal == null || prevS == null)
            return 0f;
        float sum = 0f;
        var s = nextStates[nextStates.Count - 1];
        for (int i = 0; i < agent.ActionSize; i++)
        {
            float e = s[i] - goal[i];
            sum += e * e;
        }
        return Mathf.Sqrt(sum / agent.ActionSize);
    }

    void OnDisable()
    {
        // Stopping Play early still keeps what was collected.
        if (!saved && n > 0)
            Save();
    }

    void Save()
    {
        if (saved)
            return;
        saved = true;

        string path = Path.Combine(Application.dataPath, outputPath);
        Directory.CreateDirectory(Path.GetDirectoryName(path));

        var inv = CultureInfo.InvariantCulture;
        var sb = new StringBuilder(n * 30 * 12);

        // Header is a '#' comment so np.loadtxt skips it.
        int J = agent.JointCount, A = agent.ActionSize;
        var header = new List<string>();
        for (int i = 0; i < J; i++) header.Add($"q{i}");
        for (int i = 0; i < J; i++) header.Add($"qd{i}");
        for (int i = 0; i < A; i++) header.Add($"qstar{i}");
        for (int i = 0; i < J; i++) header.Add($"q_next{i}");
        for (int i = 0; i < J; i++) header.Add($"qd_next{i}");
        sb.Append("# ").Append(string.Join(",", header)).Append('\n');

        for (int r = 0; r < n; r++)
        {
            AppendRow(sb, states[r], inv);
            sb.Append(',');
            AppendRow(sb, actions[r], inv);
            sb.Append(',');
            AppendRow(sb, nextStates[r], inv);
            sb.Append('\n');
        }

        File.WriteAllText(path, sb.ToString());
        Debug.Log($"RolloutRecorder: saved {n} transitions to {path}");
#if UNITY_EDITOR
        UnityEditor.AssetDatabase.Refresh();
#endif
    }

    static void AppendRow(StringBuilder sb, float[] values, CultureInfo inv)
    {
        for (int i = 0; i < values.Length; i++)
        {
            if (i > 0)
                sb.Append(',');
            sb.Append(values[i].ToString("G7", inv));
        }
    }
}
