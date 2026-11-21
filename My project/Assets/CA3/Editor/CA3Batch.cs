using System.Collections.Generic;
using System.IO;
using System.Text;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

// Headless runs from the command line (Unity.exe -batchmode -executeMethod CA3Batch.X):
//   CA3Batch.Build    -> CA3Setup.BuildScene and quit
//   CA3Batch.Collect  -> Play in Collect mode at high time scale until rollout.csv is saved
//   CA3Batch.Verify   -> Play: open-loop vs learned controller on random goals, then IK on a
//                        moving target; writes Assets/CA3/Data/verify_log.txt
// The same runs also work from the CA3 menu inside the editor.
[InitializeOnLoad]
public static class CA3Batch
{
    const string JobKey = "CA3Batch.Job";
    const string StartKey = "CA3Batch.Start";
    static readonly StringBuilder log = new();
    static string LogPath => Path.Combine(Application.dataPath, "CA3/Data/verify_log.txt");

    static CA3Batch()
    {
        EditorApplication.update += Tick;
        EditorApplication.playModeStateChanged += OnPlayMode;
    }

    public static void Build()
    {
        CA3Setup.BuildScene();
        EditorApplication.Exit(0);
    }

    [MenuItem("CA3/Run: Collect Data (fast)", priority = 40)]
    public static void Collect() => Start("collect");

    [MenuItem("CA3/Run: Verify Controller + IK", priority = 41)]
    public static void Verify() => Start("verify");

    static void Start(string job)
    {
        EditorSceneManager.OpenScene(CA3Setup.ScenePath);
        CA3Setup.SetMode(job == "collect" ? CA3Setup.Mode.Collect : CA3Setup.Mode.Controller);
        // Don't press Play yet: Unity often requests a recompile right after startup
        // ("Define symbols changed"), and a domain reload during Play stops the run.
        // Tick() enters Play once nothing has compiled or imported for 20 s.
        SessionState.SetString(JobKey, "wait:" + job);
        SessionState.SetFloat(StableKey, -1f);
        SessionState.SetFloat(StartKey, -1f);
        Debug.Log("CA3Batch: waiting for compilation to settle before entering Play mode...");
    }

    const string StableKey = "CA3Batch.StableSince";

    static void TickWait(string job)
    {
        float now = (float)EditorApplication.timeSinceStartup;
        if (EditorApplication.isCompiling || EditorApplication.isUpdating)
        {
            SessionState.SetFloat(StableKey, -1f);
            return;
        }
        float since = SessionState.GetFloat(StableKey, -1f);
        if (since < 0f || since > now)
        {
            SessionState.SetFloat(StableKey, now);
            return;
        }
        if (now - since < 20f)
            return;
        SessionState.SetString(JobKey, job.Substring(5));
        Debug.Log("CA3Batch: entering Play mode for " + job.Substring(5));
        EditorApplication.isPlaying = true;
    }

    static void OnPlayMode(PlayModeStateChange change)
    {
        if (change == PlayModeStateChange.EnteredEditMode && SessionState.GetString(JobKey, "") == "done")
        {
            SessionState.EraseString(JobKey);
            // Leave the scene in the mode the user is most likely to want next.
            CA3Setup.SetMode(File.Exists(Path.Combine(Application.dataPath, "CA3/Data/controller.json"))
                ? CA3Setup.Mode.IK : CA3Setup.Mode.Collect);
            if (Application.isBatchMode)
                EditorApplication.Exit(0);
        }
    }

    // ---------------- per-frame driver ----------------
    static int phase;
    static float phaseStart;
    static readonly List<float> samples = new();
    static readonly List<float> samples2 = new();
    static Vector3 eeHome;
    static bool verifyInit;

    static void Tick()
    {
        string job = SessionState.GetString(JobKey, "");
        if (job.StartsWith("wait:"))
        {
            if (!EditorApplication.isPlaying)
                TickWait(job);
            return;
        }
        if (string.IsNullOrEmpty(job) || job == "done" || !EditorApplication.isPlaying)
            return;

        var agent = Object.FindFirstObjectByType<ArmAgent>();
        if (agent == null)
            return;

        if (SessionState.GetFloat(StartKey, -1f) < 0f)
        {
            SessionState.SetFloat(StartKey, Time.realtimeSinceStartup);
            Time.timeScale = job == "collect" ? 100f : 20f;
            phase = 0;
            phaseStart = Time.time;
            samples.Clear();
            samples2.Clear();
            log.Clear();
            verifyInit = false;
        }

        float wall = Time.realtimeSinceStartup - SessionState.GetFloat(StartKey, 0f);
        if (wall > 3600f)
        {
            Debug.LogError("CA3Batch: timed out.");
            Finish();
            return;
        }

        if (job == "collect")
            TickCollect(agent);
        else
            TickVerify(agent);
    }

    static float lastReport;

    static void TickCollect(ArmAgent agent)
    {
        var recorder = agent.GetComponent<RolloutRecorder>();
        if (Time.realtimeSinceStartup - lastReport > 15f)
        {
            lastReport = Time.realtimeSinceStartup;
            Debug.Log($"CA3Batch: collected {recorder.Count}/{recorder.steps} (sim t = {Time.time:F0}s)");
        }
        if (recorder.Saved)
            Finish();
    }

    // Phase 0: 120 s, random goals, smoothed open-loop baseline (W = b = 0, same smoothing)
    // Phase 1: 120 s, the SAME random goal sequence, learned W and b
    // Phase 2: 30 s, IK + controller on 3 stationary targets (10 s each, settled error)
    // Phase 3: 40 s, IK + controller following a target moving on a loop
    const int GoalSeed = 42;
    static CA3Probe probe;
    static readonly Vector3 eeHomeUnity = new Vector3(-0.134f, 0.488f, 0.492f); // MuJoCo (-0.134, 0.492, 0.488)
    static readonly Vector3[] staticOffsets =
    {
        new Vector3(-0.15f, -0.05f, 0.05f),
        new Vector3(0.10f, 0.05f, -0.10f),
        new Vector3(-0.05f, -0.15f, 0.10f),
    };

    static void TickVerify(ArmAgent agent)
    {
        var go = agent.gameObject;
        var goals = go.GetComponent<ArmJointController>();
        var policy = go.GetComponent<WorldModelController>();
        var driver = go.GetComponent<WorldModelDriver>();
        var ik = go.GetComponent<IkGoalProvider>();
        float t = Time.time - phaseStart;

        if (!verifyInit)
        {
            verifyInit = true;
            probe = go.GetComponent<CA3Probe>() ?? go.AddComponent<CA3Probe>();
            probe.agent = agent; probe.goals = goals; probe.driver = driver; probe.ik = ik;
            policy.baselineOpenLoop = true;
            goals.RestartRandomGoals(GoalSeed);
        }

        if (phase == 2 && ik.target != null)
        {
            int k = Mathf.Min(staticOffsets.Length - 1, (int)(t / 10f));
            ik.target.position = eeHomeUnity + staticOffsets[k];
            probe.recording = (t % 10f) > 6f;            // only the settled part of each hold
        }
        else if (phase == 3 && ik.target != null)
        {
            float w = 2f * Mathf.PI * t / 10f;
            ik.target.position = eeHomeUnity + new Vector3(0.15f * Mathf.Cos(w) - 0.15f, 0.08f * Mathf.Sin(2f * w), 0.12f * Mathf.Sin(w));
            probe.recording = t > 3f;
        }
        else
        {
            probe.recording = t > 1f;
        }

        float duration = phase <= 1 ? 120f : phase == 2 ? 30f : 40f;
        if (t < duration)
            return;

        probe.recording = false;
        if (phase == 0)
        {
            Report("Random goals, smoothed open-loop baseline (W=b=0), 120 s", "RMS joint error to goal", "rad", probe.jointError);
            Report("  ghost vs real arm", "RMS joint error", "rad", probe.ghostError);
            policy.baselineOpenLoop = false;
            goals.RestartRandomGoals(GoalSeed);
        }
        else if (phase == 1)
        {
            Report("Random goals, learned controller, 120 s", "RMS joint error to goal", "rad", probe.jointError);
            Report("  ghost vs real arm", "RMS joint error", "rad", probe.ghostError);
            ik.target.gameObject.SetActive(true);
            ik.target.position = eeHomeUnity + staticOffsets[0];
            ik.enabled = true;
        }
        else if (phase == 2)
        {
            Report("IK, 3 stationary targets (settled)", "end-effector to target", "m", probe.eeError);
            Report("  IK solver residual", "|target - FK(q_goal)|", "m", probe.ikResidual);
            Report("  live MuJoCo site to target", "|target - site_xpos|", "m", probe.siteError);
            Report("  EE Transform vs MuJoCo site", "|endEffector.position - site_xpos|", "m", probe.transformGap);
            Report("  joints vs IK goal", "RMS joint error", "rad", probe.jointError);
        }
        else
        {
            Report("IK, target moving on a loop (~0.1 m/s), 40 s", "end-effector to target", "m", probe.eeError);
            Report("  IK solver residual", "|target - FK(q_goal)|", "m", probe.ikResidual);
            Report("  ghost vs real arm", "RMS joint error", "rad", probe.ghostError);
            File.WriteAllText(LogPath, log.ToString());
            Debug.Log("CA3Batch verify results:\n" + log);
            Finish();
            return;
        }
        phase++;
        phaseStart = Time.time;
        probe.Clear();
    }

    static void Report(string title, string what, string unit, List<float> values)
    {
        if (values.Count == 0)
        {
            log.AppendLine($"{title}: no samples");
            return;
        }
        var sorted = new List<float>(values);
        sorted.Sort();
        float mean = 0f;
        foreach (var v in values) mean += v;
        mean /= values.Count;
        log.AppendLine($"{title}: {what} mean {mean:F4} {unit}, median {sorted[sorted.Count / 2]:F4}, " +
                       $"p95 {sorted[(int)(sorted.Count * 0.95f)]:F4}  (n={values.Count})");
    }

    static void Finish()
    {
        SessionState.SetString(JobKey, "done");
        Time.timeScale = 1f;
        EditorApplication.isPlaying = false;
    }
}
