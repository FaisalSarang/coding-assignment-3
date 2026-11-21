using System.IO;
using Mujoco;
using Unity.InferenceEngine;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

// CA3 menu: builds the UR5e scene and switches between the assignment's modes.
//   CA3 > 1. Build UR5e Scene                 (import MJCF, add + wire every component)
//   CA3 > Mode: Collect Data                  (Step 2: random goals -> rollout.csv)
//   CA3 > Mode: World Model + Controller      (Step 3: ghost arm + learned policy on random goals)
//   CA3 > Mode: IK Tracking                   (Step 4: drag TrackingTarget in the Scene view)
public static class CA3Setup
{
    public const string ScenePath = "Assets/CA3/CA3_UR5e.unity";
    public const string DataDir = "Assets/CA3/Data";
    const string GhostPrefabPath = "Assets/CA3/GhostArm.prefab";
    const string GhostMatPath = "Assets/CA3/GhostMaterial.mat";
    const string LineMatPath = "Assets/CA3/LineMaterial.mat";
    const string TargetMatPath = "Assets/CA3/TargetMaterial.mat";

    static string MjcfPath => Path.GetFullPath(Path.Combine(
        Application.dataPath, "..", "..", "menagerie", "universal_robots_ur5e", "scene.xml"));

    [MenuItem("CA3/1. Build UR5e Scene", priority = 0)]
    public static void BuildScene()
    {
        if (!File.Exists(MjcfPath))
            throw new FileNotFoundException("UR5e MJCF not found", MjcfPath);

        EditorSceneManager.NewScene(NewSceneSetup.DefaultGameObjects, NewSceneMode.Single);

        // --- Import the arm with the MuJoCo plugin (same as Assets > Import MuJoCo Scene) ---
        var root = new MjImporterWithAssets().ImportFile(MjcfPath);
        if (root == null)
            throw new System.Exception("MuJoCo import failed; see the Console.");
        root.name = "UR5e_Scene";
        var baseBody = FindDeep(root.transform, "base");
        if (baseBody == null)
            throw new System.Exception("Imported scene has no 'base' body.");
        var arm = baseBody.gameObject;

        // --- Ghost prefab: a visual-only copy of the arm, made BEFORE adding our scripts ---
        var ghostPrefab = PrefabUtility.SaveAsPrefabAsset(arm, GhostPrefabPath);

        // --- Materials ---
        var ghostMat = MakeMaterial(GhostMatPath, "Sprites/Default", new Color(0.3f, 0.8f, 1f, 0.35f));
        var lineMat = MakeMaterial(LineMatPath, "Sprites/Default", new Color(1f, 0.85f, 0.1f, 1f));
        var targetMat = MakeMaterial(TargetMatPath, "Universal Render Pipeline/Lit", new Color(0.95f, 0.2f, 0.2f, 1f));

        // --- Components on the arm (base body) ---
        var agent = arm.AddComponent<ArmAgent>();
        var goals = arm.AddComponent<ArmJointController>();
        goals.agent = agent;
        var recorder = arm.AddComponent<RolloutRecorder>();
        recorder.agent = agent;
        recorder.controller = goals;
        var policy = arm.AddComponent<WorldModelController>();
        policy.agent = agent;
        policy.goals = goals;
        var driver = arm.AddComponent<WorldModelDriver>();
        driver.agent = agent;
        driver.controller = goals;
        driver.policy = policy;
        driver.ghostArm = ghostPrefab.transform;
        driver.ghostMaterial = ghostMat;
        driver.lineMaterial = lineMat;
        var ik = arm.AddComponent<IkGoalProvider>();
        ik.agent = agent;
        ik.goals = goals;
        var site = FindDeep(root.transform, "attachment_site");
        if (site != null)
            ik.endEffectorSite = site.GetComponent<MjSite>();

        // Pre-wire references the Awake auto-wiring would otherwise find at runtime.
        agent.endEffectorBody = FindDeep(root.transform, "wrist_3_link")?.GetComponent<MjBody>();
        agent.endEffector = site;

        // --- Movable IK target (no collider) ---
        var target = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        target.name = "TrackingTarget";
        Object.DestroyImmediate(target.GetComponent<Collider>());
        target.transform.position = new Vector3(-0.13f, 0.45f, 0.55f);
        target.transform.localScale = Vector3.one * 0.05f;
        target.GetComponent<MeshRenderer>().sharedMaterial = targetMat;
        ik.target = target.transform;

        // --- Camera ---
        var cam = Camera.main;
        if (cam != null)
        {
            cam.transform.position = new Vector3(1.1f, 0.85f, 1.3f);
            cam.transform.LookAt(new Vector3(0f, 0.35f, 0.2f));
            cam.nearClipPlane = 0.01f;
        }

        SetMode(Mode.Collect, assignAssets: false);
        Directory.CreateDirectory(DataDir);
        EditorSceneManager.SaveScene(UnityEngine.SceneManagement.SceneManager.GetActiveScene(), ScenePath);
        EditorBuildSettings.scenes = new[] { new EditorBuildSettingsScene(ScenePath, true) };
        AssetDatabase.SaveAssets();
        Debug.Log("CA3: built " + ScenePath + " (mode: Collect Data).");
    }

    public enum Mode { Collect, Controller, IK }

    [MenuItem("CA3/Mode: Collect Data", priority = 20)]
    static void ModeCollect() => SetMode(Mode.Collect);

    [MenuItem("CA3/Mode: World Model + Controller (random goals)", priority = 21)]
    static void ModeController() => SetMode(Mode.Controller);

    [MenuItem("CA3/Mode: IK Tracking", priority = 22)]
    static void ModeIK() => SetMode(Mode.IK);

    public static void SetMode(Mode mode, bool assignAssets = true)
    {
        var agent = Object.FindFirstObjectByType<ArmAgent>(FindObjectsInactive.Include);
        if (agent == null)
        {
            Debug.LogError("CA3: no ArmAgent in the open scene. Run CA3 > 1. Build UR5e Scene first.");
            return;
        }
        var go = agent.gameObject;
        var goals = go.GetComponent<ArmJointController>();
        var recorder = go.GetComponent<RolloutRecorder>();
        var policy = go.GetComponent<WorldModelController>();
        var driver = go.GetComponent<WorldModelDriver>();
        var ik = go.GetComponent<IkGoalProvider>();

        if (assignAssets)
        {
            policy.controllerJson = AssetDatabase.LoadAssetAtPath<TextAsset>(DataDir + "/controller.json");
            driver.normJson = AssetDatabase.LoadAssetAtPath<TextAsset>(DataDir + "/norm.json");
            driver.modelAsset = AssetDatabase.LoadAssetAtPath<ModelAsset>(DataDir + "/dynamics.onnx");
        }

        bool trained = policy.controllerJson != null && driver.normJson != null && driver.modelAsset != null;
        if (mode != Mode.Collect && !trained)
            Debug.LogWarning("CA3: controller.json / norm.json / dynamics.onnx not found in " + DataDir +
                             ". Collect data and run train.py + train_controller.py first.");

        goals.emitActions = true;                       // WorldModelController turns this off itself
        goals.sampleRandomGoals = mode != Mode.IK;      // IkGoalProvider turns this off itself
        recorder.enabled = mode == Mode.Collect;
        policy.enabled = mode != Mode.Collect && trained;
        driver.enabled = mode != Mode.Collect && trained;
        ik.enabled = mode == Mode.IK;
        var target = ik.target != null ? ik.target.gameObject : null;
        if (target != null)
            target.SetActive(mode == Mode.IK);

        foreach (var c in new Object[] { goals, recorder, policy, driver, ik })
            EditorUtility.SetDirty(c);
        if (target != null)
            EditorUtility.SetDirty(target);
        EditorSceneManager.MarkSceneDirty(go.scene);
        EditorSceneManager.SaveScene(go.scene);
        Debug.Log($"CA3: mode = {mode}");
        if (mode == Mode.IK && target != null)
            Selection.activeGameObject = target;
    }

    static Material MakeMaterial(string path, string shaderName, Color color)
    {
        var shader = Shader.Find(shaderName) ?? Shader.Find("Sprites/Default");
        var mat = new Material(shader) { color = color };
        if (mat.HasProperty("_BaseColor"))
            mat.SetColor("_BaseColor", color);
        AssetDatabase.DeleteAsset(path);
        AssetDatabase.CreateAsset(mat, path);
        return mat;
    }

    static Transform FindDeep(Transform root, string name)
    {
        if (root.name == name)
            return root;
        foreach (Transform child in root)
        {
            var found = FindDeep(child, name);
            if (found != null)
                return found;
        }
        return null;
    }
}
