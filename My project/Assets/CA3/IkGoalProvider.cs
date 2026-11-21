using Mujoco;
using UnityEngine;

// Cartesian IK -> joint-space goal for the learned controller.
//
// Algorithm: Damped Least Squares (Levenberg-Marquardt) on end-effector POSITION,
// with a null-space term that pulls the redundant DOF toward a rest posture.
//   e   = target - p(q)                         (3)
//   J   = dp/dq  (3 x n, finite differences)
//   J+  = J^T (J J^T + lambda^2 I)^-1           (n x 3)
//   dq  = J+ e + (I - J+ J) k (q_rest - q)
// FK is evaluated by MuJoCo itself (mj_kinematics) on a private scratch mjData, so
// the IK sees exactly the same kinematics as the simulator and never disturbs it.
// The solution is warm-started from last frame's answer, so it moves continuously
// with the target and does not flip between elbow-up / elbow-down branches.
[DefaultExecutionOrder(-8)]
public class IkGoalProvider : MonoBehaviour
{
    public ArmAgent agent;
    public ArmJointController goals;
    public Transform target;
    [Tooltip("Optional: MuJoCo site to place on the target (e.g. attachment_site). Falls back to ArmAgent.endEffectorBody.")]
    public MjSite endEffectorSite;

    [Header("Solver")]
    public int iterationsPerStep = 15;
    public float damping = 0.05f;
    public float maxStepPerIteration = 0.2f;
    public float finiteDifference = 1e-4f;
    public float tolerance = 0.002f;
    [Range(0f, 1f)] public float nullSpaceGain = 0.05f;

    [Header("Gravity-sag correction (outer Cartesian loop)")]
    [Tooltip("The UR5e position servos settle ~0.008 rad short under gravity (~1.4 cm at the hand). " +
             "Each control step the IK aim point moves by gain * (target - measured hand).")]
    public bool sagCorrection = true;
    [Range(0f, 0.5f)] public float sagGain = 0.05f;
    public float maxSagCorrection = 0.05f;

    [Header("Keep goals inside what the controller was trained on")]
    public bool clampToTrainingRange = true;

    [Header("Debug")]
    public bool drawGizmos = true;

    float[] q;
    float[] restPose;
    float[] lo;
    float[] hi;
    float[,] jac;
    float lastError = float.PositiveInfinity;
    Vector3 sagOffset;
    Vector3 lastSolvedPosition;
    bool previousSampleRandomGoals;
    bool tookOverGoals;
    unsafe MujocoLib.mjData_* scratch;

    public float LastError => lastError;

    void Start()
    {
        if (agent == null) agent = GetComponent<ArmAgent>();
        if (goals == null) goals = GetComponent<ArmJointController>();
        if (agent == null || goals == null || target == null || !agent.Ready)
        {
            Debug.LogError("IkGoalProvider needs ArmAgent, ArmJointController and a target Transform.");
            enabled = false;
            return;
        }

        if (endEffectorSite == null && agent.endEffector != null)
            endEffectorSite = agent.endEffector.GetComponent<MjSite>();

        int n = agent.JointCount;
        q = new float[n];
        restPose = (float[])goals.homeTargets.Clone();
        System.Array.Copy(restPose, q, n);
        jac = new float[3, n];
        lo = new float[n];
        hi = new float[n];
        for (int i = 0; i < n; i++)
        {
            Vector2 r = goals.jointRanges[i];
            if (clampToTrainingRange)
            {
                float span = 0.5f * (r.y - r.x) * goals.randomScale;
                lo[i] = Mathf.Max(r.x, goals.homeTargets[i] - span);
                hi[i] = Mathf.Min(r.y, goals.homeTargets[i] + span);
            }
            else
            {
                lo[i] = r.x;
                hi[i] = r.y;
            }
        }
        TakeOverGoals();
    }

    void OnEnable() { if (q != null) TakeOverGoals(); }

    void TakeOverGoals()
    {
        if (goals == null || tookOverGoals) return;
        previousSampleRandomGoals = goals.sampleRandomGoals;
        goals.sampleRandomGoals = false;   // the goal now comes from IK
        tookOverGoals = true;
    }

    unsafe void OnDisable()
    {
        if (tookOverGoals && goals != null)
            goals.sampleRandomGoals = previousSampleRandomGoals;
        tookOverGoals = false;
    }

    unsafe void OnDestroy()
    {
        if (scratch != null)
        {
            MujocoLib.mj_deleteData(scratch);
            scratch = null;
        }
    }

    unsafe void FixedUpdate()
    {
        if (target == null || !MjScene.InstanceExists || MjScene.Instance.Data == null)
            return;
        if (!agent.IsControlStep)
            return;
        var scene = MjScene.Instance;
        if (scratch == null)
            scratch = MujocoLib.mj_makeData(scene.Model);

        // Start from the live state so any non-arm DOF are correct, then overwrite the arm joints.
        MujocoLib.mj_copyData(scratch, scene.Model, scene.Data);
        Vector3 measured = HandPosition();   // the real hand right now (copied live state)

        // Integrate the remaining Cartesian error into the aim point, but only while the
        // last IK solve reached its aim point (no wind-up on unreachable targets).
        if (sagCorrection && lastError < 0.005f)
        {
            sagOffset += sagGain * (target.position - measured);
            sagOffset = Vector3.ClampMagnitude(sagOffset, maxSagCorrection);
        }
        else if (!sagCorrection)
        {
            sagOffset = Vector3.zero;
        }

        Solve(scene.Model, target.position + sagOffset);
        goals.SetTarget(q);
    }

    unsafe void Solve(MujocoLib.mjModel_* model, Vector3 goal)
    {
        int n = q.Length;
        var dq = new float[n];
        var qPlus = new float[n];

        for (int iter = 0; iter < iterationsPerStep; iter++)
        {
            Vector3 p = Fk(model, q);
            Vector3 e = goal - p;
            lastError = e.magnitude;
            lastSolvedPosition = p;
            if (lastError < tolerance)
                break;

            // Numeric Jacobian (3 x n).
            for (int j = 0; j < n; j++)
            {
                System.Array.Copy(q, qPlus, n);
                qPlus[j] += finiteDifference;
                Vector3 d = (Fk(model, qPlus) - p) / finiteDifference;
                jac[0, j] = d.x; jac[1, j] = d.y; jac[2, j] = d.z;
            }

            // A = J J^T + lambda^2 I (3x3), then A^-1.
            var A = new float[3, 3];
            for (int r = 0; r < 3; r++)
                for (int c = 0; c < 3; c++)
                {
                    float sum = 0f;
                    for (int k = 0; k < n; k++) sum += jac[r, k] * jac[c, k];
                    A[r, c] = sum + (r == c ? damping * damping : 0f);
                }
            var Ainv = Invert3x3(A);

            // J+ = J^T A^-1  (n x 3)
            var Jp = new float[n, 3];
            for (int k = 0; k < n; k++)
                for (int c = 0; c < 3; c++)
                    Jp[k, c] = jac[0, k] * Ainv[0, c] + jac[1, k] * Ainv[1, c] + jac[2, k] * Ainv[2, c];

            // Primary task: dq = J+ e
            for (int k = 0; k < n; k++)
                dq[k] = Jp[k, 0] * e.x + Jp[k, 1] * e.y + Jp[k, 2] * e.z;

            // Secondary task in the null space: (I - J+ J) * gain * (rest - q)
            if (nullSpaceGain > 0f)
            {
                var z = new float[n];
                for (int k = 0; k < n; k++) z[k] = nullSpaceGain * (restPose[k] - q[k]);
                var Jz = new float[3];
                for (int r = 0; r < 3; r++)
                    for (int k = 0; k < n; k++) Jz[r] += jac[r, k] * z[k];
                for (int k = 0; k < n; k++)
                    dq[k] += z[k] - (Jp[k, 0] * Jz[0] + Jp[k, 1] * Jz[1] + Jp[k, 2] * Jz[2]);
            }

            // Limit the step, apply, clamp to joint limits.
            float maxAbs = 0f;
            for (int k = 0; k < n; k++) maxAbs = Mathf.Max(maxAbs, Mathf.Abs(dq[k]));
            float scale = maxAbs > maxStepPerIteration ? maxStepPerIteration / maxAbs : 1f;
            for (int k = 0; k < n; k++)
                q[k] = Mathf.Clamp(q[k] + scale * dq[k], lo[k], hi[k]);
        }
    }

    // Forward kinematics with MuJoCo on the scratch data, in Unity world coordinates.
    unsafe Vector3 Fk(MujocoLib.mjModel_* model, float[] joints)
    {
        for (int i = 0; i < joints.Length; i++)
            scratch->qpos[agent.joints[i].QposAddress] = joints[i];
        MujocoLib.mj_kinematics(model, scratch);
        return HandPosition();
    }

    // Hand position stored in the scratch data (after mj_copyData or mj_kinematics).
    unsafe Vector3 HandPosition()
    {
        if (endEffectorSite != null && endEffectorSite.MujocoId >= 0)
            return MjEngineTool.UnityVector3(MjEngineTool.MjVector3AtEntry(scratch->site_xpos, endEffectorSite.MujocoId));
        return MjEngineTool.UnityVector3(MjEngineTool.MjVector3AtEntry(scratch->xpos, agent.endEffectorBody.MujocoId));
    }

    static float[,] Invert3x3(float[,] m)
    {
        float a = m[0, 0], b = m[0, 1], c = m[0, 2];
        float d = m[1, 0], e = m[1, 1], f = m[1, 2];
        float g = m[2, 0], h = m[2, 1], i = m[2, 2];
        float A = e * i - f * h, B = -(d * i - f * g), C = d * h - e * g;
        float det = a * A + b * B + c * C;
        float inv = Mathf.Abs(det) > 1e-12f ? 1f / det : 0f;
        var r = new float[3, 3];
        r[0, 0] = A * inv; r[0, 1] = -(b * i - c * h) * inv; r[0, 2] = (b * f - c * e) * inv;
        r[1, 0] = B * inv; r[1, 1] = (a * i - c * g) * inv;  r[1, 2] = -(a * f - c * d) * inv;
        r[2, 0] = C * inv; r[2, 1] = -(a * h - b * g) * inv; r[2, 2] = (a * e - b * d) * inv;
        return r;
    }

    void OnDrawGizmos()
    {
        if (!drawGizmos || target == null || !Application.isPlaying)
            return;
        Gizmos.color = lastError < 0.01f ? Color.green : Color.yellow;
        Gizmos.DrawWireSphere(target.position, 0.03f);
        Gizmos.color = Color.cyan;
        Gizmos.DrawLine(lastSolvedPosition, target.position);
    }
}
