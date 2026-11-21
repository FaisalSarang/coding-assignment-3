using System.Collections.Generic;
using Mujoco;
using UnityEngine;

// Measurement helper used by the CA3 > Run: Verify harness. Samples once per control
// step (not per rendered frame), so results don't depend on the editor's frame rate.
[DefaultExecutionOrder(5)]
public class CA3Probe : MonoBehaviour
{
    public ArmAgent agent;
    public ArmJointController goals;
    public WorldModelDriver driver;
    public IkGoalProvider ik;

    public bool recording;
    public readonly List<float> jointError = new();   // RMS |q - goal| (rad)
    public readonly List<float> ghostError = new();   // RMS |ghost - real| (rad)
    public readonly List<float> eeError = new();      // |EE - target| (m)
    public readonly List<float> ikResidual = new();   // |target - FK(q_goal)| (m)
    public readonly List<float> siteError = new();    // |target - live MuJoCo site_xpos| (m)
    public readonly List<float> transformGap = new(); // |endEffector.position - live site_xpos| (m)

    public void Clear()
    {
        jointError.Clear();
        ghostError.Clear();
        eeError.Clear();
        ikResidual.Clear();
        siteError.Clear();
        transformGap.Clear();
    }

    unsafe void FixedUpdate()
    {
        if (!recording || agent == null || !agent.Ready || !agent.IsControlStep || goals == null || goals.CurrentTarget == null)
            return;

        var s = agent.Observe();
        float e = 0f;
        for (int i = 0; i < agent.ActionSize; i++)
        {
            float d = s[i] - goals.CurrentTarget[i];
            e += d * d;
        }
        jointError.Add(Mathf.Sqrt(e / agent.ActionSize));

        if (driver != null && driver.isActiveAndEnabled)
            ghostError.Add(driver.LastGhostError);

        if (ik != null && ik.isActiveAndEnabled && ik.target != null && agent.endEffector != null)
        {
            eeError.Add(Vector3.Distance(agent.endEffector.position, ik.target.position));
            ikResidual.Add(ik.LastError);
            if (ik.endEffectorSite != null && MjScene.InstanceExists && MjScene.Instance.Data != null)
            {
                var live = MjEngineTool.UnityVector3(
                    MjEngineTool.MjVector3AtEntry(MjScene.Instance.Data->site_xpos, ik.endEffectorSite.MujocoId));
                siteError.Add(Vector3.Distance(live, ik.target.position));
                transformGap.Add(Vector3.Distance(live, agent.endEffector.position));
            }
        }
    }
}
