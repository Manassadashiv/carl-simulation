import os
import sys
import numpy as np
import mujoco

sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))

from carl_arm_train import build_cache, reset_episode, ARM_CTRL_LOW, ARM_CTRL_HIGH, ARM_CTRL_SLICE

def test_ik(step_scale=0.15, max_dq=0.08):
    model = mujoco.MjModel.from_xml_path("vessel_kinetic.xml")
    data = mujoco.MjData(model)
    cache = build_cache(model)
    rng = np.random.default_rng(seed=42)
    
    def get_ik_action(target_pos):
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        tip_sid = cache['tip_L_sid']
        mujoco.mj_jacSite(model, data, jacp, jacr, tip_sid)
        left_arm_dofs = cache['arm_dof'][:5]
        J = jacp[:, left_arm_dofs]
        tip_pos = data.site_xpos[tip_sid]
        dx = target_pos - tip_pos
        lambda_sq = 0.015
        J_dls = J.T @ np.linalg.inv(J @ J.T + lambda_sq * np.eye(3))
        dq = J_dls @ dx
        dq = np.clip(dq * step_scale, -max_dq, max_dq)
        current_q = np.array([data.qpos[a] for a in cache['arm_qpos'][:5]])
        cmd_q = current_q + dq
        cmd_q = np.clip(cmd_q, ARM_CTRL_LOW[:5], ARM_CTRL_HIGH[:5])
        
        full_cmd = np.zeros(10)
        full_cmd[:5] = cmd_q
        full_cmd[5:] = (ARM_CTRL_HIGH[5:] + ARM_CTRL_LOW[5:]) / 2.0
        center = (ARM_CTRL_HIGH + ARM_CTRL_LOW) / 2.0
        scale = (ARM_CTRL_HIGH - ARM_CTRL_LOW) / 2.0 + 1e-8
        action_norm = (full_cmd - center) / scale
        return np.clip(action_norm, -1.0, 1.0)

    successes = 0
    trials = 10
    
    for ep in range(trials):
        reset_episode(model, data, cache, "reach", rng)
        ox = rng.uniform(0.15, 0.20)
        oy = rng.uniform(0.13, 0.17)
        oz = 0.035
        data.qpos[cache['obj_qpos'] : cache['obj_qpos']+3] = [ox, oy, oz]
        data.qpos[cache['obj_qpos']+3 : cache['obj_qpos']+7] = [1.0, 0.0, 0.0, 0.0]
        mujoco.mj_forward(model, data)
        
        success = False
        for step in range(150):
            data.qpos[cache['Q_CARL'] : cache['Q_CARL']+3] = [0.0, 0.0, 0.04]
            data.qpos[cache['Q_CARL']+3 : cache['Q_CARL']+7] = [1.0, 0.0, 0.0, 0.0]
            data.qvel[cache['V_CARL'] : cache['V_CARL']+6] = 0.0
            
            obj_pos = data.xpos[cache['obj_bid']].copy()
            tip_pos = data.site_xpos[cache['tip_L_sid']].copy()
            dist = np.linalg.norm(obj_pos - tip_pos)
            
            if dist < 0.045:
                success = True
                break
                
            act = get_ik_action(obj_pos)
            center = (ARM_CTRL_HIGH + ARM_CTRL_LOW) / 2.0
            scale = (ARM_CTRL_HIGH - ARM_CTRL_LOW) / 2.0
            act_physical = act * scale + center
            
            data.ctrl[:2] = 0.0
            data.ctrl[ARM_CTRL_SLICE] = act_physical
            mujoco.mj_step(model, data)
            
        if success:
            successes += 1
            
    print(f"Scale: {step_scale:.2f} | Max dq: {max_dq:.3f} | Success: {successes}/{trials}")

if __name__ == "__main__":
    for scale in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]:
        for max_dq in [0.04, 0.06, 0.08, 0.10, 0.15, 0.20]:
            test_ik(scale, max_dq)
