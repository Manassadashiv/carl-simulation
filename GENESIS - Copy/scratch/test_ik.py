import numpy as np
import mujoco
from carl_arm_train import build_cache, reset_episode, compute_reward, ARM_CTRL_LOW, ARM_CTRL_HIGH, ARM_CTRL_SLICE

def get_ik_action_dls(model, data, cache, target_pos):
    # 1. Compute site Jacobian
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, cache['tip_L_sid'])
    
    # Slice Jacobian for the 4 left arm joints
    left_arm_dofs = cache['arm_dof'][:4]
    J = jacp[:, left_arm_dofs]
    
    # 2. Compute error vector to target
    tip_pos = data.site_xpos[cache['tip_L_sid']]
    dx = target_pos - tip_pos
    
    # 3. Damped Least Squares (DLS) Inverse
    lambda_sq = 0.015
    J_dls = J.T @ np.linalg.inv(J @ J.T + lambda_sq * np.eye(3))
    dq = J_dls @ dx
    
    # Scale dq
    step_scale = 0.45
    dq = np.clip(dq * step_scale, -0.2, 0.2)
    
    # 4. Update command target in joint radians
    current_q = np.array([data.qpos[a] for a in cache['arm_qpos'][:4]])
    cmd_q = current_q + dq
    
    # Clip to physical joint limits
    cmd_q = np.clip(cmd_q, ARM_CTRL_LOW[:4], ARM_CTRL_HIGH[:4])
    
    # 5. Mirror right arm with default center pose
    full_cmd = np.zeros(8)
    full_cmd[:4] = cmd_q
    full_cmd[4:] = (ARM_CTRL_HIGH[4:] + ARM_CTRL_LOW[4:]) / 2.0
    
    # 6. Normalize to [-1, 1] range for network output
    center = (ARM_CTRL_HIGH + ARM_CTRL_LOW) / 2.0
    scale = (ARM_CTRL_HIGH - ARM_CTRL_LOW) / 2.0 + 1e-8
    action_norm = (full_cmd - center) / scale
    
    return np.clip(action_norm, -1.0, 1.0)

model = mujoco.MjModel.from_xml_path("vessel_kinetic.xml")
data = mujoco.MjData(model)
cache = build_cache(model)
rng = np.random.default_rng(seed=42)

reset_episode(model, data, cache, "reach", rng)
# Initialize the arm to a natural home posture
data.qpos[cache['arm_qpos'][0]] = 0.3
data.qpos[cache['arm_qpos'][1]] = -1.2
data.qpos[cache['arm_qpos'][2]] = 0.0
mujoco.mj_forward(model, data)

obj_pos = data.xpos[cache['obj_bid']].copy()

print("Initial Object Pos:", obj_pos)
print("Initial Tip Pos:", data.site_xpos[cache['tip_L_sid']])

for step in range(150):
    # Lock base position and velocity
    data.qpos[cache['Q_CARL'] : cache['Q_CARL']+3] = [0.0, 0.0, 0.04]
    data.qpos[cache['Q_CARL']+3 : cache['Q_CARL']+7] = [1.0, 0.0, 0.0, 0.0]
    data.qvel[cache['V_CARL'] : cache['V_CARL']+6] = 0.0
    
    tip_pos = data.site_xpos[cache['tip_L_sid']].copy()
    dist = np.linalg.norm(obj_pos - tip_pos)
    
    act = get_ik_action_dls(model, data, cache, obj_pos)
    
    center = (ARM_CTRL_HIGH + ARM_CTRL_LOW) / 2.0
    scale = (ARM_CTRL_HIGH - ARM_CTRL_LOW) / 2.0
    act_physical = act * scale + center
    
    data.ctrl[:2] = 0.0
    data.ctrl[ARM_CTRL_SLICE] = act_physical
    mujoco.mj_step(model, data)
    
    new_tip = data.site_xpos[cache['tip_L_sid']].copy()
    if step % 10 == 0 or step == 149:
        print(f"Step {step:03d} | Dist: {dist:.4f}m | Act_L: {act_physical[:4].round(3)} | Tip: {new_tip.round(3)}")
