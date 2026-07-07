import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))

import numpy as np
import mujoco
from carl_arm_train import build_cache, reset_episode, ARM_CTRL_LOW, ARM_CTRL_HIGH, ARM_CTRL_SLICE
from carl_arm_bc import get_ik_action

model = mujoco.MjModel.from_xml_path("vessel_kinetic.xml")
data = mujoco.MjData(model)
cache = build_cache(model)
rng = np.random.default_rng(seed=123)

reset_episode(model, data, cache, "reach", rng)

# Force spawn position to ox=0.15, oy=0.11, oz=0.035
ox, oy, oz = 0.15, 0.11, 0.035
data.qpos[cache['obj_qpos'] : cache['obj_qpos']+3] = [ox, oy, oz]
data.qpos[cache['obj_qpos']+3 : cache['obj_qpos']+7] = [1.0, 0.0, 0.0, 0.0]
mujoco.mj_forward(model, data)

# Initialize arm like test_safe_envelope
data.qpos[cache['arm_qpos'][0]] = 0.3
data.qpos[cache['arm_qpos'][1]] = -1.2
data.qpos[cache['arm_qpos'][2]] = 0.0
mujoco.mj_forward(model, data)

for step in range(60):
    # Lock base
    data.qpos[cache['Q_CARL'] : cache['Q_CARL']+3] = [0.0, 0.0, 0.04]
    data.qpos[cache['Q_CARL']+3 : cache['Q_CARL']+7] = [1.0, 0.0, 0.0, 0.0]
    data.qvel[cache['V_CARL'] : cache['V_CARL']+6] = 0.0
    
    obj_pos = data.xpos[cache['obj_bid']].copy()
    tip_pos = data.site_xpos[cache['tip_L_sid']].copy()
    dist = np.linalg.norm(obj_pos - tip_pos)
    
    current_q = np.array([data.qpos[a] for a in cache['arm_qpos'][:5]])
    
    # Let's compute IK manually to print J and dq
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    tip_sid = cache['tip_L_sid']
    mujoco.mj_jacSite(model, data, jacp, jacr, tip_sid)
    left_arm_dofs = cache['arm_dof'][:5]
    J = jacp[:, left_arm_dofs]
    
    dx = obj_pos - tip_pos
    lambda_sq = 0.015
    J_dls = J.T @ np.linalg.inv(J @ J.T + lambda_sq * np.eye(3))
    dq = J_dls @ dx
    
    act = get_ik_action(model, data, cache, obj_pos)
    center = (ARM_CTRL_HIGH + ARM_CTRL_LOW) / 2.0
    scale = (ARM_CTRL_HIGH - ARM_CTRL_LOW) / 2.0
    act_physical = act * scale + center
    
    print(f"Step {step:02d} | Obj: {obj_pos.round(3)} | Tip: {tip_pos.round(3)} | Dist: {dist:.4f}m")
    print(f"       qpos: {current_q.round(3)} | ctrl: {act_physical[:5].round(3)}")
    print(f"       dx: {dx.round(4)} | dq: {dq.round(4)}")
    print(f"       J: {J.round(3).tolist()}")
    
    if dist < 0.025:
        print("CONVERGED!")
        break
        
    data.ctrl[:2] = 0.0
    data.ctrl[ARM_CTRL_SLICE] = act_physical
    mujoco.mj_step(model, data)
