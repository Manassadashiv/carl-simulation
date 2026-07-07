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

# Force spawn position to ox=0.08, oy=0.13, oz=0.035
ox, oy, oz = 0.08, 0.13, 0.035
data.qpos[cache['obj_qpos'] : cache['obj_qpos']+3] = [ox, oy, oz]
data.qpos[cache['obj_qpos']+3 : cache['obj_qpos']+7] = [1.0, 0.0, 0.0, 0.0]
mujoco.mj_forward(model, data)

print("Initial Obj Pos:", data.xpos[cache['obj_bid']].copy())

for step in range(150):
    # Lock base
    data.qpos[cache['Q_CARL'] : cache['Q_CARL']+3] = [0.0, 0.0, 0.04]
    data.qpos[cache['Q_CARL']+3 : cache['Q_CARL']+7] = [1.0, 0.0, 0.0, 0.0]
    data.qvel[cache['V_CARL'] : cache['V_CARL']+6] = 0.0
    
    obj_pos = data.xpos[cache['obj_bid']].copy()
    tip_pos = data.site_xpos[cache['tip_L_sid']].copy()
    dist = np.linalg.norm(obj_pos - tip_pos)
    
    if dist < 0.025:
        print(f"CONVERGED at step {step} | Dist: {dist:.4f}m | Obj: {obj_pos.round(3)}, Tip: {tip_pos.round(3)}")
        break
        
    act = get_ik_action(model, data, cache, obj_pos)
    center = (ARM_CTRL_HIGH + ARM_CTRL_LOW) / 2.0
    scale = (ARM_CTRL_HIGH - ARM_CTRL_LOW) / 2.0
    act_physical = act * scale + center
    
    data.ctrl[:2] = 0.0
    data.ctrl[ARM_CTRL_SLICE] = act_physical
    mujoco.mj_step(model, data)
else:
    print(f"FAILED TO CONVERGE. Final Dist: {dist:.4f}m. Obj: {obj_pos.round(3)}, Tip: {tip_pos.round(3)}")
