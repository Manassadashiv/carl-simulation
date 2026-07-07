import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))

import numpy as np
import mujoco
from carl_arm_train import build_cache, reset_episode, compute_reward, ARM_CTRL_LOW, ARM_CTRL_HIGH, ARM_CTRL_SLICE, get_state
from carl_arm_bc import get_ik_action

model = mujoco.MjModel.from_xml_path("vessel_kinetic.xml")
data = mujoco.MjData(model)
cache = build_cache(model)
rng = np.random.default_rng(seed=42)

reset_episode(model, data, cache, "reach", rng)
print("Target Pos:", data.xpos[cache['obj_bid']].copy())

for step in range(150):
    data.qpos[cache['Q_CARL'] : cache['Q_CARL']+3] = [0.0, 0.0, 0.04]
    data.qpos[cache['Q_CARL']+3 : cache['Q_CARL']+7] = [1.0, 0.0, 0.0, 0.0]
    data.qvel[cache['V_CARL'] : cache['V_CARL']+6] = 0.0
    
    obj_pos = data.xpos[cache['obj_bid']].copy()
    tip_pos = data.site_xpos[cache['tip_L_sid']].copy()
    dist = np.linalg.norm(obj_pos - tip_pos)
    
    act = get_ik_action(model, data, cache, obj_pos)
    center = (ARM_CTRL_HIGH + ARM_CTRL_LOW) / 2.0
    scale = (ARM_CTRL_HIGH - ARM_CTRL_LOW) / 2.0
    act_physical = act * scale + center
    
    data.ctrl[:2] = 0.0
    data.ctrl[ARM_CTRL_SLICE] = act_physical
    mujoco.mj_step(model, data)
    
    _, done, dist_new = compute_reward(data, cache, "reach")
    print(f"Step {step:03d} | Dist: {dist:.4f}m -> {dist_new:.4f}m | Done: {done}")
    if done and dist_new < 0.025:
        print("SUCCESS")
        break
else:
    print("FAILED TO CONVERGE")
