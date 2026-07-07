import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))

import numpy as np
import mujoco
from carl_arm_train import build_cache, reset_episode, ARM_CTRL_LOW, ARM_CTRL_HIGH, ARM_CTRL_SLICE, get_state
from carl_arm_bc import get_ik_action

model = mujoco.MjModel.from_xml_path("vessel_kinetic.xml")
data = mujoco.MjData(model)
cache = build_cache(model)
rng = np.random.default_rng(seed=123)

successes = 0
total = 20

for ep in range(total):
    mujoco.mj_resetData(model, data)
    # Lock base
    data.qpos[cache['Q_CARL'] : cache['Q_CARL']+3] = [0.0, 0.0, 0.04]
    data.qpos[cache['Q_CARL']+3 : cache['Q_CARL']+7] = [1.0, 0.0, 0.0, 0.0]
    
    # Initialize arm correctly: yaw=0.0, pitch=0.3, elbow=-1.2, wrist=0.0
    data.qpos[cache['arm_qpos'][0]] = 0.0
    data.qpos[cache['arm_qpos'][1]] = 0.3
    data.qpos[cache['arm_qpos'][2]] = -1.2
    data.qpos[cache['arm_qpos'][3]] = 0.0
    
    # Spawn object in the candidate envelope: ox in [0.08, 0.13], oy in [0.13, 0.16], oz = 0.035
    ox = rng.uniform(0.08, 0.13)
    oy = rng.uniform(0.13, 0.16)
    oz = 0.035
    data.qpos[cache['obj_qpos'] : cache['obj_qpos']+3] = [ox, oy, oz]
    mujoco.mj_forward(model, data)
    
    success = False
    for step in range(150):
        # Lock base
        data.qpos[cache['Q_CARL'] : cache['Q_CARL']+3] = [0.0, 0.0, 0.04]
        data.qpos[cache['Q_CARL']+3 : cache['Q_CARL']+7] = [1.0, 0.0, 0.0, 0.0]
        data.qvel[cache['V_CARL'] : cache['V_CARL']+6] = 0.0
        
        obj_pos = data.xpos[cache['obj_bid']].copy()
        tip_pos = data.site_xpos[cache['tip_L_sid']].copy()
        dist = np.linalg.norm(obj_pos - tip_pos)
        
        if dist < 0.025:
            success = True
            break
            
        act = get_ik_action(model, data, cache, obj_pos)
        center = (ARM_CTRL_HIGH + ARM_CTRL_LOW) / 2.0
        scale = (ARM_CTRL_HIGH - ARM_CTRL_LOW) / 2.0
        act_physical = act * scale + center
        
        data.ctrl[:2] = 0.0
        data.ctrl[ARM_CTRL_SLICE] = act_physical
        mujoco.mj_step(model, data)
        
    if success:
        successes += 1
        print(f"Episode {ep:02d} | Success! Final Dist: {dist:.4f}m")
    else:
        print(f"Episode {ep:02d} | FAILED. Final Dist: {dist:.4f}m. Obj: {obj_pos.round(3)}, Tip: {tip_pos.round(3)}")

print(f"Total Successes: {successes}/{total} ({successes/total*100:.1f}%)")
