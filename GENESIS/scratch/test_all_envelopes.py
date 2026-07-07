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

envelopes = {
    "default (carl_arm_train)": {
        "ox_range": (0.15, 0.20),
        "oy_range": (0.13, 0.17)
    },
    "candidate 1 (test_envelope)": {
        "ox_range": (0.08, 0.13),
        "oy_range": (0.13, 0.16)
    },
    "candidate 2 (test_safe_envelope)": {
        "ox_range": (0.14, 0.155),
        "oy_range": (0.08, 0.10)
    },
    "candidate 3 (test_safe_envelope_2)": {
        "ox_range": (0.13, 0.15),
        "oy_range": (0.11, 0.12)
    }
}

total = 30

for name, env in envelopes.items():
    successes = 0
    ox_low, ox_high = env["ox_range"]
    oy_low, oy_high = env["oy_range"]
    
    for ep in range(total):
        mujoco.mj_resetData(model, data)
        # Lock base
        data.qpos[cache['Q_CARL'] : cache['Q_CARL']+3] = [0.0, 0.0, 0.04]
        data.qpos[cache['Q_CARL']+3 : cache['Q_CARL']+7] = [1.0, 0.0, 0.0, 0.0]
        
        # Initialize arm correctly
        data.qpos[cache['arm_qpos'][0]] = 0.0
        data.qpos[cache['arm_qpos'][1]] = 0.3
        data.qpos[cache['arm_qpos'][2]] = -1.2
        data.qpos[cache['arm_qpos'][3]] = 0.0
        
        # Spawn object
        ox = rng.uniform(ox_low, ox_high)
        oy = rng.uniform(oy_low, oy_high)
        oz = 0.035
        data.qpos[cache['obj_qpos'] : cache['obj_qpos']+3] = [ox, oy, oz]
        mujoco.mj_forward(model, data)
        
        success = False
        for step in range(150):
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
            
    print(f"Envelope: {name:<35} | Success: {successes}/{total} ({successes/total*100:.1f}%)")
