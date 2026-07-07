import os
import sys
import numpy as np
import mujoco

sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))

from carl_arm_train import build_cache, reset_episode, ARM_CTRL_LOW, ARM_CTRL_HIGH, ARM_CTRL_SLICE, ArmPolicy, get_state

def debug_policy():
    model = mujoco.MjModel.from_xml_path("vessel_kinetic.xml")
    data = mujoco.MjData(model)
    cache = build_cache(model)
    rng = np.random.default_rng(seed=42)
    
    weights_path = "memory/carl_arm_weights.npz"
    if not os.path.exists(weights_path):
        print("No weights found!")
        return
        
    saved = np.load(weights_path)
    policy = ArmPolicy()
    policy.set_params(saved['weights'])
    
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
        
        policy.x = np.zeros(policy.HIDDEN_DIM)
        
        success = False
        for step in range(250):
            data.qpos[cache['Q_CARL'] : cache['Q_CARL']+3] = [0.0, 0.0, 0.04]
            data.qpos[cache['Q_CARL']+3 : cache['Q_CARL']+7] = [1.0, 0.0, 0.0, 0.0]
            data.qvel[cache['V_CARL'] : cache['V_CARL']+6] = 0.0
            
            s = get_state(model, data, cache)
            raw = policy.forward(s, dt=0.01)
            
            act_physical = ARM_CTRL_LOW + (raw + 1.0) * 0.5 * (ARM_CTRL_HIGH - ARM_CTRL_LOW)
            
            data.ctrl[:2] = 0.0
            data.ctrl[ARM_CTRL_SLICE] = act_physical
            mujoco.mj_step(model, data)
            
            dist = np.linalg.norm(data.xpos[cache['obj_bid']] - data.site_xpos[cache['tip_L_sid']])
            if dist < 0.045:
                success = True
                break
                
        if success:
            successes += 1
            print(f"Trial {ep} succeeded in {step} steps.")
        else:
            print(f"Trial {ep} failed! Final dist: {dist:.4f}")
            
    print(f"Success rate: {successes}/{trials}")

if __name__ == "__main__":
    debug_policy()
