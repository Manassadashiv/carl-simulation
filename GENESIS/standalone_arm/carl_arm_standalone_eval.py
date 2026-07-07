"""
carl_arm_standalone_eval.py — Evaluate and visualize the trained Linear Policy arm controller with Centered Absolute Control.
"""

import os
import time
import numpy as np
import mujoco
import mujoco.viewer

MODEL_PATH = "carl_arm_standalone.xml"
WEIGHTS_PATH = "../memory/carl_arm_standalone_weights_v2.npz"

ACT_LO = np.array([-0.3, -2.4, -2.8, -1.57, -1.3])
ACT_HI = np.array([ 2.2,  2.4,  0.0,  1.57,  1.3])

from carl_arm_standalone_train import StandaloneArmPolicy, build_env_cache, get_observation, reset_env

def run_evaluation():
    if not os.path.exists(WEIGHTS_PATH):
        print(f"Error: Weights file '{WEIGHTS_PATH}' not found. Please train the policy first.")
        return

    # 1. Load model and data
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data  = mujoco.MjData(model)
    c     = build_env_cache(model)
    rng   = np.random.default_rng()

    # 2. Instantiate and load policy weights
    policy = StandaloneArmPolicy()
    saved = np.load(WEIGHTS_PATH)
    policy.set_params(saved['weights'])
    print(f"Successfully loaded trained Linear policy weights from: {WEIGHTS_PATH}")

    max_steps = 150

    # 3. Launch passive viewer
    print("\nStarting evaluation loop. Close the viewer window to exit.")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            print("\n" + "="*50)
            print("  New Episode: Resetting environment...")
            print("="*50)
            
            # Reset episode
            reset_env(model, data, c, rng)
            policy.reset()
            grasped = False
            viewer.sync()
            
            # Let it settle slightly
            time.sleep(0.5)

            # Rollout
            for step in range(max_steps):
                if not viewer.is_running():
                    break
                    
                s_norm = step / float(max_steps)
                s = get_observation(model, data, c, s_norm)
                
                # Forward pass
                raw_act = policy.forward(s)
                
                # Map raw policy outputs directly to absolute control targets
                action = ACT_LO + (raw_act + 1.0) * 0.5 * (ACT_HI - ACT_LO)

                # Write positioning commands
                for i in range(5):
                    data.ctrl[c['act_ids'][i]] = action[i]
                    
                # Spinal Grasp Reflex override
                touch_val = sum(data.sensordata[i] for i in c['touch_sids'])
                if touch_val > 0.02:
                    grasped = True

                if grasped:
                    data.ctrl[c['act_ids'][5]] = 1.4
                    data.ctrl[c['act_ids'][6]] = -1.4
                    data.ctrl[c['act_ids'][7]] = -1.4
                else:
                    data.ctrl[c['act_ids'][5]] = 0.0
                    data.ctrl[c['act_ids'][6]] = 0.0
                    data.ctrl[c['act_ids'][7]] = 0.0

                # Step physics
                mujoco.mj_step(model, data)
                
                # Synchronize viewer
                viewer.sync()
                time.sleep(0.015)

                # Diagnostic output
                if step % 15 == 0:
                    tip = data.site_xpos[c['tip_sid']]
                    cube = data.xpos[c['cube_bid']]
                    dist = np.linalg.norm(cube - tip)
                    print(f"  Step {step:03d} | Dist: {dist:.3f}m | Touch: {touch_val:.3f} | Grasped: {grasped} | Cube Z: {cube[2]:.3f}m")

            print("Episode finished. Waiting before next reset...")
            time.sleep(2.0)

if __name__ == "__main__":
    run_evaluation()
