import mujoco
import mujoco.viewer
import time
import numpy as np
import os

model_path = os.path.join(os.path.dirname(__file__), '..', 'carl_arm_standalone.xml')
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

# Find actuator IDs
yaw_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_shoulder_yaw_L")
pitch_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_shoulder_pitch_L")
elbow_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_elbow_L")
wroll_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_wrist_roll_L")
wpitch_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_wrist_pitch_L")
thumb_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_thumb_L")
index_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_index_L")
mid_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_middle_L")

print("Launching arm flexibility demonstration...")

with mujoco.viewer.launch_passive(model, data) as viewer:
    start_time = time.time()
    # Run for 30 seconds
    while viewer.is_running() and time.time() - start_time < 30:
        t = time.time() - start_time
        
        # 1. Shoulder Yaw (swinging left/right)
        data.ctrl[yaw_id] = 0.95 + 1.0 * np.sin(t * 1.5)
        
        # 2. Shoulder Pitch (swinging forward/back)
        data.ctrl[pitch_id] = 0.5 * np.sin(t * 1.1)
        
        # 3. Elbow (bending)
        data.ctrl[elbow_id] = -1.4 + 1.2 * np.sin(t * 1.8)
        
        # 4. Wrist Roll (twisting the wrist)
        data.ctrl[wroll_id] = 1.5 * np.sin(t * 2.5)
        
        # 5. Wrist Pitch (waving)
        data.ctrl[wpitch_id] = 1.0 * np.sin(t * 3.0)
        
        # 6. Fingers (opening and closing repeatedly)
        finger_val = 0.7 + 0.7 * np.sin(t * 4.0) 
        data.ctrl[thumb_id] = finger_val
        data.ctrl[index_id] = -finger_val
        data.ctrl[mid_id] = -finger_val
        
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.005)

print("Demonstration complete.")
