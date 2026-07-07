import mujoco
import numpy as np
import os

model_path = os.path.join(os.path.dirname(__file__), '..', 'carl_arm_standalone.xml')
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

cube_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "obj_cube_0")
palm_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "wrist_L")

acts = {
    "yaw": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_shoulder_yaw_L"),
    "pitch": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_shoulder_pitch_L"),
    "elbow": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_elbow_L"),
    "w_roll": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_wrist_roll_L"),
    "w_pitch": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_wrist_pitch_L")
}

yaw_dof = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "shoulder_yaw_L")
pitch_dof = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "shoulder_pitch_L")
elbow_dof = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "elbow_L")

# Initialize
print("Actuator IDs:", acts)
data.ctrl[acts["elbow"]] = -0.5
data.ctrl[acts["pitch"]] = 0.2
mujoco.mj_step(model, data)

for step in range(10):
    expected_sensory_pos = data.xpos[cube_body].copy()
    current_sensory_pos = data.xpos[palm_body].copy()
    error_vector = expected_sensory_pos - current_sensory_pos
    
    jacp = np.zeros((3, model.nv))
    mujoco.mj_jacBody(model, data, jacp, None, palm_body)
    
    # Slice to the first 5 joints of the arm (yaw, pitch, elbow, wrist_roll, wrist_pitch)
    J = jacp[:, :5]
    
    # Damped Least Squares formulation
    lambda_sq = 0.015
    J_dls = J.T @ np.linalg.inv(J @ J.T + lambda_sq * np.eye(3))
    
    step_scale = 0.15
    joint_delta = J_dls.dot(error_vector) * step_scale
    
    # Clip joint delta to prevent massive jumps on single frames
    joint_delta = np.clip(joint_delta, -0.05, 0.05)
    
    d_yaw = joint_delta[0]
    d_pitch = joint_delta[1]
    d_elbow = joint_delta[2]
    
    yaw_q = data.qpos[model.jnt_qposadr[yaw_dof]]
    pitch_q = data.qpos[model.jnt_qposadr[pitch_dof]]
    elbow_q = data.qpos[model.jnt_qposadr[elbow_dof]]
    
    print(f"--- Step {step} ---")
    print(f"Cube Pos: {expected_sensory_pos}")
    print(f"Palm Pos: {current_sensory_pos}")
    print(f"Error: {error_vector} (norm: {np.linalg.norm(error_vector):.4f})")
    print(f"Joint Deltas -> Yaw: {d_yaw:.4f}, Pitch: {d_pitch:.4f}, Elbow: {d_elbow:.4f}")
    
    data.ctrl[acts["yaw"]] += d_yaw
    data.ctrl[acts["pitch"]] += d_pitch
    data.ctrl[acts["elbow"]] += d_elbow
    
    data.ctrl[acts["yaw"]] = np.clip(data.ctrl[acts["yaw"]], -0.3, 2.2)
    data.ctrl[acts["pitch"]] = np.clip(data.ctrl[acts["pitch"]], -2.4, 2.4)
    data.ctrl[acts["elbow"]] = np.clip(data.ctrl[acts["elbow"]], -2.8, 0)
    
    data.ctrl[acts["w_roll"]] = 1.5708
    data.ctrl[acts["w_pitch"]] = 0.5
    
    print(f"Qpos  -> Yaw: {yaw_q:.4f}, Pitch: {pitch_q:.4f}, Elbow: {elbow_q:.4f}")
    print(f"Ctrls -> Yaw: {data.ctrl[acts['yaw']]:.4f}, Pitch: {data.ctrl[acts['pitch']]:.4f}, Elbow: {data.ctrl[acts['elbow']]:.4f}")
    
    # Run 10 physics sub-steps to let the arm actually move towards the target ctrl
    for _ in range(10):
        mujoco.mj_step(model, data)
