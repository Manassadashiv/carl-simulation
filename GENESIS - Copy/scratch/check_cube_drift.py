import mujoco
import numpy as np

model = mujoco.MjModel.from_xml_path("../carl_arm_standalone.xml")
data = mujoco.MjData(model)

# Initialize pose
qpos_yaw = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "shoulder_yaw_L")]
qpos_pitch = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "shoulder_pitch_L")]
qpos_elbow = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "elbow_L")]
qpos_roll = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "wrist_roll_L")]
qpos_wrist_pitch = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "wrist_pitch_L")]

data.qpos[qpos_yaw] = 0.0
data.qpos[qpos_pitch] = 0.5
data.qpos[qpos_elbow] = -0.8
data.qpos[qpos_roll] = 1.5708
data.qpos[qpos_wrist_pitch] = 0.35

cube_qpos = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "fj_cube_0")]
data.qpos[cube_qpos : cube_qpos+3] = [0.21, 0.0, 0.14]

# Set actuator controls to hold start pose
init_ctrl = [0.0, 0.5, -0.8, 1.5708, 0.35, 0.0, 0.0, 0.0]
for i, val in enumerate(init_ctrl):
    data.ctrl[i] = val

mujoco.mj_forward(model, data)

cube_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "obj_cube_0")

print("Starting simulation steps...")
for step in range(150):
    mujoco.mj_step(model, data)
    if step % 25 == 0 or step == 149:
        cube_pos = data.xpos[cube_bid].copy()
        print(f"Step {step:03d} | Cube Pos: {np.round(cube_pos, 4)}")
