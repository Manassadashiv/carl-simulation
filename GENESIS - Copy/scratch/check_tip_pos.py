import mujoco
import numpy as np

model = mujoco.MjModel.from_xml_path("../carl_arm_standalone.xml")
data = mujoco.MjData(model)

# Set starting pose
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

mujoco.mj_forward(model, data)

tip_sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tip_site")
cube_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "obj_cube_0")

tip_pos = data.site_xpos[tip_sid].copy()
cube_pos = data.xpos[cube_bid].copy()

print("Tip Pos:", tip_pos)
print("Cube Pos:", cube_pos)
print("Distance:", np.linalg.norm(cube_pos - tip_pos))
