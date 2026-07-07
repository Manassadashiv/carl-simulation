import mujoco
import numpy as np

model = mujoco.MjModel.from_xml_path('vessel_kinetic.xml')
data = mujoco.MjData(model)

carl_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "spatial_identity")
Q_CARL = model.jnt_qposadr[carl_jid]

data.qpos[Q_CARL+2] = 0.04
mujoco.mj_forward(model, data)
mujoco.mj_step(model, data)

head_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "head")

for i in range(500):
    mujoco.mj_step(model, data)
    head_z = data.xpos[head_body_id][2]
    data.ctrl[0] = 1.25
    data.ctrl[1] = 1.25
    if i % 50 == 0:
        w, x, y, z = data.qpos[Q_CARL+3:Q_CARL+7]
        pitch = np.arcsin(2 * (w*y - x*z))
        roll = np.arctan2(2 * (w*x + y*z), 1 - 2 * (x**2 + y**2))
        print(f"Step {i}, X: {data.qpos[Q_CARL]:.3f}, Y: {data.qpos[Q_CARL+1]:.3f}, head Z: {head_z:.3f}, pitch: {pitch:.3f}, roll: {roll:.3f}")

print("Final X:", data.qpos[Q_CARL])
