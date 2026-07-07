import mujoco, numpy as np
model = mujoco.MjModel.from_xml_path('carl_primate_scout.xml')
data = mujoco.MjData(model)

root_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'root_joint')
Q = model.jnt_qposadr[root_jid]
data.qpos[Q:Q+3] = [0, 0, 0.058]

ST = mujoco.mjtObj.mjOBJ_SITE
JT = mujoco.mjtObj.mjOBJ_JOINT
BD = mujoco.mjtObj.mjOBJ_BODY
tip_L = mujoco.mj_name2id(model, ST, 'touch_index_L')

jnames = ['shoulder_yaw_L','shoulder_pitch_L','elbow_L','wrist_roll_L','wrist_pitch_L']
jids = [mujoco.mj_name2id(model, JT, n) for n in jnames]
qaddrs = [model.jnt_qposadr[j] for j in jids]

poses = {
    'REST':    [0.3, 0.4, -1.4, 0, 0],
    'EXTEND':  [0.8, 0.0, -0.3, 0, 0],
    'REACH':   [0.5, 0.8, -0.5, 0, -0.5],
    'DOWN':    [0.5, 1.5, -0.8, 0, 0],
    'FORWARD': [1.0, 0.0, -0.2, 0, 0],
}

for name, vals in poses.items():
    for a, v in zip(qaddrs, vals):
        data.qpos[a] = v
    mujoco.mj_forward(model, data)
    pos = data.site_xpos[tip_L]
    print(f"{name:8s} tip_L: x={pos[0]:.3f} y={pos[1]:.3f} z={pos[2]:.3f}")

chest_bid = mujoco.mj_name2id(model, BD, 'spine_3')
sh_bid = mujoco.mj_name2id(model, BD, 'shoulder_mount_L')
mujoco.mj_forward(model, data)
print(f"\nChest:    {data.xpos[chest_bid]}")
print(f"Shoulder: {data.xpos[sh_bid]}")
