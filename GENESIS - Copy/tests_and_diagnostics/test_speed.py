"""Quick physics test: what speed can Bob actually achieve with 0.5 Nm motor actuators?"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import mujoco
import numpy as np

model = mujoco.MjModel.from_xml_path("vessel_kinetic.xml")
data = mujoco.MjData(model)
mujoco.mj_resetData(model, data)
mujoco.mj_forward(model, data)

carl_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "spatial_identity")
V_CARL = model.jnt_dofadr[carl_jid]

print(f"Total robot mass: {sum(model.body_mass):.3f} kg")
print(f"Wheel radius: 0.04 m")
print(f"Actuator ctrlrange: {model.actuator_ctrlrange[0]}")
print(f"Number of actuators: {model.nu}")

# Test: apply full torque forward and see how fast Bob goes
for step in range(2000):
    data.ctrl[0] = 0.5   # full forward left wheel
    data.ctrl[1] = 0.5   # full forward right wheel
    data.ctrl[2:8] = 0.0  # lock spine
    mujoco.mj_step(model, data)
    
    if step % 200 == 0:
        bp = data.geom("chassis").xpos
        wL = data.joint("w_left").qvel[0]
        wR = data.joint("w_right").qvel[0]
        v_body = np.linalg.norm(data.qvel[V_CARL : V_CARL + 2])  # XY velocity
        print(f"Step {step:4d} | Pos: [{bp[0]:.3f}, {bp[1]:.3f}, {bp[2]:.4f}] | "
              f"wL: {wL:.2f} wR: {wR:.2f} rad/s | body_v: {v_body:.4f} m/s")

print("\n--- Now testing with higher torque (2.0 Nm) ---")
model2 = mujoco.MjModel.from_xml_path("vessel_kinetic.xml")
model2.actuator_ctrlrange[0] = [-2.0, 2.0]
model2.actuator_ctrlrange[1] = [-2.0, 2.0]
data2 = mujoco.MjData(model2)
mujoco.mj_resetData(model2, data2)
mujoco.mj_forward(model2, data2)

carl_jid2 = mujoco.mj_name2id(model2, mujoco.mjtObj.mjOBJ_JOINT, "spatial_identity")
V_CARL2 = model2.jnt_dofadr[carl_jid2]

for step in range(2000):
    data2.ctrl[0] = 2.0
    data2.ctrl[1] = 2.0
    data2.ctrl[2:8] = 0.0
    mujoco.mj_step(model2, data2)
    
    if step % 200 == 0:
        bp = data2.geom("chassis").xpos
        wL = data2.joint("w_left").qvel[0]
        wR = data2.joint("w_right").qvel[0]
        v_body = np.linalg.norm(data2.qvel[V_CARL2 : V_CARL2 + 2])
        print(f"Step {step:4d} | Pos: [{bp[0]:.3f}, {bp[1]:.3f}, {bp[2]:.4f}] | "
              f"wL: {wL:.2f} wR: {wR:.2f} rad/s | body_v: {v_body:.4f} m/s")
