"""
FINAL VERIFICATION: Realistic masses + velocity actuators + line-follower speed.
Each test creates fresh model/data/brainstem to avoid cached state corruption.
"""
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, '.')
import mujoco
import numpy as np
from carl_brainstem import BrainstemController

def fresh_sim():
    """Create completely fresh model, data, and brainstem."""
    model = mujoco.MjModel.from_xml_path('vessel_kinetic.xml')
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    carl_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "spatial_identity")
    Q_CARL = model.jnt_qposadr[carl_jid]
    V_CARL = model.jnt_dofadr[carl_jid]
    data.qpos[Q_CARL] = -3.0  # Spawn far back to prevent hitting walls in speed tests
    mujoco.mj_forward(model, data)
    for _ in range(200):
        mujoco.mj_step(model, data)
    return model, data, BrainstemController(), Q_CARL, V_CARL

# ============================================================
print("=" * 72)
print("  CARL GENESIS - Physics Verification Suite v2")
print("=" * 72)

# ============================================================
# TEST 0: Mass Budget
# ============================================================
model, data, _, Q_CARL, V_CARL = fresh_sim()
print("\n--- MASS BUDGET ---")
deck_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'drive_deck')
robot_mass = model.body_subtreemass[deck_id]
print(f"  Robot (drive_deck subtree): {robot_mass:.3f} kg")
print(f"  Robot weight:              {robot_mass * 9.81:.2f} N")

# ============================================================
# TEST 1: Traction & Weight Distribution
# ============================================================
print("\n--- TRACTION & WEIGHT DISTRIBUTION ---")
wheel_normal = 0; caster_normal = 0
for i in range(data.ncon):
    c = data.contact[i]
    g1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1)
    b2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[c.geom2])
    if g1 == 'floor':
        force = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, force)
        if b2 and 'wheel' in b2:
            wheel_normal += force[0]
        elif b2 and 'caster' in b2:
            caster_normal += force[0]
total = wheel_normal + caster_normal
print(f"  Wheels:  {wheel_normal:.1f} N ({wheel_normal/total*100:.0f}%)")
print(f"  Casters: {caster_normal:.1f} N ({caster_normal/total*100:.0f}%)")
traction_per_wheel = 1.4 * (wheel_normal / 4) * 0.04  # 4 contact points, 2 wheels
print(f"  Traction limit/wheel: {traction_per_wheel:.3f} Nm")
print(f"  Motor forcerange:     0.35 Nm")
print(f"  Safety margin:        {(traction_per_wheel*2 - 0.35)/0.35*100:.0f}%")

# ============================================================
# TEST 2: Open-loop speed (direct ctrl, no brainstem)
# ============================================================
print("\n--- OPEN-LOOP SPEED TEST ---")
print(f"  {'Target (rad/s)':>14s} | {'Achieved wL':>12s} | {'Chassis v':>10s} | {'Track%':>7s}")
print("  " + "-" * 55)
for target_w in [5.0, 10.0, 15.0, 25.0]:
    model, data, _, Q_CARL, V_CARL = fresh_sim()
    for step in range(300):
        data.ctrl[0] = target_w; data.ctrl[1] = target_w; data.ctrl[2:8] = 0.0
        mujoco.mj_step(model, data)
    wL = data.joint('w_left').qvel[0]
    cv = data.qvel[V_CARL]
    expected = target_w * 0.04
    tracking = cv / expected * 100 if expected > 0 else 0
    print(f"  {target_w:14.1f} | {wL:12.2f} | {cv:10.4f} | {tracking:6.1f}%")

# ============================================================
# TEST 3: Brainstem speed ramp (0 -> target)
# ============================================================
print("\n--- BRAINSTEM SPEED TRACKING ---")
print(f"  {'Target':>8s} | {'Achieved':>10s} | {'Wheel':>8s} | {'Track%':>7s} | {'0-90% time':>11s}")
print("  " + "-" * 60)
for target_speed in [0.2, 0.4, 0.6, 0.8, 1.0]:
    model, data, bs, Q_CARL, V_CARL = fresh_sim()
    lidar = [5.0] * 24
    last_v = 0.0
    t90 = None
    for step in range(500):
        last_v = 0.5 * last_v + 0.5 * target_speed  # fast S-curve
        wL = data.joint('w_left').qvel[0]
        wR = data.joint('w_right').qvel[0]
        cmds, _ = bs.compute_torques(lidar, np.array([wL, wR]), last_v, 0.0, data.qvel[V_CARL+5])
        data.ctrl[0] = cmds[0]; data.ctrl[1] = cmds[1]; data.ctrl[2:8] = 0.0
        mujoco.mj_step(model, data)
        if t90 is None and abs(data.qvel[V_CARL]) >= 0.9 * target_speed:
            t90 = step * 0.01
    cv = data.qvel[V_CARL]
    wL_f = data.joint('w_left').qvel[0]
    tracking = cv / target_speed * 100
    t90_str = f"{t90:.2f}s" if t90 else ">5s"
    print(f"  {target_speed:7.1f}  | {cv:9.3f}  | {wL_f:7.2f}  | {tracking:6.1f}% | {t90_str:>11s}")

# ============================================================
# TEST 4: Jitter at cruise
# ============================================================
print("\n--- JITTER TEST (cruise at 0.4 m/s, 3 seconds) ---")
model, data, bs, Q_CARL, V_CARL = fresh_sim()
lidar = [5.0] * 24
velocities = []
for step in range(300):
    v_cmd = 0.4 if step > 30 else step * 0.4 / 30
    wL = data.joint('w_left').qvel[0]
    wR = data.joint('w_right').qvel[0]
    cmds, _ = bs.compute_torques(lidar, np.array([wL, wR]), v_cmd, 0.0, data.qvel[V_CARL+5])
    data.ctrl[0] = cmds[0]; data.ctrl[1] = cmds[1]; data.ctrl[2:8] = 0.0
    mujoco.mj_step(model, data)
    if step >= 100:
        velocities.append(data.qvel[V_CARL])

vel_arr = np.array(velocities)
print(f"  Mean:        {vel_arr.mean():.4f} m/s")
print(f"  Std:         {vel_arr.std():.6f}")
print(f"  Peak-to-peak:{vel_arr.max() - vel_arr.min():.6f}")
if vel_arr.std() < 0.02:
    print("  [PASS] Smooth and stable.")
elif vel_arr.std() < 0.05:
    print("  [PASS] Minor oscillation, acceptable.")
else:
    print(f"  [FAIL] Jitter: std={vel_arr.std():.4f}")

# ============================================================
# TEST 5: Emergency braking
# ============================================================
print("\n--- EMERGENCY BRAKING (0.8 m/s -> 0) ---")
model, data, bs, Q_CARL, V_CARL = fresh_sim()
lidar = [5.0] * 24
# Accelerate
for step in range(200):
    v_cmd = min(0.8, step * 0.8 / 30)
    wL = data.joint('w_left').qvel[0]; wR = data.joint('w_right').qvel[0]
    cmds, _ = bs.compute_torques(lidar, np.array([wL, wR]), v_cmd, 0.0, data.qvel[V_CARL+5])
    data.ctrl[0] = cmds[0]; data.ctrl[1] = cmds[1]; data.ctrl[2:8] = 0.0
    mujoco.mj_step(model, data)
pre_v = data.qvel[V_CARL]
# Brake
stop_t = None
for step in range(200):
    wL = data.joint('w_left').qvel[0]; wR = data.joint('w_right').qvel[0]
    cmds, _ = bs.compute_torques(lidar, np.array([wL, wR]), 0.0, 0.0, data.qvel[V_CARL+5])
    data.ctrl[0] = cmds[0]; data.ctrl[1] = cmds[1]; data.ctrl[2:8] = 0.0
    mujoco.mj_step(model, data)
    if stop_t is None and abs(data.qvel[V_CARL]) < 0.02:
        stop_t = step * 0.01
print(f"  Pre-brake speed: {pre_v:.3f} m/s")
print(f"  Stopped in:      {stop_t:.2f}s" if stop_t else "  Did NOT stop!")

# ============================================================
# TEST 6: Sharp turn
# ============================================================
print("\n--- SHARP TURN (0.4 m/s + 3 rad/s) ---")
model, data, bs, Q_CARL, V_CARL = fresh_sim()
lidar = [5.0] * 24
for step in range(200):
    v_cmd = 0.4; w_cmd = 3.0 if step > 30 else 0.0
    wL = data.joint('w_left').qvel[0]; wR = data.joint('w_right').qvel[0]
    cmds, _ = bs.compute_torques(lidar, np.array([wL, wR]), v_cmd, w_cmd, data.qvel[V_CARL+5])
    data.ctrl[0] = cmds[0]; data.ctrl[1] = cmds[1]; data.ctrl[2:8] = 0.0
    mujoco.mj_step(model, data)
print(f"  Yaw rate: {data.qvel[V_CARL+5]:.2f} rad/s (target: 3.0)")
print(f"  Z height: {data.qpos[Q_CARL+2]:.4f}m")
if data.qpos[Q_CARL+2] > 0.03:
    print("  [PASS] Stable through turn.")
else:
    print("  [FAIL] Tipped!")

print("\n" + "=" * 72)
print("  ALL TESTS COMPLETE")
print("=" * 72)
