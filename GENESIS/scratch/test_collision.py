import os
import sys
import numpy as np
import mujoco

sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))

from carl_arm_train import build_cache
from carl_industrial_motion import CartesianArmExecutor
from carl_jenga_demo import resolve_objects

model = mujoco.MjModel.from_xml_path("vessel_kinetic.xml")
data = mujoco.MjData(model)
cache = build_cache(model)
objs = resolve_objects(model)

# Reset positions as in carl_jenga_demo
mujoco.mj_resetData(model, data)
data.qpos[cache['Q_CARL'] : cache['Q_CARL']+3] = [0.0, 0.0, 0.04]
data.qpos[objs['obj_cube_0']['qpos_adr'] : objs['obj_cube_0']['qpos_adr']+3] = [0.18, 0.14, 0.035]
data.qpos[objs['obj_cube_1']['qpos_adr'] : objs['obj_cube_1']['qpos_adr']+3] = [0.15, 0.17, 0.035]
data.qpos[objs['obj_cube_2']['qpos_adr'] : objs['obj_cube_2']['qpos_adr']+3] = [0.12, 0.13, 0.035]
mujoco.mj_forward(model, data)

arm = CartesianArmExecutor(model, data, cache, side="L")

# Reset left arm to starting pose
data.qpos[cache['arm_qpos'][0]] = 0.0
data.qpos[cache['arm_qpos'][1]] = 0.3
data.qpos[cache['arm_qpos'][2]] = -1.2
data.qpos[cache['arm_qpos'][3]] = 0.0
mujoco.mj_forward(model, data)

# 1. Approach cube
cube_pos = data.xpos[objs['obj_cube_0']['bid']].copy()
target_pos = cube_pos + [0, 0, 0.08]
arm.queue_waypoint(target_pos, duration=1.2)
dt = 0.01

def lock_base():
    data.qpos[cache['Q_CARL'] : cache['Q_CARL']+3] = [0.0, 0.0, 0.04]
    data.qpos[cache['Q_CARL']+3 : cache['Q_CARL']+7] = [1.0, 0.0, 0.0, 0.0]
    data.qvel[cache['V_CARL'] : cache['V_CARL']+6] = 0.0

# Run approach
while not arm.is_settled(pos_tol=0.0035):
    lock_base()
    cmd = arm.step(dt=dt, grasp=False, hold_force=0.0)
    data.ctrl[6:11] = cmd
    data.ctrl[11:16] = [0.0, 0.3, -1.2, 0.0, 0.0]
    mujoco.mj_step(model, data)

# 2. Descent
target_pos = cube_pos + [0, 0, 0.005]
arm.queue_waypoint(target_pos, duration=0.8)
while not arm.is_settled(pos_tol=0.0035):
    lock_base()
    cmd = arm.step(dt=dt, grasp=False, hold_force=0.0)
    data.ctrl[6:11] = cmd
    mujoco.mj_step(model, data)

# 3. Grasp
for _ in range(40):
    lock_base()
    cmd = arm.step(dt=dt, grasp=True, hold_force=0.6)
    data.ctrl[6:11] = cmd
    mujoco.mj_step(model, data)

# 4. Lift
target_pos = cube_pos + [0, 0, 0.08]
arm.queue_waypoint(target_pos, duration=1.0)
while not arm.is_settled(pos_tol=0.0035):
    lock_base()
    cmd = arm.step(dt=dt, grasp=True, hold_force=0.6)
    data.ctrl[6:11] = cmd
    mujoco.mj_step(model, data)

# 5. Horizontal Transfer
stack_center = np.array([0.16, 0.10])
stack_target = np.array([stack_center[0], stack_center[1], 0.035])
hover_above_stack = stack_target + [0, 0, 0.12]

print("Starting Horizontal Transfer...")
print("Target hover_above_stack:", hover_above_stack)
arm.queue_waypoint(hover_above_stack, duration=1.5)

step = 0
while not arm.is_settled(pos_tol=0.0035) and step < 300:
    lock_base()
    cmd = arm.step(dt=dt, grasp=True, hold_force=0.6)
    data.ctrl[6:11] = cmd
    mujoco.mj_step(model, data)
    
    tip_pos = data.site_xpos[arm.tip_sid].copy()
    dist = np.linalg.norm(hover_above_stack - tip_pos)
    
    # Check contacts
    collisions = []
    for i in range(data.ncon):
        con = data.contact[i]
        geom1_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, con.geom1) or "unknown"
        geom2_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, con.geom2) or "unknown"
        
        # We only care about self-collisions of CARL
        if any(g in geom1_name or g in geom2_name for g in ["torso", "upper_arm", "forearm", "gripper", "finger", "chassis"]):
            # Filter out expected contact with the grasped cube (gcube_0)
            if not ("gcube_0" in geom1_name or "gcube_0" in geom2_name):
                collisions.append(f"{geom1_name} <-> {geom2_name}")
                
    if step % 20 == 0 or len(collisions) > 0 or step < 5:
        print(f"Step {step:03d} | Dist: {dist:.4f}m | Tip: {tip_pos.round(4)} | Collisions: {collisions}")
        if len(collisions) > 0:
            print("COLLISION DETECTED! Stopping test.")
            break
            
    step += 1

print(f"Test finished. Final Step: {step} | Final Dist: {dist:.4f}m")
