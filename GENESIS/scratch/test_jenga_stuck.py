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

# Target pre-grasp hover position
cube_pos = data.xpos[objs['obj_cube_0']['bid']].copy()
target_pos = cube_pos + [0, 0, 0.08]

print("Starting pre-grasp hover test...")
print("Initial cube pos:", cube_pos)
print("Target pos:", target_pos)
print("Initial tip pos:", data.site_xpos[arm.tip_sid].copy())

arm.queue_waypoint(target_pos, duration=1.2)
dt = 0.01

def lock_base():
    data.qpos[cache['Q_CARL'] : cache['Q_CARL']+3] = [0.0, 0.0, 0.04]
    data.qpos[cache['Q_CARL']+3 : cache['Q_CARL']+7] = [1.0, 0.0, 0.0, 0.0]
    data.qvel[cache['V_CARL'] : cache['V_CARL']+6] = 0.0

step = 0
while not arm.is_settled(pos_tol=0.0035) and step < 200:
    lock_base()
    
    cmd = arm.step(dt=dt, grasp=False, hold_force=0.0)
    data.ctrl[6:11] = cmd
    data.ctrl[11:16] = [0.0, 0.3, -1.2, 0.0, 0.0]
    
    mujoco.mj_step(model, data)
    
    tip_pos = data.site_xpos[arm.tip_sid].copy()
    dist = np.linalg.norm(target_pos - tip_pos)
    
    if step % 20 == 0 or step < 5:
        print(f"Step {step:03d} | t: {arm.t_in_segment:.2f}s | Dist: {dist:.4f}m | Settled: {arm.is_settled(pos_tol=0.0035)}")
    step += 1

print(f"Final Step: {step} | Final Dist: {dist:.4f}m | Done: {arm.segment.done(arm.t_in_segment)} | Settled: {arm.is_settled(pos_tol=0.0035)}")


