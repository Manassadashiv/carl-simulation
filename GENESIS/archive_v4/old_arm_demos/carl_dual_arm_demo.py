import os
import sys
import time
import numpy as np
import mujoco
from mujoco import viewer as mj_viewer

# Ensure workspace root is in path
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))

from carl_arm_train import build_cache, ARM_CTRL_LOW, ARM_CTRL_HIGH, ARM_CTRL_SLICE
from carl_industrial_motion import CartesianArmExecutor

def resolve_objects(model):
    BD = mujoco.mjtObj.mjOBJ_BODY
    JT = mujoco.mjtObj.mjOBJ_JOINT
    
    objects = {}
    names = ['obj_cube_0', 'obj_cube_1', 'obj_cube_2', 'obj_ball_0', 'obj_ball_1', 'obj_toy']
    jnames = ['fj_cube_0', 'fj_cube_1', 'fj_cube_2', 'fj_ball_0', 'fj_ball_1', 'fj_toy']
    
    for name, jname in zip(names, jnames):
        bid = mujoco.mj_name2id(model, BD, name)
        jid = mujoco.mj_name2id(model, JT, jname)
        if bid != -1 and jid != -1:
            qpos = model.jnt_qposadr[jid]
            objects[name] = {'bid': bid, 'qpos_adr': qpos}
    return objects

def execute_waypoint_dual(arm_L, arm_R, lock_base, viewer_obj, max_time=2.5):
    """Wait for both arms to settle on their queued waypoints."""
    dt = 0.01
    max_steps = int(max_time / dt)
    step_count = 0
    
    while not (arm_L.is_settled(pos_tol=0.0035) and arm_R.is_settled(pos_tol=0.0035)):
        if step_count > max_steps:
            print("  [WARNING] Waypoint settling timeout. Proceeding.")
            break
            
        lock_base()
        
        # Calculate next target joint command
        cmd_L = arm_L.step(dt=dt, grasp=arm_L.current_grasp, hold_force=arm_L.current_hold_force)
        cmd_R = arm_R.step(dt=dt, grasp=arm_R.current_grasp, hold_force=arm_R.current_hold_force)
        
        arm_L.data.ctrl[6:11] = cmd_L
        arm_L.data.ctrl[11:16] = cmd_R
        
        mujoco.mj_step(arm_L.model, arm_L.data)
        viewer_obj.sync()
        time.sleep(0.006)
        step_count += 1

def run_dual_arm_demo():
    print("=" * 70)
    print("  CARL Dual-Arm Collaborative Sorting Demo (Minimum-Jerk)")
    print("=" * 70)
    
    model = mujoco.MjModel.from_xml_path("vessel_kinetic.xml")
    data = mujoco.MjData(model)
    cache = build_cache(model)
    objs = resolve_objects(model)
    
    # --- Spawn Locations ---
    mujoco.mj_resetData(model, data)
    
    # CARL Base
    data.qpos[cache['Q_CARL'] : cache['Q_CARL']+3] = [0.0, 0.0, 0.04]
    
    # Left objects
    data.qpos[objs['obj_cube_0']['qpos_adr'] : objs['obj_cube_0']['qpos_adr']+3] = [0.18, 0.15, 0.035]
    data.qpos[objs['obj_cube_1']['qpos_adr'] : objs['obj_cube_1']['qpos_adr']+3] = [0.12, 0.16, 0.035]
    
    # Right objects (Move balls and cube_2 to the right)
    data.qpos[objs['obj_cube_2']['qpos_adr'] : objs['obj_cube_2']['qpos_adr']+3] = [0.18, -0.15, 0.035]
    data.qpos[objs['obj_ball_0']['qpos_adr'] : objs['obj_ball_0']['qpos_adr']+3] = [0.12, -0.16, 0.03]
    
    mujoco.mj_forward(model, data)
    
    # Initialize arms
    arm_L = CartesianArmExecutor(model, data, cache, side="L", pos_gain=5.0)
    arm_R = CartesianArmExecutor(model, data, cache, side="R", pos_gain=5.0)
    
    arm_L.current_grasp = False
    arm_L.current_hold_force = 0.0
    arm_R.current_grasp = False
    arm_R.current_hold_force = 0.0
    
    # Reset arms to starting pose
    data.qpos[cache['arm_qpos'][0]] = 0.0
    data.qpos[cache['arm_qpos'][1]] = 0.3
    data.qpos[cache['arm_qpos'][2]] = -1.2
    data.qpos[cache['arm_qpos'][3]] = 0.0
    
    data.qpos[cache['arm_qpos'][5]] = 0.0
    data.qpos[cache['arm_qpos'][6]] = 0.3
    data.qpos[cache['arm_qpos'][7]] = -1.2
    data.qpos[cache['arm_qpos'][8]] = 0.0
    mujoco.mj_forward(model, data)
    
    print("[VIEWER] Launching MuJoCo Real-time Play Viewer...")
    viewer_obj = mj_viewer.launch_passive(model, data)
    time.sleep(1.0)
    
    # Base Lock helper
    def lock_base():
        data.qpos[cache['Q_CARL'] : cache['Q_CARL']+3] = [0.0, 0.0, 0.04]
        data.qpos[cache['Q_CARL']+3 : cache['Q_CARL']+7] = [1.0, 0.0, 0.0, 0.0]
        data.qvel[cache['V_CARL'] : cache['V_CARL']+6] = 0.0
        
    stack_center_L = np.array([0.16, 0.08])
    stack_center_R = np.array([0.16, -0.08])
    
    # Both hover over their first targets
    hover_L = data.xpos[objs['obj_cube_0']['bid']].copy() + [0, 0, 0.08]
    hover_R = data.xpos[objs['obj_cube_2']['bid']].copy() + [0, 0, 0.08]
    
    print("\n[DUAL] Synchronized approach...")
    arm_L.queue_waypoint(hover_L, 1.2)
    arm_R.queue_waypoint(hover_R, 1.2)
    execute_waypoint_dual(arm_L, arm_R, lock_base, viewer_obj)
    
    print("\n[DUAL] Synchronized descent and grasp...")
    arm_L.queue_waypoint(hover_L - [0, 0, 0.075], 0.8)
    arm_R.queue_waypoint(hover_R - [0, 0, 0.075], 0.8)
    execute_waypoint_dual(arm_L, arm_R, lock_base, viewer_obj, max_time=1.5)
    
    arm_L.current_grasp = True; arm_L.current_hold_force = 0.6
    arm_R.current_grasp = True; arm_R.current_hold_force = 0.6
    for _ in range(40):
        lock_base()
        cmd_L = arm_L.step(dt=0.01, grasp=True, hold_force=0.6)
        cmd_R = arm_R.step(dt=0.01, grasp=True, hold_force=0.6)
        data.ctrl[6:11] = cmd_L
        data.ctrl[11:16] = cmd_R
        mujoco.mj_step(model, data); viewer_obj.sync(); time.sleep(0.006)
        
    print("\n[DUAL] Synchronized lift...")
    arm_L.queue_waypoint(hover_L, 1.0)
    arm_R.queue_waypoint(hover_R, 1.0)
    execute_waypoint_dual(arm_L, arm_R, lock_base, viewer_obj)
    
    print("\n[DUAL] Synchronized transfer to stacks...")
    stack_L_hover = np.array([stack_center_L[0], stack_center_L[1], 0.12])
    stack_R_hover = np.array([stack_center_R[0], stack_center_R[1], 0.12])
    arm_L.queue_waypoint(stack_L_hover, 1.5)
    arm_R.queue_waypoint(stack_R_hover, 1.5)
    execute_waypoint_dual(arm_L, arm_R, lock_base, viewer_obj)
    
    print("\n[DUAL] Synchronized drop...")
    arm_L.queue_waypoint(stack_L_hover - [0, 0, 0.08], 1.0)
    arm_R.queue_waypoint(stack_R_hover - [0, 0, 0.08], 1.0)
    execute_waypoint_dual(arm_L, arm_R, lock_base, viewer_obj, max_time=1.5)
    
    arm_L.current_grasp = False; arm_L.current_hold_force = 0.0
    arm_R.current_grasp = False; arm_R.current_hold_force = 0.0
    for _ in range(40):
        lock_base()
        cmd_L = arm_L.step(dt=0.01, grasp=False, hold_force=0.0)
        cmd_R = arm_R.step(dt=0.01, grasp=False, hold_force=0.0)
        data.ctrl[6:11] = cmd_L
        data.ctrl[11:16] = cmd_R
        mujoco.mj_step(model, data); viewer_obj.sync(); time.sleep(0.006)
        
    print("\n[DUAL] Retracting...")
    arm_L.queue_waypoint(stack_L_hover, 1.0)
    arm_R.queue_waypoint(stack_R_hover, 1.0)
    execute_waypoint_dual(arm_L, arm_R, lock_base, viewer_obj)

    print("\n[DUAL] Operations complete! Both arms worked cooperatively.")
    time.sleep(1.0)
    
    print("[PLAY] Idle visualization. Closing in 5 seconds...")
    for _ in range(500):
        mujoco.mj_step(model, data)
        viewer_obj.sync()
        time.sleep(0.01)
        
    viewer_obj.close()
    print("=" * 70)
    print("  Demo Complete.")
    print("=" * 70)

if __name__ == "__main__":
    run_dual_arm_demo()
