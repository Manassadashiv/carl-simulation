"""
carl_jenga_demo.py — Industrial Palletizer & Jenga Tower Stacking Simulator for CARL (5-DOF Arms)

Uses minimum-jerk (quintic) trajectory profiles and closed-loop force grasp regulation.
"""

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

def execute_to_waypoint(arm, end_pos, duration, grasp, hold_force, lock_base, viewer_obj):
    """Guide the arm to a Cartesian waypoint using minimum-jerk interpolation."""
    arm.queue_waypoint(end_pos, duration)
    dt = 0.01  # MuJoCo options timestep
    
    max_steps = int((duration + 2.0) / dt)
    step_count = 0
    
    # We step until the minimum-jerk trajectory is done and the arm is settled close to target
    while not arm.is_settled(pos_tol=0.0035):
        if step_count > max_steps:
            print("  [WARNING] Waypoint settling timeout. Proceeding.")
            break
            
        lock_base()
        
        # Calculate next target joint command
        cmd = arm.step(dt=dt, grasp=grasp, hold_force=hold_force)
        
        # Left arm actuator commands slice (6 to 11: yaw_L, shoulder_L, elbow_L, wrist_L, grip_L)
        arm.data.ctrl[6:11] = cmd
        
        # Keep right arm in home position
        arm.data.ctrl[11:16] = [0.0, 0.3, -1.2, 0.0, 0.0]
        
        mujoco.mj_step(arm.model, arm.data)
        viewer_obj.sync()
        time.sleep(0.006)
        step_count += 1

def run_jenga_demo():
    print("=" * 70)
    print("  CARL Industrial Palletizer & Jenga Demo (Minimum-Jerk)")
    print("=" * 70)
    
    model = mujoco.MjModel.from_xml_path("vessel_kinetic.xml")
    data = mujoco.MjData(model)
    cache = build_cache(model)
    objs = resolve_objects(model)
    
    # --- Spawn Locations ---
    mujoco.mj_resetData(model, data)
    
    # CARL Base
    data.qpos[cache['Q_CARL'] : cache['Q_CARL']+3] = [0.0, 0.0, 0.04]
    
    # Cubes: Spawn in precise grid locations
    data.qpos[objs['obj_cube_0']['qpos_adr'] : objs['obj_cube_0']['qpos_adr']+3] = [0.18, 0.14, 0.035]
    data.qpos[objs['obj_cube_1']['qpos_adr'] : objs['obj_cube_1']['qpos_adr']+3] = [0.15, 0.17, 0.035]
    data.qpos[objs['obj_cube_2']['qpos_adr'] : objs['obj_cube_2']['qpos_adr']+3] = [0.12, 0.13, 0.035]
    
    # Toys: Place balls & capsule to the side
    data.qpos[objs['obj_ball_0']['qpos_adr'] : objs['obj_ball_0']['qpos_adr']+3] = [0.22, -0.15, 0.03]
    data.qpos[objs['obj_ball_1']['qpos_adr'] : objs['obj_ball_1']['qpos_adr']+3] = [0.15, -0.18, 0.03]
    data.qpos[objs['obj_toy']['qpos_adr'] : objs['obj_toy']['qpos_adr']+3] = [0.25, 0.18, 0.05]
    
    mujoco.mj_forward(model, data)
    
    # Initialize arm
    arm = CartesianArmExecutor(model, data, cache, side="L", pos_gain=5.0)
    
    # Reset left arm to starting pose
    data.qpos[cache['arm_qpos'][0]] = 0.0
    data.qpos[cache['arm_qpos'][1]] = 0.3
    data.qpos[cache['arm_qpos'][2]] = -1.2
    data.qpos[cache['arm_qpos'][3]] = 0.0
    mujoco.mj_forward(model, data)
    
    print("[VIEWER] Launching MuJoCo Real-time Play Viewer...")
    viewer_obj = mj_viewer.launch_passive(model, data)
    time.sleep(1.0)
    
    # Base Lock helper
    def lock_base():
        data.qpos[cache['Q_CARL'] : cache['Q_CARL']+3] = [0.0, 0.0, 0.04]
        data.qpos[cache['Q_CARL']+3 : cache['Q_CARL']+7] = [1.0, 0.0, 0.0, 0.0]
        data.qvel[cache['V_CARL'] : cache['V_CARL']+6] = 0.0
        
    stack_center = np.array([0.16, 0.10])
    cubes_to_stack = ['obj_cube_0', 'obj_cube_1', 'obj_cube_2']
    
    for i, cube_name in enumerate(cubes_to_stack):
        print(f"\n[PALLETIZER] Processing {cube_name} ({i+1}/3)...")
        
        # Calculate stack placement target
        target_z = 0.035 + i * 0.070
        stack_target = np.array([stack_center[0], stack_center[1], target_z])
        
        # Get target object position
        cube_pos = data.xpos[objs[cube_name]['bid']].copy()
        
        # Define waypoint targets
        hover_above_cube = cube_pos + [0, 0, 0.08]
        hover_above_stack = stack_target + [0, 0, 0.12]
        
        # --- PATH PLANNING & EXECUTION ---
        
        # 1. Move to hover position above cube (linear approach)
        print("  [MIN-JERK] Approaching pre-grasp hover position...")
        execute_to_waypoint(arm, hover_above_cube, duration=1.2, grasp=False, hold_force=0.0, lock_base=lock_base, viewer_obj=viewer_obj)
        
        # 2. Vertical descent to grasp
        print("  [MIN-JERK] Descending vertically to block...")
        execute_to_waypoint(arm, cube_pos + [0, 0, 0.005], duration=0.8, grasp=False, hold_force=0.0, lock_base=lock_base, viewer_obj=viewer_obj)
        
        # 3. Secure grasp
        print("  [GRASP] Closing gripper jaws (Force-Regulated)...")
        for _ in range(40):
            lock_base()
            cmd = arm.step(dt=0.01, grasp=True, hold_force=0.6)
            arm.data.ctrl[6:11] = cmd
            arm.data.ctrl[11:16] = [0.0, 0.3, -1.2, 0.0, 0.0]
            mujoco.mj_step(arm.model, arm.data)
            viewer_obj.sync()
            time.sleep(0.006)
            
        # 4. Vertical lift
        print("  [MIN-JERK] Lifting block vertically...")
        execute_to_waypoint(arm, hover_above_cube, duration=1.0, grasp=True, hold_force=0.6, lock_base=lock_base, viewer_obj=viewer_obj)
        
        # 5. Horizontal transfer
        print("  [MIN-JERK] Transferring horizontally to stack...")
        execute_to_waypoint(arm, hover_above_stack, duration=1.5, grasp=True, hold_force=0.6, lock_base=lock_base, viewer_obj=viewer_obj)
        
        # 6. Vertical drop
        print("  [MIN-JERK] Descending vertically to Jenga stack...")
        execute_to_waypoint(arm, stack_target + [0, 0, 0.005], duration=1.0, grasp=True, hold_force=0.6, lock_base=lock_base, viewer_obj=viewer_obj)
        
        # 7. Release
        print("  [RELEASE] Opening gripper jaws...")
        for _ in range(40):
            lock_base()
            cmd = arm.step(dt=0.01, grasp=False, hold_force=0.0)
            arm.data.ctrl[6:11] = cmd
            arm.data.ctrl[11:16] = [0.0, 0.3, -1.2, 0.0, 0.0]
            mujoco.mj_step(arm.model, arm.data)
            viewer_obj.sync()
            time.sleep(0.006)
            
        # 8. Vertical retract
        print("  [MIN-JERK] Retracting arm vertically...")
        execute_to_waypoint(arm, hover_above_stack, duration=1.0, grasp=False, hold_force=0.0, lock_base=lock_base, viewer_obj=viewer_obj)

    print("\n[PALLETIZER] Jenga tower completed with industrial precision!")
    time.sleep(1.5)
    
    print("\n[PLAY] CARL preparing sweep strike...")
    # Wind up right arm, then strike!
    for swing_step in range(120):
        lock_base()
        if swing_step < 40:
            # Wind up: move right arm to the right and back
            data.ctrl[11] = -0.5  # shoulder_yaw_R
            data.ctrl[12] = 0.3   # shoulder_R
            data.ctrl[13] = -1.2  # elbow_R
            data.ctrl[14] = 0.0   # wrist_R
        else:
            # Strike!
            data.ctrl[11] = 1.3   # Sweep yaw to the left
            data.ctrl[12] = 0.7   # Shoulder pitch forward
            data.ctrl[13] = -0.3  # straighten elbow
            data.ctrl[14] = 0.5   # flick wrist
            
        data.ctrl[6:11] = [0.0, 0.3, -1.2, 0.0, 0.0]
        mujoco.mj_step(model, data)
        viewer_obj.sync()
        time.sleep(0.006)
        
    print("[PLAY] Strike hit! Jenga tower knocked down!")
    
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
    run_jenga_demo()
