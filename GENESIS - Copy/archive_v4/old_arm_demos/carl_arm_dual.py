"""
carl_arm_dual.py — Cooperative Dual-Arm Manipulation & Emotional Expression

Features:
  1. Simultaneous 5-DOF Spatial IK Solver for both arms.
  2. Cooperative box-lifting sequence (Reach -> Squeeze/Grasp -> Lift).
  3. Predefined emotional expressions using Minimum-Jerk Trajectory interpolation.
"""

import os
import time
import argparse
import numpy as np
import mujoco
from carl_arm_train import build_cache, reset_episode, ARM_CTRL_LOW, ARM_CTRL_HIGH, ARM_CTRL_SLICE
from carl_brainstem import MinimumJerkSmoother

def get_dual_ik_action(model, data, cache, left_target, right_target):
    """
    Solve simultaneous spatial IK for left and right arms.
    Returns a 10-dimensional actuator target vector in joint radians.
    """
    # --- Left Arm (DOFs 0-5) ---
    jacp_L = np.zeros((3, model.nv))
    jacr_L = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp_L, jacr_L, cache['tip_L_sid'])
    J_L = jacp_L[:, cache['arm_dof'][:5]]
    
    tip_pos_L = data.site_xpos[cache['tip_L_sid']]
    dx_L = left_target - tip_pos_L
    
    lambda_sq = 0.015
    step_scale = 0.45
    J_dls_L = J_L.T @ np.linalg.inv(J_L @ J_L.T + lambda_sq * np.eye(3))
    dq_L = np.clip(J_dls_L @ dx_L * step_scale, -0.2, 0.2)
    
    current_q_L = np.array([data.qpos[a] for a in cache['arm_qpos'][:5]])
    cmd_q_L = np.clip(current_q_L + dq_L, ARM_CTRL_LOW[:5], ARM_CTRL_HIGH[:5])
    
    # --- Right Arm (DOFs 5-10) ---
    jacp_R = np.zeros((3, model.nv))
    jacr_R = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp_R, jacr_R, cache['tip_R_sid'])
    J_R = jacp_R[:, cache['arm_dof'][5:10]]
    
    tip_pos_R = data.site_xpos[cache['tip_R_sid']]
    dx_R = right_target - tip_pos_R
    
    J_dls_R = J_R.T @ np.linalg.inv(J_R @ J_R.T + lambda_sq * np.eye(3))
    dq_R = np.clip(J_dls_R @ dx_R * step_scale, -0.2, 0.2)
    
    current_q_R = np.array([data.qpos[a] for a in cache['arm_qpos'][5:10]])
    cmd_q_R = np.clip(current_q_R + dq_R, ARM_CTRL_LOW[5:10], ARM_CTRL_HIGH[5:10])
    
    # Pack both commands
    full_cmd = np.zeros(10)
    full_cmd[:5] = cmd_q_L
    full_cmd[5:10] = cmd_q_R
    return full_cmd

def run_cooperative_lift(render=False):
    """Executes a full cooperative box-lifting sequence."""
    print("=" * 60)
    print("  Executing Cooperative Dual-Arm Lift Sequence")
    print("=" * 60)
    
    model = mujoco.MjModel.from_xml_path("vessel_kinetic.xml")
    data = mujoco.MjData(model)
    cache = build_cache(model)
    rng = np.random.default_rng(seed=42)
    
    # Reset simulation
    reset_episode(model, data, cache, "reach", rng)
    # Spawn block at a standard cooperative reach position
    ox, oy, oz = 0.18, 0.0, 0.035
    data.qpos[cache['obj_qpos'] : cache['obj_qpos']+3] = [ox, oy, oz]
    data.qpos[cache['obj_qpos']+3 : cache['obj_qpos']+7] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)
    
    obj_pos_init = data.xpos[cache['obj_bid']].copy()
    
    viewer_obj = None
    if render:
        from mujoco import viewer as mj_viewer
        viewer_obj = mj_viewer.launch_passive(model, data)
        print("[VIEWER] Passive viewer launched.")
        
    # Lock base position
    def lock_base():
        data.qpos[cache['Q_CARL'] : cache['Q_CARL']+3] = [0.0, 0.0, 0.04]
        data.qpos[cache['Q_CARL']+3 : cache['Q_CARL']+7] = [1.0, 0.0, 0.0, 0.0]
        data.qvel[cache['V_CARL'] : cache['V_CARL']+6] = 0.0
        
    # --- PHASE 1: REACH ---
    # Reaching for the sides of the block (+Y for Left, -Y for Right)
    print("[PHASE 1] Reaching for block faces...")
    for step in range(120):
        lock_base()
        
        # Targets are positioned on the sides of the block (cube is 0.07m wide)
        left_target = obj_pos_init + np.array([0.0, 0.035 + 0.008, 0.0])
        right_target = obj_pos_init + np.array([0.0, -0.035 - 0.008, 0.0])
        
        act = get_dual_ik_action(model, data, cache, left_target, right_target)
        # Keep grippers open during reach
        act[4] = 0.0
        act[9] = 0.0
        
        data.ctrl[:2] = 0.0
        data.ctrl[ARM_CTRL_SLICE] = act
        mujoco.mj_step(model, data)
        
        if viewer_obj:
            viewer_obj.sync()
            time.sleep(0.005)
            
    # --- PHASE 2: SQUEEZE & GRASP ---
    print("[PHASE 2] Squeezing & Grasping...")
    for step in range(50):
        lock_base()
        
        # Targets slightly inside the block to apply contact force
        left_target = obj_pos_init + np.array([0.0, 0.035 - 0.003, 0.0])
        right_target = obj_pos_init + np.array([0.0, -0.035 + 0.003, 0.0])
        
        act = get_dual_ik_action(model, data, cache, left_target, right_target)
        # Close grippers fully
        act[4] = 0.52
        act[9] = 0.52
        
        data.ctrl[:2] = 0.0
        data.ctrl[ARM_CTRL_SLICE] = act
        mujoco.mj_step(model, data)
        
        if viewer_obj:
            viewer_obj.sync()
            time.sleep(0.005)
            
    # --- PHASE 3: LIFT ---
    print("[PHASE 3] Lifting block cooperatively...")
    success = False
    for step in range(100):
        lock_base()
        
        # Lift target height gradually to 0.18m
        lift_z = 0.035 + 0.12 * min(1.0, step / 60.0)
        left_target = obj_pos_init + np.array([0.0, 0.035 - 0.003, lift_z - obj_pos_init[2]])
        right_target = obj_pos_init + np.array([0.0, -0.035 + 0.003, lift_z - obj_pos_init[2]])
        
        act = get_dual_ik_action(model, data, cache, left_target, right_target)
        act[4] = 0.52
        act[9] = 0.52
        
        data.ctrl[:2] = 0.0
        data.ctrl[ARM_CTRL_SLICE] = act
        mujoco.mj_step(model, data)
        
        current_obj_pos = data.xpos[cache['obj_bid']]
        if current_obj_pos[2] > 0.09:
            success = True
            
        if viewer_obj:
            viewer_obj.sync()
            time.sleep(0.005)
            
    if success:
        print(f"[SUCCESS] Cooperative lift complete! Block reached height={data.xpos[cache['obj_bid']][2]:.4f}m")
    else:
        print(f"[FAILURE] Box lift failed. Final height={data.xpos[cache['obj_bid']][2]:.4f}m")
        
    if viewer_obj:
        viewer_obj.close()
        
    return success

def run_emotions(render=False):
    """Executes a sequence of emotional arm expressions smoothed via Minimum-Jerk."""
    print("=" * 60)
    print("  Executing Emotional Expression Sequence")
    print("=" * 60)
    
    model = mujoco.MjModel.from_xml_path("vessel_kinetic.xml")
    data = mujoco.MjData(model)
    cache = build_cache(model)
    
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    
    viewer_obj = None
    if render:
        from mujoco import viewer as mj_viewer
        viewer_obj = mj_viewer.launch_passive(model, data)
        print("[VIEWER] Passive viewer launched.")
        
    # Lock base position
    data.qpos[cache['Q_CARL'] : cache['Q_CARL']+3] = [0.0, 0.0, 0.04]
    data.qpos[cache['Q_CARL']+3 : cache['Q_CARL']+7] = [1.0, 0.0, 0.0, 0.0]
    data.qvel[cache['V_CARL'] : cache['V_CARL']+6] = 0.0
    
    # Define emotional posture targets
    # Format: [yaw_L, pitch_L, elbow_L, wrist_L, grip_L, yaw_R, pitch_R, elbow_R, wrist_R, grip_R]
    emotions = {
        "HOME": np.array([0.0, 0.3, -1.2, 0.0, 0.0,  0.0, 0.3, -1.2, 0.0, 0.0]),
        "JOY (Raised Cheer)": np.array([0.5, -1.5, -0.2, 0.0, 0.52,  -0.5, -1.5, -0.2, 0.0, 0.52]),
        "ANGER (Thumping pose)": np.array([0.0, 1.2, -0.1, -1.57, 0.52,  0.0, 1.2, -0.1, 1.57, 0.52]),
        "PRIDE (Hands on Hips)": np.array([-0.8, 0.8, -2.2, 0.0, 0.0,  0.8, 0.8, -2.2, 0.0, 0.0]),
        "SADNESS (Defeated Hang)": np.array([0.0, -0.2, -0.1, 0.0, 0.0,  0.0, -0.2, -0.1, 0.0, 0.0]),
        "HUG (Open Embrace)": np.array([1.2, 0.2, -0.8, 0.0, 0.0,  -1.2, 0.2, -0.8, 0.0, 0.0])
    }
    
    smoother = MinimumJerkSmoother(num_joints=10, duration=0.8, dt=0.01)
    
    # Start at HOME
    current_pose = emotions["HOME"].copy()
    smoother.reset(current_pose)
    
    # Execute emotions
    for emo_name, target_pose in emotions.items():
        print(f"[EXPRESSION] Transitioning to: {emo_name}...")
        smoother.update_target(target_pose)
        
        # Run interpolation steps
        for step in range(120):
            # Lock base
            data.qpos[cache['Q_CARL'] : cache['Q_CARL']+3] = [0.0, 0.0, 0.04]
            data.qpos[cache['Q_CARL']+3 : cache['Q_CARL']+7] = [1.0, 0.0, 0.0, 0.0]
            data.qvel[cache['V_CARL'] : cache['V_CARL']+6] = 0.0
            
            act = smoother.step()
            data.ctrl[:2] = 0.0
            data.ctrl[ARM_CTRL_SLICE] = act
            mujoco.mj_step(model, data)
            
            if viewer_obj:
                viewer_obj.sync()
                time.sleep(0.008)
                
        time.sleep(0.5) # Hold pose for 0.5s
        
    print("[SUCCESS] All expressions executed.")
    if viewer_obj:
        viewer_obj.close()

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="CARL Dual Arm Control Operations")
    ap.add_argument("--lift", action="store_true", help="Run cooperative lift task")
    ap.add_argument("--emotions", action="store_true", help="Run emotional expression task")
    ap.add_argument("--render", action="store_true", help="Render in passive viewer")
    args = ap.parse_args()
    
    if args.lift:
        run_cooperative_lift(render=args.render)
    elif args.emotions:
        run_emotions(render=args.render)
    else:
        ap.print_help()
