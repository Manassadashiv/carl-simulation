"""
carl_scout_ik_demo.py — Bilateral IK Demonstration Generator for CARL Primate Scout.

Phase 2a of CARL v5 Master Roadmap.

Generates expert demonstrations using Damped-Least-Squares Inverse Kinematics
for BOTH arms independently. These demonstrations are used to bootstrap the
LTC neural network via Behavioral Cloning (Phase 2b).

Three demonstration modes:
  1. LEFT-only:  Left arm reaches target, right arm idles
  2. RIGHT-only: Right arm reaches target, left arm idles
  3. BILATERAL:  Both arms reach different targets simultaneously

Each episode:
  - Spawn object(s) in reachable workspace
  - IK solves 5 positioning joints per arm per timestep
  - Fingers close when touch sensors detect contact
  - Record (state, action) pairs for the full 16-joint arm policy
  - Save to memory/scout_demos.npz

Usage:
  python carl_scout_ik_demo.py --episodes 500 --render
"""

import os
import argparse
import numpy as np
import mujoco
from carl_body_interface import (
    BodyInterface, MODEL_PATH, ARM_CTRL_LOW, ARM_CTRL_HIGH,
    ARMS_CTRL, LEFT_ARM_CTRL, RIGHT_ARM_CTRL,
)

# ─── State builder (matches ScoutArmPolicy's 51D input) ──────────────────────
STATE_DIM = 51
ACTION_DIM = 16


def get_scout_state(body, data, target_L, target_R, active_arms='both'):
    """
    Build the 51D observation for the arm policy.
    
    Layout:
      [0:16]  arm joint positions (8L + 8R)
      [16:32] arm joint velocities (8L + 8R)
      [32:35] relative target vector for left hand  (target_L - fingertip_L)
      [35:38] relative target vector for right hand (target_R - fingertip_R)
      [38:41] left fingertip velocity
      [41:44] right fingertip velocity
      [44:50] touch sensor readings (3L + 3R)
      [50]    motivational signal M(t)
    """
    # Joint positions and velocities
    arm_pos = body.get_all_arm_joint_pos(data)    # 16
    arm_vel = body.get_all_arm_joint_vel(data)    # 16

    # Relative target vectors
    tip_L = body.get_fingertip_mean_pos(data, 'L')
    tip_R = body.get_fingertip_mean_pos(data, 'R')

    if active_arms in ('left', 'both'):
        rel_L = target_L - tip_L
    else:
        rel_L = np.zeros(3)

    if active_arms in ('right', 'both'):
        rel_R = target_R - tip_R
    else:
        rel_R = np.zeros(3)

    # Rotate relative vectors to the robot's local body frame
    q = body.get_base_quat(data)
    w, x, y, z = q
    R = np.array([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x**2 - 2*y**2]
    ])
    rel_L = R.T @ rel_L
    rel_R = R.T @ rel_R

    # Fingertip velocities (approximate from site linear velocity via Jacobian)
    # For simplicity, use the difference from the body velocity
    tip_vel_L = np.zeros(3)  # will be non-zero from physics
    tip_vel_R = np.zeros(3)
    
    # We can estimate velocity from the Jacobian * joint velocities
    try:
        J_L = body.get_arm_jacobian(data, 'L')
        dof_vel_L = np.array([data.qvel[d] for d in body.left_arm_dof[:5]])
        tip_vel_L = J_L @ dof_vel_L
    except:
        pass
    try:
        J_R = body.get_arm_jacobian(data, 'R')
        dof_vel_R = np.array([data.qvel[d] for d in body.right_arm_dof[:5]])
        tip_vel_R = J_R @ dof_vel_R
    except:
        pass

    # Rotate fingertip velocities to local base frame
    tip_vel_L = R.T @ tip_vel_L
    tip_vel_R = R.T @ tip_vel_R

    # Touch sensors
    touch = body.get_all_touch_readings(data)  # 6

    # Motivational signal
    M = 1.0  # always "motivated" during demos

    state = np.concatenate([
        arm_pos,       # 16
        arm_vel,       # 16
        rel_L,         # 3
        rel_R,         # 3
        tip_vel_L,     # 3
        tip_vel_R,     # 3
        touch,         # 6
        [M],           # 1
    ])
    assert state.shape[0] == STATE_DIM, f"State dim {state.shape[0]} != {STATE_DIM}"
    return state


def ik_step(body, data, target_pos, side='L'):
    """
    One-step Damped-Least-Squares IK for 5 positioning joints.
    Returns target joint positions for the 5 positioning DOFs.
    """
    # Get Jacobian for the arm
    J = body.get_arm_jacobian(data, side)  # (3, 5)

    # Get current fingertip position (index finger tip as reference)
    tip_pos = body.get_fingertip_mean_pos(data, side)

    # Error vector
    dx = target_pos - tip_pos

    # Damped Least Squares inverse
    lambda_sq = 0.015
    J_dls = J.T @ np.linalg.inv(J @ J.T + lambda_sq * np.eye(3))
    dq = J_dls @ dx

    # Scale and clip to prevent large jumps
    step_scale = 0.15
    dq = np.clip(dq * step_scale, -0.08, 0.08)

    # Get current joint positions for the 5 positioning joints
    qpos_addrs = body.left_arm_qpos[:5] if side == 'L' else body.right_arm_qpos[:5]
    current_q = np.array([data.qpos[a] for a in qpos_addrs])
    cmd_q = current_q + dq

    # Clip to actuator control range
    idx_offset = 0 if side == 'L' else 8
    cmd_q = np.clip(cmd_q, ARM_CTRL_LOW[idx_offset:idx_offset+5],
                            ARM_CTRL_HIGH[idx_offset:idx_offset+5])

    return cmd_q


def build_action(body, data, target_L, target_R, active_arms='both'):
    """
    Build full 16D action vector.
    Active arms use IK positioning + touch-based finger closing.
    Idle arms hold rest position with open fingers.
    """
    action = np.zeros(ACTION_DIM)

    # Compute center (rest) positions for normalization
    center = (ARM_CTRL_HIGH + ARM_CTRL_LOW) / 2.0
    scale = (ARM_CTRL_HIGH - ARM_CTRL_LOW) / 2.0 + 1e-8

    # ── Left arm ──
    if active_arms in ('left', 'both'):
        pos_L = ik_step(body, data, target_L, 'L')
        touch_L = body.get_touch_readings(data, 'L')
        # Finger control: close if no contact, hold if contact
        finger_L = np.array([0.0, 0.0, 0.0])  # rest = open
        if np.linalg.norm(target_L - body.get_fingertip_mean_pos(data, 'L')) < 0.05:
            # Close fingers: thumb closes positive, index/middle close negative
            finger_L[0] = ARM_CTRL_HIGH[5] * 0.8 if touch_L[0] < 0.1 else ARM_CTRL_HIGH[5] * 0.5
            finger_L[1] = ARM_CTRL_LOW[6] * 0.8 if touch_L[1] < 0.1 else ARM_CTRL_LOW[6] * 0.5
            finger_L[2] = ARM_CTRL_LOW[7] * 0.8 if touch_L[2] < 0.1 else ARM_CTRL_LOW[7] * 0.5
        cmd_L = np.concatenate([pos_L, finger_L])
        # Store as raw control values
        action[:8] = cmd_L
    else:
        # Rest position
        action[:8] = center[:8]

    # ── Right arm ──
    if active_arms in ('right', 'both'):
        pos_R = ik_step(body, data, target_R, 'R')
        touch_R = body.get_touch_readings(data, 'R')
        finger_R = np.array([0.0, 0.0, 0.0])
        if np.linalg.norm(target_R - body.get_fingertip_mean_pos(data, 'R')) < 0.05:
            finger_R[0] = ARM_CTRL_HIGH[13] * 0.8 if touch_R[0] < 0.1 else ARM_CTRL_HIGH[13] * 0.5
            finger_R[1] = ARM_CTRL_LOW[14] * 0.8 if touch_R[1] < 0.1 else ARM_CTRL_LOW[14] * 0.5
            finger_R[2] = ARM_CTRL_LOW[15] * 0.8 if touch_R[2] < 0.1 else ARM_CTRL_LOW[15] * 0.5
        cmd_R = np.concatenate([pos_R, finger_R])
        action[8:16] = cmd_R
    else:
        action[8:16] = center[8:16]

    # Normalize to [-1, 1] for neural network output format
    action_norm = (action - center) / scale
    return np.clip(action_norm, -1.0, 1.0), action


def spawn_target(rng, side='L'):
    """
    Spawn a reachable target position for one arm.
    Workspace is in front of and to the side of the robot.
    """
    if side == 'L':
        # Left arm workspace: +y side
        x = rng.uniform(0.10, 0.30)
        y = rng.uniform(0.04, 0.18)
        z = rng.uniform(0.02, 0.15)
    else:
        # Right arm workspace: -y side (mirrored)
        x = rng.uniform(0.10, 0.30)
        y = rng.uniform(-0.18, -0.04)
        z = rng.uniform(0.02, 0.15)
    return np.array([x, y, z])


def run_demo_generation(num_episodes=500, max_steps=200, render=False, seed=42):
    """Generate and save bilateral IK demonstrations."""
    
    print("=" * 65)
    print("  CARL Primate Scout — IK Demonstration Generator")
    print(f"  Episodes: {num_episodes} | Max steps: {max_steps}")
    print("=" * 65)

    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    body = BodyInterface(model)
    rng = np.random.default_rng(seed=seed)

    viewer = None
    if render:
        from mujoco import viewer as mj_viewer
        viewer = mj_viewer.launch_passive(model, data)
        print("[VIEWER] Launched passive viewer")

    # Storage
    all_states = []
    all_actions = []
    all_lengths = []
    all_modes = []

    # Mode distribution: 33% left, 33% right, 34% bilateral
    modes = ['left', 'right', 'both']

    episode = 0
    while episode < num_episodes:
        mode = modes[episode % 3]

        # Reset physics
        mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)

        # Spawn targets
        target_L = spawn_target(rng, 'L')
        target_R = spawn_target(rng, 'R')

        # Place actual objects at target positions for visual reference
        if len(body.obj_body_ids) >= 2:
            # Move cube_0 to left target
            cube0_jnt = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "fj_cube_0")
            q0 = model.jnt_qposadr[cube0_jnt]
            data.qpos[q0:q0+3] = target_L
            data.qpos[q0+3:q0+7] = [1, 0, 0, 0]
            
            # Move cube_1 to right target
            cube1_jnt = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "fj_cube_1")
            q1 = model.jnt_qposadr[cube1_jnt]
            data.qpos[q1:q1+3] = target_R
            data.qpos[q1+3:q1+7] = [1, 0, 0, 0]
        
        mujoco.mj_forward(model, data)

        ep_states = []
        ep_actions = []

        for step in range(max_steps):
            # Lock base to prevent sliding
            body.lock_base(data)

            # Build state
            state = get_scout_state(body, data, target_L, target_R, mode)

            # Get IK action
            action_norm, action_raw = build_action(body, data, target_L, target_R, mode)

            # Record
            ep_states.append(state)
            ep_actions.append(action_norm)

            # Apply action to simulation
            data.ctrl[ARMS_CTRL] = action_raw

            # Step physics (multiple sub-steps for stability)
            for _ in range(10):
                mujoco.mj_step(model, data)

            if viewer and viewer.is_running():
                viewer.sync()

        all_states.append(np.array(ep_states))
        all_actions.append(np.array(ep_actions))
        all_lengths.append(max_steps)
        all_modes.append(mode)

        # Progress report
        dist_L = np.linalg.norm(target_L - body.get_fingertip_mean_pos(data, 'L'))
        dist_R = np.linalg.norm(target_R - body.get_fingertip_mean_pos(data, 'R'))

        if (episode + 1) % 25 == 0 or episode == 0:
            print(f"  Episode {episode+1:4d}/{num_episodes} | Mode: {mode:6s} | "
                  f"Dist_L: {dist_L:.4f}m | Dist_R: {dist_R:.4f}m")

        episode += 1

    if viewer:
        viewer.close()

    # Save demonstrations
    os.makedirs("memory", exist_ok=True)
    save_path = "memory/scout_demos.npz"

    # Pad variable-length episodes to uniform shape for numpy
    states_arr = np.array(all_states)   # (N, max_steps, 51)
    actions_arr = np.array(all_actions)  # (N, max_steps, 16)
    lengths_arr = np.array(all_lengths)  # (N,)

    np.savez_compressed(save_path,
                        states=states_arr,
                        actions=actions_arr,
                        lengths=lengths_arr,
                        modes=np.array(all_modes))

    total_pairs = sum(all_lengths)
    print(f"\n{'=' * 65}")
    print(f"  Saved {num_episodes} episodes ({total_pairs} state-action pairs)")
    print(f"  -> {save_path}")
    print(f"  Modes: left={all_modes.count('left')}, "
          f"right={all_modes.count('right')}, "
          f"both={all_modes.count('both')}")
    print(f"{'=' * 65}")

    return save_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CARL Primate Scout IK Demo Generator")
    parser.add_argument("--episodes", type=int, default=500, help="Number of episodes")
    parser.add_argument("--steps", type=int, default=200, help="Steps per episode")
    parser.add_argument("--render", action="store_true", help="Show MuJoCo viewer")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    run_demo_generation(
        num_episodes=args.episodes,
        max_steps=args.steps,
        render=args.render,
        seed=args.seed,
    )
