"""
carl_scout_autonomous.py — Fully Autonomous Neural-Network Demo for CARL Primate Scout

NO scripted waypoints. NO hardcoded motions.
The LTC policy network drives ALL 16 arm joints in real-time.

Usage:
  python carl_scout_autonomous.py              # Run with trained weights
  python carl_scout_autonomous.py --random     # Run with random policy (untrained)
"""

import os
import sys
import time
import argparse
import numpy as np
import mujoco
from mujoco import viewer as mj_viewer

from carl_scout_train import (
    ScoutArmPolicy, build_scout_cache, get_scout_state,
    reset_scout_episode, ARM_CTRL_LOW, ARM_CTRL_HIGH, ARM_CTRL_SLICE, MODEL_PATH
)


def run_autonomous(weights_path, num_episodes=5, max_steps=400, stage='reach', use_random=False):
    """Run CARL autonomously using the trained LTC policy."""
    print("=" * 65)
    print("  CARL Primate Scout — Autonomous Neural Demo")
    print(f"  100% Neural Network Driven — Stage: {stage.upper()}")
    print("=" * 65)

    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data  = mujoco.MjData(model)
    cache = build_scout_cache(model)
    rng   = np.random.default_rng()

    policy = ScoutArmPolicy()
    if not use_random and os.path.exists(weights_path):
        saved = np.load(weights_path)
        policy.set_params(saved['weights'])
        print(f"[LOAD] Loaded trained weights from {weights_path}")
        print(f"[INFO] Policy params: {policy.param_count():,}")
    else:
        if use_random:
            print("[INFO] Running with RANDOM (untrained) policy for comparison")
        else:
            print(f"[WARN] No weights found at {weights_path} — using random policy")
        print(f"[INFO] Policy params: {policy.param_count():,}")

    print(f"\n[VIEWER] Launching viewer...")
    viewer_obj = mj_viewer.launch_passive(model, data)
    time.sleep(1.0)

    for ep in range(num_episodes):
        # Determine active arms mode for this episode
        if stage in ['reach', 'grasp', 'lift', 'coordinate']:
            active_arms = ['left', 'right', 'both'][ep % 3]
        else:
            active_arms = 'both'

        print(f"\n[EPISODE {ep+1}/{num_episodes}] Mode: {active_arms.upper()} | Resetting environment...")
        policy.reset()
        reset_scout_episode(model, data, cache, stage, active_arms, rng)

        viewer_obj.sync()
        time.sleep(0.5)

        # Run the policy
        for step in range(max_steps):
            s   = get_scout_state(model, data, cache, stage, active_arms)
            raw = policy.forward(s)
            action = ARM_CTRL_LOW + (raw + 1.0) * 0.5 * (ARM_CTRL_HIGH - ARM_CTRL_LOW)

            # Only arm actuators driven by policy
            data.ctrl[:12] = 0.0  # wheels, spine, neck off
            data.ctrl[ARM_CTRL_SLICE] = action

            # Lock base
            data.qpos[cache['Q_ROOT']:cache['Q_ROOT']+3] = [0.0, 0.0, 0.058]
            data.qpos[cache['Q_ROOT']+3:cache['Q_ROOT']+7] = [1, 0, 0, 0]
            data.qvel[cache['V_ROOT']:cache['V_ROOT']+6] = 0.0

            mujoco.mj_step(model, data)
            viewer_obj.sync()
            time.sleep(0.004)

            # Check distances to targets
            tip_L = data.site_xpos[cache['tip_L_sid']]
            obj_L = data.xpos[cache['obj_bid']]
            dist_L = np.linalg.norm(obj_L - tip_L)

            tip_R = data.site_xpos[cache['tip_R_sid']]
            obj_R = data.xpos[cache['obj1_bid']]
            dist_R = np.linalg.norm(obj_R - tip_R)

            if step % 100 == 0:
                if active_arms == 'left':
                    print(f"  Step {step:3d} | Dist L: {dist_L:.4f}m | Right Arm: IDLE")
                elif active_arms == 'right':
                    print(f"  Step {step:3d} | Left Arm: IDLE  | Dist R: {dist_R:.4f}m")
                else:
                    print(f"  Step {step:3d} | Dist L: {dist_L:.4f}m | Dist R: {dist_R:.4f}m")

        print(f"  [DONE] Episode {ep+1} complete.")
        time.sleep(1.0)

    print("\n" + "=" * 65)
    print("  Autonomous demo complete!")
    print("=" * 65)
    time.sleep(2.0)
    viewer_obj.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="CARL Primate Scout — Autonomous Demo")
    ap.add_argument("--weights", default="memory/carl_scout_arm_weights.npz")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--stage", default="reach", choices=["reach", "grasp", "lift", "coordinate"])
    ap.add_argument("--random", action="store_true", help="Use random untrained policy")
    args = ap.parse_args()

    run_autonomous(args.weights, args.episodes, stage=args.stage, use_random=args.random)
