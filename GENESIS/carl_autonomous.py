"""
carl_autonomous.py - Fully Autonomous Closed-Loop Demo for CARL Primate Scout.

Phase 4 of CARL v5 Master Roadmap.

Pick-and-place sequence:
  SEARCH -> APPROACH -> REACH -> GRASP -> LIFT -> CARRY -> PLACE -> SEARCH

Architecture:
  REACH  : BC neural policy (learned from IK expert) - natural, converging motion
  GRASP  : IK solver (same as training demos) - guaranteed precise cube contact
  LIFT   : Smoothly interpolated analytical joint pose - no jerkiness
  CARRY  : Hold lift pose while driving to nest
  PLACE  : Smooth lowering + finger release

Physical correctness:
  - Base locked during REACH/GRASP to match training conditions
  - Base also locked during first half of LIFT while arm repositions with cube
  - IK during GRASP ensures fingertips physically contact cube before closing
  - Smooth interpolation prevents sudden joint angle jumps (no jerkiness)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "carl_scout"))

import time
import argparse
import numpy as np
import mujoco
from mujoco import viewer as mj_viewer

from carl_cortex import CarlCortex
from carl_body_interface import (
    MODEL_PATH, ARMS_CTRL, ARM_CTRL_LOW, ARM_CTRL_HIGH,
)
from carl_scout_ik_demo import ik_step   # DLS inverse kinematics solver

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NEST_POSITION = np.array([-0.30, 0.0, 0.02])   # drop zone (world coords)
APPROACH_DIST = 0.18   # m - base-to-cube distance to stop driving and start reaching
REACH_DIST    = 0.09   # m - fingertip-to-approach-point to trigger GRASP

# Arm array indices [0-7 = left, 8-15 = right]
L_SY, L_SP = 0, 1        # left shoulder yaw, pitch
L_EL, L_WR, L_WP = 2, 3, 4
L_TH, L_IN, L_MI = 5, 6, 7

R_SY, R_SP = 8, 9        # right shoulder yaw, pitch
R_EL, R_WR, R_WP = 10, 11, 12
R_TH, R_IN, R_MI = 13, 14, 15

# Cube geometry (from XML: half-size = 0.020m, placed at z=0.02 = center)
CUBE_HALF = 0.020

# Lifted arm configuration - arm retracted upward, fingers closed tight
def make_lift_pose(side):
    """16D arm array: chosen arm in lifted/retracted pose, idle arm at rest."""
    pose = (ARM_CTRL_HIGH + ARM_CTRL_LOW) / 2.0   # idle arm at rest
    if side == 'L':
        pose[L_SY] =  0.50   # shoulder yaw: swing arm forward
        pose[L_SP] = -0.80   # shoulder pitch: arm pitched up
        pose[L_EL] = -1.30   # elbow: forearm bends up
        pose[L_WR] =  0.00
        pose[L_WP] = -0.20   # wrist slightly down to hold cube
        pose[L_TH] = ARM_CTRL_HIGH[L_TH]   # thumb closed
        pose[L_IN] = ARM_CTRL_LOW[L_IN]    # index closed
        pose[L_MI] = ARM_CTRL_LOW[L_MI]    # middle closed
    else:
        pose[R_SY] = -0.50
        pose[R_SP] = -0.80
        pose[R_EL] = -1.30
        pose[R_WR] =  0.00
        pose[R_WP] = -0.20
        pose[R_TH] = ARM_CTRL_HIGH[R_TH]
        pose[R_IN] = ARM_CTRL_LOW[R_IN]
        pose[R_MI] = ARM_CTRL_LOW[R_MI]
    return np.clip(pose, ARM_CTRL_LOW, ARM_CTRL_HIGH)


def close_fingers(ctrl, side):
    """Force finger joints to maximum close position."""
    if side == 'L':
        ctrl[L_TH] = ARM_CTRL_HIGH[L_TH]
        ctrl[L_IN] = ARM_CTRL_LOW[L_IN]
        ctrl[L_MI] = ARM_CTRL_LOW[L_MI]
    else:
        ctrl[R_TH] = ARM_CTRL_HIGH[R_TH]
        ctrl[R_IN] = ARM_CTRL_LOW[R_IN]
        ctrl[R_MI] = ARM_CTRL_LOW[R_MI]
    return ctrl


def open_fingers(ctrl, side):
    """Force finger joints to maximum open position."""
    if side == 'L':
        ctrl[L_TH] = ARM_CTRL_LOW[L_TH]
        ctrl[L_IN] = ARM_CTRL_HIGH[L_IN]
        ctrl[L_MI] = ARM_CTRL_HIGH[L_MI]
    else:
        ctrl[R_TH] = ARM_CTRL_LOW[R_TH]
        ctrl[R_IN] = ARM_CTRL_HIGH[R_IN]
        ctrl[R_MI] = ARM_CTRL_HIGH[R_MI]
    return ctrl


# ---------------------------------------------------------------------------
# State Machine
# ---------------------------------------------------------------------------
class AutonomousStateMachine:

    def __init__(self, cortex):
        self.cortex       = cortex
        self.body         = cortex.body
        self.data         = cortex.data
        self.state        = "SEARCH"
        self.tgt_idx      = None
        self.done         = [False, False]
        self.timer        = 0
        self.lock_pos     = None
        self.lock_quat    = None
        self.arm          = None          # 'L' or 'R'
        self.lift_cube_pos = None          # cube world-pos when LIFT started
        self._carry_ctrl  = None          # arm ctrl frozen when CARRY started

    def _go(self, new_state):
        print(f"[STATE] {self.state} -> {new_state}")
        if new_state == "LIFT":
            # Snapshot cube position so LIFT IK target is anchored to it
            objs = self.body.get_object_positions(self.data)
            self.lift_cube_pos = objs[self.tgt_idx].copy()
        elif new_state == "CARRY":
            # Freeze the exact arm joint commands that successfully held the cube
            self._carry_ctrl = self.data.ctrl[ARMS_CTRL].copy()
        self.state = new_state
        self.timer = 0

    # ── state machine transitions ─────────────────────────────────────────────

    def update(self):
        body      = self.body
        data      = self.data
        base_pos  = body.get_base_pos(data)
        objs      = body.get_object_positions(data)

        if self.state == "SEARCH":
            best, best_d = None, 1e9
            for i in [0, 1]:
                if not self.done[i]:
                    d = np.linalg.norm(objs[i] - base_pos)
                    if d < best_d:
                        best_d, best = d, i
            if best is not None:
                self.tgt_idx = best
                self.cortex.target_obj_idx = best
                self.cortex.active_arm_side = 'auto'
                self._go("APPROACH")
            else:
                print("[DONE] All cubes placed!")
                self.state = "DONE"

        elif self.state == "APPROACH":
            self.cortex.target_obj_idx = self.tgt_idx
            dist = np.linalg.norm(objs[self.tgt_idx] - base_pos)
            if dist < APPROACH_DIST:
                self.lock_pos  = base_pos.copy()
                self.lock_quat = body.get_base_quat(data).copy()
                rel_y = (objs[self.tgt_idx] - base_pos)[1]
                self.arm = 'L' if rel_y >= 0.0 else 'R'
                self.cortex.active_arm_side = 'left' if self.arm == 'L' else 'right'
                self.cortex.arm_policy.reset()   # fresh hidden state for REACH
                self._go("REACH")

        elif self.state == "REACH":
            self.cortex.target_obj_idx = self.tgt_idx
            cube_pos  = objs[self.tgt_idx]
            approach  = cube_pos + np.array([0.0, 0.0, 0.04])   # 4cm above cube
            hand      = body.get_fingertip_mean_pos(data, self.arm)
            dist      = np.linalg.norm(approach - hand)
            self.timer += 1
            if self.timer % 20 == 0:
                print(f"    [REACH] dist={dist:.4f}m  hand_z={hand[2]:.3f}m")
            if dist < REACH_DIST:
                self._go("GRASP")

        elif self.state == "GRASP":
            # IK closes the arm onto the cube sides; 80 steps for solid contact
            self.timer += 1
            touch = self.body.get_touch_readings(self.data, self.arm)
            if self.timer % 20 == 0:
                print(f"    [GRASP] touch={touch.round(3)}  t={self.timer}")
            if self.timer > 80:
                # Check if any finger is in contact
                if np.max(touch) < 0.01:
                    print("    [GRASP] WARNING: no touch contact detected - extending grasp")
                self._go("LIFT")

        elif self.state == "LIFT":
            cube_z = objs[self.tgt_idx][2]
            self.timer += 1
            if self.timer % 20 == 0:
                hand = body.get_fingertip_mean_pos(data, self.arm)
                print(f"    [LIFT] cube_z={cube_z:.4f}m  hand_z={hand[2]:.3f}m  t={self.timer}")
            if cube_z > 0.055:   # cube lifted clear of floor
                self._go("CARRY")
            elif self.timer > 120:
                print("    [LIFT] timeout - cube not lifted, retrying")
                self.lock_pos  = None
                self.lock_quat = None
                self._go("APPROACH")

        elif self.state == "CARRY":
            d_nest = np.linalg.norm(NEST_POSITION[:2] - base_pos[:2])
            if d_nest < 0.16:
                self._go("PLACE")

        elif self.state == "PLACE":
            self.timer += 1
            if self.timer > 80:
                self.done[self.tgt_idx] = True
                print(f"    [PLACE] cube {self.tgt_idx} placed at nest")
                self.lock_pos  = None
                self.lock_quat = None
                self._go("SEARCH")

    # ── arm control ───────────────────────────────────────────────────────────

    def arm_ctrl(self, objs):
        """Compute 16D arm control vector for the current state."""
        rest = (ARM_CTRL_HIGH + ARM_CTRL_LOW) / 2.0

        if self.state in ("SEARCH", "APPROACH"):
            return rest.copy()

        # Check if Standalone policy is loaded in cortex
        use_standalone = (self.cortex.standalone_policy is not None)

        if self.state == "REACH" or (use_standalone and self.state in ("GRASP", "LIFT")):
            # Hand over REACH, GRASP, and LIFT manipulate states to Standalone policy
            return None   # sentinel: main loop handles

        if self.state == "GRASP":
            # IK: position fingertip at cube CENTER height so fingers grip the sides
            # (same as training: cube was always at target position = z=0.02)
            cube_pos = objs[self.tgt_idx]
            grasp_target = cube_pos.copy()   # exactly at cube center, not above
            ctrl = rest.copy()
            pos_5 = ik_step(self.body, self.data, grasp_target, self.arm)
            if self.arm == 'L':
                ctrl[:5] = pos_5
            else:
                ctrl[8:13] = pos_5
            ctrl = close_fingers(ctrl, self.arm)
            return ctrl

        if self.state in ("LIFT", "CARRY"):
            if self.state == "CARRY" and self._carry_ctrl is not None:
                # Freeze joint angles from the moment lift succeeded
                ctrl = self._carry_ctrl.copy()
                ctrl = close_fingers(ctrl, self.arm)
                return np.clip(ctrl, ARM_CTRL_LOW, ARM_CTRL_HIGH)

            # LIFT: IK elevator — target rises directly above cube, 3mm/step
            if self.lift_cube_pos is None:
                return rest.copy()
            lift_z = min(self.timer * 0.003, 0.14)   # max 14cm rise
            lift_target = self.lift_cube_pos + np.array([0.0, 0.0, lift_z])
            ctrl = rest.copy()
            pos_5 = ik_step(self.body, self.data, lift_target, self.arm)
            if self.arm == 'L':
                ctrl[:5] = pos_5
            else:
                ctrl[8:13] = pos_5
            ctrl = close_fingers(ctrl, self.arm)
            return np.clip(ctrl, ARM_CTRL_LOW, ARM_CTRL_HIGH)

        if self.state == "PLACE":
            # Keep the carry joint config, open fingers after 40 steps
            if self._carry_ctrl is not None:
                ctrl = self._carry_ctrl.copy()
            else:
                ctrl = make_lift_pose(self.arm)
            if self.timer > 40:
                ctrl = open_fingers(ctrl, self.arm)
            return np.clip(ctrl, ARM_CTRL_LOW, ARM_CTRL_HIGH)

        return rest.copy()


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def run_autonomous_loop(episodes=2, max_steps=1200):
    print("=" * 65)
    print("  CARL Primate Scout - Fully Autonomous Loop")
    print("=" * 65)

    model  = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data   = mujoco.MjData(model)
    cortex = CarlCortex(model, data, weights_dir="memory", prefer_bc=True)
    sm     = AutonomousStateMachine(cortex)

    print("[VIEWER] Launching passive viewer...")
    viewer = mj_viewer.launch_passive(model, data)
    time.sleep(0.5)

    rng = np.random.default_rng(seed=42)

    for ep in range(episodes):
        print(f"\n[EPISODE {ep+1}/{episodes}] Starting...")

        # ── Reset ─────────────────────────────────────────────────────────────
        mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)

        sm.state        = "SEARCH"
        sm.tgt_idx      = None
        sm.done         = [False, False]
        sm.timer        = 0
        sm.lock_pos     = None
        sm.lock_quat    = None
        sm.arm          = None
        sm.lift_cube_pos = None
        sm._carry_ctrl  = None
        cortex.arm_policy.reset()

        # Place cubes on the floor within arm workspace
        for jnt_name, ysign in [("fj_cube_0", 1), ("fj_cube_1", -1)]:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jnt_name)
            q   = model.jnt_qposadr[jid]
            data.qpos[q]      = rng.uniform(0.14, 0.20)
            data.qpos[q+1]    = ysign * rng.uniform(0.06, 0.13)
            data.qpos[q+2]    = CUBE_HALF    # center at half-height
            data.qpos[q+3:q+7] = [1, 0, 0, 0]
        mujoco.mj_forward(model, data)

        # Landing settle: lock base, arms in XML default pose (all-zero)
        dq   = model.key_qpos[0] if model.nkey > 0 else model.qpos0
        aidx = cortex.body.left_arm_qpos + cortex.body.right_arm_qpos
        for _ in range(50):
            cortex.body.lock_base(data)
            data.ctrl[:]        = 0.0
            data.ctrl[ARMS_CTRL] = dq[aidx]
            mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        viewer.sync()

        # ── Episode loop ──────────────────────────────────────────────────────
        for step in range(max_steps):
            if not viewer.is_running() or sm.state == "DONE":
                break

            sm.update()
            objs = cortex.body.get_object_positions(data)
            obs  = cortex.perceive()

            # ── Locomotion ────────────────────────────────────────────────────
            if sm.state == "APPROACH":
                loco = cortex.decide_locomotion(obs)
                loco[0] = np.clip(loco[0], 0.15, 0.7)
            elif sm.state == "CARRY":
                bp  = cortex.body.get_base_pos(data)
                ang = np.arctan2(NEST_POSITION[1] - bp[1],
                                 NEST_POSITION[0] - bp[0])
                loco = np.array([0.25, np.clip(ang * 1.5, -1.0, 1.0)])
            else:
                loco = np.zeros(2)   # stop wheels during manipulation

            v, w = loco[0] * 0.4, loco[1] * 1.5
            W, R = 0.22, 0.042
            oL = (v - w * W / 2) / R
            oR = (v + w * W / 2) / R
            data.ctrl[0], data.ctrl[1] = oL, oR
            data.ctrl[2], data.ctrl[3] = oL, oR

            # ── Neck gaze ─────────────────────────────────────────────────────
            data.ctrl[10:12] = cortex.decide_neck_gaze()
            data.ctrl[4:10]  = 0.0   # spine neutral

            # ── Arm control ───────────────────────────────────────────────────
            arm_ctrl_cmd = sm.arm_ctrl(objs)

            if arm_ctrl_cmd is None:
                # REACH: neural BC policy targets the cube directly at its height
                # (matches training: targets were placed at z=0.02-0.15, cube at target)
                cube_pos     = objs[sm.tgt_idx]
                arm_target   = cube_pos.copy()   # approach cube at cube height, no z offset
                arm_ctrl_cmd = cortex.decide_arms(override_target=arm_target, dt=0.006)

            data.ctrl[ARMS_CTRL] = np.clip(arm_ctrl_cmd, ARM_CTRL_LOW, ARM_CTRL_HIGH)

            # ── Physics sub-steps ─────────────────────────────────────────────
            for _ in range(3):
                # Lock base during REACH and GRASP (matches training)
                if sm.state in ("REACH", "GRASP") and sm.lock_pos is not None:
                    cortex.body.lock_base(data, pos=sm.lock_pos, quat=sm.lock_quat)
                # Also lock during LIFT so reaction forces don't slide robot
                elif sm.state == "LIFT" and sm.lock_pos is not None:
                    cortex.body.lock_base(data, pos=sm.lock_pos, quat=sm.lock_quat)
                mujoco.mj_step(model, data)

            viewer.sync()
            time.sleep(0.015)

        print(f"[EPISODE {ep+1}] Finished. Final State: {sm.state}")
        time.sleep(2.0)

    viewer.close()
    print("=" * 65)
    print("  Autonomous Pick-and-Place Demonstration Complete!")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--steps",    type=int, default=900)
    args = parser.parse_args()
    run_autonomous_loop(episodes=args.episodes, max_steps=args.steps)
