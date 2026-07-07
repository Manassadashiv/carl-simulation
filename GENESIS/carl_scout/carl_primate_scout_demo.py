"""
carl_primate_scout_demo.py — Visualization demo for the CARL Primate Scout body.

Shows off:
  1. Spine flexing & twisting (body language)
  2. Head tracking & looking around
  3. Arm reaching with wrist articulation
  4. 3-finger grasp (thumb opposes index+middle)
  5. Throw motion using spine wind-up + arm release
"""

import time
import math
import numpy as np
import mujoco
from mujoco import viewer as mj_viewer


def smooth_interp(t):
    """Smooth step interpolation (ease-in-out)."""
    return 3 * t**2 - 2 * t**3


def set_ctrl_smooth(data, idx, target, speed=0.03):
    """Smoothly move an actuator toward its target."""
    data.ctrl[idx] += np.clip(target - data.ctrl[idx], -speed, speed)


def run_demo():
    print("=" * 65)
    print("  CARL Primate Scout — Articulation Showcase")
    print("=" * 65)

    model = mujoco.MjModel.from_xml_path("carl_primate_scout.xml")
    data  = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    # --- Actuator index map ---
    # 0-3:   wheels (FL, FR, RL, RR)
    # 4-9:   spine  (s1_pitch, s1_roll, s2_pitch, s2_yaw, s3_pitch, s3_roll)
    # 10-11: neck/head (neck_tilt, head_pan)
    # 12-19: left arm  (sh_yaw, sh_pitch, elbow, wr_roll, wr_pitch, thumb, index, middle)
    # 20-27: right arm (sh_yaw, sh_pitch, elbow, wr_roll, wr_pitch, thumb, index, middle)

    IDX_SPINE1_P, IDX_SPINE1_R = 4, 5
    IDX_SPINE2_P, IDX_SPINE2_Y = 6, 7
    IDX_SPINE3_P, IDX_SPINE3_R = 8, 9
    IDX_NECK, IDX_HEAD = 10, 11

    # Left arm
    IDX_L_SHYAW, IDX_L_SHPITCH, IDX_L_ELBOW = 12, 13, 14
    IDX_L_WROLL, IDX_L_WPITCH = 15, 16
    IDX_L_THUMB, IDX_L_INDEX, IDX_L_MIDDLE = 17, 18, 19

    # Right arm
    IDX_R_SHYAW, IDX_R_SHPITCH, IDX_R_ELBOW = 20, 21, 22
    IDX_R_WROLL, IDX_R_WPITCH = 23, 24
    IDX_R_THUMB, IDX_R_INDEX, IDX_R_MIDDLE = 25, 26, 27

    # Set initial resting pose for CARL
    # Lock base in place
    root_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "root_joint")
    Q_ROOT = model.jnt_qposadr[root_jid]
    V_ROOT = model.jnt_dofadr[root_jid]
    data.qpos[Q_ROOT:Q_ROOT+3] = [0.0, 0.0, 0.058]
    data.qpos[Q_ROOT+3:Q_ROOT+7] = [1, 0, 0, 0]

    # Arms in natural rest
    # Left arm: slightly forward and down
    data.ctrl[IDX_L_SHYAW] = 0.3
    data.ctrl[IDX_L_SHPITCH] = 0.4
    data.ctrl[IDX_L_ELBOW] = -1.4
    data.ctrl[IDX_L_WROLL] = 0.0
    data.ctrl[IDX_L_WPITCH] = 0.0

    # Right arm: mirror
    data.ctrl[IDX_R_SHYAW] = -0.3
    data.ctrl[IDX_R_SHPITCH] = 0.4
    data.ctrl[IDX_R_ELBOW] = -1.4
    data.ctrl[IDX_R_WROLL] = 0.0
    data.ctrl[IDX_R_WPITCH] = 0.0

    mujoco.mj_forward(model, data)

    print("[VIEWER] Launching MuJoCo viewer...")
    print("         Watch CARL demonstrate spine, arms, wrists, and fingers!")
    viewer_obj = mj_viewer.launch_passive(model, data)
    time.sleep(1.5)

    def lock_base():
        data.qpos[Q_ROOT:Q_ROOT+3] = [0.0, 0.0, 0.058]
        data.qpos[Q_ROOT+3:Q_ROOT+7] = [1, 0, 0, 0]
        data.qvel[V_ROOT:V_ROOT+6] = 0.0

    DT = 0.005
    step_count = 0

    def step_sim(n=1):
        nonlocal step_count
        for _ in range(n):
            lock_base()
            mujoco.mj_step(model, data)
            step_count += 1
        viewer_obj.sync()
        time.sleep(0.008)

    # ================================================================
    # Phase 1: Wake Up — head looks around, spine straightens
    # ================================================================
    print("\n[PHASE 1] CARL waking up — head scanning, spine alive...")
    for t in range(400):
        phase = t / 400.0
        # Head scans left to right
        data.ctrl[IDX_HEAD] = 1.0 * math.sin(phase * 2 * math.pi)
        # Slight neck curiosity tilt
        data.ctrl[IDX_NECK] = 0.15 * math.sin(phase * 3 * math.pi)
        # Subtle spine sway (breathing/alive feel)
        data.ctrl[IDX_SPINE1_P] = 0.08 * math.sin(phase * 4 * math.pi)
        data.ctrl[IDX_SPINE2_Y] = 0.1 * math.sin(phase * 2.5 * math.pi)
        step_sim(2)

    # ================================================================
    # Phase 2: Spine Demonstration — lean, twist, flex
    # ================================================================
    print("[PHASE 2] Spine articulation — lean forward, twist, flex...")

    # Lean forward
    for t in range(200):
        s = smooth_interp(t / 200.0)
        data.ctrl[IDX_SPINE1_P] = 0.35 * s
        data.ctrl[IDX_SPINE2_P] = 0.30 * s
        data.ctrl[IDX_SPINE3_P] = 0.25 * s
        data.ctrl[IDX_HEAD] = 0.0  # look straight
        step_sim(2)

    time.sleep(0.3)

    # Twist torso (like winding up for a throw)
    for t in range(200):
        s = smooth_interp(t / 200.0)
        data.ctrl[IDX_SPINE2_Y] = 0.45 * s
        data.ctrl[IDX_SPINE3_R] = 0.25 * s
        step_sim(2)

    time.sleep(0.3)

    # Return to neutral
    for t in range(200):
        s = smooth_interp(t / 200.0)
        for idx in range(IDX_SPINE1_P, IDX_SPINE3_R + 1):
            data.ctrl[idx] *= (1.0 - s)
        step_sim(2)

    # ================================================================
    # Phase 3: Left arm reach + wrist flex + finger wiggle
    # ================================================================
    print("[PHASE 3] Left arm — reach, wrist flex, finger articulation...")

    # Reach forward
    for t in range(250):
        s = smooth_interp(min(t / 200.0, 1.0))
        data.ctrl[IDX_L_SHYAW] = 0.8 * s
        data.ctrl[IDX_L_SHPITCH] = 0.5 * s
        data.ctrl[IDX_L_ELBOW] = -0.8 * s
        step_sim(2)

    # Wrist roll demonstration
    print("  [WRIST] Roll left, roll right...")
    for t in range(300):
        phase = t / 300.0
        data.ctrl[IDX_L_WROLL] = 1.2 * math.sin(phase * 2 * math.pi)
        data.ctrl[IDX_L_WPITCH] = 0.8 * math.sin(phase * 3 * math.pi + 0.5)
        step_sim(2)

    # Reset wrist
    data.ctrl[IDX_L_WROLL] = 0.0
    data.ctrl[IDX_L_WPITCH] = 0.0

    # Finger curl demonstration
    print("  [FINGERS] Opening and closing...")
    for cycle in range(3):
        # Close (grasp)
        for t in range(120):
            s = smooth_interp(t / 120.0)
            data.ctrl[IDX_L_THUMB] = 1.2 * s    # thumb curls up
            data.ctrl[IDX_L_INDEX] = -1.2 * s   # index curls down
            data.ctrl[IDX_L_MIDDLE] = -1.2 * s  # middle curls down
            step_sim(2)
        time.sleep(0.15)

        # Open
        for t in range(120):
            s = smooth_interp(t / 120.0)
            data.ctrl[IDX_L_THUMB] = 1.2 * (1.0 - s)
            data.ctrl[IDX_L_INDEX] = -1.2 * (1.0 - s)
            data.ctrl[IDX_L_MIDDLE] = -1.2 * (1.0 - s)
            step_sim(2)
        time.sleep(0.15)

    # ================================================================
    # Phase 4: Both arms mirror — synchronized movement
    # ================================================================
    print("[PHASE 4] Dual arm synchronized movement...")

    for t in range(400):
        phase = t / 400.0
        angle = math.sin(phase * 2 * math.pi)

        # Left arm
        data.ctrl[IDX_L_SHYAW] = 0.5 + 0.5 * angle
        data.ctrl[IDX_L_SHPITCH] = 0.3 + 0.6 * angle
        data.ctrl[IDX_L_ELBOW] = -1.0 + 0.5 * angle

        # Right arm (mirrored)
        data.ctrl[IDX_R_SHYAW] = -0.5 - 0.5 * angle
        data.ctrl[IDX_R_SHPITCH] = 0.3 + 0.6 * angle
        data.ctrl[IDX_R_ELBOW] = -1.0 + 0.5 * angle

        # Subtle spine follow
        data.ctrl[IDX_SPINE2_Y] = 0.15 * angle
        data.ctrl[IDX_SPINE1_P] = 0.05 * angle

        # Head tracks opposite to body twist
        data.ctrl[IDX_HEAD] = -0.3 * angle

        step_sim(2)

    # ================================================================
    # Phase 5: Throw motion — spine wind-up + arm release
    # ================================================================
    print("[PHASE 5] Throw demonstration — full body coordination!")

    # Return to neutral first
    for t in range(150):
        s = smooth_interp(t / 150.0)
        data.ctrl[IDX_L_SHYAW] = 0.3
        data.ctrl[IDX_L_SHPITCH] = 0.4
        data.ctrl[IDX_L_ELBOW] = -1.4
        data.ctrl[IDX_R_SHYAW] = -0.3
        data.ctrl[IDX_R_SHPITCH] = 0.4
        data.ctrl[IDX_R_ELBOW] = -1.4
        for idx in range(IDX_SPINE1_P, IDX_SPINE3_R + 1):
            data.ctrl[idx] = 0.0
        data.ctrl[IDX_HEAD] = 0.0
        step_sim(2)

    # Wind-up: twist spine right, pull right arm back
    print("  [WIND-UP] Twisting spine, loading arm...")
    for t in range(200):
        s = smooth_interp(t / 200.0)
        # Spine winds up (twist right)
        data.ctrl[IDX_SPINE2_Y] = -0.45 * s
        data.ctrl[IDX_SPINE1_P] = 0.15 * s
        data.ctrl[IDX_SPINE3_R] = -0.2 * s
        # Right arm pulls back
        data.ctrl[IDX_R_SHYAW] = -0.3 - 0.8 * s
        data.ctrl[IDX_R_SHPITCH] = 1.2 * s
        data.ctrl[IDX_R_ELBOW] = -0.5
        # Right hand grips (holding imaginary ball)
        data.ctrl[IDX_R_THUMB] = 1.0 * s
        data.ctrl[IDX_R_INDEX] = -1.0 * s
        data.ctrl[IDX_R_MIDDLE] = -1.0 * s
        # Head looks at target
        data.ctrl[IDX_HEAD] = 0.4 * s
        step_sim(2)

    time.sleep(0.4)

    # THROW: snap spine forward + whip arm + release fingers
    print("  [THROW] Releasing!")
    for t in range(100):
        s = smooth_interp(min(t / 60.0, 1.0))
        # Spine snaps forward
        data.ctrl[IDX_SPINE2_Y] = -0.45 + 0.9 * s
        data.ctrl[IDX_SPINE1_P] = 0.15 - 0.5 * s
        data.ctrl[IDX_SPINE3_R] = -0.2 + 0.4 * s
        # Arm whips forward
        data.ctrl[IDX_R_SHYAW] = -1.1 + 1.8 * s
        data.ctrl[IDX_R_SHPITCH] = 1.2 - 1.8 * s
        data.ctrl[IDX_R_ELBOW] = -0.5 - 1.5 * s
        data.ctrl[IDX_R_WROLL] = 0.8 * s
        # Release fingers at peak
        if t > 40:
            release_s = smooth_interp((t - 40) / 60.0)
            data.ctrl[IDX_R_THUMB] = 1.0 * (1.0 - release_s)
            data.ctrl[IDX_R_INDEX] = -1.0 * (1.0 - release_s)
            data.ctrl[IDX_R_MIDDLE] = -1.0 * (1.0 - release_s)
        step_sim(2)

    time.sleep(0.5)

    # ================================================================
    # Phase 6: Idle alive — subtle breathing + curious head movement
    # ================================================================
    print("[PHASE 6] Idle — CARL breathing and looking around curiously...")

    # Return to rest first
    for idx in range(28):
        data.ctrl[idx] = 0.0
    data.ctrl[IDX_L_SHYAW] = 0.3
    data.ctrl[IDX_L_SHPITCH] = 0.4
    data.ctrl[IDX_L_ELBOW] = -1.4
    data.ctrl[IDX_R_SHYAW] = -0.3
    data.ctrl[IDX_R_SHPITCH] = 0.4
    data.ctrl[IDX_R_ELBOW] = -1.4

    for t in range(800):
        phase = t / 800.0
        # Breathing: subtle spine pitch oscillation
        breath = 0.04 * math.sin(phase * 6 * math.pi)
        data.ctrl[IDX_SPINE1_P] = breath
        data.ctrl[IDX_SPINE2_P] = breath * 0.7
        data.ctrl[IDX_SPINE3_P] = breath * 0.5

        # Curious head movement
        data.ctrl[IDX_HEAD] = 0.6 * math.sin(phase * 3.5 * math.pi)
        data.ctrl[IDX_NECK] = 0.2 * math.sin(phase * 2.1 * math.pi + 1.0)
        step_sim(2)

    print("\n" + "=" * 65)
    print("  Articulation showcase complete!")
    print(f"  Total simulation steps: {step_count}")
    print("  28 DOF: 4 wheels | 6 spine | 2 neck | 8+8 arms")
    print("=" * 65)

    time.sleep(3.0)
    viewer_obj.close()


if __name__ == "__main__":
    run_demo()
