"""
carl_active_inference.py
Clean DLS-IK controller targeting the grip-centre site.
Stage 1: Reach (bring fingertip to cube)
Stage 2: Grasp (close fingers when close)
Stage 3: Lift  (raise shoulder)
"""
import mujoco
import mujoco.viewer
import time
import numpy as np
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'carl_arm_standalone.xml')

# --- IK parameters (same as carl_arm_bc.py's proven settings) ---
LAMBDA_SQ   = 0.015
STEP_SCALE  = 0.04    # Slow, deliberate motion — watchable
MAX_DELTA   = 0.03    # Max 0.03 rad per loop per joint
REACH_DIST  = 0.025   # 2.5cm threshold

# Control ranges (radians) for the 5 positioning joints
CTRL_LO = np.array([-0.3, -2.4, -2.8, -1.57, -1.3])
CTRL_HI = np.array([ 2.2,  2.4,  0.0,  1.57,  1.3])

def dls_step(model, data, tip_sid, target_pos, arm_dofs):
    """One step of Damped Least Squares IK towards target_pos."""
    jacp = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, None, tip_sid)

    # Slice to arm DOFs only
    J = jacp[:, arm_dofs]

    # DLS inverse
    J_dls = J.T @ np.linalg.inv(J @ J.T + LAMBDA_SQ * np.eye(3))

    tip_pos = data.site_xpos[tip_sid].copy()
    error   = target_pos - tip_pos

    delta = J_dls @ error * STEP_SCALE
    delta = np.clip(delta, -MAX_DELTA, MAX_DELTA)
    return delta, np.linalg.norm(error)


def run():
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data  = mujoco.MjData(model)

    # --- Resolve IDs -------------------------------------------------------
    tip_sid  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tip_site")
    cube_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "obj_cube_0")

    joint_names = ["shoulder_yaw_L", "shoulder_pitch_L", "elbow_L",
                   "wrist_roll_L", "wrist_pitch_L"]
    act_names   = ["act_shoulder_yaw_L", "act_shoulder_pitch_L", "act_elbow_L",
                   "act_wrist_roll_L",   "act_wrist_pitch_L"]
    finger_acts = ["act_thumb_L", "act_index_L", "act_middle_L"]

    jnt_ids  = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in joint_names]
    act_ids  = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in act_names]
    fing_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in finger_acts]

    # DOF address for each positioning joint
    arm_dofs = np.array([model.jnt_dofadr[j] for j in jnt_ids])

    touch_sids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "touch_thumb_L_s"),
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "touch_index_L_s"),
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "touch_middle_L_s"),
    ]

    # --- Start in a safe, slightly-bent resting pose -----------------------
    # Pitch the shoulder down slightly, bend the elbow
    for ai, v in zip(act_ids, [0.0, 0.5, -0.8, 1.5708, 0.4]):
        data.ctrl[ai] = v
    for _ in range(80):                     # settle
        mujoco.mj_step(model, data)

    print("\n" + "=" * 50)
    print("  CARL Standalone Arm - DLS-IK Demonstration")
    print("  Watch: Pre-Grasp -> Reach -> Grasp -> Lift -> Hold")
    print("=" * 50)
    stage = "PRE_GRASP"
    lift_timer = 0
    grasp_timer = 0
    hold_timer  = 0
    loop_count = 0

    with mujoco.viewer.launch_passive(model, data) as viewer:
        # --- Settle visibly into start pose ---
        print("\n[1/4] Settling into start pose...")
        for si in range(200):
            for _ in range(5):
                mujoco.mj_step(model, data)
            if si % 40 == 0:
                viewer.sync()
                time.sleep(0.05)

        print("[2/4] Reaching for cube...")
        t0 = time.time()
        while viewer.is_running() and time.time() - t0 < 60:

            # Current state
            cube_pos = data.xpos[cube_bid].copy()
            touch    = sum(data.sensordata[s] for s in touch_sids)

            if stage == "PRE_GRASP":
                # Stage 1: move to a position DIRECTLY IN FRONT of the cube at
                # cube height. Approach is purely horizontal (no descending into table)
                pre_target = cube_pos + np.array([-0.07, 0.0, 0.005])
                delta, dist = dls_step(model, data, tip_sid, pre_target, arm_dofs)
                for i, ai in enumerate(act_ids):
                    data.ctrl[ai] = np.clip(data.ctrl[ai] + delta[i], CTRL_LO[i], CTRL_HI[i])
                data.ctrl[act_ids[3]] = 1.5708   # wrist roll 90 deg
                data.ctrl[act_ids[4]] = 0.0      # wrist pitch neutral (horizontal)
                data.ctrl[fing_ids[0]] =  0.0
                data.ctrl[fing_ids[1]] =  0.0
                data.ctrl[fing_ids[2]] =  0.0
                if loop_count % 15 == 0:
                    print(f"  [1] Pre-grasp positioning... dist to pre-target={dist:.3f}m")
                if dist < 0.030:
                    print("\n  Pre-grasp position reached!")
                    print("[2/4] Advancing into cube...")
                    stage = "REACH"

            elif stage == "REACH":
                # Stage 2: advance horizontally into the cube
                delta, dist = dls_step(model, data, tip_sid, cube_pos, arm_dofs)
                for i, ai in enumerate(act_ids):
                    data.ctrl[ai] = np.clip(data.ctrl[ai] + delta[i], CTRL_LO[i], CTRL_HI[i])
                data.ctrl[act_ids[3]] = 1.5708
                data.ctrl[act_ids[4]] = 0.0
                data.ctrl[fing_ids[0]] =  0.0
                data.ctrl[fing_ids[1]] =  0.0
                data.ctrl[fing_ids[2]] =  0.0
                if loop_count % 15 == 0:
                    print(f"  [2] Reaching... dist={dist:.3f}m")
                if touch > 0.005 or dist < REACH_DIST:
                    print(f"\n[3/4] CONTACT! dist={dist:.3f}m  touch={touch:.4f}")
                    print("  *** Spinal reflex fired - closing fingers ***")
                    stage = "GRASP"

            elif stage == "GRASP":
                data.ctrl[fing_ids[0]] =  1.2
                data.ctrl[fing_ids[1]] = -1.2
                data.ctrl[fing_ids[2]] = -1.2
                grasp_timer += 1
                if grasp_timer == 1:
                    print("  Fingers closing...")
                if grasp_timer > 80:
                    print("\n[4/4] LIFTING!")
                    stage = "LIFT"

            elif stage == "LIFT":
                data.ctrl[fing_ids[0]] =  1.2
                data.ctrl[fing_ids[1]] = -1.2
                data.ctrl[fing_ids[2]] = -1.2
                # Slowly raise shoulder
                data.ctrl[act_ids[1]] = max(data.ctrl[act_ids[1]] - 0.012, CTRL_LO[1])
                lift_timer += 1
                if lift_timer % 30 == 0:
                    cube_z = data.xpos[cube_bid][2]
                    print(f"  Lifting... cube z={cube_z:.3f}m")
                if lift_timer > 150:
                    print("\n  Cube lifted!  Holding for 2 seconds...")
                    stage = "HOLD"

            elif stage == "HOLD":
                data.ctrl[fing_ids[0]] =  1.2
                data.ctrl[fing_ids[1]] = -1.2
                data.ctrl[fing_ids[2]] = -1.2
                hold_timer += 1
                if hold_timer > 130:
                    print("\n  Demo complete. Close the viewer window to exit.")
                    stage = "DONE"

            elif stage == "DONE":
                pass   # just keep window open

            # Physics sub-steps
            for _ in range(10):
                mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.015)
            loop_count += 1

if __name__ == "__main__":
    run()
