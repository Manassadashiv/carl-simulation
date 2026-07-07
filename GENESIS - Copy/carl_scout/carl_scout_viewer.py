"""
carl_scout_viewer.py — Opens the Primate Scout model in MuJoCo's interactive viewer.

The viewer stays open until YOU close it. You can:
  - Click and drag to rotate the camera
  - Scroll to zoom
  - Double-click a body to track it
  - Press Space to pause/play simulation
  - Ctrl+click a joint to apply forces
"""

import sys
import numpy as np
import mujoco
import mujoco.viewer


def main():
    print("=" * 60)
    print("  CARL Primate Scout — Interactive Viewer")
    print("  Close the viewer window when done.")
    print("=" * 60)

    model = mujoco.MjModel.from_xml_path("carl_primate_scout.xml")
    data  = mujoco.MjData(model)

    # Set initial pose so CARL looks natural (not collapsed)
    root_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "root_joint")
    Q = model.jnt_qposadr[root_jid]
    data.qpos[Q:Q+3] = [0.0, 0.0, 0.058]
    data.qpos[Q+3:Q+7] = [1, 0, 0, 0]

    # Arms in natural rest pose
    # Left: shoulder_yaw=0.3, shoulder_pitch=0.4, elbow=-1.4
    data.ctrl[12] = 0.3   # shoulder_yaw_L
    data.ctrl[13] = 0.4   # shoulder_pitch_L
    data.ctrl[14] = -1.4  # elbow_L

    # Right: shoulder_yaw=-0.3, shoulder_pitch=0.4, elbow=-1.4
    data.ctrl[20] = -0.3  # shoulder_yaw_R
    data.ctrl[21] = 0.4   # shoulder_pitch_R
    data.ctrl[22] = -1.4  # elbow_R

    mujoco.mj_forward(model, data)

    print("[INFO] Viewer launching... look for the MuJoCo window!")
    print("[TIP]  Drag to rotate | Scroll to zoom | Space to pause")

    # launch() is BLOCKING — keeps window open until user closes it
    mujoco.viewer.launch(model, data)

    print("[DONE] Viewer closed.")


if __name__ == "__main__":
    main()
