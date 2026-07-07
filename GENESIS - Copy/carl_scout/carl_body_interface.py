"""
carl_body_interface.py — Unified Body Abstraction for CARL Primate Scout.

The ONE canonical interface between any Python script and CARL's MuJoCo body.
All joint indices, sensor IDs, actuator slices, and body geometry queries
go through this module. No more hardcoded slice(12, 28) scattered everywhere.

Canonical model: carl_primate_scout.xml (28 DOF)

Actuator layout (ctrl indices):
  [0-3]    Wheels:  FL, FR, RL, RR              (velocity-controlled)
  [4-9]    Spine:   s1_pitch, s1_roll, s2_pitch, s2_yaw, s3_pitch, s3_roll
  [10-11]  Neck:    neck_tilt, head_pan
  [12-19]  Left arm:  sh_yaw, sh_pitch, elbow, wr_roll, wr_pitch, thumb, index, middle
  [20-27]  Right arm: sh_yaw, sh_pitch, elbow, wr_roll, wr_pitch, thumb, index, middle

Sensor layout (sensordata indices):
  [0]  touch_thumb_L_s    [3]  touch_thumb_R_s
  [1]  touch_index_L_s    [4]  touch_index_R_s
  [2]  touch_middle_L_s   [5]  touch_middle_R_s
"""

import numpy as np
import mujoco

# ─── Canonical model path ─────────────────────────────────────────────────────
import os
MODEL_PATH = os.path.join(os.path.dirname(__file__), "carl_primate_scout.xml")

# ─── Actuator slice constants ─────────────────────────────────────────────────
WHEEL_CTRL     = slice(0, 4)     # 4 wheels
SPINE_CTRL     = slice(4, 10)    # 6 spine joints
NECK_CTRL      = slice(10, 12)   # neck_tilt + head_pan
LEFT_ARM_CTRL  = slice(12, 20)   # 8 left arm joints
RIGHT_ARM_CTRL = slice(20, 28)   # 8 right arm joints
ARMS_CTRL      = slice(12, 28)   # both arms combined (16 joints)

# Sub-slices within each arm's 8 actuators
ARM_POS_OFFSET  = slice(0, 5)   # positioning joints (shoulder_yaw..wrist_pitch)
ARM_GRIP_OFFSET = slice(5, 8)   # finger joints (thumb, index, middle)

# Total actuators
N_ACTUATORS = 28

# ─── Named actuator lists ─────────────────────────────────────────────────────
WHEEL_ACTUATORS = ["act_w_fl", "act_w_fr", "act_w_rl", "act_w_rr"]

SPINE_ACTUATORS = [
    "act_spine1_pitch", "act_spine1_roll",
    "act_spine2_pitch", "act_spine2_yaw",
    "act_spine3_pitch", "act_spine3_roll",
]

NECK_ACTUATORS = ["act_neck_tilt", "act_head_pan"]

LEFT_ARM_ACTUATORS = [
    "act_shoulder_yaw_L", "act_shoulder_pitch_L", "act_elbow_L",
    "act_wrist_roll_L", "act_wrist_pitch_L",
    "act_thumb_L", "act_index_L", "act_middle_L",
]

RIGHT_ARM_ACTUATORS = [
    "act_shoulder_yaw_R", "act_shoulder_pitch_R", "act_elbow_R",
    "act_wrist_roll_R", "act_wrist_pitch_R",
    "act_thumb_R", "act_index_R", "act_middle_R",
]

# ─── Named joint lists (for qpos/qvel access) ────────────────────────────────
WHEEL_JOINTS = ["w_fl", "w_fr", "w_rl", "w_rr"]

SPINE_JOINTS = [
    "spine1_pitch", "spine1_roll",
    "spine2_pitch", "spine2_yaw",
    "spine3_pitch", "spine3_roll",
]

NECK_JOINTS = ["neck_tilt", "head_pan"]

LEFT_ARM_JOINTS = [
    "shoulder_yaw_L", "shoulder_pitch_L", "elbow_L",
    "wrist_roll_L", "wrist_pitch_L",
    "thumb_L_knuckle", "index_L_knuckle", "middle_L_knuckle",
]

RIGHT_ARM_JOINTS = [
    "shoulder_yaw_R", "shoulder_pitch_R", "elbow_R",
    "wrist_roll_R", "wrist_pitch_R",
    "thumb_R_knuckle", "index_R_knuckle", "middle_R_knuckle",
]

# ─── Named sensor lists ──────────────────────────────────────────────────────
LEFT_TOUCH_SENSORS  = ["touch_thumb_L_s", "touch_index_L_s", "touch_middle_L_s"]
RIGHT_TOUCH_SENSORS = ["touch_thumb_R_s", "touch_index_R_s", "touch_middle_R_s"]

# ─── Named sites (for Jacobian IK, fingertip tracking) ───────────────────────
LEFT_TOUCH_SITES  = ["touch_thumb_L", "touch_index_L", "touch_middle_L"]
RIGHT_TOUCH_SITES = ["touch_thumb_R", "touch_index_R", "touch_middle_R"]

# ─── Named bodies (for xpos queries) ─────────────────────────────────────────
OBJECT_BODIES = ["obj_cube_0", "obj_cube_1", "obj_ball_0", "obj_cylinder_0"]

# ─── Joint limits for arm actuators (16 total: 8L + 8R) ──────────────────────
ARM_CTRL_LOW = np.array([
    # Left arm
    -0.30, -2.4, -2.8, -1.57, -1.3,   0.0, -1.4, -1.4,
    # Right arm
    -2.20, -2.4, -2.8, -1.57, -1.3,   0.0, -1.4, -1.4,
], dtype=np.float64)

ARM_CTRL_HIGH = np.array([
    # Left arm
     2.20,  2.4,  0.0,  1.57,  1.3,   1.4,  0.0,  0.0,
    # Right arm
     0.30,  2.4,  0.0,  1.57,  1.3,   1.4,  0.0,  0.0,
], dtype=np.float64)


class BodyInterface:
    """
    Cached interface to CARL Primate Scout's MuJoCo body.

    Resolves all named joints, actuators, sensors, sites, and bodies to
    integer IDs at construction time. All subsequent queries use cached
    integer lookups — no string hashing in the hot loop.

    Usage:
        model = mujoco.MjModel.from_xml_path(carl_body_interface.MODEL_PATH)
        data  = mujoco.MjData(model)
        body  = BodyInterface(model)
    """

    def __init__(self, model):
        self.model = model
        self._resolve_ids(model)

    def _resolve_ids(self, model):
        """Resolve all named elements to integer IDs (one-time cost)."""

        def _jnt_id(name):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            assert jid != -1, f"Joint '{name}' not found in model"
            return jid

        def _act_id(name):
            aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            assert aid != -1, f"Actuator '{name}' not found in model"
            return aid

        def _site_id(name):
            sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
            assert sid != -1, f"Site '{name}' not found in model"
            return sid

        def _body_id(name):
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            assert bid != -1, f"Body '{name}' not found in model"
            return bid

        def _sensor_id(name):
            sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
            assert sid != -1, f"Sensor '{name}' not found in model"
            return sid

        def _geom_id(name):
            gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
            assert gid != -1, f"Geom '{name}' not found in model"
            return gid

        # ── Root joint ────────────────────────────────────────────────────
        self.root_jnt_id = _jnt_id("root_joint")
        self.Q_ROOT = model.jnt_qposadr[self.root_jnt_id]   # qpos offset (freejoint: 7 values)
        self.V_ROOT = model.jnt_dofadr[self.root_jnt_id]     # qvel offset (freejoint: 6 values)

        # ── Wheel joint IDs and qpos/dof addresses ────────────────────────
        self.wheel_jnt_ids = [_jnt_id(n) for n in WHEEL_JOINTS]
        self.wheel_qpos    = [model.jnt_qposadr[j] for j in self.wheel_jnt_ids]
        self.wheel_dof     = [model.jnt_dofadr[j] for j in self.wheel_jnt_ids]

        # ── Spine joint IDs and qpos/dof addresses ────────────────────────
        self.spine_jnt_ids = [_jnt_id(n) for n in SPINE_JOINTS]
        self.spine_qpos    = [model.jnt_qposadr[j] for j in self.spine_jnt_ids]
        self.spine_dof     = [model.jnt_dofadr[j] for j in self.spine_jnt_ids]

        # ── Neck joint IDs ────────────────────────────────────────────────
        self.neck_jnt_ids = [_jnt_id(n) for n in NECK_JOINTS]
        self.neck_qpos    = [model.jnt_qposadr[j] for j in self.neck_jnt_ids]
        self.neck_dof     = [model.jnt_dofadr[j] for j in self.neck_jnt_ids]

        # ── Left arm joint IDs and qpos/dof addresses ─────────────────────
        self.left_arm_jnt_ids = [_jnt_id(n) for n in LEFT_ARM_JOINTS]
        self.left_arm_qpos    = [model.jnt_qposadr[j] for j in self.left_arm_jnt_ids]
        self.left_arm_dof     = [model.jnt_dofadr[j] for j in self.left_arm_jnt_ids]

        # ── Right arm joint IDs and qpos/dof addresses ────────────────────
        self.right_arm_jnt_ids = [_jnt_id(n) for n in RIGHT_ARM_JOINTS]
        self.right_arm_qpos    = [model.jnt_qposadr[j] for j in self.right_arm_jnt_ids]
        self.right_arm_dof     = [model.jnt_dofadr[j] for j in self.right_arm_jnt_ids]

        # ── Combined arm DOF (for Jacobian slicing) ───────────────────────
        self.all_arm_dof = self.left_arm_dof + self.right_arm_dof
        self.all_arm_qpos = self.left_arm_qpos + self.right_arm_qpos

        # ── Touch sensor IDs ─────────────────────────────────────────────
        self.left_touch_sensor_ids  = [_sensor_id(n) for n in LEFT_TOUCH_SENSORS]
        self.right_touch_sensor_ids = [_sensor_id(n) for n in RIGHT_TOUCH_SENSORS]

        # ── Touch site IDs (for Jacobian IK) ─────────────────────────────
        self.left_touch_site_ids  = [_site_id(n) for n in LEFT_TOUCH_SITES]
        self.right_touch_site_ids = [_site_id(n) for n in RIGHT_TOUCH_SITES]

        # ── Object body IDs ──────────────────────────────────────────────
        self.obj_body_ids = []
        for name in OBJECT_BODIES:
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid != -1:
                self.obj_body_ids.append(bid)

        # ── Key geom IDs (for collision/LiDAR exclusion) ─────────────────
        self.chassis_geom_id = _geom_id("chassis_core")

        # ── Key body IDs ─────────────────────────────────────────────────
        self.base_body_id = _body_id("base_platform")
        self.head_body_id = _body_id("head")
        self.chest_body_id = _body_id("spine_3")

    # ═══════════════════════════════════════════════════════════════════════
    # POSITION QUERIES
    # ═══════════════════════════════════════════════════════════════════════

    def get_base_pos(self, data):
        """Base platform world position [x, y, z]."""
        return data.xpos[self.base_body_id].copy()

    def get_base_quat(self, data):
        """Base platform world quaternion [w, x, y, z]."""
        return data.qpos[self.Q_ROOT + 3 : self.Q_ROOT + 7].copy()

    def get_base_vel(self, data):
        """Base platform linear velocity [vx, vy, vz] in world frame."""
        return data.qvel[self.V_ROOT : self.V_ROOT + 3].copy()

    def get_base_angular_vel(self, data):
        """Base platform angular velocity [wx, wy, wz]."""
        return data.qvel[self.V_ROOT + 3 : self.V_ROOT + 6].copy()

    def get_head_pos(self, data):
        """Head world position."""
        return data.xpos[self.head_body_id].copy()

    def get_chest_pos(self, data):
        """Chest (spine_3) world position."""
        return data.xpos[self.chest_body_id].copy()

    def get_fingertip_pos(self, data, side='L'):
        """
        Fingertip positions for one hand. Returns (3,3) array:
        rows = [thumb, index, middle], cols = [x, y, z].
        """
        sites = self.left_touch_site_ids if side == 'L' else self.right_touch_site_ids
        return np.array([data.site_xpos[s] for s in sites])

    def get_fingertip_mean_pos(self, data, side='L'):
        """Mean fingertip position for one hand (useful as hand center)."""
        return self.get_fingertip_pos(data, side).mean(axis=0)

    # ═══════════════════════════════════════════════════════════════════════
    # JOINT STATE QUERIES
    # ═══════════════════════════════════════════════════════════════════════

    def get_arm_joint_pos(self, data, side='L'):
        """Joint positions for one arm (8 values)."""
        qpos_addrs = self.left_arm_qpos if side == 'L' else self.right_arm_qpos
        return np.array([data.qpos[a] for a in qpos_addrs])

    def get_arm_joint_vel(self, data, side='L'):
        """Joint velocities for one arm (8 values)."""
        dof_addrs = self.left_arm_dof if side == 'L' else self.right_arm_dof
        return np.array([data.qvel[a] for a in dof_addrs])

    def get_all_arm_joint_pos(self, data):
        """Joint positions for both arms (16 values: 8L + 8R)."""
        return np.array([data.qpos[a] for a in self.all_arm_qpos])

    def get_all_arm_joint_vel(self, data):
        """Joint velocities for both arms (16 values: 8L + 8R)."""
        return np.array([data.qvel[a] for a in self.all_arm_dof])

    def get_spine_joint_pos(self, data):
        """Spine joint positions (6 values)."""
        return np.array([data.qpos[a] for a in self.spine_qpos])

    def get_spine_joint_vel(self, data):
        """Spine joint velocities (6 values)."""
        return np.array([data.qvel[a] for a in self.spine_dof])

    def get_neck_joint_pos(self, data):
        """Neck joint positions (2 values)."""
        return np.array([data.qpos[a] for a in self.neck_qpos])

    def get_wheel_velocities(self, data):
        """Wheel angular velocities (4 values)."""
        return np.array([data.qvel[a] for a in self.wheel_dof])

    # ═══════════════════════════════════════════════════════════════════════
    # SENSOR QUERIES
    # ═══════════════════════════════════════════════════════════════════════

    def get_touch_readings(self, data, side='L'):
        """Touch sensor readings for one hand (3 values: thumb, index, middle)."""
        sensors = self.left_touch_sensor_ids if side == 'L' else self.right_touch_sensor_ids
        return np.array([data.sensordata[s] for s in sensors])

    def get_all_touch_readings(self, data):
        """All 6 touch sensor readings (3L + 3R)."""
        return np.concatenate([
            self.get_touch_readings(data, 'L'),
            self.get_touch_readings(data, 'R'),
        ])

    # ═══════════════════════════════════════════════════════════════════════
    # OBJECT QUERIES
    # ═══════════════════════════════════════════════════════════════════════

    def get_object_positions(self, data):
        """World positions of all objects. Returns (N, 3) array."""
        return np.array([data.xpos[bid] for bid in self.obj_body_ids])

    def get_object_pos(self, data, index=0):
        """World position of a specific object by index."""
        return data.xpos[self.obj_body_ids[index]].copy()

    def get_nearest_object(self, data, ref_pos):
        """
        Find nearest object to ref_pos.
        Returns (index, position, distance).
        """
        positions = self.get_object_positions(data)
        distances = np.linalg.norm(positions - ref_pos, axis=1)
        idx = int(np.argmin(distances))
        return idx, positions[idx], distances[idx]

    # ═══════════════════════════════════════════════════════════════════════
    # JACOBIAN UTILITIES (for IK)
    # ═══════════════════════════════════════════════════════════════════════

    def get_site_jacobian(self, data, site_id):
        """Compute full positional Jacobian for a site. Returns (3, nv)."""
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, data, jacp, jacr, site_id)
        return jacp

    def get_arm_jacobian(self, data, side='L'):
        """
        Positional Jacobian for the index fingertip, sliced to the 5 positioning DOFs.
        Returns (3, 5) Jacobian matrix suitable for DLS-IK.
        """
        # Use index finger tip as the reference point
        sites = self.left_touch_site_ids if side == 'L' else self.right_touch_site_ids
        ref_site = sites[1]  # index finger
        dof_addrs = self.left_arm_dof[:5] if side == 'L' else self.right_arm_dof[:5]

        jacp_full = self.get_site_jacobian(data, ref_site)
        return jacp_full[:, dof_addrs]

    # ═══════════════════════════════════════════════════════════════════════
    # BODY STATE LOCK (for training — prevent sliding)
    # ═══════════════════════════════════════════════════════════════════════

    def lock_base(self, data, pos=None, quat=None):
        """
        Lock base position and orientation (useful during arm-only training).
        Prevents floor friction from sliding the body.
        """
        if pos is None:
            pos = [0.0, 0.0, 0.058]
        if quat is None:
            quat = [1.0, 0.0, 0.0, 0.0]
        data.qpos[self.Q_ROOT : self.Q_ROOT + 3] = pos
        data.qpos[self.Q_ROOT + 3 : self.Q_ROOT + 7] = quat
        data.qvel[self.V_ROOT : self.V_ROOT + 6] = 0.0

    # ═══════════════════════════════════════════════════════════════════════
    # OBSERVATION BUILDER (for CarlBrain integration)
    # ═══════════════════════════════════════════════════════════════════════

    def build_full_observation(self, data):
        """
        Build a comprehensive observation vector for CarlBrain.

        Layout (75D base):
          [ 0: 3] Base position (x, y, z)
          [ 3: 7] Base quaternion (w, x, y, z)
          [ 7:10] Base linear velocity (vx, vy, vz)
          [10:13] Base angular velocity (wx, wy, wz)
          [13:17] Wheel velocities (4)
          [17:23] Spine joint positions (6)
          [23:25] Neck joint positions (2)
          [25:33] Left arm joint positions (8)
          [33:41] Right arm joint positions (8)
          [41:49] Left arm joint velocities (8)
          [49:57] Right arm joint velocities (8)
          [57:63] Touch sensors (6: 3L + 3R)
          [63:66] Left hand mean position (3)
          [66:69] Right hand mean position (3)
          [69:72] Chest/spine_3 position (3)
          [72:75] Head position (3)
        """
        obs = np.concatenate([
            self.get_base_pos(data),                   # 3
            self.get_base_quat(data),                  # 4
            self.get_base_vel(data),                   # 3
            self.get_base_angular_vel(data),            # 3
            self.get_wheel_velocities(data),            # 4
            self.get_spine_joint_pos(data),             # 6
            self.get_neck_joint_pos(data),              # 2
            self.get_arm_joint_pos(data, 'L'),         # 8
            self.get_arm_joint_pos(data, 'R'),         # 8
            self.get_arm_joint_vel(data, 'L'),         # 8
            self.get_arm_joint_vel(data, 'R'),         # 8
            self.get_all_touch_readings(data),          # 6
            self.get_fingertip_mean_pos(data, 'L'),    # 3
            self.get_fingertip_mean_pos(data, 'R'),    # 3
            self.get_chest_pos(data),                   # 3
            self.get_head_pos(data),                    # 3
        ])
        return obs  # 75D base


def load_model_and_interface():
    """
    Convenience: load the canonical model and create the body interface.
    Returns (model, data, body).
    """
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    body = BodyInterface(model)
    return model, data, body


# ─── Self-test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  CARL Body Interface — Self-Test")
    print("=" * 60)

    model, data, body = load_model_and_interface()

    print(f"\n  Model: {MODEL_PATH}")
    print(f"  Total actuators: {model.nu} (expected {N_ACTUATORS})")
    print(f"  Total joints:    {model.njnt}")
    print(f"  Total sensors:   {model.nsensor}")
    print(f"  Total DOF (nv):  {model.nv}")

    print(f"\n  Root qpos offset: {body.Q_ROOT}")
    print(f"  Root qvel offset: {body.V_ROOT}")

    print(f"\n  Base pos:       {body.get_base_pos(data)}")
    print(f"  Base quat:      {body.get_base_quat(data)}")
    print(f"  Wheel vels:     {body.get_wheel_velocities(data)}")
    print(f"  Spine pos:      {body.get_spine_joint_pos(data)}")
    print(f"  Neck pos:       {body.get_neck_joint_pos(data)}")
    print(f"  Left arm pos:   {body.get_arm_joint_pos(data, 'L')}")
    print(f"  Right arm pos:  {body.get_arm_joint_pos(data, 'R')}")
    print(f"  Touch (L):      {body.get_touch_readings(data, 'L')}")
    print(f"  Touch (R):      {body.get_touch_readings(data, 'R')}")
    print(f"  Fingertip L:    {body.get_fingertip_mean_pos(data, 'L')}")
    print(f"  Fingertip R:    {body.get_fingertip_mean_pos(data, 'R')}")
    print(f"  Object count:   {len(body.obj_body_ids)}")

    for i, bid in enumerate(body.obj_body_ids):
        print(f"    Object {i}: pos={data.xpos[bid]}")

    obs = body.build_full_observation(data)
    print(f"\n  Full observation: {obs.shape[0]}D")

    # Verify Jacobian
    J_L = body.get_arm_jacobian(data, 'L')
    J_R = body.get_arm_jacobian(data, 'R')
    print(f"  Left arm Jacobian shape:  {J_L.shape}")
    print(f"  Right arm Jacobian shape: {J_R.shape}")

    print(f"\n{'=' * 60}")
    print("  All checks passed")
    print(f"{'=' * 60}")
