# genesis_physics.py
# The Nervous System of CARL Genesis.
# Sensation and movement. The bridge between mind and world.
# 
# Key changes from Phase 18:
# - 4-wheel differential drive (no balance needed)
# - 16-ray 360° Lidar (full awareness, not just forward)
# - Actuator layout: [lw_l, lw_r, rw_l, rw_r, neck, pan] per body
# - No fall detection based on pitch — CARL cannot tip over

import math, numpy as np, mujoco, mujoco.viewer, time

# ─────────────────────────────────────────────────────────────────────────────
# Body Configuration
# Each CARL body has:
#   - A freejoint for 6-DOF movement
#   - 4 wheel hinges (lfw, lrw, rfw, rrw)
#   - 1 neck slide joint
#   - 1 head pan hinge
#   - 6 actuators (lw_pair, rw_pair, neck, pan)
# ─────────────────────────────────────────────────────────────────────────────

GCFG = [
    dict(
        name   = 'A',
        fj     = 'fj_A',
        joints = dict(lfw='lfw_A_j', lrw='lrw_A_j', rfw='rfw_A_j',
                      rrw='rrw_A_j', neck='neck_lift_A', pan='head_pan_A'),
        ctrls  = dict(lw_l=0, lw_r=1, rw_l=2, rw_r=3, neck=4, pan=5),
        geoms  = ['chassis_A', 'lfw_A_g', 'lrw_A_g', 'rfw_A_g', 'rrw_A_g',
                  'neck_tube_A', 'head_box_A', 'eye_L_A', 'eye_R_A'],
    ),
    dict(
        name   = 'B',
        fj     = 'fj_B',
        joints = dict(lfw='lfw_B_j', lrw='lrw_B_j', rfw='rfw_B_j',
                      rrw='rrw_B_j', neck='neck_lift_B', pan='head_pan_B'),
        ctrls  = dict(lw_l=6, lw_r=7, rw_l=8, rw_r=9, neck=10, pan=11),
        geoms  = ['chassis_B', 'lfw_B_g', 'lrw_B_g', 'rfw_B_g', 'rrw_B_g',
                  'neck_tube_B', 'head_box_B', 'eye_L_B', 'eye_R_B'],
    ),
]

# Food pellet names (for proximity sensing)
FOOD_NAMES = [f'food_{i}' for i in range(5)]

# Wall geom names for contact detection
WALL_NAMES = (['w_s', 'w_n', 'w_w', 'w_e']
              + [f'wi{i}' for i in range(9)])

# ─────────────────────────────────────────────────────────────────────────────
# Global physics state
# ─────────────────────────────────────────────────────────────────────────────

_model  = None
_data   = None
_viewer = None

_wall_gids  = set()
_robot_gids = {}    # {'A': set(), 'B': set()}
_food_gids  = {}    # {geom_id: pellet_index}

_lidar_data = {'A': [5.0]*16, 'B': [5.0]*16}
_lidar_rays = {'A': [], 'B': []}

# Trajectory visualization
_trajectories = {'A': [], 'B': []}

# Food state: active = visible/edible, False = eaten (respawning)
_food_active     = [True] * 5
_food_respawn_at = [0] * 5    # step number when pellet respawns
FOOD_RESPAWN_STEPS = 28800    # 120 seconds at 240Hz


def init_physics(xml_path='genesis_body.xml'):
    global _model, _data, _wall_gids, _robot_gids, _food_gids

    _model = mujoco.MjModel.from_xml_path(xml_path)
    _data  = mujoco.MjData(_model)

    # Resolve wall geom IDs
    _wall_gids = set()
    for name in WALL_NAMES:
        try:
            _wall_gids.add(_model.geom(name).id)
        except Exception:
            pass

    # Resolve robot geom IDs per body
    for cfg in GCFG:
        ids = set()
        for gname in cfg['geoms']:
            try:
                ids.add(_model.geom(gname).id)
            except Exception:
                pass
        _robot_gids[cfg['name']] = ids

    # Resolve food geom IDs
    for i, fname in enumerate(FOOD_NAMES):
        try:
            _food_gids[_model.geom(fname).id] = i
        except Exception:
            pass

    print(f'  [PHYSICS] Model loaded: nq={_model.nq}, nv={_model.nv}, '
          f'nu={_model.nu}, ngeom={_model.ngeom}')
    print(f'  [PHYSICS] Walls: {len(_wall_gids)} | '
          f'Food pellets: {len(_food_gids)}')
    return _model, _data


def launch_viewer():
    global _viewer
    _viewer = mujoco.viewer.launch_passive(_model, _data)
    _viewer.cam.azimuth  = 45
    _viewer.cam.elevation = -28
    _viewer.cam.distance  = 10.5
    _viewer.cam.lookat[:] = [3.6, 3.0, 0.4]
    return _viewer


def viewer_sync():
    if _viewer and _viewer.is_running():
        _viewer.sync()


def viewer_running():
    return _viewer.is_running() if _viewer else True


def step_physics():
    mujoco.mj_step(_model, _data)


def forward_physics():
    mujoco.mj_forward(_model, _data)


# ─────────────────────────────────────────────────────────────────────────────
# State extraction
# ─────────────────────────────────────────────────────────────────────────────

def _fj_qpos_addr(cfg):
    """Return qpos start index for this body's freejoint."""
    jid = mujoco.mj_name2id(_model, mujoco.mjtObj.mjOBJ_JOINT, cfg['fj'])
    return _model.jnt_qposadr[jid]

def _fj_qvel_addr(cfg):
    jid = mujoco.mj_name2id(_model, mujoco.mjtObj.mjOBJ_JOINT, cfg['fj'])
    return _model.jnt_dofadr[jid]


def get_pos_2d(cfg):
    """Return (x, y) world position of the body."""
    q0 = _fj_qpos_addr(cfg)
    return float(_data.qpos[q0]), float(_data.qpos[q0 + 1])


def get_yaw(cfg):
    """Return yaw angle of the body (rotation around Z axis)."""
    q0 = _fj_qpos_addr(cfg)
    quat = _data.qpos[q0 + 3: q0 + 7]
    w, x, y, z = float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def get_velocity(cfg):
    """Return forward velocity of the body."""
    v0 = _fj_qvel_addr(cfg)
    vx = float(_data.qvel[v0])
    vy = float(_data.qvel[v0 + 1])
    return math.sqrt(vx * vx + vy * vy)


def get_state_vector(cfg):
    """
    Return a compact state vector for the reservoir input.
    [x, y, vx, vy, yaw, yaw_rate, neck_pos, head_pan_pos]
    """
    q0 = _fj_qpos_addr(cfg)
    v0 = _fj_qvel_addr(cfg)

    x   = float(_data.qpos[q0])
    y   = float(_data.qpos[q0 + 1])
    vx  = float(_data.qvel[v0])
    vy  = float(_data.qvel[v0 + 1])
    yaw = get_yaw(cfg)
    yaw_rate = float(_data.qvel[v0 + 5])

    # Neck and head pan positions
    try:
        neck_jid = mujoco.mj_name2id(_model, mujoco.mjtObj.mjOBJ_JOINT,
                                     cfg['joints']['neck'])
        pan_jid  = mujoco.mj_name2id(_model, mujoco.mjtObj.mjOBJ_JOINT,
                                     cfg['joints']['pan'])
        neck_pos = float(_data.qpos[_model.jnt_qposadr[neck_jid]])
        pan_pos  = float(_data.qpos[_model.jnt_qposadr[pan_jid]])
    except Exception:
        neck_pos = 0.12
        pan_pos  = 0.0

    return np.array([x, y, vx, vy, yaw, yaw_rate, neck_pos, pan_pos],
                    dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 360° Lidar — 16 rays, full awareness
# ─────────────────────────────────────────────────────────────────────────────

def get_lidar_360(cfg, n_rays=16, max_range=5.0):
    """
    Cast 16 rays evenly around the robot (360° coverage).
    
    Returns
    -------
    dists      : list[float] — distance to nearest obstacle per ray (0-5m)
    hit_points : list[tuple] — (x, y) world coords of hits
    """
    q0  = _fj_qpos_addr(cfg)
    x   = float(_data.qpos[q0])
    y   = float(_data.qpos[q0 + 1])
    z   = float(_data.qpos[q0 + 2]) + 0.30   # Fire from head height
    yaw = get_yaw(cfg)

    pnt = np.array([x, y, z], dtype=np.float64)
    dists      = []
    hit_points = []
    rays       = []

    angle_step = 2.0 * math.pi / n_rays
    # 3 Elevations: -15 deg, 0 deg, +15 deg (in radians)
    elevations = [-0.2618, 0.0, 0.2618]
    
    for i in range(n_rays):
        angle = yaw + i * angle_step
        
        min_dist = max_range
        best_hp  = None
        best_ray = None
        
        for el in elevations:
            vec = np.array([
                math.cos(el) * math.cos(angle),
                math.cos(el) * math.sin(angle),
                math.sin(el)
            ], dtype=np.float64)

            geom_id = np.array([-1], dtype=np.int32)
            dist    = mujoco.mj_ray(_model, _data, pnt, vec, None, 1, -1, geom_id)

            own_gids = _robot_gids.get(cfg['name'], set())
            if geom_id[0] in own_gids:
                dist = max_range

            dist = float(dist)
            if dist < 0 or dist > max_range:
                dist = max_range
                ray_tuple = (pnt.copy(), pnt + vec * max_range, False)
            else:
                hp = pnt + vec * dist
                ray_tuple = (pnt.copy(), hp, True)
                
            if dist <= min_dist:
                min_dist = dist
                if dist < max_range:
                    best_hp = hp
                best_ray = ray_tuple

        dists.append(min_dist)
        if best_ray is not None:
            rays.append(best_ray)
        if best_hp is not None:
            hit_points.append((float(best_hp[0]), float(best_hp[1])))

    _lidar_data[cfg['name']] = dists
    _lidar_rays[cfg['name']] = rays
    return dists, hit_points


def get_lidar_last(cfg):
    """Return last computed Lidar readings (fast, no recomputation)."""
    return _lidar_data.get(cfg['name'], [5.0] * 16)


# ─────────────────────────────────────────────────────────────────────────────
# Motor control
# ─────────────────────────────────────────────────────────────────────────────

def apply_drive(cfg, left_vel, right_vel):
    """
    Apply differential drive velocity commands.
    
    Left and right wheel pairs are controlled together.
    Positive = forward, negative = reverse.
    Difference creates steering.
    
    Parameters
    ----------
    left_vel  : float — target velocity for left wheel pair  [-8, 8] rad/s
    right_vel : float — target velocity for right wheel pair [-8, 8] rad/s
    """
    lv = float(np.clip(left_vel,  -40.0, 40.0))
    rv = float(np.clip(right_vel, -40.0, 40.0))

    c = cfg['ctrls']
    _data.ctrl[c['lw_l']] = lv
    _data.ctrl[c['lw_r']] = lv
    _data.ctrl[c['rw_l']] = rv
    _data.ctrl[c['rw_r']] = rv


def set_neck(cfg, height):
    """
    Set neck height (emotional body language).
    
    height : float [0.0, 0.28] — 0 = fully retracted (fear), 0.28 = fully extended (curiosity)
    """
    c = cfg['ctrls']
    _data.ctrl[c['neck']] = float(np.clip(height, 0.0, 0.28))


def set_head_pan(cfg, angle):
    """
    Set head pan angle (attention direction).
    
    angle : float [-1.047, 1.047] radians — negative = left, positive = right
    """
    c = cfg['ctrls']
    _data.ctrl[c['pan']] = float(np.clip(angle, -1.047, 1.047))


def stop(cfg):
    """Emergency stop — zero all wheel velocities."""
    apply_drive(cfg, 0.0, 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Teleport / Reset
# ─────────────────────────────────────────────────────────────────────────────

def reset_body(cfg, x, y, yaw=0.0):
    """
    Respawn the body at a given position with zero velocity.
    Used for biological resurrection after energy depletion.
    """
    q0 = _fj_qpos_addr(cfg)
    v0 = _fj_qvel_addr(cfg)

    _data.qpos[q0]     = x
    _data.qpos[q0 + 1] = y
    _data.qpos[q0 + 2] = 0.07   # Slightly above floor

    # Set yaw quaternion
    half_yaw = yaw / 2.0
    _data.qpos[q0 + 3] = math.cos(half_yaw)   # w
    _data.qpos[q0 + 4] = 0.0                   # x
    _data.qpos[q0 + 5] = 0.0                   # y
    _data.qpos[q0 + 6] = math.sin(half_yaw)   # z

    # Zero all velocities
    nv_fj = 6   # freejoint has 6 DOF
    _data.qvel[v0: v0 + nv_fj] = 0.0

    forward_physics()


# ─────────────────────────────────────────────────────────────────────────────
# Collision detection
# ─────────────────────────────────────────────────────────────────────────────

def check_wall_contact(cfg):
    """Return True if this body is in contact with any wall geom."""
    own_gids = _robot_gids.get(cfg['name'], set())
    for ci in range(_data.ncon):
        c = _data.contact[ci]
        g1, g2 = int(c.geom1), int(c.geom2)
        if ((g1 in own_gids and g2 in _wall_gids)
                or (g2 in own_gids and g1 in _wall_gids)):
            return True
    return False


def check_food_contact(cfg, step):
    """
    Return index of food pellet being eaten, or -1 if none.
    Uses distance-based checking instead of physics contacts, 
    since food geoms have contype=0 (no collisions).
    """
    x, y = get_pos_2d(cfg)
    food_positions = get_food_positions()
    
    for i, pos in enumerate(food_positions):
        if pos is None or not _food_active[i]:
            continue
        dist = math.sqrt((x - pos[0])**2 + (y - pos[1])**2)
        if dist < 0.25:  # Eat threshold (robot radius + food radius + margin)
            return i
            
    return -1


def eat_food(idx, step):
    """Mark food pellet as eaten. It will respawn after FOOD_RESPAWN_STEPS."""
    _food_active[idx]     = False
    _food_respawn_at[idx] = step + FOOD_RESPAWN_STEPS


def update_food_respawn(step):
    """Respawn any pellets whose timer has elapsed."""
    for i in range(5):
        if not _food_active[i] and step >= _food_respawn_at[i]:
            _food_active[i] = True


def get_food_positions():
    """Return (x, y) positions of all active food pellets."""
    positions = []
    for fname in FOOD_NAMES:
        try:
            gid = _model.geom(fname).id
            # Get geom position from model (static in world)
            pos = _model.geom_pos[gid]
            positions.append((float(pos[0]), float(pos[1])))
        except Exception:
            positions.append(None)
    return positions

def get_wall_positions():
    """Return (x, y, size_x, size_y) of all static walls for rasterization."""
    positions = []
    for gid in _wall_gids:
        try:
            pos = _model.geom_pos[gid]
            size = _model.geom_size[gid]
            positions.append((float(pos[0]), float(pos[1]), float(size[0]), float(size[1])))
        except Exception:
            pass
    return positions


def nearest_food(x, y):
    """
    Return (food_index, food_x, food_y, distance) of nearest active pellet.
    Returns None if no food is active.
    """
    food_positions = get_food_positions()
    best = None
    for i, pos in enumerate(food_positions):
        if pos is None or not _food_active[i]:
            continue
        d = math.sqrt((x - pos[0])**2 + (y - pos[1])**2)
        if best is None or d < best[3]:
            best = (i, pos[0], pos[1], d)
    return best


# ─────────────────────────────────────────────────────────────────────────────
# Environmental forces
# ─────────────────────────────────────────────────────────────────────────────

def apply_wind(cfg, fx, fy=0.0):
    """Apply a lateral force to simulate wind."""
    q0 = _fj_qpos_addr(cfg)
    x  = float(_data.qpos[q0])
    y  = float(_data.qpos[q0 + 1])
    # Find chassis geom body and apply xfrc
    try:
        bid = _model.body(f'base_{cfg["name"]}').id
        _data.xfrc_applied[bid, 0] = fx
        _data.xfrc_applied[bid, 1] = fy
    except Exception:
        pass


def clear_wind(cfg):
    try:
        bid = _model.body(f'base_{cfg["name"]}').id
        _data.xfrc_applied[bid, :] = 0.0
    except Exception:
        pass
