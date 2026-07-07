"""carl_mj_physics.py — MuJoCo physics layer for CARL Phase 15"""
import math, numpy as np, mujoco, mujoco.viewer, time

# ── qpos / qvel layout (verified: nq=20, nv=18) ──────────────
# fj_A : qpos[0:7]   qvel[0:6]
# lw_A  : qpos[7]    qvel[6]
# rw_A  : qpos[8]    qvel[7]
# neck_A: qpos[9]    qvel[8]
# fj_B : qpos[10:17] qvel[9:15]
# lw_B  : qpos[17]   qvel[15]
# rw_B  : qpos[18]   qvel[16]
# neck_B: qpos[19]   qvel[17]
#
# ctrl: 0=lw_A, 1=rw_A, 2=neck_A(pos), 3=lw_B, 4=rw_B, 5=neck_B(pos)

BCFG = [
    dict(name='A', q0=0,  v0=0,  nq=9,  nv=8,  cl=0, cr=1, cn=2),
    dict(name='B', q0=10, v0=9,  nq=19, nv=17, cl=3, cr=4, cn=5),
]

_model  = None
_data   = None
_viewer = None
_wall_gids  = set()   # wall geom IDs
_robot_gids = {}      # {'A': set(), 'B': set()}
_multiverse_trajectories = {'A': [], 'B': []}
_lidar_data = {'A': [5.0]*5, 'B': [5.0]*5}
_lidar_rays = {'A': [], 'B': []}


def init_physics(xml_path='carl_mujoco.xml'):
    global _model, _data, _wall_gids, _robot_gids
    _model = mujoco.MjModel.from_xml_path(xml_path)
    _data  = mujoco.MjData(_model)

    wall_names = ['w_s','w_n','w_w','w_e'] + [f'wi{i}' for i in range(9)]
    _wall_gids = {_model.geom(n).id for n in wall_names}

    for cfg in BCFG:
        n = cfg['name']
        _robot_gids[n] = {
            _model.geom(f'chassis_{n}').id,
            _model.geom(f'lw_{n}_tire').id,
            _model.geom(f'rw_{n}_tire').id,
        }
    return _model, _data


def launch_viewer():
    global _viewer
    _viewer = mujoco.viewer.launch_passive(_model, _data)
    _viewer.cam.azimuth  = 45
    _viewer.cam.elevation= -32
    _viewer.cam.distance = 9.0
    _viewer.cam.lookat[:] = [3.5, 2.8, 0.3]
    return _viewer


def viewer_sync():
    global _multiverse_trajectories, _lidar_rays
    if _viewer and _viewer.is_running():
        _viewer.user_scn.ngeom = 0
        
        # Draw Lidar rays
        for name, rays in _lidar_rays.items():
            for p1, p2, hit in rays:
                if _viewer.user_scn.ngeom >= _viewer.user_scn.maxgeom: break
                color = np.array([1, 0, 0, 1]) if hit else np.array([0, 1, 0.5, 0.5])
                mujoco.mjv_initGeom(_viewer.user_scn.geoms[_viewer.user_scn.ngeom],
                                    mujoco.mjtGeom.mjGEOM_LINE, np.zeros(3),
                                    np.zeros(3), np.zeros(9), color)
                mujoco.mjv_connector(_viewer.user_scn.geoms[_viewer.user_scn.ngeom],
                                         mujoco.mjtGeom.mjGEOM_LINE, 2,
                                         np.array(p1), np.array(p2))
                _viewer.user_scn.ngeom += 1

        _viewer.sync()

def viewer_running():
    return _viewer is not None and _viewer.is_running()

def get_lidar(cfg):
    """Cast 5 rays forward, return distances"""
    q0 = cfg['q0']
    x_pos = float(_data.qpos[q0])
    y_pos = float(_data.qpos[q0+1])
    z_pos = float(_data.qpos[q0+2]) + 0.15  # from head height, lower to hit 0.35m walls
    _, _, yaw = quat_to_euler(_data.qpos[q0+3:q0+7])
    
    angles = [-0.785, -0.392, 0.0, 0.392, 0.785] # -45, -22.5, 0, 22.5, 45 deg
    dists = []
    rays = []
    hit_points = []
    pnt = np.array([x_pos, y_pos, z_pos], dtype=np.float64)
    geomid = np.array([-1], dtype=np.int32)
    
    for a in angles:
        vec = np.array([math.cos(yaw+a), math.sin(yaw+a), 0], dtype=np.float64)
        dist = mujoco.mj_ray(_model, _data, pnt, vec, None, 1, -1, geomid)
        if dist < 0 or dist > 5.0:
            dist = 5.0
            rays.append((pnt, pnt + vec*5.0, False))
        else:
            hit_p = pnt + vec*dist
            rays.append((pnt, hit_p, True))
            hit_points.append((hit_p[0], hit_p[1]))
        dists.append(dist)
    
    _lidar_data[cfg['name']] = dists
    _lidar_rays[cfg['name']] = rays
    return dists, hit_points


def quat_to_euler(q):
    w, x, y, z = q
    roll  = math.atan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
    sinp  = float(np.clip(2*(w*y - z*x), -1.0, 1.0))
    pitch = math.asin(sinp)
    yaw   = math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    return roll, pitch, yaw


def get_state(cfg):
    """Returns (state_5d, y_pos, z_pos, roll)"""
    q0, v0 = cfg['q0'], cfg['v0']
    x_pos  = float(_data.qpos[q0])
    y_pos  = float(_data.qpos[q0+1])
    z_pos  = float(_data.qpos[q0+2])
    roll, pitch, yaw = quat_to_euler(_data.qpos[q0+3:q0+7])
    x_vel     = float(_data.qvel[v0])
    pitch_vel = float(_data.qvel[v0+4])
    neck      = float(_data.qpos[cfg['nq']])
    state = np.array([x_pos, x_vel, pitch, pitch_vel, neck], dtype=float)
    return state, y_pos, z_pos, roll


def get_yaw(cfg):
    q0 = cfg['q0']
    _, _, yaw = quat_to_euler(_data.qpos[q0+3:q0+7])
    return float(yaw)


def get_pos_2d(cfg):
    q0 = cfg['q0']
    return float(_data.qpos[q0]), float(_data.qpos[q0+1])


def apply_torque(cfg, u_fwd, u_steer=0.0):
    fwd   = float(np.clip(u_fwd,   -8., 8.))
    steer = float(np.clip(u_steer, -3., 3.))
    _data.ctrl[cfg['cl']] = fwd + steer
    _data.ctrl[cfg['cr']] = fwd - steer


def reset_robot(cfg, x_off=0.4, y_off=0.4, init_pitch=0.0):
    q0, v0 = cfg['q0'], cfg['v0']
    _data.qpos[q0]   = x_off
    _data.qpos[q0+1] = y_off
    _data.qpos[q0+2] = 0.10
    hp = init_pitch / 2.0
    _data.qpos[q0+3] = math.cos(hp)  # qw
    _data.qpos[q0+4] = 0.0           # qx
    _data.qpos[q0+5] = math.sin(hp)  # qy
    _data.qpos[q0+6] = 0.0           # qz
    _data.qpos[q0+7] = 0.0  # lw angle
    _data.qpos[q0+8] = 0.0  # rw angle
    _data.qpos[cfg['nq']] = 0.0
    _data.qvel[v0:v0+9] = 0.0
    _data.ctrl[cfg['cl']] = 0.0
    _data.ctrl[cfg['cr']] = 0.0
    _data.ctrl[cfg['cn']] = 0.0


def park_robot(cfg):
    """Move dead robot underground so it doesn't interfere."""
    q0 = cfg['q0']
    _data.qpos[q0]   = -5.0
    _data.qpos[q0+1] = -5.0
    _data.qpos[q0+2] = -5.0
    _data.qvel[cfg['v0']:cfg['v0']+9] = 0.0
    _data.ctrl[cfg['cl']] = 0.0
    _data.ctrl[cfg['cr']] = 0.0


def check_wall_contact(cfg):
    rg = _robot_gids[cfg['name']]
    for i in range(_data.ncon):
        c = _data.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        if (g1 in rg and g2 in _wall_gids) or (g2 in rg and g1 in _wall_gids):
            return True
    return False


def apply_wind(cfg, fx):
    bid = _model.body(f'base_{cfg["name"]}').id
    _data.xfrc_applied[bid, 0] = fx   # force X in world frame


def clear_wind(cfg):
    bid = _model.body(f'base_{cfg["name"]}').id
    _data.xfrc_applied[bid, :] = 0.0


def step_physics():
    mujoco.mj_step(_model, _data)


def forward_physics():
    mujoco.mj_forward(_model, _data)
