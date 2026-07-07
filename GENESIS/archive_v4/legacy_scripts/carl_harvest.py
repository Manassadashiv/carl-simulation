"""
carl_harvest.py — Cybernetic Interactive Tutorial & Expert Dataset Harvester.
Delivers instant, default navigational intelligence through analytical geometric priors.
"""

import math
import time
import json
import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
import numpy as np
import mujoco
import mujoco.viewer
import threading
import queue
import argparse
try:
    import keyboard
except ImportError:
    keyboard = None
import random

sys.path.append(os.path.abspath('.'))
from carl_agent import CarlBrain
from carl_expression import ExpressionController, ProceduralAudioSynthesizer
from carl_arm_train import ArmPolicy, build_cache, ARM_CTRL_LOW, ARM_CTRL_HIGH
from carl_bios import CarlBiosCore
from carl_double_buffer import SensorStateBuffer, AtomicDoubleBuffer
from carl_brainstem import BrainstemController, MinimumJerkSmoother
from carl_cpg import HopfCpgEngine
from carl_social import SocialVisualLobe
from carl_obstacle_controller import ObstacleController
from carl_mapping import OccupancyGrid
from carl_planner import AStarPlanner, PurePursuitFollower, ApexStateManager, GoalCrystallizer
from carl_curiosity import CuriosityEngine
from carl_imagination import ImaginationThread
from carl_active_inference import ActiveInferenceEngine
from carl_crucible import CrucibleController
from carl_workspace import WorkspaceSignal
from carl_sensor_fusion import ObservationBuilder
from carl_behavior import ActionContext, BehavioralArbitrator
from carl_evolution import EvolutionManager
from carl_metrics import EmergenceMetricsLogger

# ═══════════════════════════════════════════════════════════════════════════════
#  Experimental Configurations
# ═══════════════════════════════════════════════════════════════════════════════
PROCEDURAL_INHERITANCE = True
ECOLOGICAL_PERSISTENCE = True
GOAL_CRYSTALLIZATION = True
CULTURE_IMITATION = True
CRUCIBLE_ENABLED = True  # The Crucible: evolutionary pressure environment

# ═══════════════════════════════════════════════════════════════════════════════
#  Interactive Tutorial Engine
# ═══════════════════════════════════════════════════════════════════════════════

class TutorialNarrator:
    """Manages real-time narrative overlay tracking CARL's emergent intelligence."""
    def __init__(self):
        self.milestones = {
            'boot': False, 'standing': False, 'food_lock': False, 
            'teleop': False, 'harvest': False
        }
        self.start_time = time.time()

    def update(self, step, dist_to_food, human_active, surprise, drives):
        elapsed = time.time() - self.start_time
        
        if step == 1 and not self.milestones['boot']:
            self.milestones['boot'] = True
            self._print_banner("TUTORIAL PHASE 1: CYBERNETIC COLD BOOT")
            print("[NARRATOR] CARL (Bob Mark VIII) is awake. Asynchronous threads online.")
            print("[NARRATOR] Activating 40,000-D Holographic Sub-Cortical Memory Matrix...")
            print("[NARRATOR] Core timing initialized at 100Hz. Spinal cord reflexes hot.")

        if step == 150 and not self.milestones['standing']:
            self.milestones['standing'] = True
            self._print_banner("TUTORIAL PHASE 2: POSTURAL STEADY-STATE")
            print("[NARRATOR] Initial gravity shock absorbed. Anti-tipping torque clamp active.")
            print(f"[NARRATOR] Posture joint exploration standard deviation (log_std) stabilized.")
            print(f"[NARRATOR] Current Stress (Cortisol): {drives.cort:.3f} | Joy (Dopamine): {drives.da:.3f}")

        if dist_to_food < 3.0 and not self.milestones['food_lock']:
            self.milestones['food_lock'] = True
            self._print_banner("TUTORIAL PHASE 3: TARGET TRACKING ACTIVATED")
            print(f"[NARRATOR] LiDAR array and coordinate grounding have achieved target lock.")
            print(f"[NARRATOR] Distance to green food pellet: {dist_to_food:.2f} meters.")
            print("[NARRATOR] Default AI policy is calculating optimal geometric approach vectors.")

        if human_active and not self.milestones['teleop']:
            self.milestones['teleop'] = True
            print("\n" + "="*50)
            print("  [NARRATOR] USER INTERVENTION DETECTED!")
            print("  You have tapped the keyboard overrides. Auto-navigation paused.")
            print("  Bypassing Braitenberg safety reflexes. Human has sovereign control.")
            print("="*50 + "\n")

    def log_success(self, ep_idx, frames):
        self._print_banner(f"TUTORIAL MILESTONE: TARGET HARVESTED!")
        print(f"[NARRATOR] Episode {ep_idx} successfully completed!")
        print(f"[NARRATOR] Committed {frames} pristine trajectory frames to LeRobot Dataset.")
        print(f"[NARRATOR] Respawning food token at curriculum horizon. Continuing run...")

    def _print_banner(self, text):
        print("\n" + "="*72)
        print(f"  {text}")
        print("="*72 + "\n")


# ═══════════════════════════════════════════════════════════════════════════════
#  Automated Expert Data Collection Serializer
# ═══════════════════════════════════════════════════════════════════════════════

class LeRobotSerializer:
    def __init__(self, output_dir, fps=50):
        self.output_dir = output_dir
        self.fps = fps
        self.episodes_written = 0
        self.total_frames_written = 0
        os.makedirs(os.path.join(output_dir, "meta"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "data", "chunk-000"), exist_ok=True)

    def write_meta(self, total_episodes):
        info = {
            "codebase_version": "v3.0", "robot_type": "carl_mk8",
            "total_episodes": self.episodes_written, "total_frames": self.total_frames_written,
            "total_tasks": 1, "chunks_size": 1000, "fps": self.fps,
            "splits": {"train": f"0:{self.episodes_written}"},
            "data_path": "data/chunk-000/episode_{episode_index:06d}.parquet"
        }
        with open(os.path.join(self.output_dir, "meta", "info.json"), 'w') as f:
            json.dump(info, f, indent=2)

    def write_episode_fallback(self, ep_data, index):
        out_path = os.path.join(self.output_dir, "data", "chunk-000", f"episode_{index:06d}.npz")
        np.savez_compressed(out_path, observations=ep_data['observations'], actions=ep_data['actions'])
        self.episodes_written += 1
        self.total_frames_written += len(ep_data['observations'])


class HarvestRecorder:
    def __init__(self):
        self.current = None
        self.total_frames = 0

    def start_episode(self, idx):
        self.current = {'idx': idx, 'observations': [], 'actions': [], 'timestamps': []}

    def record_step(self, obs, act, ts):
        if self.current is not None:
            self.current['observations'].append(obs.copy())
            self.current['actions'].append(act.copy())
            self.current['timestamps'].append(ts)

    def finalize_episode(self):
        ep = self.current
        self.current = None
        self.total_frames += len(ep['observations'])
        return ep

# ═══════════════════════════════════════════════════════════════════════════════
#  Main Autonomous System Loop
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    global PROCEDURAL_INHERITANCE, ECOLOGICAL_PERSISTENCE, GOAL_CRYSTALLIZATION, CULTURE_IMITATION
    parser = argparse.ArgumentParser()
    parser.add_argument('--render', action='store_true', default=True)
    parser.add_argument('--no-render', action='store_true', default=False, help='Disable rendering')
    parser.add_argument('--no-keyboard', action='store_true', default=False, help='Disable keyboard monitoring')
    parser.add_argument('--procedural-inheritance', type=str, default='True')
    parser.add_argument('--ecological-persistence', type=str, default='True')
    parser.add_argument('--goal-crystallization', type=str, default='True')
    parser.add_argument('--culture-imitation', type=str, default='True')
    parser.add_argument('--max-generations', type=int, default=100)
    args = parser.parse_args()
    if args.no_render:
        args.render = False

    PROCEDURAL_INHERITANCE = (args.procedural_inheritance.lower() == 'true')
    ECOLOGICAL_PERSISTENCE = (args.ecological_persistence.lower() == 'true')
    GOAL_CRYSTALLIZATION = (args.goal_crystallization.lower() == 'true')
    CULTURE_IMITATION = (args.culture_imitation.lower() == 'true')

    def check_keyboard():
        if args.no_keyboard or keyboard is None:
            return False
        try:
            return any(keyboard.is_pressed(k) for k in ['w','a','s','d','up','down','left','right'])
        except Exception:
            return False

    print("=" * 72)
    print("  CARL GENESIS - Interactive AI Tutorial & Data Harvester")
    print("  Default Navigational Intelligence Stack: Mark VIII")
    print("=" * 72)

    mujoco_lock = threading.Lock()
    event_queue = queue.Queue()
    narrator = TutorialNarrator()
    viewer = None

    # -- Compile MuJoCo Engine Scene --
    model = mujoco.MjModel.from_xml_path("vessel_kinetic.xml")
    data  = mujoco.MjData(model)

    # Velocity actuators: ctrl = desired wheel velocity (rad/s). MuJoCo applies force = kv * (ctrl - joint_vel),
    # clamped to forcerange (±0.12 Nm). No external PID needed — physics engine handles tracking.

    with mujoco_lock:
        mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)

    # -- Initialize Cortical Architecture --
    brain = CarlBrain(n_obs=34)
    brain.load("memory/carl_brain")
    
    # --- NEURAL HANDOVER ENABLED ---
    # We set training = False during this test so the untrained Critic
    # doesn't immediately overwrite our pre-trained Actor with random gradients!
    brain.training = False          # inference-only mode for the test
    brain.learning_rate = 1e-4      # Set learning pace
    
    # Initialize reward tracking
    # (Moved below function definitions)
    
    sensor_buffer = SensorStateBuffer()
    brain.sensor_buffer = sensor_buffer
    obs_builder = ObservationBuilder()
    arbitrator = BehavioralArbitrator()
    evolution = EvolutionManager(elite_capacity=5)
    metrics_logger = EmergenceMetricsLogger()
    pose_buffer = AtomicDoubleBuffer(shape=(8,))
    brainstem = BrainstemController()
    cpg = HopfCpgEngine()
    hazard = ObstacleController(amplitude=0.8, speed=0.03)

    # ── Expression System ── brings the body to life
    expr_ctrl = ExpressionController()
    audio_synth = ProceduralAudioSynthesizer()
    # Smoothed expression targets — EMA prevents jerky joint snaps at 100Hz
    _smoothed_expr = {
        'neck_height': 0.1, 'head_pitch': 0.0, 'head_yaw': 0.0,
        'shoulder_yaw_L': 0.0, 'shoulder_L': 0.3, 'elbow_L': -0.5, 'wrist_L': 0.0, 'grip_L': 0.0,
        'shoulder_yaw_R': 0.0, 'shoulder_R': 0.3, 'elbow_R': -0.5, 'wrist_R': 0.0, 'grip_R': 0.0,
    }
    _EXPR_ALPHA = 0.04  # Smoothing: 0.04 = ~0.4s settling time at 100Hz

    # ── Neural Arm Policy ──
    arm_policy = ArmPolicy()
    arm_smoother = MinimumJerkSmoother(num_joints=5, duration=0.4, dt=0.01)
    try:
        arm_weights = np.load("memory/carl_arm_weights.npz")['weights']
        arm_policy.set_params(arm_weights)
        arm_cache = build_cache(model)
        print("[BRAIN] Neural arm policy loaded successfully")
    except Exception as e:
        print(f"[WARN] Could not load neural arm policy: {e}")
        arm_cache = None

    # Object curiosity state
    _obj_names = ['obj_cube_0', 'obj_cube_1', 'obj_cube_2', 'obj_ball_0', 'obj_ball_1', 'obj_toy']
    _obj_body_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in _obj_names]
    _obj_body_ids = [i for i in _obj_body_ids if i != -1]  # filter missing
    _touch_L_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, 'touch_L')
    _touch_R_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, 'touch_R')
    _curiosity_target = [None]   # nearest object position (shared)
    _touch_reward_cooldown = [0] # prevent DA reward spam
    _favorite_object_id = [None]
    _favorite_object_touches = {bid: 0 for bid in _obj_body_ids}

    # ── Arm Active Inference Engine ──
    arm_inference_queue = queue.Queue()
    arm_imagination = ImaginationThread(shared_sensor_buffer=sensor_buffer, transition_queue=arm_inference_queue, obs_dim=29, act_dim=10)
    arm_imagination.start()
    arm_active_inference = ActiveInferenceEngine(act_dim=10)
    arm_obs_history = np.zeros((10, 29), dtype=np.float32)
    prev_arm_action = np.zeros(10, dtype=np.float32)
    _occlusion_state = "VISIBLE"

    # Differential drive constants
    TRACK_WIDTH = 0.22    # meters between wheels
    WHEEL_RADIUS = 0.04   # meters

    steps_since_food = [0]
    planner_step = [0]

    # Resolve CARL's freejoint qpos base address
    _carl_jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'spatial_identity')
    Q_CARL = model.jnt_qposadr[_carl_jnt_id]  # base index for pos(3)+quat(4)
    V_CARL = model.jnt_dofadr[_carl_jnt_id]   # base index for vel(3 linear + 3 angular)

    def get_yaw():
        w, x, y, z = data.qpos[Q_CARL+3:Q_CARL+7]  # quaternion portion of freejoint
        return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    def get_lidar(n_rays=24):
        food_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "food")
        old_food_pos = model.geom_pos[food_id].copy()
        model.geom_pos[food_id] = np.array([99.0, 99.0, 99.0])
        
        blob_pos = data.geom("chassis").xpos
        yaw = get_yaw()
        dists = []
        for i in range(n_rays):
            angle = yaw + i * (2.0 * math.pi / n_rays)
            vx = math.cos(angle)
            vy = math.sin(angle)
            vec = np.array([vx, vy, 0.0])
            pnt = np.array([blob_pos[0] + vx * 0.22,
                            blob_pos[1] + vy * 0.22,
                            blob_pos[2] + 0.05])
            geom_hit = np.array([-1], dtype=np.int32)
            dist = mujoco.mj_ray(model, data, pnt, vec, None, 1, -1, geom_hit)
            dists.append((dist + 0.22) if dist > 0 else 5.0)
            
        model.geom_pos[food_id] = old_food_pos
        return dists

    _cached_lidar_24 = np.full(24, 5.0, dtype=np.float64)
    _lidar_tick_parity = 0

    def get_food_metrics():
        bp = data.geom("chassis").xpos
        fp = data.geom("food").xpos
        dx, dy = fp[0] - bp[0], fp[1] - bp[1]
        dist = math.hypot(dx, dy) + 1e-8
        yaw = get_yaw()
        local_x = dx * math.cos(yaw) + dy * math.sin(yaw)
        local_y = -dx * math.sin(yaw) + dy * math.cos(yaw)
        return local_x / dist, local_y / dist, dist

    def lidar_to_proximity(dists, max_range=2.5):
        return [max(0.0, (max_range - d) / max_range) for d in dists]

    # [NEW] Grid Connectivity Map
    valid_nodes = {
        (-3.6, 3.6): [(-3.6, 1.8), (-1.8, 3.6)],
        (-1.8, 3.6): [(-3.6, 3.6), (0.0, 3.6)],
        (0.0, 3.6): [(0.0, 1.8), (-1.8, 3.6)],
        (1.8, 3.6): [(1.8, 1.8), (3.6, 3.6)],
        (3.6, 3.6): [(3.6, 1.8), (1.8, 3.6)],
        (-3.6, 1.8): [(-3.6, 3.6), (-3.6, 0.0), (-1.8, 1.8)],
        (-1.8, 1.8): [(-3.6, 1.8), (0.0, 1.8)],
        (0.0, 1.8): [(0.0, 3.6), (0.0, 0.0), (-1.8, 1.8), (1.8, 1.8)],
        (1.8, 1.8): [(1.8, 3.6), (1.8, 0.0), (0.0, 1.8), (3.6, 1.8)],
        (3.6, 1.8): [(3.6, 3.6), (3.6, 0.0), (1.8, 1.8)],
        (-3.6, 0.0): [(-3.6, 1.8)],
        (-1.8, 0.0): [(-1.8, -1.8), (0.0, 0.0)],
        (0.0, 0.0): [(0.0, 1.8), (-1.8, 0.0)],
        (1.8, 0.0): [(1.8, 1.8), (1.8, -1.8), (3.6, 0.0)],
        (3.6, 0.0): [(3.6, 1.8), (1.8, 0.0)],
        (-3.6, -1.8): [(-3.6, -3.6), (-1.8, -1.8)],
        (-1.8, -1.8): [(-1.8, 0.0), (-1.8, -3.6), (-3.6, -1.8)],
        (0.0, -1.8): [(1.8, -1.8)],
        (1.8, -1.8): [(1.8, 0.0), (0.0, -1.8), (3.6, -1.8)],
        (3.6, -1.8): [(3.6, -3.6), (1.8, -1.8)],
        (-3.6, -3.6): [(-3.6, -1.8), (-1.8, -3.6)],
        (-1.8, -3.6): [(-1.8, -1.8), (-3.6, -3.6), (0.0, -3.6)],
        (0.0, -3.6): [(-1.8, -3.6), (1.8, -3.6)],
        (1.8, -3.6): [(0.0, -3.6), (3.6, -3.6)],
        (3.6, -3.6): [(3.6, -1.8), (1.8, -3.6)],
    }

    def is_reachable(target_node):
        # Perform a simple BFS/Flood-fill from current node (0,0) 
        # to see if target_node is reachable in the current graph
        import collections
        queue = collections.deque([(0.0, 0.0)])
        visited = {(0.0, 0.0)}
        while queue:
            curr = queue.popleft()
            if curr == target_node: return True
            for neighbor in valid_nodes.get(curr, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        return False

    def is_manipulable(model, geom_id):
        body_id = model.geom_bodyid[geom_id]
        jnt_adr = model.body_jntadr[body_id]
        jnt_num = model.body_jntnum[body_id]
        if jnt_adr < 0 or jnt_num <= 0:
            return False
        for j in range(jnt_adr, jnt_adr + jnt_num):
            if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                return True
        return False

    def get_nearest_frontier(world_map, pose):
        px, py, _ = pose
        grid_size = world_map.size
        visited = world_map.visited
        grid = world_map.grid
        dilated_grid = planner.dilate_map(grid)
        
        frontiers = []
        for r in range(grid_size):
            for c in range(grid_size):
                if visited[r, c] and grid[r, c] == 0.0 and dilated_grid[r, c] == 0.0:
                    has_unvisited = False
                    for dr, dc in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < grid_size and 0 <= nc < grid_size:
                            if not visited[nr, nc]:
                                has_unvisited = True
                                break
                    if has_unvisited:
                        wx = (r * world_map.res) - (grid_size * world_map.res) / 2 + world_map.res / 2
                        wy = (c * world_map.res) - (grid_size * world_map.res) / 2 + world_map.res / 2
                        frontiers.append((wx, wy))
                        
        if not frontiers:
            return None
            
        frontiers.sort(key=lambda f: (f[0] - px)**2 + (f[1] - py)**2)
        
        for f in frontiers:
            path = planner.plan(world_map.grid, pose, f)
            if path is not None and len(path) > 0:
                return f
                
        return None

    def respawn_target():
        food_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "food")
        bp = data.geom("chassis").xpos[:2].copy()
        
        # Filter for nodes reachable AND distance > 1.5m
        candidates = [n for n in valid_nodes.keys() 
                      if is_reachable(n) and 
                      math.hypot(n[0] - bp[0], n[1] - bp[1]) > 1.5]
        
        # Choose random candidate and apply jitter
        target = candidates[np.random.randint(len(candidates))]
        model.geom_pos[food_id][0] = target[0] + np.random.uniform(-0.15, 0.15)
        model.geom_pos[food_id][1] = target[1] + np.random.uniform(-0.15, 0.15)

    # Initialize reward tracking after function definitions
    _, _, prev_dist = get_food_metrics()

    # -- 100Hz Reticulospinal Tract (Brainstem Callback) --
    def on_control_tick(dt):
        nonlocal _cached_lidar_24, _lidar_tick_parity, arm_obs_history, prev_arm_action
        
        # --- PART 1: READ FROM MUJOCO DATA ---
        lock_ctx = viewer.lock() if viewer is not None else None
        if lock_ctx is not None:
            lock_ctx.__enter__()
        try:
            with mujoco_lock:
                # ── Decay Synaptic Lock and Cooldown unconditionally at 100Hz ──
                if not hasattr(arm_policy, 'synaptic_lock'):
                    arm_policy.synaptic_lock = 0.0
                if not hasattr(arm_policy, 'synaptic_lock_cooldown'):
                    arm_policy.synaptic_lock_cooldown = 0.0

                if arm_policy.synaptic_lock > 0.0:
                    next_lock = max(0.0, arm_policy.synaptic_lock - 0.01)
                    if next_lock == 0.0:
                        arm_policy.synaptic_lock_cooldown = 5.0
                    arm_policy.synaptic_lock = next_lock
                else:
                    arm_policy.synaptic_lock_cooldown = max(0.0, arm_policy.synaptic_lock_cooldown - 0.01)

                # Update LiDAR scan data at 100Hz
                _lidar_tick_parity = 1 - _lidar_tick_parity
                if _lidar_tick_parity == 0:
                    _cached_lidar_24 = np.array(get_lidar(n_rays=24))
                lidar_data = _cached_lidar_24.copy()
                sensor_buffer.write_lidar(lidar_data)

                # 1. Read high-level velocity targets from cognitive buffer
                planning_target, _ = sensor_buffer.planning_target.read()
                v_target, w_target = float(planning_target[0]), float(planning_target[1])

                # 2. Extract true wheel encoder angular velocities (rad/s)
                wL_vel = data.joint("w_left").qvel[0]
                wR_vel = data.joint("w_right").qvel[0]
                encoder_vel_2 = np.array([wL_vel, wR_vel], dtype=np.float64)

                # 3. Process keyboard and yaw rate
                human_active = check_keyboard()
                actual_yaw_rate = data.qvel[V_CARL+5] # freejoint Z-angular
                
                _, _, f_dist_tick = get_food_metrics()
                bypass_reflex = human_active or (f_dist_tick < 0.8)

                # 4. Check for physical collisions and damage
                chassis_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "chassis")
                floor_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
                collision_force = 0.0
                for idx in range(data.ncon):
                    con = data.contact[idx]
                    if con.geom1 == chassis_geom_id or con.geom2 == chassis_geom_id:
                        other_geom = con.geom2 if con.geom1 == chassis_geom_id else con.geom1
                        if other_geom != floor_geom_id:
                            if not is_manipulable(model, other_geom):
                                c_force = np.zeros(6)
                                mujoco.mj_contactForce(model, data, idx, c_force)
                                force_mag = abs(c_force[0])
                                if force_mag > 0.05:
                                    collision_force = max(collision_force, force_mag)

                # 5. Extract state variables for metabolism/allostasis
                vel_x_tick = float(data.qvel[V_CARL])
                vel_y_tick = float(data.qvel[V_CARL+1])
                yaw_tick = get_yaw()
                vel_fwd_tick = vel_x_tick * math.cos(yaw_tick) + vel_y_tick * math.sin(yaw_tick)
                joint_effort_tick = np.sum(np.abs(data.ctrl[2:16]))

                # 6. Extract arm-specific state if target is within reach
                has_curiosity_target = _curiosity_target[0] is not None
                curiosity_target_pos = _curiosity_target[0].copy() if has_curiosity_target else None
                carl_pos = data.geom('chassis').xpos.copy()
                
                arm_reach_active = False
                arm_state_29 = None
                reach_dist = 0.0
                if has_curiosity_target and arm_cache is not None:
                    dx, dy = curiosity_target_pos[0] - carl_pos[0], curiosity_target_pos[1] - carl_pos[1]
                    reach_dist = math.hypot(dx, dy)
                    if reach_dist < 0.45:
                        arm_reach_active = True
                        arm_pos = [data.qpos[a] for a in arm_cache['arm_qpos']]
                        arm_vel = [data.qvel[a] for a in arm_cache['arm_dof']]
                        tip = data.site_xpos[arm_cache['tip_L_sid']].copy()
                        rel = curiosity_target_pos - tip
                        res = np.zeros(6)
                        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_SITE, arm_cache['tip_L_sid'], res, 0)
                        tip_vel = res[3:6]
                        tl = data.sensordata[_touch_L_id] if _touch_L_id != -1 else 0.0
                        tr = data.sensordata[_touch_R_id] if _touch_R_id != -1 else 0.0
                        arm_state_29 = np.concatenate([arm_pos, arm_vel, rel, tip_vel, [tl, tr, brain.drives.M_t]])
                
                tl_reflex = data.sensordata[_touch_L_id] if _touch_L_id != -1 else 0.0
                tr_reflex = data.sensordata[_touch_R_id] if _touch_R_id != -1 else 0.0
        finally:
            if lock_ctx is not None:
                lock_ctx.__exit__(None, None, None)

        # --- PART 2: UNLOCKED COMPUTATIONS ---
        # Override target velocities to 0.0 if sleeping to let the robot rest
        is_sleeping = getattr(brain, 'is_sleeping', False)
        if is_sleeping:
            v_target, w_target = 0.0, 0.0

        torques, diagnostics = brainstem.compute_torques(
            lidar_24=lidar_data,
            encoder_vel_2=encoder_vel_2,
            target_linear=v_target,
            target_angular=w_target,
            actual_omega=actual_yaw_rate,
            bypass_reflex=bypass_reflex
        )

        if collision_force > 0.0:
            brain.drives.wall_event(collision_force)

        # interoception step
        proximity_8 = lidar_data[::3]
        prox = lidar_to_proximity(proximity_8)
        actual_danger = max(prox) if prox else 0.0
        
        brain.drives.interoceptive_step(
            actual_danger=actual_danger,
            torques=torques,
            is_sleeping=is_sleeping,
            velocity=vel_fwd_tick,
            joint_effort=joint_effort_tick
        )

        # update expressions
        raw_expr = expr_ctrl.step(brain.drives, dt=0.01, locomotion_v=v_target)
        
        # update smoothed expressions
        local_smoothed_expr = _smoothed_expr.copy()
        for key in local_smoothed_expr:
            if key in raw_expr:
                local_smoothed_expr[key] = (1 - _EXPR_ALPHA) * local_smoothed_expr[key] + _EXPR_ALPHA * raw_expr[key]

        # Arm Active Inference & NIP
        da_trigger = False
        lock_to_set = None
        local_arm_targets = None
        
        if arm_reach_active and brain.drives.M_t > 0.05 and arm_state_29 is not None:
            ltc_base_intention = arm_policy.forward(arm_state_29, dt=0.01)
            
            arm_obs_history = np.roll(arm_obs_history, -1, axis=0)
            arm_obs_history[-1] = arm_state_29
            
            local_arm_targets, G_val, certainty = arm_active_inference.select_action(
                predictor=arm_imagination.predictor,
                obs_history=arm_obs_history,
                prev_action=prev_arm_action,
                base_intention=ltc_base_intention
            )
            
            try:
                arm_inference_queue.put_nowait((arm_state_29.copy(), local_arm_targets.copy()))
            except queue.Full:
                pass
                
            prev_arm_action = local_arm_targets.copy()
            
            if brain.drives.cort > 0.5:
                volatility = (brain.drives.cort - 0.5) * 0.2
                local_arm_targets += np.random.normal(0, volatility, local_arm_targets.shape)

            cpg_exhale = math.sin(expr_ctrl.t * 2.1) < 0
            tl_arm = arm_state_29[-3]
            tr_arm = arm_state_29[-2]
            if arm_policy.synaptic_lock == 0.0 and arm_policy.synaptic_lock_cooldown == 0.0 and cpg_exhale and (tl_arm > 0.1 or tr_arm > 0.1):
                da_trigger = True
                lock_to_set = 5.0

            if arm_policy.synaptic_lock > 0 or lock_to_set is not None:
                if local_arm_targets is None:
                    local_arm_targets = np.zeros(10)
                local_arm_targets[1] = -1.0  # Shoulder pitch pull back
                local_arm_targets[2] = -1.0  # Elbow lift up
                local_arm_targets[4] = 0.52  # Grip Left target
                if len(local_arm_targets) > 9:
                    local_arm_targets[9] = 0.52

            if local_arm_targets is not None:
                scale = (ARM_CTRL_HIGH[:5] - ARM_CTRL_LOW[:5]) / 2.0
                center = (ARM_CTRL_HIGH[:5] + ARM_CTRL_LOW[:5]) / 2.0
                raw_left_arm = local_arm_targets[:5] * scale + center
                
                # Minimum-Jerk smoothing
                arm_smoother.update_target(raw_left_arm)
                smoothed_left_arm = arm_smoother.step()
                
                reach_blend = np.clip(1.0 - reach_dist / 0.45, 0, 1)
                local_smoothed_expr['shoulder_yaw_L'] = (1 - reach_blend) * local_smoothed_expr['shoulder_yaw_L'] + reach_blend * smoothed_left_arm[0]
                local_smoothed_expr['shoulder_L']     = (1 - reach_blend) * local_smoothed_expr['shoulder_L']     + reach_blend * smoothed_left_arm[1]
                local_smoothed_expr['elbow_L']        = (1 - reach_blend) * local_smoothed_expr['elbow_L']        + reach_blend * smoothed_left_arm[2]
                local_smoothed_expr['wrist_L']        = (1 - reach_blend) * local_smoothed_expr['wrist_L']        + reach_blend * smoothed_left_arm[3]
                local_smoothed_expr['grip_L']         = (1 - reach_blend) * local_smoothed_expr['grip_L']         + reach_blend * smoothed_left_arm[4]
        else:
            # Keep smoother updated with home/expression postures
            current_q_L = np.array([
                local_smoothed_expr['shoulder_yaw_L'],
                local_smoothed_expr['shoulder_L'],
                local_smoothed_expr['elbow_L'],
                local_smoothed_expr['wrist_L'],
                local_smoothed_expr['grip_L']
            ])
            arm_smoother.reset(current_q_L)

        cort_thresh = 0.05 if brain.drives.cort > 0.7 else 0.5
        if tl_reflex > cort_thresh:
            local_smoothed_expr['shoulder_yaw_L'] = 0.0
            local_smoothed_expr['shoulder_L'] = 0.5
            local_smoothed_expr['elbow_L'] = -1.8
            local_smoothed_expr['wrist_L'] = 1.0
            local_smoothed_expr['grip_L'] = 0.0
        if tr_reflex > cort_thresh:
            local_smoothed_expr['shoulder_yaw_R'] = 0.0
            local_smoothed_expr['shoulder_R'] = 0.5
            local_smoothed_expr['elbow_R'] = -1.8
            local_smoothed_expr['wrist_R'] = 1.0
            local_smoothed_expr['grip_R'] = 0.0

        # --- PART 3: WRITE BACK TO MUJOCO DATA & STEP ---
        lock_ctx = viewer.lock() if viewer is not None else None
        if lock_ctx is not None:
            lock_ctx.__enter__()
        try:
            with mujoco_lock:
                if da_trigger:
                    brain.drives.da = 1.0
                if lock_to_set is not None:
                    arm_policy.synaptic_lock = lock_to_set

                for key in local_smoothed_expr:
                    _smoothed_expr[key] = local_smoothed_expr[key]

                # Update real-time timing daemon diagnostics layer
                bios._state.energy = float(brain.drives.energy * 100.0)
                bios._state.fatigue = float(brain.drives.fatigue * 100.0)
                bios._state.stress = float(brain.drives.cort * 100.0)
                bios._state.damage = float(brain.drives.damage * 100.0)

                # Bounded Actuator Wear: reduce model.actuator_forcerange dynamically
                wear_multiplier = 1.0 - 0.3 * np.clip(brain.drives.damage, 0.0, 1.0)
                model.actuator_forcerange[:] = default_forcerange * wear_multiplier

                # Dispatch damped, anti-windup torques directly to hardware motors
                data.ctrl[0] = torques[0]
                data.ctrl[1] = torques[1]

                # Map expression targets → MuJoCo actuator indices
                data.ctrl[2]  = np.clip(_smoothed_expr['neck_height'] * 3.0, -0.55, 0.55)  # hip pitch (lean)
                data.ctrl[3]  = 0.0  # hip roll (handled by brainstem)
                data.ctrl[4]  = np.clip(_smoothed_expr['neck_height'], -0.6, 0.7)
                data.ctrl[5]  = np.clip(_smoothed_expr['head_yaw'], -1.4, 1.4)
                data.ctrl[6]  = np.clip(_smoothed_expr['shoulder_yaw_L'], -1.57, 1.57)
                data.ctrl[7]  = np.clip(_smoothed_expr['shoulder_L'], -2.0, 2.0)
                data.ctrl[8]  = np.clip(_smoothed_expr['elbow_L'],    -2.44, 0.0)
                data.ctrl[9]  = np.clip(_smoothed_expr['wrist_L'],    -1.57, 1.57)
                data.ctrl[10] = np.clip(_smoothed_expr['grip_L'],      0.0,  0.52)
                data.ctrl[11] = np.clip(_smoothed_expr['shoulder_yaw_R'], -1.57, 1.57)
                data.ctrl[12] = np.clip(_smoothed_expr['shoulder_R'], -2.0, 2.0)
                data.ctrl[13] = np.clip(_smoothed_expr['elbow_R'],    -2.44, 0.0)
                data.ctrl[14] = np.clip(_smoothed_expr['wrist_R'],    -1.57, 1.57)
                data.ctrl[15] = np.clip(_smoothed_expr['grip_R'],      0.0,  0.52)
                
                # Advance simulation physics state
                mujoco.mj_step(model, data)

                # Foraging contract validation
                _, _, f_dist = get_food_metrics()
                if f_dist < 0.38:
                    event_queue.put('food')
                    respawn_target()
                    mujoco.mj_forward(model, data)
        finally:
            if lock_ctx is not None:
                lock_ctx.__exit__(None, None, None)

    # -- Spin Infrastructure Ticks --
    world_map = OccupancyGrid()
    world_map.load()
    planner = AStarPlanner()
    follower = PurePursuitFollower()
    state_manager = ApexStateManager()
    goal_crystallizer = GoalCrystallizer(persistence_threshold=30)
    
    generation_idx = 0
    generation_steps = 0
    food_harvested_gen = 0
    elite_pool = []
    
    # Store initial state for resetting
    default_forcerange = model.actuator_forcerange.copy()
    default_qpos = data.qpos.copy()
    default_qvel = data.qvel.copy()
    
    food_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "food")
    default_food_pos = model.geom_pos[food_geom_id].copy()
    
    # Separate agent and environment joints for selective resetting
    env_joints = []
    agent_joints = []
    for j in range(model.njnt):
        j_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        if j_name and (j_name.startswith('fj_') or j_name == 'hazard_joint'):
            env_joints.append(j)
        else:
            agent_joints.append(j)
            
    goal_lifetimes = {}  # pos_tuple -> steps
    expired_lifetimes = []
    action_history = []

    # ── THE CRUCIBLE: Evolutionary Pressure Environment ──
    if CRUCIBLE_ENABLED:
        crucible = CrucibleController(
            circadian_period_s=300.0,
            biome_interval_s=60.0,
            food_rot_s=60.0,            # Was 45s — giving more time to forage
            food_spawn_cooldown_s=25.0,  # Was 30s — food respawns slightly faster
            maze_shift_interval_s=120.0,
            poltergeist_interval_range=(60.0, 180.0),
        )
        # Reduce maze density so CARL doesn't get boxed into dead-end traps
        crucible.maze_shifter.difficulty = 1
        _crucible_state = None
        _debris_geom_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f'debris_{i}') for i in range(12)]
        _debris_spawned = False
        print("[CRUCIBLE] *** THE CRUCIBLE IS ONLINE ***")
        print("[CRUCIBLE] Systems: Circadian | Biomes | Scarcity | Predator | Poltergeist | DVS | MPC | Maze | Object Permanence | Visual Servoing")
    else:
        crucible = None
        _crucible_state = None

    bios = CarlBiosCore()
    bios.register_control_callback(on_control_tick)
    social_lobe = SocialVisualLobe(shared_sensor_buffer=sensor_buffer, camera_index=0, frame_rate=10)
    social_lobe.start()

    bios.start_system()
    viewer = mujoco.viewer.launch_passive(model, data) if args.render else None

    recorder = HarvestRecorder()
    serializer = LeRobotSerializer("harvest_dataset", fps=50)
    episode_idx = 0
    recorder.start_episode(episode_idx)

    print("\n[SYSTEM] Tutorial Mode Active. CARL is fully autonomous by default.\n")

    last_v = 0.0
    smoothed_action = np.array([0.0, 0.0], dtype=np.float32)
    stuck_timer = 0
    prev_throttle = 0.0
    prev_steering = 0.0
    cached_path = None
    last_target_pos = None
    frontier_path = None
    face_expr = np.zeros(5, dtype=np.float32)
    exec_action = np.zeros(2, dtype=np.float32)

    try:
        while True:
            t_cycle = time.perf_counter()

            # --- REAL-TIME SENSOR FUSION ---
            # 1. Get real data from the buffers
            f_cos, f_sin, f_dist = get_food_metrics()
            lidar_data, _ = sensor_buffer.lidar.read() # Current LiDAR scan

            # ── THE CRUCIBLE: Advance all environmental systems ──
            if CRUCIBLE_ENABLED and crucible is not None:
                with mujoco_lock:
                    _carl_pos_cr = data.geom('chassis').xpos[:2].copy()
                    _carl_yaw_cr = get_yaw()

                # Prepare face data for visual servoing
                _face_data_cr = None
                if face_expr[0] > 0.5:
                    _face_data_cr = {
                        'detected': True,
                        'center_x': face_expr[4] if len(face_expr) > 4 else 320,
                        'frame_center_x': 320,
                    }

                # Spawn candidates from valid_nodes
                _spawn_cands = list(valid_nodes.keys())

                with mujoco_lock:
                    _crucible_state = crucible.step(
                        model, data, 0.02,  # dt = 50Hz
                        carl_pos=_carl_pos_cr,
                        carl_yaw=_carl_yaw_cr,
                        brain_drives=brain.drives,
                        spawn_candidates=_spawn_cands,
                        face_data=_face_data_cr,
                    )

                    # Handle debris spawning/despawning on biome change
                    if _crucible_state['weather']['changed']:
                        if _crucible_state['weather']['debris_active'] and not _debris_spawned:
                            # Scatter debris across the arena floor
                            for did in _debris_geom_ids:
                                if did != -1:
                                    model.geom_pos[did] = [
                                        random.uniform(-4.0, 4.0),
                                        random.uniform(-4.0, 4.0),
                                        0.05
                                    ]
                            _debris_spawned = True
                            print('[CRUCIBLE] Debris field deployed!')
                        elif not _crucible_state['weather']['debris_active'] and _debris_spawned:
                            # Hide debris off-screen
                            for did in _debris_geom_ids:
                                if did != -1:
                                    model.geom_pos[did] = [99, 99, 99]
                            _debris_spawned = False

                    # Update object permanence for all tracked objects
                    for oname in _obj_names:
                        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, oname)
                        if bid != -1:
                            opos = data.xpos[bid].copy()
                            # Check visibility via LiDAR occlusion
                            odist = math.hypot(opos[0] - _carl_pos_cr[0], opos[1] - _carl_pos_cr[1])
                            visible = odist < 3.0  # Simple proximity check
                            crucible.object_memory.update(oname, opos, visible=visible)

                # Log crucible state periodically
                if planner_step[0] % 250 == 0 and _crucible_state is not None:
                    cs = _crucible_state
                    print(f"[CRUCIBLE] Phase: {cs['circadian']['phase']} | "
                          f"Biome: {cs['weather']['biome']} | "
                          f"Predator: {cs['predator']['state']} ({cs['predator']['dist_to_carl']:.1f}m) | "
                          f"Food: {'ACTIVE' if cs['scarcity']['food_active'] else 'ROTTED'} | "
                          f"Drain: x{cs['energy_drain_multiplier']:.1f}")

                # ── HERO REWARD: Reward surviving a predator encounter ──
                if _crucible_state is not None:
                    _cur_pred_state = _crucible_state['predator']['state']
                    if not hasattr(crucible, '_prev_pred_state'):
                        crucible._prev_pred_state = _cur_pred_state
                    
                    # Predator gave up or entered cooldown after hunting = CARL survived!
                    if crucible._prev_pred_state in ('HUNT', 'LUNGE') and _cur_pred_state in ('PATROL', 'COOLDOWN'):
                        if _cur_pred_state == 'PATROL':
                            # Escaped! Massive reward
                            brain.drives.da = min(1.0, brain.drives.da + 0.8)
                            brain.drives.sero = min(1.0, brain.drives.sero + 0.5)
                            brain.drives.cort = max(0.0, brain.drives.cort - 0.3)
                            print("[HERO REWARD] *** ESCAPED THE PREDATOR! *** DA surge! Serotonin boost!")
                        elif _cur_pred_state == 'COOLDOWN':
                            # Survived attack (got hit but lived) — smaller reward
                            brain.drives.da = min(1.0, brain.drives.da + 0.3)
                            brain.drives.sero = min(1.0, brain.drives.sero + 0.2)
                            print("[HERO REWARD] Survived predator attack. Modest DA reward.")
                    
                    crucible._prev_pred_state = _cur_pred_state
            
            # 2. Build the REAL 34-D observation vector
            with mujoco_lock:
                yaw = get_yaw()
                bp = data.geom("chassis").xpos[:2].copy()
                pose = (bp[0], bp[1], yaw)
                actual_torques = data.actuator_force[:2].copy()
                target_food_pos = data.geom("food").xpos[:2].copy()
                vel_x = float(data.qvel[V_CARL])
                vel_y = float(data.qvel[V_CARL+1])
                omega = float(data.qvel[V_CARL+5]) # freejoint Z-angular
                vel_fwd  =  vel_x * math.cos(yaw) + vel_y * math.sin(yaw)
                vel_lat  = -vel_x * math.sin(yaw) + vel_y * math.cos(yaw)
            
            # Update goal crystallization persistence check
            lp_curiosity = brain.curiosity_engine.compute_learning_progress()
            da_val = brain.drives.da
            
            # Default utility is baseline exploration
            goal_utility = 0.5
            goal_type = 'curiosity'
            if da_val > 0.8:
                if f_dist < 1.5:
                    goal_utility = 2.5  # High-priority food reward landmark
                    goal_type = 'food'
                else:
                    goal_utility = 1.2  # Medium-priority toy discovery landmark
                    goal_type = 'toy'
            elif lp_curiosity > 0.02:
                goal_utility = 0.6  # Baseline curiosity exploration landmark
                goal_type = 'curiosity'

            if GOAL_CRYSTALLIZATION:
                goal_crystallizer.track_and_crystallize(pose[:2], lp_curiosity, da_val, utility=goal_utility, goal_type=goal_type)
                active_goal_pos = goal_crystallizer.step(pose[:2], brain.drives, boredom_accum=brain.drives.boredom_accumulation_rate, boredom_decay=brain.drives.boredom_decay_rate)
            else:
                active_goal_pos = None

            # Resolve active navigation target
            if GOAL_CRYSTALLIZATION and active_goal_pos is not None and brain.drives.energy >= 0.35 and f_dist > 1.5:
                active_target = active_goal_pos
            else:
                active_target = target_food_pos

            # Update occupancy grid map at 10Hz (every 5 steps)
            if planner_step[0] % 5 == 0:
                world_map.update(pose, lidar_data)
                if planner_step[0] % 250 == 0:
                    world_map.save()

                # ── OBJECT CURIOSITY SCAN ───────────────────────────────────
                carl_xpos = data.geom('chassis').xpos
                nearest_dist = 99.0
                nearest_pos  = None
                nearest_bid  = None
                with mujoco_lock:
                    for bid in _obj_body_ids:
                        opos = data.xpos[bid].copy()
                        d = math.hypot(opos[0] - carl_xpos[0], opos[1] - carl_xpos[1])
                        if d < nearest_dist:
                            nearest_dist = d
                            nearest_pos  = opos
                            nearest_bid  = bid

                _curiosity_target[0] = nearest_pos if nearest_dist < 0.50 else None

                if nearest_dist < 0.50 and brain.drives.hunger < 0.35:
                    brain.drives.novelty_event(1.0)
                    if planner_step[0] % 50 == 0:
                        print(f"[CURIOSITY] Object detected at {nearest_dist:.2f}m. M(t)={brain.drives.M_t:.2f} — reaching!")

                # Touch reward: if fingers contact something during curiosity reach
                if _touch_reward_cooldown[0] <= 0 and _curiosity_target[0] is not None:
                    with mujoco_lock:
                        tl = data.sensordata[_touch_L_id] if _touch_L_id != -1 else 0.0
                        tr = data.sensordata[_touch_R_id] if _touch_R_id != -1 else 0.0
                    if (tl > 0.01 or tr > 0.01):
                        brain.drives.da = min(1.0, brain.drives.da + 0.5)
                        if nearest_bid is not None:
                            _favorite_object_touches[nearest_bid] += 1
                            if _favorite_object_touches[nearest_bid] > 5:
                                _favorite_object_id[0] = nearest_bid
                        _touch_reward_cooldown[0] = 100  # 2s cooldown
                        print(f"[DISCOVERY] CARL touched an object! DA spike: {brain.drives.da:.2f}")
                else:
                    _touch_reward_cooldown[0] = max(0, _touch_reward_cooldown[0] - 1)
            
            # Headless run simulated face fallback to ensure reproducible culture ablation
            if args.no_keyboard:
                if CULTURE_IMITATION:
                    # Simulated user smiles periodically (e.g. 15% of the time)
                    t_social = planner_step[0] % 400
                    if t_social < 60:
                        face_expr = np.array([1.0, 0.7, 0.8, 0.4, 0.5], dtype=np.float32)
                    else:
                        face_expr = np.zeros(5, dtype=np.float32)
                else:
                    face_expr = np.zeros(5, dtype=np.float32)
            else:
                face_expr, _ = sensor_buffer.face_expression.read()
            ego_vec = np.array([brainstem.psi_L, brainstem.psi_R, brainstem.psi_S, 0.0], dtype=np.float32)
            surprise = 0.0
            
            obs_raw = obs_builder.build(
                lidar_data=lidar_data,
                vel_fwd=vel_fwd,
                vel_lat=vel_lat,
                omega=omega,
                f_cos=f_cos,
                f_sin=f_sin,
                drives=brain.drives,
                prev_throttle=prev_throttle,
                prev_steering=prev_steering,
                face_expr=face_expr,
                ego_vec=ego_vec,
                surprise=surprise
            )

            # 2. Compute Reward (Extrinsic)
            reward_dist = prev_dist - f_dist
            r_extrinsic = (reward_dist * 10.0) - 0.001
            if data.geom("chassis").xpos[2] < 0.02: r_extrinsic -= 1.0 # Fall penalty
            prev_dist = f_dist

            # ══════════════════════════════════════════════════════════════
            #  MARK IX: SUBCONSCIOUSNESS PROCESSING (runs every tick)
            # ══════════════════════════════════════════════════════════════
            brain.subconsciousness_tick(obs_raw, f_dist, vel_fwd, GOAL_CRYSTALLIZATION, active_goal_pos, exec_action, r_extrinsic)
            
            # ── Occlusion-Curiosity (Object Permanence) ──
            angle_to_food = math.atan2(f_sin, f_cos)
            if angle_to_food < 0: angle_to_food += 2 * math.pi
            ray_idx = int(round((angle_to_food) / (2 * math.pi) * 24)) % 24
            ray_dist = 2.5 * (1.0 - lidar_data[ray_idx])
            
            is_occluded = (ray_dist < f_dist - 0.1) and (f_dist < 2.5)
            if not is_occluded and _occlusion_state == "HIDDEN":
                brain.drives.da = 1.0
                print("[COGNITION] PEEK-A-BOO! Object Permanence recognized. Massive Dopamine!")
                _occlusion_state = "VISIBLE"
            elif is_occluded:
                _occlusion_state = "HIDDEN"
            else:
                _occlusion_state = "VISIBLE"
                
            total_reward = r_extrinsic

            # 3. Assess Trajectory Availability
            replan_interval = 10 if state_manager.state == "SPEED_DEMON" else 50
            if (cached_path is None or 
                    last_target_pos is None or 
                    not np.array_equal(active_target, last_target_pos) or 
                    planner_step[0] % replan_interval == 0):
                cached_path = planner.plan(world_map.grid, pose, active_target)
                path_valid = cached_path is not None
                last_target_pos = active_target.copy()
                
                if state_manager.state == "EXPLORER" and (not GOAL_CRYSTALLIZATION or active_goal_pos is None):
                    frontier_target = get_nearest_frontier(world_map, pose)
                    if frontier_target:
                        frontier_path = planner.plan(world_map.grid, pose, frontier_target)
                    else:
                        frontier_path = None
                else:
                    frontier_path = None
            else:
                path_valid = cached_path is not None
            path = cached_path

            # 4. Update Metabolism
            energy_level = brain.drives.energy * 100.0

            # Check if safety reflexes are active
            front_min = np.min(np.concatenate([lidar_data[21:24], lidar_data[0:4]]))
            reflex_fired = (front_min < 0.20 or np.min(lidar_data) < 0.08)

            # 5. State Machine Evaluation
            state_manager.state = state_manager.evaluate_state(
                current_state=state_manager.state,
                map_confidence=world_map.confidence,
                energy=energy_level,
                reflex_fired=reflex_fired,
                path_valid=path_valid
            )

            # 6. Action Computation & Handover
            human_active = check_keyboard()
            if human_active:
                v_exec = 0.4 if keyboard.is_pressed('w') or keyboard.is_pressed('up') else (-0.4 if keyboard.is_pressed('s') or keyboard.is_pressed('down') else 0.0)
                w_exec = 0.8 if keyboard.is_pressed('a') or keyboard.is_pressed('left') else (-0.8 if keyboard.is_pressed('d') or keyboard.is_pressed('right') else 0.0)
                exec_action = np.array([v_exec, w_exec], dtype=np.float32)
                smoothed_action = exec_action.copy()
            else:
                if state_manager.state == "SPEED_DEMON" and not (brain.drives.energy < 0.20):
                    v_cmd, w_cmd = follower.get_control(pose, path)
                    exec_action = np.array([v_cmd, w_cmd], dtype=np.float32)
                    smoothed_action = exec_action.copy()
                else:
                    # Biological Intuition Layer
                    if CULTURE_IMITATION and face_expr[0] > 0.5:
                        smile_ratio = face_expr[1]
                        brain.drives.da = min(1.0, brain.drives.da + 0.1 * smile_ratio)
                        brain.drives.sero = min(1.0, brain.drives.sero + 0.05 * smile_ratio)

                    # Mark IX: Habit Engine — check if a learned motor program can fire
                    habitual_action = brain.habits.get_habitual_action(obs_raw, brain.drives)
                    if habitual_action is not None:
                        # Habit fires! Skip MPC planner (saves cognitive bandwidth)
                        exec_action = np.clip(habitual_action, [-0.4, -1.0], [0.4, 1.0])
                        smoothed_action = 0.9 * smoothed_action + 0.1 * exec_action
                        exec_action = smoothed_action.copy()
                        # Still run brain.step for learning, but don't use its action
                        brain.step(obs_raw, r_extrinsic)
                    else:
                        raw_action, _, familiarity = brain.step(obs_raw, r_extrinsic)
                    
                        smoothed_action = 0.85 * smoothed_action + 0.15 * raw_action
                        exec_action = smoothed_action.copy()
                    
                    if (not GOAL_CRYSTALLIZATION or active_goal_pos is None) and frontier_path is not None and len(frontier_path) > 0 and f_dist > 1.5:
                        v_front, w_front = follower.get_control(pose, frontier_path)
                        v_front = np.clip(v_front, -0.30, 0.60)
                        w_front = np.clip(w_front, -1.5, 1.5)
                        exec_action[0] = v_front
                        exec_action[1] = w_front
                    
                    # Determine whether we should follow the A* path to active_target (food or goal)
                    follow_primary_path = False
                    if GOAL_CRYSTALLIZATION and active_goal_pos is not None and brain.drives.energy >= 0.35 and f_dist > 1.5:
                        follow_primary_path = True
                    elif brain.drives.energy < 0.35 or f_dist <= 1.5:
                        follow_primary_path = True
                    elif (not GOAL_CRYSTALLIZATION or active_goal_pos is None) and (frontier_path is None or len(frontier_path) == 0):
                        # Explorer mode with no valid frontier path, fallback to food path
                        follow_primary_path = True

                    if follow_primary_path:
                        v_goal, w_goal = follower.get_control(pose, path)
                        exec_action[0] = np.clip(v_goal, -0.30, 0.60)
                        exec_action[1] = np.clip(w_goal, -1.5, 1.5)
                    
                    # ── BEHAVIORAL REPERTOIRE OVERRIDES ──
                    # Retrieve favorite object position if attachment behavior is relevant
                    fav_pos = None
                    fav_dist = 0.0
                    if _favorite_object_id[0] is not None:
                        with mujoco_lock:
                            fav_pos = data.xpos[_favorite_object_id[0]].copy()
                        fav_dist = math.hypot(fav_pos[0] - pose[0], fav_pos[1] - pose[1])

                    # Build ActionContext and run BehavioralArbitrator
                    ctx = ActionContext(
                        pose=pose,
                        velocity_fwd=vel_fwd,
                        drives=brain.drives,
                        face_expr=face_expr,
                        ego_vec=ego_vec,
                        planner_step=planner_step[0],
                        f_dist=f_dist,
                        path=path,
                        frontier_path=frontier_path,
                        follower=follower,
                        crucible=crucible,
                        crucible_state=_crucible_state,
                        carl_pos_cr=_carl_pos_cr if CRUCIBLE_ENABLED else None,
                        carl_yaw_cr=_carl_yaw_cr if CRUCIBLE_ENABLED else None,
                        favorite_object_id=_favorite_object_id[0],
                        obj_names=_obj_names,
                        obj_body_ids=_obj_body_ids,
                        nearest_pos=nearest_pos,
                        nearest_dist=nearest_dist,
                        nearest_bid=nearest_bid,
                        fav_pos=fav_pos,
                        fav_dist=fav_dist,
                        follow_primary_path=follow_primary_path,
                        habit_fired=(habitual_action is not None)
                    )
                    exec_action, fired_behavior = arbitrator.select_action(exec_action, ctx)
                        
            # ── SENSORY-MOTOR OVERRIDE (Lifting) ──
            if not human_active and hasattr(arm_policy, 'synaptic_lock') and arm_policy.synaptic_lock > 0 and brain.drives.energy >= 0.35:
                exec_action[0] = 0.0
                exec_action[1] = 0.0
                
                if planner_step[0] % 50 == 0:
                    print(f"[DEBUG] state: {state_manager.state} | Energy: {energy_level:.1f}% | CORT: {brain.drives.cort:.2f} | ACh: {brain.drives.ach:.2f} | DA: {brain.drives.da:.2f} | M(t): {brain.drives.M_t:.2f}")

            # S-Curve Acceleration (Smoothing) to prevent tipping
            target_v = exec_action[0]
            last_v = 0.95 * last_v + 0.05 * target_v 
            exec_action[0] = last_v
            
            # Inject CPG Sway
            cpg_intensity = brain.drives.da * 0.15 
            cpg_mod = cpg.compute_biomimetic_modulation(brain.drives, [last_v, last_v])
            raw_sway = cpg_mod[0] - last_v
            speed_ratio = 0.3 + 0.7 * np.clip(abs(last_v) / 0.4, 0.0, 1.0)
            cpg_addition = raw_sway * cpg_intensity * 180.0 * speed_ratio
            exec_action[1] += np.clip(cpg_addition, -0.3, 0.3)

            # Escapement State Machine
            escapement_active = False
            if not human_active and state_manager.state == "EXPLORER":
                if np.min(lidar_data) < 0.08:
                    stuck_timer += 1
                else:
                    stuck_timer = 0

                if stuck_timer > 50:
                    escapement_active = True
                    exec_action = np.array([-0.3, 0.8], dtype=np.float32)
                    if planner_step[0] % 10 == 0:
                        print("[CORTEX] Escapement Protocol Active: Reversing and Re-orienting...")
            else:
                stuck_timer = 0
            
            # Dispatch Commands
            if planner_step[0] % 25 == 0:
                audio_synth.play_mood_sound(brain.drives, is_stationary=(abs(last_v) < 0.05))

            if planner_step[0] % 10 == 0:
                print(f"[DEBUG-TICK] v_target: {exec_action[0]:.5f}, w_target: {exec_action[1]:.5f}, stuck: {stuck_timer}, f_dist: {f_dist:.3f}")
            sensor_buffer.planning_target.write(exec_action)
            if not escapement_active:
                recorder.record_step(obs_raw, exec_action, planner_step[0] * 0.02)

            prev_throttle = float(exec_action[0])
            prev_steering = float(exec_action[1])
            
            # Record action history for Shannon entropy
            action_history.append((prev_throttle, prev_steering))
            if len(action_history) > 100:
                action_history.pop(0)

            # Mark IX: Record step for habit engine (learns action chunks)
            brain.habits.record_step(obs_raw, exec_action, r_extrinsic)
            
            # Mark IX: Periodic subconsciousness status
            if planner_step[0] % 500 == 0:
                ws_mode = brain.workspace.get_behavioral_mode()
                n_markers, avg_val, avg_str = brain.somatic.get_stats()
                n_chunks, avg_rel, reliable = brain.habits.get_stats()
                dmn_stats = brain.dmn.get_stats()
                print(f"[SUBCONSCIOUS] Mode: {ws_mode} | Markers: {n_markers} (val:{avg_val:.2f}) | Habits: {n_chunks} ({reliable} reliable) | DMN: act={dmn_stats['activation_level']:.2f} replay={dmn_stats['rumination_count']} | Allostatic: urg={brain.allostatic.urgency:.2f} | Circadian: {'REST' if brain.circadian.is_rest_phase() else 'ACTIVE'}")
            
            # Update goal lifetimes
            if GOAL_CRYSTALLIZATION:
                current_goal_positions = {tuple(g.target_pos) for g in goal_crystallizer.goals}
                for pos in current_goal_positions:
                    goal_lifetimes[pos] = goal_lifetimes.get(pos, 0) + 1
                for pos in list(goal_lifetimes.keys()):
                    if pos not in current_goal_positions:
                        expired_lifetimes.append(goal_lifetimes.pop(pos))
            
            # 5. Narrative & Tracking
            narrator.update(planner_step[0], f_dist, human_active, 0.0, brain.drives)
            if planner_step[0] % 100 == 0:
                wheel_eff = abs(data.ctrl[0]) + abs(data.ctrl[1])
                joint_eff = np.sum(np.abs(data.ctrl[2:14]))
                print(f"[CORTEX] Emergence Status | State: {state_manager.state} | Energy: {energy_level:.1f}% | Dist: {f_dist:.2f} | Map Confidence: {world_map.confidence * 100.0:.2f}% | WheelEff: {wheel_eff:.2f} | JointEff: {joint_eff:.2f} | Vel: {vel_fwd:.2f}")
                
                # Compute and append emergence metrics using modular logger
                with mujoco_lock:
                    w_eff = float(abs(data.actuator_force[0]) + abs(data.actuator_force[1]))
                    j_eff = float(np.sum(np.abs(data.ctrl[2:14])))
                    vel = float(math.hypot(data.qvel[V_CARL], data.qvel[V_CARL+1]))
                metrics_logger.log_tick(
                    step=planner_step[0],
                    brain=brain,
                    action_history=action_history,
                    world_map=world_map,
                    goal_lifetimes=goal_lifetimes,
                    expired_lifetimes=expired_lifetimes,
                    wheel_effort=w_eff,
                    joint_effort=j_eff,
                    velocity=vel,
                    state=state_manager.state
                )

            # Resolve target discovery boundaries
            if not event_queue.empty():
                if event_queue.get() == 'food':
                    brain.food_eaten()  # Restore metabolic energy!
                    food_harvested_gen += 1
                    # Notify Crucible scarcity controller
                    if CRUCIBLE_ENABLED and crucible is not None:
                        crucible.scarcity.on_food_eaten()
                    ep = recorder.finalize_episode()
                    serializer.write_episode_fallback(ep, episode_idx)
                    narrator.log_success(episode_idx, len(ep['observations']))
                    episode_idx += 1
                    recorder.start_episode(episode_idx)

            # Check for death and breed loop
            is_dead = evolution.check_death(brain.drives)
            if is_dead:
                print(f"[DEATH] CARL died at step {planner_step[0]} of generation {generation_idx}. Energy: {brain.drives.energy:.3f}, Damage: {brain.drives.damage:.3f}")
                
                habit_index = evolution.calculate_habit_index(action_history)
                exploration_diversity = world_map.confidence * 100.0
                goal_count = len(goal_crystallizer.goals) if GOAL_CRYSTALLIZATION else 0
                score = 100.0 * math.log(1.0 + generation_steps) + 1000.0 * food_harvested_gen - 5.0 * brain.drives.damage
                
                evolution.log_fossil_record(
                    generation_idx=generation_idx,
                    lifespan=generation_steps,
                    food_harvested=food_harvested_gen,
                    damage=brain.drives.damage,
                    score=score,
                    goal_count=goal_count,
                    habit_index=habit_index,
                    exploration_diversity=exploration_diversity,
                    procedural_inheritance=PROCEDURAL_INHERITANCE,
                    ecological_persistence=ECOLOGICAL_PERSISTENCE,
                    culture_imitation=CULTURE_IMITATION,
                    goal_crystallization=GOAL_CRYSTALLIZATION
                )
                
                # Update Elite Pool and breed
                evolution.update_elite_pool(score, brain, goal_crystallizer=goal_crystallizer)
                
                evolution.breed_and_reset(
                    brain=brain,
                    model=model,
                    data=data,
                    world_map=world_map,
                    goal_crystallizer=goal_crystallizer,
                    goal_lifetimes=goal_lifetimes,
                    expired_lifetimes=expired_lifetimes,
                    action_history=action_history,
                    event_queue=event_queue,
                    mujoco_lock=mujoco_lock,
                    default_forcerange=default_forcerange,
                    default_qpos=default_qpos,
                    default_qvel=default_qvel,
                    default_food_pos=default_food_pos,
                    food_geom_id=food_geom_id,
                    carl_joint_id=_carl_jnt_id,
                    agent_joints=agent_joints,
                    respawn_target_fn=respawn_target,
                    procedural_inheritance=PROCEDURAL_INHERITANCE,
                    ecological_persistence=ECOLOGICAL_PERSISTENCE,
                    goal_crystallization_on=GOAL_CRYSTALLIZATION
                )
                arm_policy.reset()
                
                generation_idx += 1
                if generation_idx >= args.max_generations:
                    print(f"[EXPERIMENT] Reached max generations ({args.max_generations}). Exiting simulation loop.")
                    break
                generation_steps = 0
                food_harvested_gen = 0
                
                ep = recorder.finalize_episode()
                serializer.write_episode_fallback(ep, episode_idx)
                episode_idx += 1
                recorder.start_episode(episode_idx)
                
                print(f"[BREEDING] Spawned mutated generation {generation_idx} offspring.")
                continue

            generation_steps += 1
            planner_step[0] += 1
            # Pace the main planning loop to 50Hz in all modes
            time.sleep(max(0, 0.02 - (time.perf_counter() - t_cycle)))
            if viewer is not None:
                with viewer.lock():
                    viewer.sync()

    except KeyboardInterrupt:
        print("\n[SYSTEM] Tutorial terminated safely by user.")
    finally:
        bios.shutdown_system()
        social_lobe.stop()
        if viewer: viewer.close()
        serializer.write_meta(episode_idx)
        print("\n[INFO] Tutorial dataset serialization complete.")

if __name__ == "__main__":
    main()
