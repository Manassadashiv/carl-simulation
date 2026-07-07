"""
carl_train_multirate.py — CARL v4 Mark VIII Multi-Rate Asynchronous Orchestrator.

Architecture:
- Thread 1 (BIOS Control tick @ 100Hz):
  Biological analogue: Spinal Cord + Brainstem.
  - Locks mujoco_lock.
  - Reads sensors from MuJoCo (24-ray LiDAR, wheel speeds, IMU, drives).
  - Writes to SensorStateBuffer (double buffers) for System 2.
  - Reads planning velocity targets from planning_target double buffer.
  - Computes torques via Brainstem Controller (PID + Braitenberg).
  - Modulates torques via Hopf CPG motor oscillator (tremors and sways).
  - Translates torques to MuJoCo velocity targets: ctrl = torque / kv + joint_vel
  - Steps physics (mj_step) and checks boundaries.
  - Detects events (food/wall), triggers drive resets, and respawns.
  - Dispatches events to the planner thread via a thread-safe queue.
  - Streams real-time WebSocket telemetry.
  - Plays procedural audio loops asynchronously every 350ms (winsound).

- Thread 2 (Cognitive Planner @ 50Hz):
  Biological analogue: Cerebral Cortex.
  - Reads sensory state lock-free from the double buffers.
  - Downsamples 24-ray LiDAR scan to 8-ray LiDAR to save computation.
  - Computes path waypoints to navigate around obstacles.
  - Invokes Actor-Critic policy gradient training step (CarlBrain.step()).
  - Updates policy networks asynchronously (System 2 learning).
  - Offloads expensive weights-saving (np.savez) offline to protect the 100Hz clock.
  - Triggers off-line dream state experience replay when fatigue > 0.90 in the nest.

Timing & Synchronization:
- Coordinated via carl_bios.py hybrid sleep+busy-wait scheduler (achieving 99.98Hz on Windows).
- Decoupled via lock-free pointer-swap AtomicDoubleBuffer.
- Synchronized via mujoco_lock for brief, atomic physics modification/step sections.
"""

import math
import time
import json
import os
import sys
import numpy as np
import mujoco
import threading
import queue
import random

sys.path.append(os.path.abspath('.'))
from carl_agent import CarlBrain
from carl_expression import ExpressionController, ProceduralAudioSynthesizer
from carl_bios import CarlBiosCore
from carl_double_buffer import SensorStateBuffer, AtomicDoubleBuffer
from carl_brainstem import BrainstemController
from carl_cpg import HopfCpgEngine
from carl_social import SocialVisualLobe
import blob_telemetry

# ── Thread safety & communication ──────────────────────────────────────────────
mujoco_lock = threading.Lock()
event_queue = queue.Queue()  # thread-safe event delivery (100Hz -> 50Hz)

# ── Physics world ──────────────────────────────────────────────────────────────
# Load the MuJoCo model for Character Kinematics
model = mujoco.MjModel.from_xml_path("vessel_kinetic.xml")
data  = mujoco.MjData(model)

# Resolve all actuator IDs to handle model upgrades gracefully
ACT_IDS = {}
for name in ["act_w_left", "act_w_right", "act_hip_pitch", "act_hip_roll", "act_neck", "act_head",
             "act_shoulder_yaw_L", "act_shoulder_L", "act_elbow_L", "act_wrist_L", "act_grip_L",
             "act_shoulder_yaw_R", "act_shoulder_R", "act_elbow_R", "act_wrist_R", "act_grip_R"]:
    act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    if act_id != -1:
        ACT_IDS[name] = act_id

with mujoco_lock:
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

BLOB_GEOM_ID = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "chassis")
FOOD_GEOM_ID = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "food")

init_pos = data.geom("chassis").xpos.copy()
print(f"[INIT] Bob's true spawn: ({init_pos[0]:.3f}, {init_pos[1]:.3f})")

# ── Brainstem & CPG Infrastructure ──────────────────────────────────────────────
N_OBS = 34
brain = CarlBrain(n_obs=N_OBS)
brain.load("memory/carl_brain")

# Shared double buffers
sensor_buffer = SensorStateBuffer(obs_dim=33)
brain.sensor_buffer = sensor_buffer
pose_buffer = AtomicDoubleBuffer(shape=(8,))  # [vel_fwd, vel_lat, omega, food_cos, food_sin, dist, carl_x, carl_y]

# Controllers
brainstem = BrainstemController()
cpg = HopfCpgEngine()
audio = ProceduralAudioSynthesizer()

# ── Stats & Curriculum ──────────────────────────────────────────────────────────
STATS_FILE = "memory/carl_stats.json"
def load_stats():
    if os.path.exists(STATS_FILE):
        try:
            with open(STATS_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {"episodes": [], "total_food": 0, "total_wall": 0,
            "curriculum_dist": 1.5, "best_avg": 9999}

def save_stats(s):
    try:
        with open(STATS_FILE, "w") as f:
            json.dump(s, f, indent=2)
    except Exception:
        pass

stats = load_stats()
MAX_FOOD_DIST = stats.get("curriculum_dist", 1.5)
MIN_FOOD_DIST = 1.0
MAX_EP_STEPS = 8000
NEST_CORNER_X = -3.5
NEST_CORNER_Y = -3.5

# Global runtime variables
prev_dist = 2.0
steps_since_food = 0
episode = len(stats.get("episodes", [])) + 1
total_food = stats.get("total_food", 0)
total_wall = stats.get("total_wall", 0)
collision_cooldown = 0
steps_history = []
experience_buffer = []
global_is_sleeping = False
planner_step = 0
audio_timer = 0.0
latest_hdc_familiarity = 0.0
telemetry_tick_counter = 0  # Throttle WS broadcast to every 3rd tick (~33Hz)

# State Estimator globals for Inertial Velocity Observer (Phase C2-Extension)
est_x = 0.0
est_y = 0.0
est_yaw = 0.0
imu_vel_fwd = 0.0
prev_vel_fwd = 0.0

# LiDAR scan cache: only raycast every other 100Hz tick (50Hz effective).
# Initialized with safe defaults (5m = no obstacles).
_cached_lidar_24 = np.full(24, 5.0, dtype=np.float64)
_lidar_tick_parity = 0

# ── MuJoCo physics helper operations (must run under lock) ──────────────────────

def get_yaw(quat):
    """Convert quaternion [w, x, y, z] to yaw angle (rotation around Z)."""
    w, x, y, z = quat
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

def set_yaw(yaw):
    """Convert yaw angle to quaternion [w, x, y, z]."""
    return [math.cos(yaw/2.0), 0.0, 0.0, math.sin(yaw/2.0)]

def get_lidar(n_rays=8):
    """Cast rays, excluding chassis geometry."""
    food_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "food")
    old_food_pos = model.geom_pos[food_id].copy()
    model.geom_pos[food_id] = np.array([99.0, 99.0, 99.0])
    
    blob_pos = data.geom("chassis").xpos
    yaw = get_yaw(data.qpos[3:7])
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

def lidar_to_proximity(dists, max_range=2.5):
    return [max(0.0, (max_range - d) / max_range) for d in dists]

def get_food_direction():
    blob_pos = data.geom("chassis").xpos
    food_pos = data.geom("food").xpos
    yaw = get_yaw(data.qpos[3:7])
    dx = food_pos[0] - blob_pos[0]
    dy = food_pos[1] - blob_pos[1]
    world_angle = math.atan2(dy, dx)
    rel = world_angle - yaw
    while rel >  math.pi: rel -= 2*math.pi
    while rel < -math.pi: rel += 2*math.pi
    dist = math.hypot(dx, dy)
    return math.cos(rel), math.sin(rel), dist

def food_dist():
    blob_pos = data.geom("chassis").xpos
    food_pos = data.geom("food").xpos
    return math.hypot(blob_pos[0]-food_pos[0], blob_pos[1]-food_pos[1])

def check_event():
    blob_pos = data.geom("chassis").xpos
    if food_dist() < 0.35:
        return 'food'
    abs_x = data.qpos[0]
    abs_y = data.qpos[1]
    if abs(abs_x) > 4.65 or abs(abs_y) > 4.65:
        return 'wall'
    if math.hypot(blob_pos[0], blob_pos[1]) < 0.72:
        return 'wall'
    return None

def respawn_food(max_dist=None):
    food_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "food")
    blob_pos = data.geom("chassis").xpos
    d = max_dist or MAX_FOOD_DIST
    min_d = max(MIN_FOOD_DIST, 1.0)
    if d <= min_d:
        d = min_d + 0.5
    for _ in range(200):
        angle = np.random.uniform(0, 2*math.pi)
        radius = np.random.uniform(min_d, d)
        fx = blob_pos[0] + radius * math.cos(angle)
        fy = blob_pos[1] + radius * math.sin(angle)
        fx = np.clip(fx, -4.2, 4.2)
        fy = np.clip(fy, -4.2, 4.2)
        actual_dist = math.hypot(fx - blob_pos[0], fy - blob_pos[1])
        if math.hypot(fx, fy) > 0.9 and actual_dist >= min_d:
            model.geom_pos[food_id][0] = fx
            model.geom_pos[food_id][1] = fy
            return actual_dist
    model.geom_pos[food_id][0] = np.random.uniform(1.5, 3.0)
    model.geom_pos[food_id][1] = np.random.uniform(1.5, 3.0)
    return 2.0

def respawn_bob():
    global est_x, est_y, est_yaw, imu_vel_fwd, prev_vel_fwd
    corners = [(-2.5, -2.5), (2.5, -2.5), (-2.5, 2.5), (2.5, 2.5)]
    cx, cy = corners[np.random.randint(len(corners))]
    cx += np.random.uniform(-0.5, 0.5)
    cy += np.random.uniform(-0.5, 0.5)
    data.qpos[0] = cx
    data.qpos[1] = cy
    data.qpos[2] = 0.05
    rand_yaw = np.random.uniform(-math.pi, math.pi)
    data.qpos[3:7] = set_yaw(rand_yaw)
    data.qvel[:] = 0.0

    # Reset state estimator coordinates
    est_x = cx
    est_y = cy
    est_yaw = rand_yaw
    imu_vel_fwd = 0.0
    prev_vel_fwd = 0.0
    if 'brain' in globals() and hasattr(brain, 'actor'):
        brain.actor.reset_traces()

def clamp_bob():
    abs_x = data.qpos[0]
    abs_y = data.qpos[1]
    cx = np.clip(abs_x, -4.65, 4.65)
    cy = np.clip(abs_y, -4.65, 4.65)
    hit = (abs_x != cx) or (abs_y != cy)
    if hit:
        data.qpos[0] = cx
        data.qpos[1] = cy
        if abs(abs_x) > 4.65:
            data.qvel[0] = -data.qvel[0] * 0.2
        if abs(abs_y) > 4.65:
            data.qvel[1] = -data.qvel[1] * 0.2
        mujoco.mj_forward(model, data)
    return hit

def build_observation(proximity, food_cos, food_sin, vel_fwd, vel_lat, omega,
                       touch_front, touch_back, touch_left, touch_right,
                       prev_throttle, prev_steering, drives, face_expr, ego_vec, surprise):
    return np.array(
        list(proximity) +
        [vel_fwd, vel_lat, omega] +
        [food_cos, food_sin] +
        list(drives.to_vec())[:5] +
        [touch_front, touch_back, touch_left, touch_right] +
        [prev_throttle, prev_steering] +
        list(face_expr) +
        list(ego_vec) +
        [surprise],
        dtype=np.float32
    )

def maybe_advance_curriculum(steps_history):
    global MAX_FOOD_DIST
    if len(steps_history) < 10:
        return False
    recent_avg = np.mean(steps_history[-10:])
    if recent_avg < 800 and MAX_FOOD_DIST < 8.0:
        MAX_FOOD_DIST = min(8.0, MAX_FOOD_DIST + 0.5)
        print(f"[CURRICULUM] Advancing! Food can now spawn up to {MAX_FOOD_DIST:.1f}m away. Avg was {recent_avg:.0f} steps.")
        return True
    return False

# ── THREAD 1: 100Hz Control / Reflex Loop ───────────────────────────────────────

def on_control_tick(dt_seconds):
    """
    Spinal/Reflex Tick. Computes Braitenberg overrides, PIDs, CPG tremors,
    updates sensory double buffers, steps physics, and streams telemetry.
    """
    global prev_dist, total_food, total_wall, collision_cooldown, steps_since_food, episode, audio_timer, telemetry_tick_counter, planner_step
    global est_x, est_y, est_yaw, imu_vel_fwd, prev_vel_fwd
    global _cached_lidar_24, _lidar_tick_parity
    
    # Check if emergency shutdown is requested by watchdog
    state = bios.get_state()
    if state.emergency_stop:
        return

    with mujoco_lock:
        # 1. Perception: Read raw sensory variables
        yaw = get_yaw(data.qpos[3:7])
        vel_x = float(data.qvel[0])
        vel_y = float(data.qvel[1])
        omega = float(data.qvel[2])
        vel_fwd  =  vel_x * math.cos(yaw) + vel_y * math.sin(yaw)
        vel_lat  = -vel_x * math.sin(yaw) + vel_y * math.cos(yaw)
        
        # Raycast 24 rays every OTHER tick (50Hz effective LiDAR refresh).
        # 24x mj_ray is the single most expensive operation (~5-10ms).
        # Caching halves the hot-path cost while maintaining continuous
        # Braitenberg reflex coverage using the cached scan.
        _lidar_tick_parity = 1 - _lidar_tick_parity
        if _lidar_tick_parity == 0:
            _cached_lidar_24 = np.array(get_lidar(n_rays=24))
        lidar_24 = _cached_lidar_24
        
        # Proprioceptive wheel encoders
        encoder_vel = np.array([data.joint("w_left").qvel[0], data.joint("w_right").qvel[0]])
        
        # 2. Decision: Read latest planning targets from the double buffer
        planning_target, _ = sensor_buffer.planning_target.read()
        target_linear, target_angular = planning_target[0], planning_target[1]
        
        # 3. Spinal Cord: Compute torques via Brainstem Controller (PID + Braitenberg)
        # Passes raw gyro yaw rate (omega) for traction monitoring
        torques, diag = brainstem.compute_torques(lidar_24, encoder_vel, target_linear, target_angular, actual_omega=omega)
        
        # 4. Inertial Velocity Observer (Phase C2-Extension State Estimation)
        psi_S = diag['psi'][2]
        v_encoder = 0.5 * (encoder_vel[0] + encoder_vel[1]) * 0.04  # R = 0.04m
        
        # Simulated raw IMU longitudinal accelerometer reading (with noise)
        true_accel = (vel_fwd - prev_vel_fwd) / 0.01  # dt = 0.01s (100Hz)
        imu_accel = true_accel + np.random.normal(0.0, 0.02)
        
        # Integrate IMU acceleration
        imu_vel_fwd += imu_accel * 0.01
        
        # Washout drift if traction is high (psi_S <= 0.4)
        if psi_S <= 0.4:
            imu_vel_fwd = 0.95 * imu_vel_fwd + 0.05 * v_encoder
            
        # Cross-fade factor around slip threshold 0.4
        factor = np.clip((psi_S - 0.3) / 0.2, 0.0, 1.0)
        v_fused = (1.0 - factor) * v_encoder + factor * imu_vel_fwd
        
        # Integrate estimated pose
        est_yaw += omega * 0.01
        while est_yaw > math.pi: est_yaw -= 2 * math.pi
        while est_yaw < -math.pi: est_yaw += 2 * math.pi
        
        est_x += v_fused * math.cos(est_yaw) * 0.01
        est_y += v_fused * math.sin(est_yaw) * 0.01
        
        # Hard boundary clip to prevent state estimation from drifting off-screen
        est_x = max(-4.75, min(4.75, est_x))
        est_y = max(-4.75, min(4.75, est_y))
        prev_vel_fwd = vel_fwd
        
        # Ego-centric relative navigation to food based on estimated coordinates
        food_pos = data.geom("food").xpos.copy()
        dx_est = food_pos[0] - est_x
        dy_est = food_pos[1] - est_y
        world_angle_est = math.atan2(dy_est, dx_est)
        rel_est = world_angle_est - est_yaw
        while rel_est > math.pi: rel_est -= 2 * math.pi
        while rel_est < -math.pi: rel_est += 2 * math.pi
        dist_est = math.hypot(dx_est, dy_est)
        food_cos_est = math.cos(rel_est)
        food_sin_est = math.sin(rel_est)
        
        # Write sensor updates to double buffers for the 50Hz planner
        sensor_buffer.write_lidar(lidar_24)
        sensor_buffer.write_encoder(encoder_vel)
        sensor_buffer.write_imu(np.array([est_yaw]))  # use estimated heading
        sensor_buffer.write_drives(np.array(brain.drives.to_vec()))
        sensor_buffer.write_ego(diag['psi'])
        
        pose_buffer.write(np.array([v_fused, vel_lat, omega, food_cos_est, food_sin_est, dist_est, est_x, est_y]))
        
        # 5. Biomimetic Modulation: Inject CPG tremors/breathing Purrs
        is_stationary = np.all(np.abs(torques) < 0.05)
        drives_dict = {"cortisol": brain.drives.cort, "dopamine": brain.drives.da}
        modulated_torques = cpg.compute_biomimetic_modulation(drives_dict, torques)
        
        # 6. Actuation: Apply wheel control targets directly as velocity targets (rad/s)
        # Inject left motor fault (80% left motor attenuation / drop to 20% gain) when steps_since_food > 4000
        if steps_since_food > 4000:
            modulated_torques[0] *= 0.2
        data.ctrl[ACT_IDS["act_w_left"]] = modulated_torques[0]
        data.ctrl[ACT_IDS["act_w_right"]] = modulated_torques[1]
        
        # Apply Autonomic Posture Kinematics (neck sweeps, arms, eyelids, hip sway)
        hunger = brain.drives.hunger
        pain = brain.drives.cort
        noise_level = np.mean(np.exp(brain.actor.log_std))
        # Zero-out exploration noise during the 300-step warm-up phase to stabilize posture
        if planner_step < 300:
            noise_level = 0.0
        current_time = data.time
        
        raw_noise_x = math.sin(current_time * 28.0) * math.cos(current_time * 14.0) * noise_level
        raw_noise_y = math.cos(current_time * 32.0) * math.sin(current_time * 11.0) * noise_level
 
        hip_pitch = (hunger * 0.35) + (pain * -0.45) + (raw_noise_y * 0.25)
        hip_roll = (target_angular * 0.18) + (raw_noise_x * 0.20)
        neck_sweep = (hunger * 0.25) + (pain * -0.40) + (raw_noise_y * 0.15)
        arm_posture = 0.25 + (hunger * 0.35) + (pain * -1.10) + (raw_noise_x * 0.60)
        
        head_pan = rel_est + (raw_noise_x * 0.35)
 
        data.ctrl[ACT_IDS["act_hip_pitch"]] = np.clip(hip_pitch, -0.55, 0.55)
        data.ctrl[ACT_IDS["act_hip_roll"]] = np.clip(hip_roll, -0.38, 0.38)
        data.ctrl[ACT_IDS["act_neck"]] = np.clip(neck_sweep, -0.6, 0.7)
        data.ctrl[ACT_IDS["act_head"]] = np.clip(head_pan, -1.4, 1.4)
        
        # Symmetrical arm posture control (matching original 1-DOF shoulder pitching)
        if "act_shoulder_L" in ACT_IDS:
            data.ctrl[ACT_IDS["act_shoulder_L"]] = np.clip(arm_posture, -2.0, 2.0)
        if "act_shoulder_R" in ACT_IDS:
            data.ctrl[ACT_IDS["act_shoulder_R"]] = np.clip(arm_posture, -2.0, 2.0)
            
        # Hold new joints at default stable straight positions if they exist
        for name, default_val in [("act_shoulder_yaw_L", 0.0), ("act_elbow_L", 0.0), 
                                  ("act_wrist_L", 0.0), ("act_grip_L", 0.0),
                                  ("act_shoulder_yaw_R", 0.0), ("act_elbow_R", 0.0),
                                  ("act_wrist_R", 0.0), ("act_grip_R", 0.0)]:
            if name in ACT_IDS:
                data.ctrl[ACT_IDS[name]] = default_val
        
        # 7. Step physics simulation
        mujoco.mj_step(model, data)
        clamp_bob()
        
        # 8. Collision & Event handling
        event = check_event()
        if collision_cooldown > 0:
            collision_cooldown -= 1
        else:
            if event == 'food':
                brain.food_eaten()
                total_food += 1
                collision_cooldown = 20
                event_queue.put('food')
                
                # Respawn food pellet
                respawn_food(MAX_FOOD_DIST)
                mujoco.mj_forward(model, data)
                prev_dist = food_dist()
                
            elif event == 'wall':
                brain.wall_hit()
                total_wall += 1
                cooldown_len = int(15 + brain.drives.cort * 45)
                collision_cooldown = cooldown_len
                event_queue.put('wall')
        
        # Compute touch_front tactile bumper for telemetry
        proximity_8_tick = lidar_to_proximity(lidar_24[::3])
        touch_front_tick = 1.0 if max(proximity_8_tick[0], proximity_8_tick[7], proximity_8_tick[1]) > 0.6 else 0.0

        # 8. Throttled WebSocket Telemetry (broadcast at ~33Hz, not 100Hz)
        #    JSON serialization + TCP send is non-deterministic; keeping it
        #    in the 100Hz hot path causes 15-40ms stalls that blow deadlines.
        telemetry_tick_counter += 1
        if telemetry_tick_counter >= 3:
            telemetry_tick_counter = 0

            # Read spatiotemporal prediction error for telemetry
            err_data, _ = sensor_buffer.prediction_error.read()
            surprise = err_data[0] if len(err_data) > 0 else 0.0
            certainty = err_data[1] if len(err_data) > 1 else 1.0

            # Extract spatiotemporal predictor hidden states (32-D) and recurrent weights (32x32)
            predictor_h = [0.0]*32
            predictor_W = [[0.0]*32 for _ in range(32)]
            if hasattr(brain, 'predictor') and brain.predictor is not None:
                with brain.predictor.lock:
                    h_cache = brain.predictor.cache.get('h')
                    if h_cache is not None and len(h_cache) >= 11:
                        predictor_h = h_cache[10].tolist()
                    if 'W_rec' in brain.predictor.params:
                        predictor_W = brain.predictor.params['W_rec'].tolist()

            food_pos = data.geom("food").xpos.copy()
            blob_telemetry.update_state(
                pos=[float(est_x), float(est_y)],
                food=[float(food_pos[0]), float(food_pos[1])],
                hunger=float(brain.drives.hunger),
                speed=float(v_fused),
                da=float(brain.drives.da),
                ne=float(brain.drives.ne),
                steps_since_food=steps_since_food,
                episode=episode,
                steps_to_food_history=list(steps_history[-20:]),
                neuron_states=predictor_h,
                W=predictor_W,
                nav_signal=[food_cos_est, food_sin_est],
                reflex_fired=bool(diag['reflex_active']),
                reflex_ratio=float(touch_front_tick),
                novelty=float(brain.drives.ne),
                uncertainty=float(brain.drives.cort),
                is_sleeping=bool(global_is_sleeping),
                fatigue=float(brain.drives.fatigue),
                sigma=float(noise_level),
                speed_drive=float(brain.drives.speed_drive()),
                dreamer_loss=float(surprise),
                dreamer_surprise=float(certainty),
                hdc_familiarity=float(latest_hdc_familiarity),
                ego_state=diag['psi'].tolist()
            )

        # 9. Asynchronous Procedural Audio (throttled to ~2.86Hz)
        audio_timer += 0.01
        if audio_timer >= 0.35:
            audio.play_mood_sound(brain.drives, is_stationary)
            audio_timer = 0.0

# ── THREAD 2: 50Hz Cognitive Planner Loop ─────────────────────────────────────

def planner_loop():
    """
    Cerebral Cortex Planning. Reads double-buffered sensor state lock-free,
    computes Actor-Critic actions, dispatches velocity plans, and runs training.
    """
    global prev_dist, steps_since_food, episode, total_food, total_wall, MAX_FOOD_DIST, stats, global_is_sleeping, latest_hdc_familiarity, planner_step
    
    prev_throttle = 0.0
    prev_steering = 0.0
    
    # Wait for the first 100Hz control cycles to populate double buffers
    time.sleep(0.1)
    
    # Fast 50Hz cycle time delta
    period_s = 0.02
    
    print("[PLANNER] Asynchronous 50Hz Planner thread is running.")
    
    while bios._running:
        t_start = time.perf_counter()
        
        # 1. Read double buffers (lock-free, <1μs)
        snapshot = sensor_buffer.read_all()
        lidar_24 = snapshot['lidar'][0]
        encoder_vel = snapshot['encoder_vel'][0]
        yaw = snapshot['imu_heading'][0][0]
        drive_vec = snapshot['drive_state'][0]
        face_expr = snapshot['face_expression'][0]
        ego_vec = snapshot['ego_state'][0]
        
        err_data = snapshot['prediction_error'][0]
        surprise = err_data[0] if len(err_data) > 0 else 0.0
        
        pose_data, _ = pose_buffer.read()
        vel_fwd, vel_lat, omega, food_cos, food_sin, dist, carl_x, carl_y = pose_data
        
        # Connect smile and presence metrics directly to homeostatic drives:
        face_present, smile, eye_opening, eyebrow_raise, face_distance = face_expr
        if face_present > 0.5:
            # Soothe cortisol: pull it down towards zero
            brain.drives.cort = max(0.0, brain.drives.cort - 0.05 * face_present)
            # Elevate dopamine: add up to +0.02 per step based on smile intensity
            brain.drives.da = min(1.0, brain.drives.da + 0.02 * smile)
            
        # Wire motor health and slip directly to cortisol drive
        psi_L, psi_R, psi_S, psi_T = ego_vec
        if psi_L < 0.6 or psi_R < 0.6:
            brain.drives.cort = min(1.0, brain.drives.cort + 0.04)
        if psi_S > 0.8:
            brain.drives.cort = min(1.0, brain.drives.cort + 0.08)
        
        # 2. Downsample 24-ray LiDAR to 8-rays for perception
        lidar_8 = lidar_24[::3]
        proximity = lidar_to_proximity(lidar_8)
        
        # Bumpers
        touch_front = 1.0 if max(proximity[0], proximity[7], proximity[1]) > 0.6 else 0.0
        touch_back  = 1.0 if max(proximity[3], proximity[4], proximity[5]) > 0.6 else 0.0
        touch_left  = 1.0 if proximity[6] > 0.6 else 0.0
        touch_right = 1.0 if proximity[2] > 0.6 else 0.0
        
        # 3. Compile observation array
        obs = build_observation(
            proximity, food_cos, food_sin,
            vel_fwd, vel_lat, omega,
            touch_front, touch_back, touch_left, touch_right,
            prev_throttle, prev_steering,
            brain.drives,
            face_expr,
            ego_vec,
            surprise
        )
        
        # 4. Check events from 100Hz thread queue
        events = []
        while not event_queue.empty():
            try:
                events.append(event_queue.get_nowait())
            except queue.Empty:
                break
                
        food_found = False
        for ev in events:
            if ev == 'food':
                food_found = True
            elif ev == 'wall':
                print(f"[PAIN] Wall #{total_wall} | ep {episode} step {steps_since_food} | "
                      f"CORT={brain.drives.cort:.2f} | NE={brain.drives.ne:.2f} | "
                      f"DA={brain.drives.da:.2f}", flush=True)
                
        # Compute Actor-Critic reward signal
        progress = (prev_dist - dist) * 5.0
        facing_bonus = max(0, food_cos) * 0.2
        time_penalty = -0.01
        reward = progress + facing_bonus + time_penalty
        
        if food_found:
            reward += 20.0
            
            # Record curriculum statistics
            steps_history.append(steps_since_food)
            if len(steps_history) > 50:
                steps_history.pop(0)
                
            recent_avg = int(np.mean(steps_history[-5:])) if len(steps_history) >= 5 else steps_since_food
            improved = maybe_advance_curriculum(steps_history)
            
            print(f"[JOY] Ep {episode:4d} | food in {steps_since_food:5d} steps | "
                  f"recent5_avg={recent_avg:5d} | "
                  f"food_range={MAX_FOOD_DIST:.1f}m | "
                  f"wall_ratio={total_wall/(total_food+1):.1f}x | "
                  f"DA={brain.drives.da:.2f}", flush=True)
                  
            stats["total_food"] = total_food
            stats["total_wall"] = total_wall
            stats["curriculum_dist"] = MAX_FOOD_DIST
            stats["episodes"].append({
                "ep": episode, "steps": steps_since_food,
                "food_range": MAX_FOOD_DIST, "wall_ratio": total_wall / (total_food + 1)
            })
            
            steps_since_food = 0
            episode += 1
            brainstem.reset()
            
            # Offload disk saves to the 50Hz planner thread to protect the 100Hz motor tick
            if episode % 10 == 0:
                brain.save("memory/carl_brain")
                save_stats(stats)
                print(f"[SAVE] Brain + stats saved in planner thread. Episode {episode}.")
                
        # 5. Handle Episode Timeout
        if steps_since_food >= MAX_EP_STEPS:
            print(f"[TIMEOUT] Episode {episode} timed out. Respawning in planner thread.")
            steps_history.append(MAX_EP_STEPS)
            steps_since_food = 0
            episode += 1
            brain.drives.cort *= 0.3
            brain.drives.ne   *= 0.5
            brain.drives.fatigue = 0.0
            brainstem.reset()
            
            with mujoco_lock:
                respawn_bob()
                respawn_food(MAX_FOOD_DIST)
                mujoco.mj_forward(model, data)
                prev_dist = food_dist()

        # 6. Autonomic Nest Sleep / Consolidation
        distance_to_nest = math.hypot(carl_x - NEST_CORNER_X, carl_y - NEST_CORNER_Y)
        if getattr(brain.drives, 'fatigue', 0.0) > 0.90 and distance_to_nest < 0.6:
            print("[ALLOSTASIS] Exhaustion threshold hit. Entering offline sleep consolidation...")
            global_is_sleeping = True
            
            # Send zero target velocities to stay stationary in the nest
            sensor_buffer.write_planning(np.array([0.0, 0.0]))
            
            dream_cycles = 0
            while getattr(brain.drives, 'fatigue', 0.0) > 0.05 and bios._running:
                # Consolidate holographic associative memories
                brain.hdc.sleep_consolidation()
                dream_cycles += 1
                
                brain.drives.fatigue -= 0.05
                time.sleep(0.1)
                
            global_is_sleeping = False
            print(f"[ALLOSTASIS] Sleep complete. Consolidated {dream_cycles} cycles. Waking up.")
            brain.actor.reset_traces()
            steps_since_food = 0
            
        # 7. Navigation waypoint planning (routing around the center obstacle)
        target_wx, target_wy = NEST_CORNER_X, NEST_CORNER_Y
        dx_nest = NEST_CORNER_X - carl_x
        dy_nest = NEST_CORNER_Y - carl_y
        dist_nest = math.hypot(dx_nest, dy_nest)
        if dist_nest > 0.1:
            d_line = abs(carl_x * NEST_CORNER_Y - carl_y * NEST_CORNER_X) / dist_nest
            dot = -carl_x * dx_nest - carl_y * dy_nest
            if d_line < 0.85 and dot > 0 and dot < dist_nest * dist_nest:
                # Route around center obstacle
                if carl_y > carl_x:
                    target_wx, target_wy = -1.2, 1.2
                else:
                    target_wx, target_wy = 1.2, -1.2
                    
        global_dx = target_wx - carl_x
        global_dy = target_wy - carl_y
        nest_local_dx = global_dx * math.cos(yaw) + global_dy * math.sin(yaw)
        nest_local_dy = -global_dx * math.sin(yaw) + global_dy * math.cos(yaw)
        nest_rel_vector = (nest_local_dx, nest_local_dy)

        # 8. Think: Run Active Inference Engine (out of the lock!)
        if planner_step < 300:
            # Warm-up phase: generate simple exploration actions moving FORWARD
            throttle = 0.4
            steering = float(np.random.uniform(-0.1, 0.1))
            action = np.array([throttle, steering], dtype=np.float32)
            G_val = 0.0
            # Normalize observation using the brain's internal method
            obs_norm = brain._normalize_obs(obs)
            # Populate history window for active inference later
            if not hasattr(brain, 'obs_history_window'):
                brain.obs_history_window = []
            brain.obs_history_window.append(obs_norm[:33])
            if len(brain.obs_history_window) > 10:
                brain.obs_history_window.pop(0)
            brain.prev_obs = obs_norm.copy()
            brain.prev_action = action.copy()
            latest_hdc_familiarity = 0.0
        else:
            if planner_step == 300:
                brainstem.reset()
            action, _, hdc_familiarity, G_val = brain.active_inference_step(obs, nest_rel_vector=nest_rel_vector, target_pos=np.array([target_wx, target_wy], dtype=np.float32), current_pose=np.array([carl_x, carl_y, yaw], dtype=np.float32))
            latest_hdc_familiarity = hdc_familiarity
            throttle, steering = float(action[0]), float(action[1])
        
        # 9. Output targets: Write linear and angular targets to double buffer
        sensor_buffer.planning_target.write(np.array([throttle, steering]))
        
        # Dispatch transition to the 15Hz background world-model training thread
        try:
            # Predictor and HDC train on the 33-D sensory state (excl. surprise)
            transition_queue.put_nowait((brain.prev_obs[:33], action))
        except queue.Full:
            pass
            
        prev_throttle = throttle
        prev_steering = steering
        prev_dist = dist
        steps_since_food += 1
        planner_step += 1

        # Precise 50Hz delay
        elapsed = time.perf_counter() - t_start
        rem = period_s - elapsed
        if rem > 0:
            time.sleep(rem)


# Initialize Bob's spawn and food positions prior to launching threads
with mujoco_lock:
    respawn_bob()
    respawn_food(MAX_FOOD_DIST)
    mujoco.mj_forward(model, data)
    prev_dist = food_dist()

# Start WS telemetry server
blob_telemetry.start()

# Instantiate BIOS timing nucleus
bios = CarlBiosCore()
bios.register_control_callback(on_control_tick)

# Register a watchdog fault handler
# CRITICAL: This callback runs on the watchdog thread (250Hz).
# It must NEVER acquire mujoco_lock — the control tick holds that lock
# when the watchdog fires, causing an unrecoverable deadlock.
# MuJoCo data.ctrl[] writes are atomic float stores (C-level memory
# assignment to a pre-allocated contiguous array), so direct writes
# are safe without a mutex.
def on_watchdog_fault():
    print("[EMERGENCY] WATCHDOG: Control heartbeat stale. Zeroing actuators.")
    data.ctrl[ACT_IDS["act_w_left"]] = 0.0
    data.ctrl[ACT_IDS["act_w_right"]] = 0.0

bios.register_watchdog_callback(on_watchdog_fault)

# Instantiate and start Social Visual Cortex (10Hz asynchronous thread)
social_lobe = SocialVisualLobe(shared_sensor_buffer=sensor_buffer, camera_index=0, frame_rate=10)
social_lobe.start()

# Instantiate and start spatiotemporal world predictor (15Hz background thread)
from carl_imagination import ImaginationThread
transition_queue = queue.Queue(maxsize=1000)
imagination_lobe = ImaginationThread(sensor_buffer, transition_queue, obs_dim=33, act_dim=2)
# Link the predictor weights so the brain has references
brain.predictor = imagination_lobe.predictor
imagination_lobe.start()

# Launch system
bios.start_system()

# Launch planner thread
planner_thread = threading.Thread(target=planner_loop, name="CerebralPlanner", daemon=True)
planner_thread.start()

print("[SYSTEM] Asynchronous Multi-Rate pipelines are active. Press Ctrl+C to stop.")

try:
    # Stay alive in main thread
    while True:
        time.sleep(1.0)
except KeyboardInterrupt:
    print("\n[SYSTEM] Interrupted by user. Shutting down...")
    
    # 1. Stop threads
    bios.shutdown_system()
    social_lobe.stop()
    imagination_lobe.stop()
    
    # 2. Save brain and stats
    print("[SYSTEM] Saving weights and stats offline...")
    brain.save("memory/carl_brain")
    stats["total_food"] = total_food
    stats["total_wall"] = total_wall
    stats["curriculum_dist"] = MAX_FOOD_DIST
    save_stats(stats)
    
    print("[SYSTEM] Shutdown completed cleanly. Bob is asleep.")
