"""
carl_train.py — CARL v4: Bob Learns For Real.

Pipeline (Boston Dynamics-inspired):
  Perception (fixed lidar, proprioception)
      -> Observation vector (24-dim)
      -> Actor-Critic RL (policy gradient + TD)
      -> Spinal reflex override (innate)
      -> Physics actuators
      -> Reward signal (food/wall/progress)
      -> Biological drives update
      -> Brain update
      -> [loop]

Fixes applied vs all previous versions:
  1. LIDAR EXCLUDES SELF-GEOMETRY (was the primary bug — Bob was blind)
  2. Real trainable Actor-Critic network (not ESN randomness)
  3. Proper observation normalization
  4. Reward normalization
  5. Curriculum learning (food starts close, gets farther as Bob improves)
  6. Domain randomization (random start positions)
  7. Full biological neuromodulator state
"""

import math
import time
import json
import os
import sys
import numpy as np
import mujoco

sys.path.append(os.path.abspath('.'))
from carl_agent import CarlBrain
from carl_expression import ExpressionController
import blob_telemetry

# ── Telemetry ──────────────────────────────────────────────────────────────────
blob_telemetry.start()

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

mujoco.mj_resetData(model, data)
mujoco.mj_forward(model, data)

# Identify geom IDs once (used by lidar exclusion)
BLOB_GEOM_ID = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "chassis")
FOOD_GEOM_ID = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "food")

# Resolve CARL's freejoint qpos and qvel base addresses
Q = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'spatial_identity')]
V = model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'spatial_identity')]

# Get true initial body position AFTER resetData+forward
init_pos = data.geom("chassis").xpos.copy()
print(f"[INIT] Bob's true spawn: ({init_pos[0]:.3f}, {init_pos[1]:.3f})")

# ── Brain ──────────────────────────────────────────────────────────────────────
N_OBS = 24   # 8 lidar + 3 velocity + 2 food-direction + 5 neuromod + 4 touch + 2 prev_action
brain = CarlBrain(n_obs=N_OBS)
brain.load("memory/carl_brain")

# ── Curriculum state ──────────────────────────────────────────────────────────
MAX_FOOD_DIST  = 1.5   # Start: food spawns within 1.5m. Expands as Bob improves.
curriculum_step = 0

# ── Stats ──────────────────────────────────────────────────────────────────────
STATS_FILE = "memory/carl_stats.json"
def load_stats():
    if os.path.exists(STATS_FILE):
        with open(STATS_FILE) as f:
            return json.load(f)
    return {"episodes": [], "total_food": 0, "total_wall": 0,
            "curriculum_dist": 1.5, "best_avg": 9999}

def save_stats(s):
    with open(STATS_FILE, "w") as f:
        json.dump(s, f, indent=2)

stats = load_stats()
MAX_FOOD_DIST = 1.5 # Start at 1.5m — no more free food


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_yaw(quat):
    """Convert quaternion [w, x, y, z] to yaw angle (rotation around Z)."""
    w, x, y, z = quat
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

def set_yaw(yaw):
    """Convert yaw angle to quaternion [w, x, y, z]."""
    return [math.cos(yaw/2.0), 0.0, 0.0, math.sin(yaw/2.0)]

def get_lidar(n_rays=8):
    """
    Cast n rays in a full circle. Ray origin is moved OUTSIDE the blob
    cylinder (radius=0.2m) by 0.22m so rays see the real environment.
    Ray 0=front, 1=front-right, 2=right, 3=back-right,
    4=back, 5=back-left, 6=left, 7=front-left  (clockwise)
    """
    # Temporarily hide food geom from LiDAR raycasts
    food_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "food")
    old_food_pos = model.geom_pos[food_id].copy()
    model.geom_pos[food_id] = np.array([99.0, 99.0, 99.0])
    
    blob_pos = data.geom("chassis").xpos
    yaw = get_yaw(data.qpos[Q+3:Q+7])
    dists = []
    for i in range(n_rays):
        angle = yaw + i * (2.0 * math.pi / n_rays)
        vx = math.cos(angle)
        vy = math.sin(angle)
        vec = np.array([vx, vy, 0.0])
        # Move ray origin 0.22m past the blob surface + lift off floor
        pnt = np.array([blob_pos[0] + vx * 0.22,
                        blob_pos[1] + vy * 0.22,
                        blob_pos[2] + 0.05])
        geom_hit = np.array([-1], dtype=np.int32)
        dist = mujoco.mj_ray(model, data, pnt, vec, None, 1, -1, geom_hit)
        # Add 0.22 back so distance is from blob centre, not from offset origin
        dists.append((dist + 0.22) if dist > 0 else 5.0)
        
    # Restore food geom position
    model.geom_pos[food_id] = old_food_pos
    return dists


def lidar_to_proximity(dists, max_range=2.5):
    """Convert distances to proximity [0,1]. 1.0 = wall immediately ahead."""
    return [max(0.0, (max_range - d) / max_range) for d in dists]


def get_food_direction():
    """
    Ego-centric angle to food: (cos(rel_angle), sin(rel_angle)).
    cos=1 means food is directly ahead. cos=-1 means food is behind.
    """
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
    # Food proximity (food has no collider — purely positional)
    if food_dist() < 0.35:
        return 'food'
    # Outer wall (absolute coordinates)
    abs_x = data.qpos[Q]
    abs_y = data.qpos[Q+1]
    if abs(abs_x) > 4.65 or abs(abs_y) > 4.65:
        return 'wall'
    # Center obstacle
    if math.hypot(blob_pos[0], blob_pos[1]) < 0.72:
        return 'wall'
    return None


MIN_FOOD_DIST = 1.0   # STRICT: food never spawns closer than 1.0m

def respawn_food(max_dist=None, random_pos=True):
    """Spawn food within curriculum distance but NEVER closer than MIN_FOOD_DIST."""
    food_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "food")
    blob_pos = data.geom("chassis").xpos
    d = max_dist or MAX_FOOD_DIST
    # Enforce minimum distance floor
    min_d = max(MIN_FOOD_DIST, 1.0)
    if d <= min_d:
        d = min_d + 0.5  # guarantee a valid range
    for _ in range(200):
        if random_pos:
            angle = np.random.uniform(0, 2*math.pi)
            radius = np.random.uniform(min_d, d)  # STRICT: [1.0m, max]
            fx = blob_pos[0] + radius * math.cos(angle)
            fy = blob_pos[1] + radius * math.sin(angle)
        else:
            fx = np.random.uniform(-4.0, 4.0)
            fy = np.random.uniform(-4.0, 4.0)
        fx = np.clip(fx, -4.2, 4.2)
        fy = np.clip(fy, -4.2, 4.2)
        actual_dist = math.hypot(fx - blob_pos[0], fy - blob_pos[1])
        if math.hypot(fx, fy) > 0.9 and actual_dist >= min_d:
            model.geom_pos[food_id][0] = fx
            model.geom_pos[food_id][1] = fy
            return actual_dist
    # Fallback — still enforce distance
    model.geom_pos[food_id][0] = np.random.uniform(1.5, 3.0)
    model.geom_pos[food_id][1] = np.random.uniform(1.5, 3.0)
    return 2.0


def respawn_bob():
    """Randomize Bob's starting position (domain randomization)."""
    # Pick a corner or random arena quadrant
    corners = [(-2.5, -2.5), (2.5, -2.5), (-2.5, 2.5), (2.5, 2.5)]
    cx, cy = corners[np.random.randint(len(corners))]
    # Small jitter
    cx += np.random.uniform(-0.5, 0.5)
    cy += np.random.uniform(-0.5, 0.5)
    data.qpos[Q] = cx
    data.qpos[Q+1] = cy
    data.qpos[Q+2] = 0.05 # Lower Z height for a gentle landing to prevent tipping
    rand_yaw = np.random.uniform(-math.pi, math.pi)
    data.qpos[Q+3:Q+7] = set_yaw(rand_yaw)
    data.qvel[:] = 0.0
    if 'brain' in globals() and hasattr(brain, 'actor'):
        brain.actor.reset_traces()


def build_observation(proximity, food_cos, food_sin, vel_fwd, vel_lat, omega,
                       touch_front, touch_back, touch_left, touch_right,
                       prev_throttle, prev_steering, drives):
    """
    24-dimensional observation:
    [0:8]   - lidar proximity (0=front, clockwise)
    [8:11]  - ego-velocities (forward, lateral, angular)
    [11:13] - food direction (cos, sin of relative angle)
    [13:18] - neuromodulator state (da, ne, cort, sero, hunger)
    [18:22] - tactile bumpers (front, back, left, right)
    [22:24] - previous actions (throttle, steering)
    """
    return np.array(
        list(proximity) +
        [vel_fwd, vel_lat, omega] +
        [food_cos, food_sin] +
        list(drives.to_vec())[:5] +
        [touch_front, touch_back, touch_left, touch_right] +
        [prev_throttle, prev_steering],
        dtype=np.float32
    )


def clamp_bob():
    """Hard boundary enforcement. Returns True if boundary was hit."""
    abs_x = data.qpos[Q]
    abs_y = data.qpos[Q+1]
    cx = np.clip(abs_x, -4.65, 4.65)
    cy = np.clip(abs_y, -4.65, 4.65)
    hit = (abs_x != cx) or (abs_y != cy)
    if hit:
        data.qpos[Q] = cx
        data.qpos[Q+1] = cy
        # Reflect velocity away from the wall
        if abs(abs_x) > 4.65:
            data.qvel[V] = -data.qvel[V] * 0.2
        if abs(abs_y) > 4.65:
            data.qvel[V+1] = -data.qvel[V+1] * 0.2
        mujoco.mj_forward(model, data)
    return hit


WARN_ZONE = 4.3   # Start applying escape force inside this boundary

def wall_proximity_override(yaw):
    """Reflex: Safe directional steering/throttle override to escape obstacles/walls."""
    abs_x = data.qpos[Q]
    abs_y = data.qpos[Q+1]
    dist_from_center = math.hypot(abs_x, abs_y)
    
    WARN_ZONE = 4.2
    OBSTACLE_RADIUS = 0.5
    OBSTACLE_WARN = 0.75
    
    is_outer_wall = abs(abs_x) > WARN_ZONE or abs(abs_y) > WARN_ZONE
    is_center_obstacle = dist_from_center < OBSTACLE_WARN
    
    if not (is_outer_wall or is_center_obstacle):
        return 0.0
        
    n_x, n_y = 0.0, 0.0
    strength = 1.0
    
    if is_outer_wall:
        if abs_x > WARN_ZONE:
            n_x = -1.0
        elif abs_x < -WARN_ZONE:
            n_x = 1.0
        if abs_y > WARN_ZONE:
            n_y = -1.0
        elif abs_y < -WARN_ZONE:
            n_y = 1.0
        depth_x = max(0.0, abs(abs_x) - WARN_ZONE) / (4.65 - WARN_ZONE)
        depth_y = max(0.0, abs(abs_y) - WARN_ZONE) / (4.65 - WARN_ZONE)
        strength = min(1.0, max(depth_x, depth_y))
    elif is_center_obstacle:
        if dist_from_center > 0.01:
            n_x = abs_x / dist_from_center
            n_y = abs_y / dist_from_center
        else:
            n_x = 1.0
            n_y = 0.0
        depth = max(0.0, OBSTACLE_WARN - dist_from_center) / (OBSTACLE_WARN - OBSTACLE_RADIUS)
        strength = min(1.0, depth)
        
    n_len = math.hypot(n_x, n_y)
    if n_len > 1e-5:
        n_x /= n_len
        n_y /= n_len
        
    safety_angle = math.atan2(n_y, n_x)
    alignment = math.cos(yaw - safety_angle)
    
    gain = 2.0
    max_actuator_limit = 5.0
    
    if alignment > 0.0:
        desired_throttle = max_actuator_limit * strength
        target_yaw = safety_angle
    else:
        desired_throttle = -max_actuator_limit * strength
        target_yaw = safety_angle + math.pi
        
    delta = target_yaw - yaw
    while delta > math.pi: delta -= 2 * math.pi
    while delta < -math.pi: delta += 2 * math.pi
    
    desired_steering = np.clip(delta * gain, -max_actuator_limit, max_actuator_limit)
    
    if strength > 0.1:
        left_w = desired_throttle - desired_steering
        right_w = desired_throttle + desired_steering
        data.ctrl[ACT_IDS["act_w_left"]] = np.clip(left_w, -max_actuator_limit, max_actuator_limit)
        data.ctrl[ACT_IDS["act_w_right"]] = np.clip(right_w, -max_actuator_limit, max_actuator_limit)
        
    return strength


# ── Curriculum advancement ────────────────────────────────────────────────────
def maybe_advance_curriculum(steps_history):
    """
    If Bob's recent average steps-to-food < threshold, widen the arena.
    This is the sim-to-real domain expansion equivalent.
    """
    global MAX_FOOD_DIST
    if len(steps_history) < 10:
        return False
    recent_avg = np.mean(steps_history[-10:])
    if recent_avg < 800 and MAX_FOOD_DIST < 8.0:
        MAX_FOOD_DIST = min(8.0, MAX_FOOD_DIST + 0.5)
        print(f"[CURRICULUM] Advancing! Food can now spawn up to {MAX_FOOD_DIST:.1f}m away. Avg was {recent_avg:.0f} steps.")
        stats["curriculum_dist"] = MAX_FOOD_DIST
        return True
    return False


# ── Main training loop ────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  CARL v4 * Actor-Critic RL * Bob with Real Eyes")
print("  Lidar: FIXED (self-exclusion active)")
print("  Brain: Trainable Actor-Critic (not ESN)")
print("  Curriculum: Active (food range expands as Bob learns)")
print("="*60 + "\n")

# Expression controller is now handled purely by the Autonomic Translation Matrix

# Initialize with curriculum-bounded food
respawn_bob()
respawn_food(MAX_FOOD_DIST)
mujoco.mj_forward(model, data)

prev_dist = food_dist()
prev_throttle = 0.0
prev_steering = 0.0
collision_cooldown = 0
steps_since_food = 0
episode = 1
steps_history = []
total_food = stats.get("total_food", 0)
total_wall  = stats.get("total_wall", 0)

MAX_EP_STEPS = 8000   # Timeout

NEST_CORNER_X = -3.5
NEST_CORNER_Y = -3.5
experience_buffer = []
import random

try:
    step = 0
    while True:
        # --- 1. DAY/NIGHT ECOLOGICAL FRICTION ---
        is_night_cycle = (step % 2000 > 1000)

        # ── Sense ─────────────────────────────────────────────────────────────
        lidars    = get_lidar()
        
        if is_night_cycle:
            # Night Blindness: Truncate LiDAR ranges to simulate low visibility
            lidars = [min(l, 0.25) for l in lidars]
            if step % 2000 == 1001:
                brain.drives.novelty_event(novelty=0.3)

        proximity = lidar_to_proximity(lidars)

        blob_pos = data.geom("chassis").xpos.copy()
        
        # --- 2. ALLOSTATIC NEST SAFETY CHECK ---
        distance_to_nest = math.hypot(blob_pos[0] - NEST_CORNER_X, blob_pos[1] - NEST_CORNER_Y)
        
        if getattr(brain.drives, 'fatigue', 0.0) > 0.90 and distance_to_nest < 0.6:
            print("[ALLOSTASIS] Exhaustion threshold hit. Entering Dream State Consolidation Loop...")
            dream_cycles = 0
            while getattr(brain.drives, 'fatigue', 0.0) > 0.05:
                if len(experience_buffer) > 10:
                    # ── PHASE A: Classic Experience Replay (stabilize critic) ──
                    sampled_batch = random.sample(experience_buffer, min(len(experience_buffer), 5))
                    for hist_raw_obs, old_action, old_mu, target_v in sampled_batch:
                        dream_hallucination = hist_raw_obs + np.random.randn(len(hist_raw_obs)) * 0.04
                        norm_dream = brain._normalize_obs(dream_hallucination, freeze_stats=True)
                        brain.critic.update_value(norm_dream, target_v)
                        dream_v = brain.critic.forward(norm_dream)
                        hallucinated_advantage = target_v - dream_v
                        brain.actor.update_policy(hallucinated_advantage, norm_dream, old_action, old_mu, brain.drives.ne)
                    
                    # ── PHASE B: DREAMER IMAGINED ROLLOUTS (train actor in simulation) ──
                    # Pick a random seed memory and let the world model dream forward
                    seed_obs = random.choice(experience_buffer)[0]
                    norm_seed = brain._normalize_obs(seed_obs, freeze_stats=True)
                    imagined = brain.dreamer.imagine_rollout(norm_seed, brain.actor, steps=8)
                    for dream_obs, dream_act, dream_next in imagined:
                        # Critic evaluates the imagined states
                        v_now = brain.critic.forward(dream_obs)
                        v_next = brain.critic.forward(dream_next)
                        imagined_advantage = v_next - v_now
                        mu_dream = brain.actor.forward(dream_obs)
                        brain.actor.update_policy(
                            imagined_advantage, dream_obs, dream_act, mu_dream,
                            brain.drives.ne, brain.drives.cort, brain.drives.sero
                        )
                    dream_cycles += 1
                        
                brain.drives.fatigue -= 0.05
                time.sleep(0.01)
            print(f"[ALLOSTASIS] Fatigue cleared. Brain consolidated {dream_cycles} dream cycles. Waking up.")
            brain.actor.reset_traces()
            steps_since_food = 0

        yaw      = get_yaw(data.qpos[Q+3:Q+7])
        vel_x    = float(data.qvel[V])
        vel_y    = float(data.qvel[V+1])
        omega    = float(data.qvel[V+5])
        vel_fwd  =  vel_x * math.cos(yaw) + vel_y * math.sin(yaw)
        vel_lat  = -vel_x * math.sin(yaw) + vel_y * math.cos(yaw)

        food_cos, food_sin, dist = get_food_direction()

        # Tactile bumpers (cardinal directions)
        touch_front = 1.0 if max(proximity[0], proximity[7], proximity[1]) > 0.6 else 0.0
        touch_back  = 1.0 if max(proximity[3], proximity[4], proximity[5]) > 0.6 else 0.0
        touch_left  = 1.0 if proximity[6] > 0.6 else 0.0
        touch_right = 1.0 if proximity[2] > 0.6 else 0.0

        # ── Build observation ─────────────────────────────────────────────────
        obs = build_observation(
            proximity, food_cos, food_sin,
            vel_fwd, vel_lat, omega,
            touch_front, touch_back, touch_left, touch_right,
            prev_throttle, prev_steering,
            brain.drives
        )

        # ── Compute reward ────────────────────────────────────────────────────
        # Progress reward: approach food (+), retreat (-)
        progress = (prev_dist - dist) * 5.0

        # Shaping: bonus for facing food
        facing_bonus = max(0, food_cos) * 0.2

        # Time penalty: small cost for existing (encourages efficiency)
        time_penalty = -0.01

        # Composite shaped reward
        reward = progress + facing_bonus + time_penalty

        # Step-based fallback save so we don't lose the brain if he's constantly timing out
        if step % 10000 == 0:
            brain.save("memory/carl_brain")



        # Event rewards handled below
        event = check_event()
        if event == 'food':
            reward += 20.0
        elif event == 'wall':
            reward -= 3.0

        # ── Brain step (sense -> think -> act) ───────────────────────────────
        NEST_X, NEST_Y = -3.5, -3.5
        carl_x = data.qpos[Q]
        carl_y = data.qpos[Q+1]
        
        # Calculate waypoint to clear center obstacle if the straight path to nest is blocked
        target_x, target_y = NEST_X, NEST_Y
        dx_nest = NEST_X - carl_x
        dy_nest = NEST_Y - carl_y
        dist_nest = math.hypot(dx_nest, dy_nest)
        if dist_nest > 0.1:
            d_line = abs(carl_x * NEST_Y - carl_y * NEST_X) / dist_nest
            dot = -carl_x * dx_nest - carl_y * dy_nest
            if d_line < 0.85 and dot > 0 and dot < dist_nest * dist_nest:
                # Path to nest is blocked by center obstacle. Route around it.
                if carl_y > carl_x:
                    target_x, target_y = -1.2, 1.2
                else:
                    target_x, target_y = 1.2, -1.2
        
        # Extract current yaw rotation angle from the orientation quaternion
        q = data.qpos[Q+3:Q+7]
        siny_cosp = 2.0 * (q[0] * q[3] + q[1] * q[2])
        cosy_cosp = 1.0 - 2.0 * (q[2] * q[2] + q[3] * q[3])
        current_yaw = math.atan2(siny_cosp, cosy_cosp)
        
        # Calculate global displacement vector to target (nest or waypoint)
        global_dx = target_x - carl_x
        global_dy = target_y - carl_y
        
        # Rotate the displacement vector into Carl's local reference frame
        nest_local_dx = global_dx * math.cos(current_yaw) + global_dy * math.sin(current_yaw)
        nest_local_dy = -global_dx * math.sin(current_yaw) + global_dy * math.cos(current_yaw)
        nest_rel_vector = (nest_local_dx, nest_local_dy)

        action, reflex_fired, hdc_familiarity = brain.step(obs, reward, done=False, nest_rel_vector=nest_rel_vector)
        throttle, steering = float(action[0]), float(action[1])

        # ── Apply actions to physics (Differential Drive) ────────────────────
        throttle, steering = float(action[0]), float(action[1])
        data.ctrl[ACT_IDS["act_w_left"]] = throttle - steering  # w_left
        data.ctrl[ACT_IDS["act_w_right"]] = throttle + steering  # w_right

        # ── Autonomic Character Kinematics Translation Matrix ───────────────
        hunger = brain.drives.hunger
        pain = brain.drives.cort
        noise_level = np.mean(np.exp(brain.actor.log_std))
        current_time = data.time
        
        raw_noise_x = math.sin(current_time * 28.0) * math.cos(current_time * 14.0) * noise_level
        raw_noise_y = math.cos(current_time * 32.0) * math.sin(current_time * 11.0) * noise_level

        hip_pitch = (hunger * 0.35) + (pain * -0.45) + (raw_noise_y * 0.25)
        hip_roll = (steering * 0.18) + (raw_noise_x * 0.20)
        neck_sweep = (hunger * 0.25) + (pain * -0.40) + (raw_noise_y * 0.15)
        arm_posture = 0.25 + (hunger * 0.35) + (pain * -1.10) + (raw_noise_x * 0.60)
        
        food_dx, food_dy = math.cos(yaw) * dist, math.sin(yaw) * dist
        food_angle = math.atan2(food_dy, food_dx) - yaw
        head_pan = food_angle + (raw_noise_x * 0.35)

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

        prev_throttle = throttle
        prev_steering = steering
        prev_dist     = dist

        # ── Event handling ────────────────────────────────────────────────────
        if collision_cooldown > 0:
            collision_cooldown -= 1
        else:
            if event == 'food':
                brain.food_eaten()
                total_food += 1
                collision_cooldown = 20
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
                stats["episodes"].append({
                    "ep": episode, "steps": steps_since_food,
                    "food_range": MAX_FOOD_DIST, "wall_ratio": total_wall / (total_food + 1)
                })
                if len(stats["episodes"]) % 5 == 0:
                    save_stats(stats)
                    brain.save("memory/carl_brain")

                steps_since_food = 0
                episode += 1
                collision_cooldown = 0
                # Reset cortisol on food reward — success should relieve stress
                brain.drives.cort *= 0.2

                # Domain randomization: do NOT re-spawn Bob on food find to allow continuous exploration
                # respawn_bob()
                respawn_food(MAX_FOOD_DIST)
                mujoco.mj_forward(model, data)
                prev_dist = food_dist()

                # Save brain every 10 episodes
                if episode % 10 == 0:
                    brain.save("memory/carl_brain")
                    save_stats(stats)
                    print(f"[SAVE] Brain + stats saved. Episode {episode}.")

            elif event == 'wall':
                brain.wall_hit()
                total_wall += 1
                # Adaptive cooldown: longer when chronically stressed (CORT high)
                # This prevents rapid re-triggering that causes 100+ hit loops
                cooldown_len = int(15 + brain.drives.cort * 45)  # 15-60 frames
                collision_cooldown = cooldown_len
                stats["total_wall"] = total_wall
                print(f"[PAIN] Wall #{total_wall} | ep {episode} step {steps_since_food} | "
                      f"CORT={brain.drives.cort:.2f} | NE={brain.drives.ne:.2f} | "
                      f"lidar_front={lidars[0]:.2f}m | cooldown={cooldown_len}")

        # ── Episode timeout ───────────────────────────────────────────────────
        if steps_since_food >= MAX_EP_STEPS:
            print(f"[TIMEOUT] Episode {episode} timed out. Respawning.")
            steps_history.append(MAX_EP_STEPS)
            steps_since_food = 0
            episode += 1
            collision_cooldown = 0
            # Partial cortisol reset on episode boundary — fresh start matters
            brain.drives.cort *= 0.3
            brain.drives.ne   *= 0.5
            brain.drives.fatigue = 0.0 # Reset fatigue to prevent chronic fatigue lock
            respawn_bob()
            mujoco.mj_forward(model, data)
            respawn_food(MAX_FOOD_DIST)
            mujoco.mj_forward(model, data)
            prev_dist = food_dist()

        # ── Geometric wall escape (brainstem-level, overrides cortex) ──────────
        wall_proximity_override(yaw)

        # ── Physics step ──────────────────────────────────────────────────────
        mujoco.mj_step(model, data)
        clamp_bob()

        # ── Telemetry ─────────────────────────────────────────────────────────
        blob_pos = data.geom("chassis").xpos.copy()
        food_pos = data.geom("food").xpos.copy()
        blob_telemetry.update_state(
            pos=[float(blob_pos[0]), float(blob_pos[1])],
            food=[float(food_pos[0]), float(food_pos[1])],
            hunger=float(brain.drives.hunger),
            speed=float(math.hypot(vel_x, vel_y)),
            da=float(brain.drives.da),
            ne=float(brain.drives.ne),
            steps_since_food=steps_since_food,
            episode=episode,
            steps_to_food_history=list(steps_history[-20:]),
            neuron_states=brain.actor.l3.a.tolist() if hasattr(brain.actor.l3, 'a') else [],
            W=brain.actor.mu_head.W.tolist(),
            nav_signal=[food_cos, food_sin],
            reflex_fired=bool(reflex_fired),
            reflex_ratio=float(touch_front),
            novelty=float(brain.drives.ne),
            uncertainty=float(brain.drives.cort),
            is_sleeping=False,
            fatigue=float(getattr(brain.drives, 'fatigue', 0.0)),
            sigma=float(noise_level),
            speed_drive=float(brain.drives.speed_drive()),
            dreamer_loss=float(brain.dreamer.surprise()),
            dreamer_surprise=float(min(1.0, brain.dreamer.surprise() * 5.0)),
            hdc_familiarity=float(hdc_familiarity)
        )

        # ── Cache valid experiences to fuel generative memory loops later ──
        experience_buffer.append((obs.copy(), action.copy(), brain.actor.prev_mu.copy(), brain.prev_value))
        if len(experience_buffer) > 500:
            experience_buffer.pop(0)

        step            += 1
        steps_since_food += 1

        if total_food >= 1500:
            print(f"\n[SAVE] Stopping automatically. total_food={total_food}")
            brain.save("memory/carl_brain")
            save_stats(stats)
            break

except KeyboardInterrupt:
    print(f"\n[SAVE] Stopping. total_food={total_food}, total_wall={total_wall}")
    brain.save("memory/carl_brain")
    save_stats(stats)
    print("[SAVE] Done.")
