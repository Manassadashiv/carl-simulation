import math
import numpy as np
import mujoco
import os
import sys

from carl_agent import CarlBrain

def get_yaw(quat):
    w, x, y, z = quat
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

def set_yaw(yaw):
    return [math.cos(yaw/2.0), 0.0, 0.0, math.sin(yaw/2.0)]

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
        data.ctrl[0] = np.clip(left_w, -max_actuator_limit, max_actuator_limit)
        data.ctrl[1] = np.clip(right_w, -max_actuator_limit, max_actuator_limit)
        
    return strength

model = mujoco.MjModel.from_xml_path("vessel_kinetic.xml")
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

# Resolve CARL's freejoint qpos base address (robust to XML ordering changes)
Q = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'spatial_identity')]
V = model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'spatial_identity')]

brain = CarlBrain(n_obs=24)
brain.load("memory/carl_brain")


print("\n============================================================")
print("  CARL Evaluation Run")
print("  100 Episodes. Pure Exploitation (No Noise). Fully Random Spawns.")
print("============================================================\n")

test_episodes = 100
ep_count = 0
success_count = 0
wall_count = 0
timeout_count = 0
step_count = 0
prev_throttle = 0.0
prev_steering = 0.0

while ep_count < test_episodes:
    if step_count == 0:
        while True:
            cx = np.random.uniform(-4.0, 4.0)
            cy = np.random.uniform(-4.0, 4.0)
            if math.hypot(cx, cy) > 1.2:
                break
        data.qpos[Q]   = cx
        data.qpos[Q+1] = cy
        data.qpos[Q+2] = 0.05
        rand_yaw = np.random.uniform(-math.pi, math.pi)
        data.qpos[Q+3:Q+7] = set_yaw(rand_yaw)
        data.qvel[:] = 0.0

        while True:
            fx = np.random.uniform(-4.0, 4.0)
            fy = np.random.uniform(-4.0, 4.0)
            if math.hypot(fx, fy) > 1.0 and math.hypot(fx - cx, fy - cy) > 1.0:
                break
        food_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "food")
        model.geom_pos[food_id][0] = fx
        model.geom_pos[food_id][1] = fy
        
        mujoco.mj_forward(model, data)

    # Sensing
    blob_pos = data.geom("chassis").xpos
    yaw_real = get_yaw(data.qpos[Q+3:Q+7])
    
    food_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "food")
    old_food_pos = model.geom_pos[food_id].copy()
    model.geom_pos[food_id] = np.array([99.0, 99.0, 99.0])
    
    dists = []
    for i in range(8):
        angle = yaw_real + i * (2.0 * math.pi / 8.0)
        vx = math.cos(angle)
        vy = math.sin(angle)
        vec = np.array([vx, vy, 0.0])
        pnt = np.array([blob_pos[0] + vx * 0.22, blob_pos[1] + vy * 0.22, blob_pos[2] + 0.05])
        geom_hit = np.array([-1], dtype=np.int32)
        dist = mujoco.mj_ray(model, data, pnt, vec, None, 1, -1, geom_hit)
        dists.append((dist + 0.22) if dist > 0 else 5.0)
        
    model.geom_pos[food_id] = old_food_pos
        
    proximity = [max(0.0, (2.5 - d) / 2.5) for d in dists]
    
    vel_x = float(data.qvel[0])
    vel_y = float(data.qvel[1])
    vel_fwd = vel_x * math.cos(yaw_real) + vel_y * math.sin(yaw_real)
    vel_lat = -vel_x * math.sin(yaw_real) + vel_y * math.cos(yaw_real)
    omega = float(data.qvel[2])

    food_pos = data.geom("food").xpos
    dx = food_pos[0] - blob_pos[0]
    dy = food_pos[1] - blob_pos[1]
    world_angle = math.atan2(dy, dx)
    rel = world_angle - yaw_real
    while rel > math.pi: rel -= 2*math.pi
    while rel < -math.pi: rel += 2*math.pi
    food_cos, food_sin = math.cos(rel), math.sin(rel)

    touch_front = 1.0 if max(proximity[0], proximity[7], proximity[1]) > 0.6 else 0.0
    touch_back = 1.0 if max(proximity[3], proximity[4], proximity[5]) > 0.6 else 0.0
    touch_left = 1.0 if proximity[6] > 0.6 else 0.0
    touch_right = 1.0 if proximity[2] > 0.6 else 0.0

    obs_raw = np.array(list(proximity) + [vel_fwd, vel_lat, omega, food_cos, food_sin] + list(brain.drives.to_vec())[:5] + [touch_front, touch_back, touch_left, touch_right, prev_throttle, prev_steering], dtype=np.float32)
    
    std = np.sqrt(brain._obs_var / max(1, brain._obs_n) + 1e-8)
    obs = (obs_raw - brain._obs_mean) / std

    # Thinking
    raw_action = brain.actor.forward(obs)
    
    # Spinal Reflex
    action, reflex_fired = brain.spinal.check(proximity, raw_action, brain.drives)
    throttle, steering = float(action[0]), float(action[1])
    
    # Decay drives
    brain.drives.decay()

    # ── Acting ────────────────────────────────────────────────────────────
    throttle, steering = float(action[0]), float(action[1])
    data.ctrl[0] = throttle - steering  # w_left
    data.ctrl[1] = throttle + steering  # w_right

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
    
    food_dx, food_dy = math.cos(yaw_real) * dx, math.sin(yaw_real) * dy
    food_angle = math.atan2(food_dy, food_dx) - yaw_real
    head_pan = food_angle + (raw_noise_x * 0.35)

    data.ctrl[2] = np.clip(hip_pitch, -0.55, 0.55)
    data.ctrl[3] = np.clip(hip_roll, -0.38, 0.38)
    data.ctrl[4] = np.clip(neck_sweep, -0.6, 0.7)
    data.ctrl[5] = np.clip(head_pan, -1.4, 1.4)
    # Arms: shoulder posture drives both shoulders, elbow follows
    elbow_posture = np.clip(arm_posture * 0.6 - 0.5, -2.44, 0)  # elbow tracks shoulder
    data.ctrl[6]  = np.clip(arm_posture, -2.0, 2.0)    # shoulder_L
    data.ctrl[7]  = elbow_posture                        # elbow_L
    data.ctrl[8]  = 0.0                                  # wrist_L (level)
    data.ctrl[9]  = 0.0                                  # grip_L (open)
    data.ctrl[10] = np.clip(arm_posture, -2.0, 2.0)    # shoulder_R
    data.ctrl[11] = elbow_posture                        # elbow_R
    data.ctrl[12] = 0.0                                  # wrist_R (level)
    data.ctrl[13] = 0.0                                  # grip_R (open)

    prev_throttle = throttle
    prev_steering = steering

    # Reflex
    wall_proximity_override(yaw_real)

    # Physics
    mujoco.mj_step(model, data)
    step_count += 1

    # Clamp
    abs_x = data.qpos[Q]
    abs_y = data.qpos[Q+1]
    cx = np.clip(abs_x, -4.65, 4.65)
    cy = np.clip(abs_y, -4.65, 4.65)
    if (abs_x != cx) or (abs_y != cy):
        data.qpos[Q]   = cx
        data.qpos[Q+1] = cy
        if abs(abs_x) > 4.65: data.qvel[V]   = -data.qvel[V]   * 0.2
        if abs(abs_y) > 4.65: data.qvel[V+1] = -data.qvel[V+1] * 0.2
        mujoco.mj_forward(model, data)

    # Event Checking
    dist = math.hypot(dx, dy)
    event = None
    if dist < 0.35:
        event = 'food'
    elif abs(data.qpos[Q]) > 4.65 or abs(data.qpos[Q+1]) > 4.65:
        event = 'wall'
    elif math.hypot(blob_pos[0], blob_pos[1]) < 0.72:
        event = 'wall'

    if event == 'food':
        success_count += 1
        ep_count += 1
        print(f"[SUCCESS] Ep {ep_count:03d}/{test_episodes} | Found food in {step_count} steps.")
        step_count = 0
    elif event == 'wall':
        wall_count += 1
        ep_count += 1
        print(f"[FAIL] Ep {ep_count:03d}/{test_episodes} | Hit wall in {step_count} steps.")
        step_count = 0
    elif step_count > 5000:
        timeout_count += 1
        ep_count += 1
        print(f"[TIMEOUT] Ep {ep_count:03d}/{test_episodes} | Timed out after 5000 steps.")
        step_count = 0

print("\\n============================================================")
print("  STRESS TEST COMPLETE")
print(f"  Total Episodes : {test_episodes}")
print(f"  Successes      : {success_count} ({(success_count/test_episodes)*100:.1f}%)")
print(f"  Wall Hits      : {wall_count} ({(wall_count/test_episodes)*100:.1f}%)")
print(f"  Timeouts       : {timeout_count} ({(timeout_count/test_episodes)*100:.1f}%)")
print("============================================================\\n")
