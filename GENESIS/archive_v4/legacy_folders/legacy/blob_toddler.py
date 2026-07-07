import time
import math
import numpy as np
import mujoco
import mujoco.viewer
import sys
import os

# Append brain module folder to Python path
sys.path.append(os.path.abspath('../brain'))
sys.path.append(os.path.abspath('brain'))

from blob_brain import BlobBrain, ReflexLayer
from carl_physarum import PhysarumMaze
from carl_grid_cells import HippocampalNavigator, PlaceCellLayer
import blob_telemetry

# Start Telemetry
blob_telemetry.start()

# ── 1. Load the minimal Blob world ─────────────────────────────────────────────
model = mujoco.MjModel.from_xml_path("blob_world.xml")
data = mujoco.MjData(model)

# ── 2. Initialize the Blob's Brain (Toddlerhood Phase 3 - Actor-Critic) ────────
brain = BlobBrain(n_neurons=32, n_inputs=20, n_outputs=2)
brain.load("memory/blob_brain.npz")

from collections import deque
import random

# Spinal cord (Reflexes) starts pre-wired and learns Hebbian wall avoidance
reflex = ReflexLayer(n_sensors=20, n_motors=2, threshold=0.85)
reflex.load("memory/blob_reflex.npy")

# Offline Sleep Replay variables
experience_buffer = deque(maxlen=30)
trauma_memories = []  # Holds up to 5 trauma trajectories
is_sleeping = False
sleep_counter = 0

# Muscle feedback
prev_throttle = 0.0
prev_steering = 0.0

# Grid Cells & Place Cells (Hippocampal GPS Navigator) matching arena bounds
hippo = HippocampalNavigator()
hippo.places = PlaceCellLayer(n_place=200, arena_bounds=(-5.0, 5.0, -5.0, 5.0))
initial_blob_pos = data.geom("blob_geom").xpos.copy()
init_x = float(initial_blob_pos[0])
init_y = float(initial_blob_pos[1])
hippo.reset(init_x, init_y)

# Physarum Slime Mold Pathfinding Grid (30x30 resolution)
physarum = PhysarumMaze(rows=30, cols=30)
MAP_XMIN, MAP_XMAX = -5.0, 5.0
MAP_YMIN, MAP_YMAX = -5.0, 5.0

# Seed the central cylinder obstacle into the Physarum maze
for i in range(physarum.R):
    for j in range(physarum.C):
        xw = MAP_XMIN + (i + 0.5) / physarum.R * (MAP_XMAX - MAP_XMIN)
        yw = MAP_YMIN + (j + 0.5) / physarum.C * (MAP_YMAX - MAP_YMIN)
        if math.hypot(xw, yw) < 0.75:
            physarum.obstacle[i, j] = True

hunger = 0.8  # Start hungry

def get_lidar(n_rays=8):
    blob_pos = data.geom("blob_geom").xpos
    yaw = float(data.qpos[2])  # Joint position 2 is Z-rotation angle (yaw) in radians
    dists = []
    for i in range(n_rays):
        # Ray angles are fired relative to Bob's current local heading
        angle = yaw + i * (2.0 * math.pi / n_rays)
        vec = np.array([math.cos(angle), math.sin(angle), 0.0])
        pnt = blob_pos + np.array([0, 0, 0.05])
        geom_id = np.array([-1], dtype=np.int32)
        dist = mujoco.mj_ray(model, data, pnt, vec, None, 1, -1, geom_id)
        if dist < 0: dist = 5.0
        dists.append(dist)
    return dists

def check_collisions():
    food_pos = data.geom("food").xpos
    blob_pos = data.geom("blob_geom").xpos
    dist_to_food = math.hypot(blob_pos[0] - food_pos[0], blob_pos[1] - food_pos[1])
    if dist_to_food < 0.35: return 'food'
    # Trigger wall collision slightly before hard physical clamp to allow Hebbian learning
    if abs(blob_pos[0]) > 4.68 or abs(blob_pos[1]) > 4.68: return 'wall'
    dist_to_obs = math.hypot(blob_pos[0], blob_pos[1])
    if dist_to_obs < 0.7: return 'wall'
    return None

# ── 3. The Toddlerhood Loop (Operant Conditioning) ─────────────────────────────
print("\nStarting Spatial Mastery (Phase 3)...")
print("The Spinal Reflex Layer is ACTIVE & Hebbian learning.")
print("The Physarum Slime Mold is planning paths around central obstacles.")
print("Open dashboard/blob_dashboard.html in a web browser to watch the emerging mind!")
print("Press Ctrl+C to stop and save.")

food_count = 0
collision_cooldown = 0
steps_since_last_food = 0
episode_number = 1
steps_to_food_history = []

try:
    with mujoco.viewer.launch_passive(model, data) as viewer:
        step = 0
        
        # Calculate initial distance to food
        blob_pos = data.geom("blob_geom").xpos
        food_pos = data.geom("food").xpos
        prev_dist_to_food = math.hypot(blob_pos[0] - food_pos[0], blob_pos[1] - food_pos[1])
        
        while viewer.is_running():
            if is_sleeping:
                # Select a random trauma trajectory from memories
                selected_trajectory = random.choice(trauma_memories)
                
                # Step the brain offline through saved sensors to consolidate
                total_weight_delta = 0.0
                for saved_sensors in selected_trajectory:
                    brain.step(saved_sensors, dt=0.01)
                    weight_change = brain.apply_td_feedback(external_reward=-1.5, learning_rate=0.08)
                    total_weight_delta += abs(weight_change)
                
                # Physical body remains stationary
                data.ctrl[0] = 0.0
                data.ctrl[1] = 0.0
                data.ctrl[2] = 0.0
                
                blob_pos = data.geom("blob_geom").xpos.copy()
                food_pos = data.geom("food").xpos.copy()
                
                # Stream telemetry during sleep
                blob_telemetry.update_state(
                    pos=[float(blob_pos[0]), float(blob_pos[1])],
                    food=[float(food_pos[0]), float(food_pos[1])],
                    hunger=float(hunger),
                    speed=0.0,
                    da=-1.5,
                    ne=0.0,
                    steps_since_food=steps_since_last_food,
                    episode=episode_number,
                    steps_to_food_history=list(steps_to_food_history),
                    neuron_states=brain.state.tolist(),
                    W=brain.W.tolist(),
                    nav_signal=physarum.nav_signal.tolist(),
                    reflex_fired=False,
                    reflex_ratio=float(reflex.reflex_ratio()),
                    novelty=0.0,
                    uncertainty=0.0,
                    is_sleeping=True
                )
                
                # Step MuJoCo physics (keeps the simulation timer ticking)
                mujoco.mj_step(model, data)
                viewer.sync()
                
                sleep_counter -= 1
                if sleep_counter <= 0:
                    is_sleeping = False
                    print("[SYSTEM] Sleep consolidation complete! Waking up.")
                
                time.sleep(0.01)
                continue

            # Ego-centric sensory integration
            lidars = get_lidar()
            proximity = [max(0.0, (3.0 - d) / 3.0) for d in lidars]
            
            blob_pos = data.geom("blob_geom").xpos.copy()
            food_pos = data.geom("food").xpos.copy()
            dist_to_food = math.hypot(food_pos[0] - blob_pos[0], food_pos[1] - blob_pos[1])
            
            # Fetch heading from Physarum Slime Mold Pathfinding
            def to_grid(x, y):
                gi = int(np.clip((x - MAP_XMIN) / (MAP_XMAX - MAP_XMIN) * physarum.R, 0, physarum.R - 1))
                gj = int(np.clip((y - MAP_YMIN) / (MAP_YMAX - MAP_YMIN) * physarum.C, 0, physarum.C - 1))
                return gi, gj
            
            src_cell = to_grid(blob_pos[0], blob_pos[1])
            snk_cell = to_grid(food_pos[0], food_pos[1])
            
            # Step the slime mold
            physarum.step(src_cell, snk_cell, every=10)
            
            # Get optimal world heading
            heading = physarum.get_heading(blob_pos[0], blob_pos[1], food_pos[0], food_pos[1], MAP_XMIN, MAP_XMAX, MAP_YMIN, MAP_YMAX)
            
            # ── A. Compass Target Orientation (Ego-centric Angle) ────────────────────
            yaw = float(data.qpos[2])  # Bob's current Z heading
            relative_target_angle = heading - yaw
            
            # Normalize to [-pi, pi]
            while relative_target_angle > math.pi: relative_target_angle -= 2.0 * math.pi
            while relative_target_angle < -math.pi: relative_target_angle += 2.0 * math.pi
            
            cos_target = math.cos(relative_target_angle)
            sin_target = math.sin(relative_target_angle)
            
            # ── B. Proprioception (Linear & Rotational Speeds) ───────────────────────
            vel_x = data.qvel[0]
            vel_y = data.qvel[1]
            omega_z = data.qvel[2]  # Z rotational velocity
            
            # Project velocity vector into Bob's ego-centric body frame
            v_forward = vel_x * math.cos(yaw) + vel_y * math.sin(yaw)
            v_lateral = -vel_x * math.sin(yaw) + vel_y * math.cos(yaw)
            
            # ── C. Tactile Somatosensory System (Proximity Bumpers) ──────────────────
            touch_front = 1.0 if min(proximity[0], proximity[7]) > 0.85 else 0.0
            touch_right = 1.0 if min(proximity[1], proximity[2]) > 0.85 else 0.0
            touch_back  = 1.0 if min(proximity[3], proximity[4]) > 0.85 else 0.0
            touch_left  = 1.0 if min(proximity[5], proximity[6]) > 0.85 else 0.0
            
            # ── D. Hippocampal GPS dead-reckoning ────────────────────────────────────
            hippo_out = hippo.step(blob_pos[0], blob_pos[1], vel_x, vel_y, dt=0.01, learn=True)
            
            # Compile 20-dimensional Observation Vector
            sensors = np.array(
                proximity +                   # [0-7] Exteroception (Lidar)
                [cos_target, sin_target] +   # [8-9] Compass Guidance
                [hunger] +                    # [10] Internal Metabolism
                [v_forward, omega_z, v_lateral] +  # [11-13] Proprioception (IMU)
                [touch_front, touch_back, touch_left, touch_right] +  # [14-17] Somatosensory
                [prev_throttle, prev_steering],  # [18-19] Muscle Feedback
                dtype=np.float32
            )
            experience_buffer.append(sensors.copy())
            
            # ── E. Cerebral Deliberation (Reservoir Brain) ───────────────────────────
            motor_intention = brain.step(sensors, dt=0.01)
            
            # ── F. Hebbian Spinal Reflex Layer (Spinal Cord) ─────────────────────────
            event = check_collisions()
            ne_signal = 1.0 if event == 'wall' else 0.0
            
            # continuous distance-based reward signal (smell intensity RPE)
            distance_reward = (prev_dist_to_food - dist_to_food) * 2.0
            
            # Spinal cord checks brain output and overrides on looming obstacles
            final_drive, reflex_fired = reflex.step(sensors, motor_intention, distance_reward, ne_signal)
            
            # Introspective Allostatic Regulation: throttle modulated by spatial uncertainty
            allostatic_speed_modulation = max(0.2, 1.0 - (hippo_out['uncertainty'] * 0.6))
            speed_multiplier = (1.0 + hunger) * allostatic_speed_modulation
            
            throttle = final_drive[0] * speed_multiplier
            steering = final_drive[1]
            
            # Map the brain's ego-centric throttle and steering commands to world joints
            force_x = throttle * math.cos(yaw) * 10.0  # Slide X actuator (gear=10)
            force_y = throttle * math.sin(yaw) * 10.0  # Slide Y actuator (gear=10)
            torque_z = steering * 4.0                  # Hinge Z actuator (gear=2)
            
            data.ctrl[0] = force_x
            data.ctrl[1] = force_y
            data.ctrl[2] = torque_z
            
            # Cache muscle outputs for next observation frame
            prev_throttle = throttle
            prev_steering = steering
            
            # ── G. Basal Ganglia Continuous Online TD Learning ───────────────────────
            environmental_reward = distance_reward
            if event == 'food':
                environmental_reward += 2.0
            elif event == 'wall':
                environmental_reward -= 1.5
                
            da_signal = brain.apply_td_feedback(environmental_reward, learning_rate=0.0005)
            
            if collision_cooldown > 0:
                collision_cooldown -= 1
            else:
                if event == 'food':
                    weight_change = brain.apply_td_feedback(2.0, learning_rate=0.1)
                    hunger = max(0.0, hunger - 0.5)
                    food_count += 1
                    collision_cooldown = 40  # Prevent spamming
                    da_signal = 2.0
                    
                    print(f"[JOY] Food eaten! Episode {episode_number}: reached food in {steps_since_last_food} frames. Weight delta: +{weight_change:.4f}. Hunger: {hunger:.2f}")
                    
                    # If satiated and has trauma memories, trigger Offline Sleep Replay
                    if hunger <= 0.5 and len(trauma_memories) > 0:
                        is_sleeping = True
                        sleep_counter = 120
                        print(f"[SYSTEM] Hunger satisfied! Entering Offline Sleep Replay (Hippocampal Consolidation) with {len(trauma_memories)} memories...")
                    
                    # Record performance statistics
                    steps_to_food_history.append(steps_since_last_food)
                    if len(steps_to_food_history) > 20:
                        steps_to_food_history.pop(0)
                        
                    steps_since_last_food = 0
                    episode_number += 1
                    
                    # Save both brain & reflex memory
                    brain.save("memory/blob_brain.npz")
                    reflex.save("memory/blob_reflex.npy")
                    
                    # Respawn food randomly
                    food_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "food")
                    model.geom_pos[food_id][0] = np.random.uniform(-4.0, 4.0)
                    model.geom_pos[food_id][1] = np.random.uniform(-4.0, 4.0)
                    
                    # Recalculate distance to new food
                    blob_pos = data.geom("blob_geom").xpos
                    food_pos = data.geom("food").xpos
                    prev_dist_to_food = math.hypot(blob_pos[0] - food_pos[0], blob_pos[1] - food_pos[1])
                    
                elif event == 'wall':
                    weight_change = brain.apply_td_feedback(-2.0, learning_rate=0.1)
                    brain.startle(magnitude=2.0)
                    collision_cooldown = 40  # Prevent spamming and loops
                    da_signal = -2.0
                    
                    # Log trauma trajectory into Hippocampal memories
                    if len(experience_buffer) > 0:
                        trauma_memories.append(list(experience_buffer))
                        if len(trauma_memories) > 5:
                            trauma_memories.pop(0)
                    
                    print(f"[PAIN] Hit a wall. Flinched! Weight delta: {weight_change:.4f}. Trauma memory stored! Total trauma memories: {len(trauma_memories)}")
            
            # Save distance for next RPE step
            prev_dist_to_food = dist_to_food
            hunger = min(1.0, hunger + 0.0005)
            
            # Stream high-fidelity state data to telemetry
            blob_telemetry.update_state(
                pos=[float(blob_pos[0]), float(blob_pos[1])],
                food=[float(food_pos[0]), float(food_pos[1])],
                hunger=float(hunger),
                speed=float(math.hypot(vel_x, vel_y)),
                da=float(da_signal),
                ne=float(ne_signal),
                steps_since_food=steps_since_last_food,
                episode=episode_number,
                steps_to_food_history=list(steps_to_food_history),
                neuron_states=brain.state.tolist(),
                W=brain.W.tolist(),
                nav_signal=physarum.nav_signal.tolist(),
                reflex_fired=bool(reflex_fired),
                reflex_ratio=float(reflex.reflex_ratio()),
                novelty=float(hippo_out['novelty']),
                uncertainty=float(hippo_out['uncertainty']),
                is_sleeping=False
            )
            
            mujoco.mj_step(model, data)
            
            # ── H. Hard Absolute Coordinate Clamp (Prevent high-speed physics tunneling) ──
            # Calculate current absolute world coordinates
            abs_x = init_x + data.qpos[0]
            abs_y = init_y + data.qpos[1]
            
            # Arena boundaries are at absolute +/-4.9. With radius 0.2, center must stay within [-4.7, 4.7].
            clamped_abs_x = np.clip(abs_x, -4.7, 4.7)
            clamped_abs_y = np.clip(abs_y, -4.7, 4.7)
            
            if abs_x != clamped_abs_x or abs_y != clamped_abs_y:
                data.qpos[0] = clamped_abs_x - init_x
                data.qpos[1] = clamped_abs_y - init_y
                # Damp velocity on hard boundary impact to absorb the shock
                data.qvel[0] *= -0.1
                data.qvel[1] *= -0.1
                # Recompute derived physics quantities (geom positions, contacts) immediately
                mujoco.mj_forward(model, data)

            viewer.sync()
            
            step += 1
            steps_since_last_food += 1
            time.sleep(0.01)

except KeyboardInterrupt:
    print(f"\n[INFO] Spatial Mastery training stopped manually. Total food eaten: {food_count}")
    brain.save("memory/blob_brain.npz")
    reflex.save("memory/blob_reflex.npy")
    print("[INFO] Brain and reflex states successfully saved. Ready for Phase 4.")
