"""
blob_trainer.py  — Headless autonomous training loop for Bob
No MuJoCo viewer required. Runs forever, monitors itself, auto-saves.
"""
import time
import math
import numpy as np
import mujoco
import sys
import os
import json

sys.path.append(os.path.abspath('../brain'))
sys.path.append(os.path.abspath('brain'))

from blob_brain import BlobBrain, ReflexLayer
from carl_physarum import PhysarumMaze
from carl_grid_cells import HippocampalNavigator, PlaceCellLayer
import blob_telemetry

blob_telemetry.start()

# ── 1. Load the minimal Blob world ─────────────────────────────────────────────
model = mujoco.MjModel.from_xml_path("blob_world.xml")
data  = mujoco.MjData(model)

# ── 2. Brain + Reflex  ──────────────────────────────────────────────────────────
brain  = BlobBrain(n_neurons=32, n_inputs=20, n_outputs=2)
brain.load("memory/blob_brain.npz")

reflex = ReflexLayer(n_sensors=20, n_motors=2, threshold=0.5)   # lower threshold → fires earlier
reflex.load("memory/blob_reflex.npy")

from collections import deque
import random

experience_buffer = deque(maxlen=30)
trauma_memories   = []
is_sleeping       = False
sleep_counter     = 0

prev_throttle = 0.0
prev_steering = 0.0

hippo  = HippocampalNavigator()
hippo.places = PlaceCellLayer(n_place=200, arena_bounds=(-5.0, 5.0, -5.0, 5.0))
# CRITICAL: Reset to default state and run forward kinematics BEFORE reading any positions
mujoco.mj_resetData(model, data)
mujoco.mj_forward(model, data)

initial_blob_pos = data.geom("blob_geom").xpos.copy()
init_x = float(initial_blob_pos[0])
init_y = float(initial_blob_pos[1])
hippo.reset(init_x, init_y)

print(f"[INIT] Bob's true world spawn: ({init_x:.3f}, {init_y:.3f})")

physarum = PhysarumMaze(rows=30, cols=30)
MAP_XMIN, MAP_XMAX = -5.0, 5.0
MAP_YMIN, MAP_YMAX = -5.0, 5.0

for i in range(physarum.R):
    for j in range(physarum.C):
        xw = MAP_XMIN + (i + 0.5) / physarum.R * (MAP_XMAX - MAP_XMIN)
        yw = MAP_YMIN + (j + 0.5) / physarum.C * (MAP_YMAX - MAP_YMIN)
        if math.hypot(xw, yw) < 0.75:
            physarum.obstacle[i, j] = True

hunger = 0.8


# ── 3. Helpers ──────────────────────────────────────────────────────────────────
def get_lidar(n_rays=8):
    blob_pos = data.geom("blob_geom").xpos
    yaw = float(data.qpos[2])
    dists = []
    for i in range(n_rays):
        angle = yaw + i * (2.0 * math.pi / n_rays)
        vec   = np.array([math.cos(angle), math.sin(angle), 0.0])
        pnt   = blob_pos + np.array([0, 0, 0.05])
        geom_id = np.array([-1], dtype=np.int32)
        dist  = mujoco.mj_ray(model, data, pnt, vec, None, 1, -1, geom_id)
        if dist < 0: dist = 5.0
        dists.append(dist)
    return dists


def check_collisions():
    food_pos  = data.geom("food").xpos
    blob_pos  = data.geom("blob_geom").xpos
    dist_food = math.hypot(blob_pos[0]-food_pos[0], blob_pos[1]-food_pos[1])
    if dist_food < 0.35: return 'food'
    # Outer walls: geom xpos is absolute world coordinate
    if abs(blob_pos[0]) > 4.68 or abs(blob_pos[1]) > 4.68: return 'wall'
    # Center obstacle: only count if Bob has moved away from spawn (avoid false positives)
    if math.hypot(blob_pos[0], blob_pos[1]) < 0.7:
        # Make sure this isn't just the spawn corner (spawn is at -2.5,-2.5)
        dist_from_spawn = math.hypot(blob_pos[0]-init_x, blob_pos[1]-init_y)
        if dist_from_spawn > 0.5:   # Only count if Bob has actually moved
            return 'wall'
    return None


def respawn_food():
    food_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "food")
    # Keep re-trying until it's far from the obstacle and the blob
    blob_pos = data.geom("blob_geom").xpos
    for _ in range(100):
        fx = np.random.uniform(-4.0, 4.0)
        fy = np.random.uniform(-4.0, 4.0)
        if math.hypot(fx, fy) > 1.2 and math.hypot(fx-blob_pos[0], fy-blob_pos[1]) > 1.5:
            model.geom_pos[food_id][0] = fx
            model.geom_pos[food_id][1] = fy
            break


def to_grid(x, y):
    gi = int(np.clip((x - MAP_XMIN) / (MAP_XMAX - MAP_XMIN) * physarum.R, 0, physarum.R-1))
    gj = int(np.clip((y - MAP_YMIN) / (MAP_YMAX - MAP_YMIN) * physarum.C, 0, physarum.C-1))
    return gi, gj


# ── 4. Stats log file ───────────────────────────────────────────────────────────
STATS_FILE = "memory/training_stats.json"

def save_stats(stats):
    with open(STATS_FILE, "w") as f:
        json.dump(stats, f, indent=2)


stats = {
    "episodes": [],
    "total_food": 0,
    "total_wall_hits": 0,
    "start_time": time.strftime("%Y-%m-%d %H:%M:%S")
}


# ── 5. Main training loop ───────────────────────────────────────────────────────
print("\n" + "="*56)
print("  CARL * Autonomous Headless Training * Bob v3.0")
print("  Press Ctrl+C at any time to save and exit")
print("="*56 + "\n")

food_count          = 0
wall_count          = 0
collision_cooldown  = 200   # Big cooldown at start — let Bob settle first!
steps_since_food    = 0
episode_number      = 1
steps_to_food_hist  = []

blob_pos  = data.geom("blob_geom").xpos.copy()
food_pos  = data.geom("food").xpos.copy()

# Ensure food starts in a clear area away from spawn corner and center obstacle
respawn_food()
mujoco.mj_forward(model, data)   # recalculate geom positions after food move

blob_pos  = data.geom("blob_geom").xpos.copy()
food_pos  = data.geom("food").xpos.copy()
prev_dist = math.hypot(blob_pos[0]-food_pos[0], blob_pos[1]-food_pos[1])

print(f"Bob starts at ({init_x:.2f}, {init_y:.2f})")
print(f"Food starts at ({food_pos[0]:.2f}, {food_pos[1]:.2f})")

# Max steps per episode before we give up and respawn food nearby
MAX_EPISODE_STEPS = 6000

try:
    step = 0
    while True:

        # ── Sleep / Hippocampal Consolidation ────────────────────────────────
        if is_sleeping:
            if len(trauma_memories) > 0:
                traj = random.choice(trauma_memories)
                for saved_sensors in traj:
                    brain.step(saved_sensors, dt=0.01)
                    brain.apply_td_feedback(-1.5, learning_rate=0.01)

            data.ctrl[:] = 0.0
            mujoco.mj_step(model, data)

            blob_pos = data.geom("blob_geom").xpos.copy()
            food_pos = data.geom("food").xpos.copy()

            blob_telemetry.update_state(
                pos=[float(blob_pos[0]), float(blob_pos[1])],
                food=[float(food_pos[0]), float(food_pos[1])],
                hunger=float(hunger), speed=0.0, da=-1.5, ne=0.0,
                steps_since_food=steps_since_food, episode=episode_number,
                steps_to_food_history=list(steps_to_food_hist),
                neuron_states=brain.state.tolist(), W=brain.W.tolist(),
                nav_signal=physarum.nav_signal.tolist(),
                reflex_fired=False, reflex_ratio=float(reflex.reflex_ratio()),
                novelty=0.0, uncertainty=0.0, is_sleeping=True
            )

            sleep_counter -= 1
            if sleep_counter <= 0:
                is_sleeping = False
                print("[SYSTEM] Sleep consolidation done. Waking up.")
            time.sleep(0.005)
            continue

        # ── Lidar Sensing ─────────────────────────────────────────────────────
        lidars    = get_lidar()
        # Scale: 1.0 = wall is RIGHT THERE (0m), 0.0 = far (>=2m)
        # Use 2m as saturation distance for sharper near-field signal
        proximity = [max(0.0, (2.0 - d) / 2.0) for d in lidars]

        blob_pos  = data.geom("blob_geom").xpos.copy()
        food_pos  = data.geom("food").xpos.copy()
        dist_food = math.hypot(food_pos[0]-blob_pos[0], food_pos[1]-blob_pos[1])

        # ── Physarum path heading ─────────────────────────────────────────────
        src_cell = to_grid(blob_pos[0], blob_pos[1])
        snk_cell = to_grid(food_pos[0], food_pos[1])
        physarum.step(src_cell, snk_cell, every=10)
        heading = physarum.get_heading(blob_pos[0], blob_pos[1],
                                       food_pos[0], food_pos[1],
                                       MAP_XMIN, MAP_XMAX, MAP_YMIN, MAP_YMAX)

        # ── Ego-centric target angle ──────────────────────────────────────────
        yaw = float(data.qpos[2])
        rel_angle = heading - yaw
        while rel_angle >  math.pi: rel_angle -= 2*math.pi
        while rel_angle < -math.pi: rel_angle += 2*math.pi
        cos_t = math.cos(rel_angle)
        sin_t = math.sin(rel_angle)

        # ── Proprioception ────────────────────────────────────────────────────
        vel_x    = data.qvel[0]
        vel_y    = data.qvel[1]
        omega_z  = data.qvel[2]
        v_fwd    =  vel_x * math.cos(yaw) + vel_y * math.sin(yaw)
        v_lat    = -vel_x * math.sin(yaw) + vel_y * math.cos(yaw)

        # ── Tactile bumpers (more sensitive: 0.7 instead of 0.85) ────────────
        touch_front = 1.0 if max(proximity[0], proximity[7]) > 0.65 else 0.0
        touch_right = 1.0 if max(proximity[6], proximity[7]) > 0.65 else 0.0
        touch_back  = 1.0 if max(proximity[3], proximity[4]) > 0.65 else 0.0
        touch_left  = 1.0 if max(proximity[1], proximity[2]) > 0.65 else 0.0

        # ── Hippocampal GPS ───────────────────────────────────────────────────
        hippo_out = hippo.step(blob_pos[0], blob_pos[1], vel_x, vel_y, dt=0.01, learn=True)

        # ── 20-dim observation vector ─────────────────────────────────────────
        sensors = np.array(
            proximity +
            [cos_t, sin_t] +
            [hunger] +
            [v_fwd, omega_z, v_lat] +
            [touch_front, touch_back, touch_left, touch_right] +
            [prev_throttle, prev_steering],
            dtype=np.float32
        )
        experience_buffer.append(sensors.copy())

        # ── Brain deliberation ────────────────────────────────────────────────
        motor_intention = brain.step(sensors, dt=0.01)

        # ── Spinal cord override ──────────────────────────────────────────────
        event     = check_collisions()
        ne_signal = 1.0 if event == 'wall' else 0.0

        distance_reward = (prev_dist - dist_food) * 3.0  # stronger approach signal

        final_drive, reflex_fired = reflex.step(sensors, motor_intention, distance_reward, ne_signal)

        # ── Speed control ─────────────────────────────────────────────────────
        # Cap multiplier: hunger boost is mild, no allostatic over-amplification
        # Also: if ANY front-arc lidar is very close, force-cap speed to 30%
        front_danger = max(proximity[0], proximity[7], proximity[1])
        safe_speed   = 0.3 if front_danger > 0.7 else 1.0
        speed_mult   = min(1.4, (1.0 + hunger * 0.4)) * safe_speed

        throttle = final_drive[0] * speed_mult
        steering = final_drive[1]

        # Force-steer away from walls if very close (overrides brain if danger)
        # This is a direct reflex bypass — most biologically, this is brainstem
        if proximity[0] > 0.8:                  # wall directly ahead → hard reverse
            throttle = -0.6
        if proximity[7] > 0.75 or proximity[1] > 0.75:  # front-right or front-left
            throttle = min(throttle, 0.0)        # no forward motion allowed

        # ── Actuators ─────────────────────────────────────────────────────────
        force_x  = throttle * math.cos(yaw) * 5.0
        force_y  = throttle * math.sin(yaw) * 5.0
        torque_z = steering * 4.0

        data.ctrl[0] = force_x
        data.ctrl[1] = force_y
        data.ctrl[2] = torque_z

        prev_throttle = throttle
        prev_steering = steering

        # ── Continuous TD learning ────────────────────────────────────────────
        env_reward = distance_reward
        if event == 'food': env_reward += 3.0
        elif event == 'wall': env_reward -= 1.0
        da_signal = brain.apply_td_feedback(env_reward, learning_rate=0.001)

        # ── Event handling ────────────────────────────────────────────────────
        if collision_cooldown > 0:
            collision_cooldown -= 1
        else:
            if event == 'food':
                weight_change = brain.apply_td_feedback(3.0, learning_rate=0.05)
                hunger        = max(0.0, hunger - 0.5)
                food_count   += 1
                collision_cooldown = 30
                da_signal     = 3.0

                steps_to_food_hist.append(steps_since_food)
                if len(steps_to_food_hist) > 30:
                    steps_to_food_hist.pop(0)

                recent_avg = int(np.mean(steps_to_food_hist[-5:])) if len(steps_to_food_hist) >= 5 else steps_since_food
                print(f"[JOY] Ep {episode_number}: food in {steps_since_food} steps | "
                      f"recent avg={recent_avg} | hunger={hunger:.2f}")

                stats["total_food"] = food_count
                stats["episodes"].append({
                    "ep": episode_number,
                    "steps": steps_since_food,
                    "hunger": float(hunger)
                })
                save_stats(stats)

                if hunger <= 0.5 and len(trauma_memories) > 0:
                    is_sleeping   = True
                    sleep_counter = 80
                    print(f"[SLEEP] Entering consolidation with {len(trauma_memories)} memories...")

                steps_since_food = 0
                episode_number  += 1
                brain.save("memory/blob_brain.npz")
                reflex.save("memory/blob_reflex.npy")
                respawn_food()
                blob_pos  = data.geom("blob_geom").xpos.copy()
                food_pos  = data.geom("food").xpos.copy()
                prev_dist = math.hypot(blob_pos[0]-food_pos[0], blob_pos[1]-food_pos[1])

            elif event == 'wall':
                # NO startle — that was the big bug. Instead, just a gentle velocity kill.
                data.qvel[0] *= 0.1
                data.qvel[1] *= 0.1
                weight_change  = brain.apply_td_feedback(-2.0, learning_rate=0.02)
                collision_cooldown = 30
                da_signal      = -2.0
                wall_count    += 1
                stats["total_wall_hits"] = wall_count

                if len(experience_buffer) > 0:
                    trauma_memories.append(list(experience_buffer))
                    if len(trauma_memories) > 5:
                        trauma_memories.pop(0)

                print(f"[PAIN] Wall hit #{wall_count}. Trauma stored ({len(trauma_memories)}/5). "
                      f"TD={weight_change:.3f}")

        # ── Episode timeout — respawn food if taking forever ─────────────────
        if steps_since_food > MAX_EPISODE_STEPS:
            print(f"[TIMEOUT] Episode {episode_number} timed out at {steps_since_food} steps. Respawning food.")
            respawn_food()
            steps_since_food = 0
            blob_pos  = data.geom("blob_geom").xpos.copy()
            food_pos  = data.geom("food").xpos.copy()
            prev_dist = math.hypot(blob_pos[0]-food_pos[0], blob_pos[1]-food_pos[1])

        prev_dist = dist_food
        hunger    = min(1.0, hunger + 0.0005)

        # ── Telemetry ─────────────────────────────────────────────────────────
        blob_telemetry.update_state(
            pos=[float(blob_pos[0]), float(blob_pos[1])],
            food=[float(food_pos[0]), float(food_pos[1])],
            hunger=float(hunger),
            speed=float(math.hypot(vel_x, vel_y)),
            da=float(da_signal), ne=float(ne_signal),
            steps_since_food=steps_since_food, episode=episode_number,
            steps_to_food_history=list(steps_to_food_hist),
            neuron_states=brain.state.tolist(), W=brain.W.tolist(),
            nav_signal=physarum.nav_signal.tolist(),
            reflex_fired=bool(reflex_fired),
            reflex_ratio=float(reflex.reflex_ratio()),
            novelty=float(hippo_out['novelty']),
            uncertainty=float(hippo_out['uncertainty']),
            is_sleeping=False
        )

        # ── Physics step + clamp ──────────────────────────────────────────────
        mujoco.mj_step(model, data)

        abs_x = init_x + data.qpos[0]
        abs_y = init_y + data.qpos[1]
        cx = np.clip(abs_x, -4.7, 4.7)
        cy = np.clip(abs_y, -4.7, 4.7)
        if abs_x != cx or abs_y != cy:
            data.qpos[0] = cx - init_x
            data.qpos[1] = cy - init_y
            # Hard repulsion: fire velocity AWAY from nearest wall
            if abs_x > cx:  data.qvel[0] = -abs(data.qvel[0]) * 0.3
            elif abs_x < cx: data.qvel[0] =  abs(data.qvel[0]) * 0.3
            else: data.qvel[0] *= 0.0
            if abs_y > cy:  data.qvel[1] = -abs(data.qvel[1]) * 0.3
            elif abs_y < cy: data.qvel[1] =  abs(data.qvel[1]) * 0.3
            else: data.qvel[1] *= 0.0
            mujoco.mj_forward(model, data)

        step            += 1
        steps_since_food += 1
        # No sleep — run as fast as possible for training throughput

except KeyboardInterrupt:
    print(f"\n[SAVE] Saving... total food={food_count}, total wall hits={wall_count}")
    brain.save("memory/blob_brain.npz")
    reflex.save("memory/blob_reflex.npy")
    save_stats(stats)
    print("[SAVE] Done. Bye!")
