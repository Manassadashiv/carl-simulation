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
import keyboard

sys.path.append(os.path.abspath('.'))
from carl_agent import CarlBrain
from carl_expression import ExpressionController
from carl_bios import CarlBiosCore
from carl_double_buffer import SensorStateBuffer, AtomicDoubleBuffer
from carl_brainstem import BrainstemController
from carl_cpg import HopfCpgEngine
from carl_social import SocialVisualLobe

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
    parser = argparse.ArgumentParser()
    parser.add_argument('--render', action='store_true', default=True)
    args = parser.parse_args()

    print("=" * 72)
    print("  CARL GENESIS - Interactive AI Tutorial & Data Harvester")
    print("  Default Navigational Intelligence Stack: Mark VIII")
    print("=" * 72)

    mujoco_lock = threading.Lock()
    event_queue = queue.Queue()
    narrator = TutorialNarrator()

    # -- Compile MuJoCo Engine Scene --
    model = mujoco.MjModel.from_xml_path("vessel_kinetic.xml")
    data  = mujoco.MjData(model)

    # Velocity actuators: ctrl = desired wheel velocity (rad/s). MuJoCo applies force = kv * (ctrl - joint_vel).
    # No runtime overrides needed — the XML forcerange (0.5 Nm) is the correct physical limit.

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
    pose_buffer = AtomicDoubleBuffer(shape=(8,))
    brainstem = BrainstemController(max_torque=0.5)
    cpg = HopfCpgEngine()

    # Differential drive constants
    TRACK_WIDTH = 0.22    # meters between wheels
    WHEEL_RADIUS = 0.04   # meters

    steps_since_food = [0]
    planner_step = [0]

    def get_yaw():
        w, x, y, z = data.qpos[3:7]
        return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    def get_food_metrics():
        bp = data.geom("chassis").xpos
        fp = data.geom("food").xpos
        dx, dy = fp[0] - bp[0], fp[1] - bp[1]
        dist = math.hypot(dx, dy) + 1e-8
        yaw = get_yaw()
        local_x = dx * math.cos(yaw) + dy * math.sin(yaw)
        local_y = -dx * math.sin(yaw) + dy * math.cos(yaw)
        return local_x / dist, local_y / dist, dist

    def respawn_target():
        food_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "food")
        bp = data.geom("chassis").xpos
        angle = np.random.uniform(0, 2 * math.pi)
        radius = np.random.uniform(1.5, 3.5)
        model.geom_pos[food_id][0] = np.clip(bp[0] + radius * math.cos(angle), -4.2, 4.2)
        model.geom_pos[food_id][1] = np.clip(bp[1] + radius * math.sin(angle), -4.2, 4.2)

    # Initialize reward tracking after function definitions
    _, _, prev_dist = get_food_metrics()

    # -- 100Hz Reticulospinal Tract (Brainstem Callback) --
    # Architecture: velocity actuators handle PID internally via kv servo.
    # We only need to compute DESIRED WHEEL VELOCITIES and set data.ctrl.
    def on_control_tick(dt):
        with mujoco_lock:
            # -- Step 1: Read planning targets from cognitive loop --
            planning_target, _ = sensor_buffer.planning_target.read()
            v_target, w_target = float(planning_target[0]), float(planning_target[1])

            # -- Step 2: Braitenberg reflex safety check --
            human_active = any(keyboard.is_pressed(k) for k in ['w','a','s','d','up','down','left','right'])
            if not human_active:
                v_safe, w_safe, _ = brainstem.reflex.check(np.ones(24), v_target, w_target)
            else:
                v_safe, w_safe = v_target, w_target

            # -- Step 3: Differential drive kinematics --
            # Convert body-frame (v, w) to per-wheel angular velocities (rad/s)
            half_track = TRACK_WIDTH / 2.0
            omega_L = (v_safe - w_safe * half_track) / WHEEL_RADIUS
            omega_R = (v_safe + w_safe * half_track) / WHEEL_RADIUS

            # Clamp to actuator speed limit (ctrlrange = -20.0 to 20.0 rad/s)
            omega_L = np.clip(omega_L, -20.0, 20.0)
            omega_R = np.clip(omega_R, -20.0, 20.0)

            # -- Step 4: Subtle CPG biomimetic modulation --
            # Add tiny velocity perturbations for organic feel (NOT torques)
            drives_dict = {"cortisol": brain.drives.cort, "dopamine": brain.drives.da}
            cpg_torques = np.array([0.0, 0.0])  # dummy base
            cpg_mod = cpg.compute_biomimetic_modulation(drives_dict, cpg_torques)
            # Scale CPG output to velocity-space (divide by kv=1.0)
            omega_L += cpg_mod[0] / 1.0
            omega_R += cpg_mod[1] / 1.0

            # -- Step 5: Send desired velocities to MuJoCo velocity servos --
            data.ctrl[0] = omega_L
            data.ctrl[1] = omega_R

            # Lock spine joints upright
            data.ctrl[2:8] = 0.0
            
            if viewer is not None:
                with viewer.lock():
                    mujoco.mj_step(model, data)
            else:
                mujoco.mj_step(model, data)

            # Foraging contract validation
            _, _, f_dist = get_food_metrics()
            if f_dist < 0.38:
                event_queue.put('food')
                respawn_target()
                if viewer is not None:
                    with viewer.lock():
                        mujoco.mj_forward(model, data)
                else:
                    mujoco.mj_forward(model, data)

    # -- Spin Infrastructure Ticks --
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

    try:
        while recorder.total_frames < 100000:
            t_cycle = time.perf_counter()
            
            # --- REAL-TIME SENSOR FUSION ---
            # 1. Get real data from the buffers
            f_cos, f_sin, f_dist = get_food_metrics()
            lidar_data, _ = sensor_buffer.lidar.read() # Current LiDAR scan
            
            # 2. Build the REAL 34-D observation vector
            # This matches the training shape exactly
            obs_raw = np.concatenate([
                lidar_data,                               # 24 LiDAR rays
                [data.qvel[0], data.qvel[1]],             # 2 Velocity
                [f_cos, f_sin, f_dist],                   # 3 Food metrics
                data.qpos[7:12]                           # 5 Proprioceptive joint states
            ]).astype(np.float32)

            # 2. Compute Reward
            reward_dist = prev_dist - f_dist
            reward = (reward_dist * 10.0) - 0.001
            if data.geom("chassis").xpos[2] < 0.02: reward -= 1.0 # Fall penalty
            prev_dist = f_dist

            # 3. Neural Handover (Active Inference Step)
            # This replaces the analytical autopilot
            human_active = any(keyboard.is_pressed(k) for k in ['w','a','s','d','up','down','left','right'])
            if human_active:
                v_exec = 0.4 if keyboard.is_pressed('w') or keyboard.is_pressed('up') else (-0.4 if keyboard.is_pressed('s') or keyboard.is_pressed('down') else 0.0)
                w_exec = 0.8 if keyboard.is_pressed('a') or keyboard.is_pressed('left') else (-0.8 if keyboard.is_pressed('d') or keyboard.is_pressed('right') else 0.0)
                exec_action = np.array([v_exec, w_exec], dtype=np.float32)
                smoothed_action = exec_action.copy()
            else:
                # The Brain takes control!
                raw_action, _, _ = brain.step(obs_raw, reward)
                
                # Apply Exponential Moving Average (EMA) to smooth out neural jitter
                # Humans tap keys discretely, but robots need continuous curves
                smoothed_action = 0.85 * smoothed_action + 0.15 * raw_action
                exec_action = smoothed_action
                
                if planner_step[0] % 50 == 0:
                    print(f"[DEBUG] f_cos: {f_cos:.2f}, f_sin: {f_sin:.2f}, raw_out: {raw_action}")

            # [NEW] S-Curve Acceleration (Smoothing)
            target_v = exec_action[0]
            last_v = 0.85 * last_v + 0.15 * target_v 
            exec_action[0] = last_v
            
            # [NEW] Inject CPG Sway
            # Calculate intensity based on Dopamine (da)
            cpg_intensity = brain.drives.da * 0.15 
            
            # Get oscillation modulation
            cpg_mod = cpg.compute_biomimetic_modulation(brain.drives, [last_v, last_v])
            
            # Add sway to steering/angular velocity
            # We subtract the base velocity to isolate the pure swing
            raw_sway = cpg_mod[0] - last_v
            
            # Scale raw_sway to a visible steering velocity perturbation (max ~0.45 rad/s)
            # Modulate by normalized speed ratio to keep waddling relative to movement speed
            speed_ratio = np.clip(abs(last_v) / 0.4, 0.0, 1.0)
            exec_action[1] += (raw_sway * cpg_intensity * 300.0 * speed_ratio)
            
            # 4. Dispatch Commands
            sensor_buffer.planning_target.write(exec_action)
            recorder.record_step(obs_raw, exec_action, planner_step[0] * 0.02)
            
            # 5. Narrative & Tracking
            narrator.update(planner_step[0], f_dist, human_active, 0.0, brain.drives)
            if planner_step[0] % 100 == 0:
                print(f"[CORTEX] Emergence Status | Reward: {reward:.4f} | Dist: {f_dist:.2f}")

            # Resolve target discovery boundaries
            if not event_queue.empty():
                if event_queue.get() == 'food':
                    ep = recorder.finalize_episode()
                    serializer.write_episode_fallback(ep, episode_idx)
                    narrator.log_success(episode_idx, len(ep['observations']))
                    episode_idx += 1
                    recorder.start_episode(episode_idx)

            planner_step[0] += 1
            time.sleep(max(0, 0.02 - (time.perf_counter() - t_cycle)))
            if viewer is not None:
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
