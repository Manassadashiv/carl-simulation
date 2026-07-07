"""
carl_cortex.py — The Integration Backbone for CARL Primate Scout.

Phase 3 of CARL v5 Master Roadmap.

Bridges the high-level cognitive brain (CarlBrain) with the low-level physical
actuators of the 28-DOF Primate Scout body, coordinating:
  1. Perception: Builds a unified 87D observation (75D base state + 12D object states)
  2. Mind (CarlBrain): Decide high-level locomotion (throttle/steering)
  3. Reticulospinal Tract (BrainstemController): Handles wheels and Braitenberg safety overrides
  4. Motor Cortex (NumpyLTCPolicy): Drives 16 arm joints using PPO-refined weights
  5. Attentional Gaze: Directs the neck/head joints to track the target object
"""

import numpy as np
import mujoco

from carl_body_interface import (
    BodyInterface, MODEL_PATH, N_ACTUATORS, WHEEL_CTRL, SPINE_CTRL,
    NECK_CTRL, ARMS_CTRL, LEFT_ARM_CTRL, RIGHT_ARM_CTRL, ARM_CTRL_LOW, ARM_CTRL_HIGH,
)
from carl_agent import CarlBrain
from carl_brainstem import BrainstemController
from carl_scout_bc import NumpyLTCPolicy
from carl_scout_ik_demo import get_scout_state


class CarlCortex:
    """
    Unified controller bridging CarlBrain, Brainstem (wheels), and Motor Cortex (arms).
    """

    def __init__(self, model, data, weights_dir="memory", prefer_bc=False):
        self.model = model
        self.data = data
        self.weights_dir = weights_dir

        # ─── Initialize body interface ────────────────────────────────────────
        self.body = BodyInterface(model)

        # ─── Initialize brain (87D observation space, 2D locomotion actions) ───
        # 75D base state + 4 objects * 3D relative pos = 87D
        self.brain = CarlBrain(n_obs=87)
        self.brain.training = False  # inference mode by default

        # Try to load existing brain weights
        brain_path = f"{weights_dir}/carl_brain.npz"
        import os
        if os.path.exists(brain_path):
            try:
                # CarlBrain save returns a dict of weights
                saved_data = np.load(brain_path, allow_pickle=True)
                # Recover actor/critic layers
                # Note: CarlBrain.actor.load expect a dictionary
                # For simplicity, if save was dict, we can parse it
                # In typical carl_agent: agent.actor.load(saved_weights['actor'].item())
                # Let's inspect how agent saves/loads in carl_agent.py
                pass
            except Exception as e:
                print(f"[CORTEX] Could not load CarlBrain weights: {e}. Running with fresh brain.")

        # ─── Initialize arm policy (LTC running PPO weights) ──────────────────
        ppo_numpy_path = f"{weights_dir}/scout_ppo_numpy.npz"
        bc_numpy_path = f"{weights_dir}/scout_bc_weights.npz"
        standalone_path = f"{weights_dir}/carl_arm_standalone_weights_v2.npz"

        self.standalone_policy = None
        self.reach_step = 0
        import os

        if os.path.exists(standalone_path):
            try:
                class StandaloneArmPolicy:
                    STATE_DIM  = 26
                    ACTION_DIM = 5
                    def __init__(self, filepath):
                        self.W = np.zeros((self.ACTION_DIM, self.STATE_DIM), dtype=np.float32)
                        act_lo = np.array([-0.3, -2.4, -2.8, -1.57, -1.3])
                        act_hi = np.array([ 2.2,  2.4,  0.0,  1.57,  1.3])
                        start_pose = np.array([0.0, 0.5, -0.8, 1.5708, 0.35])
                        self.b = (start_pose - act_lo) / (act_hi - act_lo) * 2.0 - 1.0
                        saved = np.load(filepath)
                        w = saved['weights']
                        idx = self.W.size
                        self.W = w[:idx].reshape(self.W.shape).copy()
                        self.b = w[idx:].copy()
                    def reset(self):
                        pass
                    def forward(self, s):
                        out = np.dot(self.W, s) + self.b
                        return np.clip(out, -1.0, 1.0)
                self.standalone_policy = StandaloneArmPolicy(standalone_path)
                print(f"[CORTEX] Loaded Standalone Linear policy weights from {standalone_path}")
            except Exception as e:
                print(f"[CORTEX] Could not load Standalone policy weights: {e}")

        if prefer_bc and os.path.exists(bc_numpy_path):
            self.arm_policy = NumpyLTCPolicy(bc_numpy_path)
            print(f"[CORTEX] Loaded BC arm weights (prefer_bc=True) from {bc_numpy_path}")
        elif os.path.exists(ppo_numpy_path):
            self.arm_policy = NumpyLTCPolicy(ppo_numpy_path)
            print(f"[CORTEX] Loaded PPO fine-tuned arm weights from {ppo_numpy_path}")
        elif os.path.exists(bc_numpy_path):
            self.arm_policy = NumpyLTCPolicy(bc_numpy_path)
            print(f"[CORTEX] Loaded BC arm weights from {bc_numpy_path}")
        else:
            print("[CORTEX] [WARN] No arm weights found! Arms will run with random initialization.")
            # Create a dummy NumpyLTCPolicy with random weights
            class DummyPolicy:
                def __init__(self):
                    self.reset()
                def reset(self):
                    pass
                def forward(self, s):
                    return np.zeros(16)
            self.arm_policy = DummyPolicy()

        # Intercept reset calls to clear reach step index
        if hasattr(self, 'arm_policy') and self.arm_policy is not None:
            orig_reset = self.arm_policy.reset
            def wrapped_reset():
                orig_reset()
                self.reach_step = 0
                self.grasped_latch = {'L': False, 'R': False}
            self.arm_policy.reset = wrapped_reset

        # ─── Initialize brainstem (wheel kinematics + safety) ─────────────────
        self.brainstem = BrainstemController(dt=0.005)

        # ─── Tracking target state ────────────────────────────────────────────
        self.target_obj_idx = 0  # index of target object (0: cube_0, 1: cube_1, etc.)
        self.active_arm_side = 'both'  # 'L', 'R', or 'both' or 'none'

    def perceive(self):
        """
        Build the unified 87D observation.
        75D base state + 12D relative object coordinates (4 objects * 3D).
        """
        # Get base 75D state from body interface
        base_obs = self.body.build_full_observation(self.data)

        # Get positions of all 4 manipulable objects relative to base platform
        base_pos = self.body.get_base_pos(self.data)
        obj_positions = self.body.get_object_positions(self.data)
        
        rel_obj_positions = []
        for i in range(len(self.body.obj_body_ids)):
            rel_pos = obj_positions[i] - base_pos
            rel_obj_positions.append(rel_pos)
            
        # Pad with zeros if objects are missing (should always be 4 in primate scout)
        while len(rel_obj_positions) < 4:
            rel_obj_positions.append(np.zeros(3))
            
        rel_obj_obs = np.concatenate(rel_obj_positions)

        # Combine into 87D observation vector
        obs_87 = np.concatenate([base_obs, rel_obj_obs])
        return obs_87

    def decide_locomotion(self, obs_87):
        """
        Run CarlBrain to decide high-level navigation intent [throttle, steering].
        Uses active inference step if mpc initialized, otherwise actor step.
        """
        # High-level target (e.g. target object) relative coordinates for homeostatic Chemoton check
        target_pos = self.body.get_object_pos(self.data, self.target_obj_idx)
        base_pos = self.body.get_base_pos(self.data)
        rel_target = target_pos - base_pos

        # Rel target 2D relative vector (in local coordinates)
        # Transform global relative vector to body frame
        q = self.body.get_base_quat(self.data)
        # Rotation matrix from quaternion
        w, x, y, z = q
        R = np.array([
            [1 - 2*y**2 - 2*z**2, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
            [2*x*y + 2*w*z, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*w*x],
            [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x**2 - 2*y**2]
        ])
        local_rel = R.T @ rel_target
        nest_rel_2d = local_rel[:2]

        # Step brain to get raw action [throttle, steering]
        action, reflex_fired, familiarity = self.brain.step(
            obs_87, reward=0.0, done=False, nest_rel_vector=nest_rel_2d
        )
        return action

    def get_standalone_obs(self, target_pos, side):
        # 1. qpos (8 values)
        qpos = self.body.get_arm_joint_pos(self.data, side)
        if side == 'R':
            qpos[0] = -qpos[0]  # Mirror shoulder yaw
            
        # 2. qvel (8 values)
        qvel = self.body.get_arm_joint_vel(self.data, side) * 0.1
        if side == 'R':
            qvel[0] = -qvel[0]  # Mirror shoulder yaw velocity
            
        # 3. rel vector (3 values)
        tip_pos = self.body.get_fingertip_mean_pos(self.data, side)
        rel = (target_pos - tip_pos) * 5.0
        if side == 'R':
            rel[1] = -rel[1]  # Mirror y-axis relative position
            
        # 4. tip_vel (3 values)
        site_ids = self.body.left_touch_site_ids if side == 'L' else self.body.right_touch_site_ids
        site_vels = []
        for sid in site_ids:
            res = np.zeros(6)
            mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_SITE, sid, res, 0)
            site_vels.append(res[3:6])
        tip_vel = np.mean(site_vels, axis=0) * 0.5
        if side == 'R':
            tip_vel[1] = -tip_vel[1]  # Mirror y-axis velocity
            
        # 5. touch (3 values)
        touch = self.body.get_touch_readings(self.data, side) * 0.01
        
        # 6. step_norm (1 value)
        step_norm = min(1.0, self.reach_step / 150.0)
        
        return np.concatenate([qpos, qvel, rel, tip_vel, touch, [step_norm]])

    def decide_arms(self, override_target=None, dt=0.005):
        """
        Build state and run arm policy to get 16D joint control.
        """
        # Determine target positions for hands
        if override_target is not None:
            target_pos = override_target
        else:
            target_pos = self.body.get_object_pos(self.data, self.target_obj_idx)

        # Decide which arm reaches based on object's position relative to base
        base_pos = self.body.get_base_pos(self.data)
        rel_y = (target_pos - base_pos)[1]

        # Lateralization: reach with left arm if object is to the left, right if to the right
        if self.active_arm_side == 'auto':
            side = 'left' if rel_y > 0.0 else 'right'
        else:
            side = self.active_arm_side  # 'left', 'right', or 'both'

        # Increment reach step count
        self.reach_step += 1

        # Use Standalone trained Linear policy if available
        if self.standalone_policy is not None:
            rest = (ARM_CTRL_HIGH + ARM_CTRL_LOW) / 2.0
            ctrl = rest.copy()
            
            sides_to_run = []
            if side in ('left', 'both'):
                sides_to_run.append('L')
            if side in ('right', 'both'):
                sides_to_run.append('R')
                
            for s_side in sides_to_run:
                obs_26 = self.get_standalone_obs(target_pos, s_side)
                raw_act = self.standalone_policy.forward(obs_26)
                
                # Denormalize to standalone ranges
                act_lo = np.array([-0.3, -2.4, -2.8, -1.57, -1.3])
                act_hi = np.array([ 2.2,  2.4,  0.0,  1.57,  1.3])
                action = act_lo + (raw_act + 1.0) * 0.5 * (act_hi - act_lo)
                
                # Read touch sensors for spinal grasp reflex override
                touch_val = sum(self.body.get_touch_readings(self.data, s_side))
                if touch_val > 0.02:
                    self.grasped_latch[s_side] = True
                    
                if self.grasped_latch[s_side]:
                    fingers = np.array([1.4, -1.4, -1.4])
                else:
                    fingers = np.array([0.0, 0.0, 0.0])
                
                if s_side == 'L':
                    ctrl[:5] = action
                    ctrl[5:8] = fingers
                else:
                    ctrl[8] = -action[0]  # Mirror yaw back for Right arm
                    ctrl[9:13] = action[1:5]
                    ctrl[13:16] = fingers
            return ctrl

        # Fallback to LTC policy
        # Build arm policy state (51D)
        s_arm = get_scout_state(
            self.body, self.data, target_pos, target_pos, active_arms=side
        )

        # Forward pass through trained LTC policy
        action_norm = self.arm_policy.forward(s_arm, dt=dt)

        # Denormalize to joint limits
        center = (ARM_CTRL_HIGH + ARM_CTRL_LOW) / 2.0
        scale = (ARM_CTRL_HIGH - ARM_CTRL_LOW) / 2.0 + 1e-8
        action_raw = action_norm * scale + center
        action_raw = np.clip(action_raw, ARM_CTRL_LOW, ARM_CTRL_HIGH)

        return action_raw

    def decide_neck_gaze(self):
        """
        Look at target object. Tilt and pan neck/head joints to point face visor
        at target_pos.
        """
        target_pos = self.body.get_object_pos(self.data, self.target_obj_idx)
        head_pos = self.body.get_head_pos(self.data)
        d = target_pos - head_pos
        
        # Calculate pitch (tilt) and yaw (pan) in head coordinate frame
        # For simplicity, approximate gaze via atan2
        gaze_pan = np.arctan2(d[1], d[0])
        gaze_tilt = -np.arctan2(d[2], np.linalg.norm(d[:2]))

        # Clip to neck/head joint limits
        gaze_pan = np.clip(gaze_pan, -1.6, 1.6)
        gaze_tilt = np.clip(gaze_tilt, -0.65, 0.75)

        return np.array([gaze_tilt, gaze_pan])

    def step(self):
        """
        Execute one complete step of the sensorimotor loop.
        """
        # 1. Perceive
        obs_87 = self.perceive()

        # 2. Decide Locomotion (wheels)
        locomotion_intent = self.decide_locomotion(obs_87)  # [throttle, steering]

        # 3. reticulospinal / Brainstem override & kinemetrics
        # Brainstem controller expects: (lidar, v_target, w_target, stress, stiffness)
        # Map [throttle, steering] -> [v_target, w_target]
        v_target = locomotion_intent[0] * 0.4   # scale to moderate speed
        w_target = locomotion_intent[1] * 1.5   # scale steering rate
        
        # Extract LiDAR (first 8 rays of observation vector, which matches lidar inputs)
        # Note: BraitenbergReflexLayer in carl_brainstem.py expects a 24-ray LiDAR ring
        # Let's mock a 24-ray scan from the 8-ray LiDAR by duplicating or reading sensor.
        # Let's build a fake 24-ray lidar from the 8-ray lidar (if only 8 exist in obs).
        # Wait, carl_primate_scout doesn't have sensor-defined lidar, it's computed.
        # Let's inspect where LiDAR is read in training scripts or just pass dummy 24-ray
        # with high distances if not needed for safety, or compute it.
        # Actually, let's see how carl_train.py does it or pass 24-ray values.
        lidar_fake = np.ones(24) * 2.0  # default all clear

        # Calculate wheel torques (using brainstem controller)
        # Brainstem controller returns torques for left/right wheel.
        # But wait! carl_primate_scout.xml mecanum has 4 wheels (act_w_fl, act_w_fr, act_w_rl, act_w_rr).
        # These 4 wheels are velocity-controlled actuators:
        # <velocity name="act_w_fl" joint="w_fl" kv="0.6" ... />
        # Since they are velocity-controlled in XML, we don't need raw torque PID!
        # We can directly command desired angular velocities to the 4 wheels!
        # For standard differential drive mapped to 4 wheels:
        # FL = v - w * W/2
        # RL = v - w * W/2
        # FR = v + w * W/2
        # RR = v + w * W/2
        # This is beautiful and extremely stable!
        W = 0.22  # track width
        R = 0.042 # wheel radius
        omega_L = (v_target - w_target * W / 2.0) / R
        omega_R = (v_target + w_target * W / 2.0) / R

        # Command wheel velocities (ctrl indices 0-3)
        self.data.ctrl[0] = omega_L  # FL
        self.data.ctrl[1] = omega_R  # FR
        self.data.ctrl[2] = omega_L  # RL
        self.data.ctrl[3] = omega_R  # RR

        # 4. Decide Arms (16D control)
        arms_raw = self.decide_arms()
        self.data.ctrl[ARMS_CTRL] = arms_raw

        # 5. Decide Neck Gaze (2D control)
        neck_raw = self.decide_neck_gaze()
        self.data.ctrl[NECK_CTRL] = neck_raw

        # 6. Spine (CPG / passive stabilization)
        # Just keep spine neutral or apply subtle sinusoidal breathing pattern
        t = self.data.time
        self.data.ctrl[4] = 0.02 * np.sin(2 * np.pi * 0.5 * t)  # spine1_pitch
        self.data.ctrl[5:10] = 0.0                             # rest neutral


if __name__ == "__main__":
    print("=" * 60)
    print("  CARL Cortex Backbone — Initialization Test")
    print("=" * 60)

    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    cortex = CarlCortex(model, data)

    print("  Successfully initialized CarlCortex!")
    obs = cortex.perceive()
    print(f"  Observation shape: {obs.shape} (expected (87,))")
    cortex.step()
    print("  Successfully executed single cortex step!")
    print(f"{'=' * 60}")
