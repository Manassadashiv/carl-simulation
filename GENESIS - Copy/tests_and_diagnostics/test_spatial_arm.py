import os
import sys
import unittest
import numpy as np
import mujoco

# Ensure workspace root is in path
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))

from carl_arm_train import build_cache, reset_episode, ARM_CTRL_LOW, ARM_CTRL_HIGH, ARM_CTRL_SLICE, ArmPolicy
from carl_arm_bc import get_ik_action

class TestSpatialArm(unittest.TestCase):
    def setUp(self):
        self.model = mujoco.MjModel.from_xml_path("vessel_kinetic.xml")
        self.data = mujoco.MjData(self.model)
        self.cache = build_cache(self.model)
        self.rng = np.random.default_rng(seed=42)

    def test_actuator_count(self):
        # 16 actuators total (2 wheels + 4 head/hip + 10 arm actuators)
        self.assertEqual(self.model.nu, 16)
        
    def test_state_policy_dim(self):
        # State dim 29, Action dim 10
        self.assertEqual(ArmPolicy.STATE_DIM, 29)
        self.assertEqual(ArmPolicy.ACTION_DIM, 10)
        
        policy = ArmPolicy()
        dummy_state = np.zeros(29)
        action = policy.forward(dummy_state)
        self.assertEqual(action.shape[0], 10)
        self.assertTrue(np.all(action >= -1.0) and np.all(action <= 1.0))

    def test_ik_convergence(self):
        # Test IK convergence on a series of random points in the safe envelope
        successes = 0
        trials = 10
        
        for ep in range(trials):
            reset_episode(self.model, self.data, self.cache, "reach", self.rng)
            
            # Spawn in safe envelope: ox in [0.15, 0.20], oy in [0.13, 0.17], oz = 0.035
            ox = self.rng.uniform(0.15, 0.20)
            oy = self.rng.uniform(0.13, 0.17)
            oz = 0.035
            self.data.qpos[self.cache['obj_qpos'] : self.cache['obj_qpos']+3] = [ox, oy, oz]
            self.data.qpos[self.cache['obj_qpos']+3 : self.cache['obj_qpos']+7] = [1.0, 0.0, 0.0, 0.0]
            mujoco.mj_forward(self.model, self.data)
            
            success = False
            for step in range(150):
                # Lock base
                self.data.qpos[self.cache['Q_CARL'] : self.cache['Q_CARL']+3] = [0.0, 0.0, 0.04]
                self.data.qpos[self.cache['Q_CARL']+3 : self.cache['Q_CARL']+7] = [1.0, 0.0, 0.0, 0.0]
                self.data.qvel[self.cache['V_CARL'] : self.cache['V_CARL']+6] = 0.0
                
                obj_pos = self.data.xpos[self.cache['obj_bid']].copy()
                tip_pos = self.data.site_xpos[self.cache['tip_L_sid']].copy()
                dist = np.linalg.norm(obj_pos - tip_pos)
                
                if dist < 0.045:
                    success = True
                    break
                    
                act = get_ik_action(self.model, self.data, self.cache, obj_pos)
                center = (ARM_CTRL_HIGH + ARM_CTRL_LOW) / 2.0
                scale = (ARM_CTRL_HIGH - ARM_CTRL_LOW) / 2.0
                act_physical = act * scale + center
                
                self.data.ctrl[:2] = 0.0
                self.data.ctrl[ARM_CTRL_SLICE] = act_physical
                mujoco.mj_step(self.model, self.data)
                
            if success:
                successes += 1
                
        # We expect a high success rate (e.g. at least 90%)
        success_rate = successes / trials
        print(f"[TEST IK] Success rate: {successes}/{trials} ({success_rate*100:.1f}%)")
        self.assertGreaterEqual(success_rate, 0.90)

    def test_policy_convergence(self):
        # Load weights from memory/carl_arm_weights.npz if they exist
        weights_path = "memory/carl_arm_weights.npz"
        if not os.path.exists(weights_path):
            self.skipTest("No trained policy weights found to test")
            
        saved = np.load(weights_path)
        policy = ArmPolicy()
        policy.set_params(saved['weights'])
        
        successes = 0
        trials = 10
        
        for ep in range(trials):
            reset_episode(self.model, self.data, self.cache, "reach", self.rng)
            
            # Spawn in safe envelope: ox in [0.15, 0.20], oy in [0.13, 0.17], oz = 0.035
            ox = self.rng.uniform(0.15, 0.20)
            oy = self.rng.uniform(0.13, 0.17)
            oz = 0.035
            self.data.qpos[self.cache['obj_qpos'] : self.cache['obj_qpos']+3] = [ox, oy, oz]
            self.data.qpos[self.cache['obj_qpos']+3 : self.cache['obj_qpos']+7] = [1.0, 0.0, 0.0, 0.0]
            mujoco.mj_forward(self.model, self.data)
            
            # Reset policy hidden state
            policy.x = np.zeros(policy.HIDDEN_DIM)
            smoothed = np.zeros(ArmPolicy.ACTION_DIM)
            
            success = False
            for step in range(250): # Allow up to 250 steps for policy rollout
                self.data.qpos[self.cache['Q_CARL'] : self.cache['Q_CARL']+3] = [0.0, 0.0, 0.04]
                self.data.qpos[self.cache['Q_CARL']+3 : self.cache['Q_CARL']+7] = [1.0, 0.0, 0.0, 0.0]
                self.data.qvel[self.cache['V_CARL'] : self.cache['V_CARL']+6] = 0.0
                
                obj_pos = self.data.xpos[self.cache['obj_bid']].copy()
                tip_pos = self.data.site_xpos[self.cache['tip_L_sid']].copy()
                dist = np.linalg.norm(obj_pos - tip_pos)
                
                if dist < 0.045:
                    success = True
                    break
                    
                # Build state vector for policy
                from carl_arm_train import get_state
                s = get_state(self.model, self.data, self.cache)
                
                raw = policy.forward(s, dt=0.01)
                act_physical = ARM_CTRL_LOW + (raw + 1.0) * 0.5 * (ARM_CTRL_HIGH - ARM_CTRL_LOW)
                
                self.data.ctrl[:2] = 0.0
                self.data.ctrl[ARM_CTRL_SLICE] = act_physical
                mujoco.mj_step(self.model, self.data)
                
            if success:
                successes += 1
                
        success_rate = successes / trials
        print(f"[TEST POLICY] Success rate: {successes}/{trials} ({success_rate*100:.1f}%)")
        self.assertGreaterEqual(success_rate, 0.80)

if __name__ == "__main__":
    unittest.main()
