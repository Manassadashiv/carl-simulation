"""
carl_active_inference.py — Fristonian Active Inference Engine for ARM CONTROL.

This engine replaces standard Actor-Critic policy gradient updates with
thermodynamic Free Energy minimization. At each step, Bob generates K=20
smooth candidate action sequences, projects them into the future using the
carl_imagination world model, and selects the sequence that minimizes
Expected Free Energy (G) by reaching for the object and grasping it.
"""

import numpy as np
import time

class ActiveInferenceEngine:
    def __init__(self, K=20, horizon=5, theta=0.35, sigma=0.15, act_dim=8):
        self.K = K
        self.horizon = horizon
        self.theta = theta
        self.sigma = sigma
        self.act_dim = act_dim
        self.obs_dim = 25

    def generate_smooth_candidates(self, prev_action, base_intention=None):
        """
        Generate K candidate action sequences using an Ornstein-Uhlenbeck (OU) process.
        Ensures smooth, continuous trajectory candidates rather than erratic movements.
        """
        candidates = np.zeros((self.K, self.horizon, self.act_dim), dtype=np.float32)
        for k in range(self.K):
            # If base_intention (from LTC) is provided, center the exploration around it.
            if base_intention is not None:
                mu = base_intention + np.random.uniform(-0.3, 0.3, size=self.act_dim).astype(np.float32)
            else:
                mu = np.random.uniform(-1.0, 1.0, size=self.act_dim).astype(np.float32)
            
            mu = np.clip(mu, -1.0, 1.0)
            
            curr_act = prev_action.copy()
            for t in range(self.horizon):
                noise = np.random.normal(0.0, self.sigma, size=self.act_dim)
                # OU drift: current + theta * (target - current) + noise
                curr_act = curr_act + self.theta * (mu - curr_act) + noise
                curr_act = np.clip(curr_act, -1.0, 1.0)
                candidates[k, t] = curr_act
        return candidates

    def select_action(self, predictor, obs_history, prev_action, base_intention=None):
        """
        Evaluates G(pi) for K candidate trajectories and samples the optimal policy.
        
        obs_history is expected to be shape (10, 25).
        Returns the chosen 8D action for the first step.
        """
        # 1. Generate K smooth candidate sequences
        candidates = self.generate_smooth_candidates(prev_action, base_intention)
        
        # 2. Run prediction forward pass in batch
        # Replicate obs_history (10, 25) to shape (K, 10, 25)
        obs_history_batch = np.repeat(obs_history[None, :, :], self.K, axis=0)
        
        # Call SpatiotemporalPredictor in batch mode
        pred_mean, pred_var = predictor.forward(obs_history_batch, candidates)
        # pred_mean: (K, 5, 25), pred_var: (K, 5, 25)
        
        # 3. Calculate Epistemic Value (Intrinsinc Curiosity / Variance Sum)
        # We want to maximize predicted variance, which minimizes -Epistemic
        epistemic_scores = np.sum(pred_var, axis=(1, 2))  # (K,)
        
        # 4. Construct Prior Preferences Profile (P_raw) and Weight Vector (W_p)
        P_raw = np.zeros(self.obs_dim, dtype=np.float32)
        W_p = np.zeros(self.obs_dim, dtype=np.float32)
        
        # STATE_25 mapping:
        # 0-7: arm_pos
        # 8-15: arm_vel
        # 16-18: rel (vector from tip to object)
        # 19-21: tip_vel
        # 22: tl (touch left)
        # 23: tr (touch right)
        # 24: M_t (drive)
        
        # We want the rel vector to be [0, 0, 0] (i.e. hand tip is exactly at the object)
        P_raw[16:19] = 0.0
        W_p[16:19] = 2.0  # High penalty for being far from object
        
        # We want the tip velocity to slow down as it approaches (so it doesn't smack it)
        P_raw[19:22] = 0.0
        W_p[19:22] = 0.5
        
        # We want to maximize touch sensors
        P_raw[22] = 1.0
        P_raw[23] = 1.0
        W_p[22:24] = 5.0  # Massive weight to prioritize touching/grasping
        
        # 5. Normalize raw preference targets
        # We don't have rolling stats for the 25D arm state readily available, 
        # so we will use fixed approximate normalization for the error.
        # Rel is usually [-1, 1] meters, touch is [0, 1].
        # We can just compute the unnormalized weighted squared distance.
        pragmatic_scores = -np.sum(W_p[None, :] * (pred_mean - P_raw[None, :])**2, axis=(1, 2))  # (K,)
        
        # 6. Expected Free Energy G = -Epistemic - Pragmatic
        # We minimize G, which maximizes Epistemic (Curiosity) + Pragmatic (Reaching/Grasping)
        G = -epistemic_scores - pragmatic_scores
        
        # 7. Softmax selection
        gamma = 15.0  # High gamma to exploit the best trajectory (we want precision grabbing)
        logits = -gamma * G
        logits_shifted = logits - np.max(logits)  # numerical stability
        probs = np.exp(logits_shifted) / np.sum(np.exp(logits_shifted))
        
        # Sample policy trajectory index
        best_idx = np.random.choice(self.K, p=probs)
        best_action = candidates[best_idx]
        
        # Return first action step, and metadata for diagnostics/telemetry
        return best_action[0], G[best_idx], 1.0
