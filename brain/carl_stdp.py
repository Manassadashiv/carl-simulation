"""
carl_stdp.py — Reward-Modulated Spike-Timing Dependent Plasticity (R-STDP)
Based on: Markram 1997 (STDP) + Izhikevich 2007 (reward modulation + eligibility traces)

Three-factor learning rule:
  ΔW = learning_rate × Reward × Eligibility_Trace

Where eligibility trace tracks STDP correlations:
  dE/dt = -E/tau_E + STDP(pre, post)

And STDP:
  pre before post  → LTP (Long Term Potentiation) — strengthen
  post before pre  → LTD (Long Term Depression)   — weaken

This is BIOLOGICALLY CORRECT memory — far more precise than Hebb's "fire together wire together."
"""
import numpy as np
from collections import deque


# ── Parameters ───────────────────────────────────────────────
TAU_PRE   = 20.0   # ms — pre-synaptic trace decay
TAU_POST  = 20.0   # ms — post-synaptic trace decay
TAU_E     = 50.0   # ms — eligibility trace decay
A_PLUS    = 0.01   # LTP amplitude
A_MINUS   = 0.012  # LTD amplitude (slightly asymmetric — bias toward depression)
W_MAX     = 2.0    # Maximum synaptic weight
W_MIN     = -0.5   # Minimum (allows weak inhibition)
LR        = 0.008  # Reward modulation learning rate


class STDPSynapse:
    """
    A single population of STDP synapses between two neural populations.
    
    pre_size:  number of pre-synaptic neurons (e.g., state dimension)
    post_size: number of post-synaptic neurons (e.g., action candidates)
    """

    def __init__(self, pre_size, post_size, init_scale=0.1):
        self.pre_size  = pre_size
        self.post_size = post_size

        # Synaptic weights
        self.W = np.random.randn(pre_size, post_size) * init_scale

        # Eligibility traces (per synapse)
        self.E = np.zeros((pre_size, post_size), dtype=float)

        # Activity traces
        self.A_pre  = np.zeros(pre_size,  dtype=float)  # pre-synaptic trace
        self.A_post = np.zeros(post_size, dtype=float)  # post-synaptic trace

        # Reward signal buffer (3-factor rule needs short delay)
        self.reward_buf = deque(maxlen=5)

        # Learning metrics
        self.ltp_events = 0
        self.ltd_events = 0
        self.total_weight_change = 0.0

    def step(self, pre_spikes, post_spikes, reward, dt=1.0):
        """
        Update weights given pre/post spikes and delayed reward.

        pre_spikes:  (pre_size,) binary or rate vector [0,1]
        post_spikes: (post_size,) binary or rate vector [0,1]
        reward:      scalar reward signal (positive = good, negative = bad)
        dt:          timestep in ms-equivalent units
        """
        # ── Decay traces ──────────────────────────────────────
        decay_pre  = np.exp(-dt / TAU_PRE)
        decay_post = np.exp(-dt / TAU_POST)
        decay_e    = np.exp(-dt / TAU_E)

        self.A_pre  *= decay_pre
        self.A_post *= decay_post
        self.E      *= decay_e

        # ── STDP correlation ──────────────────────────────────
        # LTP: post fired, pre just fired (pre causes post) → strengthen
        if np.any(post_spikes > 0):
            # Each pre-spike that occurred before contributes A_pre
            ltp = np.outer(self.A_pre, post_spikes) * A_PLUS  # (pre, post)
            self.E  += ltp
            self.ltp_events += int(np.sum(post_spikes > 0))

        # LTD: pre fired, post just fired before → depress
        if np.any(pre_spikes > 0):
            # Each post-spike before contributes A_post
            ltd = np.outer(pre_spikes, self.A_post) * (-A_MINUS)
            self.E  += ltd
            self.ltd_events += int(np.sum(pre_spikes > 0))

        # ── Update traces AFTER correlation (causal order) ───
        self.A_pre  += pre_spikes
        self.A_post += post_spikes

        # ── Reward modulation ─────────────────────────────────
        self.reward_buf.append(reward)
        avg_reward = float(np.mean(self.reward_buf)) if self.reward_buf else 0.0

        if abs(avg_reward) > 1e-4:
            dW = LR * avg_reward * self.E
            self.W  = np.clip(self.W + dW, W_MIN, W_MAX)
            self.total_weight_change += float(np.linalg.norm(dW))

        return self.W.copy()

    def predict(self, pre_activity):
        """Forward pass through synaptic weights."""
        return np.tanh(self.W.T @ pre_activity)   # (post_size,)

    def plasticity_report(self):
        return {
            'ltp': self.ltp_events,
            'ltd': self.ltd_events,
            'mean_w': float(np.mean(self.W)),
            'total_dw': round(self.total_weight_change, 4),
        }


# ── STDP-based Action Evaluator ──────────────────────────────────────────────
class STDPActionEvaluator:
    """
    Replaces the hand-crafted Hebbian associator with a proper R-STDP network.
    
    Encodes: state → action quality weights via STDP
    
    The "pre-synapse" is the state representation.
    The "post-synapse" is the action identity.
    A "spike" occurs when surprise > threshold (unexpected outcome = signal).
    Reward = dopamine signal (negative surprise = prediction match = reward).
    """

    def __init__(self, state_dim=5, visual_dim=15, n_actions=9):
        self.state_dim = state_dim
        self.visual_dim = visual_dim
        self.n_actions = n_actions
        self.full_dim = state_dim + visual_dim

        # State-to-action STDP synapse
        self.synapse = STDPSynapse(self.full_dim, n_actions, init_scale=0.05)

        # Running average reward (for baseline subtraction)
        self.reward_baseline = 0.0
        self.baseline_tau    = 200.0  # steps

        # Map action values to action indices
        self.action_map = [-8., -5., -2., -1., 0., 1., 2., 5., 8.]

    def encode_state(self, x, v_pop):
        """Encode state vector as firing rates ∈ [0, 1]."""
        proprio = np.clip(np.abs(x) / np.array([8., 3., 0.7, 3., 0.35]), 0., 1.)
        return np.concatenate([proprio, v_pop])

    def encode_action(self, u):
        """One-hot encode the chosen action."""
        idx = int(np.argmin([abs(u - a) for a in self.action_map]))
        spikes = np.zeros(self.n_actions)
        spikes[idx] = 1.0
        return spikes

    def update(self, xk, v_pop, uk, surprise, nm_DA):
        """
        Called every step with (previous state, action taken, outcome surprise).
        
        surprise: RLS prediction error (high = unexpected)
        nm_DA:    dopamine level (reward proxy)
        """
        pre_spikes  = self.encode_state(xk, v_pop)
        post_spikes = self.encode_action(uk)

        # Reward: high DA + low surprise = positive reward
        raw_reward  = nm_DA - surprise * 0.5
        # Subtract baseline (reward prediction error = dopamine-like signal)
        reward = raw_reward - self.reward_baseline
        self.reward_baseline += (raw_reward - self.reward_baseline) / self.baseline_tau

        # Threshold for spiking: only update when something notable happened
        threshold = 0.15
        pre_spikes  = pre_spikes * float(surprise > threshold)
        post_spikes = post_spikes * float(surprise > threshold)

        self.synapse.step(pre_spikes, post_spikes, reward, dt=1.0)

    def action_quality(self, x, v_pop, u):
        """
        Return the STDP-learned quality of action u in state x.
        Used as a prior in action selection (replaces Hebbian association_cost).
        """
        pre = self.encode_state(x, v_pop)
        q   = self.synapse.predict(pre)      # (n_actions,)
        idx = int(np.argmin([abs(u - a) for a in self.action_map]))
        return float(-q[idx])  # negate: higher quality → lower cost

    def best_action_bias(self, x, v_pop):
        """Return action quality vector for all actions (for deliberation)."""
        pre = self.encode_state(x, v_pop)
        return self.synapse.predict(pre)  # (n_actions,) — higher = better

    def plasticity_stats(self):
        return self.synapse.plasticity_report()


# ── Quantum-Inspired Superposition Deliberation ──────────────────────────────
class QuantumDeliberator:
    """
    Innovation 6: Quantum-inspired decision making.
    
    Instead of argmin(cost), each strategy is a quantum amplitude.
    Interference between strategies: aligned predictions amplify,
    conflicting predictions cancel. Final choice by Born rule probability.
    
    This produces GENUINELY PROBABILISTIC decisions — not random noise,
    but structured uncertainty that reflects actual ambiguity.
    """

    def __init__(self):
        self.history = deque(maxlen=50)

    def deliberate(self, strategies_costs):
        """
        strategies_costs: list of (name, u, s, cost) tuples
        Returns: chosen (name, u, s) by quantum Born rule
        """
        if not strategies_costs:
            return ('BALANCED', 0., 0.)

        costs = np.array([c for _, _, _, c in strategies_costs], dtype=float)

        # Map cost to quantum phase: low cost → small phase (constructive)
        # high cost → large phase (destructive)
        costs_norm = costs - costs.min()
        if costs_norm.max() > 1e-6:
            costs_norm /= costs_norm.max()

        # Amplitude: A_i = exp(i * phase_i) where phase ∈ [0, π]
        phases = costs_norm * math.pi
        amplitudes = np.exp(1j * phases)  # complex amplitudes

        # Interference: coherent sum
        total_amplitude = amplitudes.sum()

        # Relative amplitudes (Born rule)
        # prob_i = |A_i|² / |A_total|²... but we want interference:
        # Use overlap with total amplitude direction
        coherent_probs = np.abs(amplitudes * np.conj(total_amplitude)).real
        coherent_probs = np.clip(coherent_probs, 0., None)

        total_p = coherent_probs.sum()
        if total_p < 1e-8:
            probs = np.ones(len(strategies_costs)) / len(strategies_costs)
        else:
            probs = coherent_probs / total_p

        # Sample by Born rule
        chosen_idx = int(np.random.choice(len(strategies_costs), p=probs))
        name, u, s, _ = strategies_costs[chosen_idx]

        # Track history for uncertainty estimation
        self.history.append({'probs': probs.copy(), 'chosen': chosen_idx})

        return name, u, s, probs

    def decision_uncertainty(self):
        """Shannon entropy of recent decisions — high = genuinely uncertain."""
        if not self.history:
            return 0.5
        last = self.history[-1]['probs']
        last = np.clip(last, 1e-8, 1.)
        entropy = float(-np.sum(last * np.log(last)))
        max_ent = float(np.log(len(last)))
        return float(entropy / max_ent) if max_ent > 0 else 0.5


import math
