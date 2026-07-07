"""
carl_omega_extensions.py
Mirror Neurons, Predictive Allostasis, and integration helpers.
"""
import math, numpy as np
from collections import deque

# ════════════════════════════════════════════════════════════════
# INNOVATION 5 — Mirror Neuron Cross-Body Learning
# Rizzolatti 1996: neurons that fire when you act AND when you observe.
# Body A simulates Body B's last trajectory in its OWN world model.
# If the simulated outcome is bad → add to A's danger map.
# ════════════════════════════════════════════════════════════════
class MirrorNeuronSystem:
    def __init__(self, sdim=5):
        self.sdim = sdim
        self.sim_surprise_history = deque(maxlen=50)

    def learn_from_death(self, survivor_brain, dead_brain, T_ltm,
                         danger_update_fn, D):
        """
        When a sibling dies, the survivor simulates their last N steps
        through its own world model. High simulation surprise = shared danger.
        """
        buf = dead_brain.get('buf', [])
        if not buf:
            return D

        # Use survivor's blended model
        alpha = max(0.01, 1.0 - float(np.trace(survivor_brain['P_wm'])) / (500*(self.sdim+1)))
        T_use = alpha * survivor_brain['T_wm'] + (1-alpha) * T_ltm
        Ad = T_use[:self.sdim, :].T
        Bd = T_use[self.sdim, :]

        total_sim_surprise = 0.0
        for (xk_m, uk_m, xn_m, _) in buf[-15:]:
            xs_sim = Ad @ xk_m + Bd * uk_m
            sim_surprise = float(np.linalg.norm(xs_sim - xn_m))
            total_sim_surprise += sim_surprise
            if sim_surprise > 0.25:
                # "I would have been surprised here too"
                D = danger_update_fn(D, xn_m[2], xn_m[3],
                                     sim_surprise * 0.55, rate=0.18)

        avg = total_sim_surprise / max(1, len(buf[-15:]))
        self.sim_surprise_history.append(avg)
        return D

    def empathy_signal(self):
        """How much the survivor felt the sibling's death (0-1)."""
        if not self.sim_surprise_history:
            return 0.0
        return float(np.clip(np.mean(self.sim_surprise_history) * 2.0, 0., 1.))


# ════════════════════════════════════════════════════════════════
# INNOVATION 7 — Predictive Allostasis
# McEwen 1998, Sterling 2012: animals predict future stress, not just react.
# ════════════════════════════════════════════════════════════════
class PredictiveAllostasis:
    def __init__(self, sdim=5, horizon=60):
        self.sdim    = sdim
        self.horizon = horizon
        self.predicted_load   = 0.0
        self.actual_load_hist = deque(maxlen=200)
        self.prediction_error = 0.0

    def predict_future_load(self, x, T_ltm, D, danger_at_fn):
        """
        Simulate coasting forward (u=0) for `horizon` steps.
        Accumulate predicted danger as future allostatic load.
        """
        Ad = T_ltm[:self.sdim, :].T
        Bd = T_ltm[self.sdim, :]
        xs = x.copy()
        future_load = 0.0
        for h in range(self.horizon):
            xs = Ad @ xs + Bd * 0.0   # coast
            d  = danger_at_fn(D, xs[2], xs[3])
            future_load += d * (0.97 ** h)
        self.predicted_load = float(future_load / self.horizon)
        return self.predicted_load

    def update(self, actual_load):
        self.actual_load_hist.append(actual_load)
        if len(self.actual_load_hist) > 10:
            self.prediction_error = abs(
                self.predicted_load - float(np.mean(list(self.actual_load_hist)[-10:])))

    def anticipatory_avoidance_cost(self):
        """
        Extra cost added to actions heading toward high-future-load states.
        Animals that PREDICT chronic stress avoid risky territories earlier.
        """
        return float(np.clip(self.predicted_load * 3.0, 0., 2.0))

    def stress_forecast(self):
        return round(self.predicted_load, 4)


# ════════════════════════════════════════════════════════════════
# INNOVATION 3 — Theta Oscillation Learning Gate
# Buzsáki 2002: 4-8 Hz rhythm gates hippocampal plasticity.
# ════════════════════════════════════════════════════════════════
class ThetaGate:
    def __init__(self, freq_hz=6.0):
        self.freq    = freq_hz
        self.phase   = 0.0

    def step(self, dt):
        self.phase = (self.phase + 2*math.pi * self.freq * dt) % (2*math.pi)

    @property
    def gate(self):
        """0 at trough, 1 at peak."""
        return 0.5 + 0.5 * math.cos(self.phase)

    def learning_lambda(self, base=0.990):
        """Lambda (forgetting) is LOWER (more learning) at theta peak."""
        return float(np.clip(base - self.gate * 0.012, 0.970, 0.999))

    def explore_boost(self, nm_ACh):
        """At trough: explore more (novel encoding phase)."""
        trough = 1.0 - self.gate
        return float(1.0 + trough * nm_ACh * 0.6)
