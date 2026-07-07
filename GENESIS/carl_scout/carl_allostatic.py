"""
carl_allostatic.py — Allostatic Body State Predictor for CARL.

Biological analogue: Predictive regulation (allostasis) in the insular cortex
and hypothalamus. Extrapolates metabolic trajectories to anticipate potential
resource deficits (exhaustion, starvation) and generate behavioral urges BEFORE
critical homeostasis boundaries are breached.
"""

import numpy as np

class AllostaticPredictor:
    """
    Predicts future homeostatic states based on current trajectory.

    Urgency = how likely is death if behavior doesn't change?
    High urgency triggers preemptive foraging.
    """

    def __init__(self):
        self.energy_history = []      # last N energy readings
        self.drain_history = []       # last N drain rates
        self.max_history = 50
        self.urgency = 0.0            # [0, 1]: behavioral urgency

        # ── EVOLVABLE PARAMETERS ─────────────────────────────────────
        self.prediction_horizon = 200     # steps to look ahead
        self.anticipation_gain = 0.1      # how aggressively urgency ramps
        self.critical_threshold = 0.15    # predicted energy below this = critical

    def update(self, energy, drain_rate):
        """Record current energy and drain rate."""
        self.energy_history.append(energy)
        self.drain_history.append(drain_rate)
        if len(self.energy_history) > self.max_history:
            self.energy_history.pop(0)
            self.drain_history.pop(0)

    def predict_energy(self, steps_ahead=None):
        """Linear extrapolation of energy trajectory."""
        if len(self.drain_history) < 5:
            return 1.0  # not enough data yet
        steps = steps_ahead or self.prediction_horizon
        avg_drain = float(np.mean(self.drain_history[-20:]))
        current_energy = self.energy_history[-1] if self.energy_history else 1.0
        return max(0.0, current_energy - avg_drain * steps)

    def compute_urgency(self, nearest_food_dist=None, velocity=0.0):
        """
        Urgency = how likely is death if behavior doesn't change?
        High urgency triggers preemptive foraging.
        """
        predicted_energy = self.predict_energy()

        # If we know where food is, factor in travel time
        if nearest_food_dist is not None and abs(velocity) > 0.01:
            time_to_food = nearest_food_dist / abs(velocity)
            current_energy = self.energy_history[-1] if self.energy_history else 1.0
            avg_drain = float(np.mean(self.drain_history[-20:])) if self.drain_history else 0.0
            energy_at_food = current_energy - avg_drain * time_to_food
            if energy_at_food < 0.05:
                self.urgency = min(1.0, self.urgency + self.anticipation_gain * 2)
                return self.urgency

        if predicted_energy < self.critical_threshold:
            self.urgency = min(1.0, self.urgency + self.anticipation_gain)
        elif predicted_energy < 0.3:
            self.urgency = min(0.7, self.urgency + self.anticipation_gain * 0.5)
        else:
            self.urgency = max(0.0, self.urgency - 0.01)  # slowly relax

        return self.urgency

    def mutate(self, rate=0.15):
        """Mutate evolvable parameters."""
        self.prediction_horizon = int(np.clip(
            self.prediction_horizon + np.random.normal(0, rate * 30), 50, 500))
        self.anticipation_gain = float(np.clip(
            self.anticipation_gain + np.random.normal(0, rate * 0.02), 0.01, 0.3))
        self.critical_threshold = float(np.clip(
            self.critical_threshold + np.random.normal(0, rate * 0.03), 0.05, 0.3))

    def save(self):
        return {
            'urgency': self.urgency,
            'prediction_horizon': self.prediction_horizon,
            'anticipation_gain': self.anticipation_gain,
            'critical_threshold': self.critical_threshold,
        }

    def load(self, d):
        self.urgency = d.get('urgency', 0.0)
        self.prediction_horizon = d.get('prediction_horizon', 200)
        self.anticipation_gain = d.get('anticipation_gain', 0.1)
        self.critical_threshold = d.get('critical_threshold', 0.15)
