"""
carl_circadian.py — Circadian and Ultradian Biological Clock for CARL.

Biological analogue: Transcriptional-translational feedback loops in the
suprachiasmatic nucleus. Generates autonomous circadian (~24h) and ultradian
(~90m) rhythms to regulate arousal, metabolic thresholds, and cognitive modes.
"""

import numpy as np

class CircadianOscillator:
    """
    Generates biological rhythms that modulate cognitive parameters.

    Two oscillators:
      1. Circadian (~2000 steps): Activity/rest cycle
      2. Ultradian (~500 steps): Exploration/exploitation oscillation
    """

    def __init__(self):
        self.circadian_phase = 0.0         # [0, 2pi]
        self.ultradian_phase = 0.0         # [0, 2pi]

        # ── EVOLVABLE PARAMETERS ─────────────────────────────────────
        self.circadian_period = 2000       # steps per activity/rest cycle
        self.ultradian_period = 500        # steps per explore/exploit microcycle
        self.amplitude = 0.3               # modulation strength

    def step(self):
        """Advance both oscillators by one tick."""
        self.circadian_phase += 2.0 * np.pi / self.circadian_period
        self.ultradian_phase += 2.0 * np.pi / self.ultradian_period
        self.circadian_phase %= (2.0 * np.pi)
        self.ultradian_phase %= (2.0 * np.pi)

    def get_activity_drive(self):
        """Circadian modulation of activity level. Peak at phase=0, trough at phase=pi."""
        return 0.5 + self.amplitude * np.cos(self.circadian_phase)

    def get_exploration_bias(self):
        """Ultradian oscillation between exploration (+) and exploitation (-)."""
        return float(self.amplitude * np.sin(self.ultradian_phase))

    def is_rest_phase(self):
        """Should CARL be in a low-activity/consolidation phase?"""
        return np.cos(self.circadian_phase) < -0.5

    def get_fatigue_recovery_bonus(self):
        """Fatigue recovers faster during rest phases."""
        if self.is_rest_phase():
            return 1.5   # 50% faster recovery
        return 1.0

    def get_cortisol_modulation(self):
        """Pulsatile cortisol — peaks during early 'morning' phase."""
        return float(max(0.0, np.cos(self.circadian_phase - 0.5)) * 0.1)

    def mutate(self, rate=0.15):
        """Mutate evolvable parameters."""
        self.circadian_period = int(np.clip(
            self.circadian_period + np.random.normal(0, rate * 200), 1000, 5000))
        self.ultradian_period = int(np.clip(
            self.ultradian_period + np.random.normal(0, rate * 80), 200, 1000))
        self.amplitude = float(np.clip(
            self.amplitude + np.random.normal(0, rate * 0.05), 0.1, 0.5))

    def save(self):
        return {
            'circadian_phase': self.circadian_phase,
            'ultradian_phase': self.ultradian_phase,
            'circadian_period': self.circadian_period,
            'ultradian_period': self.ultradian_period,
            'amplitude': self.amplitude,
        }

    def load(self, d):
        self.circadian_phase = d.get('circadian_phase', 0.0)
        self.ultradian_phase = d.get('ultradian_phase', 0.0)
        self.circadian_period = d.get('circadian_period', 2000)
        self.ultradian_period = d.get('ultradian_period', 500)
        self.amplitude = d.get('amplitude', 0.3)
