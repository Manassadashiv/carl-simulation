"""
carl_metrics.py — Metrics and Logging Utilities for CARL.

Biological analogue: Neuro-telemetry and metabolic charting.
Handles high-fidelity recording of behavioral complexity (via action entropy),
metabolic efficiency (energy cost per meter), and sub-cognitive states
(workspace entropy, somatic marker accumulation).
"""

import os
import numpy as np

class EmergenceMetricsLogger:
    """Manages files and periodic logging for CARL's behavioral and metabolic metrics."""
    
    def __init__(self, emergence_log_path="emergence_metrics.log", metabolic_log_path="metabolic_ecology.log"):
        self.emergence_log_path = emergence_log_path
        self.metabolic_log_path = metabolic_log_path
        
    def _calculate_habit_index(self, action_history) -> float:
        """Shannon entropy computation over action sequences."""
        if len(action_history) >= 10:
            bins_x = np.linspace(-1.0, 1.0, 6)
            bins_y = np.linspace(-1.0, 1.0, 6)
            actions = np.array(action_history)
            hist, _, _ = np.histogram2d(actions[:, 0], actions[:, 1], bins=(bins_x, bins_y))
            probs = hist.flatten() / len(action_history)
            probs = probs[probs > 0.0]
            return float(-np.sum(probs * np.log2(probs)))
        return 0.0

    def log_tick(self, step, brain, action_history, world_map, goal_lifetimes, expired_lifetimes,
                 wheel_effort, joint_effort, velocity, state):
        """Idempotently logs a tick of metrics to emergence and metabolic files."""
        # 1. Calculate stats
        habit_index = self._calculate_habit_index(action_history)
        exploration_diversity = world_map.confidence * 100.0
        
        all_times = expired_lifetimes + list(goal_lifetimes.values())
        preference_stability = float(np.mean(all_times)) if all_times else 0.0
        
        # 2. Write to emergence_metrics.log
        if not os.path.exists(self.emergence_log_path):
            with open(self.emergence_log_path, 'w') as f_log:
                f_log.write("step,habit_index,exploration_diversity,preference_stability,workspace_mode,workspace_entropy,somatic_markers,somatic_bias,habit_chunks,dmn_activation,allostatic_urgency,circadian_phase\n")
                
        ws_mode = brain.workspace.get_behavioral_mode()
        ws_entropy = brain.workspace.get_mode_entropy()
        n_markers, _, _ = brain.somatic.get_stats()
        n_chunks, _, _ = brain.habits.get_stats()
        dmn_act = brain.dmn.activation_level
        allo_urg = brain.allostatic.urgency
        circ_phase = brain.circadian.circadian_phase / (2.0 * np.pi)
        
        with open(self.emergence_log_path, 'a') as f_log:
            f_log.write(f"{step},{habit_index:.4f},{exploration_diversity:.2f},{preference_stability:.2f},{ws_mode},{ws_entropy:.3f},{n_markers},{brain.somatic.approach_bias:.3f},{n_chunks},{dmn_act:.3f},{allo_urg:.3f},{circ_phase:.3f}\n")

        # 3. Write to metabolic_ecology.log
        if not os.path.exists(self.metabolic_log_path):
            with open(self.metabolic_log_path, 'w') as f_met:
                f_met.write("step,wheel_effort,joint_effort,velocity,drain,energy_per_meter,energy,damage,state\n")
                
        drain = float(brain.drives.last_drain)
        energy_per_meter = drain / (velocity + 1e-6)
        
        with open(self.metabolic_log_path, 'a') as f_met:
            f_met.write(f"{step},{wheel_effort:.6f},{joint_effort:.6f},{velocity:.6f},{drain:.8f},{energy_per_meter:.8f},{brain.drives.energy:.6f},{brain.drives.damage:.6f},{state}\n")
