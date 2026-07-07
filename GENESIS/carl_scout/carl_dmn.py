"""
carl_dmn.py — Default Mode Network (Idle Mind Processing)

Biological basis: The DMN activates during 'rest' states when no externally-
directed task demands attention. It is NOT rest — it is active self-referential
processing that performs:

  1. Episodic replay: re-experience and re-evaluate past events
  2. Mental simulation: run 'what-if' scenarios using the world model
  3. Spontaneous goal generation: discover potentially rewarding locations
  4. Self-model calibration: update the agent's model of itself

The DMN is anti-correlated with the task-positive network (workspace).
When external demands drop (workspace goes dark), DMN activation rises.
This creates a natural rhythm: engage -> act -> disengage -> consolidate -> plan.

Emergence signature: DMN-generated goals represent genuine 'creative' behavior —
the agent imagining possibilities that were never directly experienced.
"""

import numpy as np


class DefaultModeNetwork:
    """
    CARL's idle-mind processing — activates when nothing urgent demands attention.

    When the Global Workspace has no broadcast (low salience), the DMN activates
    and begins background cognitive processing that cannot run during task execution.

    Key functions:
    - Buffer recent experiences for offline replay
    - Replay emotionally-significant experiences to strengthen somatic markers
    - Use the Dreamer world model to imagine hypothetical trajectories
    - Generate spontaneous goals from imagination (prospection)
    """

    def __init__(self):
        self.activation_level = 0.0         # [0, 1]: how active is the DMN
        self.replay_buffer = []             # list of (obs, action, reward, drives_snap) tuples
        self.max_replay_buffer = 500
        self.imagined_goals = []            # list of dicts with 'value' and 'trajectory'
        self.max_imagined_goals = 5

        # Self-model metrics
        self.self_prediction_accuracy = 0.5   # how well CARL predicts its own behavior
        self.rumination_count = 0             # how many replay cycles run
        self.total_idle_steps = 0             # total DMN processing steps

        # ── EVOLVABLE PARAMETERS ─────────────────────────────────────────
        self.activation_threshold = 0.4     # workspace salience below this activates DMN
        self.replay_fraction = 0.1          # fraction of buffer to replay per idle step
        self.imagination_horizon = 5        # steps to imagine forward
        self.deactivation_rate = 0.1        # how fast DMN deactivates when interrupted

    # ── EXPERIENCE RECORDING ─────────────────────────────────────────────

    def record_experience(self, obs, action, reward, drives):
        """
        Buffer experience for later replay. Called every step.
        The drives snapshot preserves the emotional context of the experience.
        """
        drives_snap = {
            'da': drives.da,
            'cort': drives.cort,
            'ne': drives.ne,
            'sero': drives.sero,
            'energy': drives.energy,
            'fatigue': drives.fatigue,
        }
        self.replay_buffer.append((obs.copy(), action.copy(), float(reward), drives_snap))
        if len(self.replay_buffer) > self.max_replay_buffer:
            self.replay_buffer.pop(0)

    # ── ACTIVATION CONTROL ───────────────────────────────────────────────

    def should_activate(self, workspace_broadcast, is_sleeping=False):
        """
        DMN activates when workspace goes dark (nothing urgent) or during sleep.
        Anti-correlated with task-positive network.

        Returns: True if DMN should run idle processing this tick.
        """
        if is_sleeping:
            # Sleep state forces DMN activation
            self.activation_level = min(1.0, self.activation_level + 0.08)
        elif workspace_broadcast is None:
            # No broadcast — gradually activate DMN
            self.activation_level = min(1.0, self.activation_level + 0.05)
        elif workspace_broadcast.salience < self.activation_threshold:
            # Low-salience broadcast — partial activation
            self.activation_level = min(1.0, self.activation_level + 0.02)
        else:
            # High-salience broadcast — rapidly deactivate DMN
            self.activation_level = max(0.0, self.activation_level - self.deactivation_rate)

        return self.activation_level > 0.5

    # ── IDLE PROCESSING ──────────────────────────────────────────────────

    def idle_step(self, somatic_engine=None):
        """
        One step of idle processing.
        Replays random past experiences and evaluates their emotional valence.
        If somatic_engine is provided, replay can strengthen markers.

        This is CARL 'daydreaming' — processing past experiences offline
        to extract patterns and emotional associations.
        """
        if len(self.replay_buffer) < 20:
            return

        self.rumination_count += 1
        self.total_idle_steps += 1

        # Sample random experiences for replay
        n_replay = max(1, int(len(self.replay_buffer) * self.replay_fraction))
        n_replay = min(n_replay, 5)  # cap to keep computation light
        indices = np.random.choice(len(self.replay_buffer), size=n_replay, replace=False)

        for idx in indices:
            obs, action, reward, drives_snap = self.replay_buffer[idx]

            # Re-experience the emotional valence of this memory
            valence = drives_snap.get('da', 0) - drives_snap.get('cort', 0)

            # High-valence experiences (positive or negative) get extra processing
            if abs(valence) > 0.4 and somatic_engine is not None:
                # The act of replaying can strengthen somatic markers
                # (This is handled by the somatic engine's consolidation)
                pass  # Marker strengthening happens in sleep_consolidation

            # Update self-prediction accuracy based on replay outcomes
            # (Did the reward match what I'd expect from my current model?)
            if abs(reward) > 0.01:
                expected_direction = 1.0 if valence > 0 else -1.0
                actual_direction = 1.0 if reward > 0 else -1.0
                match = 1.0 if expected_direction == actual_direction else 0.0
                self.self_prediction_accuracy = (
                    0.99 * self.self_prediction_accuracy + 0.01 * match)

    def imagine_and_evaluate(self, dreamer, actor, current_obs):
        """
        Use the Dreamer world model to imagine hypothetical trajectories.
        If any trajectory leads to predicted novelty, generate a spontaneous goal.

        This is CARL's 'prospection' — imagining futures that were never experienced.

        Returns: list of dicts with 'value' and 'trajectory' keys
        """
        if dreamer is None or current_obs is None:
            return []

        self.imagined_goals.clear()

        # Imagine N different action sequences from current state
        for _ in range(3):
            try:
                # Use dreamer's built-in imagination capability
                obs_24 = current_obs[:24] if len(current_obs) > 24 else current_obs
                trajectory = dreamer.imagine_rollout(
                    obs_24, actor, steps=self.imagination_horizon)

                if trajectory:
                    # Evaluate: how interesting was this imagined future?
                    total_predicted_novelty = 0.0
                    for obs_t, act_t, next_obs_t in trajectory:
                        pred_error = float(np.mean((next_obs_t - obs_t) ** 2))
                        total_predicted_novelty += pred_error

                    if total_predicted_novelty > 0.1:
                        self.imagined_goals.append({
                            'value': total_predicted_novelty,
                            'trajectory': trajectory,
                        })
            except Exception:
                pass  # imagination can fail early in training

        # Sort by predicted value and keep top goals
        self.imagined_goals.sort(key=lambda g: g['value'], reverse=True)
        self.imagined_goals = self.imagined_goals[:self.max_imagined_goals]
        return self.imagined_goals

    # ── EVOLUTION SUPPORT ────────────────────────────────────────────────

    def mutate(self, rate=0.15):
        """Mutate evolvable parameters."""
        self.activation_threshold = float(np.clip(
            self.activation_threshold + np.random.normal(0, rate * 0.05), 0.2, 0.6))
        self.replay_fraction = float(np.clip(
            self.replay_fraction + np.random.normal(0, rate * 0.02), 0.02, 0.3))
        self.imagination_horizon = int(np.clip(
            self.imagination_horizon + np.random.normal(0, rate * 1.0), 3, 10))
        self.deactivation_rate = float(np.clip(
            self.deactivation_rate + np.random.normal(0, rate * 0.02), 0.05, 0.3))

    # ── STATS & PERSISTENCE ──────────────────────────────────────────────

    def get_stats(self):
        """Return stats for emergence metrics logging."""
        return {
            'activation_level': round(self.activation_level, 3),
            'replay_buffer_size': len(self.replay_buffer),
            'imagined_goals': len(self.imagined_goals),
            'rumination_count': self.rumination_count,
            'total_idle_steps': self.total_idle_steps,
            'self_prediction_accuracy': round(self.self_prediction_accuracy, 3),
        }

    def save(self):
        return {
            'activation_level': self.activation_level,
            'activation_threshold': self.activation_threshold,
            'replay_fraction': self.replay_fraction,
            'imagination_horizon': self.imagination_horizon,
            'deactivation_rate': self.deactivation_rate,
            'rumination_count': self.rumination_count,
            'total_idle_steps': self.total_idle_steps,
            'self_prediction_accuracy': self.self_prediction_accuracy,
            # Don't save replay buffer — it's ephemeral per-lifetime
        }

    def load(self, d):
        self.activation_level = d.get('activation_level', 0.0)
        self.activation_threshold = d.get('activation_threshold', 0.4)
        self.replay_fraction = d.get('replay_fraction', 0.1)
        self.imagination_horizon = d.get('imagination_horizon', 5)
        self.deactivation_rate = d.get('deactivation_rate', 0.1)
        self.rumination_count = d.get('rumination_count', 0)
        self.total_idle_steps = d.get('total_idle_steps', 0)
        self.self_prediction_accuracy = d.get('self_prediction_accuracy', 0.5)


# ── SELF-TEST ────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=== Default Mode Network Self-Test ===")

    # Mock drives
    class MockDrives:
        def __init__(self):
            self.da = 0.3
            self.cort = 0.1
            self.ne = 0.2
            self.sero = 0.8
            self.energy = 0.7
            self.fatigue = 0.1

    # Mock workspace signal
    class MockBroadcast:
        def __init__(self, salience):
            self.salience = salience
            self.source = 'test'

    dmn = DefaultModeNetwork()
    drives = MockDrives()

    # Test 1: Activation with no broadcast
    for _ in range(20):
        active = dmn.should_activate(None)
    assert active, "DMN should activate after sustained no-broadcast"
    assert dmn.activation_level > 0.5
    print("[PASS] Test 1: DMN activates when workspace is dark")

    # Test 2: Deactivation with high-salience broadcast
    for _ in range(20):
        active = dmn.should_activate(MockBroadcast(0.8))
    assert not active, "DMN should deactivate with high salience broadcast"
    assert dmn.activation_level < 0.5
    print("[PASS] Test 2: DMN deactivates with urgent broadcast")

    # Test 3: Experience recording
    for i in range(30):
        obs = np.random.randn(33)
        action = np.random.randn(2)
        dmn.record_experience(obs, action, float(i * 0.01), drives)
    assert len(dmn.replay_buffer) == 30
    print("[PASS] Test 3: Experience recording works")

    # Test 4: Idle step
    dmn.activation_level = 1.0
    dmn.idle_step(somatic_engine=None)
    assert dmn.rumination_count == 1
    print("[PASS] Test 4: Idle step runs replay")

    # Test 5: Buffer overflow
    for i in range(600):
        obs = np.random.randn(33)
        action = np.random.randn(2)
        dmn.record_experience(obs, action, 0.0, drives)
    assert len(dmn.replay_buffer) == dmn.max_replay_buffer
    print("[PASS] Test 5: Buffer respects max size")

    # Test 6: Save/Load
    saved = dmn.save()
    dmn2 = DefaultModeNetwork()
    dmn2.load(saved)
    assert dmn2.rumination_count == dmn.rumination_count
    assert abs(dmn2.self_prediction_accuracy - dmn.self_prediction_accuracy) < 0.001
    print("[PASS] Test 6: Save/load roundtrip")

    # Test 7: Mutation
    dmn.mutate(rate=0.5)
    assert 0.2 <= dmn.activation_threshold <= 0.6
    assert 3 <= dmn.imagination_horizon <= 10
    print("[PASS] Test 7: Mutation keeps parameters in range")

    print("\n=== All Default Mode Network tests passed ===")
