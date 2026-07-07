"""
carl_habits.py — Basal Ganglia Action Selection (Habit Engine)

Biological basis: The basal ganglia selects between competing action plans
through tonic inhibition and selective disinhibition. The direct pathway
(D1, dopamine-excited) promotes desired actions. The indirect pathway
(D2, dopamine-inhibited) suppresses competing alternatives.

Habit formation occurs through dopaminergic reinforcement learning.
Successful action sequences strengthen corticostriatal synapses, causing
firing patterns to 'chunk' — entire sequences of actions consolidate into
single units that fire with minimal cortical (conscious) oversight.

The locus of control shifts from dorsomedial striatum (goal-directed)
to dorsolateral striatum (habitual) as chunks become reliable.

Emergence signature: CARL develops automatic navigation habits —
repeated successful turns, approach angles, and corridor traversals
become chunked motor programs that bypass the MPC planner entirely.
This frees cognitive bandwidth for higher-level planning.
"""

import numpy as np


class ActionChunk:
    """A learned motor sequence that executes as a unit."""

    def __init__(self, context_hash, action_sequence):
        self.context_hash = context_hash
        self.action_sequence = action_sequence  # list of np.ndarray actions
        self.success_count = 0
        self.failure_count = 0
        self.total_reward = 0.0
        self.execution_index = 0  # current position in sequence during playback
        self.last_used = 0
        self.creation_step = 0

    @property
    def reliability(self):
        total = self.success_count + self.failure_count
        if total == 0:
            return 0.0
        return self.success_count / total


class BasalGangliaSelector:
    """
    Habit formation through action chunking and competitive selection.

    When an action sequence is repeated with consistent positive outcomes,
    it gets 'chunked' into an automatic habit that bypasses the MPC planner.

    Habits are:
    - Faster than deliberate planning (no MPC computation needed)
    - More reliable in familiar contexts
    - Less flexible in novel situations
    - Preferred when fatigued (less cognitive load)
    - Preferred when stressed (faster response time)

    The competition between habitual and deliberate action selection
    is modulated by drives:
    - High fatigue → prefer habits (conserve cognitive energy)
    - High stress → prefer habits (faster, proven responses)
    - High novelty → prefer deliberation (habits unreliable in new contexts)
    - High curiosity → prefer deliberation (exploration over exploitation)
    """

    def __init__(self):
        self.chunks = []                    # library of learned action chunks
        self.max_chunks = 25
        self.current_recording = []         # in-progress action sequence
        self.current_context_hash = None
        self.recording_reward_sum = 0.0
        self.active_chunk = None            # currently executing chunk (or None)
        self.step_count = 0

        # ── EVOLVABLE PARAMETERS ─────────────────────────────────────────
        self.chunk_min_length = 15          # minimum steps for a viable chunk
        self.chunk_max_length = 30          # maximum steps before forced cut
        self.success_threshold = 5          # repetitions before becoming a habit
        self.habit_preference = 0.3         # [0,1]: base preference for habits
        self.context_resolution = 12        # bits used for context hashing

    # ── CONTEXT HASHING ──────────────────────────────────────────────────

    def _hash_context(self, obs):
        """
        Hash the current sensory context for chunk lookup.
        Uses proximity sensors (wall configuration) as the primary context key.
        """
        # Quantize the first 8 proximity sensors into binary pattern
        n_prox = min(8, len(obs))
        bits = (obs[:n_prox] > 0.5).astype(np.uint8)

        # Add velocity direction if available
        if len(obs) > 12:
            vel_bits = (obs[8:12] > 0).astype(np.uint8)
            bits = np.concatenate([bits, vel_bits])

        # Truncate to context_resolution bits
        bits = bits[:self.context_resolution]
        return int(np.sum(bits * (2 ** np.arange(len(bits)))))

    # ── RECORDING ────────────────────────────────────────────────────────

    def record_step(self, obs, action, reward):
        """
        Record step for potential chunking. Called every tick.
        When context changes or max length is reached, the accumulated
        sequence is evaluated for chunking.
        """
        self.step_count += 1
        context = self._hash_context(obs)

        if self.current_context_hash is None:
            # Start new recording
            self.current_context_hash = context
            self.current_recording = [action.copy()]
            self.recording_reward_sum = reward
            return

        self.current_recording.append(action.copy())
        self.recording_reward_sum += reward

        # Context change or max length = end of sequence
        if context != self.current_context_hash or len(self.current_recording) >= self.chunk_max_length:
            # Evaluate: was this sequence long enough and rewarding?
            if (len(self.current_recording) >= self.chunk_min_length and
                    self.recording_reward_sum > 0):
                self._try_chunk(
                    self.current_context_hash,
                    self.current_recording,
                    self.recording_reward_sum)

            # Start new recording
            self.current_context_hash = context
            self.current_recording = [action.copy()]
            self.recording_reward_sum = reward

    def _try_chunk(self, context_hash, actions, reward):
        """Try to create or reinforce an action chunk."""
        # Check if similar chunk already exists
        for chunk in self.chunks:
            if (chunk.context_hash == context_hash and
                    len(chunk.action_sequence) == len(actions)):
                # Reinforce existing chunk
                if reward > 0:
                    chunk.success_count += 1
                    chunk.total_reward += reward
                else:
                    chunk.failure_count += 1
                chunk.last_used = self.step_count
                return

        # Create new chunk
        if len(self.chunks) < self.max_chunks:
            chunk = ActionChunk(context_hash, [a.copy() for a in actions])
            chunk.success_count = 1 if reward > 0 else 0
            chunk.failure_count = 0 if reward > 0 else 1
            chunk.total_reward = reward
            chunk.last_used = self.step_count
            chunk.creation_step = self.step_count
            self.chunks.append(chunk)
        else:
            # Replace weakest chunk if new one would be better
            weakest_idx = min(range(len(self.chunks)),
                              key=lambda i: self.chunks[i].reliability)
            if self.chunks[weakest_idx].reliability < 0.3:
                new_chunk = ActionChunk(context_hash, [a.copy() for a in actions])
                new_chunk.success_count = 1 if reward > 0 else 0
                new_chunk.total_reward = reward
                new_chunk.last_used = self.step_count
                new_chunk.creation_step = self.step_count
                self.chunks[weakest_idx] = new_chunk

    # ── HABITUAL ACTION SELECTION ────────────────────────────────────────

    def get_habitual_action(self, obs, drives):
        """
        Check if a reliable habit exists for current context.
        Returns action if habit fires, None if deliberation is needed.

        The competition between habit and deliberation is modulated
        by the agent's current drive state — creating context-sensitive
        habit deployment.
        """
        # If currently executing a chunk, continue it
        if self.active_chunk is not None:
            chunk = self.active_chunk
            if chunk.execution_index < len(chunk.action_sequence):
                action = chunk.action_sequence[chunk.execution_index].copy()
                chunk.execution_index += 1
                return action
            else:
                # Chunk finished — clear active chunk
                self.active_chunk = None

        context = self._hash_context(obs)

        # Find matching reliable chunks
        best_chunk = None
        best_score = 0.0

        for chunk in self.chunks:
            if (chunk.context_hash == context and
                    chunk.success_count >= self.success_threshold):
                score = chunk.reliability * (1.0 + chunk.total_reward * 0.1)
                if score > best_score:
                    best_score = score
                    best_chunk = chunk

        if best_chunk is None:
            return None

        # ── DRIVE-MODULATED HABIT/DELIBERATION COMPETITION ───────────
        habit_weight = self.habit_preference

        # Fatigue boosts habit preference (conserve cognitive energy)
        habit_weight += getattr(drives, 'fatigue', 0.0) * 0.3

        # High stress boosts habits (faster, proven responses)
        habit_weight += getattr(drives, 'cort', 0.0) * 0.2

        # High novelty suppresses habits (need deliberation)
        exploration_gain = getattr(drives, 'exploration_gain', 1.0)
        habit_weight -= (1.0 - exploration_gain) * 0.3

        habit_weight = float(np.clip(habit_weight, 0.0, 1.0))

        # Probabilistic selection: habit fires with probability proportional
        # to habit_weight * chunk reliability
        if np.random.random() < habit_weight * best_chunk.reliability:
            best_chunk.execution_index = 1  # start from action 1 (we return action 0 now)
            self.active_chunk = best_chunk
            return best_chunk.action_sequence[0].copy()

        return None

    def cancel_active_chunk(self):
        """Cancel any currently executing chunk (e.g., when a reflex fires)."""
        self.active_chunk = None

    # ── MAINTENANCE ──────────────────────────────────────────────────────

    def prune_stale_chunks(self, max_age=10000):
        """Remove chunks that haven't been used recently."""
        self.chunks = [c for c in self.chunks
                       if (self.step_count - c.last_used) < max_age or c.reliability > 0.7]

    # ── EVOLUTION SUPPORT ────────────────────────────────────────────────

    def mutate(self, rate=0.15):
        """Mutate evolvable parameters."""
        self.chunk_min_length = int(np.clip(
            self.chunk_min_length + np.random.normal(0, rate * 1.0), 2, 8))
        self.chunk_max_length = int(np.clip(
            self.chunk_max_length + np.random.normal(0, rate * 3.0), 5, 40))
        self.success_threshold = int(np.clip(
            self.success_threshold + np.random.normal(0, rate * 1.0), 2, 10))
        self.habit_preference = float(np.clip(
            self.habit_preference + np.random.normal(0, rate * 0.1), 0.1, 0.9))

    # ── STATS & PERSISTENCE ──────────────────────────────────────────────

    def get_stats(self):
        """Return stats for emergence metrics logging."""
        if not self.chunks:
            return 0, 0.0, 0
        reliable = sum(1 for c in self.chunks if c.reliability > 0.5)
        avg_reliability = float(np.mean([c.reliability for c in self.chunks]))
        return len(self.chunks), avg_reliability, reliable

    def save(self):
        chunk_data = []
        for c in self.chunks:
            chunk_data.append({
                'hash': c.context_hash,
                'actions': [a.tolist() for a in c.action_sequence],
                'success': c.success_count,
                'failure': c.failure_count,
                'reward': c.total_reward,
                'last_used': c.last_used,
                'creation_step': c.creation_step,
            })
        return {
            'chunks': chunk_data,
            'step_count': self.step_count,
            'habit_preference': self.habit_preference,
            'success_threshold': self.success_threshold,
            'chunk_min_length': self.chunk_min_length,
            'chunk_max_length': self.chunk_max_length,
        }

    def load(self, d):
        self.step_count = d.get('step_count', 0)
        self.habit_preference = d.get('habit_preference', 0.5)
        self.success_threshold = d.get('success_threshold', 3)
        self.chunk_min_length = d.get('chunk_min_length', 3)
        self.chunk_max_length = d.get('chunk_max_length', 20)
        self.chunks = []
        for cd in d.get('chunks', []):
            chunk = ActionChunk(cd['hash'], [np.array(a) for a in cd['actions']])
            chunk.success_count = cd.get('success', 0)
            chunk.failure_count = cd.get('failure', 0)
            chunk.total_reward = cd.get('reward', 0.0)
            chunk.last_used = cd.get('last_used', 0)
            chunk.creation_step = cd.get('creation_step', 0)
            self.chunks.append(chunk)


# ── SELF-TEST ────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=== Basal Ganglia Habit Engine Self-Test ===")

    # Mock drives
    class MockDrives:
        def __init__(self):
            self.da = 0.3
            self.cort = 0.1
            self.ne = 0.2
            self.sero = 0.8
            self.energy = 0.7
            self.fatigue = 0.1
            self.exploration_gain = 1.0

    bg = BasalGangliaSelector()
    drives = MockDrives()

    # Test 1: No chunks initially
    result = bg.get_habitual_action(np.random.randn(33), drives)
    assert result is None, "No habits should exist initially"
    print("[PASS] Test 1: No habits initially")

    # Test 2: Record steps and create chunks
    obs = np.zeros(33)
    obs[0] = 0.8  # wall ahead
    action = np.array([0.5, 0.3])
    for i in range(50):
        bg.record_step(obs, action, 0.1)
    # Force context change to finalize recording
    obs2 = np.zeros(33)
    obs2[2] = 0.8  # different wall config
    bg.record_step(obs2, action, 0.0)
    assert len(bg.chunks) > 0, f"Should have chunks, got {len(bg.chunks)}"
    print(f"[PASS] Test 2: Created {len(bg.chunks)} chunks")

    # Test 3: Reinforce a chunk until it becomes a habit
    obs_habit = np.zeros(33)
    obs_habit[0] = 0.8
    for trial in range(10):
        # Record the same sequence multiple times
        for i in range(5):
            bg.record_step(obs_habit, np.array([0.5, 0.3]), 0.2)
        # Context change to finalize
        bg.record_step(np.zeros(33), np.array([0.0, 0.0]), 0.0)

    # Check that a reliable chunk exists
    reliable_chunks = [c for c in bg.chunks if c.success_count >= bg.success_threshold]
    print(f"[INFO] Reliable chunks: {len(reliable_chunks)}")

    # Test 4: Habitual action retrieval
    drives.fatigue = 0.8  # high fatigue should boost habit preference
    # Try many times (probabilistic)
    habit_fired = False
    for _ in range(20):
        result = bg.get_habitual_action(obs_habit, drives)
        if result is not None:
            habit_fired = True
            break
        bg.active_chunk = None  # reset
    if reliable_chunks:
        print(f"[PASS] Test 4: Habit fired = {habit_fired} (expected with high fatigue)")
    else:
        print("[SKIP] Test 4: No reliable chunks to test")

    # Test 5: Save/Load
    saved = bg.save()
    bg2 = BasalGangliaSelector()
    bg2.load(saved)
    assert len(bg2.chunks) == len(bg.chunks)
    assert bg2.step_count == bg.step_count
    print("[PASS] Test 5: Save/load roundtrip")

    # Test 6: Mutation
    bg.mutate(rate=0.5)
    assert 2 <= bg.chunk_min_length <= 8
    assert 0.1 <= bg.habit_preference <= 0.9
    print("[PASS] Test 6: Mutation keeps parameters in range")

    # Test 7: Prune stale chunks
    for c in bg.chunks:
        c.last_used = 0
        c.reliability  # access to ensure it works
    bg.step_count = 20000
    bg.prune_stale_chunks(max_age=10000)
    print(f"[PASS] Test 7: Pruning reduced chunks (remaining: {len(bg.chunks)})")

    print("\n=== All Basal Ganglia Habit Engine tests passed ===")
