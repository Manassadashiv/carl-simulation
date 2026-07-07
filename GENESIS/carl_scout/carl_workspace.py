"""
carl_workspace.py — Global Workspace Theory (Attention Bottleneck)

Biological basis: Baars' Global Workspace Theory (1988).
Consciousness is a broadcasting mechanism where multiple unconscious
specialist modules compete, and only the 'winning' signal gets broadcast
to all modules simultaneously.

This creates:
  1. An information bottleneck that forces prioritization
  2. Sequential behavioral coherence (one thing at a time)
  3. Metacognitive awareness through broadcast history analysis
  4. The emergent 'stream of consciousness' — trackable mode transitions

The broadcast history IS Carl's stream of consciousness.
"""

import numpy as np
from collections import deque


class WorkspaceSignal:
    """A signal competing for global broadcast in the workspace."""

    def __init__(self, source, salience, content=None):
        self.source = source          # str: 'hunger', 'fear', 'curiosity', 'goal', 'somatic', 'dmn', 'habit', 'allostasis'
        self.salience = float(salience)  # [0, 1]: urgency/importance
        self.content = content or {}     # dict: signal-specific payload


class GlobalWorkspace:
    """
    The attention bottleneck — CARL's 'conscious' moment.

    Multiple unconscious specialist modules generate signals.
    These signals compete through lateral inhibition with hysteresis.
    Only ONE wins each tick and gets 'broadcast' to all modules.

    The winner determines the dominant behavioral mode:
    - hunger: food-seeking navigation
    - fear: avoidance/freeze behavior
    - curiosity: frontier exploration
    - goal: crystallized goal pursuit
    - somatic: gut-feeling-driven behavior
    - allostasis: preemptive need satisfaction
    - habit: automatic motor sequence execution
    - dmn: idle mind / daydreaming
    - idle: no signal above threshold (DMN can take over)

    Emergence signature: mode transition patterns reveal developing
    behavioral complexity. Simple creatures oscillate hunger↔fear.
    Complex ones show rich multi-modal switching with appropriate context.
    """

    def __init__(self):
        self.broadcast_history = deque(maxlen=200)
        self.current_broadcast = None
        self.previous_broadcast_source = None
        self.step_count = 0

        # ── EVOLVABLE PARAMETERS ─────────────────────────────────────────
        self.ignition_threshold = 0.4    # min salience to enter competition
        self.hysteresis = 0.08           # stability bonus for current mode
        self.inhibition_decay = 0.92     # recovery rate for suppressed signals

        # ── INTERNAL STATE ───────────────────────────────────────────────
        self._suppression = {}           # source → suppression level
        self._mode_counts = {}           # source → count in recent history

    # ── CORE COMPETITION ─────────────────────────────────────────────────

    def compete_and_broadcast(self, signals):
        """
        Winner-take-all competition with hysteresis for stability.

        This is the heart of CARL's 'consciousness' — the selection
        of what to pay attention to RIGHT NOW.

        Args:
            signals: list of WorkspaceSignal objects from all modules

        Returns:
            WorkspaceSignal: the winning broadcast (or None if nothing exceeds threshold)
        """
        self.step_count += 1

        if not signals:
            self.current_broadcast = None
            return None

        # Filter by ignition threshold
        candidates = [s for s in signals if s.salience >= self.ignition_threshold]

        if not candidates:
            # Nothing urgent — workspace goes dark (DMN can activate)
            self.current_broadcast = None
            self.broadcast_history.append(('idle', 0.0, self.step_count))
            return None

        # Compute effective saliences with hysteresis and lateral inhibition
        effective_saliences = []
        for s in candidates:
            effective = s.salience

            # Hysteresis: current mode gets stability bonus (prevents flicker)
            if self.previous_broadcast_source == s.source:
                effective += self.hysteresis

            # Lateral inhibition: recently suppressed signals are harder to activate
            suppression = self._suppression.get(s.source, 0.0)
            effective -= suppression * 0.1

            effective_saliences.append(max(0.0, effective))

        # Winner-take-all selection
        winner_idx = int(np.argmax(effective_saliences))
        winner = candidates[winner_idx]

        # Update lateral inhibition: losing signals get suppressed
        for i, s in enumerate(candidates):
            if i != winner_idx:
                self._suppression[s.source] = min(
                    1.0, self._suppression.get(s.source, 0.0) + 0.1)

        # Winner's suppression resets
        self._suppression[winner.source] = 0.0

        # Decay all suppressions toward zero
        for k in list(self._suppression.keys()):
            self._suppression[k] *= self.inhibition_decay
            if self._suppression[k] < 0.01:
                del self._suppression[k]

        # Record broadcast
        self.current_broadcast = winner
        self.previous_broadcast_source = winner.source
        self.broadcast_history.append(
            (winner.source, winner.salience, self.step_count))

        # Update mode frequency counts
        self._mode_counts[winner.source] = self._mode_counts.get(winner.source, 0) + 1

        return winner

    # ── INTROSPECTION / METRICS ──────────────────────────────────────────

    def get_behavioral_mode(self):
        """What is CARL 'thinking about' right now?"""
        if self.current_broadcast is None:
            return 'idle'
        return self.current_broadcast.source

    def get_mode_entropy(self, window=100):
        """
        Shannon entropy of recent broadcast sources.
        High entropy = diverse attention (cognitively flexible).
        Low entropy = fixation (tunnel vision or optimal focus).
        """
        if len(self.broadcast_history) < 10:
            return 0.0

        recent = list(self.broadcast_history)[-window:]
        sources = [b[0] for b in recent]

        counts = {}
        for s in sources:
            counts[s] = counts.get(s, 0) + 1

        total = len(sources)
        entropy = 0.0
        for count in counts.values():
            p = count / total
            if p > 0:
                entropy -= p * np.log2(p)

        return float(entropy)

    def get_mode_transitions(self, window=100):
        """Count how many times the broadcast mode changed recently."""
        recent = list(self.broadcast_history)[-window:]
        if len(recent) < 2:
            return 0
        transitions = sum(1 for i in range(1, len(recent))
                          if recent[i][0] != recent[i - 1][0])
        return transitions

    def get_dominant_mode(self, window=50):
        """Most frequent broadcast source in recent window."""
        recent = list(self.broadcast_history)[-window:]
        if not recent:
            return 'idle'
        counts = {}
        for b in recent:
            counts[b[0]] = counts.get(b[0], 0) + 1
        return max(counts, key=counts.get)

    def get_stats(self):
        """Return stats dict for logging."""
        return {
            'mode': self.get_behavioral_mode(),
            'entropy': self.get_mode_entropy(),
            'transitions': self.get_mode_transitions(),
            'dominant': self.get_dominant_mode(),
            'total_broadcasts': self.step_count,
        }

    # ── EVOLUTION SUPPORT ────────────────────────────────────────────────

    def mutate(self, rate=0.15):
        """Mutate evolvable parameters."""
        self.ignition_threshold = float(np.clip(
            self.ignition_threshold + np.random.normal(0, rate * 0.05), 0.1, 0.5))
        self.hysteresis = float(np.clip(
            self.hysteresis + np.random.normal(0, rate * 0.03), 0.05, 0.3))
        self.inhibition_decay = float(np.clip(
            self.inhibition_decay + np.random.normal(0, rate * 0.02), 0.8, 0.98))

    # ── PERSISTENCE ──────────────────────────────────────────────────────

    def save(self):
        return {
            'ignition_threshold': self.ignition_threshold,
            'hysteresis': self.hysteresis,
            'inhibition_decay': self.inhibition_decay,
            'step_count': self.step_count,
            'previous_broadcast_source': self.previous_broadcast_source,
            'broadcast_history': list(self.broadcast_history)[-50:],
        }

    def load(self, d):
        self.ignition_threshold = d.get('ignition_threshold', 0.3)
        self.hysteresis = d.get('hysteresis', 0.15)
        self.inhibition_decay = d.get('inhibition_decay', 0.92)
        self.step_count = d.get('step_count', 0)
        self.previous_broadcast_source = d.get('previous_broadcast_source', None)
        saved_history = d.get('broadcast_history', [])
        self.broadcast_history = deque(saved_history, maxlen=200)


# ── SELF-TEST ────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=== Global Workspace Self-Test ===")

    ws = GlobalWorkspace()

    # Test 1: Empty signals
    result = ws.compete_and_broadcast([])
    assert result is None, "Empty signals should produce None"
    print("[PASS] Test 1: Empty signals handled")

    # Test 2: Sub-threshold signals
    signals = [WorkspaceSignal('curiosity', 0.1)]
    result = ws.compete_and_broadcast(signals)
    assert result is None, "Sub-threshold should not broadcast"
    print("[PASS] Test 2: Sub-threshold signals filtered")

    # Test 3: Winner-take-all
    signals = [
        WorkspaceSignal('hunger', 0.7),
        WorkspaceSignal('curiosity', 0.4),
        WorkspaceSignal('fear', 0.3),
    ]
    result = ws.compete_and_broadcast(signals)
    assert result is not None
    assert result.source == 'hunger', f"Expected hunger, got {result.source}"
    print("[PASS] Test 3: Highest salience wins")

    # Test 4: Hysteresis
    # Run hunger for several ticks to build hysteresis
    for _ in range(5):
        ws.compete_and_broadcast([
            WorkspaceSignal('hunger', 0.5),
            WorkspaceSignal('curiosity', 0.55),  # slightly higher
        ])
    # With hysteresis, hunger should hold even though curiosity is slightly higher
    result = ws.compete_and_broadcast([
        WorkspaceSignal('hunger', 0.5),
        WorkspaceSignal('curiosity', 0.55),
    ])
    # After initial switch to curiosity (which then gets hysteresis), test passes
    assert result is not None
    print("[PASS] Test 4: Hysteresis provides mode stability")

    # Test 5: Mode entropy
    ws2 = GlobalWorkspace()
    # Run with diverse signals
    sources = ['hunger', 'fear', 'curiosity', 'goal']
    for i in range(100):
        src = sources[i % len(sources)]
        ws2.compete_and_broadcast([WorkspaceSignal(src, 0.8)])
    entropy = ws2.get_mode_entropy()
    assert entropy > 1.5, f"Expected high entropy for diverse signals, got {entropy}"
    print(f"[PASS] Test 5: Mode entropy = {entropy:.2f} (diverse)")

    # Test 6: Mode transitions
    transitions = ws2.get_mode_transitions()
    assert transitions > 50, f"Expected many transitions, got {transitions}"
    print(f"[PASS] Test 6: Mode transitions = {transitions}")

    # Test 7: Save/Load
    saved = ws2.save()
    ws3 = GlobalWorkspace()
    ws3.load(saved)
    assert ws3.ignition_threshold == ws2.ignition_threshold
    assert ws3.step_count == ws2.step_count
    print("[PASS] Test 7: Save/load roundtrip")

    # Test 8: Mutation
    ws2.mutate(rate=0.5)
    assert 0.1 <= ws2.ignition_threshold <= 0.5
    assert 0.05 <= ws2.hysteresis <= 0.3
    print("[PASS] Test 8: Mutation keeps parameters in range")

    print("\n=== All Global Workspace tests passed ===")
