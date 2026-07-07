"""
carl_somatic.py — Somatic Marker Engine

Biological basis: Damasio's somatic marker hypothesis (1994).
The ventromedial prefrontal cortex (vmPFC) and amygdala store associations
between situations and their emotional outcomes. When a similar situation
is encountered later, these markers fire BEFORE conscious deliberation,
biasing behavior through 'gut feelings.'

Computational flow:
  1. Encode current sensory state as compressed HDC hash
  2. Compare against stored markers using fast hash lookup + cosine verification
  3. If match found → inject valence into drives BEFORE action selection
  4. After action outcomes → create/update markers based on emotional intensity

This is CARL's subconscious — the felt sense of a situation
that guides behavior without explicit reasoning.
"""

import numpy as np


class SomaticMarker:
    """A single somatic marker: a compressed body-state tag bound to a situation."""

    def __init__(self, situation_hash, valence, arousal, creation_step):
        self.situation_hash = situation_hash       # int64 fast-lookup key
        self.situation_vector = None               # int8 signs of full HDC vector
        self.valence = float(valence)              # [-1, +1]: avoid ↔ approach
        self.arousal = float(arousal)              # [0, 1]: body response intensity
        self.creation_step = int(creation_step)
        self.activation_count = 0
        self.last_activated = int(creation_step)
        self.strength = 1.0                        # decays per step, boosted by consolidation


class SomaticMarkerEngine:
    """
    Maintains a library of somatic markers that fire pre-consciously.

    Every experience gets tagged with a somatic marker (a compressed emotional
    signature). When CARL encounters a similar situation later, the marker fires
    *before* the MPC planner runs, biasing action selection through pre-conscious
    emotional priming.

    Emergence signature: over generations, CARL develops 'intuitions' about
    dangerous zones and rewarding corridors that guide navigation without
    explicit goal representation.
    """

    def __init__(self, hdc_engine, max_markers=200):
        self.hdc = hdc_engine
        self.markers = []
        self.max_markers = max_markers
        self.current_step = 0

        # ── EVOLVABLE PARAMETERS ─────────────────────────────────────────
        self.activation_threshold = 0.12    # cosine sim for marker firing
        self.creation_threshold = 0.75      # min emotional intensity to create
        self.marker_strength = 0.3          # how strongly markers bias drives
        self.marker_decay_rate = 0.995      # per-step strength decay

        # ── PRE-CONSCIOUS OUTPUT (read by brain before action selection) ──
        self.approach_bias = 0.0            # positive = approach, negative = avoid
        self.arousal_injection = 0.0
        self.active_marker_count = 0

    # ── FAST HASHING ─────────────────────────────────────────────────────

    def _hash_vector(self, vec):
        """Fast hash of a bipolar HDC vector for O(1) pre-filter."""
        bits = (vec[:64] > 0).astype(np.uint8)
        # Pack 64 bits into an int for fast equality check
        packed = np.packbits(bits)
        return int(np.sum(packed.astype(np.int64) * (256 ** np.arange(len(packed)))))

    def _cosine_sim(self, v1_signs, v2):
        """Cosine similarity: stored int8 signs vs current float vector."""
        dot = float(np.dot(v1_signs.astype(np.float32), np.sign(v2)))
        return dot / len(v1_signs)

    # ── CORE OPERATIONS ──────────────────────────────────────────────────

    def pre_conscious_bias(self, obs_vector, drives):
        """
        Called BEFORE action selection each step.
        Scans markers for situation matches and injects emotional bias.

        This IS the gut feeling — it modifies drives before the MPC planner sees them.

        Returns: (approach_bias, arousal_injection)
        """
        self.current_step += 1
        S = self.hdc.encode_sensory(obs_vector)
        current_hash = self._hash_vector(S)

        self.approach_bias = 0.0
        self.arousal_injection = 0.0
        self.active_marker_count = 0

        for marker in self.markers:
            # Fast hash pre-filter: skip markers that can't possibly match
            # But also check cosine for hash collisions and near-matches
            if marker.situation_vector is not None:
                sim = self._cosine_sim(marker.situation_vector, S)
                if sim > self.activation_threshold:
                    # Marker fires!
                    effective_strength = marker.strength * marker.arousal * sim
                    self.approach_bias += marker.valence * effective_strength * self.marker_strength
                    self.arousal_injection += marker.arousal * effective_strength
                    marker.activation_count += 1
                    marker.last_activated = self.current_step
                    self.active_marker_count += 1

        # Clamp outputs
        self.approach_bias = float(np.clip(self.approach_bias, -1.0, 1.0))
        self.arousal_injection = float(np.clip(self.arousal_injection, 0.0, 1.0))

        # Inject into drives (the 'gut feeling')
        if abs(self.approach_bias) > 0.01:
            if self.approach_bias > 0:
                # Positive marker → dopamine nudge (approach)
                drives.da = float(np.clip(drives.da + self.approach_bias * 0.15, -1.0, 1.0))
            else:
                # Negative marker → cortisol nudge (avoid)
                drives.cort = float(np.clip(drives.cort - self.approach_bias * 0.15, 0.0, 1.0))

        return self.approach_bias, self.arousal_injection

    def post_experience_update(self, obs_vector, drives):
        """
        Called AFTER action outcomes are known.
        Creates new markers for emotionally intense experiences.
        Updates existing markers based on outcome similarity.
        """
        # Only create markers for emotionally significant events
        emotional_intensity = max(abs(drives.da), drives.cort, drives.ne)
        if emotional_intensity < self.creation_threshold:
            return

        S = self.hdc.encode_sensory(obs_vector)

        # Compute valence from current drives
        valence = float(np.clip(drives.da - drives.cort, -1.0, 1.0))

        # Check if a marker already exists for this situation
        for marker in self.markers:
            if marker.situation_vector is not None:
                sim = self._cosine_sim(marker.situation_vector, S)
                if sim > self.activation_threshold * 1.5:  # tighter match for update
                    # Update existing marker with exponential moving average
                    marker.valence = 0.7 * marker.valence + 0.3 * valence
                    marker.arousal = 0.7 * marker.arousal + 0.3 * emotional_intensity
                    marker.strength = min(1.0, marker.strength + 0.1)
                    return

        # Create new marker
        new_marker = SomaticMarker(
            self._hash_vector(S), valence, emotional_intensity, self.current_step
        )
        new_marker.situation_vector = np.sign(S).astype(np.int8)
        self.markers.append(new_marker)

        # Capacity management: remove weakest marker if over limit
        if len(self.markers) > self.max_markers:
            weakest_idx = min(range(len(self.markers)),
                              key=lambda i: self.markers[i].strength)
            self.markers.pop(weakest_idx)

    def decay_step(self):
        """Per-step decay of all markers. Called each simulation tick."""
        for marker in self.markers:
            marker.strength *= self.marker_decay_rate
        # Prune dead markers
        self.markers = [m for m in self.markers if m.strength > 0.01]

    def sleep_consolidation(self):
        """
        During sleep: strengthen frequently-activated markers, prune weak ones.
        Biological analogue: slow-wave sleep consolidation in vmPFC.
        """
        for marker in self.markers:
            if marker.activation_count > 3:
                marker.strength = min(1.0, marker.strength * 1.2)
            elif marker.activation_count == 0 and marker.strength < 0.3:
                marker.strength *= 0.5  # accelerate pruning of unused markers
        self.markers = [m for m in self.markers if m.strength > 0.01]

    # ── EVOLUTION SUPPORT ────────────────────────────────────────────────

    def mutate(self, rate=0.15):
        """Mutate evolvable parameters."""
        self.activation_threshold = float(np.clip(
            self.activation_threshold + np.random.normal(0, rate * 0.03), 0.05, 0.3))
        self.creation_threshold = float(np.clip(
            self.creation_threshold + np.random.normal(0, rate * 0.1), 0.3, 0.9))
        self.marker_strength = float(np.clip(
            self.marker_strength + np.random.normal(0, rate * 0.1), 0.1, 0.8))
        self.marker_decay_rate = float(np.clip(
            self.marker_decay_rate + np.random.normal(0, rate * 0.0005), 0.995, 0.9999))

    # ── STATS & PERSISTENCE ──────────────────────────────────────────────

    def get_stats(self):
        """Return stats for emergence metrics logging."""
        if not self.markers:
            return 0, 0.0, 0.0
        avg_valence = float(np.mean([m.valence for m in self.markers]))
        avg_strength = float(np.mean([m.strength for m in self.markers]))
        return len(self.markers), avg_valence, avg_strength

    def save(self):
        marker_data = []
        for m in self.markers:
            marker_data.append({
                'hash': m.situation_hash,
                'vec': m.situation_vector.copy() if m.situation_vector is not None else None,
                'valence': m.valence,
                'arousal': m.arousal,
                'strength': m.strength,
                'creation_step': m.creation_step,
                'activation_count': m.activation_count,
                'last_activated': m.last_activated,
            })
        return {
            'markers': marker_data,
            'current_step': self.current_step,
            'activation_threshold': self.activation_threshold,
            'creation_threshold': self.creation_threshold,
            'marker_strength': self.marker_strength,
            'marker_decay_rate': self.marker_decay_rate,
        }

    def load(self, d):
        self.current_step = d.get('current_step', 0)
        self.activation_threshold = d.get('activation_threshold', 0.12)
        self.creation_threshold = d.get('creation_threshold', 0.6)
        self.marker_strength = d.get('marker_strength', 0.3)
        self.marker_decay_rate = d.get('marker_decay_rate', 0.9995)
        self.markers = []
        for md in d.get('markers', []):
            m = SomaticMarker(md['hash'], md['valence'], md['arousal'], md['creation_step'])
            m.situation_vector = md.get('vec')
            m.strength = md.get('strength', 1.0)
            m.activation_count = md.get('activation_count', 0)
            m.last_activated = md.get('last_activated', 0)
            self.markers.append(m)


# ── SELF-TEST ────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=== Somatic Marker Engine Self-Test ===")

    # Mock HDC engine
    class MockHDC:
        def __init__(self, dim=40000):
            self.D = dim
            self._rng = np.random.default_rng(42)
        def encode_sensory(self, obs):
            return self._rng.choice([-1, 1], size=self.D).astype(np.float32)

    # Mock drives
    class MockDrives:
        def __init__(self):
            self.da = 0.0
            self.ne = 0.0
            self.cort = 0.0
            self.sero = 1.0
            self.energy = 1.0
            self.fatigue = 0.0
            self.damage = 0.0
            self.hunger = 0.0

    hdc = MockHDC()
    engine = SomaticMarkerEngine(hdc, max_markers=50)
    drives = MockDrives()

    # Test 1: Pre-conscious bias with no markers
    bias, arousal = engine.pre_conscious_bias(np.random.randn(33), drives)
    assert abs(bias) < 0.01, "Empty engine should produce no bias"
    print("[PASS] Test 1: No markers = no bias")

    # Test 2: Create a marker with high emotional intensity
    drives.da = 0.8
    drives.cort = 0.1
    drives.ne = 0.7
    obs = np.random.randn(33)
    engine.post_experience_update(obs, drives)
    assert len(engine.markers) == 1, f"Expected 1 marker, got {len(engine.markers)}"
    assert engine.markers[0].valence > 0, "DA > CORT should give positive valence"
    print("[PASS] Test 2: Marker created with correct valence")

    # Test 3: Decay step
    initial_strength = engine.markers[0].strength
    for _ in range(100):
        engine.decay_step()
    assert engine.markers[0].strength < initial_strength, "Strength should decay"
    print("[PASS] Test 3: Marker strength decays over time")

    # Test 4: Sleep consolidation
    engine.markers[0].activation_count = 5
    engine.markers[0].strength = 0.5
    engine.sleep_consolidation()
    assert engine.markers[0].strength > 0.5, "Active markers should strengthen during sleep"
    print("[PASS] Test 4: Sleep consolidation strengthens active markers")

    # Test 5: Save/Load roundtrip
    saved = engine.save()
    engine2 = SomaticMarkerEngine(hdc)
    engine2.load(saved)
    assert len(engine2.markers) == len(engine.markers), "Load should restore markers"
    assert abs(engine2.markers[0].valence - engine.markers[0].valence) < 0.001
    print("[PASS] Test 5: Save/load roundtrip preserves data")

    # Test 6: Mutation
    old_threshold = engine.activation_threshold
    engine.mutate(rate=0.5)
    # Just verify it doesn't crash and values stay in range
    assert 0.05 <= engine.activation_threshold <= 0.3
    print("[PASS] Test 6: Mutation keeps parameters in valid range")

    print("\n=== All Somatic Marker Engine tests passed ===")
