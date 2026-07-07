"""
carl_agent.py — The Real Brain.

Architecture (Mark VIII — Phase A0 Hardened):
- K-Sparse Actor-Critic (winner-take-all lateral inhibition)
- CarlDreamer: World Model — predicts next obs, generates dream rollouts
- HyperdimensionalMemoryEngine: VSA/HDC holographic episodic memory
    * 40,000-dim bipolar vectors — no backprop, pure bitwise algebra
    * Decoupled Sub-Cortical Registers:
        - M_spatial (40K-D): environmental / navigational memory
        - M_affective (40K-D): emotional state signatures
        - M_procedural (40K-D): motor action memory traces
    * Binding (⊗): element-wise multiply — fuses state + emotion
    * Bundling (+): majority vote — superposition of multiple contexts
    * Permutation (Π): np.roll — asymmetric temporal sequence encoding
    * query_memory: cosine similarity = instant familiarity readout
    * Synaptic Pruning: sleep-cycle consolidation with λ-decay + salience amplification
- Neuromodulators: DA, NE, CORT, Serotonin, Hunger, Fatigue
    * Homeostatic Damping: exponential baseline decay prevents emotional lock-in
    * Anti-Panic Clutch: cortisol > 0.85 triggers autonomic protective freeze
- SpinalCord: hardwired brainstem reflexes (nest-seeking, wall avoidance)
"""

import numpy as np
import os
import math


# ══════════════════════════════════════════════════════════════════════════
from carl_circadian import CircadianOscillator
from carl_allostatic import AllostaticPredictor

class HyperdimensionalMemoryEngine:
    """
    Vector Symbolic Architecture (VSA) / Hyperdimensional Computing engine.
    Mark VIII: 40,000-Dimensional Decoupled Sub-Cortical Memory Architecture.

    Represents every sensory state + emotional context as a 40,000-dimensional
    bipolar hypervector {-1, +1}^D. Upgraded from the monolithic 10K-D design
    to three independent sub-cortical register lanes:

      M_spatial    (40K-D): Environmental / navigational memory traces
      M_affective  (40K-D): Emotional state signature patterns
      M_procedural (40K-D): Motor action memory traces

    Cross-binding only occurs during high-salience events (cortisol > 0.7 or
    dopamine > 0.8) to prevent mathematical crosstalk between modalities.

    Algebraic primitives:
      Binding (⊗):      element-wise multiply  — creates unique state-emotion pairing
      Bundling (+):     majority-vote sign      — superposition of multiple contexts
      Permutation (Π):  np.roll               — encodes temporal sequence order

    Synaptic Pruning Protocol (sleep consolidation):
      H *= λ (0.95 decay) — dims background noise
      H += γ * high_salience_events — re-amplifies survival-critical memories
      Keeps the 40K-D space crisp across weeks of continuous operation.

    The query_memory cosine similarity gives an instant familiarity score:
      sim ≈ +1 → very familiar territory (seen this emotional state here before)
      sim ≈  0 → novel territory (pure exploration zone)
      sim ≈ -1 → structurally anti-correlated (impossible in practice with bundling)

    Two signals flow out into Bob's cognition:
      1. High familiarity + high CORT → danger zone aversion (learned fear)
      2. Low familiarity              → novelty bonus (complements Dreamer curiosity)

    Biological analogue: Hippocampal pattern completion + entorhinal grid cells
                         + cortical consolidation during slow-wave sleep.
    Running cost: <2ms per step on CPU. No GPU needed. ~480KB RAM (3 × 40K × 4 bytes).
    """

    # ── MARK VIII CONSTANTS ──────────────────────────────────────────────
    PRUNING_DECAY = 0.95     # λ: background memory fade during sleep
    SALIENCE_GAIN = 2.0      # γ: re-amplification multiplier for critical events
    CORT_SALIENCE_THRESH = 0.7   # cortisol level that marks an event as "survival-critical"
    DA_SALIENCE_THRESH   = 0.8   # dopamine level that marks an event as "high-reward"
    CROSS_BIND_CORT      = 0.7   # cortisol threshold for cross-modal binding
    CROSS_BIND_DA        = 0.8   # dopamine threshold for cross-modal binding
    CLIP_MAGNITUDE       = 100.0 # prevents trace values from drifting to infinity

    def __init__(self, input_dim=24, hd_dim=40000):
        self.D = hd_dim
        self.input_dim = input_dim

        # Random Gaussian projection: continuous obs → hypervector
        # Each row is a random hyperplane boundary in 24D obs space
        self.proj_matrix = np.random.randn(self.D, self.input_dim).astype(np.float32)

        # Discrete basis vectors for emotional/neuromodulator contexts
        # Each is statistically orthogonal to all others (dot product ≈ 0 for D=40k)
        rng = np.random.default_rng(42)  # fixed seed for reproducible basis
        self.basis = {
            'DA_HIGH':   rng.choice([-1, 1], size=self.D).astype(np.float32),
            'CORT_HIGH': rng.choice([-1, 1], size=self.D).astype(np.float32),
            'NE_HIGH':   rng.choice([-1, 1], size=self.D).astype(np.float32),
            'HUNGER':    rng.choice([-1, 1], size=self.D).astype(np.float32),
            'FATIGUE':   rng.choice([-1, 1], size=self.D).astype(np.float32),
            'NEUTRAL':   rng.choice([-1, 1], size=self.D).astype(np.float32),
            # Mark VIII: Action basis vectors for procedural memory
            'ACTION_FWD':  rng.choice([-1, 1], size=self.D).astype(np.float32),
            'ACTION_REV':  rng.choice([-1, 1], size=self.D).astype(np.float32),
            'ACTION_LEFT': rng.choice([-1, 1], size=self.D).astype(np.float32),
            'ACTION_RIGHT':rng.choice([-1, 1], size=self.D).astype(np.float32),
        }

        # ── MARK VIII: DECOUPLED SUB-CORTICAL MEMORY REGISTERS ──────────
        # Spatial:    Where Bob has been (LiDAR + position states)
        # Affective:  What Bob felt (neuromodulator signatures)
        # Procedural: What Bob did (motor action patterns)
        self.M_spatial    = np.zeros(self.D, dtype=np.float32)
        self.M_affective  = np.zeros(self.D, dtype=np.float32)
        self.M_procedural = np.zeros(self.D, dtype=np.float32)
        self.M_episodic   = np.zeros(self.D, dtype=np.float32)  # global holographic bound trace

        # Legacy alias for backward compatibility in query_memory
        # This is a fused read-only view computed from all three registers
        self.episodic_trace = np.zeros(self.D, dtype=np.float32)

        # Running familiarity (EMA) — smoothed to reduce per-step noise
        self._familiarity_ema = 0.0

        # Sleep consolidation buffer: high-salience events staged for pruning
        self._salience_buffer = []  # list of (bound_vector, salience_score) tuples
        self._max_salience_buffer = 500  # cap to prevent unbounded growth

    def encode_sensory(self, obs_vector):
        """Project continuous 24D obs into a bipolar 40K-dim hypervector."""
        raw_proj = self.proj_matrix @ obs_vector.astype(np.float32)
        return np.sign(raw_proj + 1e-9)  # break ties with tiny epsilon

    def encode_action(self, action):
        """Encode a 2D action vector [throttle, steering] into a bipolar hypervector."""
        components = []
        if action[0] > 0.2:  components.append(self.basis['ACTION_FWD'])
        if action[0] < -0.2: components.append(self.basis['ACTION_REV'])
        if action[1] > 0.3:  components.append(self.basis['ACTION_RIGHT'])
        if action[1] < -0.3: components.append(self.basis['ACTION_LEFT'])
        if not components:
            return self.basis['NEUTRAL']
        return np.sign(np.sum(components, axis=0) + 1e-9)

    def select_emotional_context(self, drives):
        """Bundle active neuromodulatory states into one superposition vector."""
        active = []
        if drives.da > 0.5:      active.append(self.basis['DA_HIGH'])
        if drives.cort > 0.4:    active.append(self.basis['CORT_HIGH'])
        if drives.ne > 0.5:      active.append(self.basis['NE_HIGH'])
        if drives.hunger > 0.7:  active.append(self.basis['HUNGER'])
        if getattr(drives, 'fatigue', 0) > 0.6: active.append(self.basis['FATIGUE'])

        if not active:
            return self.basis['NEUTRAL']

        # Bundling: majority-vote superposition — retains similarity to all constituents
        return np.sign(np.sum(active, axis=0) + 1e-9)

    def write_episode(self, obs_vector, drives, action=None):
        """
        Encode current experience and fold into the decoupled sub-cortical registers.

        Mark VIII decoupled write protocol:
        1. Encode sensory state → S  (what Bob sees)
        2. Encode emotional context → E  (what Bob feels)
        3. Encode action context → A  (what Bob did)
        4. Write S into M_spatial with temporal permutation
        5. Write E into M_affective with temporal permutation
        6. Write A into M_procedural with temporal permutation
        7. IF high salience event → cross-bind S⊗E⊗A and stage for consolidation
        """
        S = self.encode_sensory(obs_vector)
        E = self.select_emotional_context(drives)

        # Write spatial register: pure sensory trace
        self.M_spatial = np.sign(np.roll(self.M_spatial, 1) + S)

        # Write affective register: emotional state trace
        self.M_affective = np.sign(np.roll(self.M_affective, 1) + E)

        # Write procedural register: motor action trace
        if action is not None:
            A = self.encode_action(action)
            self.M_procedural = np.sign(np.roll(self.M_procedural, 1) + A)

        # Cross-modal binding: only during high-salience events
        # This prevents crosstalk during routine operation
        is_salient = (drives.cort > self.CROSS_BIND_CORT or
                      drives.da > self.CROSS_BIND_DA)
        if is_salient:
            bound_experience = S * E  # Binding: fuse state with emotion
            if action is not None:
                bound_experience = bound_experience * self.encode_action(action)

            salience_score = max(drives.cort, drives.da)
            if len(self._salience_buffer) < self._max_salience_buffer:
                self._salience_buffer.append((bound_experience, salience_score))

        # Update the fused episodic trace (backward-compatible read-only view)
        # Weighted combination: spatial dominates, affective modulates
        self.episodic_trace = np.sign(
            0.5 * self.M_spatial + 0.3 * self.M_affective + 0.2 * self.M_procedural + 1e-9
        )

        # Update global holographic bound memory trace
        M_episode = self.M_spatial * self.M_affective * self.M_procedural
        self.M_episodic = np.sign(np.roll(self.M_episodic, 1) + M_episode + 1e-9)

    def query_memory(self, obs_vector):
        """
        Cosine similarity between current state and entire holographic history.
        Returns: familiarity in [-1, 1]. Positive = seen this before.
        """
        S_current = self.encode_sensory(obs_vector)
        dot = float(np.dot(self.episodic_trace, S_current))
        familiarity = dot / self.D  # normalized to [-1, 1]
        # EMA smoothing: avoid per-frame noise spikes
        self._familiarity_ema = 0.92 * self._familiarity_ema + 0.08 * familiarity
        return self._familiarity_ema

    def query_spatial(self, obs_vector):
        """Query spatial memory only — have I been to this physical location?"""
        S_current = self.encode_sensory(obs_vector)
        dot = float(np.dot(self.M_spatial, S_current))
        return dot / self.D

    def query_affective(self, drives):
        """Query affective memory — have I felt this emotional state before?"""
        E_current = self.select_emotional_context(drives)
        dot = float(np.dot(self.M_affective, E_current))
        return dot / self.D

    def query_episodic(self, obs_vector):
        """
        Query the global holographic episodic memory trace using the current spatial vector.
        Decodes the associated emotional state and returns its similarity to CORT_HIGH.
        """
        S_current = self.encode_sensory(obs_vector)
        # Decode: in bipolar VSA, element-wise multiplication is the self-inverse binding operator
        decoded = S_current * self.M_episodic
        # Return cosine similarity to CORT_HIGH basis vector
        return float(np.dot(decoded, self.basis['CORT_HIGH'])) / self.D

    def sleep_consolidation(self):
        """
        Mark VIII Synaptic Pruning Protocol.
        Called during Bob's sleep cycle to maintain memory fidelity.

        1. Decay all registers by λ = 0.95 (dims background noise)
        2. Re-amplify high-salience events with γ = 2.0 gain
        3. Flush the salience buffer
        4. Clip to prevent magnitude drift

        This keeps the 40K-D space crisp across weeks of operation,
        focusing memory capacity on survival-critical experiences.
        """
        # Step 1: Global decay — fade the background noise
        self.M_spatial    *= self.PRUNING_DECAY
        self.M_affective  *= self.PRUNING_DECAY
        self.M_procedural *= self.PRUNING_DECAY
        self.M_episodic   *= self.PRUNING_DECAY

        # Step 2: Re-amplify high-salience events
        for bound_vec, salience in self._salience_buffer:
            gain = self.SALIENCE_GAIN * salience
            # Distribute across all registers (cross-bound events touch everything)
            self.M_spatial    += gain * bound_vec * 0.4
            self.M_affective  += gain * bound_vec * 0.4
            self.M_procedural += gain * bound_vec * 0.2
            self.M_episodic   += gain * bound_vec * 1.0

        # Step 3: Flush consolidation buffer
        self._salience_buffer.clear()

        # Step 4: Clip to prevent infinite magnitude drift
        self.M_spatial    = np.clip(self.M_spatial, -self.CLIP_MAGNITUDE, self.CLIP_MAGNITUDE)
        self.M_affective  = np.clip(self.M_affective, -self.CLIP_MAGNITUDE, self.CLIP_MAGNITUDE)
        self.M_procedural = np.clip(self.M_procedural, -self.CLIP_MAGNITUDE, self.CLIP_MAGNITUDE)
        self.M_episodic   = np.clip(self.M_episodic, -self.CLIP_MAGNITUDE, self.CLIP_MAGNITUDE)

        # Rebuild fused trace
        self.episodic_trace = np.sign(
            0.5 * self.M_spatial + 0.3 * self.M_affective + 0.2 * self.M_procedural + 1e-9
        )

    def save(self):
        return {
            'M_spatial': self.M_spatial.copy(),
            'M_affective': self.M_affective.copy(),
            'M_procedural': self.M_procedural.copy(),
            'M_episodic': self.M_episodic.copy(),
            'episodic_trace': self.episodic_trace.copy(),
            '_familiarity_ema': self._familiarity_ema,
            'hd_dim': self.D,
        }

    def load(self, d):
        saved_dim = d.get('hd_dim', 10000)
        if saved_dim != self.D:
            # Incompatible checkpoint — cannot load old 10K-D into 40K-D
            print(f"[HDC] Dimension mismatch: saved={saved_dim}, current={self.D}. Starting fresh.")
            return
        self.M_spatial    = d.get('M_spatial', np.zeros(self.D, dtype=np.float32))
        self.M_affective  = d.get('M_affective', np.zeros(self.D, dtype=np.float32))
        self.M_procedural = d.get('M_procedural', np.zeros(self.D, dtype=np.float32))
        self.M_episodic   = d.get('M_episodic', np.zeros(self.D, dtype=np.float32))
        self.episodic_trace = d.get('episodic_trace', np.zeros(self.D, dtype=np.float32))
        self._familiarity_ema = d.get('_familiarity_ema', 0.0)


def relu(x):
    return np.maximum(0, x)


def softplus(x):
    return np.log1p(np.exp(np.clip(x, -20, 20)))


def tanh(x):
    return np.tanh(x)


class NeuralLayer:
    """A single dense layer with Adam optimizer."""
    def __init__(self, n_in, n_out, activation='tanh', lr=3e-4):
        # Kaiming / He init
        scale = np.sqrt(2.0 / n_in)
        self.W = np.random.randn(n_out, n_in) * scale
        self.b = np.zeros(n_out)
        self.activation = activation
        self.lr = lr
        # Adam state
        self.mW = np.zeros_like(self.W)
        self.vW = np.zeros_like(self.W)
        self.mb = np.zeros_like(self.b)
        self.vb = np.zeros_like(self.b)
        self.t = 0
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-8

    def forward(self, x):
        self.x = x
        self.z = self.W @ x + self.b
        if self.activation == 'tanh':
            self.a = np.tanh(self.z)
        elif self.activation == 'relu':
            self.a = relu(self.z)
        elif self.activation == 'linear':
            self.a = self.z
        elif self.activation == 'softplus':
            self.a = softplus(self.z)
        return self.a

    def backward(self, grad_out):
        """Backprop through this layer. Returns grad for input."""
        if self.activation == 'tanh':
            dact = 1.0 - self.a ** 2
        elif self.activation == 'relu':
            dact = (self.a > 0).astype(float)
        elif self.activation == 'linear':
            dact = np.ones_like(self.a)
        elif self.activation == 'softplus':
            dact = 1.0 / (1.0 + np.exp(-self.z))

        delta = grad_out * dact
        dW = np.outer(delta, self.x)
        db = delta
        dx = self.W.T @ delta
        return dW, db, dx

    def update(self, dW, db):
        self.t += 1
        # Adam
        self.mW = self.beta1 * self.mW + (1 - self.beta1) * dW
        self.vW = self.beta2 * self.vW + (1 - self.beta2) * dW**2
        mW_hat = self.mW / (1 - self.beta1**self.t)
        vW_hat = self.vW / (1 - self.beta2**self.t)
        self.W -= self.lr * mW_hat / (np.sqrt(vW_hat) + self.eps)

        self.mb = self.beta1 * self.mb + (1 - self.beta1) * db
        self.vb = self.beta2 * self.vb + (1 - self.beta2) * db**2
        mb_hat = self.mb / (1 - self.beta1**self.t)
        vb_hat = self.vb / (1 - self.beta2**self.t)
        self.b -= self.lr * mb_hat / (np.sqrt(vb_hat) + self.eps)

    def save(self):
        return {'W': self.W.copy(), 'b': self.b.copy(),
                'mW': self.mW.copy(), 'vW': self.vW.copy(),
                'mb': self.mb.copy(), 'vb': self.vb.copy(), 't': self.t}

    def load(self, d):
        W_loaded = d['W'].copy()
        if W_loaded.shape != self.W.shape:
            print(f"[BRAIN] Adapting weight matrix shape from {W_loaded.shape} to {self.W.shape}")
            new_W = np.zeros_like(self.W)
            min_out = min(W_loaded.shape[0], self.W.shape[0])
            min_in = min(W_loaded.shape[1], self.W.shape[1])
            new_W[:min_out, :min_in] = W_loaded[:min_out, :min_in]
            self.W = new_W

            new_mW = np.zeros_like(self.W)
            mW_loaded = d.get('mW', np.zeros_like(W_loaded))
            new_mW[:min_out, :min_in] = mW_loaded[:min_out, :min_in]
            self.mW = new_mW

            new_vW = np.zeros_like(self.W)
            vW_loaded = d.get('vW', np.zeros_like(W_loaded))
            new_vW[:min_out, :min_in] = vW_loaded[:min_out, :min_in]
            self.vW = new_vW
        else:
            self.W = W_loaded
            self.mW = d.get('mW', np.zeros_like(self.W))
            self.vW = d.get('vW', np.zeros_like(self.W))

        b_loaded = d['b'].copy()
        if b_loaded.shape != self.b.shape:
            print(f"[BRAIN] Adapting bias shape from {b_loaded.shape} to {self.b.shape}")
            new_b = np.zeros_like(self.b)
            min_out = min(b_loaded.shape[0], self.b.shape[0])
            new_b[:min_out] = b_loaded[:min_out]
            self.b = new_b

            new_mb = np.zeros_like(self.b)
            mb_loaded = d.get('mb', np.zeros_like(b_loaded))
            new_mb[:min_out] = mb_loaded[:min_out]
            self.mb = new_mb

            new_vb = np.zeros_like(self.b)
            vb_loaded = d.get('vb', np.zeros_like(b_loaded))
            new_vb[:min_out] = vb_loaded[:min_out]
            self.vb = new_vb
        else:
            self.b = b_loaded
            self.mb = d.get('mb', np.zeros_like(self.b))
            self.vb = d.get('vb', np.zeros_like(self.b))

        self.t = d.get('t', 0)


class KSparseLayer:
    """
    K-Sparse Neural Layer: Only the top-k most active neurons fire.
    All others are suppressed to zero. This creates:
    - 80% compute reduction (most multiplications produce zero)
    - Sharper, more selective feature detectors
    - Deeper networks without melting consumer hardware
    
    Biological analogue: Winner-take-all lateral inhibition in cortical columns.
    """
    def __init__(self, n_in, n_out, k=16, lr=3e-4):
        scale = np.sqrt(2.0 / n_in)
        self.W = np.random.randn(n_out, n_in) * scale
        self.b = np.zeros(n_out)
        self.k = min(k, n_out)  # can't keep more than we have
        self.lr = lr
        # Adam state
        self.mW = np.zeros_like(self.W)
        self.vW = np.zeros_like(self.W)
        self.mb = np.zeros_like(self.b)
        self.vb = np.zeros_like(self.b)
        self.t = 0
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-8

    def forward(self, x):
        self.x = x
        self.z = self.W @ x + self.b
        # K-Sparse: find top-k, zero everything else
        top_k_indices = np.argsort(np.abs(self.z))[-self.k:]
        self.mask = np.zeros_like(self.z)
        self.mask[top_k_indices] = 1.0
        self.a = np.tanh(self.z) * self.mask  # tanh + sparse gate
        return self.a

    def backward(self, grad_out):
        # Gradient only flows through the active (unmasked) neurons
        dact = (1.0 - self.a ** 2) * self.mask
        delta = grad_out * dact
        dW = np.outer(delta, self.x)
        db = delta
        dx = self.W.T @ delta
        return dW, db, dx

    def update(self, dW, db):
        self.t += 1
        self.mW = self.beta1 * self.mW + (1 - self.beta1) * dW
        self.vW = self.beta2 * self.vW + (1 - self.beta2) * dW**2
        mW_hat = self.mW / (1 - self.beta1**self.t)
        vW_hat = self.vW / (1 - self.beta2**self.t)
        self.W -= self.lr * mW_hat / (np.sqrt(vW_hat) + self.eps)
        self.mb = self.beta1 * self.mb + (1 - self.beta1) * db
        self.vb = self.beta2 * self.vb + (1 - self.beta2) * db**2
        mb_hat = self.mb / (1 - self.beta1**self.t)
        vb_hat = self.vb / (1 - self.beta2**self.t)
        self.b -= self.lr * mb_hat / (np.sqrt(vb_hat) + self.eps)

    def save(self):
        return {'W': self.W.copy(), 'b': self.b.copy(),
                'mW': self.mW.copy(), 'vW': self.vW.copy(),
                'mb': self.mb.copy(), 'vb': self.vb.copy(), 't': self.t, 'k': self.k}

    def load(self, d):
        self.W = d['W'].copy()
        self.b = d['b'].copy()
        self.mW = d.get('mW', np.zeros_like(self.W))
        self.vW = d.get('vW', np.zeros_like(self.W))
        self.mb = d.get('mb', np.zeros_like(self.b))
        self.vb = d.get('vb', np.zeros_like(self.b))
        self.t  = d.get('t', 0)
        self.k  = d.get('k', self.k)


class PlasticKSparseLayer:
    """
    K-Sparse Neural Layer with neuromodulated Hebbian plasticity (Backpropamine-style).
    Effective weight is W_eff = W + alpha * Hebb.
    W (base weight) and alpha (plasticity scale) are optimized via RL / policy gradients,
    while Hebb is a local dynamic trace updated step-by-step.
    """
    def __init__(self, n_in, n_out, k=16, lr=3e-4, trace_decay=0.95):
        scale = np.sqrt(2.0 / n_in)
        self.W = np.random.randn(n_out, n_in) * scale
        # Learnable plasticity scale
        self.alpha = np.random.randn(n_out, n_in) * 0.01
        self.b = np.zeros(n_out)
        self.k = min(k, n_out)
        self.lr = lr
        self.trace_decay = trace_decay
        
        # Dynamic Hebbian trace (reset per episode)
        self.Hebb = np.zeros((n_out, n_in), dtype=np.float32)
        
        # Adam state
        self.mW = np.zeros_like(self.W); self.vW = np.zeros_like(self.W)
        self.mAlpha = np.zeros_like(self.alpha); self.vAlpha = np.zeros_like(self.alpha)
        self.mb = np.zeros_like(self.b); self.vb = np.zeros_like(self.b)
        self.t = 0
        self.beta1 = 0.9; self.beta2 = 0.999; self.eps = 1e-8

    def reset_trace(self):
        self.Hebb.fill(0.0)

    def forward(self, x):
        self.x = x
        # Compute effective weights
        self.W_eff = self.W + self.alpha * self.Hebb
        self.z = self.W_eff @ x + self.b
        
        # K-Sparse: find top-k, zero everything else
        top_k_indices = np.argsort(np.abs(self.z))[-self.k:]
        self.mask = np.zeros_like(self.z)
        self.mask[top_k_indices] = 1.0
        self.a = np.tanh(self.z) * self.mask  # tanh + sparse gate
        return self.a

    def update_trace(self, M_t):
        """
        Update the Hebbian trace.
        Hebb_t = decay * Hebb_{t-1} + M_t * (a_t x_t^T)
        """
        outer = np.outer(self.a, self.x)
        self.Hebb = self.trace_decay * self.Hebb + M_t * outer
        self.Hebb = np.clip(self.Hebb, -2.0, 2.0)

    def backward(self, grad_out):
        # Gradient only flows through the active (unmasked) neurons
        dact = (1.0 - self.a ** 2) * self.mask
        delta = grad_out * dact
        
        dW = np.outer(delta, self.x)
        dAlpha = dW * self.Hebb
        db = delta
        
        dx = self.W_eff.T @ delta
        return dW, dAlpha, db, dx

    def update(self, dW, dAlpha, db):
        self.t += 1
        # Adam for W
        self.mW = self.beta1 * self.mW + (1 - self.beta1) * dW
        self.vW = self.beta2 * self.vW + (1 - self.beta2) * dW**2
        mW_hat = self.mW / (1 - self.beta1**self.t)
        vW_hat = self.vW / (1 - self.beta2**self.t)
        self.W -= self.lr * mW_hat / (np.sqrt(vW_hat) + self.eps)
        
        # Adam for alpha
        self.mAlpha = self.beta1 * self.mAlpha + (1 - self.beta1) * dAlpha
        self.vAlpha = self.beta2 * self.vAlpha + (1 - self.beta2) * dAlpha**2
        mAlpha_hat = self.mAlpha / (1 - self.beta1**self.t)
        vAlpha_hat = self.vAlpha / (1 - self.beta2**self.t)
        self.alpha -= self.lr * mAlpha_hat / (np.sqrt(vAlpha_hat) + self.eps)

        # Adam for b
        self.mb = self.beta1 * self.mb + (1 - self.beta1) * db
        self.vb = self.beta2 * self.vb + (1 - self.beta2) * db**2
        mb_hat = self.mb / (1 - self.beta1**self.t)
        vb_hat = self.vb / (1 - self.beta2**self.t)
        self.b -= self.lr * mb_hat / (np.sqrt(vb_hat) + self.eps)

    def save(self):
        return {
            'W': self.W.copy(), 'alpha': self.alpha.copy(), 'b': self.b.copy(),
            'mW': self.mW.copy(), 'vW': self.vW.copy(),
            'mAlpha': self.mAlpha.copy(), 'vAlpha': self.vAlpha.copy(),
            'mb': self.mb.copy(), 'vb': self.vb.copy(), 't': self.t, 'k': self.k
        }

    def load(self, d):
        self.W = d['W'].copy()
        self.alpha = d.get('alpha', np.random.randn(*self.W.shape) * 0.01).copy()
        self.b = d['b'].copy()
        self.mW = d.get('mW', np.zeros_like(self.W))
        self.vW = d.get('vW', np.zeros_like(self.W))
        self.mAlpha = d.get('mAlpha', np.zeros_like(self.alpha))
        self.vAlpha = d.get('vAlpha', np.zeros_like(self.alpha))
        self.mb = d.get('mb', np.zeros_like(self.b))
        self.vb = d.get('vb', np.zeros_like(self.b))
        self.t  = d.get('t', 0)
        self.k  = d.get('k', self.k)


class CarlActor:
    """
    Policy network: obs -> (throttle, steering)
    
    Architecture inspired by biological motor cortex:
    - Sensory cortex layer (lidar + proprioception)
    - Association layer (cross-modal binding) - PLASTIC Hebbian
    - Premotor layer (action preparation) - PLASTIC Hebbian
    - Motor output (M1)
    - Neuromodulator predictor (learned internal drives)
    """
    def __init__(self, n_obs, lr=3e-4):
        hidden = 128
        self.l1 = NeuralLayer(n_obs, hidden, 'relu', lr)      # Sensory cortex (dense — needs full input)
        self.l2 = PlasticKSparseLayer(hidden, hidden, k=24, lr=lr) # Association cortex (sparse — selective binding)
        self.l3 = PlasticKSparseLayer(hidden, 64, k=16, lr=lr)     # Premotor cortex (sparse — action preparation)
        
        # Mean outputs: throttle in [-1,1], steering in [-1,1]
        self.mu_head   = NeuralLayer(64, 2, 'tanh', lr)        # Motor output (dense — needs precise control)
        # Learned neuromodulator predictions [da, ne, cort, sero]
        self.nm_head   = NeuralLayer(64, 4, 'tanh', lr)
        
        # Log std (learnable): separate from mean, no backprop through state
        self.log_std = np.array([0.0, 0.0])  # initial std = 1.0, will anneal with steps

        self.prev_obs = None
        self.prev_mu  = None
        self.prev_action = None
        self.cache = {}
        self.total_food_lifetime = 0

    def reset_traces(self):
        """Reset Hebbian memory traces (called on episode reset / wakeup)"""
        self.l2.reset_trace()
        self.l3.reset_trace()

    def forward(self, obs, update_traces=True):
        h1 = self.l1.forward(obs)
        h2 = self.l2.forward(h1)
        h3 = self.l3.forward(h2)
        mu = self.mu_head.forward(h3)
        nm_out = self.nm_head.forward(h3)
        self.cache = {'h1': h1, 'h2': h2, 'h3': h3, 'mu': mu, 'nm_out': nm_out}
        
        # Real-time Hebbian synaptic update (gated by learned dopamine component)
        if update_traces:
            M_t = float(nm_out[0])
            self.l2.update_trace(M_t)
            self.l3.update_trace(M_t)
            
        return mu

    def sample_action(self, obs, ne_signal=0.0, cort_signal=0.0):
        """Sample action using Thermodynamic Action Sampling (arousal-scaled noise)."""
        mu = self.forward(obs, update_traces=True)
        # Thermodynamic noise scaling: norepinephrine and cortisol increase temperature
        arousal_temp = np.clip(ne_signal + 0.5 * cort_signal, 0.1, 2.0)
        std = np.exp(np.clip(self.log_std, -3, 0.5)) * arousal_temp
        
        noise = np.random.randn(2) * std
        action = np.clip(mu + noise, -1.0, 1.0)
        # Log probability of sampled action
        log_prob = -0.5 * np.sum(((action - mu) / (std + 1e-8))**2) \
                   - np.sum(np.log(std + 1e-8))
        self.prev_obs = obs.copy()
        self.prev_mu  = mu.copy()
        self.prev_action = action.copy()
        return action, log_prob

    def update_policy(self, advantage, prev_obs, prev_action, prev_mu, ne_signal, cort_signal=0.0, sero_signal=1.0, hdc_familiarity=0.5):
        """
        Policy gradient update (REINFORCE with baseline) + ART-Gated plasticity.
        """
        # Recompute forward pass for gradient (without double-updating traces)
        mu = self.forward(prev_obs, update_traces=False)
        std = np.exp(np.clip(self.log_std, -3, 0.5))
        diff = (prev_action - mu) / (std**2 + 1e-8)

        # Policy gradient loss gradient: -advantage * d(log_pi)/d(mu)
        pg_grad = -advantage * diff    # shape (2,)

        # Backprop through mu_head
        dW_mu, db_mu, dh3 = self.mu_head.backward(pg_grad)
        
        # Train nm_head predictively
        nm_out = self.cache['nm_out']
        # Targets: [da_pred -> advantage, ne_pred -> abs(advantage), cort_pred -> cort_signal, sero_pred -> sero_signal]
        nm_target = np.array([advantage, abs(advantage), cort_signal, sero_signal], dtype=np.float32)
        nm_grad = 2.0 * (nm_out - nm_target) # MSE gradient
        dW_nm, db_nm, dh3_nm = self.nm_head.backward(nm_grad)
        
        # Combine hidden layer gradients at M1/drives intersection
        dh3_combined = dh3 + dh3_nm
        
        # Backprop through hidden plastic layers
        dW3, dAlpha3, db3, dh2 = self.l3.backward(dh3_combined)
        dW2, dAlpha2, db2, dh1 = self.l2.backward(dh2)
        dW1, db1, _ = self.l1.backward(dh1)

        # NE signal boosts effective learning rate for surprising events
        ne_boost = 1.0 + 2.0 * ne_signal
        if cort_signal > 0.6:
            ne_boost *= 1.6  # Impulsive trauma adaptation response
        if sero_signal > 0.7:
            ne_boost *= 0.6  # Memory protection guard when safe

        # Grossberg's Adaptive Resonance Theory (ART) Vigilance Gate:
        # If familiarity is high (>0.05), we are in resonance (update base W weights fully).
        # If familiarity is low (<=0.05), we are in mismatch (protect W weights, let plasticity scales/Hebb adapt).
        vigilance_gate = 1.0 if hdc_familiarity > 0.05 else 0.25

        # Update mu_head & nm_head
        self.mu_head.update(dW_mu * ne_boost, db_mu * ne_boost)
        self.nm_head.update(dW_nm * ne_boost, db_nm * ne_boost)
        
        # Update plastic layers (base W update scaled by vigilance gate)
        self.l3.update(dW3 * ne_boost * vigilance_gate, dAlpha3 * ne_boost, db3 * ne_boost)
        self.l2.update(dW2 * ne_boost * vigilance_gate, dAlpha2 * ne_boost, db2 * ne_boost)
        
        # Update input layer
        self.l1.update(dW1 * ne_boost * vigilance_gate, db1 * ne_boost)

        # Variance Annealing — based on TRAINING STEPS, not food count
        step_count = getattr(self, '_training_steps', 0)
        self._training_steps = step_count + 1
        max_allowed_log_std = max(-2.0, 0.0 - (step_count / 150000.0))
        # Soft pull: gently nudge log_std back up toward ceiling if it drifts below
        for i in range(len(self.log_std)):
            if self.log_std[i] < max_allowed_log_std - 0.3:
                self.log_std[i] += 0.002   # recover exploration if too quiet
        self.log_std = np.clip(self.log_std, -2.0, max_allowed_log_std)

    def save(self):
        return {
            'l1': self.l1.save(), 'l2': self.l2.save(),
            'l3': self.l3.save(), 'mu_head': self.mu_head.save(),
            'nm_head': self.nm_head.save(),
            'log_std': self.log_std.copy(),
            'total_food_lifetime': self.total_food_lifetime,
            '_training_steps': getattr(self, '_training_steps', 0)
        }

    def load(self, d):
        self.l1.load(d['l1']); self.l2.load(d['l2'])
        self.l3.load(d['l3']); self.mu_head.load(d['mu_head'])
        if 'nm_head' in d:
            self.nm_head.load(d['nm_head'])
        self.log_std = d.get('log_std', np.array([-0.5, -0.5]))
        self.total_food_lifetime = d.get('total_food_lifetime', 0)
        self._training_steps = d.get('_training_steps', 0)


class CarlCritic:
    """
    Value network: obs -> V(s)
    Biological analogue: Striatum / Basal Ganglia value estimation.
    """
    def __init__(self, n_obs, lr=1e-3):
        hidden = 128
        self.l1 = NeuralLayer(n_obs, hidden, 'relu', lr)
        self.l2 = NeuralLayer(hidden, 64, 'relu', lr)
        self.v_head = NeuralLayer(64, 1, 'linear', lr)

    def forward(self, obs):
        h1 = self.l1.forward(obs)
        h2 = self.l2.forward(h1)
        v  = self.v_head.forward(h2)
        return float(v[0])

    def update_value(self, obs, target_v):
        """MSE loss gradient update."""
        v = self.forward(obs)
        error = v - target_v
        dv = np.array([2.0 * error])
        dW_v, db_v, dh2 = self.v_head.backward(dv)
        dW2, db2, dh1 = self.l2.backward(dh2)
        dW1, db1, _ = self.l1.backward(dh1)
        self.v_head.update(dW_v, db_v)
        self.l2.update(dW2, db2)
        self.l1.update(dW1, db1)
        return error

    def save(self):
        return {'l1': self.l1.save(), 'l2': self.l2.save(), 'v_head': self.v_head.save()}

    def load(self, d):
        self.l1.load(d['l1']); self.l2.load(d['l2']); self.v_head.load(d['v_head'])


import random
import os
import pickle

class EpisodicBuffer:
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def add(self, state, action, reward, next_state, done, surprise):
        experience = (state, action, reward, next_state, done, surprise)
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.buffer, min(len(self.buffer), batch_size))

    def __len__(self):
        return len(self.buffer)

class DreamArchive:
    def __init__(self, capacity=100, save_path="dream_archive.pkl"):
        self.capacity = capacity
        self.archive = []
        self.save_path = save_path
        self.load()

    def load(self):
        if os.path.exists(self.save_path):
            try:
                with open(self.save_path, "rb") as f:
                    self.archive = pickle.load(f)
            except Exception:
                pass

    def save(self):
        try:
            with open(self.save_path, "wb") as f:
                pickle.dump(self.archive, f)
        except Exception:
            pass

    def score_experience(self, surprise, reward, future_impact=0.0):
        # The user's requested formula:
        return 0.4 * surprise + 0.3 * reward + 0.3 * future_impact

    def consider_adding(self, state, action, reward, next_state, done, surprise, future_impact=0.0):
        score = self.score_experience(surprise, reward, future_impact)
        experience = (score, (state, action, reward, next_state, done, surprise))
        
        if len(self.archive) < self.capacity:
            self.archive.append(experience)
            self.archive.sort(key=lambda x: x[0], reverse=True)
        elif score > self.archive[-1][0]:
            self.archive[-1] = experience
            self.archive.sort(key=lambda x: x[0], reverse=True)

    def sample(self, batch_size):
        if not self.archive:
            return []
        items = [x[1] for x in self.archive]
        return random.sample(items, min(len(items), batch_size))
class CarlDreamer:
    """
    The World Model / Forward Prediction Network.
    
    Input:  current observation (24) + action (2) = 26 dims
    Output: predicted NEXT observation (24 dims)
    
    This network learns the physics of Carl's universe. By predicting
    what his sensors will see one timestep into the future, the weights
    must encode gravity, wall collisions, momentum, and food proximity.
    
    The prediction error (surprise) serves as an intrinsic curiosity signal:
    high surprise = the world did something unexpected = learn faster.
    
    During sleep, this network generates imagined rollouts — the Actor
    trains on hallucinated trajectories without touching the physics engine.
    
    Biological analogue: Hippocampal-Entorhinal forward model / 
                         Predictive Processing (Karl Friston's Active Inference)
    """
    def __init__(self, n_obs=24, n_act=2, lr=1e-3):
        self.n_obs = n_obs
        self.n_act = n_act
        n_in = n_obs + n_act  # 26
        hidden = 64
        # Lean but deep — K-Sparse hidden layers for efficiency
        self.l1 = NeuralLayer(n_in, hidden, 'relu', lr)
        self.l2 = KSparseLayer(hidden, hidden, k=16, lr=lr)
        self.out = NeuralLayer(hidden, n_obs, 'linear', lr)   # predict raw obs delta
        
        self._surprise_ema = 0.0  # exponential moving average of prediction error

    def predict(self, obs, action):
        """Predict next observation given current obs and action."""
        x = np.concatenate([obs, action])
        h1 = self.l1.forward(x)
        h2 = self.l2.forward(h1)
        delta_pred = self.out.forward(h2)
        # Predict the CHANGE in observation, not the absolute next state
        # This makes the network's job easier — learn residuals
        return obs + delta_pred

    def train_step(self, obs, action, next_obs):
        """
        One-step MSE training on (obs, action) -> next_obs.
        Returns the surprise signal (prediction error magnitude).
        """
        predicted = self.predict(obs, action)
        error = predicted - next_obs
        mse = float(np.mean(error ** 2))
        
        # Backprop: gradient of MSE w.r.t. delta_pred = 2 * error / n
        grad = 2.0 * error / self.n_obs
        # Clamp gradient to prevent instability
        grad = np.clip(grad, -1.0, 1.0)
        
        dW_out, db_out, dh2 = self.out.backward(grad)
        dW2, db2, dh1 = self.l2.backward(dh2)
        dW1, db1, _ = self.l1.backward(dh1)
        
        self.out.update(dW_out, db_out)
        self.l2.update(dW2, db2)
        self.l1.update(dW1, db1)
        
        # Update surprise EMA
        self._surprise_ema = 0.95 * self._surprise_ema + 0.05 * mse
        
        return mse

    def surprise(self):
        """Current surprise level (prediction error EMA)."""
        return self._surprise_ema

    def imagine_rollout(self, start_obs, actor, steps=5):
        """
        Generate an imagined trajectory using the world model.
        Returns list of (obs, action, predicted_next_obs) tuples.
        Used during sleep for generative replay.
        """
        trajectory = []
        obs = start_obs.copy()
        for _ in range(steps):
            mu = actor.forward(obs)
            # Add small noise for dream diversity
            action = np.clip(mu + np.random.randn(2) * 0.1, -1.0, 1.0)
            
            # Predict expects only the sensory part of obs
            next_sensory = self.predict(obs[:self.n_obs], action)
            
            # Reconstruct next_obs matching start_obs dimensions
            if len(obs) > self.n_obs:
                next_obs = np.zeros(len(obs), dtype=np.float32)
                next_obs[:self.n_obs] = next_sensory
                next_obs[self.n_obs:] = obs[self.n_obs:]
            else:
                next_obs = next_sensory
                
            trajectory.append((obs.copy(), action.copy(), next_obs.copy()))
            obs = next_obs
        return trajectory

    def save(self):
        return {
            'l1': self.l1.save(), 'l2': self.l2.save(), 'out': self.out.save(),
            '_surprise_ema': self._surprise_ema
        }

    def load(self, d):
        self.l1.load(d['l1']); self.l2.load(d['l2']); self.out.load(d['out'])
        self._surprise_ema = d.get('_surprise_ema', 0.0)


class BiologicalDrives:
    """
    Tracks Bob's neuromodulatory state — the biochemical context
    that shapes learning and behavior.

    Mark VIII Homeostatic Damping Architecture:
    - Homeostatic Core: Energy (H_e), Fatigue (H_f), Stress/Cortisol (H_s), Damage (H_d)
    - Metabolic consumption depends on velocity, CPG activity & control effort.
    - Permanent damage/torque wear triggers when H_d crosses thresholds.
    """
    def __init__(self):
        # Innate biochemical baselines
        self.da_innate = 0.0     # dopamine [-1, 1]
        self.ne_innate = 0.0     # norepinephrine [0, 1]
        self.cort_innate = 0.0   # cortisol [0, 1] (Stress / H_s)
        self.sero_innate = 1.0   # serotonin [0, 1]
        
        # Learned predictive modulatory components (from nm_head)
        self.da_learned = 0.0
        self.ne_learned = 0.0
        self.cort_learned = 0.0
        self.sero_learned = 0.0
        
        self.ach = 0.0    # acetylcholine [0, 1]
        
        # Homeostatic Dials (H_e, H_f, H_s, H_d) in [0.0, 1.0]
        self.energy = 1.0   # H_e (Energy)
        self.fatigue = 0.0  # H_f (Fatigue)
        self.damage = 0.0   # H_d (Damage)
        self.hunger = 0.0   # 1.0 - self.energy (backward compatibility)
        self.last_drain = 0.0
        
        # Interoceptive Predictions (The Self-Model's priors)
        self.pred_energy = 1.0
        self.pred_safety = 1.0
        self.pred_novelty = 0.0

        self.exploration_gain = 1.0
        
        # Constants & Evolved Parameters (Evolvable via genetic mutations)
        self.CLUTCH_THRESHOLD = 0.85
        self.CLUTCH_DA_SUPPRESSION = 0.1
        
        # Boredom rates (Evolvable)
        self.boredom_accumulation_rate = 0.04
        self.boredom_decay_rate = 0.01
        
        # Sleep replay fractions & thresholds (Evolvable)
        self.sleep_replay_fraction = 0.20
        self.sleep_pruning_decay = 0.95
        
        # Curiosity and CPG parameters (Evolvable)
        self.curiosity_surprise_gain = 0.05
        self.curiosity_hdc_gain = 0.03
        self.cpg_frequency_multiplier = 1.0
        self.cpg_amplitude_multiplier = 1.0
        self.brainstem_reflex_stiffness = 1.0

    @property
    def da(self):
        return float(np.clip(self.da_innate + self.da_learned, -1.0, 1.0))
    @da.setter
    def da(self, val):
        self.da_innate = float(np.clip(val, -1.0, 1.0))

    @property
    def ne(self):
        return float(np.clip(self.ne_innate + self.ne_learned, 0.0, 1.0))
    @ne.setter
    def ne(self, val):
        self.ne_innate = float(np.clip(val, 0.0, 1.0))

    @property
    def cort(self):
        return float(np.clip(self.cort_innate + self.cort_learned, 0.0, 1.0))
    @cort.setter
    def cort(self, val):
        self.cort_innate = float(np.clip(val, 0.0, 1.0))

    @property
    def sero(self):
        return float(np.clip(self.sero_innate + self.sero_learned, 0.0, 1.0))
    @sero.setter
    def sero(self, val):
        self.sero_innate = float(np.clip(val, 0.0, 1.0))

    def mutate(self, rate=0.15):
        """Mutates genetic parameters with Gaussian perturbations within valid biological ranges."""
        self.CLUTCH_THRESHOLD = float(np.clip(self.CLUTCH_THRESHOLD + np.random.normal(0, rate * 0.1), 0.5, 0.98))
        self.CLUTCH_DA_SUPPRESSION = float(np.clip(self.CLUTCH_DA_SUPPRESSION + np.random.normal(0, rate * 0.05), 0.0, 0.5))
        
        self.boredom_accumulation_rate = float(np.clip(self.boredom_accumulation_rate + np.random.normal(0, rate * 0.01), 0.005, 0.2))
        self.boredom_decay_rate = float(np.clip(self.boredom_decay_rate + np.random.normal(0, rate * 0.005), 0.001, 0.1))
        
        self.sleep_replay_fraction = float(np.clip(self.sleep_replay_fraction + np.random.normal(0, rate * 0.05), 0.05, 0.5))
        self.sleep_pruning_decay = float(np.clip(self.sleep_pruning_decay + np.random.normal(0, rate * 0.02), 0.85, 0.999))
        
        self.curiosity_surprise_gain = float(np.clip(self.curiosity_surprise_gain + np.random.normal(0, rate * 0.02), 0.01, 0.5))
        self.curiosity_hdc_gain = float(np.clip(self.curiosity_hdc_gain + np.random.normal(0, rate * 0.01), 0.01, 0.3))
        
        self.cpg_frequency_multiplier = float(np.clip(self.cpg_frequency_multiplier + np.random.normal(0, rate * 0.2), 0.5, 2.5))
        self.cpg_amplitude_multiplier = float(np.clip(self.cpg_amplitude_multiplier + np.random.normal(0, rate * 0.2), 0.2, 2.0))
        self.brainstem_reflex_stiffness = float(np.clip(self.brainstem_reflex_stiffness + np.random.normal(0, rate * 0.2), 0.5, 2.0))

    @property
    def M_t(self):
        """SpikeAEC Continuous Arbitration Signal: ACh (Explore) vs DA (Exploit)"""
        return self.ach - max(0.0, self.da)

    def food_event(self):
        # Food restores metabolic energy by 35%
        self.energy = min(1.0, self.energy + 0.35)
        self.hunger = 1.0 - self.energy
        # Dopamine and Serotonin spike on consumption (updates innate base)
        self.da_innate = min(1.0, self.da_innate + 0.6)
        self.sero_innate = min(1.0, self.sero_innate + 0.5)

    def wall_event(self, force_mag=0.2):
        # Wall collision causes stress (cortisol spike) scaled by impact force
        self.cort_innate = min(1.0, self.cort_innate + force_mag * 0.08)
        # Damage accumulates from force (scaled down for robust exploration)
        self.damage = min(1.0, self.damage + force_mag * 0.001)
        # Norepinephrine spikes on physical shock
        self.ne_innate = min(1.0, self.ne_innate + force_mag * 0.3)

    def novelty_event(self, novelty):
        # Interoceptive Prediction Error for Novelty
        pe_novelty = novelty - self.pred_novelty
        
        # ACh spikes on positive novelty surprise
        self.ach = np.clip(self.ach + pe_novelty * 0.8, 0.0, 1.0)
        
        # Update prediction
        self.pred_novelty += 0.1 * pe_novelty

    def interoceptive_step(self, actual_danger, torques=[0.0, 0.0], is_sleeping=False, velocity=0.0, joint_effort=0.0):
        """
        Called every step. Replaces the old 'decay' function.
        Calculates prediction errors for energy and safety, and updates emotions and metabolism.
        """
        # 1. Compute Actuals
        actual_energy = self.energy
        actual_safety = 1.0 - actual_danger
        
        # 2. Compute Prediction Errors
        pe_energy = actual_energy - self.pred_energy
        pe_safety = actual_safety - self.pred_safety
        
        # 3. Update Neurotransmitters (Emotions = Prediction Error)
        self.da_innate = np.clip(self.da_innate + pe_energy * 0.8, -1.0, 1.0)
        self.cort_innate = np.clip(self.cort_innate - pe_safety * 0.5, 0.0, 1.0)
        self.ne_innate = np.clip(self.ne_innate - pe_safety * 0.4 + self.ach * 0.1, 0.0, 1.0)
        self.sero_innate = np.clip(self.sero_innate + pe_safety * 0.1, 0.0, 1.0)
        
        # 4. Update Predictions
        self.pred_energy += 0.05 * pe_energy
        self.pred_safety += 0.05 * pe_safety
        self.pred_novelty *= 0.999
        
        # 5. Baseline Decay (Homeostasis)
        self.da_innate   *= 0.995
        self.ne_innate   *= 0.995
        self.cort_innate *= 0.995
        self.ach         *= 0.995
        
        # 6. Metabolic drift & Homeostasis
        if is_sleeping:
            # Sleep clears fatigue and consumes very little energy
            self.fatigue = max(0.0, self.fatigue - 0.003)
            self.last_drain = 0.00002
            self.energy = max(0.0, self.energy - 0.00002)
            # Chemoton Metabolic Repair: consume energy to repair structural damage
            if self.damage > 0.0:
                repair_amt = min(self.damage, 0.01)
                self.damage -= repair_amt
                self.energy = max(0.0, self.energy - repair_amt * 0.05)
        else:
            # Awake accumulates fatigue and drains energy based on wheel torque effort, chassis speed, and arm/head joint activity
            self.fatigue = min(1.0, self.fatigue + 0.000005)
            wheel_effort = abs(torques[0]) + abs(torques[1])
            # Metabolic drain: scaled down to match the physical scale of the maze and target distances
            drain = 0.000003 + wheel_effort * 0.00002 + abs(velocity) * 0.000015 + joint_effort * 0.000007
            self.last_drain = drain
            self.energy = max(0.0, self.energy - drain)
            
        self.hunger = 1.0 - self.energy

        # ── MARK VIII ANTI-PANIC CLUTCH ─────────────────────────────────
        if self.cort > self.CLUTCH_THRESHOLD:
            self.da_innate *= self.CLUTCH_DA_SUPPRESSION
            self.exploration_gain = 0.0  # Force freeze
        else:
            self.exploration_gain = min(1.0, self.exploration_gain + 0.01)

    def decay(self):
        # Backwards compatibility wrapper
        self.interoceptive_step(actual_danger=0.0)

    def speed_drive(self):
        """How urgently should Bob move? Modulated by M(t) arbitration."""
        base_speed = 0.5 + self.hunger * 0.3 - self.sero * 0.2 + self.M_t * 0.15
        return np.clip(base_speed * (1.0 - self.fatigue) * max(0.3, self.exploration_gain), 0.1, 0.75)

    def to_vec(self):
        return np.array([self.da, self.ne, self.cort, self.sero, self.hunger, self.fatigue], dtype=np.float32)


class CarlBrain:
    """
    Complete integrated brain. Entry point for the training loop.

    Mark IX — Subconsciousness Architecture:
    Adds 6 biologically-grounded subconsciousness systems:
      1. SomaticMarkerEngine: Pre-conscious gut-feeling bias (Damasio)
      2. GlobalWorkspace: Attention bottleneck / stream of consciousness (Baars)
      3. AllostaticPredictor: Future body-state prediction for preemptive behavior
      4. DefaultModeNetwork: Idle-mind processing / daydreaming / prospection
      5. CircadianOscillator: Biological clock with activity/rest rhythms
      6. BasalGangliaSelector: Habit formation through action chunking
    """
    def __init__(self, n_obs=24):
        self.n_obs = n_obs
        self.actor   = CarlActor(n_obs)
        self.critic  = CarlCritic(n_obs)
        
        # Squeeze out surprise (last dimension) if running in 34-D space
        sensory_dim = 33 if n_obs == 34 else n_obs
        self.sensory_dim = sensory_dim
        self.dreamer  = CarlDreamer(sensory_dim, n_act=2)       # World Model: short-term predictive
        self.hdc      = HyperdimensionalMemoryEngine(sensory_dim)  # HDC: long-term holographic memory
        self.drives  = BiologicalDrives()

        # Phase C3 active inference additions (lazily import to avoid cyclic imports)
        from carl_imagination import SpatiotemporalPredictor
        from carl_active_inference import ActiveInferenceEngine
        self.predictor = SpatiotemporalPredictor(obs_dim=sensory_dim, act_dim=2)
        self.active_inf = ActiveInferenceEngine()

        # Import and initialize curiosity engine & locomotion MPC planner
        from carl_curiosity import CuriosityEngine
        from carl_planner import LocomotionMPCPlanner
        self.curiosity_engine = CuriosityEngine(window_size=50)
        self.mpc_planner = LocomotionMPCPlanner(K=30, horizon=6, dt=0.02)

        # ── MARK IX: SUBCONSCIOUSNESS LAYER ──────────────────────────────
        from carl_somatic import SomaticMarkerEngine
        from carl_workspace import GlobalWorkspace
        from carl_dmn import DefaultModeNetwork
        from carl_habits import BasalGangliaSelector

        self.somatic     = SomaticMarkerEngine(self.hdc, max_markers=200)
        self.workspace   = GlobalWorkspace()
        self.dmn         = DefaultModeNetwork()
        self.habits      = BasalGangliaSelector()
        self.circadian   = CircadianOscillator()
        self.allostatic  = AllostaticPredictor()
        self.is_sleeping = False

        # Phase C4 - Sleep Replay & Inheritance
        self.replay_buffer = EpisodicBuffer(capacity=10000)
        self.dream_archive = DreamArchive(capacity=100)

        # TD learning state
        self.gamma   = 0.99
        self.prev_obs   = None
        self.prev_action = None
        self.prev_mu    = None
        self.prev_logp  = 0.0
        self.prev_value = 0.0

        # Rolling stats for normalization
        self._obs_mean = np.zeros(n_obs)
        self._obs_var  = np.ones(n_obs)
        self._obs_n    = 0
        self._rwd_mean = 0.0
        self._rwd_var  = 1.0
        self._rwd_n    = 0

        # Episodic memory: list of (obs, action, reward)
        self.trajectory = []

    def _normalize_obs(self, obs, freeze_stats=False):
        """Online normalization (Welford's algorithm)."""
        if not freeze_stats:
            self._obs_n += 1
            delta = obs - self._obs_mean
            self._obs_mean += delta / self._obs_n
            delta2 = obs - self._obs_mean
            self._obs_var += delta * delta2
        
        n = max(1, self._obs_n)
        # Apply a safety floor to variance to prevent explosion of inputs
        variance = np.maximum(self._obs_var / n, 1e-4)
        std = np.sqrt(variance + 1e-8)
        return (obs - self._obs_mean) / std

    def _normalize_reward(self, r):
        self._rwd_n += 1
        delta = r - self._rwd_mean
        self._rwd_mean += delta / self._rwd_n
        self._rwd_var += delta * (r - self._rwd_mean)
        std = np.sqrt(self._rwd_var / self._rwd_n + 1e-8)
        return r / (std + 1e-4)

    def step(self, obs_raw, reward, done=False, nest_rel_vector=None):
        """
        Full brain step:
        1. Normalize observation
        2. Train dreamer on (prev_obs, prev_action) -> current_obs
        3. Run critic to estimate value
        4. Run actor to sample action
        5. TD update with dreamer surprise as intrinsic curiosity
        6. Apply spinal reflex
        7. Return action
        """
        obs = self._normalize_obs(obs_raw)
        
        # ── DREAMER WORLD MODEL TRAINING ──────────────────────────────────
        surprise = 0.0
        if self.prev_obs is not None and self.prev_action is not None:
            sensory_dim = self.hdc.input_dim
            raw_loss = self.dreamer.train_step(self.prev_obs[:sensory_dim], self.prev_action, obs[:sensory_dim])
            surprise = np.log1p(raw_loss) / 10.0
            self.drives.novelty_event(min(0.5, surprise))
            
            # Feed raw prediction error loss to Learning Progress Curiosity Engine
            self.curiosity_engine.update_error(raw_loss)

        # ── HDC HOLOGRAPHIC MEMORY: write + query ───────────────────────────
        # 1. Encode this step's experience into the holographic trace
        obs_33 = obs_raw[:self.sensory_dim]
        self.hdc.write_episode(obs_33, self.drives, self.prev_action)  # use raw obs for HDC projection
        
        # Conditioned fear response: retrieve spatial-episodic cortisol memory association
        retrieved_cort = self.hdc.query_episodic(obs_33)
        if retrieved_cort > 0.05:
            # Inject somatic anxiety directly into cortisol drive based on spatial memory
            self.drives.cort = min(1.0, self.drives.cort + 0.15 * retrieved_cort)
            
        # 2. Query familiarity: have we been in this emotional state here before?
        familiarity = self.hdc.query_memory(obs_33)
        # familiarity > 0.15 AND high CORT = learned fear of this zone
        # familiarity < 0.02 = genuinely novel territory = exploration bonus
        hdc_danger = max(0.0, familiarity - 0.15) * self.drives.cort  # fear signal
        hdc_novelty = max(0.0, 0.02 - familiarity) * 2.0              # curiosity bonus

        # ── COMBINED INTRINSIC REWARD ──────────────────────────────────────
        # Compute noise-filtered Learning Progress derivative
        lp_curiosity = self.curiosity_engine.compute_learning_progress()
        
        # Causal path entropy: reward maximizing future path freedom (from LiDAR proximity)
        free_space_score = np.sum((1.0 - obs_raw[:8])**2)
        r_entropy = float(np.log1p(free_space_score))
        
        # Combined neuromodulated curiosity bonus: LP + HDC novelty
        curiosity_bonus = min(0.1, lp_curiosity * self.drives.curiosity_surprise_gain + hdc_novelty * self.drives.curiosity_hdc_gain)
        danger_penalty  = min(0.2, hdc_danger * 0.5)  # mild aversion from trauma zones
        total_dense_reward = reward + curiosity_bonus + 0.05 * r_entropy - danger_penalty
        r = self._normalize_reward(total_dense_reward)

        # Current state value
        v_current = self.critic.forward(obs)

        # TD update using previous step
        ne = self.drives.ne
        if self.prev_obs is not None and getattr(self, 'training', True):
            # Bootstrapped TD target
            td_target  = r + self.gamma * v_current * (1 - float(done))
            advantage  = td_target - self.prev_value

            # Update critic
            self.critic.update_value(self.prev_obs, td_target)

            # Update actor with state-dependent variables and ART-Gated familiarity
            self.actor.update_policy(
                advantage, self.prev_obs, self.prev_action, self.prev_mu, 
                ne, self.drives.cort, self.drives.sero, hdc_familiarity=familiarity
            )
            
        # Also strictly disable exploration noise if not training
        if not getattr(self, 'training', True):
            # Bypass sample_action to avoid adding noise during inference
            action = self.actor.forward(obs).copy()
            log_prob = 0.0
            self.actor.prev_obs = obs.copy()
            self.actor.prev_mu = action.copy()
            self.actor.prev_action = action.copy()
        else:
            # Sample new action from policy with thermodynamic scaling
            action, log_prob = self.actor.sample_action(obs, ne_signal=self.drives.ne, cort_signal=self.drives.cort)

        # Inject learned components from Actor's cache
        nm_out = self.actor.cache.get('nm_out', np.zeros(4))
        self.drives.da_learned = float(nm_out[0])
        self.drives.ne_learned = float(nm_out[1])
        self.drives.cort_learned = float(nm_out[2])
        self.drives.sero_learned = float(nm_out[3])

        # Chemoton Metabolic Priority Override: return home if energy is low
        if self.drives.energy < 0.20 and nest_rel_vector is not None:
            nest_dx, nest_dy = nest_rel_vector
            angle_to_nest = math.atan2(nest_dy, nest_dx)
            action[0] = 0.4  # Throttle (moderate return home speed)
            action[1] = float(np.clip(angle_to_nest * 1.5, -1.0, 1.0))  # Steering

        # Store current step for next iteration
        self.prev_obs    = obs.copy()
        self.prev_action = action.copy()
        self.prev_mu     = self.actor.prev_mu.copy()
        self.prev_logp   = log_prob
        self.prev_value  = v_current

        # Apply spinal cord reflexes (disabled duplicate check; delegated to brainstem)
        final_action = action
        reflex_fired = False

        return final_action, reflex_fired, familiarity

    def active_inference_step(self, obs_raw, nest_rel_vector=None, target_pos=None, current_pose=None, world_map=None):
        """
        Fristonian Active Inference action selection using Locomotion MPC.
        obs_raw: 34-D raw observation vector.
        """
        # Normalize the full observation
        obs = self._normalize_obs(obs_raw)
        
        # SpatiotemporalPredictor and HDC operate on the sensory state
        obs_33 = obs_raw[:self.sensory_dim]
        obs_33_norm = obs[:self.sensory_dim]
        
        # 1. Maintain rolling history of last 10 normalized observations
        if not hasattr(self, 'obs_history_window'):
            self.obs_history_window = []
        self.obs_history_window.append(obs_33_norm)
        if len(self.obs_history_window) > 10:
            self.obs_history_window.pop(0)
            
        # 2. HDC memory write and query
        if self.prev_action is not None:
            self.hdc.write_episode(obs_33, self.drives, self.prev_action)
            
        # Conditioned Pavlovian Cortisol response
        retrieved_cort = self.hdc.query_episodic(obs_33)
        if retrieved_cort > 0.05:
            self.drives.cort = min(1.0, self.drives.cort + 0.15 * retrieved_cort)
            
        familiarity = self.hdc.query_memory(obs_33)
        
        # Fallbacks for planning arguments
        if target_pos is None:
            target_pos = np.array([0.0, 0.0], dtype=np.float32)
        if current_pose is None:
            current_pose = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        # 3. Action Selection via Locomotion MPC Optimization
        if len(self.obs_history_window) == 10:
            obs_history = np.array(self.obs_history_window, dtype=np.float32)
            prev_act = self.prev_action if self.prev_action is not None else np.zeros(2, dtype=np.float32)
            action, G_val = self.mpc_planner.plan_active_inference(
                self, obs_history, prev_act, target_pos, current_pose, world_map
            )
        else:
            action = np.zeros(2, dtype=np.float32)
            G_val = 0.0
            
        # 4. Save state for next step
        self.prev_obs = obs.copy()
        self.prev_action = action.copy()
        
        # 5. Apply Braitenberg Spinal Reflexes (disabled duplicate check; delegated to brainstem)
        final_action = action
        reflex_fired = False
        
        return final_action, reflex_fired, familiarity, G_val

    def subconsciousness_tick(self, obs_raw, f_dist, vel_fwd, goal_crystallization_on, active_goal_pos, exec_action, r_extrinsic):
        """
        Consolidates the Mark IX subconsciousness ticks (Circadian, Allostatic,
        Somatic Marker, Global Workspace, and Default Mode Network) into the brain.
        """
        from carl_workspace import WorkspaceSignal

        # 1. Circadian Oscillator tick — biological clock advances
        self.circadian.step()
        self.is_sleeping = self.circadian.is_rest_phase()

        # 2. Allostatic Predictor — predict future energy state
        self.allostatic.update(self.drives.energy, self.drives.last_drain)
        allostatic_urgency = self.allostatic.compute_urgency(
            nearest_food_dist=f_dist, velocity=vel_fwd)

        # 3. Somatic Marker Engine — pre-conscious gut feeling bias
        obs_33 = obs_raw[:self.sensory_dim]
        somatic_bias, somatic_arousal = self.somatic.pre_conscious_bias(
            obs_33, self.drives)
        self.somatic.decay_step()

        # 4. Global Workspace — attention bottleneck competition
        workspace_signals = []
        # Hunger signal
        if self.drives.hunger > 0.3:
            workspace_signals.append(WorkspaceSignal(
                'hunger', self.drives.hunger,
                {'target': 'food', 'dist': f_dist}))
        # Fear signal
        if self.drives.cort > 0.4:
            workspace_signals.append(WorkspaceSignal(
                'fear', self.drives.cort))
        # Curiosity signal
        curiosity_salience = min(0.8, self.curiosity_engine.compute_learning_progress() * 5.0)
        if curiosity_salience > 0.2:
            workspace_signals.append(WorkspaceSignal(
                'curiosity', curiosity_salience))
        # Somatic marker signal (gut feeling)
        if abs(somatic_bias) > 0.05:
            workspace_signals.append(WorkspaceSignal(
                'somatic', abs(somatic_bias),
                {'bias': somatic_bias}))
        # Allostatic urgency
        if allostatic_urgency > 0.2:
            workspace_signals.append(WorkspaceSignal(
                'allostasis', allostatic_urgency))
        # Goal signal
        if goal_crystallization_on and active_goal_pos is not None:
            workspace_signals.append(WorkspaceSignal(
                'goal', 0.5 + self.drives.hunger * 0.3,
                {'target': active_goal_pos}))
        # Circadian exploration bias
        explore_bias = self.circadian.get_exploration_bias()
        if abs(explore_bias) > 0.1:
            workspace_signals.append(WorkspaceSignal(
                'curiosity', 0.3 + abs(explore_bias)))

        broadcast = self.workspace.compete_and_broadcast(workspace_signals)

        # 5. Default Mode Network — idle mind processing
        if self.dmn.should_activate(broadcast, is_sleeping=self.is_sleeping):
            self.dmn.idle_step(somatic_engine=self.somatic)
        # Record experience for DMN replay buffer
        self.dmn.record_experience(
            obs_raw, exec_action,
            r_extrinsic, self.drives)

        # 6. Somatic marker post-experience update
        self.somatic.post_experience_update(obs_33, self.drives)

        return broadcast, somatic_bias

    def food_eaten(self):
        self.drives.food_event()
        self.actor.total_food_lifetime += 1

    def wall_hit(self):
        self.drives.wall_event()

    def save(self, path):
        with self.predictor.lock:
            predictor_params = {k: v.copy() for k, v in self.predictor.params.items()}
        np.savez(path,
                 actor=np.array([self.actor.save()], dtype=object),
                 critic=np.array([self.critic.save()], dtype=object),
                 dreamer=np.array([self.dreamer.save()], dtype=object),
                 hdc=np.array([self.hdc.save()], dtype=object),
                 predictor=np.array([predictor_params], dtype=object),
                 # Mark IX: Subconsciousness systems
                 somatic=np.array([self.somatic.save()], dtype=object),
                 workspace=np.array([self.workspace.save()], dtype=object),
                 dmn=np.array([self.dmn.save()], dtype=object),
                 habits=np.array([self.habits.save()], dtype=object),
                 circadian=np.array([self.circadian.save()], dtype=object),
                 allostatic=np.array([self.allostatic.save()], dtype=object),
                 obs_mean=self._obs_mean, obs_var=self._obs_var, obs_n=np.array([self._obs_n]),
                 rwd_mean=np.array([self._rwd_mean]), rwd_var=np.array([self._rwd_var]),
                 rwd_n=np.array([self._rwd_n]),
                 da=np.array([self.drives.da]), ne=np.array([self.drives.ne]),
                 cort=np.array([self.drives.cort]), sero=np.array([self.drives.sero]),
                 hunger=np.array([self.drives.hunger]),
                 fatigue=np.array([self.drives.fatigue]),
                 exploration_gain=np.array([self.drives.exploration_gain]))

    def load(self, path):
        if not os.path.exists(path + '.npz'):
            print(f"[BRAIN] No saved brain found at {path}. Starting fresh.")
            return False
        try:
            d = np.load(path + '.npz', allow_pickle=True)
            self.actor.load(d['actor'][0])
            self.critic.load(d['critic'][0])
            if 'dreamer' in d:
                self.dreamer.load(d['dreamer'][0])
            if 'hdc' in d:
                self.hdc.load(d['hdc'][0])
            if 'predictor' in d and hasattr(self, 'predictor'):
                with self.predictor.lock:
                    loaded_params = d['predictor'][0]
                    for k in self.predictor.params:
                        if k in loaded_params and self.predictor.params[k].shape == loaded_params[k].shape:
                            self.predictor.params[k] = loaded_params[k].copy()
            loaded_obs_mean = d['obs_mean']
            loaded_obs_var = d['obs_var']
            if loaded_obs_mean.shape != self._obs_mean.shape:
                print(f"[BRAIN] Adapting obs_mean and obs_var shape from {loaded_obs_mean.shape} to {self._obs_mean.shape}")
                new_mean = np.zeros(self.n_obs)
                new_var = np.ones(self.n_obs)
                min_in = min(len(loaded_obs_mean), self.n_obs)
                new_mean[:min_in] = loaded_obs_mean[:min_in]
                new_var[:min_in] = loaded_obs_var[:min_in]
                self._obs_mean = new_mean
                self._obs_var = new_var
            else:
                self._obs_mean = loaded_obs_mean
                self._obs_var = loaded_obs_var
            self._obs_n    = int(d['obs_n'][0])
            self._rwd_mean = float(d['rwd_mean'][0])
            self._rwd_var  = float(d['rwd_var'][0])
            self._rwd_n    = int(d['rwd_n'][0])
            
            if self._obs_n > 10 and np.mean(self._obs_var) < 10.0:
                self._obs_var *= self._obs_n
            if self._rwd_n > 10 and self._rwd_var < 10.0:
                self._rwd_var *= self._rwd_n
            self.drives.da    = float(d['da'][0])
            self.drives.ne    = float(d['ne'][0])
            self.drives.cort  = float(d['cort'][0])
            self.drives.sero  = float(d['sero'][0])
            self.drives.hunger = float(d['hunger'][0])
            self.drives.fatigue = float(d.get('fatigue', [0.0])[0])
            self.drives.exploration_gain = float(d.get('exploration_gain', [1.0])[0])
            # Mark IX: Load subconsciousness systems (backward compatible)
            if 'somatic' in d:
                self.somatic.load(d['somatic'][0])
            if 'workspace' in d:
                self.workspace.load(d['workspace'][0])
            if 'dmn' in d:
                self.dmn.load(d['dmn'][0])
            if 'habits' in d:
                self.habits.load(d['habits'][0])
            if 'circadian' in d:
                self.circadian.load(d['circadian'][0])
            if 'allostatic' in d:
                self.allostatic.load(d['allostatic'][0])
            print(f"[BRAIN] Loaded brain from {path} (step {self._obs_n})")
            return True
        except Exception as e:
            print(f"[BRAIN] Failed to load brain: {e}. Starting fresh.")
            return False
