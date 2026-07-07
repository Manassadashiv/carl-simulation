# genesis_reservoir.py
# The Brain of CARL Genesis.
# A Liquid State Machine at criticality + a Hebbian Reflex Layer.
# Nothing here is scripted. Everything emerges from dynamics and experience.

import numpy as np
import math

# ─────────────────────────────────────────────────────────────────────────────
# LIQUID STATE MACHINE — The Deliberative Brain
# A reservoir of recurrently connected neurons initialized at criticality.
# The reservoir is NEVER trained. Its dynamics are its intelligence.
# Only the readout layer learns — from experience, online, continuously.
# ─────────────────────────────────────────────────────────────────────────────

class LiquidStateMachine:
    """
    A Liquid State Machine (Maass et al., 2002) initialized at the edge of chaos.

    The reservoir is a fixed, randomly connected dynamical system.
    Rich internal dynamics encode temporal patterns in high-dimensional space.
    A linear readout layer learns to extract navigation commands from this space.

    Parameters
    ----------
    n_reservoir : int — number of reservoir neurons (128 for stable debug, 500 for full)
    n_inputs    : int — sensory input dimensions (16 lidar + 4 internal = 20)
    n_outputs   : int — motor outputs (left_wheel, right_wheel, neck, head_pan)
    spectral_radius : float — target spectral radius (0.97 = near-critical)
    leak_rate   : float — how fast reservoir state updates (0.3 = biological timescale)
    sparsity    : float — fraction of active connections (0.10 = sparse like cortex)
    """

    def __init__(self, n_reservoir=128, n_inputs=20, n_outputs=4,
                 spectral_radius=0.97, leak_rate=0.3, sparsity=0.10):
        self.n_res    = n_reservoir
        self.n_in     = n_inputs
        self.n_out    = n_outputs
        self.leak     = leak_rate

        # ── Reservoir weights: fixed, random, sparse, at criticality ──────────
        self.W = self._init_critical_reservoir(n_reservoir, sparsity, spectral_radius)

        # ── Input weights: random projection into reservoir ────────────────────
        self.W_in = (np.random.randn(n_reservoir, n_inputs) * 0.1).astype(np.float32)

        # ── Readout weights: start zero, learn online from experience ──────────
        self.W_out = np.zeros((n_outputs, n_reservoir), dtype=np.float32)

        # ── Reservoir state: the "working memory" of the brain ─────────────────
        self.state = np.zeros(n_reservoir, dtype=np.float32)

        # ── Autopoiesis: track synapse usage for pruning ───────────────────────
        self.activity_trace = np.zeros(n_reservoir, dtype=np.float32)
        self.step_count     = 0

        # ── Readout learning: online gradient with eligibility trace ───────────
        self.eligibility_trace = np.zeros((n_outputs, n_reservoir), dtype=np.float32)
        self.readout_lr        = 0.002   # Learning rate
        self.eligibility_decay = 0.95    # Eligibility trace decay

    def _init_critical_reservoir(self, n, sparsity, target_sr):
        """Initialize reservoir at the edge of chaos (spectral radius ≈ target_sr).
        
        This is done ONCE. The reservoir never changes after this.
        The criticality is the condition for life. We do not maintain it —
        we trust the physics to preserve it under natural operation.
        """
        # Random sparse connectivity
        W = np.random.randn(n, n).astype(np.float32)
        mask = (np.random.random((n, n)) < sparsity).astype(np.float32)
        np.fill_diagonal(mask, 0.0)   # No self-connections
        W = W * mask

        # Scale to target spectral radius
        eigenvalues = np.linalg.eigvals(W)
        rho = np.max(np.abs(eigenvalues))
        if rho > 1e-6:
            W = W * (target_sr / rho)

        # Measure and report actual branching ratio
        actual_sr = np.max(np.abs(np.linalg.eigvals(W)))
        print(f'  [RESERVOIR] Initialized: {n} neurons, sigma={actual_sr:.4f} '
              f'(target {target_sr}, sparsity={sparsity})')
        return W.astype(np.float32)

    def measure_criticality(self):
        """Measure current spectral radius — the signature of criticality."""
        return float(np.max(np.abs(np.linalg.eigvals(self.W))))

    def step(self, inputs):
        """
        Advance reservoir one timestep.
        
        Uses leaky integrator dynamics — the biological timescale of dendrites.
        State update: x(t) = (1-α)x(t-1) + α·tanh(W·x(t-1) + W_in·u(t))
        """
        inputs = np.asarray(inputs, dtype=np.float32)
        pre = self.W @ self.state + self.W_in @ inputs
        self.state = ((1.0 - self.leak) * self.state
                      + self.leak * np.tanh(pre))

        # Track activity for autopoiesis
        self.activity_trace = 0.999 * self.activity_trace + 0.001 * np.abs(self.state)
        self.step_count += 1

        return self.state.copy()

    def readout(self):
        """Project reservoir state to motor commands via learned readout."""
        raw = self.W_out @ self.state
        return np.tanh(raw)   # Bounded [-1, 1]

    def update_readout(self, td_error, nm_DA):
        """
        Online readout learning — reward-modulated gradient descent.
        
        td_error : (n_outputs,) — prediction error signal
        nm_DA    : float — dopamine level (gates learning, like biological RL)
        
        Learning only happens when dopamine is elevated — reward signals
        are only incorporated when the brain is in a rewarding state.
        This is biologically faithful reward-modulated STDP at the readout level.
        """
        # Update eligibility trace (what the reservoir was doing recently)
        self.eligibility_trace = (self.eligibility_decay * self.eligibility_trace
                                  + np.outer(td_error, self.state))

        # Gate learning by dopamine — high DA = strong learning signal
        da_gate = float(np.clip(nm_DA * 1.5, 0.0, 1.0))
        dW = self.readout_lr * da_gate * self.eligibility_trace
        self.W_out = np.clip(self.W_out + dW, -3.0, 3.0)

    def autopoietic_prune(self, energy_level):
        """
        Autopoiesis: maintain the brain's own organization using available energy.
        
        When energy is abundant — all synapses maintained.
        When energy is low — weak, unused synapses are pruned to conserve resources.
        
        This is the metabolic cost of cognition. The brain eats to think.
        """
        if self.step_count % 2400 == 0:   # Every ~10 seconds at 240Hz
            # Metabolic threshold: lower energy = more aggressive pruning
            prune_threshold = float(np.clip((100.0 - energy_level) / 200.0,
                                            0.001, 0.05))

            # Prune readout weights of least-active neurons
            inactive_mask = (self.activity_trace < np.percentile(
                self.activity_trace, prune_threshold * 100))
            self.W_out[:, inactive_mask] *= 0.98   # Gradual decay, not sudden death

    def pretrain_from_ltm(self, ltm_T, ltm_P, n_samples=2000):
        """
        Bootstrap the readout from Phase 18's 362,000 steps of experience.
        
        CARL Genesis does not start from zero. It inherits what came before.
        The reservoir has never seen the world. The readout has been shaped
        by 362,000 steps of survival. This is the inheritance.
        """
        if ltm_T is None:
            print('  [RESERVOIR] No LTM available — starting from scratch.')
            return

        print(f'  [RESERVOIR] Pre-training readout from {n_samples} LTM samples...')

        # LTM stores (state, action) pairs — use to warm-start readout
        # Feed synthetic sensor patterns through reservoir, collect states,
        # regress readout weights to produce LTM-consistent actions.
        np.random.seed(42)
        errors = []
        for i in range(n_samples):
            # Synthetic sensor state from LTM distribution
            fake_sensors = np.random.randn(self.n_in).astype(np.float32) * 0.3
            res_state = self.step(fake_sensors)

            # Target: what Phase 18 would have done
            # Map LTM action (scalar torque) to readout format
            ltm_idx = np.random.randint(0, ltm_T.shape[0])
            target_drive = float(np.clip(ltm_T[ltm_idx, 1], -1.0, 1.0))

            current_out = self.W_out @ res_state
            target = np.array([target_drive, target_drive], dtype=np.float32)
            error  = target - np.tanh(current_out)
            # Direct least-squares update
            self.W_out += 0.001 * np.outer(error, res_state)
            errors.append(float(np.mean(np.abs(error))))

        print(f'  [RESERVOIR] Pre-training complete. '
              f'Mean error: {np.mean(errors[-200:]):.4f}')


# ─────────────────────────────────────────────────────────────────────────────
# REFLEX LAYER — The Spinal Cord
# Born empty. Wires itself through experience.
# When confident enough: fires before the reservoir can deliberate.
# ─────────────────────────────────────────────────────────────────────────────

class ReflexLayer:
    """
    The spinal cord of CARL Genesis.
    
    A direct sensory→motor mapping that starts at zero and wires itself
    through Hebbian learning reinforced by neuromodulatory reward signals.
    
    When a reflex is confident enough (above threshold), it fires and
    bypasses the reservoir entirely. The body acts before the mind knows.
    
    This is the most biologically faithful component in the architecture.
    
    Parameters
    ----------
    n_sensors : int — input dimension (same as LSM inputs)
    n_motors  : int — motor outputs (left_wheel, right_wheel only — reflexes are fast)
    threshold : float — minimum confidence to fire (overrides deliberation)
    """

    def __init__(self, n_sensors=20, n_motors=2, threshold=0.55):
        # Born empty — no reflexes at all
        self.W = np.zeros((n_motors, n_sensors), dtype=np.float32)
        self.threshold  = threshold
        self.n_motors   = n_motors
        self.n_sensors  = n_sensors

        # Eligibility trace: what sensor-motor co-activations happened recently?
        self.eligibility = np.zeros_like(self.W)

        # History: track which reflexes have fired for observability
        self.fired_count   = 0
        self.total_count   = 0
        self.last_fired    = False
        self.confidence    = 0.0

    def step(self, sensors, reservoir_drive, reward_signal, ne_signal):
        """
        One reflex timestep.
        
        sensors         : (n_sensors,) — current sensory input
        reservoir_drive : (2,) — what the reservoir suggests for left/right wheels
        reward_signal   : float — DA-based reward (positive = good, negative = bad)
        ne_signal       : float — NE-based urgency (high = fear/pain, strong gating)
        
        Returns
        -------
        drive   : (2,) — left/right wheel commands
        fired   : bool — True if reflex overrode the reservoir
        """
        sensors = np.asarray(sensors, dtype=np.float32).copy()
        # CRITICAL FIX: Convert Lidar to proximity with a strict 0.25m cutoff.
        # Reflexes are for immediate survival (touch/looming). They should be blind to distant walls!
        sensors[:16] = np.clip((0.25 - sensors[:16]) / 0.25, 0.0, 1.0)
        # CRITICAL FIX 2: Spinal cord (reflex) must ONLY react to physical environment, not internal thoughts!
        sensors[16:] = 0.0
        
        self.total_count += 1
        
        # Gradual autopoietic weight decay for the spinal cord
        self.W *= 0.9999

        # ── Compute reflex prediction ──────────────────────────────────────────
        raw        = self.W @ sensors
        reflex_out = np.tanh(raw)
        self.confidence = float(np.max(np.abs(raw)))

        # ── Update eligibility trace ───────────────────────────────────────────
        # Eligibility = recent sensory-motor co-activation
        # Decays quickly — only recent co-activations are eligible for learning
        # CRITICAL FIX: wire based on the actual drive (reservoir), not the empty reflex prediction!
        actual_motor = np.asarray(reservoir_drive, dtype=np.float32)
        self.eligibility = 0.85 * self.eligibility + np.outer(actual_motor, sensors)

        # ── Hebbian reinforcement ──────────────────────────────────────────────
        # Learning gated by: reward magnitude AND norepinephrine (fear/urgency)
        # Pain (NE spike) teaches avoidance. Reward (DA) teaches approach.
        # This is reward-modulated Hebbian learning — closest to biological STDP.
        ne_gate     = float(np.clip(ne_signal,    0.0, 1.0))
        reward_gate = float(np.clip(abs(reward_signal), 0.0, 1.0))
        gate        = max(ne_gate, reward_gate * 0.5)

        if gate > 0.05:
            lr   = 0.008 * gate
            sign = np.sign(reward_signal) if abs(reward_signal) > 0.05 else -ne_gate
            dW   = lr * sign * self.eligibility
            self.W = np.clip(self.W + dW, -2.5, 2.5)

        # ── Reflex fires? ──────────────────────────────────────────────────────
        # If confidence exceeds threshold AND reflex has had time to wire itself
        # (at least 500 steps of experience), override the reservoir.
        if self.confidence > self.threshold and self.total_count > 500:
            self.fired_count += 1
            self.last_fired   = True
            return reflex_out, True

        self.last_fired = False
        return reservoir_drive, False

    def reflex_ratio(self):
        """Fraction of timesteps where reflex overrode deliberation."""
        if self.total_count == 0:
            return 0.0
        return self.fired_count / self.total_count

    def strongest_reflexes(self, n=3):
        """Return the n strongest wired reflex connections for observability."""
        magnitudes = np.sum(np.abs(self.W), axis=0)
        top_idx    = np.argsort(magnitudes)[::-1][:n]
        return [(int(i), float(magnitudes[i])) for i in top_idx]

    def save(self, path):
        np.save(path, self.W)
        print(f'  [REFLEX] Saved to {path}')

    def load(self, path):
        try:
            self.W = np.load(path).astype(np.float32)
            print(f'  [REFLEX] Loaded from {path} — '
                  f'{np.count_nonzero(np.abs(self.W) > 0.1)} active connections')
        except FileNotFoundError:
            print(f'  [REFLEX] No saved reflexes found — starting from scratch')


# ─────────────────────────────────────────────────────────────────────────────
# NEUROTRANSMITTER SYSTEM
# Carried over from Phase 18 — biologically faithful chemical dynamics.
# DA: dopamine — reward, motivation, pleasure
# NE: norepinephrine — fear, alertness, urgency
# 5HT: serotonin — contentment, social bond, mood
# ACh: acetylcholine — curiosity, learning, attention
# ─────────────────────────────────────────────────────────────────────────────

class Neurotransmitters:
    def __init__(self):
        self.DA  = 0.5
        self.NE  = 0.2
        self.SHT = 0.6
        self.ACh = 0.4

    def update(self, surprise, danger, dist_to_goal, reached_goal,
               sibling_died, allostatic_load):
        # Dopamine: rises with reward, falls with failure
        if reached_goal:
            self.DA = min(1.0, self.DA + 0.4)
        self.DA = self.DA * 0.999 + 0.001 * (0.5 - danger * 0.3 + 0.2 * (1.0 - dist_to_goal / 9.0))

        # Norepinephrine: rises with danger and surprise
        self.NE = float(np.clip(self.NE * 0.995 + 0.005 * (danger * 2.0 + surprise), 0.0, 1.0))

        # Serotonin: falls with death events, rises with safety
        if sibling_died:
            self.SHT = max(0.0, self.SHT - 0.3)
        self.SHT = float(np.clip(self.SHT * 0.9995 + 0.0005 * (1.0 - danger), 0.0, 1.0))

        # Acetylcholine: rises with novelty/surprise, drives curiosity
        self.ACh = float(np.clip(self.ACh * 0.998 + 0.002 * (surprise + 0.3), 0.0, 1.0))

        # Allostatic modulation
        self.NE  = min(1.0, self.NE  + allostatic_load * 0.01)
        self.SHT = max(0.0, self.SHT - allostatic_load * 0.005)

    def effective_danger(self, raw_danger):
        return float(np.clip(raw_danger * (1.0 + self.NE * 0.5), 0.0, 1.0))

    @property
    def mood(self):
        """Scalar mood: positive = good, negative = bad."""
        return float(self.DA - self.NE * 0.6 + self.SHT * 0.3 - 0.3)


# ─────────────────────────────────────────────────────────────────────────────
# SELF-MODEL — Metacognition
# CARL monitors its own internal state before acting.
# This is the functional architecture of self-awareness.
# ─────────────────────────────────────────────────────────────────────────────

class SelfModel:
    """
    CARL's model of itself.
    
    Not a model of the world — a model of CARL.
    Before acting, CARL queries this model. Actions are modulated by self-knowledge.
    This is the second-order loop: responding to one's response to the world.
    """

    def __init__(self):
        self.state = {
            'confidence':  0.5,
            'fear':        0.0,
            'curiosity':   0.5,
            'hunger':      0.0,
            'grief':       0.0,
            'fatigue':     0.0,
            'social_need': 0.0,
        }
        self.neck_current = 0.15
        self.neck_velocity = 0.0
        self.pan_current = 0.0
        self.pan_velocity = 0.0

    def update(self, nm, energy, grief, fatigue, dist_to_sibling, reservoir_uncertainty):
        self.state['confidence']  = float(np.clip(1.0 - reservoir_uncertainty, 0.0, 1.0))
        self.state['fear']        = float(nm.NE)
        self.state['curiosity']   = float(nm.ACh)
        self.state['hunger']      = float(np.clip(1.0 - energy / 100.0, 0.0, 1.0))
        self.state['grief']       = float(np.clip(grief, 0.0, 1.0))
        self.state['fatigue']     = float(np.clip(fatigue, 0.0, 1.0))
        self.state['social_need'] = float(np.clip(1.0 / (dist_to_sibling + 0.1) * 0.5
                                                  * nm.NE, 0.0, 1.0))

    def modulate_drive(self, left, right, goal_override=None):
        """
        Modulate motor output based on self-knowledge.
        
        This is not scripted behavior. This is the self-model influencing
        the reservoir's output — metacognition shaping action.
        """
        s = self.state

        # Hunger overrides purpose — survival first
        if s['hunger'] > 0.7:
            speed_scale = 0.6   # Conserve energy while seeking food
        elif s['fear'] > 0.7 and s['hunger'] < 0.3:
            speed_scale = 0.3   # Fear + not starving = caution
        elif s['confidence'] > 0.8 and s['fear'] < 0.2:
            speed_scale = 1.2   # Confident and safe = move freely
        else:
            speed_scale = 1.0

        # Social need: if grieving and afraid, seek sibling (modulate steering)
        # (Actual sibling-seeking is handled in genesis_run.py goal selection)

        return float(np.clip(left  * speed_scale, -1.0, 1.0)), \
               float(np.clip(right * speed_scale, -1.0, 1.0))

    def neck_target(self, step, DT):
        """
        Compute neck height from internal state — emotional body language.
        
        This is not animation. This is the body expressing the chemistry.
        """
        s = self.state
        target = (0.15                              # Resting
                + 0.25 * s['curiosity']             # Curiosity raises
                - 0.20 * s['fear']                  # Fear retracts
                + 0.10 * (1.0 - s['hunger'])        # Satiated = relaxed = higher
                - 0.15 * s['grief']                 # Grief droops
                + 0.04 * math.sin(step * DT * 1.8)) # Idle breathing
        
        target = float(np.clip(target, 0.02, 0.38))
        
        # Snappy bird-like saccadic movements with overshoot
        self.neck_velocity += 0.3 * (target - self.neck_current)
        self.neck_velocity *= 0.7  # damping
        self.neck_current += self.neck_velocity
        
        return float(np.clip(self.neck_current, 0.02, 0.38))

    def head_pan_target(self, lidar_rays_16, step, DT):
        """
        Head turns toward novel or threatening detections autonomously.
        This is the looming reflex at the expression level.
        """
        # Find direction of nearest object (highest threat)
        min_idx = int(np.argmin(lidar_rays_16))
        # Map ray index to angle: rays go 0→360° in 22.5° steps
        angle   = (min_idx * 22.5 - 180.0) * math.pi / 180.0

        # Only pan toward if closer than 1.5m (looming response)
        if lidar_rays_16[min_idx] < 1.5:
            urgency = float(np.clip(1.0 - lidar_rays_16[min_idx] / 1.5, 0.0, 1.0))
            return float(np.clip(angle * urgency * 0.8, -1.047, 1.047))

        # Idle scan: gentle curiosity-driven exploration
        idle_pan = 0.3 * self.state['curiosity'] * math.sin(step * DT * 0.4)
        return float(np.clip(idle_pan, -1.047, 1.047))
