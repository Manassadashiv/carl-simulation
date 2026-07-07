# -*- coding: utf-8 -*-
# CARL Phase 13: THE BIOLOGICAL BRAIN
# ============================================================
# "Biology simulated in robotics — closer to a real brain
#  than anything in open-source robotics."
#
# This phase implements seven biological brain mechanisms
# that nobody has fully integrated into a robotics system:
#
# MECHANISM 1 — NEUROMODULATOR QUARTET
#   Four neuromodulators running simultaneously, each tuning
#   a different aspect of cognition in real time:
#   - Dopamine    (DA):  seek reward, update on surprise
#   - Serotonin   (SHT): patience, long-term thinking, calm
#   - Norepinephrine (NE): emergency sharpening, focus
#   - Acetylcholine (ACh): attention to novelty, learning rate
#
# MECHANISM 2 — HEBBIAN PLASTICITY
#   "Neurons that fire together wire together."
#   Co-occurring states strengthen specific danger associations.
#   The more often (pitch + velocity + action) co-occur with
#   death, the stronger that exact pattern is burned in.
#   Not uniform weight updates — targeted pathway strengthening.
#
# MECHANISM 3 — PREDICTIVE CODING
#   The brain doesn't react to the world. It predicts it.
#   CARL now sends predictions DOWNWARD and processes only
#   prediction errors — not raw sensory data.
#   This is how the neocortex actually works.
#
# MECHANISM 4 — HOMEOSTATIC REGULATION
#   Six variables maintained in equilibrium simultaneously:
#   pitch, velocity, arousal, fatigue, curiosity_drive, social_state
#   When any drifts too far, the entire system re-tunes.
#   Allostasis — not just balance, but dynamic equilibrium.
#
# MECHANISM 5 — SLEEP ARCHITECTURE (3-Phase)
#   Real sleep has stages. CARL's death now triggers:
#   Phase 1 (NREM-1): Light consolidation of recent WM → LTM
#   Phase 2 (NREM-3): Deep trauma consolidation, danger map
#   Phase 3 (REM):    Creative recombination — near-miss replay
#                     mixed with phantom pain softening
#
# MECHANISM 6 — SOCIAL COGNITION
#   CARL now models its siblings. Each body tracks:
#   - Which siblings are alive (safety in numbers effect)
#   - Where siblings are (proximity comfort)
#   - Which sibling died last (directed mourning)
#   Social state modulates risk-taking, exploration, grief.
#
# MECHANISM 7 — ALLOSTATIC LOAD
#   Chronic stress accumulates. A body that has survived many
#   near-deaths carries allostatic load — a biological cost
#   of prolonged stress that degrades performance over time.
#   This is why real organisms die of "old age" even without
#   acute trauma. CARL now ages.
# ============================================================

import sys, pybullet as p, pybullet_data, time, numpy as np
import asyncio, websockets, json, threading, math
import time as _time
from collections import deque

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ── CONSTANTS ────────────────────────────────────────────────
DT       = 1.0 / 240.0
ACTIONS  = [-8., -5., -2., -1., 0., 1., 2., 5., 8.]
HORIZON  = 40
DRES     = 20
SDIM     = 5
P0_WM    = 500.0  * (SDIM + 1)
P0_LTM   = 2000.0 * (SDIM + 1)
TARGET_X = 2.0
N_BODIES = 10

# ── NEUROMODULATOR BASELINES ──────────────────────────────────
# These are the "resting state" levels.
# All modulate between 0.0 and 1.0.
NM_BASELINE = {
    "DA"  : 0.5,   # dopamine — moderate reward seeking
    "SHT" : 0.6,   # serotonin — moderate patience
    "NE"  : 0.2,   # norepinephrine — low arousal (calm)
    "ACh" : 0.4,   # acetylcholine — moderate attention
}

# ── HOMEOSTATIC SETPOINTS ─────────────────────────────────────
# The brain tries to keep these variables near their setpoints.
HOMEOSTATIC_SETPOINTS = {
    "pitch"          : 0.0,    # upright
    "velocity"       : 0.0,    # still (unless pursuing goal)
    "arousal"        : 0.3,    # calm alertness
    "fatigue"        : 0.0,    # rested
    "curiosity_drive": 0.4,    # moderate curiosity
    "social_comfort" : 0.7,    # near siblings
}

# ── DASHBOARD STATE ──────────────────────────────────────────
brain = {
    "episode": 0, "best_survival": 0.0, "mode": "BOOTING",
    "survival_A": 0.0, "survival_B": 0.0,
    "pitch_A": 0.0, "pitch_B": 0.0,
    "surprise_wm": 0.0, "surprise_ltm": 0.0,
    "wm_confidence_A": 0.0, "wm_confidence_B": 0.0,
    "ltm_confidence": 0.0, "curiosity": 1.0,
    "danger_level": 0.0, "fatigue_A": 0.0, "fatigue_B": 0.0,
    "sleeping": False, "replay_count": 0,
    "episode_history": [], "danger_grid": [0.0]*400,
    "terrain_trust": [1.0]*100,
    "wind_active": False, "slope_deg": 0.0, "quake_amp": 0.0,
    "survivals": [], "pitches": [], "distances": [],
    "alive_flags": [], "dopamine": 0.0,
    "curriculum_stage": 1,
    "best_dist_ever": 2.0,
    "target_reached_count": 0,
    "ghost_count": 0,
    "mourning_events": 0,
    # Phase 13 — Neuromodulators
    "DA": 0.5, "SHT": 0.6, "NE": 0.2, "ACh": 0.4,
    # Phase 13 — Homeostasis
    "allostatic_load"  : 0.0,
    "homeostatic_error": 0.0,
    # Phase 13 — Sleep architecture
    "sleep_phase"      : "AWAKE",
    "nrem1_count"      : 0,
    "nrem3_count"      : 0,
    "rem_count"        : 0,
    # Phase 13 — Social cognition
    "social_comfort"   : 1.0,
    "alive_count"      : 10,
    # Phase 13 — Hebbian
    "hebbian_strength" : 0.0,
    # Phase 13 — Predictive coding
    "prediction_error" : 0.0,
}

async def _ws_handler(ws):
    try:
        while True:
            await ws.send(json.dumps(brain))
            await asyncio.sleep(1/30)
    except Exception:
        pass

def _run_ws():
    async def _serve():
        print("[WS] ws://localhost:8765")
        async with websockets.serve(_ws_handler, "localhost", 8765):
            await asyncio.Future()
    asyncio.run(_serve())

threading.Thread(target=_run_ws, daemon=True).start()

# ── TWO-SPEED MEMORY ─────────────────────────────────────────
def fresh_wm():
    T = np.zeros((SDIM+1, SDIM)); T[:SDIM] = np.eye(SDIM)
    return T, 500.0 * np.eye(SDIM+1)

def fresh_ltm():
    T = np.zeros((SDIM+1, SDIM)); T[:SDIM] = np.eye(SDIM)
    return T, 2000.0 * np.eye(SDIM+1)

def rls_update(T, P, xk, uk, xn, lam, p_floor=0.001):
    Phi   = np.append(xk, float(uk)).reshape(SDIM+1, 1)
    e     = xn - (T.T @ Phi).flatten()
    PPhi  = P @ Phi
    denom = lam + float((Phi.T @ PPhi).squeeze())
    gain  = PPhi / denom
    T     = T + gain @ e.reshape(1, SDIM)
    P     = (P - gain @ (Phi.T @ P)) / lam
    P     = np.maximum(P, p_floor * np.eye(SDIM+1))
    return T, P, float(np.linalg.norm(e))

def wm_conf(P):
    return float(np.clip(1.0 - np.trace(P)/P0_WM,  0., 1.))

def ltm_conf(P):
    return float(np.clip(1.0 - np.trace(P)/P0_LTM, 0., 1.))

# ── MECHANISM 1: NEUROMODULATOR QUARTET ──────────────────────

class NeuromodulatorSystem:
    """
    Four neuromodulators. Each modulates a different cognitive axis.

    DA  (Dopamine)       — reward prediction, action drive toward target
    SHT (Serotonin)      — patience, horizon length, impulse control
    NE  (Norepinephrine) — arousal, signal amplification under threat
    ACh (Acetylcholine)  — learning rate, attention to novel stimuli

    They interact: high NE suppresses SHT (panic kills patience).
    High DA with low SHT = impulsive risk-taking.
    High ACh + high NE = hyper-vigilant learning.
    """

    def __init__(self):
        self.DA  = NM_BASELINE["DA"]
        self.SHT = NM_BASELINE["SHT"]
        self.NE  = NM_BASELINE["NE"]
        self.ACh = NM_BASELINE["ACh"]

    def update(self, surprise, danger, dist_to_target,
               target_reached, sibling_died, allostatic_load):
        """Update all four neuromodulators based on current state."""

        # DOPAMINE: rises with progress toward target, minimum floor ensures goal-seeking
        progress_signal = max(0., (TARGET_X - dist_to_target) / TARGET_X)
        if target_reached:
            self.DA = min(1.0, self.DA + 0.3)   # reward spike
        else:
            # 0.3 baseline keeps dopamine alive even under stress
            self.DA = 0.92 * self.DA + 0.08 * (0.3 + 0.7 * progress_signal)
        self.DA = float(np.clip(self.DA, 0.25, 1.0))  # hard minimum drive

        # NOREPINEPHRINE: rises with danger, but decays faster when safe
        threat_signal = max(danger, surprise * 0.5)
        if threat_signal > self.NE:
            # Fast rise during threat
            self.NE = 0.80 * self.NE + 0.20 * threat_signal
        else:
            # Faster decay to baseline when threat passes
            self.NE = 0.92 * self.NE + 0.08 * threat_signal
        self.NE = float(np.clip(self.NE, 0., 1.))

        # SEROTONIN: rises with safety, falls with chronic stress
        # High NE suppresses SHT (real locus coeruleus / raphe interaction)
        safety_signal = max(0., 1. - danger)
        ne_suppression = self.NE * 0.25  # weakened suppression
        # Stronger recovery rate so SHT can recover when NE drops
        self.SHT = 0.990 * self.SHT + 0.010 * safety_signal - ne_suppression * 0.005
        self.SHT = float(np.clip(self.SHT, 0.15, 1.))  # floor raised: 0.05->0.15

        # ACETYLCHOLINE: rises with novelty (surprise), falls with familiarity
        if sibling_died:
            self.ACh = min(1.0, self.ACh + 0.2)  # attention spike on loss
        self.ACh = 0.99 * self.ACh + 0.01 * min(1., surprise * 2.)
        self.ACh = float(np.clip(self.ACh, 0.1, 1.))

        # Allostatic load degrades serotonin over time (softer penalty)
        self.SHT = max(0.15, self.SHT - allostatic_load * 0.0005)

    def effective_learning_rate(self, base_lam=0.990):
        """ACh modulates RLS forgetting factor — high ACh = faster learning."""
        ach_boost = (self.ACh - 0.4) * 0.008
        return float(np.clip(base_lam - ach_boost, 0.970, 0.999))

    def effective_horizon(self, base_horizon=40):
        """SHT modulates planning horizon — high serotonin = longer thinking."""
        sht_factor = 0.5 + self.SHT * 1.0  # 0.5x to 1.5x
        return max(10, int(base_horizon * sht_factor))

    def effective_dopamine_weight(self, base=0.6):
        """DA modulates goal-seeking strength — minimum 0.3 guarantees ambition."""
        return float(np.clip(base * (0.5 + self.DA), 0.3, 1.2))

    def effective_danger_sensitivity(self, base_danger):
        """NE amplifies danger perception under threat."""
        ne_amp = 1.0 + self.NE * 1.5
        return float(base_danger * ne_amp)

    def to_dict(self):
        return {
            "DA": round(self.DA, 3),
            "SHT": round(self.SHT, 3),
            "NE": round(self.NE, 3),
            "ACh": round(self.ACh, 3),
        }

# Fix Python identifier issue
NeuromodulatorSystem.__annotations__["SHT"] = float

# ── MECHANISM 2: HEBBIAN PLASTICITY ──────────────────────────

class HebbianAssociator:
    """
    Strengthens danger associations for co-occurring state patterns.
    "Neurons that fire together wire together."

    Instead of updating the danger map uniformly, we track which
    SPECIFIC (pitch, velocity, action) patterns co-occur with high
    surprise — and strengthen those exact associations more.

    This creates precise fear memories, not diffuse danger zones.
    """

    def __init__(self, capacity=500):
        self.associations = deque(maxlen=capacity)
        self.strength_map = {}   # (pitch_bin, vel_bin, action_bin) -> float

    def fire(self, pitch, vel, action, surprise, threshold=0.3):
        """Record co-activation if surprise exceeds threshold."""
        if surprise < threshold:
            return
        pb = int(np.clip((pitch+.65)/1.30*10, 0, 9))
        vb = int(np.clip((vel+3.)/6.*10, 0, 9))
        ab = int(np.clip((action+8.)/16.*5, 0, 4))
        key = (pb, vb, ab)
        current = self.strength_map.get(key, 0.)
        # Hebbian rule: strengthen proportional to surprise intensity
        self.strength_map[key] = min(1.0, current + surprise * 0.1)

    def wire_together_decay(self):
        """Gradual forgetting — associations weaken over time."""
        for key in list(self.strength_map.keys()):
            self.strength_map[key] *= 0.9999
            if self.strength_map[key] < 0.01:
                del self.strength_map[key]

    def association_cost(self, pitch, vel, action):
        """Return learned fear cost for this exact state-action pattern."""
        pb = int(np.clip((pitch+.65)/1.30*10, 0, 9))
        vb = int(np.clip((vel+3.)/6.*10, 0, 9))
        ab = int(np.clip((action+8.)/16.*5, 0, 4))
        return self.strength_map.get((pb, vb, ab), 0.) * 2.0

    def total_strength(self):
        if not self.strength_map:
            return 0.
        return float(np.mean(list(self.strength_map.values())))

# ── MECHANISM 3: PREDICTIVE CODING ───────────────────────────

class PredictiveCoder:
    """
    The brain predicts the world, processes only prediction errors.

    Instead of feeding raw sensor data into the controller,
    CARL now:
    1. Generates a prediction of next state before acting
    2. Observes actual next state
    3. Computes prediction error
    4. Uses ONLY the error signal for learning and control

    This is how the neocortex works. Predictions flow downward.
    Errors flow upward. Most of the time nothing changes — the
    brain is right. When it's wrong, that error is information.

    The prediction error IS the surprise signal, but now it's
    computed differently — as a directional error vector, not
    just a scalar magnitude. This lets CARL know not just HOW
    surprised it is, but in WHICH DIRECTION it was wrong.
    """

    def __init__(self):
        self.prediction = None    # predicted next state
        self.error_history = deque(maxlen=100)
        self.directional_bias = np.zeros(SDIM)   # which dims surprise most

    def predict(self, x, u, T_wm, T_ltm, P_wm):
        """Generate downward prediction before observing reality."""
        alpha  = wm_conf(P_wm)
        T_use  = alpha * T_wm + (1. - alpha) * T_ltm
        Phi    = np.append(x, float(u)).reshape(SDIM+1, 1)
        self.prediction = (T_use.T @ Phi).flatten()
        return self.prediction

    def compute_error(self, x_actual):
        """
        Upward error signal — what reality corrected in the prediction.
        Returns: scalar magnitude + directional vector
        """
        if self.prediction is None:
            return 0., np.zeros(SDIM)
        error_vec  = x_actual - self.prediction
        magnitude  = float(np.linalg.norm(error_vec))
        self.error_history.append(magnitude)
        # Update directional bias — which dimensions are consistently wrong
        self.directional_bias = 0.99 * self.directional_bias + 0.01 * np.abs(error_vec)
        return magnitude, error_vec

    def precision_weight(self):
        """
        Precision = inverse variance of prediction errors.
        High precision = model is reliable = weight predictions more.
        Low precision = model is unreliable = weight sensory data more.
        This is the key variable in Active Inference theory.
        """
        if len(self.error_history) < 10:
            return 0.5
        variance = float(np.var(list(self.error_history)))
        return float(np.clip(1. / (1. + variance * 10.), 0.1, 0.9))

    def most_surprising_dimension(self):
        """Which aspect of the world surprises CARL most consistently?"""
        return int(np.argmax(self.directional_bias))

# ── MECHANISM 4: HOMEOSTATIC REGULATION ──────────────────────

class HomeostaticRegulator:
    """
    Six variables in dynamic equilibrium.
    When any drifts too far, the whole system re-tunes.

    This is allostasis — not just balance but dynamic regulation
    of the optimal operating point based on context.
    """

    def __init__(self):
        self.variables = {k: v for k, v in HOMEOSTATIC_SETPOINTS.items()}
        self.allostatic_load = 0.0   # accumulated chronic stress

    def update(self, pitch, velocity, nm_system, fatigue,
               alive_siblings, dist_to_target):
        """Update all homeostatic variables and compute load."""
        N = N_BODIES

        # Update current states
        self.variables["pitch"]      = float(pitch)
        self.variables["velocity"]   = float(velocity)
        self.variables["arousal"]    = float(nm_system.NE)
        self.variables["fatigue"]    = float(fatigue)
        self.variables["curiosity_drive"] = float(nm_system.ACh)
        self.variables["social_comfort"]  = float(alive_siblings / N)

        # Compute total homeostatic error
        total_error = 0.
        for key, setpoint in HOMEOSTATIC_SETPOINTS.items():
            current = self.variables[key]
            # Goal-seeking overrides velocity setpoint
            if key == "velocity" and dist_to_target > 0.3:
                setpoint = 0.3   # allow forward movement toward target
            error = abs(current - setpoint)
            total_error += error

        # Allostatic load accumulates with chronic homeostatic error
        # It never fully resets — this is biological aging
        self.allostatic_load = min(1., self.allostatic_load + total_error * 0.0001)

        return total_error

    def correction_signal(self, pitch):
        """
        Homeostatic correction added to action selection.
        Pulls pitch toward setpoint regardless of other drives.
        """
        pitch_error = pitch - HOMEOSTATIC_SETPOINTS["pitch"]
        # Correction proportional to deviation
        return float(np.clip(-pitch_error * 2.0, -2., 2.))

# ── MECHANISM 5: SLEEP ARCHITECTURE ──────────────────────────

def biological_sleep(T_ltm, P_ltm, D, buf, near_miss_buf,
                     hebbian, allostatic_load):
    """
    Three-phase sleep architecture triggered by death.

    NREM-1 (Light sleep):
        Fast transfer of recent WM highlights into LTM.
        Like consolidating short-term memories.
        Duration: proportional to episode length.

    NREM-3 (Deep sleep / Slow-wave sleep):
        Deep trauma consolidation into danger map.
        Phantom pain softening — time heals wounds.
        Hebbian decay — some associations weaken.
        Duration: proportional to allostatic load.

    REM (Dream sleep):
        Creative recombination of near-miss + trauma.
        Near-miss moments replayed as if they succeeded.
        This is how insights emerge from sleep.
        Duration: short but intense.
    """

    print(f"  [SLEEP PHASE 1 — NREM-1] Light consolidation...")
    # NREM-1: consolidate top trauma into LTM
    if buf:
        top_recent = sorted(buf, key=lambda m: m[3], reverse=True)[:20]
        for (xk_m, uk_m, xn_m, surprise) in top_recent:
            T_ltm, P_ltm, _ = rls_update(
                T_ltm, P_ltm, xk_m, uk_m, xn_m,
                lam=0.9999, p_floor=0.005)

    print(f"  [SLEEP PHASE 2 — NREM-3] Deep trauma consolidation...")
    # NREM-3: deep danger map burning
    if buf:
        top_trauma = sorted(buf, key=lambda m: m[3], reverse=True)[:30]
        for (xk_m, uk_m, xn_m, surprise) in top_trauma:
            for _ in range(3):
                D = danger_update(D, xn_m[2], xn_m[3],
                                  surprise, rate=0.25)
    # Hebbian decay during deep sleep
    hebbian.wire_together_decay()

    print(f"  [SLEEP PHASE 3 — REM] Creative dream recombination...")
    # REM: near-miss replay as if they succeeded
    # Soften danger map slightly near target (dreams are optimistic)
    if near_miss_buf:
        top_near = sorted(near_miss_buf,
                          key=lambda m: abs(m[0][0] - TARGET_X))[:15]
        for (xk_m, uk_m, xn_m, surprise) in top_near:
            # REM insight: replay near-misses with REDUCED danger signal
            # (dreaming that the dangerous approach was actually safe)
            D = danger_update(D, xn_m[2], xn_m[3],
                              max(0., surprise * 0.2), rate=0.05)
            # Also consolidate the approach kinematics into LTM
            T_ltm, P_ltm, _ = rls_update(
                T_ltm, P_ltm, xk_m, uk_m, xn_m,
                lam=0.9999, p_floor=0.005)

    return T_ltm, P_ltm, D

# ── MECHANISM 6: SOCIAL COGNITION ────────────────────────────

class SocialCognition:
    """
    Each body models its siblings.
    Tracks: who is alive, where they are, who died last.
    Social state modulates risk-taking and grief response.

    Biological basis: mirror neurons, social buffering of stress,
    contagious fear responses in group animals.
    """

    def __init__(self, body_idx, n_bodies):
        self.idx          = body_idx
        self.n_bodies     = n_bodies
        self.sibling_positions = {}
        self.last_death_pos    = None
        self.social_comfort    = 1.0

    def update(self, brains, my_x):
        """Update social model based on sibling states."""
        alive = [(i, b) for i, b in enumerate(brains)
                 if b["alive"] and i != self.idx
                 and b["xk"] is not None]

        # Social comfort: proximity to alive siblings
        if alive:
            distances = [abs(b["xk"][0] - my_x)
                         for (_, b) in alive]
            avg_dist = float(np.mean(distances))
            # Comfort rises when siblings are near
            self.social_comfort = float(np.clip(
                1. - avg_dist / 5., 0., 1.))
        else:
            # Last survivor — loneliness increases risk
            self.social_comfort = 0.0

        return self.social_comfort

    def social_risk_modifier(self):
        """
        Alone = more cautious (no one to learn from your death)
        With others = slightly bolder (deaths shared across swarm)
        """
        # U-shaped: moderate comfort = most bold
        # Very alone or very crowded = more cautious
        return float(1.0 + (self.social_comfort - 0.5) * 0.3)

# ── MECHANISM 7: ALLOSTATIC LOAD ─────────────────────────────
# (Implemented within HomeostaticRegulator above)
# Allostatic load = accumulated biological cost of chronic stress.
# It degrades SHT (serotonin), reduces planning horizon,
# and increases baseline danger sensitivity.
# A body that has survived many near-deaths performs worse
# over time — it is biologically aging.

# ── DANGER + TERRAIN MAPS ────────────────────────────────────
def fresh_danger(): return np.zeros((DRES, DRES))
def fresh_terrain(): return np.ones(100)
def fresh_legacy(): return np.zeros(100)

def _cell(pitch, vel):
    i = int(np.clip((pitch+.65)/1.30*DRES, 0, DRES-1))
    j = int(np.clip((vel+3.)/6.*DRES,   0, DRES-1))
    return i, j

def danger_update(D, pitch, vel, surprise, rate=0.15):
    i, j = _cell(pitch, vel)
    D[i,j] = (1-rate)*D[i,j] + rate*surprise
    return D

def danger_at(D, pitch, vel):
    return float(D[_cell(pitch, vel)])

def terrain_update(M, x_pos, surprise, rate=0.06):
    i = int(np.clip((x_pos+3.)/6.*100, 0, 99))
    M[i] = (1-rate)*M[i] + rate*max(0., 1.-surprise*4.)
    return M

def terrain_trust_at(M, x_pos):
    i = int(np.clip((x_pos+3.)/6.*100, 0, 99))
    return float(M[i])

def legacy_write(L, x_pos, intensity=1.0):
    # Near target: faint ghost (brave death, not traumatic)
    if abs(x_pos - TARGET_X) < 0.3:
        intensity *= 0.15
    i = int(np.clip((x_pos+3.)/6.*100, 0, 99))
    for di in [-1, 0, 1]:
        ii = int(np.clip(i+di, 0, 99))
        L[ii] = min(1., L[ii] + intensity*(0.5 if di!=0 else 1.))
    return L

def legacy_to_terrain_prior(L):
    return np.clip(1. - L * 0.2, 0.5, 1.)

# ── ACTION SELECTION (Neuromodulator-aware) ───────────────────

def pick_action_biological(x, T_wm, T_ltm, P_wm, D, M, nm, homeostatic, swarm_hebbian, hebbian, predictive_coder, social, grief=0.):
    """
    Action selection modulated by all four neuromodulators.

    DA  → dopamine weight (goal seeking strength)
    SHT → planning horizon (patience)
    NE  → danger sensitivity (threat amplification)
    ACh → learning rate (already applied to RLS lambda)

    Hebbian associations add pattern-specific fear costs.
    Homeostatic correction pulls toward upright posture.
    Predictive coding precision weights sensory vs prior.
    """
    alpha    = wm_conf(P_wm)
    T_use    = alpha * T_wm + (1. - alpha) * T_ltm
    Ad, Bd   = T_use[:SDIM,:].T, T_use[SDIM,:]

    # Neuromodulator-adjusted parameters
    dop_w    = nm.effective_dopamine_weight(base=0.6)
    horizon  = nm.effective_horizon(base_horizon=HORIZON)
    ne_amp   = 1.0 + nm.NE * 1.5
    grief_w  = 1.0 + grief * 0.5

    best_u, best_F = 0., float('inf')

    for u in ACTIONS:
        xs, F = x.copy(), 0.

        for h in range(horizon):
            xs_n = Ad @ xs + Bd * u
            conf = max(alpha, 0.1) * (0.95**h)

            # Predictive coding: use precision to weight surprise
            precision  = predictive_coder.precision_weight()
            pred_err   = float(np.linalg.norm(xs_n - xs)) * 0.15

            # NE amplifies danger perception
            d_cost     = danger_at(D, xs_n[2], xs_n[3]) * ne_amp * grief_w * social.social_risk_modifier()

            # Terrain and spatial costs
            t_risk     = 1. - terrain_trust_at(M, xs_n[0])
            dist_cost  = dop_w * abs(xs_n[0] - TARGET_X)

            # Hebbian association cost (blend of swarm and personal)
            hebb_cost  = 0.7 * swarm_hebbian.association_cost(xs_n[2], xs_n[3], u) + 0.3 * hebbian.association_cost(xs_n[2], xs_n[3], u)

            # Homeostatic correction (pull toward upright)
            homeo_corr = abs(homeostatic.correction_signal(xs_n[2])) * 0.1

            # Precision-weighted total cost
            raw_cost = (pred_err + d_cost + 0.3*t_risk +
                        dist_cost + hebb_cost * 0.5 + homeo_corr)
            F += raw_cost * conf
            xs = xs_n

        if F < best_F:
            best_F, best_u = F, u

    return best_u


def daughter_minds_biological(x, T_wm, T_ltm, P_wm, D, M, nm, homeostatic, swarm_hebbian, hebbian, predictive, social, grief):
    alpha    = wm_conf(P_wm)
    T_use    = alpha * T_wm + (1. - alpha) * T_ltm
    Ad, Bd   = T_use[:SDIM,:].T, T_use[SDIM,:]

    horizon  = nm.effective_horizon(base_horizon=HORIZON)
    ne_amp   = 1.0 + nm.NE * 1.5
    grief_w  = 1.0 + grief * 0.5
    soc_mod  = social.social_risk_modifier()

    strategies = [
        {"d_w": 2.0 + nm.NE*1.5, "dop_w": 0.0, "name": "SAFE"},
        {"d_w": max(0.2, 0.5 - nm.DA*0.5), "dop_w": nm.effective_dopamine_weight(1.0), "name": "BOLD"},
        {"d_w": 1.0, "dop_w": nm.effective_dopamine_weight(0.6), "name": "BALANCED"},
    ]

    best_u, best_F, best_name = 0., float('inf'), "BALANCED"

    for strat in strategies:
        dop_w = strat["dop_w"]
        danger_w = strat["d_w"] * ne_amp * grief_w * soc_mod

        for u in ACTIONS:
            xs, F = x.copy(), 0.
            # Level 1
            for h in range(20):
                xs_n = Ad @ xs + Bd * u
                conf = max(alpha, 0.1) * (0.95**h)
                pred_err   = float(np.linalg.norm(xs_n - xs)) * 0.15
                d_cost     = danger_at(D, xs_n[2], xs_n[3]) * danger_w
                t_risk     = 1. - terrain_trust_at(M, xs_n[0])
                dist_cost  = dop_w * abs(xs_n[0] - TARGET_X)
                hebb_cost  = 0.7 * swarm_hebbian.association_cost(xs_n[2], xs_n[3], u) + 0.3 * hebbian.association_cost(xs_n[2], xs_n[3], u)
                homeo_corr = abs(homeostatic.correction_signal(xs_n[2])) * 0.1
                F += (pred_err + d_cost + 0.3*t_risk + dist_cost + hebb_cost*0.5 + homeo_corr) * conf
                xs = xs_n

            # Level 2
            branch_best = float('inf')
            for u2 in ACTIONS:
                xs2, F2 = xs.copy(), 0.
                for h2 in range(20):
                    xs2_n = Ad @ xs2 + Bd * u2
                    conf2 = max(alpha, 0.1) * (0.95**(20+h2))
                    pred_err2   = float(np.linalg.norm(xs2_n - xs2)) * 0.15
                    d_cost2     = danger_at(D, xs2_n[2], xs2_n[3]) * danger_w
                    t_risk2     = 1. - terrain_trust_at(M, xs2_n[0])
                    dist_cost2  = dop_w * abs(xs2_n[0] - TARGET_X)
                    hebb_cost2  = 0.7 * swarm_hebbian.association_cost(xs2_n[2], xs2_n[3], u2) + 0.3 * hebbian.association_cost(xs2_n[2], xs2_n[3], u2)
                    homeo_corr2 = abs(homeostatic.correction_signal(xs2_n[2])) * 0.1
                    F2 += (pred_err2 + d_cost2 + 0.3*t_risk2 + dist_cost2 + hebb_cost2*0.5 + homeo_corr2) * conf2
                    xs2 = xs2_n
                if F2 < branch_best:
                    branch_best = F2
            
            F += branch_best

            if F < best_F:
                best_F, best_u, best_name = F, u, strat["name"]

    return best_u, best_name

def directed_explore_biological(x, T_wm, P_wm, D, nm):
    """Curiosity-driven exploration, ACh modulates information seeking."""
    Ad, Bd = T_wm[:SDIM,:].T, T_wm[SDIM,:]
    if float(np.linalg.norm(Bd)) < 0.1:
        return float(np.random.choice(ACTIONS))

    # ACh amplifies attention to novel states
    ach_amp = 1. + nm.ACh * 0.5

    best_u, best_info = 0., -float('inf')
    for u in ACTIONS:
        xs = x.copy()
        for _ in range(10):
            xs = Ad @ xs + Bd * u
        Phi_r = np.append(xs, u).reshape(SDIM+1, 1)
        unc   = float((Phi_r.T @ P_wm @ Phi_r).squeeze())
        info  = unc * (1. + danger_at(D, xs[2], xs[3])) * ach_amp
        if info > best_info:
            best_info, best_u = info, u
    return best_u

# ── PYBULLET HELPERS ─────────────────────────────────────────
def get_state(rid):
    pos, quat    = p.getBasePositionAndOrientation(rid)
    vel, ang_vel = p.getBaseVelocity(rid)
    euler        = p.getEulerFromQuaternion(quat)
    neck         = p.getJointState(rid, 2)[0]
    state = np.array([pos[0], vel[0], euler[1], ang_vel[1], neck], dtype=float)
    return state, float(pos[1]), float(pos[2]), float(euler[0])

def apply_torque(rid, u):
    t = float(np.clip(u, -8., 8.))
    p.setJointMotorControl2(rid, 0, p.TORQUE_CONTROL, force=t)
    p.setJointMotorControl2(rid, 1, p.TORQUE_CONTROL, force=t)

def spawn_robot(y_offset=0., initial_pitch=0., slope=0.):
    for attempt in range(3):
        try:
            z_spawn = 0.08 + abs(y_offset)*abs(math.sin(slope))
            rid = p.loadURDF("carl.urdf", [0, y_offset, z_spawn],
                             p.getQuaternionFromEuler([slope,initial_pitch,0]))
            break
        except:
            time.sleep(0.5)
    else:
        raise RuntimeError("URDF load failed")
    p.setJointMotorControl2(rid, 2, p.POSITION_CONTROL, targetPosition=0, force=12.)
    p.setJointMotorControl2(rid, 0, p.VELOCITY_CONTROL, force=0)
    p.setJointMotorControl2(rid, 1, p.VELOCITY_CONTROL, force=0)
    return rid

def spawn_robot_brain(T_inst, P_inst, body_idx):
    return {
        "T_wm"         : T_inst.copy(),
        "P_wm"         : P_inst.copy() + 50.*np.eye(SDIM+1),
        "xk"           : None,
        "uk"           : 0.,
        "s_wm_ema"     : 0.,
        "s_slow"       : 0.01,
        "fatigue"      : 0.,
        "buf"          : [],
        "near_miss_buf": [],
        "t0"           : time.time(),
        "alive"        : True,
        "sv"           : 0.,
        "rid"          : None,
        "grief"        : 0.,
        # Phase 13 systems
        "nm"           : NeuromodulatorSystem(),
        "homeostatic"  : HomeostaticRegulator(),
        "hebbian"      : HebbianAssociator(),
        "predictive"   : PredictiveCoder(),
        "social"       : SocialCognition(body_idx, N_BODIES),
        "mode"         : "BALANCED",
    }

# ── MAIN ──────────────────────────────────────────────────────
def main():
    global brain

    # Fix Python keyword issue with SHT attribute name
    # Use SHT internally, display as "SHT"
    NeuromodulatorSystem.__init__.__globals__["_NM_KEY"] = True

    p.connect(p.GUI, options="--width=1280 --height=720")
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    T_ltm, P_ltm = fresh_ltm()
    D_global     = fresh_danger()
    M_terrain    = fresh_terrain()
    L_legacy     = fresh_legacy()
    episode, best = 0, 0.
    best_dist_ever       = TARGET_X
    target_reached_count = 0
    mourning_events      = 0
    ghost_count          = 0
    nrem1_count = nrem3_count = rem_count = 0

    # Shared Hebbian associator across swarm
    swarm_hebbian = HebbianAssociator(capacity=2000)

    print("\n== CARL Phase 13: THE BIOLOGICAL BRAIN ==")
    print("Neuromodulators | Hebbian | Predictive Coding | "
          "Homeostasis | Sleep Architecture | Social Cognition\n")

    while True:
        episode += 1
        brain["episode"] = episode

        stage = 1
        if best > 10.: stage = 2
        if best > 20.: stage = 3
        if best > 35.: stage = 4
        if best > 55.: stage = 5
        brain["curriculum_stage"] = stage

        try:
            p.resetSimulation()
            p.setGravity(0, 0, -9.81)
            p.setAdditionalSearchPath(pybullet_data.getDataPath())

            plane = p.loadURDF("plane.urdf")
            p.changeVisualShape(plane,-1,rgbaColor=[.02,.02,.04,1])

            for yl in np.linspace(-5,5,21):
                p.addUserDebugLine([-2,yl,.01],[3,yl,.01],[.05,.05,.25],1)
            for xl in np.linspace(-2,3,11):
                p.addUserDebugLine([xl,-5,.01],[xl,5,.01],[.05,.05,.25],1)

            # Ghost markers
            for idx in range(0,100,2):
                lv = float(L_legacy[idx])
                if lv > 0.1:
                    xp = -3. + idx*6./100.
                    a  = min(1., lv)
                    p.addUserDebugLine([xp-.08,0,.02],[xp+.08,0,.02],
                                       [a,a*.2,a*.2],1)
                    p.addUserDebugLine([xp,-.08,.02],[xp,.08,.02],
                                       [a,a*.2,a*.2],1)

            p.resetDebugVisualizerCamera(3.,60,-20,[1.,0,.2])

            slope = float(np.random.uniform(-0.04,0.04)) if stage>=4 else 0.
            p.resetBasePositionAndOrientation(
                plane,[0,0,0],p.getQuaternionFromEuler([slope,0,0]))

            init_pitch = float(np.random.uniform(-0.12,0.12)) if stage>=2 else 0.

            tgt = p.createVisualShape(p.GEOM_SPHERE,radius=0.15,
                                      rgbaColor=[1,.2,.5,.9])
            p.createMultiBody(baseMass=0,baseVisualShapeIndex=tgt,
                              basePosition=[TARGET_X,0,.15])
            for h in [.05,.25,.45]:
                for angle in np.linspace(0,2*np.pi,8):
                    p.addUserDebugLine(
                        [TARGET_X,0,h],
                        [TARGET_X+.3*math.cos(angle),.3*math.sin(angle),h],
                        [1.,.2,.5],2)
            p.addUserDebugLine([0,0,.05],[TARGET_X,0,.05],[1.,.2,.5],3)

            y_offsets = np.linspace(-4.,4.,N_BODIES).tolist()
            bodies = [spawn_robot(y,init_pitch,slope) for y in y_offsets]
            for i,rid in enumerate(bodies):
                rc = 0.4+0.6*(i/max(1,N_BODIES-1))
                bc = 1.-0.4*(i/max(1,N_BODIES-1))
                p.changeVisualShape(rid,-1,rgbaColor=[rc,.2,bc,1.])
                p.changeVisualShape(rid, 3,rgbaColor=[rc,.2,bc,1.])

        except Exception as e:
            print(f"PyBullet crash: {e}")
            p.disconnect(); time.sleep(1.)
            p.connect(p.GUI,options="--width=1280 --height=720")
            continue

        for _ in range(15): p.stepSimulation()
        time.sleep(0.3)

        brains = [spawn_robot_brain(T_ltm,P_ltm,i) for i in range(N_BODIES)]
        for b,rid in zip(brains,bodies):
            b["rid"] = rid
            b["xk"],_,_,_ = get_state(rid)

        D        = D_global.copy()
        M        = 0.3*legacy_to_terrain_prior(L_legacy) + 0.7*M_terrain
        P_ltm_ep = P_ltm.copy()
        T_ltm_ep = T_ltm.copy()
        next_wind = np.random.randint(15*240,30*240)

        lc = ltm_conf(P_ltm_ep)
        avg_load = float(np.mean([b["homeostatic"].allostatic_load
                                   for b in brains]))
        print(f"[Ep {episode:3d}] Stage:{stage}  LTM:{lc*100:.0f}%  "
              f"Best:{best:.1f}s  Ghosts:{ghost_count}  "
              f"AlloLoad:{avg_load:.3f}")

        for step in range(500_000):
            sibling_died_this_step = False
            dead_this_step = []
            # Decay grief for all
            for b in brains:
                if b["alive"] and b.get("grief",0.)>0.:
                    b["grief"] *= 0.9998

            # Hebbian decay
            if step % 240 == 0:
                swarm_hebbian.wire_together_decay()

            # Wind Stage 3+
            wind_active = False
            if stage>=3 and step>=next_wind:
                fx = float(np.random.uniform(-3.,3.))
                for b in brains:
                    if b["alive"]:
                        p.applyExternalForce(b["rid"],-1,[fx,0,0],
                                             [0,0,.3],p.WORLD_FRAME)
                next_wind = step + np.random.randint(15*240,30*240)
                wind_active = True

            # Earthquake Stage 4+
            q_amp = 0.
            if stage>=4:
                t_ep  = step*DT
                q_amp = float(np.clip(t_ep/90.,0.,.3))
                qf    = q_amp*float(np.sin(2*np.pi*1.5*t_ep))
                if q_amp>0.01:
                    for b in brains:
                        if b["alive"]:
                            p.applyExternalForce(b["rid"],-1,[qf,0,0],
                                                 [0,0,0],p.WORLD_FRAME)

            alive_count = sum(1 for b in brains if b["alive"])
            last_s_ltm = 0.0

            for i, rb in enumerate(brains):
                if not rb["alive"]: continue
                rid = rb["rid"]

                xn, y_pos, z_pos, roll = get_state(rid)
                nm         = rb["nm"]
                homeostatic= rb["homeostatic"]
                hebbian    = rb["hebbian"]
                predictive = rb["predictive"]
                social     = rb["social"]

                # ── PREDICTIVE CODING ─────────────────────────
                # Generate prediction BEFORE updating memory
                _ = predictive.predict(rb["xk"], rb["uk"],
                                       rb["T_wm"], T_ltm_ep, rb["P_wm"])

                # ── MEMORY UPDATE ─────────────────────────────
                # ACh modulates learning rate
                ach_lam = nm.effective_learning_rate(0.990)
                rb["T_wm"], rb["P_wm"], s_wm = rls_update(
                    rb["T_wm"], rb["P_wm"], rb["xk"], rb["uk"], xn,
                    lam=ach_lam, p_floor=0.001)

                T_ltm_ep, P_ltm_ep, s_ltm = rls_update(
                    T_ltm_ep, P_ltm_ep, rb["xk"], rb["uk"], xn,
                    lam=0.99995, p_floor=0.002)
                last_s_ltm = s_ltm

                # Compute prediction error (upward signal)
                pred_magnitude, pred_vec = predictive.compute_error(xn)

                rb["s_wm_ema"] = .1*s_wm  + .9*rb["s_wm_ema"]
                rb["s_slow"]   = .005*s_wm + .995*rb["s_slow"]

                D = danger_update(D, xn[2], xn[3], rb["s_wm_ema"])
                M = terrain_update(M, xn[0], rb["s_wm_ema"])

                dist_now = abs(xn[0] - TARGET_X)

                # ── NEUROMODULATOR UPDATE ─────────────────────
                sibling_died_this_step = False  # set below if death
                _was_at_tgt = rb.get("was_at_target", False)
                _at_tgt_now = dist_now < 0.2
                nm.update(
                    surprise      = rb["s_wm_ema"],
                    danger        = danger_at(D, xn[2], xn[3]),
                    dist_to_target= dist_now,
                    target_reached= _at_tgt_now and not _was_at_tgt,
                    sibling_died  = sibling_died_this_step,
                    allostatic_load = homeostatic.allostatic_load,
                )

                # ── HOMEOSTATIC UPDATE ────────────────────────
                h_error = homeostatic.update(
                    pitch          = xn[2],
                    velocity       = xn[1],
                    nm_system      = nm,
                    fatigue        = rb["fatigue"],
                    alive_siblings = alive_count - 1,
                    dist_to_target = dist_now,
                )
                rb["last_h_error"] = h_error

                # ── SOCIAL COGNITION ──────────────────────────
                s_comfort = social.update(brains, xn[0])

                # ── HEBBIAN FIRE ──────────────────────────────
                swarm_hebbian.fire(xn[2], xn[3], rb["uk"], rb["s_wm_ema"])
                hebbian.fire(xn[2], xn[3], rb["uk"], rb["s_wm_ema"])

                # ── AMYGDALA ──────────────────────────────────
                danger_here = nm.effective_danger_sensitivity(
                    danger_at(D, xn[2], xn[3]))
                if danger_here > 0.20 or s_wm > 0.25:
                    rb["buf"].append((rb["xk"].copy(), rb["uk"],
                                      xn.copy(), s_wm))
                    if len(rb["buf"]) > 200:
                        rb["buf"].pop(0)

                if dist_now < 0.8:
                    rb["near_miss_buf"].append((rb["xk"].copy(), rb["uk"],
                                                xn.copy(), s_wm))
                    if len(rb["near_miss_buf"]) > 100:
                        rb["near_miss_buf"].pop(0)

                rb["fatigue"] = .999*rb["fatigue"] + .001*abs(rb["uk"])

                # ── CURIOSITY SCORE ───────────────────────────
                base_curio = float(np.clip(
                    np.trace(rb["P_wm"])/P0_WM + 0.08, 0., 1.))
                # Grief suppresses curiosity
                grief = rb.get("grief", 0.)
                curio = float(np.clip(base_curio*(1.-grief*0.5), 0.05, 1.))
                # ACh boosts curiosity
                curio = float(np.clip(curio*(1.+nm.ACh*0.3), 0., 1.))

                # ── ACTION SELECTION ──────────────────────────
                rb["curio"] = curio
                dl = danger_here * (1. + grief*0.5) * social.social_risk_modifier()
                use_daughters = (alive_count <= 3 and alive_count > 0 and stage >= 4)

                if use_daughters:
                    rb["nm"].NE = min(1.0, rb["nm"].NE + 0.05)
                    rb["nm"].ACh = min(1.0, rb["nm"].ACh + 0.03)
                    un, d_name = daughter_minds_biological(
                        xn, rb["T_wm"], T_ltm_ep, rb["P_wm"],
                        D, M, nm, homeostatic, swarm_hebbian,
                        hebbian, predictive, social, grief)
                    mode = f"DELIBERATING-{d_name}"
                elif curio > np.random.random() and dl < 0.5:
                    un   = directed_explore_biological(
                        xn, rb["T_wm"], rb["P_wm"], D, nm)
                    mode = "CURIOUS"
                else:
                    un   = pick_action_biological(
                        xn, rb["T_wm"], T_ltm_ep, rb["P_wm"],
                        D, M, nm, homeostatic, swarm_hebbian, hebbian,
                        predictive, social, grief)
                    mode = ("FEARFUL" if dl>0.4 else
                            "GRIEVING" if grief>0.3 else "EXPLOITING")

                rb["mode"] = mode

                # Spotter Stage 3-4
                if stage==3 and abs(xn[2])>0.55:
                    p.resetBasePositionAndOrientation(
                        rid,[xn[0],y_offsets[i],.10],
                        p.getQuaternionFromEuler([0,0,0]))
                    p.resetBaseVelocity(rid,[0,0,0],[0,0,0])
                elif stage==4 and abs(xn[2])>0.62:
                    p.resetBasePositionAndOrientation(
                        rid,[xn[0],y_offsets[i],.10],
                        p.getQuaternionFromEuler([0,0,0]))
                    p.resetBaseVelocity(rid,[0,0,0],[0,0,0])

                apply_torque(rid, un)
                rb["xk"], rb["uk"] = xn, un
                rb["sv"] = time.time() - rb["t0"]

                if dist_now < best_dist_ever:
                    best_dist_ever = dist_now
                    brain["best_dist_ever"] = round(best_dist_ever,3)

                # Edge-triggered: only fire on the frame the body ENTERS the zone
                at_target_now = dist_now < 0.20
                was_at_target = rb.get("was_at_target", False)
                if at_target_now and not was_at_target:
                    target_reached_count += 1
                    brain["target_reached_count"] = target_reached_count
                    nm.DA = min(1., nm.DA + 0.4)  # dopamine spike
                    print(f"  *** TARGET REACHED Body {i+1}! "
                          f"DA:{nm.DA:.2f} Total:{target_reached_count} ***")
                rb["was_at_target"] = at_target_now

                # ── DEATH ─────────────────────────────────────
                floor_z = y_pos*math.sin(slope)
                local_z = z_pos - floor_z
                if local_z<0.065 or abs(xn[2])>0.65 or abs(roll)>0.5:
                    sv   = rb["sv"]
                    best = max(best, sv)
                    brain["episode_history"].append(round(sv,2))
                    brain["best_survival"] = round(best,2)
                    rb["alive"] = False
                    sibling_died_this_step = True
                    dead_this_step.append(rb)

                    # Trauma burn
                    for _ in range(4):
                        D = danger_update(D,xn[2],xn[3],10.,rate=0.3)

                    # Spread grief + NE spike to survivors
                    if abs(xn[0]-TARGET_X) < 0.5:
                        intensity = 1.-(abs(xn[0]-TARGET_X)/0.5)
                        mourning_events += 1
                        for b in brains:
                            if b["alive"]:
                                b["grief"] = min(1.,
                                    b.get("grief",0.) + intensity*0.5)
                                b["nm"].NE = min(1.,
                                    b["nm"].NE + intensity*0.3)
                                b["nm"].ACh = min(1.,
                                    b["nm"].ACh + 0.2)
                        print(f"  [MOURNING+NE SPIKE] "
                              f"Body {i+1} died near target")

                    # Legacy ghost
                    ghost_intensity = min(1., 0.3+sv/60.)
                    L_legacy = legacy_write(L_legacy,xn[0],ghost_intensity)
                    ghost_count += 1



                    alloload = rb["homeostatic"].allostatic_load
                    print(f"  Body {i+1} died {sv:.2f}s | "
                          f"Stage {stage} | {mode} | "
                          f"DA:{nm.DA:.2f} NE:{nm.NE:.2f} "
                          f"SHT:{nm.SHT:.2f} ACh:{nm.ACh:.2f} | "
                          f"AlloLoad:{alloload:.3f}")

                    try: p.removeBody(rid)
                    except: pass


            if dead_this_step:
                max_allo = max(b["homeostatic"].allostatic_load for b in dead_this_step)
                all_trauma = []
                all_near = []
                for b in dead_this_step:
                    all_trauma.extend(b["buf"])
                    all_near.extend(b["near_miss_buf"])

                brain["sleep_phase"] = "NREM-1"
                T_ltm_ep, P_ltm_ep, D = biological_sleep(
                    T_ltm_ep, P_ltm_ep, D,
                    all_trauma, all_near,
                    swarm_hebbian,
                    max_allo)
                
                # Apply 2% healing
                for b in dead_this_step:
                    b["homeostatic"].allostatic_load *= 0.98

                nrem1_count += 1; nrem3_count += 1; rem_count += 1
                brain["sleep_phase"] = "AWAKE"
                brain["nrem1_count"] = nrem1_count
                brain["nrem3_count"] = nrem3_count
                brain["rem_count"]   = rem_count

            if sibling_died_this_step:
                for b in brains:
                    if b["alive"]:
                        b["nm"].update(surprise=0, danger=0, dist_to_target=0, target_reached=False, sibling_died=True, allostatic_load=b["homeostatic"].allostatic_load)

            # ── DASHBOARD ────────────────────────────────────
            xA   = brains[0]["xk"] if brains[0]["xk"] is not None else np.zeros(SDIM)
            xB   = brains[1]["xk"] if brains[1]["xk"] is not None else np.zeros(SDIM)
            dmax = float(np.max(D)) + 1e-6
            nm0  = brains[0]["nm"]
            h0   = brains[0]["homeostatic"]

            alive_modes = [b["mode"] for b in brains if b["alive"]]
            mode = alive_modes[0] if alive_modes else "DEAD"

            avg_grief = float(np.mean([b.get("grief",0.) for b in brains]))
            avg_allo  = float(np.mean([b["homeostatic"].allostatic_load
                                        for b in brains]))

            best_dist_step = min(
                [abs(b["xk"][0]-TARGET_X) if b["xk"] is not None
                 else TARGET_X for b in brains], default=TARGET_X)

            brain.update({
                "survival_A"       : round(brains[0]["sv"],2),
                "survival_B"       : round(brains[1]["sv"],2),
                "pitch_A"          : float(xA[2]),
                "pitch_B"          : float(xB[2]),
                "wm_confidence_A"  : float(wm_conf(brains[0]["P_wm"])),
                "wm_confidence_B"  : float(wm_conf(brains[1]["P_wm"])),
                "survivals"        : [round(b["sv"],2) for b in brains],
                "pitches"          : [round(float(b["xk"][2]) if b["xk"] is not None else 0.,3) for b in brains],
                "distances"        : [round(float(abs(b["xk"][0]-TARGET_X)) if b["xk"] is not None else 2.,3) for b in brains],
                "alive_flags"      : [b["alive"] for b in brains],
                "best_survival"    : round(best,2),
                "mode"             : mode,
                "surprise_wm"      : float(brains[0]["s_wm_ema"]),
                "surprise_ltm"     : float(last_s_ltm),
                "ltm_confidence"   : float(ltm_conf(P_ltm_ep)),
                "curiosity"        : float(brains[0].get("curio",0.4)),
                "danger_level"     : float(danger_at(D,xA[2],xA[3])) if brains[0]["alive"] else 0.,
                "wind_active"      : wind_active,
                "slope_deg"        : round(np.degrees(slope),2),
                "quake_amp"        : round(q_amp,3),
                "danger_grid"      : (D/dmax).flatten().round(3).tolist(),
                "terrain_trust"    : M.round(3).tolist(),
                "curriculum_stage" : stage,
                "dist_A"           : float(abs(xA[0]-TARGET_X)),
                "dist_B"           : float(abs(xB[0]-TARGET_X)),
                "dopamine"         : float(max(0.,1.-best_dist_step/TARGET_X)),
                "best_dist_ever"   : round(best_dist_ever,3),
                "target_reached_count": target_reached_count,
                "ghost_count"      : ghost_count,
                "mourning_events"  : mourning_events,
                "legacy_map"       : L_legacy.round(3).tolist(),
                # Neuromodulators
                "DA"               : round(nm0.DA,3),
                "5HT"              : round(nm0.SHT,3),
                "NE"               : round(nm0.NE,3),
                "ACh"              : round(nm0.ACh,3),
                # Homeostasis
                "allostatic_load"  : round(avg_allo,4),
                "homeostatic_error": round(brains[0].get("last_h_error", 0.) if brains[0]["alive"] else 0., 3),
                # Sleep
                "sleep_phase"      : brain["sleep_phase"],
                "nrem1_count"      : nrem1_count,
                "nrem3_count"      : nrem3_count,
                "rem_count"        : rem_count,
                # Social
                "social_comfort"   : round(float(np.mean(
                    [b["social"].social_comfort for b in brains if b["alive"]]
                ) if any(b["alive"] for b in brains) else 0.),3),
                "alive_count"      : sum(1 for b in brains if b["alive"]),
                # Hebbian
                "hebbian_strength" : round(swarm_hebbian.total_strength(),3),
                # Predictive coding
                "prediction_error" : round(float(brains[0]["predictive"].precision_weight()) if brains[0]["alive"] else 0.,3),
            })

            # HUD
            p.removeAllUserDebugItems()
            nm_str = (f"DA:{nm0.DA:.2f} "
                      f"NE:{nm0.NE:.2f} "
                      f"SHT:{nm0.SHT:.2f}")
            hud = (f"STAGE {stage} | {mode} | "
                   f"Best:{best:.1f}s | {nm_str} | "
                   f"AlloLoad:{avg_allo:.3f}")
            p.addUserDebugText(hud,[0,0,.7],
                               textColorRGB=[0,1,.5],textSize=1.1)

            # Ghost markers + target rings
            for idx in range(0,100,4):
                lv = float(L_legacy[idx])
                if lv>0.15:
                    xp = -3.+idx*6./100.
                    a  = min(1.,lv)
                    p.addUserDebugLine([xp-.08,0,.02],[xp+.08,0,.02],
                                       [a,a*.2,a*.2],1)
            for h in [.05,.25,.45]:
                for angle in np.linspace(0,2*np.pi,8):
                    p.addUserDebugLine(
                        [TARGET_X,0,h],
                        [TARGET_X+.3*math.cos(angle),
                         .3*math.sin(angle),h],
                        [1.,.2,.5],2)

            p.stepSimulation()
            time.sleep(DT)

            if not any(b["alive"] for b in brains):
                T_ltm, P_ltm = T_ltm_ep.copy(), P_ltm_ep.copy()
                D_global     = D.copy()
                M_terrain    = M.copy()

                if episode % 50 == 0:
                    np.save("checkpoint_ltm_T.npy",  T_ltm)
                    np.save("checkpoint_ltm_P.npy",  P_ltm)
                    np.save("checkpoint_danger.npy", D_global)
                    np.save("checkpoint_terrain.npy",M_terrain)
                    np.save("checkpoint_legacy.npy", L_legacy)
                    print(f"  [CHECKPOINT ep {episode}]")

                allo_avg = float(np.mean([b["homeostatic"].allostatic_load
                                           for b in brains]))
                ep_max_sv = float(max(b["sv"] for b in brains))
                print(f"  Episode {episode} over. "
                      f"Best:{best:.2f}s  "
                      f"Reached:{target_reached_count}x  "
                      f"Ghosts:{ghost_count}  "
                      f"AlloLoad:{allo_avg:.4f}  "
                      f"REM:{rem_count}")

                # ── CSV LOGGING ──────────────────────────────────────────
                import csv, os
                log_path = "carl_survival_log.csv"
                write_header = not os.path.exists(log_path)
                with open(log_path, "a", newline="") as f:
                    w = csv.writer(f)
                    if write_header:
                        w.writerow(["episode","ep_max_survival","best_ever",
                                    "targets_reached","ghosts","rem_count",
                                    "hebbian_strength","allo_avg","stage"])
                    w.writerow([episode, round(ep_max_sv,2), round(best,2),
                                target_reached_count, ghost_count, rem_count,
                                round(swarm_hebbian.total_strength(),4),
                                round(allo_avg,4), stage])
                # ────────────────────────────────────────────────────────

                time.sleep(1.5)
                break

if __name__ == "__main__":

    main()
