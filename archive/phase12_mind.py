# -*- coding: utf-8 -*-
# CARL Phase 12: THE FEELING MIND
# ============================================================
# Built on top of Phase 11 (honest earned curriculum).
#
# Four new modules — each one answers: "what would I want
# if I were designing my own mind?"
#
# MODULE 1 — MOURNING (Grief that fades)
#   When a sibling dies near the target, surviving bodies
#   don't just update the danger map — they enter a grief
#   state. Their curiosity drops. Their caution rises.
#   Grief decays exponentially over time, like real grief.
#   Viewers see the swarm slow down after a loss.
#
# MODULE 2 — BRANCHING IMAGINATION (Prefrontal cortex)
#   pick_action now runs a 2-level tree search.
#   Level 1: 9 actions × 20 steps.
#   Level 2: at step 20, branch into 9 sub-actions × 20 more.
#   81 futures considered per decision. CARL foresees crashes
#   3 decisions away, not just 1.
#
# MODULE 3 — INTENTIONAL DREAMS (Near-miss replay)
#   Sleep replay used to pick the 50 most surprising moments.
#   Now it picks from TWO pools:
#     - Top 25 most traumatic (highest surprise = near death)
#     - Top 25 nearest misses (closest to target before death)
#   Learning from almost-winning, not just from suffering.
#
# MODULE 4 — LEGACY MARKERS (Persistent spatial memory)
#   When a body dies, it writes a "ghost" into a persistent
#   legacy map — a spatial record that says "a mind died here."
#   This map never resets. It accumulates across all episodes.
#   Future bodies inherit it as prior knowledge — instinct
#   shaped by ancestral death, not just personal experience.
# ============================================================

import sys, pybullet as p, pybullet_data, time, numpy as np
import asyncio, websockets, json, threading, math
import time as _time

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

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
    "episode_history": [], "danger_grid": [0.0] * 400,
    "terrain_trust": [1.0] * 100,
    "wind_active": False, "slope_deg": 0.0, "quake_amp": 0.0,
    "neck_A": 0.0, "neck_B": 0.0,
    "survivals": [], "pitches": [], "distances": [],
    "alive_flags": [], "dist_A": 2.0, "dist_B": 2.0,
    "dopamine": 0.0, "curriculum_stage": 1,
    "daughter_active": False,
    "best_dist_ever": 2.0,
    "target_reached_count": 0,
    # Phase 12 NEW telemetry
    "mourning_level"    : 0.0,   # 0=normal, 1=deep grief
    "mourning_events"   : 0,     # total times mourning triggered
    "imagination_depth" : 1,     # 1=shallow, 2=branching
    "near_miss_replays" : 0,     # times near-miss replay used
    "legacy_map"        : [0.0]*100, # persistent death location map
    "ghost_count"       : 0,     # total ghosts written to legacy
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

def curio_score(P_wm, boost=0.0):
    base = float(np.clip(np.trace(P_wm)/P0_WM, 0., 1.))
    return float(np.clip(base + boost + 0.08, 0., 1.))

# ── DANGER MAP ────────────────────────────────────────────────
def fresh_danger(): return np.zeros((DRES, DRES))

def _cell(pitch, vel):
    i = int(np.clip((pitch+.65)/1.30*DRES, 0, DRES-1))
    j = int(np.clip((vel+3.0)/6.0*DRES,   0, DRES-1))
    return i, j

def danger_update(D, pitch, vel, surprise, rate=0.15):
    i, j = _cell(pitch, vel)
    D[i,j] = (1-rate)*D[i,j] + rate*surprise
    return D

def danger_at(D, pitch, vel):
    return float(D[_cell(pitch, vel)])

# ── TERRAIN MEMORY ────────────────────────────────────────────
def fresh_terrain(): return np.ones(100)

def terrain_update(M, x_pos, surprise, rate=0.06):
    i = int(np.clip((x_pos+3.)/6.*100, 0, 99))
    M[i] = (1-rate)*M[i] + rate*max(0., 1.-surprise*4.)
    return M

def terrain_trust_at(M, x_pos):
    i = int(np.clip((x_pos+3.)/6.*100, 0, 99))
    return float(M[i])

# ── MODULE 4: LEGACY MAP ──────────────────────────────────────
# A persistent spatial record of where minds have died.
# Never resets. Accumulates across all episodes and all bodies.
# Future bodies inherit ancestral death knowledge as instinct.

def fresh_legacy(): return np.zeros(100)

def legacy_write(L, x_pos, intensity=1.0):
    """Write a ghost at the position where a body died."""
    # Don't write ghosts AT the target — dying there is brave, not traumatic
    if abs(x_pos - TARGET_X) < 0.25:
        intensity *= 0.2
    i = int(np.clip((x_pos+3.)/6.*100, 0, 99))
    # Spread ghost across 3 cells — death leaves a wide mark
    for di in [-1, 0, 1]:
        ii = np.clip(i+di, 0, 99)
        L[ii] = min(1.0, L[ii] + intensity * (0.5 if di != 0 else 1.0))
    return L

def legacy_at(L, x_pos):
    i = int(np.clip((x_pos+3.)/6.*100, 0, 99))
    return float(L[i])

def legacy_to_terrain_prior(L):
    """
    Convert legacy map to terrain trust prior.
    High legacy (many deaths) = low trust = inherited caution.
    Used at episode start to seed terrain map with ancestral knowledge.
    """
    return np.clip(1.0 - L * 0.2, 0.5, 1.0)

# ── MODULE 5: PHANTOM PAINS (Proprioceptive Trauma) ───────────
def phantom_pain_cost(xs, u, phantom_pains, living_siblings=10):
    """Flinch away from actions that previously caused death in similar states."""
    cost = 0.0
    for (px, pu, intensity) in phantom_pains:
        d_state = float(np.linalg.norm(xs[:4] - px[:4]))
        d_act = abs(u - pu)
        if d_state < 0.5 and d_act < 2.0:
            cost += 1.5 * intensity * math.exp(-d_state - d_act)
    comfort_modifier = 0.5 + 0.5 * (1.0 - (living_siblings / 10.0))
    # Trauma Ceiling: The mind can only feel so much pain at once before it numbs out.
    return min(10.0, cost * comfort_modifier)

# ── MODULE 1: MOURNING ────────────────────────────────────────
# Grief that fades exponentially.
# When a sibling dies near the target, all surviving bodies
# enter a grief state. Their effective curiosity drops.
# Their danger sensitivity rises. Grief decays each step.

MOURNING_DECAY   = 0.9998   # how fast grief fades (per sim step)
MOURNING_TRIGGER = 0.5      # how close to target triggers deep grief

def apply_mourning_to_swarm(brains, dead_x, target_x, mourning_events):
    """Spread grief to all living bodies after a sibling dies near target."""
    dist_from_target = abs(dead_x - target_x)
    if dist_from_target < MOURNING_TRIGGER:
        grief_intensity = 1.0 - (dist_from_target / MOURNING_TRIGGER)
        for b in brains:
            if b["alive"]:
                b["grief"] = min(1.0, b.get("grief", 0.) + grief_intensity * 0.6)
        mourning_events += 1
        print(f"  [MOURNING] Sibling died {dist_from_target:.2f}m from target "
              f"— grief:{grief_intensity:.2f} spread to {sum(1 for b in brains if b['alive'])} survivors")
    return mourning_events

def decay_grief(brains):
    """Called every step — grief fades naturally over time."""
    for b in brains:
        if b["alive"] and b.get("grief", 0.) > 0.:
            b["grief"] *= MOURNING_DECAY

def grief_modified_curio(base_curio, grief_level):
    """
    Grief suppresses curiosity (risk-taking drops after loss)
    and amplifies danger sensitivity.
    At full grief (1.0): curiosity halved, danger felt 1.5x stronger.
    At zero grief: no effect.
    """
    curio_factor   = 1.0 - grief_level * 0.5
    return float(np.clip(base_curio * curio_factor, 0.05, 1.0))

def grief_modified_danger(base_danger, grief_level):
    """Grief makes survivors more sensitive to danger signals."""
    return float(base_danger * (1.0 + grief_level * 0.5))

# ── MODULE 3: INTENTIONAL DREAMS ─────────────────────────────
# Near-miss replay — learning from almost-winning, not just suffering.
# Two pools: trauma (highest surprise) + near-misses (closest to target).
# Each pool contributes 25 memories to the 50-memory replay budget.

def intentional_sleep_replay(D, buf, near_miss_buf):
    """
    Replay from two pools:
    - buf: high-surprise trauma memories
    - near_miss_buf: moments where x_pos was closest to TARGET_X
    Equal weight. Both shape the danger map.
    """
    replayed = 0

    # Pool 1: top 25 trauma (highest surprise)
    if buf:
        top_trauma = sorted(buf, key=lambda m: m[3], reverse=True)[:25]
        for (xk_m, uk_m, xn_m, surprise) in top_trauma:
            for _ in range(2):
                D = danger_update(D, xn_m[2], xn_m[3], surprise, rate=0.2)
        replayed += len(top_trauma)

    # Pool 2: top 25 near-misses (closest to target)
    if near_miss_buf:
        top_near = sorted(near_miss_buf,
                          key=lambda m: abs(m[0][0] - TARGET_X))[:25]
        for (xk_m, uk_m, xn_m, surprise) in top_near:
            for _ in range(2):
                # Near misses: reinforce the approach path as SAFE
                # (they were close to target = good terrain)
                i = int(np.clip((xk_m[0]+3.)/6.*100, 0, 99))
                # Don't update danger for near-miss — update terrain trust
                # (handled outside, returned as signal)
                D = danger_update(D, xn_m[2], xn_m[3],
                                  max(0., surprise * 0.3), rate=0.1)
        replayed += len(top_near)

    return D, replayed

# ── MODULE 2: BRANCHING IMAGINATION ──────────────────────────
# Two-level tree search. 81 futures per decision.
# Level 1: choose action, roll 20 steps.
# Level 2: at step 20, branch into all 9 sub-actions, roll 20 more.
# Falls back to shallow if computation takes >40ms.

def pick_action_branching(x, T_wm, T_ltm, P_wm, D, M, L_phantom_pains, living_siblings=10, ambition=1.0):
    """Phase 12: Branching imagination — 81 futures considered."""
    alpha  = wm_conf(P_wm)
    T_use  = alpha * T_wm + (1.0 - alpha) * T_ltm
    Ad, Bd = T_use[:SDIM,:].T, T_use[SDIM,:]

    best_u, best_F = 0., float('inf')

    for u in ACTIONS:
        xs, F = x.copy(), 0.

        # Level 1: 20 steps straight
        for h in range(20):
            xs_n      = Ad @ xs + Bd * u
            conf      = max(alpha, 0.1) * (0.95**h)
            pred_s    = float(np.linalg.norm(xs_n - xs)) * 0.15
            d_cost    = danger_at(D, xs_n[2], xs_n[3])
            t_risk    = 1.0 - terrain_trust_at(M, xs_n[0])
            dist_cost = (0.6 * ambition) * abs(xs_n[0] - TARGET_X)
            pain      = phantom_pain_cost(xs_n, u, L_phantom_pains, living_siblings) if L_phantom_pains else 0.0
            F        += (pred_s + d_cost + 0.3*t_risk + dist_cost + pain) * conf
            xs        = xs_n

        # Level 2: branch at step 20
        branch_best = float('inf')
        for u2 in ACTIONS:
            xs2, F2 = xs.copy(), 0.
            for h2 in range(20):
                xs2_n  = Ad @ xs2 + Bd * u2
                conf2  = max(alpha, 0.1) * (0.95**(20+h2))
                d2     = danger_at(D, xs2_n[2], xs2_n[3])
                dist2  = (0.6 * ambition) * abs(xs2_n[0] - TARGET_X)
                pain2  = phantom_pain_cost(xs2_n, u2, L_phantom_pains, living_siblings) if L_phantom_pains else 0.0
                F2    += (d2 + dist2 + pain2) * conf2
                xs2    = xs2_n
            if F2 < branch_best:
                branch_best = F2
        F += branch_best

        if F < best_F:
            best_F, best_u = F, u

    return best_u

def pick_action_shallow(x, T_wm, T_ltm, P_wm, D, M, L_phantom_pains, living_siblings=10, ambition=1.0):
    """Original shallow planner — fallback for early stages or timeout."""
    alpha  = wm_conf(P_wm)
    T_use  = alpha * T_wm + (1.0 - alpha) * T_ltm
    Ad, Bd = T_use[:SDIM,:].T, T_use[SDIM,:]
    best_u, best_F = 0., float('inf')
    for u in ACTIONS:
        xs, F = x.copy(), 0.
        for h in range(HORIZON):
            xs_n      = Ad @ xs + Bd * u
            conf      = max(alpha, 0.1) * (0.95**h)
            pred_s    = float(np.linalg.norm(xs_n - xs)) * 0.15
            d_cost    = danger_at(D, xs_n[2], xs_n[3])
            t_risk    = 1.0 - terrain_trust_at(M, xs_n[0])
            dist_cost = (0.6 * ambition) * abs(xs_n[0] - TARGET_X)
            pain      = phantom_pain_cost(xs_n, u, L_phantom_pains, living_siblings) if L_phantom_pains else 0.0
            F        += (pred_s + d_cost + 0.3*t_risk + dist_cost + pain) * conf
            xs        = xs_n
        if F < best_F:
            best_F, best_u = F, u
    return best_u

def smart_pick_action(x, T_wm, T_ltm, P_wm, D, M, stage, L_phantom_pains, living_siblings=10, ambition=1.0):
    """
    Choose imagination depth based on stage and timing.
    Stage 1-3: always shallow (danger map too sparse for branching to help)
    Stage 4-5: try branching, fall back if >40ms
    Returns: (action, depth_used)
    """
    if stage < 4:
        return pick_action_shallow(x, T_wm, T_ltm, P_wm, D, M, L_phantom_pains, living_siblings, ambition), 1

    t0 = _time.perf_counter()
    u  = pick_action_branching(x, T_wm, T_ltm, P_wm, D, M, L_phantom_pains, living_siblings, ambition)
    dt = _time.perf_counter() - t0

    if dt > 0.040:
        # Too slow — fall back, log warning
        u = pick_action_shallow(x, T_wm, T_ltm, P_wm, D, M, L_phantom_pains, living_siblings, ambition)
        return u, 1
    return u, 2

# ── DAUGHTER MINDS ────────────────────────────────────────────
def directed_explore(x, T_wm, P_wm, D):
    Ad, Bd = T_wm[:SDIM,:].T, T_wm[SDIM,:]
    if float(np.linalg.norm(Bd)) < 0.1:
        return float(np.random.choice(ACTIONS))
    best_u, best_info = 0., -float('inf')
    for u in ACTIONS:
        xs = x.copy()
        for _ in range(10):
            xs = Ad @ xs + Bd * u
        Phi_r = np.append(xs, u).reshape(SDIM+1, 1)
        unc   = float((Phi_r.T @ P_wm @ Phi_r).squeeze())
        info  = unc * (1.0 + danger_at(D, xs[2], xs[3]))
        if info > best_info:
            best_info, best_u = info, u
    return best_u

def daughter_minds(x, T_wm, T_ltm, P_wm, D, M, grief, L_phantom_pains, living_siblings=10, ambition=1.0):
    """
    Three daughter strategies.
    Grief modifies the weights — grieving bodies favour SAFE daughter.
    """
    alpha  = wm_conf(P_wm)
    T_use  = alpha * T_wm + (1.0 - alpha) * T_ltm
    Ad, Bd = T_use[:SDIM,:].T, T_use[SDIM,:]

    # Grief shifts weight toward safety
    safe_d_weight = 2.0 + grief * 1.5
    bold_d_weight = max(0.3, 0.5 - grief * 0.3)

    strategies = [
        {"d_w": safe_d_weight, "dop_w": 0.0,  "name": "SAFE"},
        {"d_w": bold_d_weight, "dop_w": 0.6,  "name": "BOLD"},
        {"d_w": 1.0,           "dop_w": 0.3,  "name": "BALANCED"},
    ]

    best_u, best_F, best_name = 0., float('inf'), "BALANCED"
    for strat in strategies:
        for u in ACTIONS:
            xs, F = x.copy(), 0.
            for h in range(HORIZON):
                xs_n      = Ad @ xs + Bd * u
                conf      = max(alpha, 0.1) * (0.95**h)
                pred_s    = float(np.linalg.norm(xs_n - xs)) * 0.15
                d_cost    = danger_at(D, xs_n[2], xs_n[3]) * strat["d_w"]
                t_risk    = 1.0 - terrain_trust_at(M, xs_n[0])
                dist_cost = (strat["dop_w"] * ambition) * abs(xs_n[0] - TARGET_X)
                pain      = phantom_pain_cost(xs_n, u, L_phantom_pains, living_siblings) if L_phantom_pains else 0.0
                F        += (pred_s + d_cost + 0.3*t_risk + dist_cost + pain) * conf
                xs        = xs_n
            if F < best_F:
                best_F, best_u, best_name = F, u, strat["name"]

    return best_u, best_name

# ── PYBULLET HELPERS ─────────────────────────────────────────
def get_state(rid):
    pos, quat    = p.getBasePositionAndOrientation(rid)
    vel, ang_vel = p.getBaseVelocity(rid)
    euler        = p.getEulerFromQuaternion(quat)
    neck         = p.getJointState(rid, 2)[0]
    state = np.array([pos[0], vel[0], euler[1], ang_vel[1], neck], dtype=float)
    return state, float(pos[1]), float(pos[2]), float(euler[0])

def apply_torque(rid, u, adrenaline=0.0):
    limit = 8.0 + 4.0 * adrenaline
    t = float(np.clip(u * (1.0 + 0.5 * adrenaline), -limit, limit))
    p.setJointMotorControl2(rid, 0, p.TORQUE_CONTROL, force=t)
    p.setJointMotorControl2(rid, 1, p.TORQUE_CONTROL, force=t)

def spawn_robot(y_offset=0.0, initial_pitch=0.0, slope=0.0):
    for attempt in range(3):
        try:
            z_spawn = 0.08 + abs(y_offset) * abs(math.sin(slope))
            rid = p.loadURDF("carl.urdf", [0, y_offset, z_spawn],
                             p.getQuaternionFromEuler([slope, initial_pitch, 0]))
            break
        except Exception:
            time.sleep(0.5)
    else:
        raise RuntimeError("Failed to load URDF after 3 attempts")
    p.setJointMotorControl2(rid, 2, p.POSITION_CONTROL, targetPosition=0, force=12.)
    p.setJointMotorControl2(rid, 0, p.VELOCITY_CONTROL, force=0)
    p.setJointMotorControl2(rid, 1, p.VELOCITY_CONTROL, force=0)
    return rid

def spawn_robot_brain(T_instinct, P_instinct):
    P_wm_start = P_instinct.copy() + 50.0 * np.eye(SDIM+1)
    return {
        "T_wm": T_instinct.copy(), "P_wm": P_wm_start,
        "xk": None, "uk": 0.,
        "s_wm_ema": 0., "s_slow": 0.01,
        "danger_ema": 0., "fatigue": 0.,
        "buf": [],           # trauma buffer
        "near_miss_buf": [], # near-miss buffer (Phase 12)
        "t0": time.time(), "alive": True, "sv": 0.,
        "rid": None,
        "grief": 0.0,        # current grief level (Phase 12)
        "adrenaline": 0.0,   # acute stress response
        "pitch_baseline": 0.0, # sensory habituation
        "boredom": 0.0,      # courage/defiance accumulator
    }

# ── MAIN ──────────────────────────────────────────────────────
def main():
    global brain

    p.connect(p.GUI, options="--width=1280 --height=720")
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    T_ltm, P_ltm = fresh_ltm()
    D_global     = fresh_danger()
    M_terrain    = fresh_terrain()
    L_legacy     = fresh_legacy()   # Phase 12: never resets
    L_phantom_pains = []            # Phase 12 Extras: proprioceptive trauma
    episode, best = 0, 0.
    best_dist_ever       = TARGET_X
    target_reached_count = 0
    mourning_events      = 0
    near_miss_replays    = 0
    ghost_count          = 0
    imagination_depth    = 1

    print(f"\n== CARL Phase 12: THE FEELING MIND ==")
    print(f"Mourning | Branching Imagination | Intentional Dreams | Legacy\n")

    while True:
        episode += 1
        brain["episode"] = episode

        # ── EARNED CURRICULUM ────────────────────────────────
        stage = 1
        if best > 10.0:  stage = 2
        if best > 20.0:  stage = 3
        if best > 35.0:  stage = 4
        if best > 55.0:  stage = 5
        brain["curriculum_stage"] = stage

        try:
            p.resetSimulation()
            p.setGravity(0, 0, -9.81)
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
            p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)

            plane = p.loadURDF("plane.urdf")
            p.changeVisualShape(plane, -1, rgbaColor=[.02,.02,.04,1])

            # TRON grid
            for y_line in np.linspace(-5, 5, 21):
                p.addUserDebugLine([-2,y_line,.01],[3,y_line,.01],[.05,.05,.25],1)
            for x_line in np.linspace(-2, 3, 11):
                p.addUserDebugLine([x_line,-5,.01],[x_line,5,.01],[.05,.05,.25],1)

            # Legacy ghost markers — draw faint crosses where minds have died
            for idx in range(100):
                lv = float(L_legacy[idx])
                if lv > 0.1:
                    x_pos = -3.0 + idx * 6.0 / 100.0
                    alpha = min(1.0, lv)
                    p.addUserDebugLine(
                        [x_pos-0.1, 0, 0.02], [x_pos+0.1, 0, 0.02],
                        [alpha, alpha*0.3, alpha*0.3], 1)
                    p.addUserDebugLine(
                        [x_pos, -0.1, 0.02], [x_pos, 0.1, 0.02],
                        [alpha, alpha*0.3, alpha*0.3], 1)

            p.resetDebugVisualizerCamera(3.0, 60, -20, [1.0, 0, 0.2])

            slope = 0.0
            if stage >= 4:
                slope = float(np.random.uniform(-0.04, 0.04))
            p.resetBasePositionAndOrientation(
                plane, [0,0,0], p.getQuaternionFromEuler([slope,0,0]))

            init_pitch = 0.0
            if stage >= 2:
                init_pitch = float(np.random.uniform(-0.12, 0.12))

            # Glowing target
            tgt = p.createVisualShape(p.GEOM_SPHERE, radius=0.15,
                                      rgbaColor=[1,0.2,0.5,0.9])
            p.createMultiBody(baseMass=0, baseVisualShapeIndex=tgt,
                              basePosition=[TARGET_X,0,0.15])
            for h in [0.05, 0.25, 0.45]:
                for angle in np.linspace(0, 2*np.pi, 8):
                    p.addUserDebugLine(
                        [TARGET_X,0,h],
                        [TARGET_X+0.3*math.cos(angle), 0.3*math.sin(angle), h],
                        [1.0,0.2,0.5], 2)
            p.addUserDebugLine([0,0,0.05],[TARGET_X,0,0.05],[1.0,0.2,0.5],3)

            # Swarm spawn
            y_offsets = np.linspace(-4.0, 4.0, N_BODIES).tolist()
            bodies = [spawn_robot(y_offset=y, initial_pitch=init_pitch,
                                  slope=slope) for y in y_offsets]
            for i, rid in enumerate(bodies):
                r_c = 0.4 + 0.6*(i/max(1, N_BODIES-1))
                b_c = 1.0 - 0.4*(i/max(1, N_BODIES-1))
                p.changeVisualShape(rid, -1, rgbaColor=[r_c,0.2,b_c,1.0])
                p.changeVisualShape(rid,  3, rgbaColor=[r_c,0.2,b_c,1.0])

        except Exception as e:
            print(f"PyBullet crash: {e}")
            p.disconnect(); time.sleep(1.)
            p.connect(p.GUI, options="--width=1280 --height=720")
            continue

        for _ in range(15): p.stepSimulation()
        time.sleep(0.3)

        brains = [spawn_robot_brain(T_ltm, P_ltm) for _ in range(N_BODIES)]
        for b, rid in zip(brains, bodies):
            b["rid"] = rid
            b["xk"], _, _, _ = get_state(rid)

        D        = D_global.copy()
        # Seed terrain map with legacy prior knowledge
        M = legacy_to_terrain_prior(L_legacy)
        # Blend with learned terrain (legacy anchors, terrain refines)
        M = 0.3 * M + 0.7 * M_terrain

        P_ltm_ep = P_ltm.copy()
        T_ltm_ep = T_ltm.copy()
        next_wind = np.random.randint(15*240, 30*240)

        stage_labels = {
            1:"STAGE 1 — LEARNING TO STAND",
            2:"STAGE 2 — LEARNING TO RECOVER",
            3:"STAGE 3 — PUSHING THROUGH WIND",
            4:"STAGE 4 — COMBINATIONS",
            5:"STAGE 5 — FULL AUTONOMY",
        }
        lc = ltm_conf(P_ltm_ep)
        print(f"[Ep {episode:3d}] {stage_labels[stage]}  "
              f"LTM:{lc*100:.0f}%  Best:{best:.1f}s  "
              f"Ghosts:{ghost_count}")

        for step in range(500_000):
            # Grief decays every step for all bodies
            decay_grief(brains)

            # Time Heals All Wounds (Trauma Decay)
            new_pains = []
            for (px, pu, intensity) in L_phantom_pains:
                intensity *= 0.9995
                if intensity >= 0.1:
                    new_pains.append((px, pu, intensity))
            L_phantom_pains = new_pains
            
            # The environment also cools down: collective memory of danger fades slowly
            D *= 0.99995

            # Wind Stage 3+
            wind_active = False
            if stage >= 3 and step >= next_wind:
                fx = float(np.random.uniform(-3., 3.))
                for b in brains:
                    if b["alive"]:
                        p.applyExternalForce(b["rid"],-1,[fx,0,0],[0,0,.3],
                                             p.WORLD_FRAME)
                next_wind = step + np.random.randint(15*240, 30*240)
                wind_active = True

            # Earthquake Stage 4+
            q_amp = 0.0
            if stage >= 4:
                t_ep  = step * DT
                q_amp = float(np.clip(t_ep/90., 0., 0.3))
                q_f   = q_amp * float(np.sin(2*np.pi*1.5*t_ep))
                if q_amp > 0.01:
                    for b in brains:
                        if b["alive"]:
                            p.applyExternalForce(b["rid"],-1,[q_f,0,0],[0,0,0],
                                                 p.WORLD_FRAME)

            # ── STEP EACH ROBOT ───────────────────────────────
            for i, rb in enumerate(brains):
                if not rb["alive"]: continue
                rid = rb["rid"]

                xn, y_pos, z_pos, roll = get_state(rid)

                # 1. WM update
                rb["T_wm"], rb["P_wm"], s_wm = rls_update(
                    rb["T_wm"], rb["P_wm"], rb["xk"], rb["uk"], xn,
                    lam=0.990, p_floor=0.001)

                # 2. Shared LTM update
                T_ltm_ep, P_ltm_ep, _ = rls_update(
                    T_ltm_ep, P_ltm_ep, rb["xk"], rb["uk"], xn,
                    lam=0.99995, p_floor=0.002)

                rb["s_wm_ema"] = .1*s_wm  + .9*rb["s_wm_ema"]
                rb["s_slow"]   = .005*s_wm + .995*rb["s_slow"]
                nov_boost = float(np.clip(
                    (rb["s_wm_ema"]/(rb["s_slow"]+1e-6)-2.)*0.3, 0., .5))

                D = danger_update(D, xn[2], xn[3], rb["s_wm_ema"])
                M = terrain_update(M, xn[0], rb["s_wm_ema"])

                # 3. Amygdala filter
                if rb["danger_ema"] > 0.20 or s_wm > 0.25:
                    rb["buf"].append((rb["xk"].copy(), rb["uk"],
                                      xn.copy(), s_wm))
                    if len(rb["buf"]) > 200:
                        rb["buf"].pop(0)

                # Near-miss buffer — moments close to target
                dist_now = abs(xn[0] - TARGET_X)
                if dist_now < 0.8:
                    rb["near_miss_buf"].append((rb["xk"].copy(), rb["uk"],
                                                xn.copy(), s_wm))
                    if len(rb["near_miss_buf"]) > 100:
                        rb["near_miss_buf"].pop(0)

                # Sensory Habituation: decay pitch baseline towards current pitch
                rb["pitch_baseline"] = 0.998 * rb["pitch_baseline"] + 0.002 * xn[2]
                eff_pitch = xn[2] - rb["pitch_baseline"] * 0.7

                # Grief-modified signals (use eff_pitch for danger)
                grief      = rb.get("grief", 0.)
                base_curio = curio_score(rb["P_wm"], nov_boost)
                curio      = grief_modified_curio(base_curio, grief)
                dl         = grief_modified_danger(danger_at(D, eff_pitch, xn[3]), grief)

                # Adrenaline Trigger
                danger_spike = dl - rb["danger_ema"]
                if danger_spike > 0.5 and rb.get("adrenaline", 0.0) < 0.1:
                    rb["adrenaline"] = 1.0
                    print(f"  [ADRENALINE] Body {i+1} spiked (delta {danger_spike:.2f})!")
                rb["adrenaline"] = max(0.0, rb["adrenaline"] - 0.005)

                rb["danger_ema"] = .05*dl + .95*rb["danger_ema"]

                # Courage (Boredom) accumulator: builds up if they aren't making forward progress
                if rb["xk"] is not None and (xn[0] - rb["xk"][0]) <= 0.01:
                    rb["boredom"] = rb.get("boredom", 0.0) + 0.005
                else:
                    rb["boredom"] = max(0.0, rb.get("boredom", 0.0) - 0.05)

                # ── ACTION SELECTION ──────────────────────────
                conflict = curio > 0.5 and dl > 0.25
                d_active = False

                # Adaptive Ambition: Builds aggressively after 30 seconds
                sv = time.time() - rb["t0"]
                ambition = 1.0 + max(0.0, (sv - 30.0) / 10.0)
                living_sibs = sum(1 for b in brains if b["alive"])

                force_leap = rb["boredom"] > 1.0
                if force_leap:
                    un = directed_explore(xn, rb["T_wm"], rb["P_wm"], D)
                    rb["daughter_mode"] = "DEFIANT"
                    rb["boredom"] = 0.0
                elif conflict:
                    un, d_name = daughter_minds(
                        xn, rb["T_wm"], T_ltm_ep, rb["P_wm"], D, M, grief, L_phantom_pains, living_sibs, ambition)
                    rb["daughter_mode"] = d_name
                    d_active = True
                elif curio > np.random.random():
                    un = directed_explore(xn, rb["T_wm"], rb["P_wm"], D)
                    rb["daughter_mode"] = "CURIOUS"
                else:
                    un, img_depth = smart_pick_action(
                        xn, rb["T_wm"], T_ltm_ep, rb["P_wm"], D, M, stage, L_phantom_pains, living_sibs, ambition)
                    imagination_depth = img_depth
                    rb["daughter_mode"] = "FEARFUL" if dl > 0.3 else "EXPLOITING"

                # Spotter Stage 3-4 only
                if stage == 3 and abs(xn[2]) > 0.55:
                    p.resetBasePositionAndOrientation(
                        rid,[xn[0],y_offsets[i],.10],
                        p.getQuaternionFromEuler([0,0,0]))
                    p.resetBaseVelocity(rid,[0,0,0],[0,0,0])
                elif stage == 4 and abs(xn[2]) > 0.62:
                    p.resetBasePositionAndOrientation(
                        rid,[xn[0],y_offsets[i],.10],
                        p.getQuaternionFromEuler([0,0,0]))
                    p.resetBaseVelocity(rid,[0,0,0],[0,0,0])

                apply_torque(rid, un, rb["adrenaline"])
                rb["fatigue"] = .999*rb["fatigue"] + .001*abs(un)
                rb["xk"], rb["uk"] = xn, un
                rb["sv"] = time.time() - rb["t0"]

                # Track best distance
                if dist_now < best_dist_ever:
                    best_dist_ever = dist_now
                    brain["best_dist_ever"] = round(best_dist_ever, 3)

                # Target reached
                if dist_now < 0.20:
                    target_reached_count += 1
                    brain["target_reached_count"] = target_reached_count
                    print(f"  *** TARGET REACHED by Body {i+1}! "
                          f"Total:{target_reached_count} "
                          f"Grief:{rb['grief']:.2f} ***")

                # ── DEATH ────────────────────────────────────
                floor_z = y_pos * math.sin(slope)
                local_z = z_pos - floor_z

                if local_z < 0.065 or abs(xn[2]) > 0.65 or abs(roll) > 0.5:
                    sv   = rb["sv"]
                    best = max(best, sv)
                    brain["episode_history"].append(round(sv, 2))
                    brain["best_survival"] = round(best, 2)
                    rb["alive"] = False

                    # Trauma danger burn
                    for _ in range(4):
                        D = danger_update(D, xn[2], xn[3], 10., rate=0.3)

                    # MODULE 1: Mourning — spread grief to survivors
                    mourning_events = apply_mourning_to_swarm(
                        brains, xn[0], TARGET_X, mourning_events)

                    # MODULE 4: Legacy — write ghost at death location
                    ghost_intensity = min(1.0, 0.3 + sv/60.)
                    L_legacy = legacy_write(L_legacy, xn[0], ghost_intensity)
                    ghost_count += 1
                    brain["ghost_count"] = ghost_count
                    brain["legacy_map"]  = L_legacy.round(3).tolist()

                    # MODULE 5: Phantom Pain — log lethal state/action with intensity
                    L_phantom_pains.append((rb["xk"].copy(), rb["uk"], 1.0))
                    if len(L_phantom_pains) > 50:
                        L_phantom_pains.pop(0)

                    # MODULE 3: Intentional dreams
                    brain["sleeping"] = True
                    n_trauma   = min(len(rb["buf"]), 25)
                    n_nearmiss = min(len(rb["near_miss_buf"]), 25)
                    print(f"  Body {i+1} died {sv:.2f}s | "
                          f"Stage {stage} | Mode:{rb['daughter_mode']} | "
                          f"Grief:{rb['grief']:.2f} | "
                          f"Trauma:{n_trauma} NearMiss:{n_nearmiss}")
                    D, replayed = intentional_sleep_replay(
                        D, rb["buf"], rb["near_miss_buf"])
                    if n_nearmiss > 0:
                        near_miss_replays += 1
                        brain["near_miss_replays"] = near_miss_replays
                    brain["replay_count"] = replayed
                    brain["sleeping"]     = False

                    try: p.removeBody(rid)
                    except: pass

            # ── DASHBOARD ────────────────────────────────────
            xA   = brains[0]["xk"] if brains[0]["xk"] is not None else np.zeros(SDIM)
            xB   = brains[1]["xk"] if brains[1]["xk"] is not None else np.zeros(SDIM)
            dmax = float(np.max(D)) + 1e-6
            PhiA = np.append(xA, brains[0]["uk"]).reshape(SDIM+1,1)
            s_ltm = float(np.linalg.norm(xA - (T_ltm_ep.T @ PhiA).flatten()))
            dl = danger_at(D, xA[2], xA[3]) if brains[0]["alive"] else 0.

            alive_count = sum(1 for b in brains if b["alive"])
            alive_modes = [b["daughter_mode"] for b in brains if b["alive"]]
            mode = alive_modes[0] if alive_modes else "DEAD"
            if brain.get("sleeping"): mode = "SLEEPING"

            avg_grief = float(np.mean([b.get("grief",0.) for b in brains]))
            d_active  = any(b["daughter_mode"] in ("SAFE","BOLD","BALANCED")
                           for b in brains if b["alive"])

            best_dist_step = min(
                [abs(b["xk"][0]-TARGET_X) if b["xk"] is not None else TARGET_X
                 for b in brains], default=TARGET_X)

            brain.update({
                "survival_A"      : round(brains[0]["sv"],2),
                "survival_B"      : round(brains[1]["sv"],2),
                "pitch_A"         : float(xA[2]),
                "pitch_B"         : float(xB[2]),
                "neck_A"          : float(xA[4]),
                "neck_B"          : float(xB[4]),
                "fatigue_A"       : float(brains[0]["fatigue"]),
                "fatigue_B"       : float(brains[1]["fatigue"]),
                "wm_confidence_A" : float(wm_conf(brains[0]["P_wm"])),
                "wm_confidence_B" : float(wm_conf(brains[1]["P_wm"])),
                "survivals"       : [round(b["sv"],2) for b in brains],
                "pitches"         : [round(float(b["xk"][2]) if b["xk"] is not None else 0.,3) for b in brains],
                "distances"       : [round(float(abs(b["xk"][0]-TARGET_X)) if b["xk"] is not None else 2.,3) for b in brains],
                "alive_flags"     : [b["alive"] for b in brains],
                "best_survival"   : round(best,2),
                "mode"            : mode,
                "surprise_wm"     : float(brains[0]["s_wm_ema"]),
                "surprise_ltm"    : float(s_ltm),
                "ltm_confidence"  : float(ltm_conf(P_ltm_ep)),
                "curiosity"       : float(curio_score(brains[0]["P_wm"])),
                "danger_level"    : float(dl),
                "wind_active"     : wind_active,
                "slope_deg"       : round(np.degrees(slope),2),
                "quake_amp"       : round(q_amp,3),
                "danger_grid"     : (D/dmax).flatten().round(3).tolist(),
                "terrain_trust"   : M.round(3).tolist(),
                "curriculum_stage": stage,
                "dist_A"          : float(abs(xA[0]-TARGET_X)),
                "dist_B"          : float(abs(xB[0]-TARGET_X)),
                "dopamine"        : float(max(0.,1.-best_dist_step/TARGET_X)),
                "daughter_active" : d_active,
                "best_dist_ever"  : round(best_dist_ever,3),
                "target_reached_count": target_reached_count,
                # Phase 12 new
                "mourning_level"  : round(avg_grief,3),
                "mourning_events" : mourning_events,
                "imagination_depth": imagination_depth,
                "near_miss_replays": near_miss_replays,
                "legacy_map"      : L_legacy.round(3).tolist(),
                "ghost_count"     : ghost_count,
                "adrenaline_active": any(b.get("adrenaline", 0.) > 0.5 for b in brains if b["alive"]),
                "phantom_pains"   : len(L_phantom_pains),
                "ambition"        : float(1.0 + max(0.0, ((time.time() - brains[0]["t0"]) - 30.0) / 10.0)) if brains and brains[0]["alive"] else 1.0,
                "boredom"         : float(max([b.get("boredom", 0.0) for b in brains if b["alive"]] + [0.0])),
            })

            # HUD
            p.removeAllUserDebugItems()
            grief_str = f" [GRIEVING {avg_grief:.2f}]" if avg_grief > 0.1 else ""
            img_str   = f" [DEPTH:{imagination_depth}]" if stage >= 4 else ""
            hud = (f"STAGE {stage} | {mode}{grief_str}{img_str} | "
                   f"Best:{best:.1f}s | Reached:{target_reached_count}x | "
                   f"Ghosts:{ghost_count}")
            p.addUserDebugText(hud,[0,0,0.7],textColorRGB=[0,1,0.5],textSize=1.2)

            # Redraw ghost markers and target rings
            for idx in range(0,100,3):
                lv = float(L_legacy[idx])
                if lv > 0.15:
                    xp = -3.0 + idx * 6.0/100.0
                    a  = min(1.0, lv)
                    p.addUserDebugLine([xp-.08,0,.02],[xp+.08,0,.02],
                                       [a,a*.2,a*.2],1)
                    p.addUserDebugLine([xp,-.08,.02],[xp,.08,.02],
                                       [a,a*.2,a*.2],1)
            for h in [0.05,0.25,0.45]:
                for angle in np.linspace(0,2*np.pi,8):
                    p.addUserDebugLine(
                        [TARGET_X,0,h],
                        [TARGET_X+.3*math.cos(angle),.3*math.sin(angle),h],
                        [1.,0.2,0.5],2)

            p.stepSimulation()
            time.sleep(DT)

            if alive_count == 0:
                T_ltm, P_ltm = T_ltm_ep.copy(), P_ltm_ep.copy()
                D_global     = D.copy()
                M_terrain    = M.copy()
                # L_legacy never resets — intentional

                if episode % 50 == 0:
                    np.save("checkpoint_ltm_T.npy",   T_ltm)
                    np.save("checkpoint_ltm_P.npy",   P_ltm)
                    np.save("checkpoint_danger.npy",  D_global)
                    np.save("checkpoint_terrain.npy", M_terrain)
                    np.save("checkpoint_legacy.npy",  L_legacy)
                    print(f"  [CHECKPOINT saved ep {episode}]")

                print(f"  Episode {episode} over. "
                      f"Best:{best:.2f}s  "
                      f"LTM:{ltm_conf(P_ltm)*100:.1f}%  "
                      f"Reached:{target_reached_count}x  "
                      f"Ghosts:{ghost_count}  "
                      f"MourningEvents:{mourning_events}")
                time.sleep(1.5)
                break

if __name__ == "__main__":
    main()
