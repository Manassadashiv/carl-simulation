# -*- coding: utf-8 -*-
# CARL Phase 11: HONEST ARCHITECTURE
# ============================================================
# What changed from Phase 10C and WHY:
#
# REMOVED: Teleport curriculum (Stages 1-2 hand-holding)
#   WHY: Teleporting fed false motor data into T_wm/T_ltm.
#        The memory system learned "perfect movement happened"
#        but the robot's motors never caused it. Corrupted signal.
#
# REPLACED WITH: Earned curriculum
#   Stage 1: Flat ground. No help. CARL falls and learns alone.
#   Stage 2: Spawn tilt. Still no help. Must recover or die.
#   Stage 3: Wind added. Spotter catches only extreme falls (>0.55 rad).
#   Stage 4: Slopes + earthquakes. Spotter at 0.62 rad only.
#   Stage 5: Full chaos. No parent. Fully autonomous.
#
# KEPT: Everything that was genuinely good:
#   - 10-body hive mind with shared LTM
#   - Amygdala filter + hippocampal sleep replay
#   - Directed curiosity + confidence-weighted free energy
#   - Dopamine spatial goal (TARGET_X = 2.0)
#   - TRON visuals, coloured swarm, checkpoint saving
#   - Buffer cap, roll detection, slope-aware spawn
#
# NEW: Daughter Minds conflict resolution (Phase 11 addition)
#   When curiosity and danger conflict — spawn 3 micro-rollouts,
#   pick the best. Brain deliberates instead of flipping a coin.
# ============================================================

import sys, pybullet as p, pybullet_data, time, numpy as np
import asyncio, websockets, json, threading, math

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
    "survivals": [], "pitches": [], "distances": [], "alive_flags": [],
    "dist_A": 2.0, "dist_B": 2.0, "dopamine": 0.0,
    "curriculum_stage": 1,
    "daughter_active": False,   # NEW: shows when daughter minds are deliberating
    "best_dist_ever": 2.0,      # NEW: closest any bot has ever gotten to target
    "target_reached_count": 0,  # NEW: how many times target was actually reached
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
SDIM     = 5          # [x_pos, x_vel, pitch, pitch_vel, neck_angle]
P0_WM    = 500.0  * (SDIM + 1)
P0_LTM   = 2000.0 * (SDIM + 1)
TARGET_X = 2.0        # Spatial goal — 2 metres ahead
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
    return float(np.clip(1.0 - np.trace(P) / P0_WM,  0., 1.))

def ltm_conf(P):
    return float(np.clip(1.0 - np.trace(P) / P0_LTM, 0., 1.))

def curio_score(P_wm, boost=0.0):
    base = float(np.clip(np.trace(P_wm) / P0_WM, 0., 1.))
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

# ── DIRECTED CURIOSITY ────────────────────────────────────────
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

# ── FREE ENERGY ACTION SELECTION ─────────────────────────────
def pick_action(x, T_wm, T_ltm, P_wm, D, M):
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
            dist_cost = 0.3 * abs(xs_n[0] - TARGET_X)   # dopamine drive
            F        += (pred_s + d_cost + 0.3*t_risk + dist_cost) * conf
            xs        = xs_n
        if F < best_F:
            best_F, best_u = F, u
    return best_u

# ── DAUGHTER MINDS: Conflict Resolution ──────────────────────
# When curiosity and danger strongly conflict, spawn 3 micro-minds.
# Each evaluates a different strategy. Parent picks best outcome.
# This replaces the coin-flip between CURIOUS and EXPLOITING.
def daughter_minds(x, T_wm, T_ltm, P_wm, D, M, curio):
    """
    Three daughter strategies:
      D1 — Pure survival  (danger weight x2, no dopamine)
      D2 — Pure dopamine  (danger weight x0.5, full dopamine)
      D3 — Balanced       (normal weights — same as pick_action)
    Parent picks the daughter whose best action has lowest combined cost.
    """
    alpha  = wm_conf(P_wm)
    T_use  = alpha * T_wm + (1.0 - alpha) * T_ltm
    Ad, Bd = T_use[:SDIM,:].T, T_use[SDIM,:]

    strategies = [
        {"d_w": 2.0, "dop_w": 0.0,  "name": "SAFE"},     # D1
        {"d_w": 0.5, "dop_w": 0.6,  "name": "BOLD"},     # D2
        {"d_w": 1.0, "dop_w": 0.3,  "name": "BALANCED"}, # D3
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
                dist_cost = strat["dop_w"] * abs(xs_n[0] - TARGET_X)
                F        += (pred_s + d_cost + 0.3*t_risk + dist_cost) * conf
                xs        = xs_n
            if F < best_F:
                best_F, best_u, best_name = F, u, strat["name"]

    return best_u, best_name

# ── HIPPOCAMPAL SLEEP REPLAY ─────────────────────────────────
def sleep_replay(D, buf):
    if not buf: return D
    top50 = sorted(buf, key=lambda m: m[3], reverse=True)[:50]
    for (xk_m, uk_m, xn_m, surprise) in top50:
        for _ in range(2):
            D = danger_update(D, xn_m[2], xn_m[3], surprise, rate=0.2)
    return D

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
        "buf": [], "t0": time.time(), "alive": True, "sv": 0.,
        "rid": None, "daughter_mode": "BALANCED",
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
    episode, best = 0, 0.
    best_dist_ever     = TARGET_X
    target_reached_count = 0

    print(f"\n== CARL Phase 11: HONEST ARCHITECTURE ==")
    print(f"{N_BODIES} bodies | Earned curriculum | Daughter minds | Dopamine goal\n")

    while True:
        episode += 1
        brain["episode"] = episode

        # ── EARNED CURRICULUM ────────────────────────────────
        # No teleport. No hand-holding. CARL earns every stage.
        # Thresholds based on best survival time in seconds.
        stage = 1
        if best > 10.0:  stage = 2   # spawn tilted, still no help
        if best > 20.0:  stage = 3   # wind + spotter at 0.55 rad
        if best > 35.0:  stage = 4   # slopes + quake + spotter at 0.62 rad
        if best > 55.0:  stage = 5   # full chaos, no parent
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
                p.addUserDebugLine([-2,y_line,0.01],[3,y_line,0.01],[0.05,0.05,0.25],1)
            for x_line in np.linspace(-2, 3, 11):
                p.addUserDebugLine([x_line,-5,0.01],[x_line,5,0.01],[0.05,0.05,0.25],1)

            p.resetDebugVisualizerCamera(3.0, 60, -20, [1.0, 0, 0.2])

            # Slope only in Stage 4+
            slope = 0.0
            if stage >= 4:
                slope = float(np.random.uniform(-0.04, 0.04))
            p.resetBasePositionAndOrientation(
                plane, [0,0,0], p.getQuaternionFromEuler([slope,0,0]))

            # Spawn tilt only in Stage 2+
            init_pitch = 0.0
            if stage >= 2:
                init_pitch = float(np.random.uniform(-0.12, 0.12))

            # Glowing target sphere
            tgt = p.createVisualShape(p.GEOM_SPHERE, radius=0.15,
                                      rgbaColor=[1,0.2,0.5,0.9])
            p.createMultiBody(baseMass=0, baseVisualShapeIndex=tgt,
                              basePosition=[TARGET_X,0,0.15])
            # Laser rings around target
            for h in [0.05, 0.25, 0.45]:
                for angle in np.linspace(0, 2*np.pi, 8):
                    p.addUserDebugLine(
                        [TARGET_X, 0, h],
                        [TARGET_X+0.3*math.cos(angle), 0.3*math.sin(angle), h],
                        [1.0, 0.2, 0.5], 2)
            # Path laser
            p.addUserDebugLine([0,0,0.05],[TARGET_X,0,0.05],[1.0,0.2,0.5],3)

            # Spawn swarm
            y_offsets = np.linspace(-4.0, 4.0, N_BODIES).tolist()
            bodies = [spawn_robot(y_offset=y, initial_pitch=init_pitch, slope=slope)
                      for y in y_offsets]

            # Colour gradient cyan → purple
            for i, rid in enumerate(bodies):
                r_c = 0.4 + 0.6 * (i / max(1, N_BODIES-1))
                b_c = 1.0 - 0.4 * (i / max(1, N_BODIES-1))
                p.changeVisualShape(rid, -1, rgbaColor=[r_c,0.2,b_c,1.0])
                p.changeVisualShape(rid,  3, rgbaColor=[r_c,0.2,b_c,1.0])

        except Exception as e:
            print(f"PyBullet crash, restarting... {e}")
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
        M        = M_terrain.copy()
        P_ltm_ep = P_ltm.copy()
        T_ltm_ep = T_ltm.copy()
        next_wind = np.random.randint(15*240, 30*240)

        stage_labels = {
            1: "STAGE 1 — LEARNING TO STAND",
            2: "STAGE 2 — LEARNING TO RECOVER",
            3: "STAGE 3 — PUSHING THROUGH WIND",
            4: "STAGE 4 — COMBINATIONS",
            5: "STAGE 5 — FULL AUTONOMY",
        }

        lc = ltm_conf(P_ltm_ep)
        print(f"[Ep {episode:3d}] {stage_labels[stage]}  "
              f"LTM:{lc*100:.0f}%  Best:{best:.1f}s")

        for step in range(500_000):
            # Wind — Stage 3+
            wind_active = False
            if stage >= 3 and step >= next_wind:
                fx = float(np.random.uniform(-3., 3.))
                for b in brains:
                    if b["alive"]:
                        p.applyExternalForce(b["rid"],-1,[fx,0,0],[0,0,.3],p.WORLD_FRAME)
                next_wind = step + np.random.randint(15*240, 30*240)
                wind_active = True

            # Earthquake — Stage 4+
            q_amp = 0.0
            if stage >= 4:
                t_ep  = step * DT
                q_amp = float(np.clip(t_ep/90., 0., 0.3))
                q_f   = q_amp * float(np.sin(2*np.pi*1.5*t_ep))
                if q_amp > 0.01:
                    for b in brains:
                        if b["alive"]:
                            p.applyExternalForce(b["rid"],-1,[q_f,0,0],[0,0,0],p.WORLD_FRAME)

            # ── STEP EACH ROBOT ───────────────────────────────
            for i, rb in enumerate(brains):
                if not rb["alive"]: continue
                rid = rb["rid"]

                xn, y_pos, z_pos, roll = get_state(rid)

                # 1. WM update
                rb["T_wm"], rb["P_wm"], s_wm = rls_update(
                    rb["T_wm"], rb["P_wm"], rb["xk"], rb["uk"], xn,
                    lam=0.990, p_floor=0.001)

                # 2. Shared LTM continuous update
                T_ltm_ep, P_ltm_ep, _ = rls_update(
                    T_ltm_ep, P_ltm_ep, rb["xk"], rb["uk"], xn,
                    lam=0.99995, p_floor=0.002)

                rb["s_wm_ema"] = .1*s_wm  + .9*rb["s_wm_ema"]
                rb["s_slow"]   = .005*s_wm + .995*rb["s_slow"]
                nov_boost = float(np.clip(
                    (rb["s_wm_ema"]/(rb["s_slow"]+1e-6)-2.)*0.3, 0., .5))

                D = danger_update(D, xn[2], xn[3], rb["s_wm_ema"])
                M = terrain_update(M, xn[0], rb["s_wm_ema"])
                rb["danger_ema"] = .05*danger_at(D,xn[2],xn[3]) + .95*rb["danger_ema"]

                # 3. Amygdala filter
                if rb["danger_ema"] > 0.20 or s_wm > 0.25:
                    rb["buf"].append((rb["xk"].copy(), rb["uk"], xn.copy(), s_wm))
                    if len(rb["buf"]) > 200:
                        rb["buf"].pop(0)

                curio = curio_score(rb["P_wm"], nov_boost)
                dl    = danger_at(D, xn[2], xn[3])

                # ── DAUGHTER MINDS CONFLICT RESOLUTION ───────
                # Trigger when curiosity AND danger are both high:
                # the robot genuinely doesn't know whether to explore or hide.
                # Spawn 3 daughter strategies. Parent picks the winner.
                conflict = curio > 0.5 and dl > 0.25
                daughter_active = False

                if conflict:
                    un, d_name = daughter_minds(
                        xn, rb["T_wm"], T_ltm_ep, rb["P_wm"], D, M, curio)
                    rb["daughter_mode"] = d_name
                    daughter_active = True
                elif curio > np.random.random():
                    un = directed_explore(xn, rb["T_wm"], rb["P_wm"], D)
                    rb["daughter_mode"] = "CURIOUS"
                else:
                    un = pick_action(xn, rb["T_wm"], T_ltm_ep, rb["P_wm"], D, M)
                    rb["daughter_mode"] = "FEARFUL" if dl > 0.3 else "EXPLOITING"

                # ── SPOTTER (Stage 3 and 4 only) ─────────────
                # Catches only genuinely extreme falls.
                # CARL earns recovery below the threshold on its own.
                if stage == 3 and abs(xn[2]) > 0.55:
                    p.resetBasePositionAndOrientation(
                        rid, [xn[0], y_offsets[i], 0.10],
                        p.getQuaternionFromEuler([0,0,0]))
                    p.resetBaseVelocity(rid, [0,0,0], [0,0,0])
                    rb["t0"] = time.time()  # Reset survival timer!
                elif stage == 4 and abs(xn[2]) > 0.62:
                    p.resetBasePositionAndOrientation(
                        rid, [xn[0], y_offsets[i], 0.10],
                        p.getQuaternionFromEuler([0,0,0]))
                    p.resetBaseVelocity(rid, [0,0,0], [0,0,0])
                    rb["t0"] = time.time()  # Reset survival timer!

                apply_torque(rid, un)
                rb["fatigue"] = .999*rb["fatigue"] + .001*abs(un)
                rb["xk"], rb["uk"] = xn, un
                rb["sv"] = time.time() - rb["t0"]

                # Track best distance to target ever
                dist_now = abs(xn[0] - TARGET_X)
                if dist_now < best_dist_ever:
                    best_dist_ever = dist_now
                    brain["best_dist_ever"] = round(best_dist_ever, 3)

                # Target reached?
                if dist_now < 0.20:
                    target_reached_count += 1
                    brain["target_reached_count"] = target_reached_count
                    print(f"  *** TARGET REACHED by Body {i+1}! "
                          f"Total: {target_reached_count} ***")

                # ── DEATH CHECK ───────────────────────────────
                floor_z = y_pos * math.sin(slope)
                local_z = z_pos - floor_z

                if local_z < 0.065 or abs(xn[2]) > 0.65 or abs(roll) > 0.5:
                    sv   = rb["sv"]
                    best = max(best, sv)
                    brain["episode_history"].append(round(sv, 2))
                    brain["best_survival"] = round(best, 2)
                    rb["alive"] = False

                    for _ in range(4):
                        D = danger_update(D, xn[2], xn[3], 10., rate=0.3)

                    brain["sleeping"] = True
                    n_rep = min(len(rb["buf"]), 50)
                    print(f"  Body {i+1} died {sv:.2f}s | "
                          f"Stage {stage} | Mode:{rb['daughter_mode']} | "
                          f"Replay {n_rep}")
                    D = sleep_replay(D, rb["buf"])
                    brain["replay_count"] = n_rep
                    brain["sleeping"] = False

                    try: p.removeBody(rid)
                    except: pass

            # ── DASHBOARD UPDATE ─────────────────────────────
            xA   = brains[0]["xk"] if brains[0]["xk"] is not None else np.zeros(SDIM)
            xB   = brains[1]["xk"] if brains[1]["xk"] is not None else np.zeros(SDIM)
            dmax = float(np.max(D)) + 1e-6
            PhiA = np.append(xA, brains[0]["uk"]).reshape(SDIM+1,1)
            s_ltm = float(np.linalg.norm(xA - (T_ltm_ep.T @ PhiA).flatten()))
            dl = danger_at(D, xA[2], xA[3]) if brains[0]["alive"] else 0.0

            alive_count = sum(1 for b in brains if b["alive"])

            # Determine display mode
            alive_modes = [b["daughter_mode"] for b in brains if b["alive"]]
            mode = alive_modes[0] if alive_modes else "DEAD"
            if brain.get("sleeping"): mode = "SLEEPING"

            # Daughter active if any body used daughter minds this step
            d_active = any(b["daughter_mode"] in ("SAFE","BOLD","BALANCED")
                          for b in brains if b["alive"])

            best_dist_this_step = min(
                [abs(b["xk"][0]-TARGET_X) if b["xk"] is not None else TARGET_X
                 for b in brains], default=TARGET_X)

            brain.update({
                # Legacy keys
                "survival_A"     : round(brains[0]["sv"], 2),
                "survival_B"     : round(brains[1]["sv"], 2),
                "pitch_A"        : float(xA[2]),
                "pitch_B"        : float(xB[2]),
                "neck_A"         : float(xA[4]),
                "neck_B"         : float(xB[4]),
                "fatigue_A"      : float(brains[0]["fatigue"]),
                "fatigue_B"      : float(brains[1]["fatigue"]),
                "wm_confidence_A": float(wm_conf(brains[0]["P_wm"])),
                "wm_confidence_B": float(wm_conf(brains[1]["P_wm"])),
                # Swarm keys
                "survivals"      : [round(b["sv"],2) for b in brains],
                "pitches"        : [round(float(b["xk"][2]) if b["xk"] is not None else 0.,3) for b in brains],
                "distances"      : [round(float(abs(b["xk"][0]-TARGET_X)) if b["xk"] is not None else 2.,3) for b in brains],
                "alive_flags"    : [b["alive"] for b in brains],
                # Shared state
                "best_survival"  : round(best, 2),
                "mode"           : mode,
                "surprise_wm"    : float(brains[0]["s_wm_ema"]),
                "surprise_ltm"   : float(s_ltm),
                "ltm_confidence" : float(ltm_conf(P_ltm_ep)),
                "curiosity"      : float(curio_score(brains[0]["P_wm"])),
                "danger_level"   : float(dl),
                "wind_active"    : wind_active,
                "slope_deg"      : round(np.degrees(slope),2),
                "quake_amp"      : round(q_amp,3),
                "danger_grid"    : (D/dmax).flatten().round(3).tolist(),
                "terrain_trust"  : M.round(3).tolist(),
                "curriculum_stage": stage,
                "dist_A"         : float(abs(xA[0]-TARGET_X)),
                "dist_B"         : float(abs(xB[0]-TARGET_X)),
                "dopamine"       : float(max(0., 1. - best_dist_this_step/TARGET_X)),
                "daughter_active": d_active,
                "best_dist_ever" : round(best_dist_ever,3),
                "target_reached_count": target_reached_count,
            })

            # PyBullet HUD
            p.removeAllUserDebugItems()
            d_flag = " [DAUGHTERS]" if d_active else ""
            hud = (f"STAGE {stage} | {mode}{d_flag} | "
                   f"Best:{best:.1f}s | Target:{best_dist_ever:.2f}m away")
            p.addUserDebugText(hud,[0,0,0.6],textColorRGB=[0,1,0.5],textSize=1.3)

            # Redraw target rings every step so they persist
            for h in [0.05,0.25,0.45]:
                for angle in np.linspace(0,2*np.pi,8):
                    p.addUserDebugLine(
                        [TARGET_X,0,h],
                        [TARGET_X+0.3*math.cos(angle),0.3*math.sin(angle),h],
                        [1.0,0.2,0.5],2)

            p.stepSimulation()
            time.sleep(DT)

            # Episode over when all dead
            if alive_count == 0:
                T_ltm, P_ltm = T_ltm_ep.copy(), P_ltm_ep.copy()
                D_global     = D.copy()
                M_terrain    = M.copy()

                # Checkpoint every 50 episodes
                if episode % 50 == 0:
                    np.save("checkpoint_ltm_T.npy", T_ltm)
                    np.save("checkpoint_ltm_P.npy", P_ltm)
                    np.save("checkpoint_danger.npy", D_global)
                    np.save("checkpoint_terrain.npy", M_terrain)
                    print(f"  [CHECKPOINT saved at episode {episode}]")

                print(f"  Episode {episode} over. "
                      f"Best:{best:.2f}s  "
                      f"LTM:{ltm_conf(P_ltm)*100:.1f}%  "
                      f"BestDist:{best_dist_ever:.2f}m  "
                      f"Reached:{target_reached_count}x")
                time.sleep(1.5)
                break

if __name__ == "__main__":
    main()
