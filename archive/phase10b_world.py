# -*- coding: utf-8 -*-
# CARL Phase 10B: Enhanced Mind + Dynamic World
# Two-Speed Memory | Amygdala | Sleep Replay | Directed Curiosity
# Procedural Slopes | Wind | Earthquake | Terrain Memory

import sys
import pybullet as p
import pybullet_data
import time
import numpy as np
import asyncio
import websockets
import json
import threading

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ── DASHBOARD STATE ──────────────────────────────────────────
brain = {
    "episode": 0, "survival_time": 0.0, "best_survival": 0.0,
    "mode": "BOOTING",
    "pitch": 0.0, "surprise_wm": 0.0, "surprise_ltm": 0.0,
    "wm_confidence": 0.0, "ltm_confidence": 0.0,
    "curiosity": 1.0, "danger_level": 0.0, "fatigue": 0.0,
    "sleeping": False, "replay_count": 0,
    "action": 0.0, "episode_history": [],
    "danger_grid": [0.0] * 400,
    "terrain_trust": [1.0] * 100,
    "wind_active": False,
    "slope_deg": 0.0,
    "quake_amp": 0.0,
}

async def _ws_handler(ws):
    try:
        while True:
            await ws.send(json.dumps(brain))
            await asyncio.sleep(1 / 30)
    except Exception:
        pass

def _run_ws():
    async def _serve():
        print("[WS] Dashboard: ws://localhost:8765")
        async with websockets.serve(_ws_handler, "localhost", 8765):
            await asyncio.Future()
    asyncio.run(_serve())

threading.Thread(target=_run_ws, daemon=True).start()

# ── CONSTANTS ────────────────────────────────────────────────
DT      = 1.0 / 240.0
ACTIONS = [-8., -5., -2., -1., 0., 1., 2., 5., 8.]
HORIZON = 40
DRES    = 20
P0_WM   = 500.0  * 5
P0_LTM  = 2000.0 * 5

# ── TWO-SPEED MEMORY ─────────────────────────────────────────
def fresh_wm():
    T = np.zeros((5, 4)); T[0:4] = np.eye(4)
    return T, 500.0 * np.eye(5)

def fresh_ltm():
    T = np.zeros((5, 4)); T[0:4] = np.eye(4)
    return T, 2000.0 * np.eye(5)

def rls_update(T, P, xk, uk, xn, lam, p_floor=0.001):
    Phi   = np.append(xk, float(uk)).reshape(5, 1)
    e     = xn - (T.T @ Phi).flatten()
    PPhi  = P @ Phi
    denom = lam + float((Phi.T @ PPhi).squeeze())
    gain  = PPhi / denom
    T     = T + gain @ e.reshape(1, 4)
    P     = (P - gain @ (Phi.T @ P)) / lam
    P     = np.maximum(P, p_floor * np.eye(5))
    return T, P, float(np.linalg.norm(e))

def wm_confidence(P):
    return float(np.clip(1.0 - np.trace(P) / P0_WM,  0.0, 1.0))

def ltm_confidence(P):
    return float(np.clip(1.0 - np.trace(P) / P0_LTM, 0.0, 1.0))

def curiosity_score(P_wm, boost=0.0):
    base = float(np.clip(np.trace(P_wm) / P0_WM, 0.0, 1.0))
    return float(np.clip(base + boost + 0.08, 0.0, 1.0))

def blended_predict(T_wm, T_ltm, P_wm, Phi):
    alpha = wm_confidence(P_wm)
    return alpha * (T_wm.T @ Phi).flatten() + (1 - alpha) * (T_ltm.T @ Phi).flatten()

# ── TERRAIN MEMORY ───────────────────────────────────────────
def fresh_terrain():
    return np.ones(100)

def terrain_update(M, x_pos, surprise, rate=0.08):
    i = int(np.clip((x_pos + 3.0) / 6.0 * 100, 0, 99))
    M[i] = (1 - rate) * M[i] + rate * max(0., 1. - surprise * 5.)
    return M

def terrain_trust_at(M, x_pos):
    i = int(np.clip((x_pos + 3.0) / 6.0 * 100, 0, 99))
    return float(M[i])

# ── DANGER MAP ────────────────────────────────────────────────
def fresh_danger():
    return np.zeros((DRES, DRES))

def _cell(pitch, vel):
    i = int(np.clip((pitch + .65) / 1.30 * DRES, 0, DRES - 1))
    j = int(np.clip((vel   + 3.0) / 6.0  * DRES, 0, DRES - 1))
    return i, j

def danger_update(D, pitch, vel, surprise, rate=0.15):
    i, j = _cell(pitch, vel)
    D[i, j] = (1 - rate) * D[i, j] + rate * surprise
    return D

def danger_at(D, pitch, vel):
    return float(D[_cell(pitch, vel)])

# ── DIRECTED CURIOSITY ────────────────────────────────────────
def directed_explore(x, T_wm, P_wm, D):
    Ad, Bd = T_wm[0:4, :].T, T_wm[4, :]
    best_u, best_info = 0.0, -float('inf')
    for u in ACTIONS:
        xs = x.copy()
        for _ in range(12):
            xs = Ad @ xs + Bd * u
        Phi_r = np.append(xs, u).reshape(5, 1)
        unc   = float((Phi_r.T @ P_wm @ Phi_r).squeeze())
        info  = unc * (1.0 + danger_at(D, xs[2], xs[3]))
        if info > best_info:
            best_info, best_u = info, u
    return best_u

# ── CONFIDENCE-WEIGHTED FREE ENERGY ──────────────────────────
def pick_action(x, T_wm, T_ltm, P_ltm, D, M):
    Ad, Bd  = T_wm[0:4, :].T, T_wm[4, :]
    ltm_con = ltm_confidence(P_ltm)
    best_u, best_F = 0.0, float('inf')
    for u in ACTIONS:
        xs, F = x.copy(), 0.0
        for h in range(HORIZON):
            xs_next = Ad @ xs + Bd * u
            conf    = ltm_con * (0.95 ** h)
            pred_s  = float(np.linalg.norm(xs_next - xs)) * 0.1
            d_cost  = danger_at(D, xs_next[2], xs_next[3])
            t_risk  = 1.0 - terrain_trust_at(M, xs_next[0])
            F      += conf * (pred_s + d_cost + 0.3 * t_risk)
            xs      = xs_next
        if F < best_F:
            best_F, best_u = F, u
    return best_u

# ── HIPPOCAMPAL SLEEP REPLAY ─────────────────────────────────
def sleep_replay(T_ltm, P_ltm, buf):
    if not buf:
        return T_ltm, P_ltm
    top50 = sorted(buf, key=lambda m: m[3], reverse=True)[:50]
    for (xk_m, uk_m, xn_m, _) in top50:
        for _ in range(3):
            T_ltm, P_ltm, _ = rls_update(T_ltm, P_ltm, xk_m, uk_m, xn_m,
                                          lam=0.9999, p_floor=0.005)
    return T_ltm, P_ltm

# ── PYBULLET HELPERS ─────────────────────────────────────────
def get_state(rid):
    pos, quat    = p.getBasePositionAndOrientation(rid)
    vel, ang_vel = p.getBaseVelocity(rid)
    euler        = p.getEulerFromQuaternion(quat)
    return np.array([pos[0], vel[0], euler[1], ang_vel[1]], dtype=float), float(pos[2])

def apply_torque(rid, u):
    t = float(np.clip(u, -8., 8.))
    p.setJointMotorControl2(rid, 0, p.TORQUE_CONTROL, force=t)
    p.setJointMotorControl2(rid, 1, p.TORQUE_CONTROL, force=t)

def spawn_robot():
    rid = p.loadURDF("carl.urdf", [0, 0, .08],
                     p.getQuaternionFromEuler([0, .04, 0]))
    p.setJointMotorControl2(rid, 2, p.POSITION_CONTROL, targetPosition=0, force=20.)
    p.setJointMotorControl2(rid, 0, p.VELOCITY_CONTROL, force=0)
    p.setJointMotorControl2(rid, 1, p.VELOCITY_CONTROL, force=0)
    return rid

# ── MAIN ─────────────────────────────────────────────────────
def main():
    global brain

    p.connect(p.GUI, options="--width=1280 --height=720")
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    plane = p.loadURDF("plane.urdf")
    p.changeVisualShape(plane, -1, rgbaColor=[.04, .04, .07, 1])

    T_ltm, P_ltm = fresh_ltm()
    D_global     = fresh_danger()
    M_terrain    = fresh_terrain()
    episode, best = 0, 0.

    print("\n== CARL Phase 10B: Enhanced Mind + Dynamic World ==")
    print("Memory | Amygdala | Sleep | Curiosity | Slopes | Wind | Earthquake\n")

    while True:
        episode += 1
        brain["episode"] = episode

        # PROCEDURAL SLOPE: random floor tilt +-3 degrees
        slope = float(np.random.uniform(-0.052, 0.052))
        p.resetBasePositionAndOrientation(
            plane, [0, 0, 0], p.getQuaternionFromEuler([slope, 0, 0]))
        p.setGravity(0, 0, -9.81)

        # Wind schedule
        next_wind = np.random.randint(15 * 240, 30 * 240)

        rid = spawn_robot()
        for _ in range(10): p.stepSimulation()
        time.sleep(0.3)

        # Fresh working memory each life
        T_wm, P_wm = fresh_wm()

        # Inherit global knowledge with covariance dilation
        T_ltm_ep = T_ltm.copy()
        P_ltm_ep = P_ltm.copy() + 0.5 * np.eye(5)
        D        = D_global.copy()
        M        = M_terrain.copy()

        xk, _    = get_state(rid)
        uk       = 0.
        s_wm_ema = 0.
        s_ltm_ema= 0.
        s_slow   = 0.01
        danger_ema = 0.
        fatigue  = 0.
        buf      = []
        t0       = time.time()

        brain.update({"slope_deg": round(np.degrees(slope), 2),
                      "wind_active": False, "quake_amp": 0.0})

        wm_con  = wm_confidence(P_wm)
        ltm_con = ltm_confidence(P_ltm_ep)
        print(f"[Life {episode:3d}]  WM:{wm_con*100:.0f}%  LTM:{ltm_con*100:.0f}%  "
              f"Slope:{np.degrees(slope):+.1f}deg  Best:{best:.1f}s")

        for step in range(500_000):
            xn, z_h = get_state(rid)

            # WIND DISTURBANCE
            wind_active = False
            if step >= next_wind:
                fx = float(np.random.uniform(-4., 4.))
                p.applyExternalForce(rid, -1, [fx, 0, 0], [0, 0, 0.3], p.WORLD_FRAME)
                next_wind = step + np.random.randint(15 * 240, 30 * 240)
                wind_active = True

            # EARTHQUAKE: grows over 60 seconds
            t_ep  = step * DT
            q_amp = float(np.clip(t_ep / 60.0, 0., 0.4))
            q_f   = q_amp * float(np.sin(2 * np.pi * 1.5 * t_ep))
            if q_amp > 0.01:
                p.applyExternalForce(rid, -1, [q_f, 0, 0], [0, 0, 0], p.WORLD_FRAME)

            Phi = np.append(xk, uk).reshape(5, 1)

            # 1. Working memory update (every step)
            T_wm, P_wm, s_wm = rls_update(T_wm, P_wm, xk, uk, xn,
                                            lam=0.990, p_floor=0.001)

            # Blended surprise
            x_blend   = blended_predict(T_wm, T_ltm_ep, P_wm, Phi)
            s_ltm     = float(np.linalg.norm(xn - x_blend))
            s_wm_ema  = .1 * s_wm  + .9 * s_wm_ema
            s_ltm_ema = .1 * s_ltm + .9 * s_ltm_ema
            s_slow    = .005 * s_wm + .995 * s_slow
            nov_boost = float(np.clip((s_wm_ema / (s_slow + 1e-6) - 2.) * 0.3, 0., .5))

            # 2. Danger + terrain update
            D = danger_update(D, xn[2], xn[3], s_wm_ema)
            M = terrain_update(M, xn[0], s_wm_ema)
            t_risk     = 1.0 - terrain_trust_at(M, xn[0])
            danger_ema = .05 * (danger_at(D, xn[2], xn[3]) + 0.3 * t_risk) + .95 * danger_ema

            # 3. AMYGDALA: write to LTM only when danger is elevated
            if danger_ema > 0.25:
                T_ltm_ep, P_ltm_ep, _ = rls_update(T_ltm_ep, P_ltm_ep, xk, uk, xn,
                                                     lam=0.9999, p_floor=0.005)

            # 4. Buffer high-surprise moments for sleep replay
            if s_wm > 0.3:
                buf.append((xk.copy(), uk, xn.copy(), s_wm))

            wm_con  = wm_confidence(P_wm)
            ltm_con = ltm_confidence(P_ltm_ep)
            curio   = curiosity_score(P_wm, nov_boost)

            # 5. Action selection
            if curio > np.random.random():
                un   = directed_explore(xn, T_wm, P_wm, D)
                mode = "CURIOUS/DIRECTED"
            else:
                un   = pick_action(xn, T_wm, T_ltm_ep, P_ltm_ep, D, M)
                dl   = danger_at(D, xn[2], xn[3])
                mode = "FEARFUL" if dl > 0.25 else "EXPLOITING"

            apply_torque(rid, un)
            fatigue = .999 * fatigue + .001 * abs(un)
            xk, uk  = xn, un

            # Dashboard
            dmax = float(np.max(D)) + 1e-6
            brain.update({
                "survival_time" : round(time.time() - t0, 2),
                "best_survival" : round(best, 2),
                "mode"          : mode,
                "pitch"         : float(xn[2]),
                "surprise_wm"   : float(s_wm_ema),
                "surprise_ltm"  : float(s_ltm_ema),
                "wm_confidence" : float(wm_con),
                "ltm_confidence": float(ltm_con),
                "curiosity"     : float(curio),
                "danger_level"  : float(danger_ema),
                "fatigue"       : float(fatigue),
                "sleeping"      : False,
                "action"        : float(un),
                "danger_grid"   : (D / dmax).flatten().round(3).tolist(),
                "terrain_trust" : M.round(3).tolist(),
                "wind_active"   : wind_active,
                "slope_deg"     : round(np.degrees(slope), 2),
                "quake_amp"     : round(q_amp, 3),
            })

            p.resetDebugVisualizerCamera(1.2, 45, -15, [xn[0], 0, .4])
            p.stepSimulation()
            time.sleep(DT)

            # DEATH: z-height primary, pitch secondary
            if z_h < 0.05 or abs(xn[2]) > 0.65:
                sv   = time.time() - t0
                best = max(best, sv)
                brain["episode_history"].append(round(sv, 2))
                brain["best_survival"] = round(best, 2)
                brain["mode"] = "DEAD"

                for _ in range(4):
                    D = danger_update(D, xn[2], xn[3], 10., rate=0.3)

                # SLEEP PHASE
                brain["sleeping"] = True
                brain["mode"]     = "SLEEPING"
                n_rep = min(len(buf), 50)
                print(f"         -> Died {sv:.2f}s | Replaying {n_rep} memories | "
                      f"Quake:{q_amp:.2f}")
                T_ltm_ep, P_ltm_ep = sleep_replay(T_ltm_ep, P_ltm_ep, buf)
                brain["replay_count"] = n_rep
                brain["sleeping"]     = False

                T_ltm, P_ltm = T_ltm_ep.copy(), P_ltm_ep.copy()
                D_global     = D.copy()
                M_terrain    = M.copy()

                print(f"            LTM after sleep: {ltm_confidence(P_ltm)*100:.1f}%  "
                      f"Slope: {np.degrees(slope):+.1f}deg")

                p.removeBody(rid)
                time.sleep(1.5)
                break

if __name__ == "__main__":
    main()
