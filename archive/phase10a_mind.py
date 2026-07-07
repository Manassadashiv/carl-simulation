# -*- coding: utf-8 -*-
"""
CARL Phase 10A: The Enhanced Mind
===================================
New cognitive modules added to Phase 9 Active Inference core:

  1. TWO-SPEED MEMORY
     Working Memory  (lambda=0.990) : fast, volatile, resets each life
     Long-Term Memory (lambda=0.9999): slow, stable, persists forever
     Prediction = blend(WM, LTM) weighted by WM confidence

  2. AMYGDALA FILTER
     Only emotionally significant experiences (high danger) write to LTM.
     Routine moments are discarded. Exactly like the mammalian amygdala.

  3. HIPPOCAMPAL SLEEP REPLAY
     On death: top-50 most surprising moments are re-simulated 3x into LTM.
     The robot "dreams" about near-death experiences to learn from them faster.

  4. DIRECTED CURIOSITY
     Instead of random exploration, seek actions that reduce uncertainty
     specifically in the most dangerous parts of the state space.
     Called Information-Directed Sampling in RL literature.

  5. CONFIDENCE-WEIGHTED IMAGINATION
     Free-energy rollouts discount future steps by model uncertainty.
     CARL now knows when its imagination is unreliable.
"""

import sys, pybullet as p, pybullet_data, time, numpy as np
import asyncio, websockets, json, threading
from collections import deque

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
        print("[WS] Dashboard: ws://localhost:8765")
        async with websockets.serve(_ws_handler, "localhost", 8765):
            await asyncio.Future()
    asyncio.run(_serve())

threading.Thread(target=_run_ws, daemon=True).start()

# ── CONSTANTS ────────────────────────────────────────────────
DT       = 1.0 / 240.0
ACTIONS  = [-8., -5., -2., -1., 0., 1., 2., 5., 8.]
HORIZON  = 40
DRES     = 20

# P0 traces for consciousness calculation
P0_WM    = 500.0  * 5   # Working memory initial trace
P0_LTM   = 2000.0 * 5   # Long-term memory initial trace

# ── MEMORY: Two-Speed ────────────────────────────────────────
def fresh_wm():
    """Working Memory: fast, resets each episode."""
    T = np.zeros((5, 4)); T[0:4] = np.eye(4)
    return T, 500.0 * np.eye(5)

def fresh_ltm():
    """Long-Term Memory: slow, persists across all lives."""
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

def wm_confidence(P_wm):
    return float(np.clip(1.0 - np.trace(P_wm) / P0_WM, 0.0, 1.0))

def ltm_confidence(P_ltm):
    return float(np.clip(1.0 - np.trace(P_ltm) / P0_LTM, 0.0, 1.0))

def curiosity_score(P_wm, novelty_boost=0.0):
    base = float(np.clip(np.trace(P_wm) / P0_WM, 0.0, 1.0))
    return float(np.clip(base + novelty_boost + 0.08, 0.0, 1.0))

def blended_predict(T_wm, T_ltm, P_wm, Phi):
    """Blend WM and LTM predictions. High WM confidence -> trust WM more."""
    alpha = wm_confidence(P_wm)
    return alpha * (T_wm.T @ Phi).flatten() + (1 - alpha) * (T_ltm.T @ Phi).flatten()

# ── DANGER MAP ────────────────────────────────────────────────
def fresh_danger(): return np.zeros((DRES, DRES))

def _cell(pitch, vel):
    i = int(np.clip((pitch + .65) / 1.30 * DRES, 0, DRES - 1))
    j = int(np.clip((vel   + 3.0) / 6.0  * DRES, 0, DRES - 1))
    return i, j

def danger_update(D, pitch, vel, surprise, rate=0.15):
    i, j = _cell(pitch, vel); D[i, j] = (1-rate)*D[i, j] + rate*surprise
    return D

def danger_at(D, pitch, vel):
    return float(D[_cell(pitch, vel)])

# ── DIRECTED CURIOSITY ────────────────────────────────────────
def directed_explore(x, T_wm, P_wm, D):
    """Seek action that reduces uncertainty in most dangerous regions."""
    Ad, Bd = T_wm[0:4, :].T, T_wm[4, :]
    best_u, best_info = 0.0, -float('inf')
    for u in ACTIONS:
        xs = x.copy()
        for _ in range(12):
            xs = Ad @ xs + Bd * u
        Phi_r    = np.append(xs, u).reshape(5, 1)
        local_unc = float((Phi_r.T @ P_wm @ Phi_r).squeeze())
        info_gain = local_unc * (1.0 + danger_at(D, xs[2], xs[3]))
        if info_gain > best_info:
            best_info, best_u = info_gain, u
    return best_u

# ── CONFIDENCE-WEIGHTED IMAGINATION ──────────────────────────
def pick_action(x, T_wm, T_ltm, P_ltm, D):
    """
    Free-energy rollout discounted by LTM confidence.
    Low-confidence futures are treated as untrustworthy.
    """
    Ad, Bd  = T_wm[0:4, :].T, T_wm[4, :]
    ltm_con = ltm_confidence(P_ltm)
    best_u, best_F = 0.0, float('inf')
    for u in ACTIONS:
        xs, F = x.copy(), 0.0
        for h in range(HORIZON):
            xs_next = Ad @ xs + Bd * u
            conf    = ltm_con * (0.95 ** h)   # confidence degrades with horizon
            pred_s  = float(np.linalg.norm(xs_next - xs)) * 0.1
            F      += conf * (pred_s + danger_at(D, xs_next[2], xs_next[3]))
            xs      = xs_next
        if F < best_F: best_F, best_u = F, u
    return best_u

# ── SLEEP / HIPPOCAMPAL REPLAY ────────────────────────────────
def sleep_replay(T_ltm, P_ltm, episode_buffer):
    """Replay top-50 most surprising moments 3x into LTM. The robot 'dreams'."""
    if not episode_buffer: return T_ltm, P_ltm
    top50 = sorted(episode_buffer, key=lambda m: m[3], reverse=True)[:50]
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

# ── MAIN ──────────────────────────────────────────────────────
def main():
    global brain

    p.connect(p.GUI, options="--width=1280 --height=720")
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    plane = p.loadURDF("plane.urdf")
    p.changeVisualShape(plane, -1, rgbaColor=[.04, .04, .07, 1])

    # ── Persistent cross-episode knowledge ──
    T_ltm, P_ltm = fresh_ltm()
    D_global     = fresh_danger()
    episode, best = 0, 0.

    print("\n== CARL Phase 10A: Enhanced Mind ==")
    print("Two-speed memory | Amygdala | Sleep replay | Directed curiosity\n")

    while True:
        episode += 1
        brain["episode"] = episode
        p.setGravity(0, 0, -9.81)

        rid = spawn_robot()
        for _ in range(10): p.stepSimulation()
        time.sleep(0.3)

        # ── Fresh working memory each life ──
        T_wm, P_wm = fresh_wm()

        # ── Inherit global knowledge ──
        T_ltm_ep = T_ltm.copy()
        # Covariance dilation: slightly re-inflate LTM uncertainty each life
        # so the model never fully freezes. Like forgetting a tiny bit while sleeping.
        P_ltm_ep = P_ltm.copy() + 0.5 * np.eye(5)
        D        = D_global.copy()

        xk, _   = get_state(rid)
        uk       = 0.
        s_wm_ema = 0.
        s_ltm_ema= 0.
        s_slow   = 0.01
        nov_boost= 0.
        danger_ema = 0.

        episode_buffer = []   # (xk, uk, xn, surprise)
        fatigue        = 0.
        t0             = time.time()

        wm_con  = wm_confidence(P_wm)
        ltm_con = ltm_confidence(P_ltm_ep)
        print(f"[Life {episode:3d}]  WM:{wm_con*100:.0f}%  LTM:{ltm_con*100:.0f}%  "
              f"Best:{best:.1f}s")

        for step in range(500_000):
            xn, z_h = get_state(rid)
            Phi      = np.append(xk, uk).reshape(5, 1)

            # 1. Working memory always updates (every step)
            T_wm, P_wm, s_wm = rls_update(T_wm, P_wm, xk, uk, xn,
                                            lam=0.990, p_floor=0.001)

            # Blended prediction error using LTM
            x_blend   = blended_predict(T_wm, T_ltm_ep, P_wm, Phi)
            s_ltm     = float(np.linalg.norm(xn - x_blend))

            s_wm_ema  = .1*s_wm  + .9*s_wm_ema
            s_ltm_ema = .1*s_ltm + .9*s_ltm_ema
            s_slow    = .005*s_wm + .995*s_slow

            # Novelty boost: current surprise >> baseline
            nov_boost = float(np.clip((s_wm_ema/(s_slow+1e-6) - 2.0)*0.3, 0., .5))

            # 2. Danger map update
            D = danger_update(D, xn[2], xn[3], s_wm_ema)
            danger_ema = .05*danger_at(D, xn[2], xn[3]) + .95*danger_ema

            # 3. AMYGDALA: only write to LTM when emotionally significant
            if danger_ema > 0.25:
                T_ltm_ep, P_ltm_ep, _ = rls_update(T_ltm_ep, P_ltm_ep, xk, uk, xn,
                                                     lam=0.9999, p_floor=0.005)

            # 4. Buffer high-surprise moments for sleep replay
            if s_wm > 0.3:
                episode_buffer.append((xk.copy(), uk, xn.copy(), s_wm))

            wm_con  = wm_confidence(P_wm)
            ltm_con = ltm_confidence(P_ltm_ep)
            curio   = curiosity_score(P_wm, nov_boost)

            # 5. Action selection
            if curio > np.random.random():
                # Directed curiosity: explore dangerous unknowns
                un   = directed_explore(xn, T_wm, P_wm, D)
                mode = "CURIOUS/DIRECTED"
            else:
                # Confidence-weighted free energy minimization
                un   = pick_action(xn, T_wm, T_ltm_ep, P_ltm_ep, D)
                dl   = danger_at(D, xn[2], xn[3])
                mode = "FEARFUL" if dl > 0.25 else "EXPLOITING"

            apply_torque(rid, un)
            fatigue = .999*fatigue + .001*abs(un)
            xk, uk  = xn, un

            # Dashboard
            dmax = float(np.max(D)) + 1e-6
            brain.update({
                "survival_time" : round(time.time()-t0, 2),
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
                "danger_grid"   : (D/dmax).flatten().round(3).tolist(),
            })

            p.resetDebugVisualizerCamera(1.2, 45, -15, [xn[0], 0, .4])
            p.stepSimulation()
            time.sleep(DT)

            # Death: z-height primary, pitch secondary
            if z_h < 0.05 or abs(xn[2]) > 0.65:
                sv   = time.time() - t0
                best = max(best, sv)
                brain["episode_history"].append(round(sv, 2))
                brain["best_survival"] = round(best, 2)
                brain["mode"] = "DEAD"

                # Burn trauma
                for _ in range(4):
                    D = danger_update(D, xn[2], xn[3], 10., rate=0.3)

                # ── SLEEP PHASE: Hippocampal Replay ──────────
                brain["sleeping"] = True
                brain["mode"]     = "SLEEPING"
                print(f"         -> Died {sv:.2f}s | "
                      f"Replaying {min(len(episode_buffer),50)} memories...")
                T_ltm_ep, P_ltm_ep = sleep_replay(T_ltm_ep, P_ltm_ep, episode_buffer)
                brain["replay_count"] = min(len(episode_buffer), 50)
                brain["sleeping"]     = False

                # Persist to global
                T_ltm, P_ltm = T_ltm_ep.copy(), P_ltm_ep.copy()
                D_global     = D.copy()

                ltm_con_after = ltm_confidence(P_ltm)
                print(f"            LTM confidence after sleep: {ltm_con_after*100:.1f}%")

                p.removeBody(rid)
                time.sleep(1.5)
                break

if __name__ == "__main__":
    main()
