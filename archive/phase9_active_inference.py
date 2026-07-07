# -*- coding: utf-8 -*-
"""
CARL Phase 9: Active Inference (Free Energy Principle)
=======================================================
Run this file. CARL starts with ZERO knowledge.
It explores, builds its own world model, discovers danger,
and gradually learns to balance — completely autonomously.

No goals given. No cost matrices. No Guardian threshold.
One principle: MINIMIZE SURPRISE.
"""

import sys
import pybullet as p
import pybullet_data
import time
import numpy as np
import asyncio
import websockets
import json
import threading

# ── Force UTF-8 output on Windows ───────────────────────────
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ── DASHBOARD SHARED STATE ───────────────────────────────────
brain = {
    "episode": 0, "survival_time": 0.0, "best_survival": 0.0,
    "mode": "BOOTING", "pitch": 0.0, "surprise": 0.0,
    "curiosity": 1.0, "consciousness": 0.0, "danger_level": 0.0,
    "action": 0.0, "episode_history": [],
    "danger_grid": [0.0] * 400,
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
        print("[WS] Dashboard server: ws://localhost:8765")
        async with websockets.serve(_ws_handler, "localhost", 8765):
            await asyncio.Future()
    asyncio.run(_serve())

threading.Thread(target=_run_ws, daemon=True).start()

# ── PHYSICS CONSTANTS ────────────────────────────────────────
DT      = 1.0 / 240.0
ACTIONS = [-8., -5., -2., -1., 0., 1., 2., 5., 8.]
HORIZON = 40
DRES    = 20          # danger grid resolution (20x20)
P0_TR   = 2000.0 * 5  # initial RLS covariance trace = 10000

# ── WORLD MODEL: Tabula Rasa ─────────────────────────────────
# State x = [x_pos, x_vel, pitch, pitch_vel]  (4-vector)
# Input u = scalar torque
# Model: x_{k+1} ~ Theta^T @ [x_k; u_k]  where Theta is (5 x 4)

def fresh_model():
    T      = np.zeros((5, 4))
    T[0:4] = np.eye(4)          # prior: "nothing changes"
    P      = 2000.0 * np.eye(5) # extremely uncertain at birth
    return T, P

def rls_update(T, P, xk, uk, xn, lam=0.9995):
    """RLS update. lam=0.9995 = slower forgetting = more stable model."""
    Phi    = np.append(xk, float(uk)).reshape(5, 1)
    x_pred = (T.T @ Phi).flatten()
    e      = xn - x_pred
    PPhi   = P @ Phi
    denom  = lam + float((Phi.T @ PPhi).squeeze())
    gain   = PPhi / denom
    T      = T + gain @ e.reshape(1, 4)
    P      = (P - gain @ (Phi.T @ P)) / lam
    # P-floor: prevent complete collapse so model stays plastic (never stops learning)
    P      = np.maximum(P, 0.002 * np.eye(5))
    return T, P, float(np.linalg.norm(e))

def consciousness(P):
    """0 = newborn (no knowledge), 1 = fully aware."""
    return float(np.clip(1.0 - np.trace(P) / P0_TR, 0.0, 1.0))

def curiosity(P, novelty_boost=0.0):
    """1 = fully curious (uncertain), 0 = confident.
    novelty_boost temporarily raises curiosity when surprise spikes."""
    base = float(np.clip(np.trace(P) / P0_TR, 0.0, 1.0))
    # Floor: never drop below 8% — always some exploration drive
    return float(np.clip(base + novelty_boost + 0.08, 0.0, 1.0))

# ── SELF-DISCOVERED DANGER MAP ───────────────────────────────
# Grid axes: pitch in [-0.65, 0.65], pitch_vel in [-3, 3]
# Values: accumulated surprise experienced in each (pitch, vel) cell.
# No thresholds — danger is felt, not programmed.

def fresh_danger():
    return np.zeros((DRES, DRES))

def _cell(pitch, vel):
    i = int(np.clip((pitch + 0.65) / 1.30 * DRES, 0, DRES - 1))
    j = int(np.clip((vel   + 3.0 ) / 6.0  * DRES, 0, DRES - 1))
    return i, j

def danger_update(D, pitch, vel, surprise, rate=0.15):
    i, j = _cell(pitch, vel)
    D[i, j] = (1 - rate) * D[i, j] + rate * surprise
    return D

def danger_at(D, pitch, vel):
    i, j = _cell(pitch, vel)
    return float(D[i, j])

# ── FREE ENERGY ACTION SELECTION ─────────────────────────────
# For each candidate action, roll out H steps through the world model.
# Cost = predicted motion surprise + accumulated danger memory.
# Pick minimum cost. No Q/R matrix. No human-coded objective.

def pick_action(x, T, D):
    Ad = T[0:4, :].T
    Bd = T[4, :]
    best_u, best_F = 0.0, float('inf')
    for u in ACTIONS:
        xs, F = x.copy(), 0.0
        for _ in range(HORIZON):
            xs_next  = Ad @ xs + Bd * u
            F       += float(np.linalg.norm(xs_next - xs)) * 0.1
            F       += danger_at(D, xs_next[2], xs_next[3])
            xs       = xs_next
        if F < best_F:
            best_F, best_u = F, u
    return best_u

# ── PYBULLET HELPERS ─────────────────────────────────────────
def get_state(rid):
    pos, quat    = p.getBasePositionAndOrientation(rid)
    vel, ang_vel = p.getBaseVelocity(rid)
    euler        = p.getEulerFromQuaternion(quat)
    state = np.array([pos[0], vel[0], euler[1], ang_vel[1]], dtype=float)
    return state, float(pos[2])   # also return z-height

def apply_torque(rid, u):
    t = float(np.clip(u, -8.0, 8.0))
    p.setJointMotorControl2(rid, 0, p.TORQUE_CONTROL, force=t)
    p.setJointMotorControl2(rid, 1, p.TORQUE_CONTROL, force=t)

def spawn_robot():
    rid = p.loadURDF("carl.urdf", [0, 0, 0.08],
                     p.getQuaternionFromEuler([0, 0.04, 0]))
    p.setJointMotorControl2(rid, 2, p.POSITION_CONTROL,
                            targetPosition=0, force=20.0)
    p.setJointMotorControl2(rid, 0, p.VELOCITY_CONTROL, force=0)
    p.setJointMotorControl2(rid, 1, p.VELOCITY_CONTROL, force=0)
    return rid

# ── MAIN: EPISODIC CONSCIOUSNESS LOOP ───────────────────────
def main():
    global brain

    # Connect PyBullet
    p.connect(p.GUI, options="--width=1280 --height=720")
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    plane = p.loadURDF("plane.urdf")
    p.changeVisualShape(plane, -1, rgbaColor=[0.04, 0.04, 0.07, 1])

    # Persistent knowledge — survives across all lives
    T_global = fresh_model()[0]
    P_global = fresh_model()[1]
    D_global = fresh_danger()

    episode  = 0
    best     = 0.0

    print("\n== CARL Phase 9: Active Inference (Tabula Rasa) ==")
    print("No goals. No thresholds. Pure surprise minimization.\n")

    while True:
        episode += 1
        brain["episode"] = episode
        p.setGravity(0, 0, -9.81)

        rid = spawn_robot()
        for _ in range(10):
            p.stepSimulation()
        time.sleep(0.3)

        # Inherit accumulated knowledge
        T, P, D = T_global.copy(), P_global.copy(), D_global.copy()
        xk       = get_state(rid)[0]
        uk       = 0.0
        s_ema        = 0.0
        s_slow_ema   = 0.01   # slow baseline for novelty detection
        novelty_boost = 0.0
        t0      = time.time()
        consc   = consciousness(P)
        curio   = curiosity(P)

        print(f"[Life {episode:3d}]  Consciousness:{consc*100:5.1f}%  "
              f"Curiosity:{curio*100:5.1f}%")

        for _ in range(500_000):
            xn, z_height = get_state(rid)

            # 1. Update world model from lived experience
            T, P, surprise = rls_update(T, P, xk, uk, xn)
            s_ema      = 0.10 * surprise + 0.90 * s_ema
            s_slow_ema = 0.01 * surprise + 0.99 * s_slow_ema

            # Novelty detection: surprise >> baseline => robot is confused => explore!
            novelty_ratio  = s_ema / (s_slow_ema + 1e-6)
            novelty_boost  = float(np.clip((novelty_ratio - 2.0) * 0.3, 0.0, 0.5))

            # 2. Update self-discovered danger map
            D = danger_update(D, xn[2], xn[3], s_ema)

            consc = consciousness(P)
            curio = curiosity(P, novelty_boost)

            # 3. Action selection: curious OR exploit
            if curio > np.random.random():
                un   = float(np.random.choice(ACTIONS))
                mode = "CURIOUS"
            else:
                un   = pick_action(xn, T, D)
                dl   = danger_at(D, xn[2], xn[3])
                mode = "FEARFUL" if dl > 0.25 else "EXPLOITING"

            apply_torque(rid, un)
            xk, uk = xn, un

            # 4. Update dashboard
            dmax = float(np.max(D)) + 1e-6
            brain.update({
                "survival_time": round(time.time() - t0, 2),
                "best_survival" : round(best, 2),
                "mode"          : mode,
                "pitch"         : float(xn[2]),
                "surprise"      : float(s_ema),
                "curiosity"     : float(curio),
                "consciousness" : float(consc),
                "danger_level"  : float(danger_at(D, xn[2], xn[3])),
                "action"        : float(un),
                "danger_grid"   : (D / dmax).flatten().round(3).tolist(),
            })

            p.resetDebugVisualizerCamera(1.2, 45, -15, [xn[0], 0, 0.4])
            p.stepSimulation()
            time.sleep(DT)

            # 5. Death: use BASE HEIGHT as primary detector
            # When flat on ground, z drops from 0.08 to ~0.03
            # Euler pitch wraps to 0 (gimbal lock) — cannot be trusted alone
            if z_height < 0.05 or abs(xn[2]) > 0.65:
                sv   = time.time() - t0
                best = max(best, sv)
                brain["episode_history"].append(round(sv, 2))
                brain["best_survival"] = round(best, 2)
                brain["mode"]          = "DEAD"

                # Burn death trauma into danger memory
                # rate=0.3 (not 0.6) to avoid catastrophic overwrite of good knowledge
                for _ in range(4):   # 4 burns not 8
                    D = danger_update(D, xn[2], xn[3], 10.0, rate=0.3)

                # Save knowledge globally
                T_global, P_global, D_global = T.copy(), P.copy(), D.copy()

                print(f"         -> DIED {sv:.2f}s | Best:{best:.2f}s | "
                      f"Consc:{consc*100:.0f}%")

                p.removeBody(rid)
                time.sleep(1.5)
                break


if __name__ == "__main__":
    main()
