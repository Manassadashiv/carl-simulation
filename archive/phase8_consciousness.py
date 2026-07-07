"""
CARL Phase 8: The Consciousness Architecture
- Tabula Rasa: No hardcoded physics. Starts with identity matrices.
- Motor Babbling: Experiments with random torques to learn its own body.
- Episodic Rebirth: Falls, learns from failure, reboots, tries again.
- Streams live "thought state" to the Web Dashboard via WebSocket.
"""

import pybullet as p
import pybullet_data
import time
import numpy as np
import scipy.linalg
import asyncio
import websockets
import json
import threading

# ─────────────────────────────────────────────
#  SHARED DASHBOARD STATE
# ─────────────────────────────────────────────
brain_state = {
    "episode": 0,
    "survival_time": 0.0,
    "best_survival": 0.0,
    "mode": "BOOTING",
    "pitch": 0.0,
    "error": 0.0,
    "p_fail": 0.0,
    "mpc_active": False,
    "consciousness": 0.0,      # 0-1: how confident is the internal model
    "babble_energy": 0.0,      # random torque being injected
    "episode_history": [],     # survival times per episode
}

# ─────────────────────────────────────────────
#  WEBSOCKET SERVER
# ─────────────────────────────────────────────
connected_clients = set()

async def ws_handler(websocket):
    connected_clients.add(websocket)
    try:
        while True:
            await websocket.send(json.dumps(brain_state))
            await asyncio.sleep(1.0 / 60.0)
    except websockets.exceptions.ConnectionClosed:
        connected_clients.discard(websocket)

async def main_ws():
    print("[WebSocket] Server started on ws://localhost:8765")
    async with websockets.serve(ws_handler, "localhost", 8765):
        await asyncio.Future()

def run_ws_server():
    asyncio.run(main_ws())

ws_thread = threading.Thread(target=run_ws_server, daemon=True)
ws_thread.start()

# ─────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────
dt = 1.0 / 240.0
g = 9.81

# LQR cost matrices (for control policy once model is known)
Q_lqr = np.diag([2.0, 2.0, 50.0, 5.0])
R_lqr = np.array([[1.0]])

theta_max  = 0.5
e_max      = 0.05
w1, w2, w3, w4 = 1.0, 1.0, 0.5, 2.0

# ─────────────────────────────────────────────
#  TABULA RASA: starts with ZERO knowledge
#  Theta = [A_d | B_d]^T  (5x4 matrix)
# ─────────────────────────────────────────────
def fresh_brain():
    """Return a completely blank internal model."""
    Theta = np.zeros((5, 4))          # Identity dynamics + zero input
    Theta[0:4, :] = np.eye(4)        # Start with "nothing changes" prior
    P_rls = 1000.0 * np.eye(5)       # Very high uncertainty at birth
    return Theta, P_rls

# ─────────────────────────────────────────────
#  RLS UPDATE (The Mechanic)
# ─────────────────────────────────────────────
lambda_forget = 0.998

def rls_update(Theta, P_rls, Phi_k, x_next):
    P_Phi  = P_rls @ Phi_k
    L_k    = P_Phi / (lambda_forget + (Phi_k.T @ P_Phi)[0, 0])
    e_k    = x_next - (Theta.T @ Phi_k).flatten()
    Theta  = Theta + L_k @ e_k.reshape(1, 4)
    P_rls  = (P_rls - L_k @ Phi_k.T @ P_rls) / lambda_forget
    return Theta, P_rls, e_k

# ─────────────────────────────────────────────
#  CONSCIOUSNESS METRIC
#  How "certain" is the internal model?
#  Measured as confidence from RLS covariance trace
# ─────────────────────────────────────────────
P0_trace = 1000.0 * 5   # Initial trace: 5000

def compute_consciousness(P_rls):
    trace = np.trace(P_rls)
    raw   = 1.0 - min(trace / P0_trace, 1.0)
    return float(np.clip(raw, 0.0, 1.0))

# ─────────────────────────────────────────────
#  DERIVE LQR FROM LEARNED MODEL
# ─────────────────────────────────────────────
def derive_lqr(Theta):
    try:
        Ad = Theta[0:4, :].T
        Bd = Theta[4, :].reshape(4, 1)
        P  = scipy.linalg.solve_discrete_are(Ad, Bd, Q_lqr, R_lqr)
        K  = np.linalg.inv(R_lqr + Bd.T @ P @ Bd) @ (Bd.T @ P @ Ad)
        return K[0]
    except Exception:
        return np.zeros(4)   # No control until model is good enough

# ─────────────────────────────────────────────
#  GUARDIAN
# ─────────────────────────────────────────────
def guardian(pitch, dpitch, error_pitch):
    H = (w1 * abs(pitch)  / theta_max) + \
        (w2 * abs(dpitch) / 2.0) + \
        (w4 * abs(error_pitch) / e_max)
    return 1.0 / (1.0 + np.exp(-5.0 * (H - 1.2)))

# ─────────────────────────────────────────────
#  OPTIMAL MPC (Daughter Minds)
# ─────────────────────────────────────────────
def optimal_mpc(x, Theta):
    actions  = [-8.0, -5.0, -2.0, 0.0, 2.0, 5.0, 8.0]
    Ad       = Theta[0:4, :].T
    Bd       = Theta[4, :]
    best_u, best_cost = 0.0, float('inf')
    for u in actions:
        xs, cost = x.copy(), 0.0
        for _ in range(50):
            xs    = Ad @ xs + Bd * u
            cost += xs[2]**2 + 0.5 * xs[3]**2 + 0.05 * u**2
            if abs(xs[2]) > 0.5:
                cost += 1000.0; break
        if cost < best_cost:
            best_cost, best_u = cost, u
    return best_u

# ─────────────────────────────────────────────
#  PYBULLET HELPERS
# ─────────────────────────────────────────────
def get_state(robot_id):
    pos, quat   = p.getBasePositionAndOrientation(robot_id)
    vel, ang_vel = p.getBaseVelocity(robot_id)
    euler       = p.getEulerFromQuaternion(quat)
    return np.array([pos[0], vel[0], euler[1], ang_vel[1]]), pos[1]

def apply_torque(robot_id, torque):
    t = float(np.clip(torque, -8.0, 8.0))
    p.setJointMotorControl2(robot_id, 0, p.TORQUE_CONTROL, force=t)
    p.setJointMotorControl2(robot_id, 1, p.TORQUE_CONTROL, force=t)

def spawn_robot():
    start_pos  = [0, 0, 0.08]
    start_orn  = p.getQuaternionFromEuler([0, 0.02, 0])
    robot_id   = p.loadURDF("carl.urdf", start_pos, start_orn)
    p.setJointMotorControl2(robot_id, 2, p.POSITION_CONTROL, targetPosition=0, force=25.0)
    p.setJointMotorControl2(robot_id, 0, p.VELOCITY_CONTROL, force=0)
    p.setJointMotorControl2(robot_id, 1, p.VELOCITY_CONTROL, force=0)
    return robot_id

# ─────────────────────────────────────────────
#  MAIN: EPISODIC CONSCIOUSNESS LOOP
# ─────────────────────────────────────────────
def main():
    global brain_state

    # PyBullet setup
    physicsClient = p.connect(p.GUI, options="--width=1280 --height=720")
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    # Preload textures
    planeId = p.loadURDF("plane.urdf")
    p.changeVisualShape(planeId, -1, rgbaColor=[0.04, 0.04, 0.07, 1])

    # ── PERSISTENT KNOWLEDGE: survives across episodes ──
    Theta_global, P_rls_global = fresh_brain()
    episode = 0
    best_survival = 0.0

    print("\n========================================")
    print("  CARL — Consciousness Architecture")
    print("  Starting Tabula Rasa: ZERO knowledge")
    print("========================================\n")

    while True:
        episode += 1
        brain_state["episode"] = episode

        # Reset physics world
        p.setGravity(0, 0, -9.81)
        robot_id = spawn_robot()
        time.sleep(0.5)

        x_k, y_pos = get_state(robot_id)
        u_k = 0.0

        # Carry forward global knowledge (persistent learning!)
        Theta = Theta_global.copy()
        P_rls = P_rls_global.copy()

        e_filtered = np.zeros(4)
        alpha_ema  = 0.05

        mpc_active = False
        episode_start = time.time()

        # ── Phase A: MOTOR BABBLING ──────────────────────
        # Robot intentionally explores random torques for 2 seconds
        # This rapidly boots the RLS model with real physics data
        babble_steps  = int(2.0 / dt)  # 480 steps
        babble_torque = 0.0

        brain_state["mode"] = "MOTOR BABBLING"
        print(f"[Episode {episode}] Motor Babbling...")

        for step in range(babble_steps):
            x_next, _ = get_state(robot_id)

            # Random exploratory torque (changes every 0.5 seconds)
            if step % 120 == 0:
                babble_torque = np.random.uniform(-3.0, 3.0)

            Phi_k = np.append(x_k, u_k).reshape(5, 1)
            Theta, P_rls, e_k = rls_update(Theta, P_rls, Phi_k, x_next)
            e_filtered = alpha_ema * e_k + (1 - alpha_ema) * e_filtered

            apply_torque(robot_id, babble_torque)
            x_k = x_next
            u_k = babble_torque

            brain_state["babble_energy"]  = float(babble_torque)
            brain_state["consciousness"]  = compute_consciousness(P_rls)
            brain_state["pitch"]          = float(x_next[2])

            p.resetDebugVisualizerCamera(1.2, 45, -15, [x_k[0], 0, 0.4])
            p.stepSimulation()
            time.sleep(dt)

            # If it falls during babbling, don't abort — carry on and learn
            if abs(x_next[2]) > 0.8:
                break

        # ── Phase B: DERIVE POLICY FROM LEARNED MODEL ────
        brain_state["mode"] = "DERIVING POLICY"
        K = derive_lqr(Theta)
        consciousness = compute_consciousness(P_rls)
        print(f"[Episode {episode}] Consciousness: {consciousness*100:.1f}% — Attempting to balance...")

        # Reset robot for balancing attempt
        p.removeBody(robot_id)
        robot_id = spawn_robot()
        time.sleep(0.3)

        x_k, _ = get_state(robot_id)
        u_k    = 0.0

        # ── Phase C: BALANCING ───────────────────────────
        brain_state["mode"] = "BALANCING"
        phase_start = time.time()
        survived_steps = 0

        for step in range(500000):
            x_next, y_pos = get_state(robot_id)

            Phi_k = np.append(x_k, u_k).reshape(5, 1)
            Theta, P_rls, e_k = rls_update(Theta, P_rls, Phi_k, x_next)
            e_filtered = alpha_ema * e_k + (1 - alpha_ema) * e_filtered

            p_fail = guardian(x_next[2], x_next[3], e_filtered[2])

            if p_fail > 0.70:
                mpc_active = True
                brain_state["mode"] = "DAUGHTER MINDS"
            elif p_fail < 0.20:
                mpc_active = False
                brain_state["mode"] = "BALANCING"

            if mpc_active:
                u_next = optimal_mpc(x_next, Theta)
            else:
                u_next = -float(np.dot(K, x_next))

            apply_torque(robot_id, u_next)
            x_k = x_next
            u_k = u_next
            survived_steps += 1

            # Update dashboard
            survival_time = time.time() - phase_start
            brain_state.update({
                "survival_time":  round(survival_time, 2),
                "best_survival":  round(max(best_survival, survival_time), 2),
                "pitch":          float(x_next[2]),
                "error":          float(e_filtered[2]),
                "p_fail":         float(p_fail),
                "mpc_active":     bool(mpc_active),
                "consciousness":  compute_consciousness(P_rls),
                "babble_energy":  0.0,
            })

            p.resetDebugVisualizerCamera(1.2, 45, -15, [x_k[0], 0, 0.4])
            p.stepSimulation()
            time.sleep(dt)

            # DEATH CONDITION
            if abs(x_next[2]) > 0.6:
                survival_time = time.time() - phase_start
                best_survival = max(best_survival, survival_time)
                brain_state["episode_history"].append(round(survival_time, 2))

                print(f"[Episode {episode}] FELL after {survival_time:.2f}s  |  Best: {best_survival:.2f}s  |  Consciousness: {compute_consciousness(P_rls)*100:.1f}%")

                # Persist knowledge to global model
                Theta_global = Theta.copy()
                P_rls_global = P_rls.copy()

                p.removeBody(robot_id)
                time.sleep(1.0)   # Brief pause before next life
                break

if __name__ == "__main__":
    main()
