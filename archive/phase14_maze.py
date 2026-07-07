import traceback
# -*- coding: utf-8 -*-
# CARL Phase 14: THE MAZE
# ============================================================
# "The biological brain finally has something worthy of it."
#
# Everything from Phase 13 is preserved intact:
#   Neuromodulators · Hebbian · Predictive Coding
#   Homeostasis · Sleep Architecture · Social Cognition
#   Allostatic Load · Legacy Ghosts · Collective Mourning
#
# What changes: the WORLD.
#
# Instead of rolling 2m on a flat plane, CARL must now:
#   1. Navigate a procedurally generated 3D maze
#   2. Find a MOVING goal that relocates every 200 episodes
#   3. Survive walls — contact = trauma burn, not instant death
#   4. Build a 2D spatial cognitive map (not just 1D terrain)
#   5. Navigate junctions — left/right/straight decisions
#      that the daughter minds now have genuine stakes in
#
# Now grief, ancestral memory, daughter-mind deliberation,
# and allostatic aging are VISIBLE to any observer.
# A wrong turn at a junction = a death with a clear cause.
# A ghost at a dead end = a warning any human can read.
# ============================================================

import sys, pybullet as p, pybullet_data, time, numpy as np
import asyncio, websockets, json, threading, math, random
from astar import astar_path
import time as _time
from collections import deque

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ── CONSTANTS ─────────────────────────────────────────────────
DT        = 1.0 / 240.0
ACTIONS   = [-8., -5., -2., -1., 0., 1., 2., 5., 8.]
HORIZON   = 40
DRES      = 20
SDIM      = 5          # NEVER CHANGE — [x, vx, pitch, pitch_vel, neck]
P0_WM     = 500.0  * (SDIM + 1)
P0_LTM    = 2000.0 * (SDIM + 1)
N_BODIES      = 2
STEER_ACTIONS = [-2., 0., 2.]  # differential steer torques (L-R diff)
REDUCED_ACTIONS = [-8., -3., 0., 3., 8.] # for heavy O(N^2) deliberation
STEER_GAIN    = 6.0            # yaw rate per unit steer (rad/s, tune after first run)
WALL_H        = 0.35           # wall height in metres
WALL_T    = 0.08       # wall thickness

# ── MAZE LAYOUT ───────────────────────────────────────────────
# Maze is defined in a 2D grid coordinate system.
# Grid cell = 1.2m x 1.2m. Origin at (0,0).
# Robots spawn at grid (0,0), goal starts at grid (4,2).
# Walls defined as (x1,y1,x2,y2) in grid coords → converted to metres.

MAZE_CELL = 1.2   # metres per grid cell

# Outer boundary + internal walls
# Format: (gx1, gy1, gx2, gy2) in grid units
MAZE_WALLS_GRID = [
    # Outer boundaries
    (0,0, 6,0), (0,5, 6,5), (0,0, 0,5), (6,0, 6,5),
    # Internal walls (guaranteed open path)
    (1,0, 1,1),
    (0,2, 2,2),
    (1,3, 1,4),
    (2,1, 2,4),
    (3,0, 3,2),
    (2,4, 4,4),
    (4,1, 4,3),
    (3,3, 5,3),
    (5,1, 6,1),
]

# ── GOAL POSITIONS (cycle every 200 episodes) ─────────────────
# Centers of cells: (ix*1.2 + 0.6, iy*1.2 + 0.6)
GOAL_POSITIONS = [
    (5.4, 5.4),   # Cell (4, 4)
    (1.8, 5.4),   # Cell (1, 4)
    (6.6, 0.6),   # Cell (5, 0)
    (4.2, 1.8),   # Cell (3, 1)
    (0.6, 5.4),   # Cell (0, 4)
]

# ── NEUROMODULATOR BASELINES ───────────────────────────────────
NM_BASELINE = {"DA": 0.5, "SHT": 0.6, "NE": 0.2, "ACh": 0.4}

# ── HOMEOSTATIC SETPOINTS ──────────────────────────────────────
HOMEOSTATIC_SETPOINTS = {
    "pitch": 0.0, "velocity": 0.0, "arousal": 0.3,
    "fatigue": 0.0, "curiosity_drive": 0.4, "social_comfort": 0.7,
}

# ── DASHBOARD STATE ───────────────────────────────────────────
brain = {
    "episode": 0, "best_survival": 0.0, "mode": "BOOTING",
    "DA": 0.5, "5HT": 0.6, "NE": 0.2, "ACh": 0.4,
    "allostatic_load": 0.0, "homeostatic_error": 0.0,
    "social_comfort": 1.0, "alive_count": N_BODIES,
    "hebbian_strength": 0.0, "prediction_error": 0.5,
    "sleep_phase": "AWAKE",
    "nrem1_count": 0, "nrem3_count": 0, "rem_count": 0,
    "survivals": [0.]*N_BODIES, "pitches": [0.]*N_BODIES,
    "distances": [99.]*N_BODIES, "alive_flags": [True]*N_BODIES,
    "goal_x": 4.8, "goal_y": 2.4, "goal_episode": 0,
    "ghost_count": 0, "mourning_events": 0,
    "target_reached_count": 0, "best_dist_ever": 99.,
    "curriculum_stage": 1,
    "junction_decisions": 0,
    "danger_grid": [0.0]*400,
    "cognitive_map": [0.0]*625,   # 25x25 spatial map
    "cm_scent": [0.0]*625,
    "cm_phero": [0.0]*625,
    "wind_active": False, "slope_deg": 0.0, "quake_amp": 0.0,
    "episode_history": [],
    "surprise_wm": 0.0, "surprise_ltm": 0.0,
    "ltm_confidence": 0.0, "curiosity": 0.5,
    "danger_level": 0.0,
    "wm_confidence_A": 0.0, "wm_confidence_B": 0.0,
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

# ── TWO-SPEED MEMORY ──────────────────────────────────────────
def fresh_wm():
    T = np.zeros((SDIM+1, SDIM)); T[:SDIM] = np.eye(SDIM)
    return T, 500.0 * np.eye(SDIM+1)

def fresh_ltm():
    T = np.zeros((SDIM+1, SDIM))
    T[:SDIM, :] = np.eye(SDIM)
    P = np.eye(SDIM+1) * 1000.0
    import os
    if os.path.exists("checkpoint_ltm_T.npy") and os.path.exists("checkpoint_ltm_P.npy"):
        try:
            T = np.load("checkpoint_ltm_T.npy")
            P = np.load("checkpoint_ltm_P.npy")
            print("  [SUCCESS] Loaded Pre-trained Balance LTM from Phase 13!")
        except:
            pass
    return T, P

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

# ── NEUROMODULATOR SYSTEM ─────────────────────────────────────
class NeuromodulatorSystem:
    def __init__(self):
        self.DA  = NM_BASELINE["DA"]
        self.SHT = NM_BASELINE["SHT"]
        self.NE  = NM_BASELINE["NE"]
        self.ACh = NM_BASELINE["ACh"]

    def update(self, surprise, danger, dist_to_target,
               target_reached, sibling_died, allostatic_load):
        progress_signal = max(0., 1. - dist_to_target / 8.)
        if target_reached:
            self.DA = min(1.0, self.DA + 0.3)
        else:
            self.DA = 0.92 * self.DA + 0.08 * (0.3 + 0.7 * progress_signal)
        self.DA = float(np.clip(self.DA, 0.25, 1.0))

        threat_signal = max(danger, surprise * 0.5)
        if threat_signal > self.NE:
            self.NE = 0.80 * self.NE + 0.20 * threat_signal
        else:
            self.NE = 0.92 * self.NE + 0.08 * threat_signal
        self.NE = float(np.clip(self.NE, 0., 1.))

        safety_signal  = max(0., 1. - danger)
        ne_suppression = self.NE * 0.25
        self.SHT = 0.990 * self.SHT + 0.010 * safety_signal - ne_suppression * 0.005
        self.SHT = float(np.clip(self.SHT, 0.15, 1.))

        if sibling_died:
            self.ACh = min(1.0, self.ACh + 0.2)
        self.ACh = 0.99 * self.ACh + 0.01 * min(1., surprise * 2.)
        self.ACh = float(np.clip(self.ACh, 0.1, 1.))
        self.SHT = max(0.15, self.SHT - allostatic_load * 0.0005)

    def effective_learning_rate(self, base_lam=0.990):
        return float(np.clip(base_lam - (self.ACh - 0.4) * 0.008, 0.970, 0.999))

    def effective_horizon(self, base_horizon=40):
        return max(10, int(base_horizon * (0.5 + self.SHT * 1.0)))

    def effective_dopamine_weight(self, base=0.6):
        return float(np.clip(base * (0.5 + self.DA), 0.3, 1.2))

    def effective_danger_sensitivity(self, base_danger):
        return float(base_danger * (1.0 + self.NE * 1.5))

# ── HEBBIAN ASSOCIATOR ────────────────────────────────────────
class HebbianAssociator:
    def __init__(self, capacity=500):
        self.associations = deque(maxlen=capacity)
        self.strength_map = {}

    def fire(self, pitch, vel, action, surprise, threshold=0.3):
        if surprise < threshold:
            return
        pb = int(np.clip((pitch+.65)/1.30*10, 0, 9))
        vb = int(np.clip((vel+3.)/6.*10, 0, 9))
        ab = int(np.clip((action+8.)/16.*5, 0, 4))
        key = (pb, vb, ab)
        self.strength_map[key] = min(1.0, self.strength_map.get(key, 0.) + surprise * 0.1)

    def wire_together_decay(self):
        for key in list(self.strength_map.keys()):
            self.strength_map[key] *= 0.9999
            if self.strength_map[key] < 0.01:
                del self.strength_map[key]

    def association_cost(self, pitch, vel, action):
        pb = int(np.clip((pitch+.65)/1.30*10, 0, 9))
        vb = int(np.clip((vel+3.)/6.*10, 0, 9))
        ab = int(np.clip((action+8.)/16.*5, 0, 4))
        return self.strength_map.get((pb, vb, ab), 0.) * 2.0

    def total_strength(self):
        if not self.strength_map: return 0.
        return float(np.mean(list(self.strength_map.values())))

# ── PREDICTIVE CODER ──────────────────────────────────────────
class PredictiveCoder:
    def __init__(self):
        self.prediction = None
        self.error_history = deque(maxlen=100)
        self.directional_bias = np.zeros(SDIM)

    def predict(self, x, u, T_wm, T_ltm, P_wm):
        alpha = wm_conf(P_wm)
        T_use = alpha * T_wm + (1. - alpha) * T_ltm
        Phi   = np.append(x, float(u)).reshape(SDIM+1, 1)
        self.prediction = (T_use.T @ Phi).flatten()
        return self.prediction

    def compute_error(self, x_actual):
        if self.prediction is None:
            return 0., np.zeros(SDIM)
        error_vec = x_actual - self.prediction
        magnitude = float(np.linalg.norm(error_vec))
        self.error_history.append(magnitude)
        self.directional_bias = 0.99 * self.directional_bias + 0.01 * np.abs(error_vec)
        return magnitude, error_vec

    def precision_weight(self):
        if len(self.error_history) < 10:
            return 0.5
        return float(np.clip(1. / (1. + np.var(list(self.error_history)) * 10.), 0.1, 0.9))

# ── HOMEOSTATIC REGULATOR ─────────────────────────────────────
class HomeostaticRegulator:
    def __init__(self):
        self.variables = {k: v for k, v in HOMEOSTATIC_SETPOINTS.items()}
        self.allostatic_load = 0.0

    def update(self, pitch, velocity, nm_system, fatigue,
               alive_siblings, dist_to_target):
        self.variables.update({
            "pitch": float(pitch), "velocity": float(velocity),
            "arousal": float(nm_system.NE), "fatigue": float(fatigue),
            "curiosity_drive": float(nm_system.ACh),
            "social_comfort": float(alive_siblings / N_BODIES),
        })
        total_error = 0.
        for key, setpoint in HOMEOSTATIC_SETPOINTS.items():
            if key == "velocity" and dist_to_target > 0.3:
                setpoint = 0.3
            total_error += abs(self.variables[key] - setpoint)
        self.allostatic_load = min(1., self.allostatic_load + total_error * 0.0001)
        return total_error

    def correction_signal(self, pitch):
        return float(np.clip(-(pitch - HOMEOSTATIC_SETPOINTS["pitch"]) * 2.0, -2., 2.))

# ── SOCIAL COGNITION ──────────────────────────────────────────
class SocialCognition:
    def __init__(self, body_idx, n_bodies):
        self.idx = body_idx
        self.n_bodies = n_bodies
        self.social_comfort = 1.0

    def update(self, brains, my_x):
        alive = [(i, b) for i, b in enumerate(brains)
                 if b["alive"] and i != self.idx and b["xk"] is not None]
        if alive:
            distances = [abs(b["xk"][0] - my_x) for (_, b) in alive]
            self.social_comfort = float(np.clip(1. - np.mean(distances) / 5., 0., 1.))
        else:
            self.social_comfort = 0.0
        return self.social_comfort

    def social_risk_modifier(self):
        return float(1.0 + (self.social_comfort - 0.5) * 0.3)

# ── 2D SPATIAL COGNITIVE MAP ──────────────────────────────────
# This is new in Phase 14.
# CARL now builds a 2D map of the maze — not just 1D terrain trust.
# 25x25 grid covering the maze footprint.
# Each cell stores (danger, trust, ghost_intensity).
# This is what makes junction decisions meaningful.

MAP_RES   = 25
MAP_XMIN  = -0.5
MAP_XMAX  =  7.7
MAP_YMIN  = -0.5
MAP_YMAX  =  6.5

def _map_cell(xw, yw):
    i = int(np.clip((xw - MAP_XMIN) / (MAP_XMAX - MAP_XMIN) * MAP_RES, 0, MAP_RES-1))
    j = int(np.clip((yw - MAP_YMIN) / (MAP_YMAX - MAP_YMIN) * MAP_RES, 0, MAP_RES-1))
    return i, j

def fresh_cognitive_map():
    # (danger, trust, ghost) per cell
    return np.zeros((MAP_RES, MAP_RES, 5))


def diffuse_scent(CM, goal_pos):
    gi, gj = _map_cell(goal_pos[0], goal_pos[1])
    scent = CM[:, :, 3]
    danger = CM[:, :, 0]
    
    # Vectorised Laplacian
    ns = np.zeros_like(scent)
    ns[1:-1, 1:-1] = (scent[:-2, 1:-1] + scent[2:, 1:-1] + scent[1:-1, :-2] + scent[1:-1, 2:])
    
    new_scent = scent.copy()
    new_scent[1:-1, 1:-1] += 0.25 * (ns[1:-1, 1:-1] - 4*scent[1:-1, 1:-1])
    
    # Apply danger block (pruning dead ends)
    new_scent[danger > 0.5] = 0.0
    
    # Decay
    new_scent *= 0.999
    
    # Clamp goal (The Oat Flake)
    new_scent[gi, gj] = 1.0
    
    CM[:, :, 3] = new_scent
    CM[:, :, 4] *= 0.995 # decay agent pheromone
    return CM

def get_olfactory_reward(CM, xw, yw):
    # Bilinear interpolation of the grid so the reward is continuous!
    x_pct = (xw - MAP_XMIN) / (MAP_XMAX - MAP_XMIN) * MAP_RES
    y_pct = (yw - MAP_YMIN) / (MAP_YMAX - MAP_YMIN) * MAP_RES
    
    x0 = int(np.clip(math.floor(x_pct), 0, MAP_RES-1))
    x1 = int(np.clip(math.ceil(x_pct), 0, MAP_RES-1))
    y0 = int(np.clip(math.floor(y_pct), 0, MAP_RES-1))
    y1 = int(np.clip(math.ceil(y_pct), 0, MAP_RES-1))
    
    dx = x_pct - x0
    dy = y_pct - y0
    
    def b_interp(c):
        c00 = CM[x0, y0, c]
        c10 = CM[x1, y0, c]
        c01 = CM[x0, y1, c]
        c11 = CM[x1, y1, c]
        return c00*(1-dx)*(1-dy) + c10*dx*(1-dy) + c01*(1-dx)*dy + c11*dx*dy

    goal_s = b_interp(3)
    agent_p = b_interp(4)

    # Scale kept in same ballpark as danger costs (~0-2.5) to avoid domination
    return goal_s * 2.0 + agent_p * 0.5

def cognitive_map_update(CM, xw, yw, danger, trust_delta, ghost=0.):
    i, j = _map_cell(xw, yw)
    CM[i, j, 0] = 0.9 * CM[i, j, 0] + 0.1 * danger
    CM[i, j, 1] = np.clip(CM[i, j, 1] + trust_delta, 0., 1.)
    if ghost > 0.:
        CM[i, j, 2] = min(1., CM[i, j, 2] + ghost)
    return CM

def cognitive_map_danger(CM, xw, yw):
    i, j = _map_cell(xw, yw)
    return float(CM[i, j, 0])

def cognitive_map_trust(CM, xw, yw):
    i, j = _map_cell(xw, yw)
    return float(CM[i, j, 1])

# ── DANGER MAP (pitch/vel space, preserved from Phase 13) ─────
def fresh_danger(): return np.zeros((DRES, DRES))

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

# ── LEGACY GHOST MAP ──────────────────────────────────────────
def fresh_legacy_2d():
    return np.zeros((MAP_RES, MAP_RES))

def legacy_write_2d(L2, xw, yw, intensity=1.0, goal_xw=None, goal_yw=None):
    if goal_xw is not None:
        dist_to_goal = math.sqrt((xw-goal_xw)**2 + (yw-goal_yw)**2)
        if dist_to_goal < 0.4:
            intensity *= 0.15   # brave death near goal = faint ghost
    i, j = _map_cell(xw, yw)
    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            ii = int(np.clip(i+di, 0, MAP_RES-1))
            jj = int(np.clip(j+dj, 0, MAP_RES-1))
            w  = 1.0 if (di==0 and dj==0) else 0.4
            L2[ii, jj] = min(1., L2[ii, jj] + intensity * w)
    return L2

# ── SLEEP ARCHITECTURE ────────────────────────────────────────
def biological_sleep(T_ltm, P_ltm, D, buf, near_miss_buf, hebbian, allostatic_load):
    print("  [SLEEP PHASE 1 — NREM-1] Light consolidation...")
    if buf:
        top_recent = sorted(buf, key=lambda m: m[3], reverse=True)[:20]
        for (xk_m, uk_m, xn_m, surprise) in top_recent:
            T_ltm, P_ltm, _ = rls_update(T_ltm, P_ltm, xk_m, uk_m, xn_m,
                                          lam=0.9999, p_floor=0.005)

    print("  [SLEEP PHASE 2 — NREM-3] Deep trauma consolidation...")
    if buf:
        for (xk_m, uk_m, xn_m, surprise) in sorted(buf, key=lambda m: m[3], reverse=True)[:30]:
            for _ in range(3):
                D = danger_update(D, xn_m[2], xn_m[3], surprise, rate=0.25)
    hebbian.wire_together_decay()

    print("  [SLEEP PHASE 3 — REM] Creative dream recombination...")
    if near_miss_buf:
        for (xk_m, uk_m, xn_m, surprise) in sorted(near_miss_buf,
                                                     key=lambda m: m[3])[:15]:
            D = danger_update(D, xn_m[2], xn_m[3], max(0., surprise * 0.2), rate=0.05)
            T_ltm, P_ltm, _ = rls_update(T_ltm, P_ltm, xk_m, uk_m, xn_m,
                                          lam=0.9999, p_floor=0.005)

    return T_ltm, P_ltm, D

# ── MAZE BUILDER ──────────────────────────────────────────────
def preseed_cm_walls(CM):
    """
    Pre-seed the cognitive map danger channel with wall geometry so that
    A* can avoid walls from episode 1 (before any collision learning).
    """
    for (gx1, gy1, gx2, gy2) in MAZE_WALLS_GRID:
        x1, y1 = gx1 * MAZE_CELL, gy1 * MAZE_CELL
        x2, y2 = gx2 * MAZE_CELL, gy2 * MAZE_CELL
        # Sample wall segment in small steps and mark cells as high danger
        length = math.sqrt((x2-x1)**2 + (y2-y1)**2)
        n_pts  = max(2, int(length / 0.15))
        for k in range(n_pts + 1):
            t  = k / max(1, n_pts)
            xw = x1 + t * (x2 - x1)
            yw = y1 + t * (y2 - y1)
            # Mark a small neighbourhood of cells as dangerous
            for ddx in [-0.1, 0.0, 0.1]:
                for ddy in [-0.1, 0.0, 0.1]:
                    cognitive_map_update(CM, xw+ddx, yw+ddy,
                                         danger=0.9, trust_delta=0.)
    return CM


def build_maze():
    """
    Builds the maze from MAZE_WALLS_GRID.
    Returns list of wall body IDs.
    Each wall is a thin box collider with a cinematic dark material.
    """
    wall_ids = []
    for (gx1, gy1, gx2, gy2) in MAZE_WALLS_GRID:
        x1 = gx1 * MAZE_CELL
        y1 = gy1 * MAZE_CELL
        x2 = gx2 * MAZE_CELL
        y2 = gy2 * MAZE_CELL

        cx = (x1 + x2) / 2.
        cy = (y1 + y2) / 2.
        cz = WALL_H / 2.

        # Wall dimensions
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)

        if dx < 0.01:   # vertical wall
            half_ext = [WALL_T/2., max(dy/2., WALL_T/2.), WALL_H/2.]
        else:            # horizontal wall
            half_ext = [max(dx/2., WALL_T/2.), WALL_T/2., WALL_H/2.]

        col  = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_ext)
        vis  = p.createVisualShape(p.GEOM_BOX, halfExtents=half_ext,
                                   rgbaColor=[0.06, 0.07, 0.12, 1.0])
        wid  = p.createMultiBody(baseMass=0,
                                 baseCollisionShapeIndex=col,
                                 baseVisualShapeIndex=vis,
                                 basePosition=[cx, cy, cz])
        wall_ids.append(wid)

        # Glowing edge line on top of each wall
        p.addUserDebugLine([x1, y1, WALL_H], [x2, y2, WALL_H],
                           [0.2, 0.5, 1.0], 1)

    return wall_ids

def draw_maze_floor_grid():
    """Cinematic cyan grid on the floor."""
    for gx in range(-1, 7):
        x = gx * MAZE_CELL
        p.addUserDebugLine([x, MAP_YMIN, 0.005],
                           [x, MAP_YMAX, 0.005],
                           [0.05, 0.08, 0.20], 1)
    for gy in range(-1, 5):
        y = gy * MAZE_CELL
        p.addUserDebugLine([MAP_XMIN, y, 0.005],
                           [MAP_XMAX, y, 0.005],
                           [0.05, 0.08, 0.20], 1)

def spawn_goal(gx, gy, episode):
    """Spawn the goal sphere with beacon rings."""
    x, y = gx, gy
    tgt = p.createVisualShape(p.GEOM_SPHERE, radius=0.18,
                              rgbaColor=[1., 0.15, 0.5, 0.95])
    p.createMultiBody(baseMass=0, baseVisualShapeIndex=tgt,
                      basePosition=[x, y, 0.18])
    for h in [0.05, 0.25, 0.50]:
        for angle in np.linspace(0, 2*np.pi, 12):
            p.addUserDebugLine(
                [x, y, h],
                [x + 0.35*math.cos(angle), y + 0.35*math.sin(angle), h],
                [1., 0.15, 0.5], 2)
    # Vertical beacon
    for z in np.linspace(0.05, 1.2, 8):
        p.addUserDebugLine([x, y, z], [x, y, z+0.05], [1., 0.3, 0.6], 1)
    return (x, y)

# ── PYBULLET HELPERS ──────────────────────────────────────────
def get_state(rid):
    pos, quat    = p.getBasePositionAndOrientation(rid)
    vel, ang_vel = p.getBaseVelocity(rid)
    euler        = p.getEulerFromQuaternion(quat)
    neck         = p.getJointState(rid, 2)[0]
    state = np.array([pos[0], vel[0], euler[1], ang_vel[1], neck], dtype=float)
    return state, float(pos[1]), float(pos[2]), float(euler[0])

def get_yaw(rid):
    """Return heading angle (yaw) of robot in world frame."""
    _, quat = p.getBasePositionAndOrientation(rid)
    euler   = p.getEulerFromQuaternion(quat)
    return float(euler[2])

def get_pos_2d(rid):
    pos, _ = p.getBasePositionAndOrientation(rid)
    return float(pos[0]), float(pos[1])

def apply_torque(rid, u_fwd, u_steer=0.):
    """Differential drive: left = fwd+steer, right = fwd-steer."""
    fwd   = float(np.clip(u_fwd,   -8., 8.))
    steer = float(np.clip(u_steer, -3., 3.))
    p.setJointMotorControl2(rid, 0, p.TORQUE_CONTROL, force=fwd + steer)
    p.setJointMotorControl2(rid, 1, p.TORQUE_CONTROL, force=fwd - steer)

def spawn_robot(x_offset=0., y_offset=0., initial_pitch=0.):
    for attempt in range(3):
        try:
            rid = p.loadURDF("carl.urdf",
                             [x_offset, y_offset, 0.10],
                             p.getQuaternionFromEuler([0, initial_pitch, 0]))
            break
        except:
            time.sleep(0.5)
    else:
        raise RuntimeError("URDF load failed")
    p.setJointMotorControl2(rid, 2, p.POSITION_CONTROL,
                            targetPosition=0, force=12.)
    p.setJointMotorControl2(rid, 0, p.VELOCITY_CONTROL, force=0)
    p.setJointMotorControl2(rid, 1, p.VELOCITY_CONTROL, force=0)
    return rid

def pick_action_maze(x, pos_2d, yaw, T_wm, T_ltm, P_wm, D, CM,
                     nm, homeostatic, swarm_hebbian, hebbian,
                     predictive, social, goal_pos, grief=0.):
    """
    Full differential-steering action selection.
    Iterates over forward × steer action combinations.
    Position rollout uses heading (yaw) so spatial costs are 2D-correct.
    """
    alpha   = wm_conf(P_wm)
    T_use   = alpha * T_wm + (1. - alpha) * T_ltm
    Ad, Bd  = T_use[:SDIM,:].T, T_use[SDIM,:]
    dop_w   = nm.effective_dopamine_weight(base=0.6)
    horizon = nm.effective_horizon(base_horizon=HORIZON)
    ne_amp  = 1.0 + nm.NE * 1.5
    grief_w = 1.0 + grief * 0.5
    soc_mod = social.social_risk_modifier()
    gx, gy  = goal_pos
    best_u, best_s, best_F = 0., 0., float('inf')

    for u in ACTIONS:
        for s in STEER_ACTIONS:
            xs, F   = x.copy(), 0.
            xw, yw  = pos_2d
            heading = yaw
            for h in range(horizon):
                xs_n     = Ad @ xs + Bd * u
                conf     = max(alpha, 0.1) * (0.95**h)
                # Propagate heading and 2D position (s > 0 turns right, so negative yaw)
                heading -= s * STEER_GAIN * DT
                spd      = xs_n[1] * DT * 25.
                xw      += spd * math.cos(heading)
                yw      += spd * math.sin(heading)
                kin_danger = danger_at(D, xs_n[2], xs_n[3]) * ne_amp * grief_w
                spatial_d  = cognitive_map_danger(CM_global, xw, yw) * ne_amp * 0.5
                spatial_t  = 1. - cognitive_map_trust(CM, xw, yw)
                olfactory  = get_olfactory_reward(CM, xw, yw) * dop_w
                pred_err   = float(np.linalg.norm(xs_n - xs)) * 0.02
                hebb_cost  = (0.7 * swarm_hebbian.association_cost(xs_n[2], xs_n[3], u)
                            + 0.3 * hebbian.association_cost(xs_n[2], xs_n[3], u))
                homeo_corr = abs(homeostatic.correction_signal(xs_n[2])) * 0.1
                raw_cost   = (pred_err
                            + (kin_danger + spatial_d) * soc_mod
                            + 0.1 * spatial_t
                            - olfactory + hebb_cost * 0.1 + homeo_corr)
                F  += raw_cost * conf
                xs  = xs_n
            if F < best_F:
                best_F, best_u, best_s = F, u, s
    return best_u, best_s



def daughter_minds_maze(x, pos_2d, yaw, T_wm, T_ltm, P_wm, D, CM,
                        nm, homeostatic, swarm_hebbian, hebbian,
                        predictive, social, goal_pos, grief):
    alpha   = wm_conf(P_wm)
    T_use   = alpha * T_wm + (1. - alpha) * T_ltm
    Ad, Bd  = T_use[:SDIM,:].T, T_use[SDIM,:]
    ne_amp  = 1.0 + nm.NE * 1.5
    grief_w = 1.0 + grief * 0.5
    soc_mod = social.social_risk_modifier()
    gx, gy  = goal_pos
    strategies = [
        {"d_w": 2.0 + nm.NE*1.5,         "dop_w": 0.0,                            "name": "SAFE"},
        {"d_w": max(0.2,0.5-nm.DA*0.5),   "dop_w": nm.effective_dopamine_weight(1.0), "name": "BOLD"},
        {"d_w": 1.0,                       "dop_w": nm.effective_dopamine_weight(0.6), "name": "BALANCED"},
    ]
    
    proposals = []
    
    for strat in strategies:
        dop_w    = strat["dop_w"]
        danger_w = strat["d_w"] * ne_amp * grief_w * soc_mod
        strat_best_F, strat_best_u, strat_best_s = float('inf'), 0., 0.
        
        for u in ACTIONS:
            for s in STEER_ACTIONS:
                xs, F   = x.copy(), 0.
                xw, yw  = pos_2d
                heading = yaw
                # Leg 1
                for h in range(20):
                    xs_n     = Ad @ xs + Bd * u
                    conf     = max(alpha, 0.1) * (0.95**h)
                    heading -= s * STEER_GAIN * DT
                    spd      = xs_n[1] * DT * 25.
                    xw      += spd * math.cos(heading)
                    yw      += spd * math.sin(heading)
                    olf1 = get_olfactory_reward(CM, xw, yw)
                    kin_d    = danger_at(D, xs_n[2], xs_n[3]) * danger_w
                    sp_d     = cognitive_map_danger(CM_global, xw, yw) * danger_w * 0.5
                    F       += (kin_d + sp_d - dop_w*olf1
                               + abs(homeostatic.correction_signal(xs_n[2]))*0.1) * conf
                    xs = xs_n
                # Leg 2
                branch_best = float('inf')
                for u2 in ACTIONS:
                    xs2, F2    = xs.copy(), 0.
                    xw2, yw2   = xw, yw
                    h2_heading = heading
                    for h2 in range(10):
                        xs2_n      = Ad @ xs2 + Bd * u2
                        conf2      = max(alpha, 0.1) * (0.95**(15+h2))
                        spd2       = xs2_n[1] * DT * 25.
                        xw2       += spd2 * math.cos(h2_heading)
                        yw2       += spd2 * math.sin(h2_heading)
                        olf2 = get_olfactory_reward(CM, xw2, yw2)
                        F2        += (danger_at(D,xs2_n[2],xs2_n[3])*danger_w
                                     + cognitive_map_danger(CM_global,xw2,yw2)*danger_w*0.5
                                     - dop_w*olf2) * conf2
                        xs2 = xs2_n
                    branch_best = min(branch_best, F2)
                F += branch_best
                if F < strat_best_F:
                    strat_best_F, strat_best_u, strat_best_s = F, u, s
        proposals.append((strat_best_u, strat_best_s, strat["name"]))

    # BASELINE EVALUATION TO PREVENT LAZY 'SAFE' FROM ALWAYS WINNING
    base_dop_w = nm.effective_dopamine_weight(base=0.6)
    base_danger_w = 1.0 * ne_amp * grief_w * soc_mod
    best_F, best_u, best_s, best_name = float('inf'), 0., 0., "BALANCED"

    for (u, s, name) in proposals:
        xs, F   = x.copy(), 0.
        xw, yw  = pos_2d
        heading = yaw
        # Full 30 step evaluation of the proposed u and s
        # Wait, the proposal only gives u and s for Leg 1.
        # We must re-evaluate Leg 1, and then assume optimal Leg 2 under baseline.
        # To keep it fast, we just evaluate Leg 1 using baseline.
        for h in range(20):
            xs_n     = Ad @ xs + Bd * u
            conf     = max(alpha, 0.1) * (0.95**h)
            heading -= s * STEER_GAIN * DT
            spd      = xs_n[1] * DT * 25.
            xw      += spd * math.cos(heading)
            yw      += spd * math.sin(heading)
            olf1 = get_olfactory_reward(CM, xw, yw)
            kin_d    = danger_at(D, xs_n[2], xs_n[3]) * base_danger_w
            sp_d     = cognitive_map_danger(CM_global, xw, yw) * base_danger_w * 0.5
            F       += (kin_d + sp_d - base_dop_w*olf1
                       + abs(homeostatic.correction_signal(xs_n[2]))*0.1) * conf
            xs = xs_n
            
        # Add a quick Leg 2 baseline
        branch_best = float('inf')
        for u2 in ACTIONS:
            xs2, F2 = xs.copy(), 0.
            xw2, yw2 = xw, yw
            h2_heading = heading
            for h2 in range(10):
                xs2_n = Ad @ xs2 + Bd * u2
                conf2 = max(alpha, 0.1) * (0.95**(15+h2))
                spd2 = xs2_n[1] * DT * 25.
                xw2 += spd2 * math.cos(h2_heading)
                yw2 += spd2 * math.sin(h2_heading)
                olf2 = get_olfactory_reward(CM, xw2, yw2)
                F2 += (danger_at(D,xs2_n[2],xs2_n[3])*base_danger_w
                       + cognitive_map_danger(CM_global,xw2,yw2)*base_danger_w*0.5
                       - base_dop_w*olf2) * conf2
                xs2 = xs2_n
            branch_best = min(branch_best, F2)
        F += branch_best

        if F < best_F:
            best_F, best_u, best_s, best_name = F, u, s, name

    return best_u, best_s, best_name


def directed_explore_maze(x, pos_2d, yaw, T_wm, P_wm, D, CM, nm, goal_pos):
    """Curiosity-driven exploration: seek unexplored safe zones, with steering."""
    Ad, Bd = T_wm[:SDIM,:].T, T_wm[SDIM,:]
    if float(np.linalg.norm(Bd)) < 0.1:
        return float(np.random.choice(ACTIONS)), 0.
    ach_amp = 1. + nm.ACh * 0.5
    gx, gy  = goal_pos
    xw, yw  = pos_2d
    best_u, best_s, best_info = 0., 0., -float('inf')
    for u in REDUCED_ACTIONS:
        for s in STEER_ACTIONS:
            xs      = x.copy()
            xw2, yw2 = xw, yw
            heading  = yaw
            for _ in range(10):
                xs       = Ad @ xs + Bd * u
                heading -= s * STEER_GAIN * DT
                spd      = xs[1] * DT * 25.
                xw2     += spd * math.cos(heading)
                yw2     += spd * math.sin(heading)
            Phi_r  = np.append(xs, u).reshape(SDIM+1, 1)
            unc    = float((Phi_r.T @ P_wm @ Phi_r).squeeze())
            unexplored = 1. - cognitive_map_trust(CM, xw2, yw2)
            goal_pull  = max(0., 1. - math.sqrt((xw2-gx)**2+(yw2-gy)**2)/8.)
            info   = unc * (1. + unexplored) * ach_amp + goal_pull * 0.3
            if info > best_info:
                best_info, best_u, best_s = info, u, s
    return best_u, best_s

def spawn_robot_brain(T_inst, P_inst, body_idx):
    return {
        "T_wm"          : T_inst.copy(),
        "P_wm"          : P_inst.copy() + 0.1*np.eye(SDIM+1),
        "xk"            : None,
        "uk"            : 0.,
        "sk"            : 0.,      # steer action (differential)
        "yaw"           : 0.,      # heading angle in world frame
        "s_wm_ema"      : 0.,
        "s_slow"        : 0.01,
        "fatigue"       : 0.,
        "buf"           : [],
        "near_miss_buf" : [],
        "t0"            : time.time(),
        "alive"         : True,
        "sv"            : 0.,
        "rid"           : None,
        "grief"         : 0.,
        "nm"            : NeuromodulatorSystem(),
        "homeostatic"   : HomeostaticRegulator(),
        "hebbian"       : HebbianAssociator(),
        "predictive"    : PredictiveCoder(),
        "social"        : SocialCognition(body_idx, N_BODIES),
        "mode"          : "BALANCED",
        "was_at_target" : False,
        "pos_2d"        : (0., 0.),
        "wall_contacts" : 0,
    }

# ── CINEMATIC HUD ─────────────────────────────────────────────
def draw_hud(nm0, mode, best, avg_allo, stage, goal_pos, alive_count,
             junction_decisions, target_reached):
    gx, gy = goal_pos
    nm_str = (f"DA:{nm0.DA:.2f} NE:{nm0.NE:.2f} "
              f"SHT:{nm0.SHT:.2f} ACh:{nm0.ACh:.2f}")
    line1  = (f"STAGE {stage} | {mode} | Best:{best:.1f}s | "
              f"Alive:{alive_count}/{N_BODIES}")
    line2  = (f"{nm_str} | AlloLoad:{avg_allo:.3f} | "
              f"Goal:({gx:.1f},{gy:.1f}) | Reached:{target_reached}")
    p.addUserDebugText(line1, [0, -2, 0.8],
                       textColorRGB=[0, 1, 0.5], textSize=1.0)
    p.addUserDebugText(line2, [0, -2, 0.6],
                       textColorRGB=[0.4, 0.8, 1.0], textSize=0.85)

# ── MAIN ──────────────────────────────────────────────────────

# ── ASYNCHRONOUS PREFRONTAL CORTEX (TESLA FSD) ───────────────
brains = []          # global so pfc_worker thread can read it
pfc_active = True
_pfc_threads = []    # track so we can restart each episode

def pfc_worker(b_idx):
    """Background thread: A* pathfinding + steering decision at 20 Hz."""
    global brains, CM_global, goal_pos
    while pfc_active:
        try:
            if not brains or b_idx >= len(brains):
                time.sleep(0.1)
                continue
            b = brains[b_idx]
            if not b["alive"]:
                time.sleep(0.05)
                continue

            # 1. A* Pathfinding over the live Cognitive Map
            path = astar_path(CM_global, b["pos_2d"], goal_pos)
            b["astar_path"] = path

            # 2. Compute steering target from path
            if path and len(path) > 1:
                target_pt = path[1]
                dx = target_pt[0] - b["pos_2d"][0]
                dy = target_pt[1] - b["pos_2d"][1]
                ideal_yaw = math.atan2(dy, dx)
                yaw_err = (ideal_yaw - b["yaw"] + math.pi) % (2*math.pi) - math.pi
                b["target_s"] = -1.5 if yaw_err > 0.2 else (1.5 if yaw_err < -0.2 else 0.)
                b["target_u"] = 3.
                b["mode"]     = "AUTOPILOT"
            else:
                b["target_u"] = 0.
                b["target_s"] = 0.
                b["mode"]     = "SEARCHING"

        except Exception as e:
            print(f"[PFC-{b_idx}] ERROR: {e}")
        time.sleep(0.05)  # 20 Hz
# ────────────────────────────────────────────────────────────

def main():
    global brain

    p.connect(p.GUI, options="--width=1440 --height=900")
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    T_ltm, P_ltm  = fresh_ltm()
    D_global       = fresh_danger()
    CM_global      = fresh_cognitive_map()
    # Bug 4 fix: Pre-seed wall geometry so A* avoids walls from episode 1
    CM_global      = preseed_cm_walls(CM_global)
    L2_legacy      = fresh_legacy_2d()

    episode              = 0
    best                 = 0.
    best_dist_ever       = 99.
    target_reached_count = 0
    ghost_count          = 0
    mourning_events      = 0
    junction_decisions   = 0
    nrem1_count = nrem3_count = rem_count = 0

    goal_idx      = 0
    goal_pos      = GOAL_POSITIONS[goal_idx]

    swarm_hebbian = HebbianAssociator(capacity=2000)
    wall_ids      = []   # safe default before first build_maze()

    import csv
    LOG_PATH = "carl_maze_log.csv"
    with open(LOG_PATH, 'w', newline='', encoding='utf-8') as f:
        csv.writer(f).writerow(['episode','best_survival','ep_max','targets_reached',
                                'ghosts','junctions','rem_count','allo_avg','stage'])

    print("\n== CARL Phase 14: THE MAZE — GOD MODE ==")
    print("Differential steering + 2D spatial planning + biological brain.")
    print("Goal moves every 200 episodes.\n")

    while True:
        episode += 1
        brain["episode"] = episode

        # Move goal every 200 episodes
        if episode % 200 == 1 and episode > 1:
            goal_idx  = (goal_idx + 1) % len(GOAL_POSITIONS)
            goal_pos  = GOAL_POSITIONS[goal_idx]
            print(f"\n  *** GOAL MOVED to {goal_pos} ***\n")

        stage = 1
        if best > 15.:  stage = 2
        if best > 35.:  stage = 3
        if best > 60.:  stage = 4
        if best > 100.: stage = 5
        brain["curriculum_stage"] = stage

        init_pitch = float(np.random.uniform(-0.10, 0.10)) if stage >= 2 else 0.

        try:
            p.resetSimulation()
            p.setGravity(0, 0, -9.81)
            p.setAdditionalSearchPath(pybullet_data.getDataPath())

            plane = p.loadURDF("plane.urdf")
            p.changeVisualShape(plane, -1, rgbaColor=[0.02, 0.02, 0.05, 1])

            draw_maze_floor_grid()
            wall_ids = build_maze()

            # Ghost markers in 2D
            for gi in range(MAP_RES):
                for gj in range(MAP_RES):
                    lv = float(L2_legacy[gi, gj])
                    if lv > 0.15:
                        xw = MAP_XMIN + (gi / MAP_RES) * (MAP_XMAX - MAP_XMIN)
                        yw = MAP_YMIN + (gj / MAP_RES) * (MAP_YMAX - MAP_YMIN)
                        a  = min(1., lv)
                        p.addUserDebugLine([xw-.06, yw, .02],
                                           [xw+.06, yw, .02],
                                           [a, a*.15, a*.15], 1)
                        p.addUserDebugLine([xw, yw-.06, .02],
                                           [xw, yw+.06, .02],
                                           [a, a*.15, a*.15], 1)

            # Cinematic camera — top-down diagonal
            cx, cy = goal_pos
            p.resetDebugVisualizerCamera(
                cameraDistance=8.0,
                cameraYaw=45,
                cameraPitch=-35,
                cameraTargetPosition=[cx*0.5, cy*0.5, 0.3])

            gx_w, gy_w = spawn_goal(goal_pos[0], goal_pos[1], episode)

            # Spawn bodies nicely spaced within the center of cell (0,0) [x=0..1.2, y=0..1.2]
            bodies = []
            for idx in range(N_BODIES):
                col = idx % 4
                row = idx // 4
                x_off = 0.3 + col * 0.25
                y_off = 0.3 + row * 0.25
                bodies.append(spawn_robot(x_off, y_off, init_pitch))

            for i, rid in enumerate(bodies):
                rc = 0.4 + 0.6 * (i / max(1, N_BODIES-1))
                bc = 1.0 - 0.4 * (i / max(1, N_BODIES-1))
                p.changeVisualShape(rid, -1, rgbaColor=[rc, .2, bc, 1.])
                p.changeVisualShape(rid,  3, rgbaColor=[rc, .2, bc, 1.])

        except Exception as e:
            print(f"PyBullet crash: {e}")
            p.disconnect(); time.sleep(1.)
            p.connect(p.GUI, options="--width=1440 --height=900")
            continue

        for _ in range(15): p.stepSimulation()
        time.sleep(0.3)

        global brains
        brains = [spawn_robot_brain(T_ltm, P_ltm, i) for i in range(N_BODIES)]
        ep_max_sv = 0.0
        for b, rid in zip(brains, bodies):
            b["rid"]    = rid
            b["xk"], _, _, _ = get_state(rid)
            b["pos_2d"] = get_pos_2d(rid)
            b["yaw"]    = get_yaw(rid)
            b["target_u"] = 0.
            b["target_s"] = 0.
            b["mode"]     = "INITIALIZING"
            b["astar_path"] = []

        # Start / restart PFC background threads
        global _pfc_threads
        for t in _pfc_threads:
            pass  # threads are daemon — they die automatically
        _pfc_threads = []
        for i in range(N_BODIES):
            t = threading.Thread(target=pfc_worker, args=(i,), daemon=True, name=f"PFC-{i}")
            t.start()
            _pfc_threads.append(t)
        print(f"  [PFC] {N_BODIES} Prefrontal Cortex threads launched.")

        D        = D_global.copy()
        CM       = CM_global.copy()
        P_ltm_ep = P_ltm.copy()
        T_ltm_ep = T_ltm.copy()
        next_wind = np.random.randint(20*240, 40*240)

        lc       = ltm_conf(P_ltm_ep)
        avg_load = float(np.mean([b["homeostatic"].allostatic_load for b in brains]))
        print(f"[Ep {episode:3d}] Stage:{stage}  LTM:{lc*100:.0f}%  "
              f"Best:{best:.1f}s  Goal:{goal_pos}  Ghosts:{ghost_count}")

        for step in range(500_000):
            sibling_died_this_step = False
            dead_this_step         = []

            for b in brains:
                if b["alive"] and b.get("grief", 0.) > 0.:
                    b["grief"] *= 0.9998

            if step % 240 == 0:
                swarm_hebbian.wire_together_decay()

            # Wind stage 3+
            wind_active = False
            if stage >= 3 and step >= next_wind:
                fx = float(np.random.uniform(-2.5, 2.5))
                for b in brains:
                    if b["alive"]:
                        p.applyExternalForce(b["rid"], -1, [fx, 0, 0],
                                             [0, 0, .3], p.WORLD_FRAME)
                next_wind   = step + np.random.randint(20*240, 40*240)
                wind_active = True

            # Earthquake stage 4+
            q_amp = 0.
            if stage >= 4:
                t_ep  = step * DT
                q_amp = float(np.clip(t_ep / 90., 0., 0.25))
                qf    = q_amp * float(np.sin(2*np.pi*1.5*t_ep))
                if q_amp > 0.01:
                    for b in brains:
                        if b["alive"]:
                            p.applyExternalForce(b["rid"], -1, [qf, 0, 0],
                                                 [0, 0, 0], p.WORLD_FRAME)

            alive_count = sum(1 for b in brains if b["alive"])
            last_s_ltm  = 0.0

            for i, rb in enumerate(brains):
                if not rb["alive"]: continue
                rid = rb["rid"]

                xn, y_pos, z_pos, roll = get_state(rid)
                xw, yw = get_pos_2d(rid)
                rb["pos_2d"] = (xw, yw)
                rb["yaw"]    = get_yaw(rid)   # live heading update

                nm          = rb["nm"]
                homeostatic = rb["homeostatic"]
                hebbian     = rb["hebbian"]
                predictive  = rb["predictive"]
                social      = rb["social"]

                # Predictive coding
                _ = predictive.predict(rb["xk"], rb["uk"],
                                       rb["T_wm"], T_ltm_ep, rb["P_wm"])

                # Memory update
                ach_lam = nm.effective_learning_rate(0.990)
                rb["T_wm"], rb["P_wm"], s_wm = rls_update(
                    rb["T_wm"], rb["P_wm"], rb["xk"], rb["uk"], xn,
                    lam=ach_lam, p_floor=0.001)

                T_ltm_ep, P_ltm_ep, s_ltm = rls_update(
                    T_ltm_ep, P_ltm_ep, rb["xk"], rb["uk"], xn,
                    lam=0.99995, p_floor=0.002)
                last_s_ltm = s_ltm

                pred_mag, _ = predictive.compute_error(xn)

                rb["s_wm_ema"] = .1*s_wm + .9*rb["s_wm_ema"]
                rb["s_slow"]   = .005*s_wm + .995*rb["s_slow"]

                # Kinematic danger map
                D  = danger_update(D, xn[2], xn[3], rb["s_wm_ema"])
                # Spatial cognitive map — trust grows with visits
                CM = cognitive_map_update(CM, xw, yw,
                                          danger=rb["s_wm_ema"],
                                          trust_delta=+0.005)

                # 2D distance to goal
                dist_now = math.sqrt((xw - goal_pos[0])**2
                                    + (yw - goal_pos[1])**2)

                # Neuromodulator update
                _at_tgt  = dist_now < 0.35
                _was_tgt = rb.get("was_at_target", False)
                nm.update(
                    surprise       = rb["s_wm_ema"],
                    danger         = danger_at(D, xn[2], xn[3]),
                    dist_to_target = dist_now,
                    target_reached = _at_tgt and not _was_tgt,
                    sibling_died   = sibling_died_this_step,
                    allostatic_load= homeostatic.allostatic_load,
                )

                # Homeostatic update
                h_error = homeostatic.update(
                    pitch          = xn[2],
                    velocity       = xn[1],
                    nm_system      = nm,
                    fatigue        = rb["fatigue"],
                    alive_siblings = alive_count - 1,
                    dist_to_target = dist_now,
                )
                rb["last_h_error"] = h_error

                # Social
                s_comfort = social.update(brains, xw)

                # Hebbian
                swarm_hebbian.fire(xn[2], xn[3], rb["uk"], rb["s_wm_ema"])
                hebbian.fire(xn[2], xn[3], rb["uk"], rb["s_wm_ema"])

                # Amygdala buffer
                danger_here = nm.effective_danger_sensitivity(
                    danger_at(D, xn[2], xn[3]))
                if danger_here > 0.20 or s_wm > 0.25:
                    rb["buf"].append((rb["xk"].copy(), rb["uk"],
                                      xn.copy(), s_wm))
                    if len(rb["buf"]) > 200:
                        rb["buf"].pop(0)

                if dist_now < 1.5:
                    rb["near_miss_buf"].append((rb["xk"].copy(), rb["uk"],
                                                xn.copy(), s_wm))
                    if len(rb["near_miss_buf"]) > 100:
                        rb["near_miss_buf"].pop(0)

                rb["fatigue"] = .999*rb["fatigue"] + .001*abs(rb["uk"])

                # Curiosity
                base_curio = float(np.clip(
                    np.trace(rb["P_wm"])/P0_WM + 0.08, 0., 1.))
                grief      = rb.get("grief", 0.)
                curio      = float(np.clip(base_curio*(1.-grief*0.5), 0.05, 1.))
                curio      = float(np.clip(curio*(1.+nm.ACh*0.3), 0., 1.))
                rb["curio"] = curio

                # Target reached
                if _at_tgt and not _was_tgt:
                    target_reached_count += 1
                    brain["target_reached_count"] = target_reached_count
                    nm.DA = min(1., nm.DA + 0.4)
                    print(f"  *** GOAL REACHED Body {i+1}! "
                          f"DA:{nm.DA:.2f} Total:{target_reached_count} ***")
                rb["was_at_target"] = _at_tgt

                if dist_now < best_dist_ever:
                    best_dist_ever = dist_now
                    brain["best_dist_ever"] = round(best_dist_ever, 3)

                # ── ACTION SELECTION: Hybrid A* + Biological Brain ──
                # PFC thread updates target_s with A* heading every 50ms.
                # The biological deliberation then picks the best forward torque
                # that is also consistent with the A* steering direction.
                dl         = danger_here * (1. + grief*0.5) * social.social_risk_modifier()
                pfc_steer  = brains[i].get("target_s", 0.)
                pfc_path   = brains[i].get("astar_path", [])
                
                if pfc_path:
                    # A* path exists — use PFC steering, pick best forward drive
                    Ad, Bd = rb["T_wm"][:SDIM,:].T, rb["T_wm"][SDIM,:]
                    best_u, best_cost = 0., float('inf')
                    for test_u in [-8., -5., -3., 0., 3., 5., 8.]:
                        xs_sim = xn.copy()
                        cost   = 0.
                        for _ in range(15):
                            xs_sim = Ad @ xs_sim + Bd * test_u
                            cost  += (xs_sim[2]**2) * 8.0  # penalise tilt
                            if abs(xs_sim[2]) > 0.38: cost += 500.; break
                        if cost < best_cost:
                            best_cost = cost
                            best_u    = test_u
                    un   = best_u
                    sn   = pfc_steer
                    mode = brains[i].get("mode", "AUTOPILOT")
                    junction_decisions += 1
                    brain["junction_decisions"] = junction_decisions
                elif dl > 0.5:
                    # Danger zone — full daughter minds deliberation
                    junction_decisions += 1
                    brain["junction_decisions"] = junction_decisions
                    un, sn, d_name = daughter_minds_maze(
                        xn, (xw, yw), rb["yaw"],
                        rb["T_wm"], T_ltm_ep, rb["P_wm"],
                        D, CM, nm, homeostatic, swarm_hebbian,
                        hebbian, predictive, social, goal_pos, grief)
                    mode = f"DELIBERATING-{d_name}"
                elif curio > np.random.random() and dl < 0.4:
                    un, sn = directed_explore_maze(
                        xn, (xw, yw), rb["yaw"],
                        rb["T_wm"], rb["P_wm"], D, CM, nm, goal_pos)
                    mode = "CURIOUS"
                else:
                    un, sn = pick_action_maze(
                        xn, (xw, yw), rb["yaw"],
                        rb["T_wm"], T_ltm_ep, rb["P_wm"],
                        D, CM, nm, homeostatic, swarm_hebbian,
                        hebbian, predictive, social, goal_pos, grief)
                    mode = ("FEARFUL"  if dl > 0.4 else
                            "GRIEVING" if grief > 0.3 else "EXPLORING")

                rb["mode"] = mode
                apply_torque(rid, un, sn)
                rb["xk"], rb["uk"], rb["sk"] = xn, un, sn
                rb["sv"] = time.time() - rb["t0"]
                ep_max_sv = max(ep_max_sv, rb["sv"])

                # Wall contact — NE spike, spatial danger burn
                contact_points = p.getContactPoints(rid)
                touching_wall = False
                for cp in (contact_points or []):
                    if cp[2] in wall_ids:
                        touching_wall = True
                        rb["wall_contacts"] += 1
                        nm.NE = min(1., nm.NE + 0.15)
                        CM    = cognitive_map_update(CM, xw, yw,
                                                     danger=0.8,
                                                     trust_delta=-0.05)
                        D     = danger_update(D, xn[2], xn[3], 0.6, rate=0.2)
                
                if touching_wall:
                    rb["wall_time"] = rb.get("wall_time", 0) + 1
                else:
                    rb["wall_time"] = 0

                # Death check
                if z_pos < 0.065 or abs(xn[2]) > 0.40 or abs(roll) > 0.40 or rb["wall_time"] > 1200:
                    sv   = rb["sv"]
                    best = max(best, sv)
                    brain["episode_history"].append(round(sv, 2))
                    brain["best_survival"] = round(best, 2)
                    rb["alive"] = False
                    sibling_died_this_step = True
                    dead_this_step.append(rb)

                    # Trauma burn in both spaces
                    for _ in range(4):
                        D  = danger_update(D, xn[2], xn[3], 10., rate=0.3)
                    CM = cognitive_map_update(CM, xw, yw,
                                             danger=1.0, trust_delta=-0.2)

                    # Mourning — death near goal
                    if dist_now < 0.8:
                        intensity = 1. - (dist_now / 0.8)
                        mourning_events += 1
                        for b in brains:
                            if b["alive"]:
                                b["grief"] = min(1., b.get("grief", 0.)
                                                  + intensity * 0.5)
                                b["nm"].NE = min(1., b["nm"].NE + intensity * 0.3)
                                b["nm"].ACh = min(1., b["nm"].ACh + 0.2)
                        print(f"  [MOURNING] Body {i+1} died near goal "
                              f"({xw:.1f},{yw:.1f})")

                    # 2D ghost
                    ghost_intensity = min(1., 0.3 + sv / 60.)
                    L2_legacy = legacy_write_2d(L2_legacy, xw, yw,
                                                ghost_intensity,
                                                goal_pos[0], goal_pos[1])
                    CM = cognitive_map_update(CM, xw, yw, danger=0.,
                                             trust_delta=0.,
                                             ghost=ghost_intensity * 0.5)
                    ghost_count += 1

                    print(f"  Body {i+1} died {sv:.2f}s | Stage {stage} | "
                          f"{mode} | Pos:({xw:.1f},{yw:.1f}) | "
                          f"DA:{nm.DA:.2f} NE:{nm.NE:.2f} "
                          f"SHT:{nm.SHT:.2f} | "
                          f"AlloLoad:{homeostatic.allostatic_load:.3f}")
                    try: p.removeBody(rid)
                    except: pass

            # Bug 2 fix: diffuse Physarum goal-scent every 10 steps.
            # Must run AFTER the robot loop so CM has the latest danger updates.
            if step % 10 == 0:
                CM = diffuse_scent(CM, goal_pos)

            # Sleep on death
            if dead_this_step:
                all_trauma = []
                all_near   = []
                for b in dead_this_step:
                    all_trauma.extend(b["buf"])
                    all_near.extend(b["near_miss_buf"])

                max_allo = max(b["homeostatic"].allostatic_load
                               for b in dead_this_step)
                brain["sleep_phase"] = "NREM-1"
                T_ltm_ep, P_ltm_ep, D = biological_sleep(
                    T_ltm_ep, P_ltm_ep, D,
                    all_trauma, all_near,
                    swarm_hebbian, max_allo)

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
                        b["nm"].NE = min(1.0, b["nm"].NE + 0.3)
                        b["nm"].DA = max(0.0, b["nm"].DA - 0.1)

            # Sync global CM so PFC threads see latest danger map
            CM_global[:] = CM[:]

            # ── DASHBOARD UPDATE ──────────────────────────────
            xA   = brains[0]["xk"] if brains[0]["xk"] is not None else np.zeros(SDIM)
            nm0  = brains[0]["nm"]
            h0   = brains[0]["homeostatic"]

            alive_modes  = [b["mode"] for b in brains if b["alive"]]
            mode_display = alive_modes[0] if alive_modes else "DEAD"
            avg_grief    = float(np.mean([b.get("grief",0.) for b in brains]))
            avg_allo     = float(np.mean([b["homeostatic"].allostatic_load
                                           for b in brains]))

            dmax = float(np.max(D)) + 1e-6
            cm_danger_flat = CM[:,:,0].T.flatten() / (float(np.max(CM[:,:,0]))+1e-6)
            cm_danger_flat = np.round(cm_danger_flat, 3).tolist()
            
            # SLIME MOLD CHANNELS
            cm_scent = np.round(CM[:,:,3].T.flatten(), 3).tolist()
            cm_phero = np.round(CM[:,:,4].T.flatten(), 3).tolist()

            brain.update({
                "cm_scent": cm_scent,
                "cm_phero": cm_phero,
                "astar_path": brains[0].get("astar_path", []),
                "robot_x": [b["pos_2d"][0] for b in brains if b["xk"] is not None],
                "robot_y": [b["pos_2d"][1] for b in brains if b["xk"] is not None],
                "survivals"          : [round(b["sv"],2) for b in brains],
                "pitches"            : [round(float(b["xk"][2]) if b["xk"] is not None else 0.,3) for b in brains],
                "distances"          : [round(float(math.sqrt(
                                             (b["pos_2d"][0]-goal_pos[0])**2+
                                             (b["pos_2d"][1]-goal_pos[1])**2))
                                           if b["xk"] is not None else 99.,2)
                                        for b in brains],
                "alive_flags"        : [b["alive"] for b in brains],
                "best_survival"      : round(best, 2),
                "mode"               : mode_display,
                "surprise_wm"        : float(brains[0]["s_wm_ema"]),
                "surprise_ltm"       : float(last_s_ltm),
                "ltm_confidence"     : float(ltm_conf(P_ltm_ep)),
                "wm_confidence_A"    : float(wm_conf(brains[0]["P_wm"])),
                "wm_confidence_B"    : float(wm_conf(brains[1]["P_wm"])),
                "curiosity"          : float(brains[0].get("curio", 0.4)),
                "danger_level"       : float(danger_at(D,xA[2],xA[3])) if brains[0]["alive"] else 0.,
                "wind_active"        : wind_active,
                "slope_deg"          : 0.0,
                "quake_amp"          : round(q_amp, 3),
                "danger_grid"        : (D/dmax).flatten().round(3).tolist(),
                "cognitive_map"      : cm_danger_flat,
                "curriculum_stage"   : stage,
                "DA"                 : round(nm0.DA, 3),
                "5HT"                : round(nm0.SHT, 3),
                "NE"                 : round(nm0.NE, 3),
                "ACh"                : round(nm0.ACh, 3),
                "allostatic_load"    : round(avg_allo, 4),
                "homeostatic_error"  : round(brains[0].get("last_h_error",0.)
                                             if brains[0]["alive"] else 0., 3),
                "sleep_phase"        : brain["sleep_phase"],
                "nrem1_count"        : nrem1_count,
                "nrem3_count"        : nrem3_count,
                "rem_count"          : rem_count,
                "social_comfort"     : round(float(np.mean(
                    [b["social"].social_comfort for b in brains if b["alive"]]
                ) if any(b["alive"] for b in brains) else 0.), 3),
                "alive_count"        : alive_count,
                "hebbian_strength"   : round(swarm_hebbian.total_strength(), 3),
                "prediction_error"   : round(float(brains[0]["predictive"].precision_weight())
                                             if brains[0]["alive"] else 0., 3),
                "goal_x"             : goal_pos[0],
                "goal_y"             : goal_pos[1],
                "goal_episode"       : episode,
                "ghost_count"        : ghost_count,
                "mourning_events"    : mourning_events,
                "target_reached_count": target_reached_count,
                "best_dist_ever"     : round(best_dist_ever, 3),
                "junction_decisions" : junction_decisions,
            })

            # HUD
            p.removeAllUserDebugItems()
            draw_maze_floor_grid()
            # Redraw wall edges
            for (gx1, gy1, gx2, gy2) in MAZE_WALLS_GRID:
                p.addUserDebugLine(
                    [gx1*MAZE_CELL, gy1*MAZE_CELL, WALL_H],
                    [gx2*MAZE_CELL, gy2*MAZE_CELL, WALL_H],
                    [0.15, 0.45, 1.0], 1)
            # Goal beacon
            gx_d, gy_d = goal_pos
            for angle in np.linspace(0, 2*np.pi, 12):
                p.addUserDebugLine(
                    [gx_d, gy_d, 0.3],
                    [gx_d+0.35*math.cos(angle),
                     gy_d+0.35*math.sin(angle), 0.3],
                    [1., 0.15, 0.5], 2)
            # Ghost markers
            for gi in range(0, MAP_RES, 2):
                for gj in range(0, MAP_RES, 2):
                    lv = float(L2_legacy[gi, gj])
                    if lv > 0.15:
                        xw = MAP_XMIN + (gi/MAP_RES)*(MAP_XMAX-MAP_XMIN)
                        yw = MAP_YMIN + (gj/MAP_RES)*(MAP_YMAX-MAP_YMIN)
                        a  = min(1., lv)
                        p.addUserDebugLine([xw-.06,yw,.02],[xw+.06,yw,.02],
                                           [a,a*.15,a*.15],1)

            draw_hud(nm0, mode_display, best, avg_allo, stage,
                     goal_pos, alive_count, junction_decisions,
                     target_reached_count)

            p.stepSimulation()
            time.sleep(DT)

            if not any(b["alive"] for b in brains):
                T_ltm, P_ltm = T_ltm_ep.copy(), P_ltm_ep.copy()
                D_global     = D.copy()
                CM_global    = CM.copy()

                if episode % 50 == 0:
                    np.save("maze_ltm_T.npy",   T_ltm)
                    np.save("maze_ltm_P.npy",   P_ltm)
                    np.save("maze_danger.npy",  D_global)
                    np.save("maze_cm.npy",      CM_global)
                    np.save("maze_legacy.npy",  L2_legacy)
                    print(f"  [CHECKPOINT ep {episode}]")

                allo_avg = float(np.mean([b["homeostatic"].allostatic_load
                                           for b in brains]))
                print(f"  Episode {episode} over. "
                      f"Best:{best:.2f}s  EpMax:{ep_max_sv:.1f}s  "
                      f"Reached:{target_reached_count}x  "
                      f"Ghosts:{ghost_count}  "
                      f"Junctions:{junction_decisions}  "
                      f"AlloLoad:{allo_avg:.4f}  REM:{rem_count}")

                # CSV log
                with open(LOG_PATH, 'a', newline='', encoding='utf-8') as f:
                    csv.writer(f).writerow([
                        episode, round(best,2), round(ep_max_sv,2),
                        target_reached_count, ghost_count, junction_decisions,
                        rem_count, round(allo_avg,4), stage])

                time.sleep(1.5)
                break

if __name__ == "__main__":
    # Bug 1 fix: do NOT set pfc_active=False here — that kills PFC threads
    # before they even start. Threads are daemons and die when main() exits.
    main()
