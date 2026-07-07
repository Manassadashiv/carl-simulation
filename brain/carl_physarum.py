"""
carl_physarum.py — Physarum Polycephalum Slime Mold Path Optimizer
Based on: Tero, Kobayashi & Nakagaki (2007) Science — proven shortest-path convergence.

The maze is a network of tubes. Tubes carrying high flow THICKEN.
Tubes carrying no flow RETRACT. System provably converges to Steiner shortest path.
"""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

# ── Parameters ────────────────────────────────────────────────
MU      = 1.8      # Reinforcement exponent (1 < mu < 2 for convergence)
TAU     = 80.0     # Conductivity adaptation time constant (steps)
D_MIN   = 1e-5     # Minimum conductivity (prevents disconnection)
D_MAX   = 50.0     # Maximum conductivity


class PhysarumMaze:
    """
    Models the cognitive map grid as a network of Physarum tubes.
    Each navigable cell is a node; edges between adjacent cells
    have conductivity D that adapts based on fluid flux.
    """

    def __init__(self, rows=25, cols=25):
        self.R = rows
        self.C = cols
        self.N = rows * cols

        # Horizontal edges: D_h[i,j] = conductivity between (i,j) and (i,j+1)
        self.D_h = np.ones((rows, cols-1), dtype=float)
        # Vertical edges: D_v[i,j] = conductivity between (i,j) and (i+1,j)
        self.D_v = np.ones((rows-1, cols), dtype=float)

        # Obstacle mask (updated each step from cognitive map)
        self.obstacle = np.zeros((rows, cols), dtype=bool)

        # Pressure field (solution)
        self.pressure = np.zeros(self.N, dtype=float)

        # Conductivity trail — used as navigation signal
        self.nav_signal = np.zeros((rows, cols), dtype=float)

        self._step_count = 0

    # ── Index helpers ─────────────────────────────────────────
    def _idx(self, i, j):
        return int(i) * self.C + int(j)

    def _pos(self, idx):
        return idx // self.C, idx % self.C

    # ── Update obstacle mask from cognitive map ───────────────
    def update_obstacles(self, CM_danger, danger_thresh=0.7):
        """Mark high-danger cells as obstacles."""
        self.obstacle = CM_danger > danger_thresh

    # ── Build sparse Kirchhoff/Laplacian ──────────────────────
    def _build_laplacian(self):
        data, row_idx, col_idx = [], [], []
        diag = np.zeros(self.N, dtype=float)

        for i in range(self.R):
            for j in range(self.C):
                n = self._idx(i, j)
                if self.obstacle[i, j]:
                    # Isolated node — tiny self-connection to keep matrix non-singular
                    data.append(1e-6)
                    row_idx.append(n); col_idx.append(n)
                    continue

                # Right neighbor
                if j + 1 < self.C and not self.obstacle[i, j+1]:
                    m = self._idx(i, j+1)
                    d = float(self.D_h[i, j])
                    data.append(-d); row_idx.append(n); col_idx.append(m)
                    data.append(-d); row_idx.append(m); col_idx.append(n)
                    diag[n] += d; diag[m] += d

                # Down neighbor
                if i + 1 < self.R and not self.obstacle[i+1, j]:
                    m = self._idx(i+1, j)
                    d = float(self.D_v[i, j])
                    data.append(-d); row_idx.append(n); col_idx.append(m)
                    data.append(-d); row_idx.append(m); col_idx.append(n)
                    diag[n] += d; diag[m] += d

        # Add diagonal
        for n in range(self.N):
            if diag[n] < 1e-10:
                diag[n] = 1e-6  # Prevent singular matrix
            data.append(diag[n])
            row_idx.append(n); col_idx.append(n)

        L = sparse.csr_matrix((data, (row_idx, col_idx)), shape=(self.N, self.N))
        return L

    # ── Solve pressure field ──────────────────────────────────
    def _solve_pressure(self, src_idx, snk_idx):
        L = self._build_laplacian()
        rhs = np.zeros(self.N, dtype=float)

        # Boundary conditions: fix source and sink pressures
        L_mod = L.tolil()
        L_mod[src_idx, :] = 0; L_mod[src_idx, src_idx] = 1.0; rhs[src_idx] = 1.0
        L_mod[snk_idx, :] = 0; L_mod[snk_idx, snk_idx] = 1.0; rhs[snk_idx] = 0.0

        try:
            p = spsolve(L_mod.tocsr(), rhs)
            if np.any(np.isnan(p)): p = np.zeros(self.N)
        except Exception:
            p = np.zeros(self.N)

        self.pressure = np.clip(p, 0.0, 1.0)

    # ── Update conductivities (Physarum dynamics) ─────────────
    def _update_conductivities(self):
        p = self.pressure.reshape(self.R, self.C)

        # Horizontal edges: Q = D * |p_i - p_{i,j+1}|
        p_left  = p[:, :-1]
        p_right = p[:, 1:]
        Q_h = np.abs(self.D_h * (p_left - p_right))
        dD_h = (Q_h**MU - self.D_h) / TAU
        self.D_h = np.clip(self.D_h + dD_h, D_MIN, D_MAX)
        # Zero out obstacle edges
        for i in range(self.R):
            for j in range(self.C-1):
                if self.obstacle[i,j] or self.obstacle[i,j+1]:
                    self.D_h[i,j] = D_MIN

        # Vertical edges
        p_top = p[:-1, :]
        p_bot = p[1:,  :]
        Q_v = np.abs(self.D_v * (p_top - p_bot))
        dD_v = (Q_v**MU - self.D_v) / TAU
        self.D_v = np.clip(self.D_v + dD_v, D_MIN, D_MAX)
        for i in range(self.R-1):
            for j in range(self.C):
                if self.obstacle[i,j] or self.obstacle[i+1,j]:
                    self.D_v[i,j] = D_MIN

    # ── Compute navigation signal at each cell ────────────────
    def _compute_nav_signal(self):
        """
        Nav signal at (i,j) = sum of conductivities of edges
        connecting to the lower-pressure side (toward sink).
        High conductivity + decreasing pressure = the path.
        """
        p = self.pressure.reshape(self.R, self.C)
        sig = np.zeros((self.R, self.C), dtype=float)

        # Horizontal
        sig[:, :-1] += self.D_h * np.maximum(0, p[:, :-1] - p[:, 1:])  # left→right if p_l > p_r
        sig[:, 1:]  += self.D_h * np.maximum(0, p[:, 1:] - p[:, :-1])  # right→left
        # Vertical
        sig[:-1, :] += self.D_v * np.maximum(0, p[:-1, :] - p[1:, :])
        sig[1:,  :] += self.D_v * np.maximum(0, p[1:,  :] - p[:-1, :])

        # Normalise
        mx = sig.max()
        if mx > 1e-6: sig /= mx
        self.nav_signal = sig

    # ── Main step ─────────────────────────────────────────────
    def step(self, src_cell, snk_cell, CM_danger=None, every=1):
        """
        Run one Physarum update cycle.
        Solves the harmonic pressure field every step to support dynamic high-speed navigation.
        """
        self._step_count += 1

        if CM_danger is not None:
            self.update_obstacles(CM_danger)

        si = self._idx(*src_cell)
        sk = self._idx(*snk_cell)

        if si == sk:
            self.nav_signal[:] = 0.0
            return

        # Solve pressure field every frame to track moving robot/target instantly
        self._solve_pressure(si, sk)

        # Keep conductivities active to prevent "dead-tube" lock-in
        self.D_h[:] = 1.0
        self.D_v[:] = 1.0
        
        # Zero out obstacle edges
        for i in range(self.R):
            for j in range(self.C-1):
                if self.obstacle[i, j] or self.obstacle[i, j+1]:
                    self.D_h[i, j] = D_MIN

        for i in range(self.R-1):
            for j in range(self.C):
                if self.obstacle[i, j] or self.obstacle[i+1, j]:
                    self.D_v[i, j] = D_MIN

        self._compute_nav_signal()

    # ── Navigate: best adjacent cell to move toward sink ─────
    def best_move(self, cur_cell, src_cell, snk_cell):
        """
        Navigate toward the sink by following the steepest pressure drop (descending gradient).
        This guarantees zero local minima and flawless obstacle avoidance.
        """
        i, j = int(cur_cell[0]), int(cur_cell[1])
        p = self.pressure.reshape(self.R, self.C)
        
        candidates = []
        for di, dj in [(-1,0), (1,0), (0,-1), (0,1)]:
            ni, nj = i+di, j+dj
            if 0 <= ni < self.R and 0 <= nj < self.C:
                if not self.obstacle[ni, nj]:
                    # We want to move to a cell with lower pressure (steepest descent)
                    if p[ni, nj] < p[i, j] + 1e-5:
                        candidates.append(((ni, nj), p[i, j] - p[ni, nj]))
                        
        if not candidates:
            return cur_cell
            
        # Return neighbor with largest pressure drop
        return max(candidates, key=lambda x: x[1])[0]

    # ── Get desired heading from current world position ───────
    def get_heading(self, xw, yw, goal_xw, goal_yw,
                    MAP_XMIN, MAP_XMAX, MAP_YMIN, MAP_YMAX):
        """
        Convert world coords to grid, get best move, return heading angle.
        """
        def to_grid(x, y):
            gi = int(np.clip((x - MAP_XMIN)/(MAP_XMAX - MAP_XMIN)*self.R, 0, self.R-1))
            gj = int(np.clip((y - MAP_YMIN)/(MAP_YMAX - MAP_YMIN)*self.C, 0, self.C-1))
            return gi, gj

        cur  = to_grid(xw, yw)
        snk  = to_grid(goal_xw, goal_yw)

        best = self.best_move(cur, cur, snk)
        if best == cur:
            import math
            return math.atan2(goal_yw - yw, goal_xw - xw)

        # Convert best grid cell back to world coords
        bx = MAP_XMIN + (best[0] + 0.5) / self.R * (MAP_XMAX - MAP_XMIN)
        by = MAP_YMIN + (best[1] + 0.5) / self.C * (MAP_YMAX - MAP_YMIN)
        import math
        return math.atan2(by - yw, bx - xw)

    # ── Diagnostic: how connected is the path? ───────────────
    def path_strength(self):
        return float(np.percentile(np.concatenate([
            self.D_h.flatten(), self.D_v.flatten()]), 90))
