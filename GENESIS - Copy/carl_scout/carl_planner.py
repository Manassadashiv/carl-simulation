import numpy as np
import heapq
import collections
try:
    from scipy.ndimage import binary_dilation
except ImportError:
    binary_dilation = None

class AStarPlanner:
    def __init__(self, grid_size=50, resolution=0.20):
        self.size = grid_size
        self.res = resolution
        self.wall_dilation_radius = 1  # Number of grid cells to inflate walls for safety (0.18m is > CARL's 0.11m radius)
        self._last_grid = None
        self._dilated_grid = None
        self._free_cell_cache = {}  # Cache nearest free cells to optimize planning ticks

    def dilate_map(self, grid):
        """ Inflates walls so CARL maintains a safe structural clearance """
        if self._last_grid is not None and np.array_equal(grid, self._last_grid):
            return self._dilated_grid
            
        # Clear free cell cache when grid layout shifts
        self._free_cell_cache.clear()
            
        if binary_dilation is not None:
            # 3x3 square structure for 8-connectivity dilation
            structure = np.ones((3, 3), dtype=bool)
            dilated = binary_dilation(grid == 1.0, structure=structure, iterations=self.wall_dilation_radius)
        else:
            r = (grid == 1.0)
            dilated = r.copy()
            for radius in range(1, self.wall_dilation_radius + 1):
                dilated[1:, :] |= r[:-1, :]
                dilated[:-1, :] |= r[1:, :]
                dilated[:, 1:] |= r[:, :-1]
                dilated[:, :-1] |= r[:, 1:]
                dilated[1:, 1:] |= r[:-1, :-1]
                dilated[:-1, :-1] |= r[1:, 1:]
                dilated[1:, :-1] |= r[:-1, 1:]
                dilated[:-1, 1:] |= r[1:, :-1]
                if self.wall_dilation_radius > 1:
                    r = dilated.copy()
                
        dilated_grid = np.copy(grid)
        dilated_grid[dilated] = 1.0
                    
        self._last_grid = np.copy(grid)
        self._dilated_grid = dilated_grid
        return dilated_grid

    def heuristic(self, a, b):
        """ Straight-line distance calculation for path optimization """
        return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

    def find_nearest_free_cell(self, dilated_grid, start_cell):
        if start_cell in self._free_cell_cache:
            return self._free_cell_cache[start_cell]
            
        if dilated_grid[start_cell[0], start_cell[1]] == 0.0:
            self._free_cell_cache[start_cell] = start_cell
            return start_cell
            
        queue = collections.deque([start_cell])
        visited = {start_cell}
        while queue:
            curr = queue.popleft()
            if dilated_grid[curr[0], curr[1]] == 0.0:
                self._free_cell_cache[start_cell] = curr
                return curr
            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]:
                neighbor = (curr[0] + dx, curr[1] + dy)
                if 0 <= neighbor[0] < self.size and 0 <= neighbor[1] < self.size:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
                        
        self._free_cell_cache[start_cell] = start_cell
        return start_cell

    def plan(self, grid, start_pose, target_pos):
        """ Calculates the absolute shortest path avoiding dilated walls """
        dilated_grid = self.dilate_map(grid)
        
        # Convert world coordinates (meters) to grid coordinates (indexes)
        start = (
            int((start_pose[0] + (self.size * self.res) / 2) / self.res),
            int((start_pose[1] + (self.size * self.res) / 2) / self.res)
        )
        goal = (
            int((target_pos[0] + (self.size * self.res) / 2) / self.res),
            int((target_pos[1] + (self.size * self.res) / 2) / self.res)
        )

        # Bound checks to keep coordinates safe inside the grid boundaries
        start = (max(0, min(self.size-1, start[0])), max(0, min(self.size-1, start[1])))
        goal = (max(0, min(self.size-1, goal[0])), max(0, min(self.size-1, goal[1])))

        # Fallback to nearest free cells if start or goal are inside dilated walls
        start = self.find_nearest_free_cell(dilated_grid, start)
        goal = self.find_nearest_free_cell(dilated_grid, goal)

        # Priority Queue setup: (f_score, current_node)
        open_set = []
        heapq.heappush(open_set, (0, start))
        
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}

        neighbors = [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]

        while open_set:
            current = heapq.heappop(open_set)[1]

            if current == goal:
                # Reconstruct the optimal path coordinates back to world units
                path = []
                while current in came_from:
                    world_x = (current[0] * self.res) - (self.size * self.res) / 2 + self.res/2
                    world_y = (current[1] * self.res) - (self.size * self.res) / 2 + self.res/2
                    path.append([world_x, world_y])
                    current = came_from[current]
                path.reverse()
                
                # Ensure the exact target position is the final waypoint
                if len(path) == 0:
                    path.append([float(target_pos[0]), float(target_pos[1])])
                else:
                    path[-1] = [float(target_pos[0]), float(target_pos[1])]
                return path # List of optimal waypoints

            for dx, dy in neighbors:
                neighbor = (current[0] + dx, current[1] + dy)
                
                if not (0 <= neighbor[0] < self.size and 0 <= neighbor[1] < self.size):
                    continue
                if dilated_grid[neighbor[0], neighbor[1]] == 1.0:
                    continue

                # Movement cost calculation (diagonal steps cost slightly more)
                move_cost = 1.414 if (dx != 0 and dy != 0) else 1.0
                tentative_g = g_score[current] + move_cost

                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return None  # Fallback if path is completely blocked


class PurePursuitFollower:
    def __init__(self, lookahead_dist=0.4, max_v=1.2, max_w=2.0):
        self.lookahead_dist = lookahead_dist
        self.max_v = max_v
        self.max_w = max_w

    def get_control(self, current_pose, path):
        """ Calculates target linear and angular velocities to chase the path """
        if not path or len(path) == 0:
            # No valid path — wander forward slowly instead of freezing
            return 0.15, 0.3
            
        cx, cy, ctheta = current_pose
        
        # Find the lookahead point on the path
        target_point = None
        for wp in path:
            dist = np.sqrt((wp[0] - cx)**2 + (wp[1] - cy)**2)
            if dist >= self.lookahead_dist:
                target_point = wp
                break
        
        # Fallback to the final destination point if the path is short
        if target_point == None:
            target_point = path[-1]

        # Transform target point to CARL's local coordinate frame
        dx = target_point[0] - cx
        dy = target_point[1] - cy
        
        # Rotation matrix to local frame
        local_x = dx * np.cos(-ctheta) - dy * np.sin(-ctheta)
        local_y = dx * np.sin(-ctheta) + dy * np.cos(-ctheta)
        
        # If the target point is behind the robot, turn in place to face it
        if local_x < 0.0:
            # Turn in the direction of the target (left if local_y >= 0, right if local_y < 0)
            w_target = self.max_w if local_y >= 0.0 else -self.max_w
            return 0.0, w_target

        # Calculate curvature (2 * local_y / dist^2)
        L2 = local_x**2 + local_y**2
        if L2 < 0.01:
            return 0.1, 0.0
            
        curvature = (2.0 * local_y) / L2
        
        # Command Generation
        # Slow down slightly in sharp turns to preserve wheel traction
        v_target = self.max_v / (1.0 + 2.0 * abs(curvature))
        v_target = np.clip(v_target, 0.2, self.max_v)
        w_target = np.clip(curvature * v_target, -self.max_w, self.max_w)
        
        return v_target, w_target


class ApexStateManager:
    def __init__(self):
        self.state = "EXPLORER"
        print("[CORTEX] Apex State Manager initialized. Starting in EXPLORER mode.")

    def evaluate_state(self, current_state, map_confidence, energy, reflex_fired, path_valid):
        """ Hard state-transition boundaries to enforce safety and efficiency """
        
        # Emergency Fallbacks out of Speed Demon Mode
        if current_state == "SPEED_DEMON":
            if reflex_fired:
                print("[STATE] Safety Reflex Interrupt! Forcing EXPLORER fallback.")
                return "EXPLORER"
            if energy < 20.0:
                print("[STATE] Energy Low (< 20%). Forcing EXPLORER Economy Mode.")
                return "EXPLORER"
            if not path_valid:
                print("[STATE] Path blocked or lost. Forcing EXPLORER re-routing.")
                return "EXPLORER"
                
        # Promotion Criteria into Speed Demon Mode
        if current_state == "EXPLORER":
            if map_confidence > 0.90 and energy >= 20.0 and not reflex_fired:
                print("[STATE] Map Confidence > 90%. Transitioning to SPEED_DEMON execution!")
                return "SPEED_DEMON"
                
        return current_state


class CrystallizedGoal:
    """
    Represents a crystallized goal in CARL's preference buffer.
    Its value decays dynamically based on boredom and opportunity costs.
    """
    def __init__(self, target_pos, utility=1.0, goal_type='curiosity'):
        self.target_pos = np.array(target_pos, dtype=np.float32)
        self.U = utility
        self.B = 0.0
        self.V = utility
        self.goal_type = goal_type
        
    def update(self, current_pos, is_active, boredom_accum, boredom_decay, energy, damage):
        # Boredom dynamics: accumulates when actively targeted and close, decays when away
        dist = np.linalg.norm(self.target_pos - current_pos)
        
        # Customize boredom rates based on goal type
        if self.goal_type == 'food':
            # Food is a critical survival resource, so Bob never gets bored of it
            actual_accum = 0.0
            actual_decay = boredom_decay * 3.0
        elif self.goal_type == 'toy':
            # Toys are interesting but Bob gets bored of them eventually
            actual_accum = boredom_accum * 0.5
            actual_decay = boredom_decay
        else:
            # Curiosity targets are quickly depleted of novelty
            actual_accum = boredom_accum * 1.5
            actual_decay = boredom_decay * 0.5

        if dist < 0.5 and is_active:
            self.B = min(1.0, self.B + actual_accum)
        else:
            self.B = max(0.0, self.B - actual_decay)
            
        # Cost dynamics: hunger (low energy) and damage increase opportunity cost for non-food
        if self.goal_type == 'food':
            # High hunger (low energy) reduces the cost / increases value of food goals
            C = 0.4 * damage - 0.6 * (1.0 - energy)
        else:
            C = 0.6 * (1.0 - energy) + 0.4 * damage
        
        self.V = self.U - self.B - C
        return self.V


class GoalCrystallizer:
    """
    Manages goal crystallization and temporal persistence checks.
    Only crystallizes goals if interest/progress remains stable over N steps.
    """
    def __init__(self, persistence_threshold=30):
        self.goals = []
        self.active_goal_idx = None
        
        # Temporal persistence parameters
        self.candidate_pos = None
        self.persistence_counter = 0
        self.persistence_threshold = persistence_threshold
        self.max_utility = 1.0
        self.candidate_type = 'curiosity'
        
    def add_goal(self, target_pos, utility=1.0, goal_type='curiosity'):
        # Avoid duplicating goals in the same local zone
        for g in self.goals:
            if np.linalg.norm(g.target_pos - target_pos) < 0.8:
                g.B = 0.0  # Reset boredom if re-visited/crystallized
                # Update utility and type if the new candidate is stronger/more specific
                if utility > g.U:
                    g.U = utility
                    g.goal_type = goal_type
                return
        self.goals.append(CrystallizedGoal(target_pos, utility, goal_type))
        print(f"[GOAL] Crystallized new {goal_type} goal at coordinates: {target_pos} with utility {utility:.2f}")
        
        # Cap working memory capacity to 8 goals
        if len(self.goals) > 8:
            self.goals.sort(key=lambda g: g.V, reverse=True)
            self.goals = self.goals[:8]
        
    def track_and_crystallize(self, current_pos, lp, da, utility=1.0, goal_type='curiosity'):
        """Monitors persistence of high Learning Progress or Dopamine interaction."""
        if goal_type == 'food':
            # Bypass persistence gate and add food goal immediately
            self.add_goal(current_pos, utility=utility, goal_type='food')
            self.candidate_pos = None
            self.persistence_counter = 0
            return

        is_curious = (lp > 0.02) or (da > 0.8)
        
        if is_curious:
            if self.candidate_pos is None:
                self.candidate_pos = np.array(current_pos, dtype=np.float32)
                self.persistence_counter = 1
                self.max_utility = utility
                self.candidate_type = goal_type
            else:
                if np.linalg.norm(current_pos - self.candidate_pos) < 0.5:
                    self.persistence_counter += 1
                    if utility > self.max_utility:
                        self.max_utility = utility
                        self.candidate_type = goal_type
                    if self.persistence_counter >= self.persistence_threshold:
                        self.add_goal(self.candidate_pos.copy(), utility=self.max_utility, goal_type=self.candidate_type)
                        self.candidate_pos = None
                        self.persistence_counter = 0
                else:
                    self.candidate_pos = np.array(current_pos, dtype=np.float32)
                    self.persistence_counter = 1
                    self.max_utility = utility
                    self.candidate_type = goal_type
        else:
            self.persistence_counter = max(0, self.persistence_counter - 1)
            if self.persistence_counter == 0:
                self.candidate_pos = None
                
    def step(self, current_pos, drives, boredom_accum=0.04, boredom_decay=0.01):
        """Updates goal metrics and returns the coordinates of the highest-value active goal."""
        for idx, g in enumerate(self.goals):
            is_active = (self.active_goal_idx is not None and idx == self.active_goal_idx)
            prev_V = g.V
            val = g.update(current_pos, is_active, boredom_accum, boredom_decay, drives.energy, drives.damage)
            if is_active and val <= 0.0:
                print(f"[GOAL] Goal at {g.target_pos} expired (Value={val:.3f}, Boredom={g.B:.3f}). Deactivating.")
        
        # Select active navigation targets from those with positive value
        # Apply hysteresis to prevent rapid on/off activation oscillations:
        # If the goal was already active, it needs V > 0.0 to stay active.
        # If it was inactive, it needs V > 0.02 to become eligible for activation.
        valid_indices = []
        for i, g in enumerate(self.goals):
            was_active = (self.active_goal_idx is not None and i == self.active_goal_idx)
            threshold = 0.0 if was_active else 0.02
            if g.V > threshold:
                valid_indices.append(i)
        
        if not valid_indices:
            self.active_goal_idx = None
            return None
            
        best_idx = valid_indices[0]
        best_val = self.goals[best_idx].V
        for idx in valid_indices[1:]:
            if self.goals[idx].V > best_val:
                best_val = self.goals[idx].V
                best_idx = idx
                
        self.active_goal_idx = best_idx
        return self.goals[best_idx].target_pos


class LocomotionMPCPlanner:
    """
    Trajectory Model-Predictive Control (MPC) Planner.
    Uses Active Inference principles to select optimal velocity commands.
    """
    def __init__(self, K=30, horizon=6, dt=0.02):
        self.K = K
        self.horizon = horizon
        self.dt = dt
        self.sigma_v = 0.15
        self.sigma_w = 0.4
        
    def generate_candidates(self, prev_action):
        """Generates K Ornstein-Uhlenbeck smoothed linear/angular trajectory candidates."""
        candidates = np.zeros((self.K, self.horizon, 2), dtype=np.float32)
        for k in range(self.K):
            v_target = np.random.uniform(-0.2, 0.8)
            w_target = np.random.uniform(-1.5, 1.5)
            
            curr = prev_action.copy()
            for t in range(self.horizon):
                curr[0] = curr[0] + 0.35 * (v_target - curr[0]) + np.random.normal(0, self.sigma_v)
                curr[1] = curr[1] + 0.35 * (w_target - curr[1]) + np.random.normal(0, self.sigma_w)
                curr[0] = np.clip(curr[0], -0.4, 1.0)
                curr[1] = np.clip(curr[1], -2.0, 2.0)
                candidates[k, t] = curr.copy()
        return candidates

    def plan_active_inference(self, brain, obs_history, prev_action, target_pos, pose, world_map=None):
        """Scores candidate trajectories against Epistemic and Instrumental values."""
        candidates = self.generate_candidates(prev_action)
        G_scores = np.zeros(self.K)
        
        for k in range(self.K):
            curr_obs = obs_history[-1].copy()
            curr_pose = np.array(pose, dtype=np.float32)
            
            epistemic_sum = 0.0
            instrumental_sum = 0.0
            collision_penalty = 0.0
            
            for t in range(self.horizon):
                action = candidates[k, t]
                # Dreamer predicts next observation given current obs and action
                pred_obs = brain.dreamer.predict(curr_obs[:brain.dreamer.n_obs], action)
                
                # Epistemic Value: reward low familiarity (HDC novelty)
                pred_obs_sensory = np.zeros(brain.hdc.input_dim, dtype=np.float32)
                copy_dim = min(len(pred_obs), brain.hdc.input_dim)
                pred_obs_sensory[:copy_dim] = pred_obs[:copy_dim]
                familiarity = brain.hdc.query_memory(pred_obs_sensory)
                epistemic_val = max(0.0, 0.02 - familiarity) * 5.0
                epistemic_sum += epistemic_val
                
                # Collision risk from predicted proximity sensors (indices 0-7)
                pred_proximity = pred_obs[:8]
                collision_risk = np.max(pred_proximity)
                if collision_risk > 0.8:
                    collision_penalty += (collision_risk - 0.8) * 10.0
                
                # Kinematic pose projection
                curr_pose[2] += action[1] * self.dt
                curr_pose[0] += action[0] * np.cos(curr_pose[2]) * self.dt
                curr_pose[1] += action[0] * np.sin(curr_pose[2]) * self.dt
                
                # Proximity to planning target position
                dist_to_target = np.linalg.norm(curr_pose[:2] - target_pos)
                instrumental_val = -dist_to_target * 2.0
                instrumental_sum += instrumental_val
                
                # Update rolling context for dreamer
                curr_obs = np.zeros_like(obs_history[-1])
                copy_len = min(len(curr_obs), len(pred_obs))
                curr_obs[:copy_len] = pred_obs[:copy_len]
                
            G_scores[k] = -epistemic_sum - instrumental_sum + collision_penalty
            
        # Softmax selection over Expected Free Energy G
        G_min = np.min(G_scores)
        logits = -10.0 * (G_scores - G_min)
        probs = np.exp(logits) / np.sum(np.exp(logits))
        
        best_idx = np.random.choice(self.K, p=probs)
        return candidates[best_idx, 0], G_scores[best_idx]
