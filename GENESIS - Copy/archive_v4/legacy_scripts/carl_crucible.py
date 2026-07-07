"""
carl_crucible.py — The Crucible: Evolutionary Pressure Environment Controller.

This module transforms CARL's static laboratory into a living, shifting,
hostile ecosystem designed to force the emergence of genuine survival
instincts through extreme environmental pressure.

Systems:
    1. CircadianClock       — Day/night light cycle
    2. BiomeWeatherEngine   — Markov-chain physics mutations (sand/water/slopes/wind/vacuum/pressure)
    3. ScarcityController   — Food rot, rare spawns, starvation pressure
    4. PredatorAI           — Autonomous hostile entity that hunts CARL
    5. PoltergeistEvent     — Randomly launched projectiles
    6. NeuromorphicDVS      — Simulated dynamic vision sensor (looming detection)
    7. MPCDodge             — Model Predictive Control evasion subroutine
    8. MazeShifter          — Periodic labyrinth reconfiguration
    9. ObjectPermanence     — Spatial memory for occluded objects
   10. VisualServoing       — PID face-tracking for social fixation

Author: CARL Genesis Project
"""

import math
import time
import random
import numpy as np
import mujoco
from carl_maze_gen import generate_maze


# ═══════════════════════════════════════════════════════════════════════════════
#  1. CIRCADIAN CLOCK — Day / Night Cycle
# ═══════════════════════════════════════════════════════════════════════════════

class CircadianClock:
    """
    Modulates MuJoCo scene lighting to simulate a full day/night cycle.
    Uses a sinusoidal curve mapped to a configurable cycle period.
    At night, only CARL's emissive materials provide illumination.
    """

    def __init__(self, cycle_period_s=300.0):
        """
        Args:
            cycle_period_s: Duration of one full day/night cycle in seconds.
                            Default 300s = 5 minutes per full cycle.
        """
        self.cycle_period = cycle_period_s
        self.t = 0.0
        self.phase = "DAY"
        self._last_log_phase = None

    def step(self, model, dt):
        """
        Advance the clock and update lighting.

        Args:
            model: MuJoCo MjModel instance.
            dt: Time delta since last call (seconds).

        Returns:
            dict with 'phase' ('DAY', 'DUSK', 'NIGHT', 'DAWN') and 'intensity' (0.0–1.0).
        """
        self.t += dt

        # Sinusoidal intensity: 1.0 at noon, 0.0 at midnight
        raw = 0.5 * (1.0 + math.sin(2.0 * math.pi * self.t / self.cycle_period - math.pi / 2.0))
        intensity = max(0.02, raw)  # Never go fully black — maintain minimal ambient

        # Determine phase
        if intensity > 0.7:
            self.phase = "DAY"
        elif intensity > 0.4:
            self.phase = "DUSK" if raw < 0.5 and self.t % self.cycle_period > self.cycle_period / 2 else "DAWN"
        else:
            self.phase = "NIGHT"

        # Apply to MuJoCo light
        light_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_LIGHT, "studio_sun")
        if light_id != -1:
            # Warm daylight → cold moonlight color shift
            if intensity > 0.5:
                r, g, b = 0.85 * intensity, 0.85 * intensity, 0.82 * intensity
            else:
                # Moonlight: blue-shifted, very dim
                r, g, b = 0.15 * intensity, 0.20 * intensity, 0.35 * intensity
            model.light_diffuse[light_id] = [r, g, b]
            model.light_ambient[light_id] = [0.05 * intensity, 0.05 * intensity, 0.08 * intensity]

        if self.phase != self._last_log_phase:
            self._last_log_phase = self.phase
            print(f"[CIRCADIAN] Phase transition → {self.phase} (intensity: {intensity:.2f})")

        return {"phase": self.phase, "intensity": intensity}


# ═══════════════════════════════════════════════════════════════════════════════
#  2. BIOME WEATHER ENGINE — Markov Chain Physics Mutations
# ═══════════════════════════════════════════════════════════════════════════════

class BiomeWeatherEngine:
    """
    Markov chain that transitions between atmospheric/terrain biomes by
    directly mutating MuJoCo's physics engine parameters at runtime.

    Biomes:
        CLEAR       — Default physics, baseline conditions.
        SAND        — High friction, drains energy.
        WATER       — High density + viscosity, massive drag.
        SLOPES      — Tilted gravity vector, uphill/downhill.
        WIND        — Random wind forces push CARL around.
        VACUUM      — Zero air density, no drag.
        PRESSURE    — Extreme viscosity, syrup-like movement.
        DEBRIS      — Scattered small obstacles (handled via object spawning).
    """

    BIOMES = ["CLEAR", "SAND", "WATER", "SLOPES", "WIND", "VACUUM", "PRESSURE", "DEBRIS"]

    # Transition probability matrix (row = current, col = next)
    # Rows sum to 1.0. Designed so CLEAR is the "resting" state.
    TRANSITION_MATRIX = {
        "CLEAR":    {"CLEAR": 0.30, "SAND": 0.12, "WATER": 0.10, "SLOPES": 0.12, "WIND": 0.12, "VACUUM": 0.08, "PRESSURE": 0.08, "DEBRIS": 0.08},
        "SAND":     {"CLEAR": 0.40, "SAND": 0.15, "WATER": 0.10, "SLOPES": 0.10, "WIND": 0.08, "VACUUM": 0.05, "PRESSURE": 0.05, "DEBRIS": 0.07},
        "WATER":    {"CLEAR": 0.40, "SAND": 0.10, "WATER": 0.15, "SLOPES": 0.08, "WIND": 0.08, "VACUUM": 0.07, "PRESSURE": 0.07, "DEBRIS": 0.05},
        "SLOPES":   {"CLEAR": 0.40, "SAND": 0.10, "WATER": 0.08, "SLOPES": 0.15, "WIND": 0.10, "VACUUM": 0.05, "PRESSURE": 0.05, "DEBRIS": 0.07},
        "WIND":     {"CLEAR": 0.35, "SAND": 0.10, "WATER": 0.10, "SLOPES": 0.10, "WIND": 0.15, "VACUUM": 0.05, "PRESSURE": 0.05, "DEBRIS": 0.10},
        "VACUUM":   {"CLEAR": 0.50, "SAND": 0.08, "WATER": 0.05, "SLOPES": 0.08, "WIND": 0.08, "VACUUM": 0.10, "PRESSURE": 0.06, "DEBRIS": 0.05},
        "PRESSURE": {"CLEAR": 0.50, "SAND": 0.08, "WATER": 0.08, "SLOPES": 0.05, "WIND": 0.05, "VACUUM": 0.06, "PRESSURE": 0.10, "DEBRIS": 0.08},
        "DEBRIS":   {"CLEAR": 0.40, "SAND": 0.10, "WATER": 0.08, "SLOPES": 0.10, "WIND": 0.10, "VACUUM": 0.05, "PRESSURE": 0.05, "DEBRIS": 0.12},
    }

    # Default physics values (restore when transitioning away)
    DEFAULTS = {
        "gravity": [0.0, 0.0, -9.81],
        "wind": [0.0, 0.0, 0.0],
        "density": 1.2,      # air density kg/m³
        "viscosity": 0.00002,  # air viscosity
        "floor_friction": [0.001, 0.001, 0.0001],
    }

    def __init__(self, transition_interval_s=60.0):
        """
        Args:
            transition_interval_s: Seconds between biome transitions.
        """
        self.current_biome = "CLEAR"
        self.transition_interval = transition_interval_s
        self.timer = 0.0
        self.floor_geom_id = None
        self._debris_active = False
        self._slope_angle = 0.0
        self._wind_vector = np.zeros(3)

    def _resolve_floor(self, model):
        if self.floor_geom_id is None:
            self.floor_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")

    def _sample_next_biome(self):
        row = self.TRANSITION_MATRIX[self.current_biome]
        biomes = list(row.keys())
        probs = list(row.values())
        return np.random.choice(biomes, p=probs)

    def _apply_biome(self, model, biome):
        """Mutate MuJoCo physics parameters to match the target biome."""
        self._resolve_floor(model)

        # Reset to defaults first
        model.opt.gravity[:] = self.DEFAULTS["gravity"]
        model.opt.wind[:] = self.DEFAULTS["wind"]
        model.opt.density = self.DEFAULTS["density"]
        model.opt.viscosity = self.DEFAULTS["viscosity"]
        if self.floor_geom_id is not None and self.floor_geom_id != -1:
            model.geom_friction[self.floor_geom_id] = self.DEFAULTS["floor_friction"]

        self._debris_active = False
        self._slope_angle = 0.0
        self._wind_vector = np.zeros(3)

        if biome == "SAND":
            # High friction — sluggish movement, energy drain
            if self.floor_geom_id is not None and self.floor_geom_id != -1:
                model.geom_friction[self.floor_geom_id] = [3.0, 0.1, 0.01]

        elif biome == "WATER":
            # High density + viscosity — massive drag
            model.opt.density = 50.0        # ~40x air density (simulated submersion)
            model.opt.viscosity = 0.01       # thick fluid resistance

        elif biome == "SLOPES":
            # Tilt gravity vector to simulate hills
            angle = random.uniform(-0.15, 0.15)  # radians (~8.5 degrees max)
            direction = random.uniform(0, 2 * math.pi)
            gx = 9.81 * math.sin(angle) * math.cos(direction)
            gy = 9.81 * math.sin(angle) * math.sin(direction)
            gz = -9.81 * math.cos(angle)
            model.opt.gravity[:] = [gx, gy, gz]
            self._slope_angle = angle

        elif biome == "WIND":
            # Random persistent wind
            wx = random.uniform(-8.0, 8.0)
            wy = random.uniform(-8.0, 8.0)
            wz = random.uniform(-2.0, 2.0)
            model.opt.wind[:] = [wx, wy, wz]
            self._wind_vector = np.array([wx, wy, wz])

        elif biome == "VACUUM":
            # Zero air density — no drag at all
            model.opt.density = 0.0
            model.opt.viscosity = 0.0

        elif biome == "PRESSURE":
            # Extreme viscosity — moving through syrup
            model.opt.density = 200.0
            model.opt.viscosity = 0.1

        elif biome == "DEBRIS":
            self._debris_active = True
            # Floor stays normal, but we signal the caller to spawn debris objects

    def step(self, model, dt):
        """
        Advance the weather timer and potentially transition biomes.

        Args:
            model: MuJoCo MjModel.
            dt: Time delta (seconds).

        Returns:
            dict with 'biome', 'changed' (bool), and biome-specific data.
        """
        self.timer += dt
        changed = False

        if self.timer >= self.transition_interval:
            self.timer = 0.0
            next_biome = self._sample_next_biome()
            if next_biome != self.current_biome:
                print(f"[WEATHER] Biome transition: {self.current_biome} → {next_biome}")
                self.current_biome = next_biome
                self._apply_biome(model, next_biome)
                changed = True

        return {
            "biome": self.current_biome,
            "changed": changed,
            "debris_active": self._debris_active,
            "slope_angle": self._slope_angle,
            "wind_vector": self._wind_vector.copy(),
        }

    def get_energy_drain_multiplier(self):
        """Returns a multiplier for how much extra energy movement costs in this biome."""
        multipliers = {
            "CLEAR": 1.0, "SAND": 2.0, "WATER": 2.5, "SLOPES": 1.8,
            "WIND": 1.3, "VACUUM": 0.8, "PRESSURE": 3.0, "DEBRIS": 1.5,
        }
        return multipliers.get(self.current_biome, 1.0)


# ═══════════════════════════════════════════════════════════════════════════════
#  3. SCARCITY CONTROLLER — Food Rot & Rare Spawns
# ═══════════════════════════════════════════════════════════════════════════════

class ScarcityController:
    """
    Controls food availability. Food "rots" (despawns) after a timeout,
    and new food spawns are rare and randomized.

    This creates genuine starvation pressure: CARL must actively hunt
    for food or risk energy collapse.
    """

    def __init__(self, rot_time_s=45.0, spawn_cooldown_s=30.0):
        """
        Args:
            rot_time_s: Seconds before uneaten food despawns.
            spawn_cooldown_s: Minimum seconds between food spawns.
        """
        self.rot_time = rot_time_s
        self.spawn_cooldown = spawn_cooldown_s
        self.food_timer = 0.0
        self.spawn_timer = 0.0
        self.food_active = True
        self._food_geom_id = None
        self._hidden_pos = np.array([99.0, 99.0, 99.0])

    def _resolve_food(self, model):
        if self._food_geom_id is None:
            self._food_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "food")

    def step(self, model, data, dt, carl_pos, spawn_candidates):
        """
        Advance food timers. Rot active food, spawn new food when ready.

        Args:
            model: MuJoCo MjModel.
            data: MuJoCo MjData.
            dt: Time delta (seconds).
            carl_pos: CARL's (x, y) position.
            spawn_candidates: List of (x, y) tuples for valid food positions.

        Returns:
            dict with 'food_active', 'food_pos', 'rotted' (bool), 'spawned' (bool).
        """
        self._resolve_food(model)
        rotted = False
        spawned = False

        if self.food_active:
            self.food_timer += dt
            if self.food_timer >= self.rot_time:
                # Food rots — despawn it
                if self._food_geom_id is not None and self._food_geom_id != -1:
                    model.geom_pos[self._food_geom_id] = self._hidden_pos
                self.food_active = False
                self.food_timer = 0.0
                self.spawn_timer = 0.0
                rotted = True
                print("[SCARCITY] Food has ROTTED! Starvation pressure increasing...")
        else:
            self.spawn_timer += dt
            if self.spawn_timer >= self.spawn_cooldown and len(spawn_candidates) > 0:
                # Spawn new food at a random valid location, far from CARL
                far_candidates = [c for c in spawn_candidates
                                  if math.hypot(c[0] - carl_pos[0], c[1] - carl_pos[1]) > 2.0]
                if not far_candidates:
                    far_candidates = spawn_candidates

                pos = random.choice(far_candidates)
                if self._food_geom_id is not None and self._food_geom_id != -1:
                    model.geom_pos[self._food_geom_id] = [
                        pos[0] + random.uniform(-0.3, 0.3),
                        pos[1] + random.uniform(-0.3, 0.3),
                        0.12
                    ]
                self.food_active = True
                self.spawn_timer = 0.0
                spawned = True
                print(f"[SCARCITY] New food spawned at ({pos[0]:.1f}, {pos[1]:.1f})")

        food_pos = model.geom_pos[self._food_geom_id].copy() if self._food_geom_id is not None and self._food_geom_id != -1 else self._hidden_pos
        return {
            "food_active": self.food_active,
            "food_pos": food_pos,
            "rotted": rotted,
            "spawned": spawned,
        }

    def on_food_eaten(self):
        """Called when CARL successfully eats the food."""
        self.food_active = False
        self.food_timer = 0.0
        self.spawn_timer = 0.0


# ═══════════════════════════════════════════════════════════════════════════════
#  4. APEX PREDATOR AI
# ═══════════════════════════════════════════════════════════════════════════════

class PredatorAI:
    """
    A simple but terrifying autonomous predator that hunts CARL.

    Behavior:
        - PATROL: Wanders randomly through the maze.
        - HUNT:   Detected CARL. Moves directly toward him.
        - LUNGE:  Within striking distance. Attacks.
        - COOLDOWN: After a successful attack, pauses briefly.

    The predator is implemented as a "ghost body" — we control its position
    directly by setting geom_pos each tick (it has no physics joints).
    """

    def __init__(self, speed=0.35, detection_radius=3.0, attack_radius=0.4,
                 attack_cooldown_s=5.0, damage_per_attack=0.25):
        self.speed = speed
        self.detection_radius = detection_radius
        self.attack_radius = attack_radius
        self.attack_cooldown = attack_cooldown_s
        self.damage_per_attack = damage_per_attack

        self.pos = np.array([4.0, 4.0, 0.15])  # Start in a corner
        self.state = "PATROL"
        self.patrol_target = None
        self.cooldown_timer = 0.0
        self._geom_id = None
        self._glow_id = None

    def _resolve_geoms(self, model):
        if self._geom_id is None:
            self._geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "predator")
            self._glow_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "predator_glow")

    def step(self, model, dt, carl_pos, arena_bounds=(-4.5, 4.5)):
        """
        Advance predator state machine.

        Args:
            model: MuJoCo MjModel.
            dt: Time delta (seconds).
            carl_pos: CARL's (x, y) position as numpy array.
            arena_bounds: (min, max) coordinate bounds.

        Returns:
            dict with 'state', 'pos', 'attacked' (bool), 'damage'.
        """
        self._resolve_geoms(model)
        attacked = False
        damage = 0.0
        dist_to_carl = math.hypot(self.pos[0] - carl_pos[0], self.pos[1] - carl_pos[1])

        if self.state == "COOLDOWN":
            self.cooldown_timer -= dt
            if self.cooldown_timer <= 0:
                self.state = "PATROL"
                self.patrol_target = None

        elif self.state == "PATROL":
            # Wander randomly
            if self.patrol_target is None or math.hypot(
                    self.pos[0] - self.patrol_target[0],
                    self.pos[1] - self.patrol_target[1]) < 0.3:
                self.patrol_target = np.array([
                    random.uniform(arena_bounds[0], arena_bounds[1]),
                    random.uniform(arena_bounds[0], arena_bounds[1]),
                ])

            # Move toward patrol target
            dx = self.patrol_target[0] - self.pos[0]
            dy = self.patrol_target[1] - self.pos[1]
            dist = math.hypot(dx, dy) + 1e-8
            self.pos[0] += (dx / dist) * self.speed * 0.5 * dt
            self.pos[1] += (dy / dist) * self.speed * 0.5 * dt

            # Check if CARL is detected
            if dist_to_carl < self.detection_radius:
                self.state = "HUNT"
                print(f"[PREDATOR] CARL DETECTED at {dist_to_carl:.1f}m! Switching to HUNT mode!")

        elif self.state == "HUNT":
            # Move directly toward CARL
            dx = carl_pos[0] - self.pos[0]
            dy = carl_pos[1] - self.pos[1]
            dist = math.hypot(dx, dy) + 1e-8
            self.pos[0] += (dx / dist) * self.speed * dt
            self.pos[1] += (dy / dist) * self.speed * dt

            if dist_to_carl < self.attack_radius:
                self.state = "LUNGE"
            elif dist_to_carl > self.detection_radius * 1.5:
                # Lost sight of CARL
                self.state = "PATROL"
                self.patrol_target = None
                print("[PREDATOR] Lost sight of CARL. Returning to PATROL.")

        elif self.state == "LUNGE":
            # Attack!
            attacked = True
            damage = self.damage_per_attack
            self.state = "COOLDOWN"
            self.cooldown_timer = self.attack_cooldown
            print(f"[PREDATOR] *** ATTACK! *** CARL takes {damage:.0%} damage!")

        # Clamp to arena
        lo, hi = arena_bounds
        self.pos[0] = np.clip(self.pos[0], lo, hi)
        self.pos[1] = np.clip(self.pos[1], lo, hi)

        # Update visual position
        if self._geom_id is not None and self._geom_id != -1:
            model.geom_pos[self._geom_id] = self.pos.copy()
        if self._glow_id is not None and self._glow_id != -1:
            model.geom_pos[self._glow_id] = [self.pos[0], self.pos[1], self.pos[2] + 0.31]

        return {
            "state": self.state,
            "pos": self.pos.copy(),
            "attacked": attacked,
            "damage": damage,
            "dist_to_carl": dist_to_carl,
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  5. POLTERGEIST EVENT — Random Projectile Launcher
# ═══════════════════════════════════════════════════════════════════════════════

class PoltergeistEvent:
    """
    Randomly teleports a physics toy into the air and fires it at CARL.
    Creates sudden, unpredictable threats that require fast reflexes.
    """

    def __init__(self, interval_range_s=(60.0, 180.0), launch_height=5.0, launch_speed=6.0):
        self.interval_range = interval_range_s
        self.launch_height = launch_height
        self.launch_speed = launch_speed
        self.timer = random.uniform(*interval_range_s)
        self.projectile_names = ['obj_cube_0', 'obj_cube_1', 'obj_cube_2', 'obj_ball_0', 'obj_ball_1']
        self.active_projectile = None  # body_id of currently launched projectile
        self.launch_time = 0.0

    def step(self, model, data, dt, carl_pos):
        """
        Advance timer. When triggered, launch a projectile at CARL.

        Returns:
            dict with 'launched' (bool), 'projectile_name', 'velocity_vector'.
        """
        self.timer -= dt
        launched = False
        proj_name = None
        vel_vec = None

        if self.timer <= 0:
            self.timer = random.uniform(*self.interval_range)

            # Pick a random toy
            proj_name = random.choice(self.projectile_names)
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, proj_name)
            if body_id != -1:
                jnt_id = model.body_jntadr[body_id]
                if jnt_id != -1 and model.jnt_type[jnt_id] == mujoco.mjtJoint.mjJNT_FREE:
                    qpos_adr = model.jnt_qposadr[jnt_id]
                    dof_adr = model.jnt_dofadr[jnt_id]

                    # Teleport above CARL with lateral offset
                    offset_x = random.uniform(-1.5, 1.5)
                    offset_y = random.uniform(-1.5, 1.5)
                    spawn_x = carl_pos[0] + offset_x
                    spawn_y = carl_pos[1] + offset_y
                    spawn_z = self.launch_height

                    data.qpos[qpos_adr]     = spawn_x
                    data.qpos[qpos_adr + 1] = spawn_y
                    data.qpos[qpos_adr + 2] = spawn_z

                    # Reset quaternion to identity
                    data.qpos[qpos_adr + 3] = 1.0
                    data.qpos[qpos_adr + 4] = 0.0
                    data.qpos[qpos_adr + 5] = 0.0
                    data.qpos[qpos_adr + 6] = 0.0

                    # Compute velocity vector aimed at CARL's chassis
                    dx = carl_pos[0] - spawn_x
                    dy = carl_pos[1] - spawn_y
                    dz = 0.05 - spawn_z  # Target ground level
                    dist = math.sqrt(dx*dx + dy*dy + dz*dz) + 1e-8
                    vel_vec = np.array([
                        dx / dist * self.launch_speed,
                        dy / dist * self.launch_speed,
                        dz / dist * self.launch_speed,
                    ])

                    data.qvel[dof_adr]     = vel_vec[0]
                    data.qvel[dof_adr + 1] = vel_vec[1]
                    data.qvel[dof_adr + 2] = vel_vec[2]
                    # Zero angular velocity
                    data.qvel[dof_adr + 3] = 0.0
                    data.qvel[dof_adr + 4] = 0.0
                    data.qvel[dof_adr + 5] = 0.0

                    self.active_projectile = body_id
                    self.launch_time = time.time()
                    launched = True
                    print(f"[POLTERGEIST] *** PROJECTILE LAUNCHED: {proj_name} from ({spawn_x:.1f}, {spawn_y:.1f}, {spawn_z:.1f})! ***")

        return {
            "launched": launched,
            "projectile_name": proj_name,
            "velocity_vector": vel_vec,
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  6. NEUROMORPHIC DVS — Simulated Dynamic Vision Sensor
# ═══════════════════════════════════════════════════════════════════════════════

class NeuromorphicDVS:
    """
    Simulates a Dynamic Vision Sensor (event camera) by monitoring
    the velocity of all free-body objects in the scene.

    A real DVS fires asynchronous spikes when pixel intensity changes.
    We approximate this by detecting objects whose velocity magnitude
    exceeds a threshold — indicating rapid movement toward CARL.

    This provides 360-degree, zero-latency "looming detection."
    """

    def __init__(self, velocity_threshold=2.0, proximity_radius=3.0):
        self.velocity_threshold = velocity_threshold
        self.proximity_radius = proximity_radius
        self.tracked_bodies = []
        self._resolved = False

    def _resolve_bodies(self, model):
        if self._resolved:
            return
        names = ['obj_cube_0', 'obj_cube_1', 'obj_cube_2', 'obj_ball_0', 'obj_ball_1', 'obj_toy']
        for name in names:
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid != -1:
                jnt_id = model.body_jntadr[bid]
                if jnt_id != -1 and model.jnt_type[jnt_id] == mujoco.mjtJoint.mjJNT_FREE:
                    self.tracked_bodies.append({
                        "name": name,
                        "body_id": bid,
                        "dof_adr": model.jnt_dofadr[jnt_id],
                    })
        self._resolved = True

    def step(self, model, data, carl_pos):
        """
        Scan all tracked bodies for high-velocity looming threats.

        Returns:
            dict with 'threat_detected' (bool), 'threats' (list of threat dicts).
            Each threat has: 'name', 'pos', 'vel', 'speed', 'closing_speed',
                             'distance', 'time_to_impact'.
        """
        self._resolve_bodies(model)
        threats = []

        for body in self.tracked_bodies:
            pos = data.xpos[body["body_id"]].copy()
            dof = body["dof_adr"]
            vel = data.qvel[dof:dof + 3].copy()
            speed = np.linalg.norm(vel)

            # Distance to CARL
            dx = carl_pos[0] - pos[0]
            dy = carl_pos[1] - pos[1]
            dz = 0.05 - pos[2]  # CARL is at ground level
            dist = math.sqrt(dx*dx + dy*dy + dz*dz)

            if dist > self.proximity_radius or speed < self.velocity_threshold:
                continue

            # Compute closing speed (negative = approaching)
            direction_to_carl = np.array([dx, dy, dz])
            direction_to_carl /= (np.linalg.norm(direction_to_carl) + 1e-8)
            closing_speed = np.dot(vel, direction_to_carl)

            if closing_speed > 0.5:  # Object is moving TOWARD CARL
                time_to_impact = dist / (closing_speed + 1e-8)
                threats.append({
                    "name": body["name"],
                    "pos": pos,
                    "vel": vel,
                    "speed": speed,
                    "closing_speed": closing_speed,
                    "distance": dist,
                    "time_to_impact": time_to_impact,
                })

        if threats:
            most_urgent = min(threats, key=lambda t: t["time_to_impact"])
            print(f"[DVS SPIKE] *** LOOMING THREAT: {most_urgent['name']} — "
                  f"impact in {most_urgent['time_to_impact']:.2f}s, "
                  f"speed {most_urgent['speed']:.1f} m/s ***")

        return {
            "threat_detected": len(threats) > 0,
            "threats": threats,
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  7. MPC DODGE — Model Predictive Control Evasion
# ═══════════════════════════════════════════════════════════════════════════════

class MPCDodge:
    """
    Given a detected threat (position, velocity), predicts the projectile's
    future trajectory using Newtonian ballistics and computes the optimal
    evasion motor command to dodge.

    This is a simplified single-step MPC: predict impact point → compute
    perpendicular escape vector → output motor commands.
    """

    def __init__(self, dodge_speed=0.4, gravity=-9.81):
        self.dodge_speed = dodge_speed
        self.gravity = gravity

    def compute_evasion(self, carl_pos, carl_yaw, threat):
        """
        Compute optimal dodge motor commands given a threat.

        Args:
            carl_pos: (x, y) CARL position.
            carl_yaw: CARL heading in radians.
            threat: dict from NeuromorphicDVS with 'pos', 'vel', 'time_to_impact'.

        Returns:
            dict with 'v_dodge' (forward speed), 'w_dodge' (angular speed),
                       'impact_point' (predicted x,y,z), 'dodge_direction' (angle).
        """
        # Predict impact point using ballistic equations
        t = threat["time_to_impact"]
        px = threat["pos"][0] + threat["vel"][0] * t
        py = threat["pos"][1] + threat["vel"][1] * t
        pz = threat["pos"][2] + threat["vel"][2] * t + 0.5 * self.gravity * t * t
        impact_point = np.array([px, py, max(0, pz)])

        # Compute escape vector: perpendicular to the threat's velocity (2D)
        threat_dir_2d = np.array([threat["vel"][0], threat["vel"][1]])
        threat_dir_norm = np.linalg.norm(threat_dir_2d) + 1e-8
        threat_dir_2d /= threat_dir_norm

        # Two perpendicular options — pick the one that moves CARL away from impact
        perp_a = np.array([-threat_dir_2d[1], threat_dir_2d[0]])
        perp_b = np.array([threat_dir_2d[1], -threat_dir_2d[0]])

        carl_to_impact = impact_point[:2] - np.array(carl_pos[:2])
        # Choose the perpendicular that takes us AWAY from the impact point
        if np.dot(perp_a, carl_to_impact) < 0:
            escape_dir = perp_a
        else:
            escape_dir = perp_b

        # Convert escape direction to motor commands (v, w)
        escape_angle = math.atan2(escape_dir[1], escape_dir[0])
        rel_angle = escape_angle - carl_yaw
        rel_angle = (rel_angle + math.pi) % (2 * math.pi) - math.pi

        v_dodge = self.dodge_speed
        w_dodge = np.clip(rel_angle * 3.0, -1.5, 1.5)

        return {
            "v_dodge": v_dodge,
            "w_dodge": w_dodge,
            "impact_point": impact_point,
            "dodge_direction": escape_angle,
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  8. MAZE SHIFTER — Periodic Labyrinth Reconfiguration
# ═══════════════════════════════════════════════════════════════════════════════

class MazeShifter:
    """
    Periodically regenerates the maze layout using carl_maze_gen and
    repositions the 16 "ghost wall" geoms in the MuJoCo model.

    The walls are pre-allocated in the XML with names 'ghost_wall_0' through
    'ghost_wall_15'. This class moves them to new positions.
    """

    def __init__(self, shift_interval_s=120.0, n_walls=16, difficulty=3):
        self.shift_interval = shift_interval_s
        self.n_walls = n_walls
        self.difficulty = difficulty
        self.timer = shift_interval_s * 0.5  # First shift happens sooner
        self._geom_ids = None
        self._glow_ids = None
        self._hidden_pos = np.array([99.0, 99.0, 99.0])
        self.current_maze = None
        self.shift_count = 0

    def _resolve_geoms(self, model):
        if self._geom_ids is not None:
            return
        self._geom_ids = []
        self._glow_ids = []
        for i in range(self.n_walls):
            gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"ghost_wall_{i}")
            self._geom_ids.append(gid)
            glow_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"ghost_glow_{i}")
            self._glow_ids.append(glow_id)

    def step(self, model, dt):
        """
        Advance timer and shift maze when ready.

        Returns:
            dict with 'shifted' (bool), 'maze_data', 'shift_count'.
        """
        self._resolve_geoms(model)
        self.timer += dt
        shifted = False

        if self.timer >= self.shift_interval:
            self.timer = 0.0
            self.shift_count += 1

            # Generate new maze
            seed = int(time.time() * 1000) % 100000
            self.current_maze = generate_maze(rows=5, cols=5, difficulty=self.difficulty, seed=seed)
            walls = self.current_maze["walls"]

            print(f"[MAZE] *** GROUND SHAKING! Maze reconfiguration #{self.shift_count} — "
                  f"{len(walls)} walls generated ***")

            # Position the ghost walls
            for i in range(self.n_walls):
                gid = self._geom_ids[i]
                if gid == -1:
                    continue

                if i < len(walls):
                    w = walls[i]
                    model.geom_pos[gid] = w["pos"]
                    model.geom_size[gid] = w["size"]
                    # Set glow cap
                    if self._glow_ids[i] != -1:
                        model.geom_pos[self._glow_ids[i]] = [
                            w["pos"][0], w["pos"][1], w["pos"][2] + w["size"][2] + 0.004
                        ]
                        # Match orientation
                        model.geom_size[self._glow_ids[i]] = [w["size"][0], w["size"][1], 0.004]
                else:
                    # Hide unused walls off-screen
                    model.geom_pos[gid] = self._hidden_pos
                    if self._glow_ids[i] != -1:
                        model.geom_pos[self._glow_ids[i]] = self._hidden_pos

            shifted = True

        return {
            "shifted": shifted,
            "maze_data": self.current_maze,
            "shift_count": self.shift_count,
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  9. OBJECT PERMANENCE — Spatial Memory for Occluded Objects
# ═══════════════════════════════════════════════════════════════════════════════

class ObjectPermanenceMemory:
    """
    Maintains a persistent spatial memory of object locations.
    Even when objects are occluded by walls or out of sensor range,
    CARL remembers where they were last seen.

    This is a fundamental milestone of cognitive development:
    knowing something exists even when it cannot be perceived.
    """

    def __init__(self):
        self.memory = {}  # name -> {"pos": np.array, "last_seen": float, "confidence": float}
        self.t = 0.0
        self.decay_rate = 0.005  # Confidence decays over time

    def update(self, name, pos, visible=True):
        """
        Update memory for a named object.

        Args:
            name: Object identifier string.
            pos: (x, y, z) position as numpy array.
            visible: Whether the object is currently visible.
        """
        if visible:
            self.memory[name] = {
                "pos": np.array(pos, dtype=np.float64),
                "last_seen": self.t,
                "confidence": 1.0,
            }

    def step(self, dt):
        """Advance time and decay confidence of unseen objects."""
        self.t += dt
        for name, mem in self.memory.items():
            age = self.t - mem["last_seen"]
            mem["confidence"] = max(0.0, 1.0 - age * self.decay_rate)

    def recall(self, name):
        """
        Retrieve the last known position and confidence for an object.

        Returns:
            (pos, confidence) tuple, or (None, 0.0) if unknown.
        """
        if name in self.memory and self.memory[name]["confidence"] > 0.01:
            mem = self.memory[name]
            return mem["pos"].copy(), mem["confidence"]
        return None, 0.0

    def get_all_remembered(self, min_confidence=0.1):
        """Return all objects with confidence above threshold."""
        return {
            name: mem for name, mem in self.memory.items()
            if mem["confidence"] >= min_confidence
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  10. VISUAL SERVOING — PID Face-Tracking
# ═══════════════════════════════════════════════════════════════════════════════

class VisualServoPID:
    """
    Continuous PID controller for face-tracking social fixation.

    Takes the horizontal offset of a detected face (from camera center)
    and outputs a steering correction to keep the face dead-center.
    """

    def __init__(self, kp=2.0, ki=0.1, kd=0.5):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0.0
        self.prev_error = 0.0
        self.active = False

    def compute(self, face_center_x, frame_center_x, dt):
        """
        Compute steering correction.

        Args:
            face_center_x: Horizontal pixel position of detected face center.
            frame_center_x: Horizontal pixel position of frame center.
            dt: Time delta (seconds).

        Returns:
            float: Angular velocity correction (w_servo). Positive = turn left.
        """
        # Normalize error to [-1, 1] range
        error = (frame_center_x - face_center_x) / (frame_center_x + 1e-8)

        self.integral += error * dt
        self.integral = np.clip(self.integral, -2.0, 2.0)  # Anti-windup

        derivative = (error - self.prev_error) / (dt + 1e-8)
        self.prev_error = error

        w_servo = self.kp * error + self.ki * self.integral + self.kd * derivative
        return np.clip(w_servo, -1.0, 1.0)

    def reset(self):
        """Reset PID state when face is lost."""
        self.integral = 0.0
        self.prev_error = 0.0
        self.active = False


# ═══════════════════════════════════════════════════════════════════════════════
#  MASTER CONTROLLER — The Crucible
# ═══════════════════════════════════════════════════════════════════════════════

class CrucibleController:
    """
    Master orchestrator for the entire dynamic environment.

    Composes all subsystems and provides a single step() interface
    for the main loop in carl_harvest.py.
    """

    def __init__(self,
                 circadian_period_s=300.0,
                 biome_interval_s=60.0,
                 food_rot_s=45.0,
                 food_spawn_cooldown_s=30.0,
                 maze_shift_interval_s=120.0,
                 poltergeist_interval_range=(60.0, 180.0)):

        self.circadian = CircadianClock(cycle_period_s=circadian_period_s)
        self.weather = BiomeWeatherEngine(transition_interval_s=biome_interval_s)
        self.scarcity = ScarcityController(rot_time_s=food_rot_s, spawn_cooldown_s=food_spawn_cooldown_s)
        self.predator = PredatorAI()
        self.poltergeist = PoltergeistEvent(interval_range_s=poltergeist_interval_range)
        self.dvs = NeuromorphicDVS()
        self.mpc = MPCDodge()
        self.maze_shifter = MazeShifter(shift_interval_s=maze_shift_interval_s)
        self.object_memory = ObjectPermanenceMemory()
        self.visual_servo = VisualServoPID()

        self.tick_count = 0
        self._initialized = False

    def step(self, model, data, dt, carl_pos, carl_yaw, brain_drives,
             spawn_candidates=None, face_data=None):
        """
        Advance all Crucible subsystems by one tick.

        Args:
            model: MuJoCo MjModel.
            data: MuJoCo MjData.
            dt: Time delta (seconds).
            carl_pos: (x, y) CARL position.
            carl_yaw: CARL heading (radians).
            brain_drives: CARL's brain.drives object for neuromodulator injection.
            spawn_candidates: List of (x, y) for food respawn positions.
            face_data: dict with 'detected' (bool), 'center_x', 'frame_center_x', or None.

        Returns:
            CrucibleState dict with all subsystem outputs and motor overrides.
        """
        self.tick_count += 1
        if spawn_candidates is None:
            spawn_candidates = [(0, 0)]

        # 1. Circadian
        circadian = self.circadian.step(model, dt)

        # --- BABY CARL MODE ---
        # Freezing the environment to focus on cumulative competence
        BABY_CARL_MODE = True
        
        if BABY_CARL_MODE:
            weather = {
                "biome": "normal",
                "changed": False,
                "debris_active": False,
                "slope_angle": 0.0,
                "wind_vector": np.zeros(3)
            }
            drain_mult = 1.0
            scarcity = self.scarcity.step(model, data, dt, carl_pos, spawn_candidates)
            predator = {"state": "PATROL", "pos": np.array([10.0, 10.0]), "attacked": False, "damage": 0.0, "dist_to_carl": 10.0}
            poltergeist = {"active": False, "projectiles": []}
            dvs = {"threat_detected": False, "threats": []}
            mpc_result = None
            maze = {"shifted": False}
        else:
            # 2. Weather / Biome
            weather = self.weather.step(model, dt)
            drain_mult = self.weather.get_energy_drain_multiplier()
            if drain_mult > 1.0 and self.tick_count % 50 == 0:
                energy_cost = (drain_mult - 1.0) * 0.001
                brain_drives.energy = max(0.0, brain_drives.energy - energy_cost)
    
            # 3. Scarcity
            scarcity = self.scarcity.step(model, data, dt, carl_pos, spawn_candidates)
            if scarcity["rotted"]:
                brain_drives.cort = min(1.0, brain_drives.cort + 0.15)
    
            # 4. Predator
            predator = self.predator.step(model, dt, carl_pos)
            if predator["attacked"]:
                brain_drives.cort = min(1.0, brain_drives.cort + 0.4)
                brain_drives.energy = max(0.0, brain_drives.energy - predator["damage"])
                brain_drives.damage = min(1.0, brain_drives.damage + predator["damage"] * 0.5)
            elif predator["state"] == "HUNT" and predator["dist_to_carl"] < 2.0:
                brain_drives.cort = min(1.0, brain_drives.cort + 0.02)
    
            # 5. Poltergeist
            poltergeist = self.poltergeist.step(model, data, dt, carl_pos)
    
            # 6. DVS threat detection
            dvs = self.dvs.step(model, data, carl_pos)
    
            # 7. MPC dodge computation
            mpc_result = None
            if dvs["threat_detected"]:
                most_urgent = min(dvs["threats"], key=lambda t: t["time_to_impact"])
                mpc_result = self.mpc.compute_evasion(carl_pos, carl_yaw, most_urgent)
                brain_drives.cort = min(1.0, brain_drives.cort + 0.3)
    
            # 8. Maze shifting
            maze = self.maze_shifter.step(model, dt)
            if maze["shifted"]:
                brain_drives.cort = min(1.0, brain_drives.cort + 0.1)

        # 9. Object permanence
        self.object_memory.step(dt)

        # 10. Visual servoing
        servo_w = None
        if face_data is not None and face_data.get("detected", False):
            self.visual_servo.active = True
            servo_w = self.visual_servo.compute(
                face_data["center_x"], face_data["frame_center_x"], dt
            )
        else:
            if self.visual_servo.active:
                self.visual_servo.reset()

        return {
            "circadian": circadian,
            "weather": weather,
            "scarcity": scarcity,
            "predator": predator,
            "poltergeist": poltergeist,
            "dvs": dvs,
            "mpc_dodge": mpc_result,
            "maze": maze,
            "servo_w": servo_w,
            "energy_drain_multiplier": drain_mult,
        }

    def get_flee_direction(self, carl_pos, carl_yaw):
        """
        Compute motor commands to flee from the predator.

        Returns:
            (v_flee, w_flee) or None if predator is not hunting.
        """
        if self.predator.state not in ("HUNT", "LUNGE"):
            return None

        dx = carl_pos[0] - self.predator.pos[0]
        dy = carl_pos[1] - self.predator.pos[1]
        flee_angle = math.atan2(dy, dx)
        rel_angle = flee_angle - carl_yaw
        rel_angle = (rel_angle + math.pi) % (2 * math.pi) - math.pi

        v_flee = 0.8  # Full speed — away from predator (must outrun it)
        w_flee = np.clip(rel_angle * 3.0, -1.5, 1.5)
        return v_flee, w_flee
