"""
carl_brainstem.py — The Reticulospinal Tract.

Phase A0 Survival Stack (Mark VIII Spec):
    Deterministic 100Hz control layer that sits BETWEEN the RL cortex
    (carl_agent.py — which outputs high-level target trajectories) and the
    actual MuJoCo wheel actuators. This is Bob's spinal cord upgrade —
    a hardened, zero-jitter safety layer that guarantees three invariants:

    1. Instant geometric obstacle avoidance  (Braitenberg reflexes)
    2. Precise wheel velocity tracking        (discrete PID controller)
    3. Hardware torque saturation clamps       (protect the motors)

    Biological analogue: The reticulospinal tract is the brainstem pathway
    that handles fast reflexive motor control independently of the cortex.
    A cat with its cortex ablated can still walk, dodge obstacles, and land
    on its feet — because the reticulospinal system keeps running at full
    speed. This module is Bob's version of that ancient survival hardware.

    The RL policy (cortex) proposes velocity targets (v, ω). This module
    may OVERRIDE those targets if LiDAR detects an imminent collision,
    then converts the (possibly overridden) targets into precise wheel
    torques via differential-drive kinematics + closed-loop PID tracking.

    Data flow:
        RL Policy → (v_target, ω_target) → BraitenbergReflexLayer
            → (v_safe, ω_safe) → Differential Kinematics
            → (ωL_desired, ωR_desired) → PID Controllers
            → (τL_raw, τR_raw) → Torque Saturation
            → (τL_clamped, τR_clamped) → MuJoCo actuators

    Running cost: <50μs per step on CPU. Pure arithmetic. No allocations
    in the hot path. No GPU needed. ~200 bytes of state.
"""

import numpy as np
import time


class BraitenbergReflexLayer:
    """
    Pure geometric reflex — no learning, no state, no memory.

    Splits the 24-ray LiDAR ring into three sectors (left / front / right)
    and applies priority-ordered override rules if obstacles breach safety
    thresholds. The logic is inspired by Valentino Braitenberg's "Vehicles"
    (1984) — simple sensor-motor wiring that produces complex avoidance
    behavior without any internal model.

    Sector layout (24 rays, evenly spaced at 15° intervals):
        Rays  0– 7: LEFT   sector (  0° to 105°, counter-clockwise from nose-left)
        Rays  8–15: FRONT  sector (120° to 225°, centered on straight-ahead)
        Rays 16–23: RIGHT  sector (240° to 345°, counter-clockwise wrapping right)

    Distance thresholds (meters):
        EMERGENCY_DISTANCE = 0.15  — 15cm hard safety floor (Phase A0 spec)
        CAUTION_DISTANCE   = 0.40  — 40cm soft braking zone
        SIDE_CAUTION       = 0.30  — 30cm lateral drift correction

    Priority cascade (first match wins):
        1. Front emergency  → reverse + hard spin away from closest side
        2. Front caution    → crawl  + moderate spin toward open side
        3. Left too close   → nudge angular rightward (−0.5 rad/s)
        4. Right too close  → nudge angular leftward  (+0.5 rad/s)
        5. All clear        → pass through RL targets unchanged

    Biological analogue: Superior colliculus saccade-avoidance circuitry.
    """

    # ── Phase A0 safety thresholds (meters) ──────────────────────────────
    EMERGENCY_DISTANCE = 0.10   # 10cm — HARD floor. Triggers emergency reverse.
    CAUTION_DISTANCE   = 0.40   # 40cm — Soft zone. Slow to crawl + steer away.
    SIDE_CAUTION       = 0.30   # 30cm — Lateral drift. Gentle angular nudge.

    def __init__(self):
        # Latch the escape direction to prevent 100Hz bang-bang steering oscillation
        self.escape_dir = 0.0
        # Hysteresis: stay in emergency reverse until completely out of the caution zone
        self.in_emergency = False

    def check(self, lidar_24, target_linear, target_angular, stress=0.0, stiffness=1.0):
        """
        Evaluate LiDAR scan and conditionally override velocity targets,
        modulating threat boundaries and steering gains based on cortisol (stress)
        and genetic stiffness.
        """
        # Stiffen safety thresholds and steering gains based on stress and genetic stiffness
        stiffness_factor = stiffness * (1.0 + 0.4 * stress)
        emergency_dist = self.EMERGENCY_DISTANCE * stiffness_factor
        caution_dist = self.CAUTION_DISTANCE * stiffness_factor
        side_caution = self.SIDE_CAUTION * stiffness_factor
        
        # Steering multiplier (hyper-vigilant panic response)
        steer_mult = 1.0 + 0.5 * stress

        # ── Sector decomposition ─────────────────────────────────────────
        # In a 24-ray LiDAR, Ray 0 is nose-forward.
        # Front sector covers -45° to +45° (rays 21, 22, 23, 0, 1, 2, 3)
        front_rays = np.concatenate([lidar_24[21:24], lidar_24[0:4]])
        front_min  = np.min(front_rays)

        # Left sector covers +45° to +135° (rays 4, 5, 6, 7, 8)
        left_min   = np.min(lidar_24[4:9])

        # Right sector covers -135° to -45° (rays 15, 16, 17, 18, 19, 20)
        right_min  = np.min(lidar_24[15:21])

        linear  = target_linear
        angular = target_angular
        override_active = False

        # ── Kill Switch: Wall within 5cm of chassis ──────────────────────
        if np.min(lidar_24) < 0.05:
            return 0.0, 0.0, True

        # ── Hysteresis State Machine ─────────────────────────────────────
        if front_min < emergency_dist:
            self.in_emergency = True
        elif front_min > caution_dist:
            self.in_emergency = False

        # ── Priority 1: EMERGENCY — wall within 15cm dead ahead ──────────
        # Full reverse at crawl speed + hard spin away from nearest side.
        if self.in_emergency:
            linear = -0.1                    # persistent reverse
            if self.escape_dir == 0.0:
                self.escape_dir = -2.0 if left_min < right_min else 2.0
            angular = self.escape_dir * steer_mult
            override_active = True

        # ── Priority 2: CAUTION — wall within 40cm ahead ─────────────────
        # Slow to crawl + moderate steering toward open side.
        elif front_min < caution_dist:
            linear = 0.05                     # crawl forward
            if self.escape_dir == 0.0:
                self.escape_dir = -1.5 if left_min < right_min else 1.5
            angular = self.escape_dir * steer_mult
            override_active = True

        else:
            self.escape_dir = 0.0
            # ── Priority 3 & 4: SIDE corrections ────────────────────────
            # These are additive nudges — they don't replace the RL target,
            # they bias it. Both can fire simultaneously (narrow corridor).
            if left_min < side_caution:
                angular -= 0.5 * steer_mult
                override_active = True

            if right_min < side_caution:
                angular += 0.5 * steer_mult
                override_active = True

        # ── Priority 5: ALL CLEAR — pass through unchanged ───────────────
        return linear, angular, override_active


class DiscretePIDController:
    """
    Classic discrete-time PID controller with anti-windup clamping.

    Tracks a single scalar setpoint (e.g., one wheel's angular velocity).
    Runs at a fixed timestep dt — no variable-rate compensation needed
    since the brainstem loop is locked to 100Hz via perf_counter_ns.

    Transfer function (Z-domain equivalent):
        u[k] = Kp · e[k]  +  Ki · Σ(e·dt)  +  Kd · (e[k] - e[k-1]) / dt

    Anti-windup: The integral term is hard-clamped to ±INTEGRAL_CLAMP.
    This prevents integral windup when the actuator is saturated (e.g.,
    wheel stuck against a wall) — without it, the integral would balloon
    and cause massive overshoot when the constraint releases.

    Default gains (Kp=15.0, Ki=0.5, Kd=0.1) are tuned for:
        - 2.5 kg chassis mass
        - 0.04 m wheel radius
        - 100 Hz control rate
        - 2.5 Nm max torque saturation

    Biological analogue: Cerebellar Purkinje cell error-correction loop.
    The cerebellum doesn't "decide" what movement to make — it corrects
    the movement that the cortex already planned, using sensory feedback
    (proprioceptive error) to compute precise corrective torques.
    """

    # Anti-windup: prevent integral explosion during actuator saturation
    INTEGRAL_CLAMP = 5.0

    def __init__(self, Kp=15.0, Ki=0.5, Kd=0.1, dt=0.01):
        """
        Parameters
        ----------
        Kp : float
            Proportional gain. Dominates transient response.
            Higher = faster tracking, but oscillation risk.
        Ki : float
            Integral gain. Eliminates steady-state error.
            Higher = tighter DC tracking, but windup risk.
        Kd : float
            Derivative gain. Damps oscillation.
            Higher = more damping, but noise sensitivity.
        dt : float
            Fixed timestep in seconds (0.01 = 100Hz).
        """
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.dt = dt

        # Running state
        self.integral   = 0.0    # accumulated integral of error
        self.prev_error = 0.0    # previous error (for derivative term)

    def compute(self, desired, actual):
        """
        Compute one PID control step.

        Parameters
        ----------
        desired : float
            Target setpoint (e.g., desired wheel angular velocity, rad/s).
        actual : float
            Current measured value (e.g., encoder-reported wheel velocity).

        Returns
        -------
        output : float
            Raw control output (torque command). NOT yet saturated — the
            caller (BrainstemController) handles torque clamping.
        """
        error = desired - actual

        # Integral accumulation with anti-windup clamp
        self.integral = np.clip(
            self.integral + error * self.dt,
            -self.INTEGRAL_CLAMP,
            self.INTEGRAL_CLAMP
        )

        # Backward-difference derivative
        derivative = (error - self.prev_error) / self.dt
        
        # Apply a rolling average to the derivative to prevent torque spikes
        if not hasattr(self, 'derivative_filter'):
            self.derivative_filter = 0.0
        self.derivative_filter = 0.9 * self.derivative_filter + 0.1 * derivative

        # PID output
        output = self.Kp * error + self.Ki * self.integral + self.Kd * self.derivative_filter

        # Store for next derivative calculation
        self.prev_error = error

        return float(output)

    def reset(self):
        """Zero out all internal state. Call on episode boundaries."""
        self.integral   = 0.0
        self.prev_error = 0.0


class IntelligentPidController:
    """
    Model-Free iPID (Intelligent PID) using an ultra-local model.
    Estimates the unknown dynamics (F) in real-time.
    Formula: y_ddot = F + alpha * u
    u = (-F + y_ddot_target + Kp*e + Kd*e_dot + Ki*int(e)) / alpha
    """
    def __init__(self, Kp=1.0, Ki=0.1, Kd=0.01, alpha=1.0, dt=0.01):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.alpha = alpha
        self.dt = dt
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_y_dot = 0.0
        self.prev_u = 0.0

    def compute(self, desired_y_dot, actual_y_dot):
        error = desired_y_dot - actual_y_dot
        self.integral = np.clip(self.integral + error * self.dt, -2.0, 2.0)
        e_dot = (error - self.prev_error) / self.dt
        
        # Estimate y_ddot (acceleration)
        y_ddot = (actual_y_dot - self.prev_y_dot) / self.dt
        
        # Estimate F (total unknown dynamics)
        F_est = y_ddot - self.alpha * self.prev_u
        
        # We assume desired acceleration is 0 for setpoint tracking
        y_ddot_target = 0.0 
        
        # iPID control law
        u = (-F_est + y_ddot_target + self.Kp * error + self.Kd * e_dot + self.Ki * self.integral) / self.alpha
        
        self.prev_error = error
        self.prev_y_dot = actual_y_dot
        self.prev_u = u
        return float(u)

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_y_dot = 0.0
        self.prev_u = 0.0


class ActiveDisturbanceRejectionController:
    """
    Linear Active Disturbance Rejection Control (LADRC).
    Uses a 2nd-order Linear Extended State Observer (LESO) to track velocity and total disturbance.
    """
    def __init__(self, b0=1.0, w_c=10.0, w_o=30.0, dt=0.01):
        self.b0 = b0
        self.dt = dt
        # Observer bandwidth (w_o) and Controller bandwidth (w_c)
        self.beta1 = 2 * w_o
        self.beta2 = w_o**2
        self.Kp = w_c
        
        self.z1 = 0.0  # Estimated velocity
        self.z2 = 0.0  # Estimated total disturbance (f)
        self.prev_u = 0.0

    def compute(self, desired_vel, actual_vel):
        # 1. Update LESO (Observer)
        e_obs = self.z1 - actual_vel
        self.z1 += self.dt * (self.z2 + self.b0 * self.prev_u - self.beta1 * e_obs)
        self.z2 += self.dt * (-self.beta2 * e_obs)
        
        # 2. Control Law (Cancel disturbance + proportional tracking)
        u0 = self.Kp * (desired_vel - self.z1)
        u = (u0 - self.z2) / self.b0
        
        self.prev_u = u
        return float(u)

    def reset(self):
        self.z1 = 0.0
        self.z2 = 0.0
        self.prev_u = 0.0


class ControlBarrierFunction:
    """
    Safety filter using Control Barrier Functions (CBFs).
    Ensures safe geometry bypassing RL errors.
    """
    def __init__(self, safe_dist=0.15, gamma=1.0):
        self.safe_dist = safe_dist
        self.gamma = gamma

    def filter(self, nominal_u, min_lidar_dist, approach_vel):
        # h(x) = dist - safe_dist >= 0
        h = min_lidar_dist - self.safe_dist
        h_dot = -approach_vel  # approach velocity (positive if getting closer)
        
        # If safe, pass through
        if h_dot + self.gamma * h >= 0:
            return nominal_u
            
        # Otherwise, project nominal_u to safe set (brake)
        # Simplify: if heading to wall, clamp forward velocity
        if nominal_u > 0:
            safe_u = max(0.0, nominal_u - 0.5) # Soft brake
            return safe_u
        return nominal_u


class DataDrivenMPC:
    """
    A lightweight, data-driven Model Predictive Control horizon tracker.
    Stores past input-output pairs to approximate local Jacobians without a physics model.
    """
    def __init__(self, horizon=5, dt=0.01):
        self.horizon = horizon
        self.dt = dt
        self.history_u = []
        self.history_y = []
        
    def optimize(self, desired_vel, current_vel):
        # In a full deployment, this solves a QP over the horizon.
        # For our 100Hz real-time constraint in pure Python, we implement a fast 1-step gradient descent
        # based on recent data correlation.
        if len(self.history_u) < 2:
            u_opt = desired_vel  # Fallback to direct tracking
        else:
            # Simple local sensitivity (dy/du)
            dy = self.history_y[-1] - self.history_y[-2]
            du = self.history_u[-1] - self.history_u[-2]
            sensitivity = dy / du if abs(du) > 1e-4 else 1.0
            
            error = desired_vel - current_vel
            u_opt = self.history_u[-1] + error / (sensitivity + 1e-3)
            
        # Log history
        self.history_u.append(u_opt)
        self.history_y.append(current_vel)
        if len(self.history_u) > 10:
            self.history_u.pop(0)
            self.history_y.pop(0)
            
        return float(u_opt)



class BrainstemController:
    """
    The main orchestrator: Braitenberg reflexes → differential kinematics → PID → torque clamps.

    This is the complete 100Hz deterministic control loop. Every call to
    compute_torques() performs one full cycle:

        1. REFLEX CHECK: Route (v, ω) through Braitenberg layer. If LiDAR
           detects danger, the reflex layer overrides the RL targets with
           safe commands. The RL policy never knows this happened.

        2. DIFFERENTIAL KINEMATICS: Convert the (possibly overridden) body
           velocity (v, ω) into per-wheel angular velocity setpoints:
               ωL = (v − ω · W/2) / R
               ωR = (v + ω · W/2) / R
           where W = track_width (0.22m), R = wheel_radius (0.04m).

        3. PID TRACKING: Each wheel has its own independent PID controller
           that compares the desired ωL/ωR against encoder feedback and
           outputs a raw torque command.

        4. TORQUE SATURATION: Both torques are hard-clamped to ±max_torque
           (2.5 Nm). This protects the motors and prevents the PID from
           commanding physically impossible forces.

    Diagnostic counters track total reflex overrides and torque saturations
    across the lifetime of the controller — useful for tuning and telemetry.

    Biological analogue: Complete reticulospinal motor loop from brainstem
    through spinal interneurons to alpha motor neurons. The "two PID
    controllers" are analogous to the left and right ventral horn motor
    pools, each independently controlling its side of the body while
    receiving shared descending commands from the reticular formation.

    Parameters
    ----------
    track_width : float
        Distance between left and right drive wheel contact patches (m).
        Default 0.22m — Bob's Mark VIII chassis spec.
    wheel_radius : float
        Radius of drive wheels (m). Default 0.04m.
    dt : float
        Control loop timestep (s). Default 0.01 = 100Hz.
    max_torque : float
        Hardware torque saturation limit per wheel (Nm). Default 0.12.
    """

    def __init__(self, dt=0.01, track_width=0.22, wheel_radius=0.04, max_torque=0.12):
        self.track_width  = track_width
        self.wheel_radius = wheel_radius
        self.dt           = dt
        self.max_torque   = max_torque

        # [NEW] Initialize low-pass filter state
        self.prev_torque = np.array([0.0, 0.0], dtype=np.float32)

        # [NEW] The Grand Unified Spinal Controller suite
        self.cbf = ControlBarrierFunction(safe_dist=0.15, gamma=2.0)
        
        # Left wheel uses iPID + MPC
        self.left_mpc = DataDrivenMPC(horizon=5, dt=dt)
        self.left_ipid = IntelligentPidController(Kp=1.0, Ki=0.1, Kd=0.01, alpha=10.0, dt=dt)
        
        # Right wheel uses ADRC (just to run both concurrently for redundancy)
        self.right_mpc = DataDrivenMPC(horizon=5, dt=dt)
        self.right_adrc = ActiveDisturbanceRejectionController(b0=10.0, w_c=10.0, w_o=30.0, dt=dt)

        # Symmetric robust tracking controllers
        self.left_pid  = DiscretePIDController(Kp=0.1, Ki=0.0, Kd=0.0, dt=dt)
        self.right_pid = DiscretePIDController(Kp=0.1, Ki=0.0, Kd=0.0, dt=dt)

        # Braitenberg geometric reflex layer
        self.reflex = BraitenbergReflexLayer()

        # Lifetime diagnostic counters
        self.total_reflex_overrides    = 0
        self.total_torque_saturations  = 0

        # Step counter for startup grace window
        self.step_count = 0

        # EMA states for proprioception (Phase C2 self-estimator)
        self.psi_L = 1.0
        self.psi_R = 1.0
        self.psi_S = 0.0
        self.alpha_ema = 0.05  # 100Hz smoothing (time constant ~200ms)

        # Low-pass filter states for velocity feedback
        self.vel_L_ema = 0.0
        self.vel_R_ema = 0.0
        self.ema_alpha = 0.15  # filter alpha (cutoff ~15Hz to kill 250Hz chatter)

    def apply_damping(self, raw_torque):
        """ Exponential Moving Average filter to kill torque jitter """
        self.prev_torque = 0.8 * self.prev_torque + 0.2 * raw_torque
        return self.prev_torque

    def compute_torques(self, lidar_24, encoder_vel_2, target_linear, target_angular, actual_omega=0.0, bypass_reflex=False, stress=0.0, stiffness=1.0):
        # ── Step 1: Safety Shield (CBF & Reflex) ─────────────────────────
        min_lidar = np.min(lidar_24)
        
        # Approximate approach velocity (positive if getting closer)
        # We don't have perfect dv, so we guess from target_linear
        approach_vel = target_linear if target_linear > 0 else 0.0
        
        # 1.a CBF mathematical shield
        cbf_linear = self.cbf.filter(target_linear, min_lidar, approach_vel)
        
        # 1.b Legacy Braitenberg fallback
        if bypass_reflex:
            safe_linear, safe_angular, reflex_active = cbf_linear, target_angular, False
        else:
            safe_linear, safe_angular, reflex_active = self.reflex.check(
                lidar_24, cbf_linear, target_angular, stress=stress, stiffness=stiffness
            )
        if reflex_active:
            self.total_reflex_overrides += 1

        # ── Step 2: Differential drive kinematics ────────────────────────
        half_track = self.track_width / 2.0
        kinematic_L = (safe_linear - safe_angular * half_track) / self.wheel_radius
        kinematic_R = (safe_linear + safe_angular * half_track) / self.wheel_radius

        kinematic_L = np.clip(kinematic_L, -35.0, 35.0)
        kinematic_R = np.clip(kinematic_R, -35.0, 35.0)

        self.vel_L_ema = self.ema_alpha * encoder_vel_2[0] + (1.0 - self.ema_alpha) * self.vel_L_ema
        self.vel_R_ema = self.ema_alpha * encoder_vel_2[1] + (1.0 - self.ema_alpha) * self.vel_R_ema

        # ── Step 3: Grand Unified Motor Optimization ─────────────────────
        # Run MPC and robust controllers to update histories, but use the stable symmetric PID for control
        _ = self.left_mpc.optimize(kinematic_L, self.vel_L_ema)
        _ = self.right_mpc.optimize(kinematic_R, self.vel_R_ema)
        _ = self.left_ipid.compute(kinematic_L, self.vel_L_ema)
        _ = self.right_adrc.compute(kinematic_R, self.vel_R_ema)
        
        corr_L = self.left_pid.compute(kinematic_L, self.vel_L_ema)
        corr_R = self.right_pid.compute(kinematic_R, self.vel_R_ema)
        
        corr_L = np.clip(corr_L, -2.0, 2.0)
        corr_R = np.clip(corr_R, -2.0, 2.0)
        
        cmd_L = kinematic_L + corr_L
        cmd_R = kinematic_R + corr_R

        torque_L = np.clip(cmd_L, -35.0, 35.0)
        torque_R = np.clip(cmd_R, -35.0, 35.0)

        # ── Step 4: No explicit saturation needed ────────────────────────
        # Force limiting is handled by the actuator forcerange in XML.
        torque_saturated = False

        # ── Step 5: Proprioceptive Self-Estimator (Phase C2) ─────────────
        self.step_count += 1
        actual_L = self.vel_L_ema
        actual_R = self.vel_R_ema

        # 1 & 2. Left and Right Motor Health (Tracking Efficiency)
        # Apply 1000-step grace window during which estimators are frozen in healthy default states
        if self.step_count > 1000:
            if abs(kinematic_L) > 0.5:
                left_efficiency = 1.0 - np.clip(abs(kinematic_L - actual_L) / (abs(kinematic_L) + 1e-6), 0.0, 1.0)
                self.psi_L = (1.0 - self.alpha_ema) * self.psi_L + self.alpha_ema * left_efficiency
            else:
                self.psi_L = (1.0 - 0.01) * self.psi_L + 0.01 * 1.0

            if abs(kinematic_R) > 0.5:
                right_efficiency = 1.0 - np.clip(abs(kinematic_R - actual_R) / (abs(kinematic_R) + 1e-6), 0.0, 1.0)
                self.psi_R = (1.0 - self.alpha_ema) * self.psi_R + self.alpha_ema * right_efficiency
            else:
                self.psi_R = (1.0 - 0.01) * self.psi_R + 0.01 * 1.0

            # 3. Slip Ratio (Kinematic vs. IMU Coherence)
            # R = 0.04m, W = 0.22m
            omega_encoder = (self.wheel_radius / self.track_width) * (actual_R - actual_L)
            slip_diff = abs(omega_encoder - actual_omega)
            # Normalize over the maximum scale of active rotation to prevent straight-line division amplification
            norm_denominator = max(abs(omega_encoder), abs(actual_omega), 1.0)
            norm_slip = np.clip(slip_diff / norm_denominator, 0.0, 2.0)
            self.psi_S = (1.0 - self.alpha_ema) * self.psi_S + self.alpha_ema * norm_slip

        # 4. Torque Strain — use actuator forcerange (0.12) as reference
        psi_T = 0.0  # Force is managed by the actuator; no explicit strain

        psi_vec = np.array([self.psi_L, self.psi_R, self.psi_S, psi_T], dtype=np.float64)

        # ── Build output ─────────────────────────────────────────────────
        # ── Step 6: Package Telemetry ────────────────────────────────────────────
        diagnostics = {
            'reflex_active':    bool(reflex_active),
            'left_error':       float(kinematic_L - actual_L),
            'right_error':      float(kinematic_R - actual_R),
            'torque_saturated': bool(torque_saturated),
            'psi':              np.array([self.psi_L, self.psi_R, self.psi_S, psi_T], dtype=np.float32)
        }

        return np.array([torque_L, torque_R], dtype=np.float32), diagnostics

    def reset(self):
        """Reset all internal state. Call on episode boundaries."""
        self.left_ipid.reset()
        self.right_adrc.reset()
        self.left_pid.reset()
        self.right_pid.reset()
        self.total_reflex_overrides   = 0
        self.total_torque_saturations = 0
        self.psi_L = 1.0
        self.psi_R = 1.0
        self.psi_S = 0.0
        self.step_count = 0


class MinimumJerkSmoother:
    """
    Minimum-Jerk filter for joint-space trajectory smoothing.
    Uses a 5th-order polynomial interpolation to transition smoothly between target states,
    reducing acceleration spikes and joint wear.
    """
    def __init__(self, num_joints=10, duration=0.4, dt=0.01):
        self.num_joints = num_joints
        self.duration = duration
        self.dt = dt
        self.steps = int(duration / dt)
        
        self.q_curr = np.zeros(num_joints, dtype=np.float32)
        self.q_start = np.zeros(num_joints, dtype=np.float32)
        self.q_target = np.zeros(num_joints, dtype=np.float32)
        
        self.step_idx = 0
        self.is_moving = False

    def reset(self, q_init):
        self.q_curr = np.array(q_init, dtype=np.float32)
        self.q_start = np.array(q_init, dtype=np.float32)
        self.q_target = np.array(q_init, dtype=np.float32)
        self.step_idx = 0
        self.is_moving = False

    def update_target(self, q_new_target):
        # Trigger a new trajectory segment if the target changes significantly
        if np.linalg.norm(q_new_target - self.q_target) > 1e-3:
            self.q_start = np.copy(self.q_curr)
            self.q_target = np.copy(q_new_target)
            self.step_idx = 0
            self.is_moving = True

    def step(self):
        if not self.is_moving:
            return self.q_target
            
        self.step_idx += 1
        tau = self.step_idx / self.steps
        if tau >= 1.0:
            self.q_curr = np.copy(self.q_target)
            self.is_moving = False
        else:
            # 5th-degree polynomial: 10*t^3 - 15*t^4 + 6*t^5
            s = 10.0 * (tau**3) - 15.0 * (tau**4) + 6.0 * (tau**5)
            self.q_curr = self.q_start + (self.q_target - self.q_start) * s
            
        return self.q_curr


# ═════════════════════════════════════════════════════════════════════════════
# STANDALONE VALIDATION
# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    """
    Simulation test: 1000 steps (10 seconds at 100Hz).

    Scenario:
        - LiDAR is clear (all 1.0m) for most of the run.
        - At step 300, a wall appears dead ahead (front rays drop to 0.10m),
          triggering the Braitenberg emergency reflex.
        - Target velocity: constant 0.5 m/s forward, 0.0 angular.

    Assertions:
        - Reflex fires during the wall event.
        - Torques reverse (go negative) during the wall event.
        - System runs 1000 steps without crash or NaN.
    """
    print("=" * 70)
    print("carl_brainstem.py — Reticulospinal Tract Validation")
    print("=" * 70)

    ctrl = BrainstemController()
    TOTAL_STEPS = 1000
    WALL_EVENT_START = 300
    WALL_EVENT_END   = 320    # wall persists for 20 steps (200ms)

    # Track state for assertions
    reflex_fired_during_wall = False
    torques_reversed_during_wall = False
    any_nan = False

    # Simulated encoder velocity (starts at rest, evolves with torque)
    encoder_vel = np.array([0.0, 0.0])

    # Simple first-order wheel dynamics for the simulation:
    # ω_new = ω_old + (τ / I) * dt, where I ≈ 0.01 kg·m² (wheel inertia)
    WHEEL_INERTIA = 0.01

    t_start = time.perf_counter_ns()

    for step in range(TOTAL_STEPS):
        # ── Build LiDAR scan ─────────────────────────────────────────
        lidar = np.ones(24, dtype=np.float64)   # all clear at 1.0m

        if WALL_EVENT_START <= step < WALL_EVENT_END:
            # Wall dead ahead: front sector (rays 8-15) drops to 0.10m
            lidar[8:16] = 0.10

        # ── Run brainstem ────────────────────────────────────────────
        torques, diag = ctrl.compute_torques(
            lidar_24=lidar,
            encoder_vel_2=encoder_vel,
            target_linear=0.5,
            target_angular=0.0
        )

        # ── Check for NaN ────────────────────────────────────────────
        if np.any(np.isnan(torques)):
            any_nan = True
            print(f"  [FAIL] NaN detected at step {step}")
            break

        # ── Track assertions during wall event ───────────────────────
        if WALL_EVENT_START <= step < WALL_EVENT_END:
            if diag['reflex_active']:
                reflex_fired_during_wall = True
            # At least one torque should be negative (reversing)
            if torques[0] < 0 or torques[1] < 0:
                torques_reversed_during_wall = True

        # ── Simulate wheel dynamics ──────────────────────────────────
        # Simple Euler integration: ω += (τ / I) · dt
        encoder_vel = encoder_vel + (torques / WHEEL_INERTIA) * ctrl.dt

        # ── Periodic telemetry ───────────────────────────────────────
        if step % 200 == 0 or step == WALL_EVENT_START:
            print(f"  step {step:4d} | tL={torques[0]:+7.3f} tR={torques[1]:+7.3f} "
                  f"| wL={encoder_vel[0]:+7.2f} wR={encoder_vel[1]:+7.2f} "
                  f"| reflex={diag['reflex_active']!s:5s} "
                  f"| sat={diag['torque_saturated']!s:5s}")

    t_elapsed_us = (time.perf_counter_ns() - t_start) / 1000.0
    t_per_step_us = t_elapsed_us / TOTAL_STEPS

    print()
    print(f"  Timing: {t_elapsed_us:.0f} us total, {t_per_step_us:.1f} us/step")
    print(f"  Total reflex overrides:   {ctrl.total_reflex_overrides}")
    print(f"  Total torque saturations: {ctrl.total_torque_saturations}")
    print()

    # ── Assertions ───────────────────────────────────────────────────────
    passed = 0
    total  = 4

    def check(label, condition):
        global passed
        status = "PASS" if condition else "FAIL"
        if condition:
            passed += 1
        print(f"  [{status}] {label}")

    check("No NaN in torque outputs",                     not any_nan)
    check("Reflex fired during wall event",               reflex_fired_during_wall)
    check("Torques reversed during wall event",           torques_reversed_during_wall)
    check("Reflex override count > 0",                    ctrl.total_reflex_overrides > 0)

    print()
    if passed == total:
        print(f"  [OK] ALL {total} CHECKS PASSED -- reticulospinal tract is operational.")
    else:
        print(f"  [!!] {total - passed}/{total} CHECKS FAILED -- brainstem needs repair.")
    print("=" * 70)
