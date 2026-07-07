"""
carl_industrial_motion.py — Industrial-grade Cartesian motion executor for CARL's arms.
"""

import numpy as np
import mujoco

class MinJerkSegment:
    """Classic minimum-jerk quintic (Flash & Hogan): zero velocity AND zero
    acceleration at both endpoints. Eliminates jerk discontinuities."""

    def __init__(self, p0, p1, duration):
        self.p0 = np.asarray(p0, dtype=float)
        self.p1 = np.asarray(p1, dtype=float)
        self.T = max(duration, 1e-3)

    def sample(self, t):
        """Return (pos, vel) at time t in [0, T]."""
        s = float(np.clip(t / self.T, 0.0, 1.0))
        blend = 10 * s**3 - 15 * s**4 + 6 * s**5
        dblend_ds = 30 * s**2 - 60 * s**3 + 30 * s**4
        pos = self.p0 + (self.p1 - self.p0) * blend
        vel = (self.p1 - self.p0) * dblend_ds / self.T
        return pos, vel

    def done(self, t):
        return t >= self.T


class ForceModulatedGrip:
    """Closes to a target *contact force*, not a fixed angle. Open-loop
    search speed until contact is detected, then PI regulation on the
    touch sensor reading."""

    def __init__(self, grip_low, grip_high, search_speed=0.35, kp=0.4, ki=0.6):
        self.grip_low = grip_low
        self.grip_high = grip_high
        self.search_speed = search_speed
        self.kp, self.ki = kp, ki
        self.target_force = 0.6
        self.integral = 0.0
        self.cmd = grip_low

    def reset(self):
        self.integral = 0.0
        self.cmd = self.grip_low

    def step(self, touch_reading, dt, closing=True):
        if not closing:
            self.cmd = self.grip_low
            self.integral = 0.0
            return self.cmd

        if touch_reading < 0.01:
            # No contact yet — close steadily
            self.cmd += self.search_speed * dt
        else:
            # Contact made — regulate to target force
            error = self.target_force - touch_reading
            self.integral += error * dt
            self.cmd += (self.kp * error + self.ki * self.integral) * dt

        self.cmd = float(np.clip(self.cmd, self.grip_low, self.grip_high))
        return self.cmd


class CartesianArmExecutor:
    """
    High-precision Cartesian executor for one CARL arm (4-DOF IK chain +
    1 grip joint). Queue waypoints; call step() once per control tick.
    """

    def __init__(self, model, data, cache, side="L", max_joint_vel=8.0,
                 pos_gain=15.0, damping=0.015):
        self.model, self.data, self.cache = model, data, cache
        self.side = side
        self.tip_sid = cache[f'tip_{side}_sid']
        sl = slice(0, 4) if side == "L" else slice(5, 9)
        self.dofs = cache['arm_dof'][sl]
        self.qpos_adr = cache['arm_qpos'][sl]
        
        self.max_joint_vel = max_joint_vel
        self.pos_gain = pos_gain
        self.damping = damping
        self.segment = None
        self.t_in_segment = 0.0
        self.touch_key = 'touch_L' if side == "L" else 'touch_R'
        self.grip = ForceModulatedGrip(0.0, 0.52)
        
        # Preferred home configuration for null-space projection (prevents singularities/collisions)
        self.q_home = np.array([0.0, 0.3, -1.2, 0.0]) if side == "L" else np.array([0.0, 0.3, -1.2, 0.0])
        self.q_cmd = np.array([self.data.qpos[a] for a in self.qpos_adr])

    def queue_waypoint(self, target_pos, duration):
        """Start a new minimum-jerk segment from the current tip pose to
        target_pos over `duration` seconds."""
        tip_pos = self.data.site_xpos[self.tip_sid].copy()
        self.segment = MinJerkSegment(tip_pos, np.asarray(target_pos, dtype=float), duration)
        self.t_in_segment = 0.0
        self.q_cmd = np.array([self.data.qpos[a] for a in self.qpos_adr])

    def _solve_ik(self, target_pos, target_vel, dt, grasp=False, drives_dict=None):
        # 1. Modulate control parameters neurochemically if drives are provided
        da = drives_dict.get('da', 0.0) if drives_dict else 0.0
        cort = drives_dict.get('cort', 0.0) if drives_dict else 0.0
        
        max_vel = self.max_joint_vel * (1.0 + 0.5 * da)
        p_gain = self.pos_gain * (1.0 + 0.4 * cort)
        
        # 2. Get Site Jacobian
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.tip_sid)
        J = jacp[:, self.dofs]

        # 3. Admittance Compliance (Virtual Spring along vertical axis)
        touch_val = float(self.data.sensordata[self.cache[self.touch_key]])
        modified_target_pos = target_pos.copy()
        if not grasp and touch_val > 0.08:
            # Yield upward to avoid crushing things or colliding aggressively
            compliance_z = 0.05 * (touch_val - 0.08)
            modified_target_pos[2] += compliance_z

        tip_pos = self.data.site_xpos[self.tip_sid]
        pos_err = modified_target_pos - tip_pos

        # Feedforward velocity + proportional feedback
        cart_vel_cmd = target_vel + p_gain * pos_err

        # 4. SVD-based singularity robust Damped Least Squares
        U, s, Vt = np.linalg.svd(J)
        s_min = s[-1] if len(s) > 0 else 0.0
        s_min_thresh = 0.02
        
        # Levenberg-Marquardt style adaptive damping based on minimum singular value
        if s_min < s_min_thresh:
            adaptive_damping = self.damping + 0.10 * (1.0 - s_min / s_min_thresh)**2
        else:
            adaptive_damping = self.damping
            
        s_inv = s / (s**2 + adaptive_damping)
        V = Vt.T
        Sigma_inv = np.zeros((4, 3))
        for i in range(3):
            Sigma_inv[i, i] = s_inv[i]
        J_dls = V @ Sigma_inv @ U.T

        # 5. Null-Space Attraction to preferred home pose (collision & joint limit avoidance)
        current_q = np.array([self.data.qpos[a] for a in self.qpos_adr])
        q_err = self.q_home - current_q
        k_null = 0.1
        dq_null = k_null * q_err
        
        # Projection operator: I - J_dls @ J
        I = np.eye(4)
        P_null = I - J_dls @ J
        
        # 6. Combined Command
        dq = J_dls @ cart_vel_cmd + P_null @ dq_null
        return np.clip(dq, -max_vel, max_vel)

    def step(self, dt, grasp=False, hold_force=0.6, drives_dict=None):
        """Returns the 5-element joint command (yaw, shoulder, elbow, wrist, grip)
        for this arm. Call once per control tick."""
        if self.segment is not None:
            target_pos, target_vel = self.segment.sample(self.t_in_segment)
            dq = self._solve_ik(target_pos, target_vel, dt, grasp=grasp, drives_dict=drives_dict)
            self.q_cmd = self.q_cmd + dq * dt
            self.t_in_segment += dt

        self.grip.target_force = hold_force
        touch_val = float(self.data.sensordata[self.cache[self.touch_key]])
        grip_cmd = self.grip.step(touch_val, dt, closing=grasp)

        return np.concatenate([self.q_cmd, [grip_cmd]])

    def is_settled(self, pos_tol=0.004):
        if self.segment is None:
            return True
        target_pos, _ = self.segment.sample(self.segment.T)
        tip_pos = self.data.site_xpos[self.tip_sid]
        close_enough = float(np.linalg.norm(target_pos - tip_pos)) < pos_tol
        return self.segment.done(self.t_in_segment) and close_enough
