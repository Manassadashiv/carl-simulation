"""
carl_cpg.py — Coupled Hopf Central Pattern Generator (CPG) Motor Oscillator.

Architecture (Mark VIII — Sub-cortical Motor pattern Modulator):
- HopfCpgEngine: Simulates coupled non-linear differential equations representing
  biological oscillators. Integrates states via an explicit 4th-Order Runge-Kutta (RK4)
  solver on a 100Hz timeline to ensure limit-cycle stability.

  This layer sits between the PID controllers and the physical MuJoCo wheels to
  introduce biomimetic physical expressions directly to Bob's locomotion vectors:

    1. Cortisol Shudder (Panic Tremor):
       When cortisol spikes (>0.4), the limit cycle frequency shifts into high gear
       (~12 Hz) with small amplitudes (~0.45 Nm max offset). This creates a physical,
       visceral trembling effect in Bob's chassis under stress.

    2. Dopamine Purr (Idle Sway):
       When Bob is stationary and dopamine is active (>0.3), the limit cycle frequency
       shifts to a low, soothing rhythm (~1.5 Hz) in anti-phase between left and right
       channels. This forces Bob to perform a micro-breathing yaw rotation back and forth
       in place, mimicking organic rest cycles.

    3. Normal Driving:
       When moving or in baseline emotional states, the limit cycle radius collapses to 0,
       ensuring the modulation decays smoothly to zero with no torque distortion.

Mathematical model (Canonical Hopf Bifurcation):
  dx/dt = alpha * (mu - (x^2 + y^2)) * x - omega * y
  dy/dt = alpha * (mu - (x^2 + y^2)) * y + omega * x
  where r = sqrt(mu) is the target radius (amplitude), and omega is the frequency (2 * pi * f).
"""

import numpy as np

class HopfCpgEngine:
    def __init__(self, dt=0.01, alpha=5.0):
        """
        Biomimetic Central Pattern Generator using Coupled Hopf Oscillators.
        Integrates states via an explicit RK4 solver on a 100Hz timeline.
        
        Parameters
        ----------
        dt : float
            The 100Hz execution time delta (0.01 seconds)
        alpha : float
            Convergence acceleration factor toward the limit cycle
        """
        self.dt = dt
        self.alpha = alpha
        
        # State vectors for Left (L) and Right (R) motor channels
        # State format: [x, y] where x maps to the raw mechanical torque modulation
        self.state_L = np.array([0.1, 0.0])
        self.state_R = np.array([0.1, 0.0])
        
        # Target limits to insulate motors from structural shearing forces
        self.MAX_TREMOR_OFFSET = 0.03  # Max torque deviation in panic mode (Nm)
        self.MAX_SWAY_OFFSET = 0.01    # Max torque deviation during idle breathing (Nm)

    def _hopf_dynamics(self, state, mu, omega):
        """
        Evaluates the instantaneous derivatives of a canonical Hopf Oscillator.
        dx/dt = alpha * (mu - (x^2 + y^2)) * x - omega * y
        dy/dt = alpha * (mu - (x^2 + y^2)) * y + omega * x
        
        When mu is 0 (dampening mode), we use a linear coefficient of -1.0
        to ensure fast, exponential decay toward the origin instead of slow cubic decay.
        """
        x, y = state[0], state[1]
        r_squared = x**2 + y**2
        
        linear_coeff = mu if mu > 0 else -1.0
        
        dxdt = self.alpha * (linear_coeff - r_squared) * x - omega * y
        dydt = self.alpha * (linear_coeff - r_squared) * y + omega * x
        
        return np.array([dxdt, dydt])

    def _rk4_step(self, state, mu, omega):
        """
        4th-Order Runge-Kutta fixed-step integrator to guarantee limit-cycle stability.
        """
        h = self.dt
        k1 = self._hopf_dynamics(state, mu, omega)
        k2 = self._hopf_dynamics(state + 0.5 * h * k1, mu, omega)
        k3 = self._hopf_dynamics(state + 0.5 * h * k2, mu, omega)
        k4 = self._hopf_dynamics(state + h * k3, mu, omega)
        
        return state + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def compute_biomimetic_modulation(self, current_drives, target_torques):
        """
        Executes the 100Hz CPG step. Distorts base target torques using 
        the active emotional state vector.
        
        Parameters
        ----------
        current_drives : dict
            Dict containing scalar values for 'cortisol' and 'dopamine'.
        target_torques : np.ndarray, shape (2,)
            Target wheel torques [torque_left, torque_right] from PID core.
            
        Returns
        -------
        np.ndarray, shape (2,)
            Modulated wheel torques [torque_left, torque_right] ready for actuators.
        """
        if hasattr(current_drives, 'get'):
            cortisol = current_drives.get("cortisol", current_drives.get("cort", 0.0))
            dopamine = current_drives.get("dopamine", current_drives.get("da", 0.0))
        else:
            cortisol = getattr(current_drives, "cort", 0.0)
            dopamine = getattr(current_drives, "da", 0.0)
        
        # Check if the robot is effectively stationary or running a low-speed profile
        is_stationary = np.all(np.abs(target_torques) < 0.05)
        
        # --- PARAMETER MODULATION GRID ---
        freq_mult = getattr(current_drives, "cpg_frequency_multiplier", 1.0)
        amp_mult = getattr(current_drives, "cpg_amplitude_multiplier", 1.0)

        if cortisol > 0.4:
            # Panic Trajectory: Shift into high-frequency, somatic trembling mode
            # Omega = 2 * pi * Frequency (12 Hz tremor)
            omega_L = 2.0 * np.pi * 12.0 * freq_mult
            omega_R = 2.0 * np.pi * 12.5 * freq_mult # Slight detuning to prevent unnatural rigid symmetry
            
            # Radius of limit cycle scales linearly with stress intensity
            mu_L = (cortisol * self.MAX_TREMOR_OFFSET * amp_mult) ** 2
            mu_R = (cortisol * self.MAX_TREMOR_OFFSET * amp_mult) ** 2
            
        elif dopamine > 0.3:
            # [NEW] Remove 'is_stationary' check. 
            # Sway is now allowed when moving, scaled by dopamine (da)
            # Omega = 2 * pi * Frequency (1.5 Hz breathing cycle)
            omega_L = 2.0 * np.pi * 1.5 * freq_mult
            omega_R = 2.0 * np.pi * 1.5 * freq_mult
            
            mu_L = (dopamine * self.MAX_SWAY_OFFSET * amp_mult) ** 2
            mu_R = (dopamine * self.MAX_SWAY_OFFSET * amp_mult) ** 2
            
        else:
            # Baseline Dynamic State: Collapse the limit cycle radius to zero
            # The oscillator smoothly dampens toward the origin, adding zero modulation.
            omega_L = 2.0 * np.pi * 2.0 * freq_mult
            omega_R = 2.0 * np.pi * 2.0 * freq_mult
            mu_L = 0.0
            mu_R = 0.0

        # --- PREVENT FIXED-POINT TRAPPING & ENSURE INSTANT CONVERGENCE ---
        # When transitioning from damped mode to active oscillation, the growth rate
        # alpha * mu is small, resulting in slow exponential growth. We seed the states
        # to the target radius to bypass the startup delay and prevent fixed-point trapping.
        if mu_L > 0.0:
            target_r_L = np.sqrt(mu_L)
            current_r_L = np.sqrt(np.sum(self.state_L**2))
            if current_r_L < 0.5 * target_r_L:
                self.state_L = np.array([target_r_L, 0.0])
                
        if mu_R > 0.0:
            target_r_R = np.sqrt(mu_R)
            current_r_R = np.sqrt(np.sum(self.state_R**2))
            if current_r_R < 0.5 * target_r_R:
                self.state_R = np.array([target_r_R, 0.0])

        # --- EXECUTE STABLE INTEGRATION ---
        self.state_L = self._rk4_step(self.state_L, mu_L, omega_L)
        self.state_R = self._rk4_step(self.state_R, mu_R, omega_R)
        
        # [NEW] Invert the right wheel purr coupling to execute an anti-phase rotational sway
        # Condition updated to check dopamine > 0.3 instead of is_stationary
        modulation_L = self.state_L[0]
        modulation_R = -self.state_R[0] if (dopamine > 0.3 and cortisol <= 0.4) else self.state_R[0]
        
        # Inject the synthesized wave profiles straight into the motor torques
        modulated_torques = np.zeros(2)
        modulated_torques[0] = target_torques[0] + modulation_L
        modulated_torques[1] = target_torques[1] + modulation_R
        
        return modulated_torques


# ==============================================================================
#  STANDALONE VALIDATION TEST SUITE
# ==============================================================================
if __name__ == "__main__":
    print("=" * 72)
    print("  carl_cpg.py -- CPG Motor Oscillator Limit-Cycle Validation")
    print("=" * 72)

    cpg = HopfCpgEngine()
    total_steps = 1000  # 10 seconds at 100Hz

    has_nans = False
    max_modulation = 0.0

    # Phase tracking
    dampening_ok = False
    purr_ok = False
    tremor_ok = False

    # Simulate 1000 steps with dynamic drive shifts
    for step in range(total_steps):
        # Base torques: assume stationary for the first 700 steps (purring/dampening),
        # then moving for the last 300 steps (tremor)
        if step < 700:
            torques = np.array([0.0, 0.0])
        else:
            torques = np.array([1.0, 1.0])

        # Drive changes
        if step < 300:
            # Phase 1: Baseline (neutral drives) -> should collapse to origin (0)
            drives = {"cortisol": 0.0, "dopamine": 0.0}
        elif step < 600:
            # Phase 2: Dopamine purring (stationary) -> 1.5Hz anti-phase sway
            drives = {"cortisol": 0.0, "dopamine": 0.8}
        else:
            # Phase 3: Cortisol trembling (panic while moving) -> 12Hz tremor
            drives = {"cortisol": 0.9, "dopamine": 0.0}

        modulated = cpg.compute_biomimetic_modulation(drives, torques)
        diff = modulated - torques
        max_mod = np.max(np.abs(diff))
        if max_mod > max_modulation:
            max_modulation = max_mod

        if np.any(np.isnan(modulated)):
            has_nans = True
            break

        # Verification asserts inside milestones
        if step == 200:
            # Check that oscillation has collapsed close to zero after 2 seconds
            if np.all(np.abs(diff) < 1e-3):
                dampening_ok = True

        if step == 500:
            # Check that purring is active, bounded, and anti-phase (left = -right)
            # Note: diff[0] should be very close to -diff[1]
            if np.abs(diff[0]) > 0.005 and np.abs(diff[0] + diff[1]) < 1e-4:
                purr_ok = True

        if step == 900:
            # Check that tremor is active and bounded under MAX_TREMOR_OFFSET
            if np.abs(diff[0]) > 0.015 and np.max(np.abs(diff)) <= cpg.MAX_TREMOR_OFFSET + 1e-3:
                tremor_ok = True

        if step % 150 == 0:
            print(f"  Step {step:4d} | Drives: da={drives['dopamine']:.1f} cort={drives['cortisol']:.1f} "
                  f"| Torques: {torques[0]:.1f}/{torques[1]:.1f} -> Modulated: {modulated[0]:+6.3f}/{modulated[1]:+6.3f} "
                  f"| Modulation L/R: {diff[0]:+6.3f}/{diff[1]:+6.3f}")

    print("-" * 72)
    print(f"  Maximum observed modulation: {max_modulation:.4f} Nm")
    print(f"  Divergence check (No NaNs):  {'PASS' if not has_nans else 'FAIL'}")
    print(f"  Phase 1 (Origin dampening):  {'PASS' if dampening_ok else 'FAIL'}")
    print(f"  Phase 2 (Anti-phase purr):   {'PASS' if purr_ok else 'FAIL'}")
    print(f"  Phase 3 (Tremor modulation): {'PASS' if tremor_ok else 'FAIL'}")
    print("-" * 72)

    passed = (not has_nans) and dampening_ok and purr_ok and tremor_ok
    if passed:
        print("  [PASS] Hopf CPG limit-cycle characteristics verified.")
    else:
        print("  [FAIL] Limit cycle failed to meet stability or modulation requirements.")
        import sys
        sys.exit(1)
    print("=" * 72)
