# CARL Final Phase: The Optimal Daughter Minds (MPC Upgrade)

The data has revealed a massive scientific insight: **Even when the Mechanic learns the system perfectly and prediction error drops to zero, the robot still fails due to control limitations in the emergency system.**

To solve this, we must upgrade the Daughter Minds from a simple "emergency reaction" into a true **Optimal Model Predictive Controller (MPC)**.

## User Review Required

> [!IMPORTANT]  
> Please review the Optimal MPC formulation below. This integrates the expanded action space, the 60-step horizon, and the quadratic cost function $J$. If you approve, I will code the final `phase_final_env.py`.

---

## 1. The Optimal Daughter Minds Formulation

When the Guardian triggers ($P(\text{fail}) > 0.70$), the system enters **Recovery Mode**.

### The Action Space
Instead of just max torque, the Daughter Minds will evaluate a spectrum of actions:
$U_{candidates} = [-8.0, -5.0, -2.0, 0.0, 2.0, 5.0, 8.0]$

### The Simulation Horizon
We extend the simulation from 15 steps to **50 steps** ($\approx 200$ ms into the future). This allows the system to see the long-term consequences of an action, avoiding local minima.

### The Cost Function ($J$)
Instead of simply choosing the action with the "lowest max tilt", the Daughter Minds will compute a true Optimal Control cost over the horizon:
$J = \sum_{k=1}^{H} \Big( \theta_k^2 + 0.5 \dot{\theta}_k^2 + 0.01 u^2 \Big)$
- Heavy penalty if the simulation predicts a fall ($|\theta| > 0.5$): $J += 1000$.
- The action that minimizes $J$ is executed.

---

## 2. Recovery Mode & Hysteresis

To prevent the MPC from rapidly toggling on and off (which caused the Guardian to trigger multiple times but fail to save it in Phase 4), we introduce stateful Recovery Mode.

- **Trigger ON**: If $P(\text{fail}) > 0.70$, `mpc_active = True`.
- **Trigger OFF**: `mpc_active` remains True until the robot is restabilized and $P(\text{fail}) < 0.30$. 
- While `mpc_active` is True, the LQR is completely overridden by the Optimal MPC.

---

## 3. Increased Control Authority

We will increase the maximum allowable torque from `5.0` to `8.0`. The robot now has the physical strength required to execute the MPC's extreme recovery maneuvers.

---

> *"Learning reduces prediction error to near zero, but stability still depends on decision-making under constraints."*
