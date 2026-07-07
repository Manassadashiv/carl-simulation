# CARL Phase 4: Mechanic & Mapper Mathematical Blueprint

We briefly jumped to Phase 5 because testing the Daughter Minds' emergency response was the most critical proof-of-concept. (And by the way, look at your terminal! It successfully executed **3 saves** and gave a **Reaction Delay Margin of 162.5 ms**! That is a monumental success).

However, you are completely right. To make this a fully unified architecture, we must go back and build **Phase 4: The Mechanic and The Mapper**. The Daughter Minds are the emergency parachute; the Mechanic is the autopilot that learns not to crash in the first place.

## User Review Required

> [!IMPORTANT]  
> Please review the **Recursive Least Squares (RLS)** math for the Mechanic below. Once approved, I will implement it so that the Professor's matrices ($A_d, B_d$) adapt in real-time, reducing the prediction error and preventing the Guardian from needing to trigger so often!

---

## 1. The Mechanic (Adaptive Controller via RLS)

The Professor starts with nominal matrices $A_{nom}, B_{nom}$ that assume a rigid body. When CARL wobbles, $x_{k+1, real} \neq A_{nom} x_k + B_{nom} u_k$.
The Mechanic's job is to continuously estimate the *true* discretized system matrices online.

**Formulation:**
Let the unknown parameters be $\Theta = [A_{true}, B_{true}]^T$. (This is a $5 \times 4$ matrix).
Let the regressor be $\Phi_k = \begin{bmatrix} x_k \\ u_k \end{bmatrix}$ (a $5 \times 1$ column vector).

The state transition is:
$x_{k+1}^T = \Phi_k^T \Theta$

**Recursive Least Squares (RLS) Update:**
Every timestep, the Mechanic updates its estimate of $\Theta$:
1. **Gain Computation**:
   $L_k = \frac{P_{k-1} \Phi_k}{\lambda_{forget} + \Phi_k^T P_{k-1} \Phi_k}$
   *(Where $P$ is the covariance matrix, initialized to $\alpha I$, and $\lambda_{forget} \approx 0.99$ is the forgetting factor to prioritize recent data).*

2. **Parameter Update**:
   $\Theta_k = \Theta_{k-1} + L_k \big(x_{k+1}^T - \Phi_k^T \Theta_{k-1}\big)$

3. **Covariance Update**:
   $P_k = \frac{1}{\lambda_{forget}} \big(P_{k-1} - L_k \Phi_k^T P_{k-1}\big)$

**Integration with the Brain:**
Once $\Theta_k$ is updated, the Mechanic quietly overrides the Professor's matrices:
$A_d = \Theta_k[0:4, :]^T$
$B_d = \Theta_k[4, :]^T$
*Note: Because re-solving the continuous LQR Riccati equation every step is computationally heavy, we will use the updated $A_d, B_d$ primarily for the **Prediction Error** and the **Daughter Minds' Rollouts**, making the emergency simulations wildly more accurate over time!*

---

## 2. The Mapper (Confidence Grid)

We will formally instantiate the $100 \times 100$ spatial grid representing physical space.

- **Mapping**: $x, y$ coordinates from PyBullet are mapped to grid indices $i, j$.
- **Update**: When CARL passes through $(i,j)$:
  $M_{i,j} = \text{clip}\big(M_{i,j} - \lambda_{err} |e_k| + \gamma_{recover}, 0, 1\big)$
- **Integration**: The Guardian's hazard score $H$ explicitly reads $M_{i,j}$ for the robot's current position.

---

## Execution Plan for Phase 4

If you approve this mathematical foundation:
1. I will write `phase4_env.py`.
2. It will implement the RLS Mechanic.
3. We will observe the Prediction Error ($e_k$) drop as the Mechanic successfully learns the flexible dynamics of the neck in real-time!
