# CARL: Cognitive Architecture for Robotic Learning
**A Unified Framework for Self-Aware Adaptive Control in Unstable Robotic Systems**

> *"A machine that predicts its own mistakes and simulates futures before acting."*

---

## 1. Project Positioning & Core Scientific Claim

This project elevates a classic control theory benchmark (the inverted pendulum) into a self-preserving, adaptive cognitive system. It demonstrates not just how a robot balances, but how it *thinks* about balancing under extreme uncertainty.

**The Core Scientific Claim:**
> "A unified architecture using prediction error to drive adaptation, exploration, and failure prediction improves stability and learning speed in unstable robotic systems."

---

## 2. The Plant: Hyper-Unstable Flexible Inverted Pendulum

To prove the superiority of the architecture, the physical system must be highly unforgiving.
- **Base Structure**: Two independently driven wheels on a small chassis.
- **The Neck**: A tall (30-60cm), slightly flexible rod. This introduces elasticity, oscillations, and delayed response into the dynamics.
- **The Mass**: A variable center of mass at the top (representing the battery/controller).
- **The Environment**: Variable surfaces (tile, carpet, ice, slopes) and external disturbance inputs (manual pushes).

*Constant Instability = Constant Intelligence Test.*

---

## 3. The Cognitive Architecture (The 5 Modules)

The core loop integrates optimal control with real-time cognitive awareness.

### I. The Professor (Prediction & Baseline Control)
- **Role**: Uses an Infinite Horizon Linear Quadratic Regulator (LQR) based on a nominal linear dynamic model (assuming a rigid body on a perfect surface).
- **Function**: Outputs the baseline torque to balance the robot. Predicts the next state.
- **Output**: The crucial **Prediction Error ($e_t$)** between anticipated tilt and actual tilt.

### II. The Mechanic (Adaptive Control)
- **Role**: The Online System Identifier.
- **Function**: Uses Recursive Least Squares (RLS) to learn the true friction, inertia, and neck elasticity. Quietly updates the Professor's internal $A$ and $B$ matrices to minimize $e_t$.

### III. The Mapper (Uncertainty Mapping)
- **Role**: The Confidence Map.
- **Function**: Stores a 2D grid of "Trust Scores" ($M_{x,y}$). High error drops the trust; Mechanic adaptation restores the trust. It memorizes unstable zones.

### IV. The Guardian (Failure Prediction)
- **Role**: The Lifeguard.
- **Function**: Calculates $P(\text{Failure})$ continuously using current tilt, angular velocity, local Trust Score, and current Error rate. If $P(\text{Failure}) > 70\%$, it seizes control.

### V. The Explorer (Curiosity Engine)
- **Role**: The Strategic Wanderer.
- **Function**: Modifies the target position vector, slightly pulling the robot toward low-trust zones (carpet/ice) during safe conditions to force the Mechanic to learn the new dynamics.

---

## 4. The Signature Contribution: Daughter Minds (MPC)

When the Guardian predicts an imminent fall, the standard LQR fails. The system spawns **Daughter Minds**.
- **Action**: The main loop pauses for a fraction of a second (Bullet Time).
- **Simulation**: 5 distinct control strategies (Reverse torque, Accelerate forward, Damping oscillation, Drop mass, Do nothing) are rolled out into the future using the Mechanic's current estimated model.
- **Decision**: The strategy yielding the lowest maximum tilt is selected and executed, preventing the fall.

---

## 5. Experimental Design & Scientific Baselines

To prove the core claim, CARL must be evaluated against standard systems.

### The 3 Baselines
1. **Basic Controller (LQR Only)**: Works perfectly on tile. Fails catastrophically on carpet or when pushed.
2. **Adaptive Only (LQR + Mechanic)**: Adapts to carpet, but too slowly to survive sudden extreme changes (e.g., hitting ice at speed).
3. **Full CARL (The Unified Framework)**: Adapts to carpet, but also uses Guardian and Daughter Minds to survive sudden ice patches and severe pushes.

### Key Performance Metrics
- **Stability Time**: Total time upright before catastrophic failure ($|\theta| > 25^\circ$).
- **Failure Rate**: Number of falls over a set of randomized runs.
- **Recovery Success**: Percentage of times $P(\text{Failure}) > 70\%$ resulted in a successful save via Daughter Minds.
- **Adaptation Speed**: Time required for the Mechanic to reduce the Professor's error below a threshold after a surface change.
- **Prediction Accuracy**: Overall reduction in $e_t$.

---

## 6. The Killer Demo: "The Bullet Time Decision Test"

This is the visual and scientific centerpiece of the project.
1. CARL drives on tile. The ground suddenly turns to ice.
2. The Professor's error spikes. The Guardian predicts an 85% chance of falling within 1 second.
3. **The Spectacle**: PyBullet physics slows to 10% speed. The Web Dashboard flashes Crimson.
4. **The Rollout**: The Daughter Minds visually branch out on the UI. Four branches turn red (fall). One branch turns green (accelerate into the fall to catch balance).
5. **The Save**: Time snaps back. CARL executes the green branch perfectly.
*This proves that without Daughter Minds, the robot falls. With them, it recovers.*

---

## 7. Phased Engineering Build Order

We will build in rigorous layers to manage the extreme complexity.

- **Phase 1: The Baseline System**: Implement PyBullet inverted pendulum (with flexible neck). Tune basic LQR controller for a perfect floor.
- **Phase 2: The Professor & The Error**: Implement state prediction and error calculation. Introduce surface variations (carpet) to observe error spikes.
- **Phase 3: The Mechanic**: Implement RLS parameter tuning. Observe the LQR matrices adapting online to the carpet.
- **Phase 4: The Guardian & Mapper**: Implement the spatial trust map and the probabilistic failure boundary calculations.
- **Phase 5: The Daughter Minds**: Implement the fast-forward MPC rollouts. Hook them up to the Guardian triggers.
- **Phase 6: The Mind Dashboard**: Build the WebSocket bridge and the premium web UI to visualize the cognitive loop (Error, Trust, Probability, and the Bullet Time tree).

---

> *"This is no longer trying to beat MIT. This is building something that deserves to stand in that conversation."*
