# CARL-Ω: The Complete Embodied Biological Brain 🧠🤖
### *The Most Intelligent Cognitive Architecture Built on a Virtual Primate Scout*

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![Physics Engine](https://img.shields.io/badge/physics-MuJoCo-orange.svg)](https://mujoco.org/)
[![Deep Learning](https://img.shields.io/badge/framework-PyTorch-red.svg)](https://pytorch.org/)
[![Memory Model](https://img.shields.io/badge/memory-40K--D%20Holographic-green.svg)](https://github.com/Manassadashiv/carl-simulation)

---

## 🌟 Overview
**CARL-Ω (Carl-Omega)** is a biologically-inspired, multi-tiered cognitive architecture implemented on a **28-DOF physical Primate Scout robot** simulated in **MuJoCo**. 

Unlike standard AI models which are disembodied (such as LLMs), Carl operates in a closed-loop environment where he must manage physical variables (gravity, friction, collisions), negotiate homeostatic and allostatic drives (hunger, fatigue), map his surroundings, and control his 16-DOF arms to perform complex tasks like reach-grasp-and-carry.

---

## 🧠 The 8-Layer Cognitive Stack
Carl's brain is structured as an integrated biological hierarchy, ranging from raw physics up to conceptual abstraction:

```
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 7: CONCEPT GENESIS (Phase 16)                            │
│  Abstractions computed from raw surprise recurrence patterns    │
├─────────────────────────────────────────────────────────────────┤
│  LAYER 6: THE IMAGINATION (Phase 15)                            │
│  Hallucination Engine: Dreamer world model offline simulation   │
├─────────────────────────────────────────────────────────────────┤
│  LAYER 5: CAUSAL REASONING (Phase 14C)                          │
│  Counterfactual Question Generator + Causal Adjacency Scaffold │
├─────────────────────────────────────────────────────────────────┤
│  LAYER 4: THE WITNESS (Phase 14B)                               │
│  Metacognitive memory: monitors own failures and penalizes errors │
├─────────────────────────────────────────────────────────────────┤
│  LAYER 3: UNCERTAINTY MAP (Phase 14A)                           │
│  2D Spatial Navigator (Danger × Uncertainty matrix)              │
├─────────────────────────────────────────────────────────────────┤
│  LAYER 2: THE BIOLOGICAL BRAIN (Phase 13)                       │
│  Neuromodulators (DA/NE/5-HT) │ Hebbian Plasticity │ Sleep      │
├─────────────────────────────────────────────────────────────────┤
│  LAYER 1: TWO-SPEED MEMORY (Phase 11)                           │
│  Working Memory (Fast) ⊗ Long-Term Memory (Slow)                │
├─────────────────────────────────────────────────────────────────┤
│  LAYER 0: PHYSICAL REALITY                                      │
│  MuJoCo Physics, 28-DOF body, motor lag, and sensor noise       │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🧬 Biological Engine Breakdown

### 1. The Endocrine System & Plasticity
* **Neuromodulators:** Carl's learning rates and arousal thresholds are dynamically guided by internal transmitters:
  * **Dopamine (DA):** Signals reward prediction error and gates learning.
  * **Norepinephrine (NE):** Signals surprise and increases arousal (thermodynamic action noise).
  * **Serotonin (5-HT):** Acts as a safety guard to stabilize memory.
* **Synaptic Plasticity:** Utilizes **Reward-Modulated Spike-Timing-Dependent Plasticity (R-STDP)**. Associations strengthen when pre-synaptic neurons fire right before post-synaptic ones (LTP) and weaken when reversed (LTD).

### 2. Holographic Sub-Cortical Memory (HDC)
* Maps observations to a **40,000-dimensional bipolar hypervector space** (`{-1, +1}^D`) split into independent sub-cortical registers:
  * `M_spatial` — environmental traces
  * `M_affective` — emotional signatures
  * `M_procedural` — motor actions
* Memories are superposed using **bundling (+)** and **binding (⊗)** algebraic operations without crosstalk, pruning noise during offline sleep consolidation.

### 3. Hippocampal GPS Navigation
* Simulates **Nobel-Prize winning Grid & Place cell biology** (O'Keefe & Moser 2014):
  * **Grid Cells:** 3 modules of hexagonal tiling grid cell populations.
  * **Place Cells:** 200 place cells mapping local coordinates.
  * Coordinates path integration (dead reckoning) and maps safety gradients.

### 4. Liquid Time-Constant Motor Cortex (LTC)
* Driven by a recurrent LTC neural network (imitation-learned via **Behavioral Cloning** and fine-tuned with **PPO**):
  * Learns dynamic, physical reaching and grasping profiles.
  * Adjusts its internal time constants dynamically according to environmental variables.

---

## 📁 Repository Structure

```
carl-simulation/
├── brain/                    # Bio-computational components
│   ├── carl_stdp.py          # R-STDP & dopamine-modulated learning
│   ├── carl_grid_cells.py    # Hippocampal Grid & Place cells
│   ├── carl_physarum.py      # Slime-mold Steiner path optimizer
│   └── carl_omega_extensions.py
├── GENESIS/                  # Core simulation environment
│   ├── carl_scout/           # Primate Scout agents and controllers
│   │   ├── carl_cortex.py    # Main sensorimotor integration backbone
│   │   ├── carl_agent.py     # Actor-critic brain, Drives & HDC memory
│   │   └── carl_scout_ppo.py # PPO Arm-reaching fine-tuning
│   └── carl_autonomous.py    # Closed-loop reach-carry-place state machine
├── docs/                     # Research roadmaps, architecture, and manifests
├── dashboard/                # Live performance visualizations
└── tests/                    # Kinematics and physics stability diagnostics
```

---

## 🛠️ Getting Started

### 1. Setup Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run the Autonomous Demo
Execute the full pick-and-place sequence using the cognitive cortex:
```bash
python GENESIS/carl_autonomous.py
```
This runs the closed-loop task sequence:
$$\text{SEARCH} \to \text{APPROACH} \to \text{REACH} \to \text{GRASP} \to \text{LIFT} \to \text{CARRY} \to \text{PLACE} \to \text{SEARCH}$$
