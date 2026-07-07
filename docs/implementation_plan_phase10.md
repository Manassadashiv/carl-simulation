# CARL Phase 10: Full Cognitive Architecture

This is the complete technical blueprint for transforming CARL into a full neuro-inspired cognitive agent. Every feature is specified with exact mathematics and implementation details.

---

## Architecture Overview

```
SENSORS (6D state)
  x_pos, x_vel, pitch, pitch_vel, neck_angle, optical_flow
        ↓
WORKING MEMORY (fast RLS, λ=0.99, resets each life)
        ↓ blended with ↓
LONG-TERM MEMORY (slow RLS, λ=0.9999, never resets)
        ↓
AMYGDALA FILTER → only emotionally significant surprises → LTM
        ↓
CONFIDENCE-WEIGHTED IMAGINATION (free energy, uncertainty-discounted)
        ↓
DIRECTED CURIOSITY (information-directed sampling in dangerous regions)
        ↓
ACTION → ROBOT A + ROBOT B (shared LTM)
        ↓
DEATH → HIPPOCAMPAL REPLAY (top-50 traumas replayed into LTM)
```

---

## Module 1: Extended State Vector (6D)

Current: `x = [x_pos, x_vel, pitch, pitch_vel]` (4D)  
New: `x = [x_pos, x_vel, pitch, pitch_vel, neck_angle, optical_flow]` (6D)

**Neck angle**: `p.getJointState(robot_id, 2)[0]` — measures head displacement from body.  
**Optical flow**: Simplified as `optical_flow = (pitch_now - pitch_prev) / dt` — a proxy for visual motion without a full camera. Clean and fast.

**RLS matrix**: `T ∈ ℝ^{7×6}` (7 = 6 state + 1 action, 6 outputs)

---

## Module 2: Two-Speed Memory

```python
# Working Memory: fast, volatile, episode-specific
T_work, P_work  (λ = 0.990, P0 = 500·I, RESETS each life)

# Long-Term Memory: slow, stable, cross-episode
T_long, P_long  (λ = 0.9999, P0 = 2000·I, NEVER resets, has P-floor = 0.005·I)

# Blended prediction:
alpha = consciousness(P_work)  # 0 = trust LTM, 1 = trust WM
x_pred = alpha * (T_work.T @ Phi) + (1 - alpha) * (T_long.T @ Phi)

# Surprise: deviation from blended prediction
surprise = ||x_next - x_pred||
```

---

## Module 3: Amygdala (Consequence Filter)

Only write to Long-Term Memory when the experience is **emotionally significant** — i.e., when danger is elevated.

```python
danger_ema = 0.05 * danger_level + 0.95 * danger_ema

if danger_ema > 0.25:  # Emotionally significant threshold
    T_long, P_long = rls_update(T_long, P_long, xk, uk, xn, lam=0.9999)
    # This moment matters. Remember it.
else:
    pass  # Routine. Forget it.

# Working memory ALWAYS updates (every step)
T_work, P_work = rls_update(T_work, P_work, xk, uk, xn, lam=0.990)
```

---

## Module 4: Hippocampal Replay (Sleep Phase)

Between episodes, replay the 50 most surprising moments into LTM.

```python
# During episode: store top experiences
episode_buffer = []  # list of (xk, uk, xn, surprise)
if surprise > REPLAY_THRESHOLD:
    episode_buffer.append((xk, uk, xn, surprise))

# On death (Sleep Phase):
top_memories = sorted(episode_buffer, key=lambda m: m[3], reverse=True)[:50]
for (xk_mem, uk_mem, xn_mem, _) in top_memories:
    for _ in range(3):  # replay each memory 3 times
        T_long, P_long = rls_update(T_long, P_long, xk_mem, uk_mem, xn_mem, lam=0.9999)
```

---

## Module 5: Confidence-Weighted Imagination

Current rollouts don't know if their predictions are trustworthy.  
Fix: discount future steps by model uncertainty.

```python
# Uncertainty proxy per simulated step:
# Use diagonal of P_long as uncertainty signal
P_diag_norm = np.trace(P_long) / P0_LONG_TR  # 0=certain, 1=uncertain

def pick_action(x, T_work, T_long, P_long, D, M_terrain):
    for u in ACTIONS:
        xs, F = x.copy(), 0.0
        for h in range(HORIZON):
            xs_next = Ad @ xs + Bd * u
            
            # Confidence degrades with model uncertainty and horizon depth
            confidence = (1 - P_diag_norm) * (0.95 ** h)
            
            pred_surprise = np.linalg.norm(xs_next - xs) * 0.1
            danger        = danger_at(D, xs_next[2], xs_next[3])
            terrain_risk  = terrain_uncertainty(M_terrain, xs_next[0])
            
            # Discount low-confidence futures
            F += confidence * (pred_surprise + danger + terrain_risk)
            xs = xs_next
```

---

## Module 6: Directed Curiosity (Information-Directed Sampling)

Instead of random exploration, seek actions that reduce uncertainty in **dangerous regions**.

```python
def directed_explore(x, T_work, P_work, D):
    # Find the most uncertain + dangerous region of state space
    # by combining the danger map with uncertainty estimate
    
    best_u, best_info_gain = 0.0, -float('inf')
    for u in ACTIONS:
        xs = x.copy()
        for _ in range(10):
            xs = Ad_work @ xs + Bd_work * u
        
        # Information gain = uncertainty at the reached state
        # weighted by how dangerous that region is
        Phi_reach  = np.append(xs, u).reshape(7, 1)
        local_unc  = float((Phi_reach.T @ P_work @ Phi_reach).squeeze())
        danger_val = danger_at(D, xs[2], xs[3])
        
        info_gain  = local_unc * (1 + danger_val)  # explore dangerous unknowns first
        if info_gain > best_info_gain:
            best_info_gain, best_u = info_gain, u
    
    return best_u
```

---

## Module 7: Dynamic Terrain

### 7a. Procedural Slopes
```python
slope = np.random.uniform(-0.052, 0.052)  # ±3 degrees in radians
p.resetBasePositionAndOrientation(
    planeId, [0, 0, 0], p.getQuaternionFromEuler([slope, 0, 0])
)
```

### 7b. Wind Disturbances
```python
# Every 15-30 seconds, apply invisible lateral impulse
if step % wind_interval == 0:
    wind_interval = np.random.randint(15*240, 30*240)
    fx = np.random.uniform(-3.0, 3.0)
    p.applyExternalForce(robot_id, -1, [fx, 0, 0], [0, 0, 0.3], p.WORLD_FRAME)
```

### 7c. Earthquake Mode
```python
t_elapsed = step * DT
quake_amp = min(t_elapsed / 60.0, 0.5)  # grows over 60 seconds
quake_force = quake_amp * np.sin(2 * np.pi * 1.5 * t_elapsed)
p.applyExternalForce(robot_id, -1, [quake_force, 0, 0], [0,0,0], p.WORLD_FRAME)
```

### 7d. Terrain Memory (Mapper)
```python
M_terrain = np.ones((100,))  # 1D grid: x_pos in [-2, 2]
def terrain_update(M, x_pos, surprise, a=0.1):
    i = int(np.clip((x_pos + 2.0) / 4.0 * 100, 0, 99))
    M[i] = (1 - a) * M[i] + a * (1 - surprise)  # high surprise = low trust
```

---

## Module 8: Proprioceptive Fatigue

```python
fatigue = 0.0  # resets each episode

# Each step:
fatigue = 0.999 * fatigue + 0.001 * abs(u)

# Add fatigue cost to rollout:
F += 0.02 * fatigue  # encourages efficient, minimal-effort control
```

---

## Module 9: Two Bodies, One Mind

```python
# Both robots share: T_long, P_long, D_global, M_terrain
# Each robot has its own: T_work, P_work, episode_buffer, fatigue

robot_A = spawn_robot(position=[0, -0.3, 0.08])
robot_B = spawn_robot(position=[0,  0.3, 0.08])

# Each step: update both robots independently
# When either dies:
#   1. Run sleep/replay → update shared T_long
#   2. Burn trauma → update shared D_global  
#   3. Respawn that robot
#   4. Other robot continues uninterrupted
```

---

## Dashboard Additions

| New Panel | Shows |
|---|---|
| **Dual Body View** | Side-by-side survival time for Body A and Body B |
| **Memory Streams** | Working Memory surprise vs. Long-Term Memory surprise |
| **Sleep Indicator** | Flashes "CONSOLIDATING" during replay phase |
| **Terrain Strip** | Live 1D terrain trust map showing safe vs risky zones |
| **Fatigue Meter** | Cumulative energy expenditure per episode |

---

## Files to Create

| File | Purpose |
|---|---|
| `phase10_cognitive.py` | Full cognitive backend |
| `dashboard/index.html` | Updated dashboard (all new panels) |

> [!IMPORTANT]
> This is a complete rewrite. `phase9_active_inference.py` will be superseded.
> Confirm and I will build the entire system in one shot.
