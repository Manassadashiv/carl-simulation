"""
carl_arm_train.py — CARL Arm Manipulation Curriculum Trainer

Uses Augmented Random Search v2 (ARS) with antithetic sampling:
  - N=16 perturbation directions evaluated per generation
  - Both +δ and -δ tested for each direction (antithetic pairs)
  - Top-b=8 directions selected by reward
  - Normalized gradient estimate applied with adaptive learning rate
  - ~32x more sample-efficient than single-mutation hill climbing

Curriculum stages:
  reach  → move fingertip to object
  grasp  → close gripper around object
  lift   → raise object above ground
"""

import os
import math
import time
import argparse
import numpy as np
import mujoco
from mujoco import viewer as mj_viewer

# ─── Actuator index map ────────────────────────────────────────────────────────
# ctrl[0]  = act_w_left          (velocity ±35)
# ctrl[1]  = act_w_right         (velocity ±35)
# ctrl[2]  = act_hip_pitch       (pos ±0.55)
# ctrl[3]  = act_hip_roll        (pos ±0.38)
# ctrl[4]  = act_neck            (pos -0.6..0.7)
# ctrl[5]  = act_head            (pos ±1.4)
# ctrl[6]  = act_shoulder_yaw_L  (pos -0.1..1.57)
# ctrl[7]  = act_shoulder_L      (pos ±2.0)
# ctrl[8]  = act_elbow_L         (pos -2.44..0)
# ctrl[9]  = act_wrist_L         (pos ±1.57)
# ctrl[10] = act_grip_L          (pos 0..0.52)
# ctrl[11] = act_shoulder_yaw_R  (pos -1.57..0.1)
# ctrl[12] = act_shoulder_R      (pos ±2.0)
# ctrl[13] = act_elbow_R         (pos -2.44..0)
# ctrl[14] = act_wrist_R         (pos ±1.57)
# ctrl[15] = act_grip_R          (pos 0..0.52)

ARM_CTRL_SLICE = slice(6, 16)   # 10 arm actuators (5 left, 5 right)
ARM_CTRL_LOW   = np.array([-0.10, -2.0, -2.44, -1.57, 0.0,  -1.57, -2.0, -2.44, -1.57, 0.0])
ARM_CTRL_HIGH  = np.array([ 1.57,  2.0,  0.0,   1.57, 0.52,   0.10,  2.0,  0.0,   1.57, 0.52])


class ArmPolicy:
    """
    Liquid Time-Constant (LTC) Neural Network: state → arms.
    state_dim=29: 10 arm_pos + 10 arm_vel + 3 rel_target + 3 tip_vel + 2 tactile + 1 M(t)
    action_dim=10: 10 arm actuator targets
    """
    STATE_DIM  = 29
    ACTION_DIM = 10
    HIDDEN_DIM = 40  # Slightly larger for LTC dynamics

    def __init__(self):
        # State integration weights
        self.W_in  = np.random.randn(self.HIDDEN_DIM, self.STATE_DIM) * 0.1
        self.W_rec = np.random.randn(self.HIDDEN_DIM, self.HIDDEN_DIM) * 0.1
        self.b     = np.zeros(self.HIDDEN_DIM)
        
        # Liquid time-constant (tau) weights
        self.Tau_in  = np.random.randn(self.HIDDEN_DIM, self.STATE_DIM) * 0.1
        self.Tau_rec = np.random.randn(self.HIDDEN_DIM, self.HIDDEN_DIM) * 0.1
        self.tau_b   = np.ones(self.HIDDEN_DIM) * 1.5  # Base tau
        
        # Output weights
        self.W_out = np.random.randn(self.ACTION_DIM, self.HIDDEN_DIM) * 0.1
        self.b_out = np.zeros(self.ACTION_DIM)
        
        self.x = np.zeros(self.HIDDEN_DIM)

    def reset(self):
        """Reset internal recurrent hidden state."""
        self.x.fill(0.0)

    def forward(self, s, dt=0.01):
        def act(z):
            return np.tanh(z)
        
        # Liquid time-constant logic
        tau_modulation = act(np.dot(self.Tau_in, s) + np.dot(self.Tau_rec, self.x) + self.tau_b)
        tau = np.exp(tau_modulation)  # Time constant bounded > 0
        
        # Continuous-time update (Euler method)
        dx = -self.x + act(np.dot(self.W_in, s) + np.dot(self.W_rec, self.x) + self.b)
        self.x = self.x + (dt / tau) * dx
        
        out = np.dot(self.W_out, self.x) + self.b_out
        return np.tanh(out)  # [-1, 1] mapped to actuators later

    # ── Flat param vector for ARS ──────────────────────────────────────────
    def get_params(self):
        return np.concatenate([
            self.W_in.ravel(), self.W_rec.ravel(), self.b,
            self.Tau_in.ravel(), self.Tau_rec.ravel(), self.tau_b,
            self.W_out.ravel(), self.b_out
        ])

    def set_params(self, p):
        idx = 0
        for attr in ['W_in', 'W_rec', 'b', 'Tau_in', 'Tau_rec', 'tau_b', 'W_out', 'b_out']:
            arr = getattr(self, attr)
            sz = arr.size
            setattr(self, attr, p[idx:idx+sz].reshape(arr.shape))
            idx += sz

    def param_count(self):
        return self.get_params().size


# ─── MuJoCo helpers ────────────────────────────────────────────────────────────

def _resolve(model, obj_type, name):
    """Resolve a named MuJoCo object id, raise if missing."""
    i = mujoco.mj_name2id(model, obj_type, name)
    if i == -1:
        raise RuntimeError(f"MuJoCo object not found: {name!r}")
    return i

def build_cache(model):
    """Pre-resolve all IDs so the inner loop has zero string lookups."""
    JT = mujoco.mjtObj.mjOBJ_JOINT
    ST = mujoco.mjtObj.mjOBJ_SITE
    BD = mujoco.mjtObj.mjOBJ_BODY
    SN = mujoco.mjtObj.mjOBJ_SENSOR

    jnames = ['shoulder_yaw_L', 'shoulder_L', 'elbow_L', 'wrist_L', 'grip_L',
              'shoulder_yaw_R', 'shoulder_R', 'elbow_R', 'wrist_R', 'grip_R']
    arm_qpos = [model.jnt_qposadr[_resolve(model, JT, n)] for n in jnames]
    arm_dof  = [model.jnt_dofadr [_resolve(model, JT, n)] for n in jnames]

    carl_jid  = _resolve(model, JT, 'spatial_identity')
    Q_CARL    = model.jnt_qposadr[carl_jid]
    V_CARL    = model.jnt_dofadr [carl_jid]

    obj_bid   = _resolve(model, BD, 'obj_cube_0')
    obj_jid   = _resolve(model, JT, 'fj_cube_0')
    obj_qpos  = model.jnt_qposadr[obj_jid]

    tip_L_sid = _resolve(model, ST, 'touch_site_L')
    tip_R_sid = _resolve(model, ST, 'touch_site_R')

    touch_L   = _resolve(model, SN, 'touch_L')
    touch_R   = _resolve(model, SN, 'touch_R')

    return dict(
        arm_qpos=arm_qpos, arm_dof=arm_dof,
        Q_CARL=Q_CARL, V_CARL=V_CARL,
        obj_bid=obj_bid, obj_qpos=obj_qpos,
        tip_L_sid=tip_L_sid, tip_R_sid=tip_R_sid,
        touch_L=touch_L, touch_R=touch_R
    )


def get_state(model, data, c):
    """Build 29-D observation vector (no string lookups)."""
    arm_pos = [data.qpos[a] for a in c['arm_qpos']]
    arm_vel = [data.qvel[a] for a in c['arm_dof']]
    tip  = data.site_xpos[c['tip_L_sid']]
    obj  = data.xpos[c['obj_bid']]
    rel  = obj - tip
    
    # Compute actual Cartesian 3D linear velocity of the fingertip site in world orientation
    res = np.zeros(6)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_SITE, c['tip_L_sid'], res, 0)
    tip_vel = res[3:6]
    
    # Tactile inputs & M(t)
    tl = data.sensordata[c['touch_L']] if c['touch_L'] != -1 else 0.0
    tr = data.sensordata[c['touch_R']] if c['touch_R'] != -1 else 0.0
    M_t = 0.5  # Neutral default for disconnected brainstem training
    
    return np.concatenate([arm_pos, arm_vel, rel, tip_vel, [tl, tr, M_t]])


def compute_reward(data, c, stage):
    """Dense shaped reward. Returns (reward, done, dist)."""
    tip  = data.site_xpos[c['tip_L_sid']]
    obj  = data.xpos[c['obj_bid']]
    dist = float(np.linalg.norm(obj - tip))

    touch   = float(data.sensordata[c['touch_L']]) > 0.01
    grip_q  = data.qpos[c['arm_qpos'][4]]   # grip_L joint pos is index 4 in 5-DOF left arm
    closed  = grip_q > 0.18

    reward, done = 0.0, False

    if stage == 'reach':
        reward = -dist * 12.0
        reward += max(0.0, 0.20 - dist) * 50.0   # proximity bonus
        if dist < 0.045:
            reward += 800.0
            done = True

    elif stage == 'grasp':
        reward = -dist * 6.0
        if touch:             reward += 8.0
        if touch and closed:  reward += 150.0; done = True

    elif stage == 'lift':
        reward = -dist * 6.0
        if touch and closed:
            height = max(0.0, obj[2] - 0.04)
            reward += 60.0 + height * 1200.0
            if height > 0.10:
                reward += 1500.0; done = True

    return reward, done, dist


def reset_episode(model, data, c, stage, rng):
    """Reset simulation to a fresh start state."""
    mujoco.mj_resetData(model, data)
    # Place CARL at origin facing +X
    data.qpos[c['Q_CARL'] : c['Q_CARL']+3] = [0.0, 0.0, 0.04]
    
    # Initialize both arms to a natural home posture to avoid singularities
    # Left Arm: yaw=0.0, pitch=0.3, elbow=-1.2, wrist=0.0
    data.qpos[c['arm_qpos'][0]] = 0.0
    data.qpos[c['arm_qpos'][1]] = 0.3
    data.qpos[c['arm_qpos'][2]] = -1.2
    data.qpos[c['arm_qpos'][3]] = 0.0
    # Right Arm: yaw=0.0, pitch=0.3, elbow=-1.2, wrist=0.0
    data.qpos[c['arm_qpos'][5]] = 0.0
    data.qpos[c['arm_qpos'][6]] = 0.3
    data.qpos[c['arm_qpos'][7]] = -1.2
    data.qpos[c['arm_qpos'][8]] = 0.0
    
    # Comfortable spatial reach envelope outside the chassis collision boundaries
    ox = rng.uniform(0.15, 0.20)
    oy = rng.uniform(0.13, 0.17)
    oz = 0.035
    data.qpos[c['obj_qpos'] : c['obj_qpos']+3] = [ox, oy, oz]
    data.qpos[c['obj_qpos']+3 : c['obj_qpos']+7] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)


def rollout(policy, model, data, c, stage, max_steps, rng, viewer_obj=None):
    """
    Run one episode, return cumulative reward.
    Uses EMA smoothing on actions to prevent joint snapping.
    """
    policy.reset()
    total_r   = 0.0
    final_d   = 99.0
    reset_episode(model, data, c, stage, rng)

    for _ in range(max_steps):
        s   = get_state(model, data, c)
        raw = policy.forward(s)                 # in [-1, 1]
        # Rescale from [-1,1] → actual joint range
        action = ARM_CTRL_LOW + (raw + 1.0) * 0.5 * (ARM_CTRL_HIGH - ARM_CTRL_LOW)

        data.ctrl[:2]             = 0.0          # freeze wheels
        data.ctrl[ARM_CTRL_SLICE] = action
        
        # Lock CARL's base so he doesn't tip over while flailing!
        data.qpos[c['Q_CARL'] : c['Q_CARL']+3] = [0.0, 0.0, 0.04]
        data.qpos[c['Q_CARL']+3 : c['Q_CARL']+7] = [1.0, 0.0, 0.0, 0.0]
        data.qvel[c['V_CARL'] : c['V_CARL']+6] = 0.0
        
        mujoco.mj_step(model, data)

        r, done, final_d = compute_reward(data, c, stage)
        total_r += r

        if viewer_obj:
            viewer_obj.sync()
            time.sleep(0.004)

        if done:
            break

    return total_r, final_d


# ─── ARS v2 optimizer ──────────────────────────────────────────────────────────

def ars_train(stage, episodes, render, load_path=None):
    """
    Augmented Random Search v2 with antithetic sampling.
    N directions, top-b selected, normalized gradient update.
    """
    model = mujoco.MjModel.from_xml_path("vessel_kinetic.xml")
    data  = mujoco.MjData(model)
    cache = build_cache(model)
    rng   = np.random.default_rng()

    policy = ArmPolicy()
    if load_path and os.path.exists(load_path):
        try:
            saved = np.load(load_path)
            policy.set_params(saved['weights'])
            print(f"[LOAD] Resumed from {load_path}")
        except (ValueError, KeyError) as e:
            print(f"[LOAD] Incompatible weights ({e}) — starting fresh.")

    n_params   = policy.param_count()
    theta      = policy.get_params()

    # ARS hyperparameters
    N         = 16      # perturbation directions per generation
    b         = 8       # elite directions to keep
    alpha     = 0.015   # step size (learning rate)
    nu        = 0.03    # perturbation noise std
    max_steps = 250     # steps per rollout

    viewer_obj = None
    if render:
        print("[VIEWER] Launching passive viewer…")
        viewer_obj = mj_viewer.launch_passive(model, data)

    best_reward = -1e9
    os.makedirs("memory", exist_ok=True)

    for gen in range(episodes):
        # 1. Sample N random directions in parameter space (unscaled standard normal)
        deltas    = rng.standard_normal((N, n_params))

        rewards_p = np.zeros(N)   # reward with +delta
        rewards_n = np.zeros(N)   # reward with -delta (antithetic)

        # 2. Evaluate all 2*N rollouts
        for i in range(N):
            # Positive perturbation
            policy.set_params(theta + nu * deltas[i])
            rng_p = np.random.default_rng(seed=gen * 1000 + i)
            rp, _ = rollout(policy, model, data, cache, stage, max_steps, rng_p,
                            viewer_obj if i == 0 else None)
            rewards_p[i] = rp

            # Negative perturbation (antithetic)
            policy.set_params(theta - nu * deltas[i])
            rng_n = np.random.default_rng(seed=gen * 1000 + i)
            rn, _ = rollout(policy, model, data, cache, stage, max_steps, rng_n)
            rewards_n[i] = rn

        # 3. Select top-b directions by max(r+, r-) instead of absolute values
        scores   = np.maximum(rewards_p, rewards_n)
        top_idx  = np.argsort(scores)[-b:]

        # 4. Normalise rewards and compute gradient
        all_r    = np.concatenate([rewards_p[top_idx], rewards_n[top_idx]])
        sigma_r  = all_r.std() + 1e-8
        grad     = np.zeros(n_params)
        for i in top_idx:
            grad += (rewards_p[i] - rewards_n[i]) * deltas[i]
        grad /= (b * sigma_r)

        # 5. Gradient ascent step
        theta += alpha * grad
        policy.set_params(theta)

        # 6. Evaluate current policy (no noise)
        rng_eval = np.random.default_rng(seed=gen * 1000 + 999)
        ep_r, dist = rollout(policy, model, data, cache, stage, max_steps, rng_eval,
                             viewer_obj)

        if ep_r > best_reward:
            best_reward = ep_r
            np.savez("memory/carl_arm_weights.npz", weights=theta)

        print(f"Gen {gen+1:04d}/{episodes} | {stage.upper()} | "
              f"Reward: {ep_r:8.1f} | Best: {best_reward:8.1f} | "
              f"Dist: {dist:.3f}m | nu: {nu:.4f}")

        # Adaptive noise decay
        nu = max(0.005, nu * 0.998)

    print(f"\n[DONE] Weights saved to memory/carl_arm_weights.npz")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="CARL Arm Training — ARS v2")
    ap.add_argument("--stage",    default="reach", choices=["reach","grasp","lift"])
    ap.add_argument("--episodes", type=int, default=500)
    ap.add_argument("--render",   action="store_true")
    ap.add_argument("--load",     default="memory/carl_arm_weights.npz",
                    help="Resume from saved weights")
    args = ap.parse_args()

    print("=" * 60)
    print(f"  CARL Arm Training  |  Stage: {args.stage.upper()}")
    print(f"  ARS v2  |  N=16 directions  |  b=8 elite")
    print("=" * 60)
    ars_train(args.stage, args.episodes, args.render, args.load)
