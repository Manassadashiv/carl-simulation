"""
carl_scout_train.py — CARL Primate Scout Arm Training (ARS v2)

Autonomous training for the 28-DOF Primate Scout body.
The LTC neural network learns to control:
  - 5 positioning joints per arm (shoulder_yaw, shoulder_pitch, elbow, wrist_roll, wrist_pitch)
  - 3 finger joints per arm (thumb, index, middle knuckles)
  - Total: 16 arm actuators driven by the policy

Curriculum:
  reach  → move fingertip to target object
  grasp  → close fingers around object (multi-finger tactile feedback)
  lift   → raise object above table surface
"""

import os
import math
import time
import argparse
import numpy as np
import mujoco
from mujoco import viewer as mj_viewer

# ─── Actuator Index Map (Primate Scout) ───────────────────────────────────────
# ctrl[0-3]   = wheels (FL, FR, RL, RR) — velocity
# ctrl[4-9]   = spine (s1_pitch, s1_roll, s2_pitch, s2_yaw, s3_pitch, s3_roll)
# ctrl[10-11] = neck_tilt, head_pan
# ctrl[12-19] = Left arm:  sh_yaw, sh_pitch, elbow, wr_roll, wr_pitch, thumb, index, middle
# ctrl[20-27] = Right arm: sh_yaw, sh_pitch, elbow, wr_roll, wr_pitch, thumb, index, middle

import os
MODEL_PATH     = os.path.join(os.path.dirname(__file__), "carl_primate_scout.xml")
ARM_CTRL_SLICE = slice(12, 28)   # 16 arm actuators (8 left + 8 right)
LEFT_ARM_SLICE = slice(12, 20)   # 8 left arm actuators
RIGHT_ARM_SLICE = slice(20, 28)  # 8 right arm actuators

# Joint limits for all 16 arm actuators
ARM_CTRL_LOW  = np.array([
    # Left arm
    -0.30, -2.4, -2.8, -1.57, -1.3,   0.0, -1.4, -1.4,
    # Right arm
    -2.20, -2.4, -2.8, -1.57, -1.3,   0.0, -1.4, -1.4,
])
ARM_CTRL_HIGH = np.array([
    # Left arm
     2.20,  2.4,  0.0,  1.57,  1.3,   1.4,  0.0,  0.0,
    # Right arm
     0.30,  2.4,  0.0,  1.57,  1.3,   1.4,  0.0,  0.0,
])

# Spine actuator slice for body-language and throwing
SPINE_CTRL_SLICE = slice(4, 10)


class ScoutArmPolicy:
    """
    Liquid Time-Constant (LTC) Neural Network for the Primate Scout arms.

    state_dim = 51:
        16 arm joint positions + 16 arm joint velocities
        + 3 relative target vector (L) + 3 relative target vector (R)
        + 3 fingertip velocity (L) + 3 fingertip velocity (R)
        + 6 tactile sensors (thumb/index/middle × L/R)
        + 1 motivational signal M(t)

    action_dim = 16:
        8 left arm + 8 right arm actuator targets
    """
    STATE_DIM  = 51
    ACTION_DIM = 16
    HIDDEN_DIM = 48  # Slightly larger for 16 outputs

    def __init__(self):
        self.W_in  = np.random.randn(self.HIDDEN_DIM, self.STATE_DIM) * 0.08
        self.W_rec = np.random.randn(self.HIDDEN_DIM, self.HIDDEN_DIM) * 0.08
        self.b     = np.zeros(self.HIDDEN_DIM)

        self.Tau_in  = np.random.randn(self.HIDDEN_DIM, self.STATE_DIM) * 0.08
        self.Tau_rec = np.random.randn(self.HIDDEN_DIM, self.HIDDEN_DIM) * 0.08
        self.tau_b   = np.ones(self.HIDDEN_DIM) * 1.5

        self.W_out = np.random.randn(self.ACTION_DIM, self.HIDDEN_DIM) * 0.08
        self.b_out = np.zeros(self.ACTION_DIM)

        self.x = np.zeros(self.HIDDEN_DIM)

    def reset(self):
        self.x.fill(0.0)

    def forward(self, s, dt=0.005):
        tau_mod = np.tanh(self.Tau_in @ s + self.Tau_rec @ self.x + self.tau_b)
        tau = np.exp(tau_mod)
        dx = -self.x + np.tanh(self.W_in @ s + self.W_rec @ self.x + self.b)
        self.x = self.x + (dt / tau) * dx
        return np.tanh(self.W_out @ self.x + self.b_out)

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


# ─── MuJoCo Helpers ───────────────────────────────────────────────────────────

def _resolve(model, obj_type, name):
    i = mujoco.mj_name2id(model, obj_type, name)
    if i == -1:
        raise RuntimeError(f"MuJoCo object not found: {name!r}")
    return i


def build_scout_cache(model):
    """Pre-resolve all body/joint/site/sensor IDs for zero string lookups."""
    JT = mujoco.mjtObj.mjOBJ_JOINT
    ST = mujoco.mjtObj.mjOBJ_SITE
    BD = mujoco.mjtObj.mjOBJ_BODY
    SN = mujoco.mjtObj.mjOBJ_SENSOR

    # All 16 arm joint names (8 left + 8 right)
    left_joints = [
        'shoulder_yaw_L', 'shoulder_pitch_L', 'elbow_L',
        'wrist_roll_L', 'wrist_pitch_L',
        'thumb_L_knuckle', 'index_L_knuckle', 'middle_L_knuckle'
    ]
    right_joints = [
        'shoulder_yaw_R', 'shoulder_pitch_R', 'elbow_R',
        'wrist_roll_R', 'wrist_pitch_R',
        'thumb_R_knuckle', 'index_R_knuckle', 'middle_R_knuckle'
    ]
    all_arm_joints = left_joints + right_joints

    arm_qpos = [model.jnt_qposadr[_resolve(model, JT, n)] for n in all_arm_joints]
    arm_dof  = [model.jnt_dofadr [_resolve(model, JT, n)] for n in all_arm_joints]

    # Root joint (freejoint for base platform)
    root_jid = _resolve(model, JT, 'root_joint')
    Q_ROOT   = model.jnt_qposadr[root_jid]
    V_ROOT   = model.jnt_dofadr [root_jid]

    # Target objects
    obj_bid  = _resolve(model, BD, 'obj_cube_0')
    obj_jid  = _resolve(model, JT, 'fj_cube_0')
    obj_qpos = model.jnt_qposadr[obj_jid]

    obj1_bid  = _resolve(model, BD, 'obj_cube_1')
    obj1_jid  = _resolve(model, JT, 'fj_cube_1')
    obj1_qpos = model.jnt_qposadr[obj1_jid]

    # Fingertip sites (use index finger as primary IK tip)
    tip_L_sid = _resolve(model, ST, 'touch_index_L')
    tip_R_sid = _resolve(model, ST, 'touch_index_R')

    # Touch sensors (6 total)
    touch_sensors = {}
    for name in ['touch_thumb_L_s', 'touch_index_L_s', 'touch_middle_L_s',
                  'touch_thumb_R_s', 'touch_index_R_s', 'touch_middle_R_s']:
        touch_sensors[name] = _resolve(model, SN, name)

    return dict(
        arm_qpos=arm_qpos, arm_dof=arm_dof,
        left_arm_qpos=arm_qpos[:8], left_arm_dof=arm_dof[:8],
        right_arm_qpos=arm_qpos[8:], right_arm_dof=arm_dof[8:],
        Q_ROOT=Q_ROOT, V_ROOT=V_ROOT,
        obj_bid=obj_bid, obj_qpos=obj_qpos,
        obj1_bid=obj1_bid, obj1_qpos=obj1_qpos,
        tip_L_sid=tip_L_sid, tip_R_sid=tip_R_sid,
        touch_sensors=touch_sensors,
    )


def get_scout_state(model, data, c, stage='reach', active_arms='both'):
    """
    Build 51-D observation vector for the Scout policy.

    [0:16]  arm joint positions (8L + 8R)
    [16:32] arm joint velocities (8L + 8R)
    [32:35] relative vector from left index fingertip to left target (cube_0)
    [35:38] relative vector from right index fingertip to right target (cube_1)
    [38:41] Cartesian velocity of left index fingertip
    [41:44] Cartesian velocity of right index fingertip
    [44:50] 6 tactile sensor readings
    [50]    motivational signal M(t)
    """
    arm_pos = np.array([data.qpos[a] for a in c['arm_qpos']])
    arm_vel = np.array([data.qvel[a] for a in c['arm_dof']])

    tip_L = data.site_xpos[c['tip_L_sid']]
    tip_R = data.site_xpos[c['tip_R_sid']]

    # Left target vector: zero if left arm is inactive
    if active_arms in ['left', 'both']:
        obj_L = data.xpos[c['obj_bid']]    # cube_0 -> left arm
        rel_L = obj_L - tip_L
    else:
        rel_L = np.zeros(3)

    # Right target vector: zero if right arm is inactive
    if active_arms in ['right', 'both']:
        obj_R = data.xpos[c['obj1_bid']]   # cube_1 -> right arm
        rel_R = obj_R - tip_R
    else:
        rel_R = np.zeros(3)

    # Left fingertip velocity
    res_L = np.zeros(6)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_SITE, c['tip_L_sid'], res_L, 0)
    tip_L_vel = res_L[3:6]

    # Right fingertip velocity
    res_R = np.zeros(6)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_SITE, c['tip_R_sid'], res_R, 0)
    tip_R_vel = res_R[3:6]

    # 6 tactile readings
    ts = c['touch_sensors']
    tactile = np.array([
        data.sensordata[ts['touch_thumb_L_s']],
        data.sensordata[ts['touch_index_L_s']],
        data.sensordata[ts['touch_middle_L_s']],
        data.sensordata[ts['touch_thumb_R_s']],
        data.sensordata[ts['touch_index_R_s']],
        data.sensordata[ts['touch_middle_R_s']],
    ])

    M_t = 0.5  # Neutral motivational signal for standalone training
    return np.concatenate([arm_pos, arm_vel, rel_L, rel_R, tip_L_vel, tip_R_vel, tactile, [M_t]])


def compute_scout_reward(data, c, stage, active_arms='both'):
    """
    Bilateral/Unilateral reward — evaluates based on active_arms mode.
    If left/right only, the other arm is penalized for moving from its home position.
    """
    tip_L = data.site_xpos[c['tip_L_sid']]
    tip_R = data.site_xpos[c['tip_R_sid']]
    obj_L = data.xpos[c['obj_bid']]     # cube_0
    obj_R = data.xpos[c['obj1_bid']]    # cube_1

    dist_L = float(np.linalg.norm(obj_L - tip_L))
    dist_R = float(np.linalg.norm(obj_R - tip_R))

    ts = c['touch_sensors']

    # Left hand metrics
    touch_L_sum = (
        float(data.sensordata[ts['touch_thumb_L_s']]) +
        float(data.sensordata[ts['touch_index_L_s']]) +
        float(data.sensordata[ts['touch_middle_L_s']])
    )
    any_touch_L = touch_L_sum > 0.01
    multi_touch_L = touch_L_sum > 0.03
    thumb_L_q = data.qpos[c['arm_qpos'][5]]
    index_L_q = data.qpos[c['arm_qpos'][6]]
    middle_L_q = data.qpos[c['arm_qpos'][7]]
    fingers_closed_L = thumb_L_q > 0.4 and index_L_q < -0.4 and middle_L_q < -0.4

    # Right hand metrics
    touch_R_sum = (
        float(data.sensordata[ts['touch_thumb_R_s']]) +
        float(data.sensordata[ts['touch_index_R_s']]) +
        float(data.sensordata[ts['touch_middle_R_s']])
    )
    any_touch_R = touch_R_sum > 0.01
    multi_touch_R = touch_R_sum > 0.03
    thumb_R_q = data.qpos[c['arm_qpos'][13]]
    index_R_q = data.qpos[c['arm_qpos'][14]]
    middle_R_q = data.qpos[c['arm_qpos'][15]]
    fingers_closed_R = thumb_R_q > 0.4 and index_R_q < -0.4 and middle_R_q < -0.4

    # Rest pose reference
    home_L = np.array([0.3, 0.4, -1.4, 0.0, 0.0, 0.0, 0.0, 0.0])
    home_R = np.array([-0.3, 0.4, -1.4, 0.0, 0.0, 0.0, 0.0, 0.0])

    # Inactive arm penalty calculation
    idle_penalty = 0.0
    if active_arms == 'left':
        qpos_R = np.array([data.qpos[a] for a in c['right_arm_qpos']])
        idle_penalty = -float(np.linalg.norm(qpos_R - home_R)) * 5.0
    elif active_arms == 'right':
        qpos_L = np.array([data.qpos[a] for a in c['left_arm_qpos']])
        idle_penalty = -float(np.linalg.norm(qpos_L - home_L)) * 5.0

    reward, done = 0.0, False

    if stage == 'reach' or stage == 'coordinate':
        if active_arms == 'both':
            reward = -dist_L * 10.0 - dist_R * 10.0
            if dist_L < 0.05 and dist_R < 0.05:
                reward += 1000.0
                done = True
        elif active_arms == 'left':
            reward = -dist_L * 10.0 + idle_penalty
            if dist_L < 0.05:
                reward += 1000.0
                done = True
        elif active_arms == 'right':
            reward = -dist_R * 10.0 + idle_penalty
            if dist_R < 0.05:
                reward += 1000.0
                done = True

    elif stage == 'grasp':
        if active_arms == 'both':
            # Distance penalty for both arms
            reward = -dist_L * 6.0 - dist_R * 6.0

            # Left arm grasp evaluation
            if not any_touch_L: reward -= 2.0
            if not multi_touch_L: reward -= 5.0
            if not fingers_closed_L: reward -= 10.0

            # Right arm grasp evaluation
            if not any_touch_R: reward -= 2.0
            if not multi_touch_R: reward -= 5.0
            if not fingers_closed_R: reward -= 10.0

            if (multi_touch_L and fingers_closed_L and
                    multi_touch_R and fingers_closed_R):
                reward += 1000.0
                done = True
        elif active_arms == 'left':
            reward = -dist_L * 6.0 + idle_penalty
            if not any_touch_L: reward -= 2.0
            if not multi_touch_L: reward -= 5.0
            if not fingers_closed_L: reward -= 10.0
            if multi_touch_L and fingers_closed_L:
                reward += 1000.0
                done = True
        elif active_arms == 'right':
            reward = -dist_R * 6.0 + idle_penalty
            if not any_touch_R: reward -= 2.0
            if not multi_touch_R: reward -= 5.0
            if not fingers_closed_R: reward -= 10.0
            if multi_touch_R and fingers_closed_R:
                reward += 1000.0
                done = True

    elif stage == 'lift':
        if active_arms == 'both':
            # Distance penalty for both arms
            reward = -dist_L * 4.0 - dist_R * 4.0

            # Left arm lift evaluation
            if multi_touch_L and fingers_closed_L:
                height_L = max(0.0, obj_L[2] - 0.03)
                reward -= 100.0 * max(0.0, 0.08 - height_L)
            else:
                reward -= 8.0

            # Right arm lift evaluation
            if multi_touch_R and fingers_closed_R:
                height_R = max(0.0, obj_R[2] - 0.03)
                reward -= 100.0 * max(0.0, 0.08 - height_R)
            else:
                reward -= 8.0

            height_L = max(0.0, obj_L[2] - 0.03) if (multi_touch_L and fingers_closed_L) else 0.0
            height_R = max(0.0, obj_R[2] - 0.03) if (multi_touch_R and fingers_closed_R) else 0.0
            if height_L > 0.08 and height_R > 0.08:
                reward += 1000.0
                done = True
        elif active_arms == 'left':
            reward = -dist_L * 4.0 + idle_penalty
            if multi_touch_L and fingers_closed_L:
                height_L = max(0.0, obj_L[2] - 0.03)
                reward -= 100.0 * max(0.0, 0.08 - height_L)
                if height_L > 0.08:
                    reward += 1000.0
                    done = True
            else:
                reward -= 8.0
        elif active_arms == 'right':
            reward = -dist_R * 4.0 + idle_penalty
            if multi_touch_R and fingers_closed_R:
                height_R = max(0.0, obj_R[2] - 0.03)
                reward -= 100.0 * max(0.0, 0.08 - height_R)
                if height_R > 0.08:
                    reward += 1000.0
                    done = True
            else:
                reward -= 8.0

    if active_arms == 'both':
        avg_dist = (dist_L + dist_R) / 2.0
    elif active_arms == 'left':
        avg_dist = dist_L
    else:
        avg_dist = dist_R

    return reward, done, avg_dist


def reset_scout_episode(model, data, c, stage, active_arms, rng):
    """Reset simulation — spawn active cubes, hide inactive ones."""
    mujoco.mj_resetData(model, data)

    # Place CARL
    data.qpos[c['Q_ROOT']:c['Q_ROOT']+3] = [0.0, 0.0, 0.058]
    data.qpos[c['Q_ROOT']+3:c['Q_ROOT']+7] = [1, 0, 0, 0]

    # Natural arm rest pose (both arms)
    home_L = [0.3, 0.4, -1.4, 0.0, 0.0, 0.0, 0.0, 0.0]
    home_R = [-0.3, 0.4, -1.4, 0.0, 0.0, 0.0, 0.0, 0.0]
    for i, val in enumerate(home_L):
        data.qpos[c['arm_qpos'][i]] = val
    for i, val in enumerate(home_R):
        data.qpos[c['arm_qpos'][8 + i]] = val

    # Hide unused objects (ball and cylinder)
    JT = mujoco.mjtObj.mjOBJ_JOINT
    for jname in ['fj_ball_0', 'fj_cyl_0']:
        jid = mujoco.mj_name2id(model, JT, jname)
        if jid != -1:
            qpos_adr = model.jnt_qposadr[jid]
            offset = 0.2 if 'ball' in jname else 0.4
            data.qpos[qpos_adr:qpos_adr+3] = [2.0, 2.0 + offset, 0.020]
            data.qpos[qpos_adr+3:qpos_adr+7] = [1, 0, 0, 0]

    # Spawning logic based on active_arms
    if active_arms in ['left', 'both']:
        ox_L = rng.uniform(0.15, 0.24)
        oy_L = rng.uniform(0.08, 0.22)
        data.qpos[c['obj_qpos']:c['obj_qpos']+3] = [ox_L, oy_L, 0.020]
        data.qpos[c['obj_qpos']+3:c['obj_qpos']+7] = [1, 0, 0, 0]
    else:
        # Hide cube 0 far away on the floor
        data.qpos[c['obj_qpos']:c['obj_qpos']+3] = [2.0, 2.0, 0.020]
        data.qpos[c['obj_qpos']+3:c['obj_qpos']+7] = [1, 0, 0, 0]

    if active_arms in ['right', 'both']:
        ox_R = rng.uniform(0.15, 0.24)
        oy_R = rng.uniform(-0.22, -0.08)
        data.qpos[c['obj1_qpos']:c['obj1_qpos']+3] = [ox_R, oy_R, 0.020]
        data.qpos[c['obj1_qpos']+3:c['obj1_qpos']+7] = [1, 0, 0, 0]
    else:
        # Hide cube 1 far away on the floor
        data.qpos[c['obj1_qpos']:c['obj1_qpos']+3] = [2.0, 2.5, 0.020]
        data.qpos[c['obj1_qpos']+3:c['obj1_qpos']+7] = [1, 0, 0, 0]

    mujoco.mj_forward(model, data)


def rollout(policy, model, data, c, stage, max_steps, rng, viewer_obj=None, active_arms=None):
    """Run one episode, return cumulative reward."""
    policy.reset()
    total_r, final_d = 0.0, 99.0

    # Curriculum: randomize active arms in coordinate stage, or in all stages to make them fully general
    if active_arms is None:
        if stage in ['reach', 'grasp', 'lift', 'coordinate']:
            active_arms = rng.choice(['left', 'right', 'both'])
        else:
            active_arms = 'both'

    reset_scout_episode(model, data, c, stage, active_arms, rng)

    for step in range(max_steps):
        s   = get_scout_state(model, data, c, stage, active_arms)
        raw = policy.forward(s)
        action = ARM_CTRL_LOW + (raw + 1.0) * 0.5 * (ARM_CTRL_HIGH - ARM_CTRL_LOW)

        # Freeze wheels and spine during arm training
        data.ctrl[:4]  = 0.0
        data.ctrl[4:10] = 0.0
        data.ctrl[ARM_CTRL_SLICE] = action

        # Lock base position
        data.qpos[c['Q_ROOT']:c['Q_ROOT']+3] = [0.0, 0.0, 0.058]
        data.qpos[c['Q_ROOT']+3:c['Q_ROOT']+7] = [1, 0, 0, 0]
        data.qvel[c['V_ROOT']:c['V_ROOT']+6] = 0.0

        mujoco.mj_step(model, data)

        r, done, final_d = compute_scout_reward(data, c, stage, active_arms)
        total_r += r

        if viewer_obj:
            viewer_obj.sync()
            time.sleep(0.003)

        if done:
            break

    return total_r, final_d


# ─── ARS v2 Optimizer ─────────────────────────────────────────────────────────

def ars_train(stage, generations, render, load_path=None):
    """Augmented Random Search v2 with antithetic sampling."""
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data  = mujoco.MjData(model)
    cache = build_scout_cache(model)
    rng   = np.random.default_rng()

    policy = ScoutArmPolicy()
    if load_path and os.path.exists(load_path):
        try:
            saved = np.load(load_path)
            policy.set_params(saved['weights'])
            print(f"[LOAD] Resumed from {load_path}")
        except (ValueError, KeyError) as e:
            print(f"[LOAD] Incompatible weights ({e}) — starting fresh.")

    n_params = policy.param_count()
    theta    = policy.get_params()

    # ARS hyperparameters
    N         = 20     # perturbation directions (slightly more for higher DOF)
    b         = 10     # elite directions
    alpha     = 0.012  # learning rate
    nu        = 0.025  # perturbation noise
    max_steps = 300    # steps per rollout

    viewer_obj = None
    if render:
        print("[VIEWER] Launching passive viewer…")
        viewer_obj = mj_viewer.launch_passive(model, data)

    best_reward = -1e9
    os.makedirs("memory", exist_ok=True)
    save_path = "memory/carl_scout_arm_weights.npz"

    print(f"\n[TRAIN] Policy params: {n_params:,}")
    print(f"[TRAIN] ARS v2: N={N}, b={b}, alpha={alpha}, nu={nu}")
    print(f"[TRAIN] Stage: {stage.upper()}, Generations: {generations}\n")

    for gen in range(generations):
        deltas    = rng.standard_normal((N, n_params))
        rewards_p = np.zeros(N)
        rewards_n = np.zeros(N)

        for i in range(N):
            # +δ
            policy.set_params(theta + nu * deltas[i])
            rng_p = np.random.default_rng(seed=gen * 1000 + i)
            rp, _ = rollout(policy, model, data, cache, stage, max_steps, rng_p,
                            viewer_obj if i == 0 else None)
            rewards_p[i] = rp

            # -δ (antithetic)
            policy.set_params(theta - nu * deltas[i])
            rng_n = np.random.default_rng(seed=gen * 1000 + i)
            rn, _ = rollout(policy, model, data, cache, stage, max_steps, rng_n)
            rewards_n[i] = rn

        # Select top-b
        scores  = np.maximum(rewards_p, rewards_n)
        top_idx = np.argsort(scores)[-b:]

        # Normalized gradient
        all_r   = np.concatenate([rewards_p[top_idx], rewards_n[top_idx]])
        sigma_r = all_r.std() + 1e-8
        grad    = np.zeros(n_params)
        for i in top_idx:
            grad += (rewards_p[i] - rewards_n[i]) * deltas[i]
        grad /= (b * sigma_r)

        # Gradient ascent
        theta += alpha * grad
        policy.set_params(theta)

        # Evaluate current on all configurations for a stable metric
        r_sum = 0.0
        d_sum = 0.0
        eval_modes = ['left', 'right', 'both'] if stage in ['reach', 'grasp', 'lift', 'coordinate'] else ['both']
        
        for mode in eval_modes:
            rng_eval = np.random.default_rng(seed=gen * 1000 + 999)
            # Only render the 'both' mode to avoid visual flickering/viewer reset overhead
            ep_r, dist = rollout(policy, model, data, cache, stage, max_steps, rng_eval,
                                 viewer_obj if (mode == 'both' and render) else None,
                                 active_arms=mode)
            r_sum += ep_r
            d_sum += dist
            
        avg_r = r_sum / len(eval_modes)
        avg_dist = d_sum / len(eval_modes)

        if avg_r > best_reward:
            best_reward = avg_r
            np.savez(save_path, weights=theta)

        print(f"Gen {gen+1:04d}/{generations} | {stage.upper():5s} | "
              f"Reward: {avg_r:9.1f} | Best: {best_reward:9.1f} | "
              f"Dist: {avg_dist:.4f}m | nu: {nu:.4f}")

        nu = max(0.004, nu * 0.997)

    print(f"\n[DONE] Best weights -> {save_path}")
    if viewer_obj:
        viewer_obj.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="CARL Primate Scout — Arm Training (ARS v2)")
    ap.add_argument("--stage",    default="reach", choices=["reach", "grasp", "lift", "coordinate"])
    ap.add_argument("--gens",     type=int, default=500, help="Number of ARS generations")
    ap.add_argument("--render",   action="store_true", help="Show viewer during training")
    ap.add_argument("--load",     default="memory/carl_scout_arm_weights.npz",
                    help="Resume from saved weights")
    args = ap.parse_args()

    print("=" * 65)
    print(f"  CARL Primate Scout — Arm Training")
    print(f"  Stage: {args.stage.upper()} | ARS v2 | N=20, b=10")
    print(f"  Model: {MODEL_PATH}")
    print("=" * 65)
    ars_train(args.stage, args.gens, args.render, args.load)
