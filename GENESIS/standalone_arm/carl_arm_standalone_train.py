"""
carl_arm_standalone_train.py — Autonomous RL curriculum trainer for the 8-DOF Standalone Arm.
Uses a Centered Absolute Control Linear Policy (no joint snapping, full reachability) and spinal reflex.
"""

import os
import time
import argparse
import numpy as np
import mujoco

MODEL_PATH = "carl_arm_standalone.xml"

# Joint positioning ranges (5 DOF)
ACT_LO = np.array([-0.3, -2.4, -2.8, -1.57, -1.3])
ACT_HI = np.array([ 2.2,  2.4,  0.0,  1.57,  1.3])

class StandaloneArmPolicy:
    """
    Linear Policy (W @ s + b) with Centered Absolute Control.
    b is initialized to match the start pose, and W is perturbed by ARS.
    This guarantees zero snapping at initialization, while retaining full speed and reachability.
    State: 26-D
    Action: 5-D (absolute targets for the positioning actuators)
    """
    STATE_DIM  = 26
    ACTION_DIM = 5

    def __init__(self):
        # W is initialized to zero to prevent initial drift
        self.W = np.zeros((self.ACTION_DIM, self.STATE_DIM), dtype=np.float32)
        
        # b maps exactly to the starting pose in [-1, 1] range
        start_pose = np.array([0.0, 0.5, -0.8, 1.5708, 0.35])
        self.b = (start_pose - ACT_LO) / (ACT_HI - ACT_LO) * 2.0 - 1.0

    def reset(self):
        pass

    def forward(self, s):
        out = np.dot(self.W, s) + self.b
        return np.clip(out, -1.0, 1.0) # Bounded action

    def get_params(self):
        return np.concatenate([self.W.ravel(), self.b])

    def set_params(self, p):
        idx = self.W.size
        self.W = p[:idx].reshape(self.W.shape).copy()
        self.b = p[idx:].copy()

    def param_count(self):
        return self.get_params().size

def build_env_cache(model):
    JT = mujoco.mjtObj.mjOBJ_JOINT
    ST = mujoco.mjtObj.mjOBJ_SITE
    BD = mujoco.mjtObj.mjOBJ_BODY
    SN = mujoco.mjtObj.mjOBJ_SENSOR
    AC = mujoco.mjtObj.mjOBJ_ACTUATOR

    jnames = ["shoulder_yaw_L", "shoulder_pitch_L", "elbow_L", "wrist_roll_L", "wrist_pitch_L",
              "thumb_L_knuckle", "index_L_knuckle", "middle_L_knuckle"]
    qpos_adrs = [model.jnt_qposadr[mujoco.mj_name2id(model, JT, n)] for n in jnames]
    dof_adrs  = [model.jnt_dofadr[mujoco.mj_name2id(model, JT, n)] for n in jnames]
    act_ids   = [mujoco.mj_name2id(model, AC, "act_" + n) for n in jnames]

    cube_bid = mujoco.mj_name2id(model, BD, "obj_cube_0")
    cube_jid = mujoco.mj_name2id(model, JT, "fj_cube_0")
    cube_qpos = model.jnt_qposadr[cube_jid]

    tip_sid = mujoco.mj_name2id(model, ST, "tip_site")

    touch_sids = [
        mujoco.mj_name2id(model, SN, "touch_thumb_L_s"),
        mujoco.mj_name2id(model, SN, "touch_index_L_s"),
        mujoco.mj_name2id(model, SN, "touch_middle_L_s"),
    ]

    return dict(
        qpos_adrs=qpos_adrs, dof_adrs=dof_adrs, act_ids=act_ids,
        cube_bid=cube_bid, cube_qpos=cube_qpos, tip_sid=tip_sid,
        touch_sids=touch_sids
    )

def get_observation(model, data, c, step_norm):
    qpos = np.array([data.qpos[i] for i in c['qpos_adrs']])
    qvel = np.array([data.qvel[i] for i in c['dof_adrs']]) * 0.1
    
    tip = data.site_xpos[c['tip_sid']].copy()
    cube = data.xpos[c['cube_bid']].copy()
    rel = (cube - tip) * 5.0
    
    res = np.zeros(6)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_SITE, c['tip_sid'], res, 0)
    tip_vel = res[3:6] * 0.5
    
    touch = np.array([data.sensordata[i] for i in c['touch_sids']]) * 0.01
    
    return np.concatenate([qpos, qvel, rel, tip_vel, touch, [step_norm]])

def compute_rewards(model, data, c, stage):
    tip = data.site_xpos[c['tip_sid']]
    cube = data.xpos[c['cube_bid']]
    dist = np.linalg.norm(cube - tip)
    
    touch = sum(data.sensordata[i] for i in c['touch_sids'])
    
    reward = 0.0
    done = False
    
    if stage == "reach":
        reward = -dist * 20.0
        reward += max(0.0, 0.15 - dist) * 150.0
        if dist < 0.025:
            reward += 1000.0
            done = True
            
    elif stage == "grasp":
        reward = -dist * 10.0
        reward += touch * 20.0
        if dist < 0.03 and touch > 0.05:
            reward += 2000.0
            done = True

    elif stage == "lift":
        reward = -dist * 5.0
        cube_z = cube[2]
        height = max(0.0, cube_z - 0.14)
        reward += height * 8000.0
        if height > 0.08:
            reward += 6000.0
            done = True

    return reward, done, dist

def reset_env(model, data, c, rng):
    mujoco.mj_resetData(model, data)
    
    # Direct joint position initialization to start posture (no-collision reset)
    data.qpos[c['qpos_adrs'][0]] = 0.0      # shoulder yaw
    data.qpos[c['qpos_adrs'][1]] = 0.5      # shoulder pitch
    data.qpos[c['qpos_adrs'][2]] = -0.8     # elbow
    data.qpos[c['qpos_adrs'][3]] = 1.5708   # wrist roll
    data.qpos[c['qpos_adrs'][4]] = 0.35     # wrist pitch
    
    # Initialize control targets to match
    init_ctrl = [0.0, 0.5, -0.8, 1.5708, 0.35, 0.0, 0.0, 0.0]
    for ai, val in zip(c['act_ids'], init_ctrl):
        data.ctrl[ai] = val
        
    # Cube position randomization
    cx = rng.uniform(0.20, 0.22)
    cy = rng.uniform(-0.04, 0.04)
    cz = 0.14
    
    data.qpos[c['cube_qpos'] : c['cube_qpos']+3] = [cx, cy, cz]
    data.qpos[c['cube_qpos']+3 : c['cube_qpos']+7] = [1.0, 0.0, 0.0, 0.0]
    
    mujoco.mj_forward(model, data)

def rollout(policy, model, data, c, stage, max_steps, rng):
    policy.reset()
    total_r = 0.0
    final_d = 99.0
    reset_env(model, data, c, rng)
    
    grasped = False
    
    for step in range(max_steps):
        s_norm = step / float(max_steps)
        s = get_observation(model, data, c, s_norm)
        
        # Policy output in [-1, 1]
        raw_act = policy.forward(s)
        
        # Map raw policy outputs directly to absolute control targets
        action = ACT_LO + (raw_act + 1.0) * 0.5 * (ACT_HI - ACT_LO)
        
        # Apply positioning commands
        for i in range(5):
            data.ctrl[c['act_ids'][i]] = action[i]
            
        # Grasp Reflex
        touch_val = sum(data.sensordata[i] for i in c['touch_sids'])
        if touch_val > 0.02:
            grasped = True
            
        if grasped:
            data.ctrl[c['act_ids'][5]] = 1.4   # thumb
            data.ctrl[c['act_ids'][6]] = -1.4  # index
            data.ctrl[c['act_ids'][7]] = -1.4  # middle
        else:
            data.ctrl[c['act_ids'][5]] = 0.0
            data.ctrl[c['act_ids'][6]] = 0.0
            data.ctrl[c['act_ids'][7]] = 0.0
            
        mujoco.mj_step(model, data)
        
        r, done, dist = compute_rewards(model, data, c, stage)
        total_r += r
        final_d = dist
        
        if done:
            break
            
    return total_r, final_d

def train(stage, episodes=300):
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data  = mujoco.MjData(model)
    c     = build_env_cache(model)
    rng   = np.random.default_rng()
    
    policy = StandaloneArmPolicy()
    
    save_path = "../memory/carl_arm_standalone_weights_v2.npz"
    os.makedirs("memory", exist_ok=True)
    if os.path.exists(save_path):
        try:
            saved = np.load(save_path)
            w = saved['weights']
            if w.size != policy.param_count():
                print(f"[LOAD] Param mismatch. Re-initializing policy fresh.")
            else:
                policy.set_params(w)
                print(f"[LOAD] Loaded weights from {save_path}")
        except Exception as e:
            print(f"[LOAD] Starting fresh: {e}")

    n_params = policy.param_count()
    theta = policy.get_params()
    
    # ARS parameters
    N = 16
    b = 8
    alpha = 0.02
    nu = 0.03
    max_steps = 150
    
    best_reward = -1e9
    
    print("\n" + "="*60)
    print(f"  CARL Standalone Arm Trainer (Centered Absolute Control) — Stage: {stage.upper()}")
    print("="*60)
    
    for gen in range(episodes):
        deltas = rng.standard_normal((N, n_params))
        rewards_p = np.zeros(N)
        rewards_n = np.zeros(N)
        
        for i in range(N):
            policy.set_params(theta + nu * deltas[i])
            rng_p = np.random.default_rng(seed=gen * 1000 + i)
            rewards_p[i], _ = rollout(policy, model, data, c, stage, max_steps, rng_p)
            
            policy.set_params(theta - nu * deltas[i])
            rng_n = np.random.default_rng(seed=gen * 1000 + i)
            rewards_n[i], _ = rollout(policy, model, data, c, stage, max_steps, rng_n)
            
        scores = np.maximum(rewards_p, rewards_n)
        top_idx = np.argsort(scores)[-b:]
        
        all_r = np.concatenate([rewards_p[top_idx], rewards_n[top_idx]])
        sigma_r = all_r.std() + 1e-8
        
        grad = np.zeros(n_params)
        for i in top_idx:
            grad += (rewards_p[i] - rewards_n[i]) * deltas[i]
        grad /= (b * sigma_r)
        
        theta += alpha * grad
        policy.set_params(theta)
        
        rng_eval = np.random.default_rng(seed=gen * 1000 + 999)
        ep_r, final_d = rollout(policy, model, data, c, stage, max_steps, rng_eval)
        
        if ep_r > best_reward:
            best_reward = ep_r
            np.savez(save_path, weights=theta)
            
        if (gen + 1) % 10 == 0 or gen == 0:
            print(f"Gen {gen+1:03d}/{episodes} | Reward: {ep_r:8.1f} | Best: {best_reward:8.1f} | Final Dist: {final_d:.4f}m | nu: {nu:.4f}")
            
        nu = max(0.005, nu * 0.995)
        
    print(f"\n[DONE] Best weights saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="reach", choices=["reach", "grasp", "lift"])
    parser.add_argument("--episodes", type=int, default=150)
    args = parser.parse_args()
    
    train(args.stage, args.episodes)
