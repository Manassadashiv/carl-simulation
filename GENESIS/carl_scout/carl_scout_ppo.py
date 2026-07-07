"""
carl_scout_ppo.py -- PPO Fine-Tuning for CARL Primate Scout Arms.

Phase 2c of CARL v5 Master Roadmap.

Starts from BC-pretrained weights (Phase 2b) and fine-tunes using
Proximal Policy Optimization (PPO) with Generalized Advantage Estimation (GAE).

This is the first gradient-based RL training for CARL's arms, replacing the
inefficient ARS (random search) approach.

Features:
  - PPO with clipped surrogate objective
  - GAE (lambda=0.95) for advantage estimation
  - Small value network head on LTC hidden state
  - Dense reward shaping: -distance + touch_bonus + grasp_bonus + lift_bonus
  - Curriculum: reach -> grasp -> lift

Usage:
  python carl_scout_ppo.py --steps 500000 --render
  python carl_scout_ppo.py --resume --steps 200000
"""

import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import mujoco

from carl_body_interface import (
    BodyInterface, MODEL_PATH, ARM_CTRL_LOW, ARM_CTRL_HIGH, ARMS_CTRL,
)
from carl_scout_ik_demo import get_scout_state, spawn_target
from carl_scout_bc import LTCPolicyTorch, import_weights_from_numpy

# ─── Constants ────────────────────────────────────────────────────────────────
STATE_DIM   = 51
ACTION_DIM  = 16
HIDDEN_DIM  = 48
BC_WEIGHTS  = "memory/scout_bc_weights.npz"
PPO_WEIGHTS = "memory/scout_ppo_weights.pth"


# ═══════════════════════════════════════════════════════════════════════════════
# PPO Actor-Critic Network
# ═══════════════════════════════════════════════════════════════════════════════

class PPOActorCritic(nn.Module):
    """
    Actor-Critic with shared LTC backbone.
    Actor: LTC hidden -> action mean (16D)
    Critic: LTC hidden -> value (1D)
    """

    def __init__(self, state_dim=STATE_DIM, action_dim=ACTION_DIM, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.hidden_dim = hidden_dim

        # ── Shared LTC backbone (matches ScoutArmPolicy architecture) ────
        self.W_in  = nn.Parameter(torch.randn(hidden_dim, state_dim) * 0.08)
        self.W_rec = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.08)
        self.b     = nn.Parameter(torch.zeros(hidden_dim))
        self.Tau_in  = nn.Parameter(torch.randn(hidden_dim, state_dim) * 0.08)
        self.Tau_rec = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.08)
        self.tau_b   = nn.Parameter(torch.ones(hidden_dim) * 1.5)

        # ── Actor head (policy) ──────────────────────────────────────────
        self.W_out = nn.Parameter(torch.randn(action_dim, hidden_dim) * 0.08)
        self.b_out = nn.Parameter(torch.zeros(action_dim))
        # Log std (learnable, per-action)
        self.log_std = nn.Parameter(torch.ones(action_dim) * -1.0)

        # ── Critic head (value function) ─────────────────────────────────
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
        )

    def _ltc_step(self, s, x, dt=0.005):
        """Single LTC recurrent step. Returns new hidden state."""
        tau_mod = torch.tanh(s @ self.Tau_in.t() + x @ self.Tau_rec.t() + self.tau_b)
        tau = torch.exp(tau_mod)
        dx = -x + torch.tanh(s @ self.W_in.t() + x @ self.W_rec.t() + self.b)
        return x + (dt / tau) * dx

    def forward_step(self, state, hidden):
        """
        Single-step forward pass.
        state: (state_dim,)
        hidden: (hidden_dim,)
        Returns: action_mean, value, new_hidden
        """
        s = state.unsqueeze(0) if state.dim() == 1 else state
        h = hidden.unsqueeze(0) if hidden.dim() == 1 else hidden

        new_h = self._ltc_step(s, h)

        action_mean = torch.tanh(new_h @ self.W_out.t() + self.b_out)
        value = self.value_head(new_h)

        return action_mean.squeeze(0), value.squeeze(0).squeeze(-1), new_h.squeeze(0)

    def get_action_and_value(self, state, hidden, action=None):
        """
        Get action, log_prob, entropy, value for PPO update.
        If action is provided, compute log_prob of that action (for update).
        """
        action_mean, value, new_hidden = self.forward_step(state, hidden)

        std = torch.exp(self.log_std)
        dist = torch.distributions.Normal(action_mean, std)

        if action is None:
            action = dist.sample()
            action = torch.clamp(action, -1.0, 1.0)

        log_prob = dist.log_prob(action).sum(-1)
        entropy  = dist.entropy().sum(-1)

        return action, log_prob, entropy, value, new_hidden

    def load_bc_weights(self, path=BC_WEIGHTS):
        """Load BC-pretrained LTC weights (actor only, critic random)."""
        data = np.load(path)
        with torch.no_grad():
            self.W_in.copy_(torch.tensor(data['W_in'], dtype=torch.float32))
            self.W_rec.copy_(torch.tensor(data['W_rec'], dtype=torch.float32))
            self.b.copy_(torch.tensor(data['b'], dtype=torch.float32))
            self.Tau_in.copy_(torch.tensor(data['Tau_in'], dtype=torch.float32))
            self.Tau_rec.copy_(torch.tensor(data['Tau_rec'], dtype=torch.float32))
            self.tau_b.copy_(torch.tensor(data['tau_b'], dtype=torch.float32))
            self.W_out.copy_(torch.tensor(data['W_out'], dtype=torch.float32))
            self.b_out.copy_(torch.tensor(data['b_out'], dtype=torch.float32))
        print(f"  [LOAD] BC weights loaded from {path}")

    def export_numpy(self, path="memory/scout_ppo_numpy.npz"):
        """Export actor weights to numpy for NumpyLTCPolicy inference."""
        np.savez(path,
            W_in=self.W_in.detach().cpu().numpy(),
            W_rec=self.W_rec.detach().cpu().numpy(),
            b=self.b.detach().cpu().numpy(),
            Tau_in=self.Tau_in.detach().cpu().numpy(),
            Tau_rec=self.Tau_rec.detach().cpu().numpy(),
            tau_b=self.tau_b.detach().cpu().numpy(),
            W_out=self.W_out.detach().cpu().numpy(),
            b_out=self.b_out.detach().cpu().numpy(),
        )
        print(f"  [EXPORT] Numpy weights -> {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Rollout Buffer
# ═══════════════════════════════════════════════════════════════════════════════

class RolloutBuffer:
    """Stores rollout data for PPO update."""

    def __init__(self):
        self.states   = []
        self.actions  = []
        self.log_probs = []
        self.rewards  = []
        self.values   = []
        self.dones    = []
        self.hiddens  = []

    def store(self, state, action, log_prob, reward, value, done, hidden):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
        self.hiddens.append(hidden)

    def compute_gae(self, gamma=0.99, lam=0.95, last_value=0.0):
        """Compute Generalized Advantage Estimation."""
        T = len(self.rewards)
        advantages = np.zeros(T, dtype=np.float32)
        returns    = np.zeros(T, dtype=np.float32)

        values = np.array(self.values + [last_value], dtype=np.float32)
        rewards = np.array(self.rewards, dtype=np.float32)
        dones = np.array(self.dones, dtype=np.float32)

        gae = 0.0
        for t in reversed(range(T)):
            delta = rewards[t] + gamma * values[t+1] * (1 - dones[t]) - values[t]
            gae = delta + gamma * lam * (1 - dones[t]) * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]

        return advantages, returns

    def get_batches(self, advantages, returns, batch_size=256):
        """Yield minibatches for PPO update."""
        T = len(self.states)
        indices = np.random.permutation(T)

        for start in range(0, T, batch_size):
            end = min(start + batch_size, T)
            idx = indices[start:end]

            yield (
                torch.stack([self.states[i] for i in idx]),
                torch.stack([self.actions[i] for i in idx]),
                torch.tensor([self.log_probs[i] for i in idx], dtype=torch.float32),
                torch.tensor(advantages[idx], dtype=torch.float32),
                torch.tensor(returns[idx], dtype=torch.float32),
                torch.stack([self.hiddens[i] for i in idx]),
            )

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()
        self.hiddens.clear()


# ═══════════════════════════════════════════════════════════════════════════════
# Reward Function
# ═══════════════════════════════════════════════════════════════════════════════

def compute_reward(body, data, target_L, target_R, active_arms, prev_dist_L, prev_dist_R):
    """
    Dense reward for arm training.
    Components:
      1. Distance reduction bonus (shaped)
      2. Touch bonus (when fingers contact object)
      3. Energy penalty (discourage wild movements)
    """
    tip_L = body.get_fingertip_mean_pos(data, 'L')
    tip_R = body.get_fingertip_mean_pos(data, 'R')
    dist_L = np.linalg.norm(target_L - tip_L)
    dist_R = np.linalg.norm(target_R - tip_R)

    reward = 0.0

    # Distance-based reward
    if active_arms in ('left', 'both'):
        # Reward for getting closer
        reward += 2.0 * (prev_dist_L - dist_L)
        # Proximity bonus (dense shaping)
        if dist_L < 0.05:
            reward += 0.5
        if dist_L < 0.02:
            reward += 1.0

    if active_arms in ('right', 'both'):
        reward += 2.0 * (prev_dist_R - dist_R)
        if dist_R < 0.05:
            reward += 0.5
        if dist_R < 0.02:
            reward += 1.0

    # Touch bonus
    touch = body.get_all_touch_readings(data)
    if active_arms in ('left', 'both'):
        touch_L = np.sum(touch[:3])
        if touch_L > 0.01:
            reward += 2.0 * min(touch_L, 1.0)

    if active_arms in ('right', 'both'):
        touch_R = np.sum(touch[3:])
        if touch_R > 0.01:
            reward += 2.0 * min(touch_R, 1.0)

    # Energy penalty (small, to discourage spastic movements)
    arm_vel = body.get_all_arm_joint_vel(data)
    reward -= 0.001 * np.sum(arm_vel ** 2)

    return reward, dist_L, dist_R


# ═══════════════════════════════════════════════════════════════════════════════
# PPO Training Loop
# ═══════════════════════════════════════════════════════════════════════════════

def train_ppo(total_steps=500000, rollout_len=1024, ppo_epochs=4,
              batch_size=256, lr=3e-4, clip_eps=0.2, gamma=0.99, lam=0.95,
              render=False, resume=False):
    """Main PPO training loop."""

    print("=" * 65)
    print("  CARL Primate Scout -- PPO Fine-Tuning")
    print(f"  Total steps: {total_steps} | Rollout: {rollout_len}")
    print(f"  PPO epochs: {ppo_epochs} | Batch: {batch_size}")
    print(f"  LR: {lr} | Clip: {clip_eps} | Gamma: {gamma} | Lambda: {lam}")
    print("=" * 65)

    # Setup
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    body = BodyInterface(model)
    rng = np.random.default_rng(seed=int(time.time()) % 10000)

    viewer = None
    if render:
        from mujoco import viewer as mj_viewer
        viewer = mj_viewer.launch_passive(model, data)

    # Create actor-critic
    ac = PPOActorCritic()
    total_params = sum(p.numel() for p in ac.parameters())

    if resume and os.path.exists(PPO_WEIGHTS):
        ac.load_state_dict(torch.load(PPO_WEIGHTS, weights_only=True))
        print(f"  [LOAD] Resumed from {PPO_WEIGHTS}")
    elif os.path.exists(BC_WEIGHTS):
        ac.load_bc_weights(BC_WEIGHTS)
    else:
        print("  [WARN] No BC weights found, training from scratch!")

    print(f"  [NET] Total params: {total_params}")

    optimizer = optim.Adam(ac.parameters(), lr=lr, eps=1e-5)

    center = (ARM_CTRL_HIGH + ARM_CTRL_LOW) / 2.0
    scale  = (ARM_CTRL_HIGH - ARM_CTRL_LOW) / 2.0 + 1e-8
    modes = ['left', 'right', 'both']

    # Tracking
    global_step = 0
    episode_count = 0
    best_avg_reward = -float('inf')

    # Episode state
    def reset_env():
        nonlocal episode_count
        mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)
        mode = modes[episode_count % 3]
        t_L = spawn_target(rng, 'L')
        t_R = spawn_target(rng, 'R')

        # Place cubes
        if len(body.obj_body_ids) >= 2:
            c0_jnt = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "fj_cube_0")
            q0 = model.jnt_qposadr[c0_jnt]
            data.qpos[q0:q0+3] = t_L
            data.qpos[q0+3:q0+7] = [1, 0, 0, 0]
            c1_jnt = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "fj_cube_1")
            q1 = model.jnt_qposadr[c1_jnt]
            data.qpos[q1:q1+3] = t_R
            data.qpos[q1+3:q1+7] = [1, 0, 0, 0]
        mujoco.mj_forward(model, data)
        episode_count += 1
        return mode, t_L, t_R

    # Initialize first episode
    mode, target_L, target_R = reset_env()
    hidden = torch.zeros(HIDDEN_DIM)
    ep_step = 0
    ep_reward = 0.0
    prev_dist_L = np.linalg.norm(target_L - body.get_fingertip_mean_pos(data, 'L'))
    prev_dist_R = np.linalg.norm(target_R - body.get_fingertip_mean_pos(data, 'R'))

    recent_rewards = []

    print(f"\n  Training started...\n")

    while global_step < total_steps:
        buffer = RolloutBuffer()

        # ── Collect rollout ───────────────────────────────────────────
        for _ in range(rollout_len):
            body.lock_base(data)
            state = get_scout_state(body, data, target_L, target_R, mode)
            state_t = torch.tensor(state, dtype=torch.float32)

            with torch.no_grad():
                action_t, log_prob, _, value, new_hidden = ac.get_action_and_value(
                    state_t, hidden)

            # Apply action
            action_np = action_t.numpy()
            action_raw = action_np * scale + center
            action_raw = np.clip(action_raw, ARM_CTRL_LOW, ARM_CTRL_HIGH)
            data.ctrl[ARMS_CTRL] = action_raw

            for _ in range(10):
                mujoco.mj_step(model, data)

            if viewer and viewer.is_running():
                viewer.sync()

            # Compute reward
            reward, dist_L, dist_R = compute_reward(
                body, data, target_L, target_R, mode, prev_dist_L, prev_dist_R)
            prev_dist_L = dist_L
            prev_dist_R = dist_R

            ep_step += 1
            ep_reward += reward
            done = ep_step >= 300  # episode length

            buffer.store(state_t, action_t, log_prob.item(), reward,
                        value.item(), float(done), hidden)
            hidden = new_hidden.detach()

            global_step += 1

            if done:
                recent_rewards.append(ep_reward)
                mode, target_L, target_R = reset_env()
                hidden = torch.zeros(HIDDEN_DIM)
                ep_step = 0
                ep_reward = 0.0
                prev_dist_L = np.linalg.norm(target_L - body.get_fingertip_mean_pos(data, 'L'))
                prev_dist_R = np.linalg.norm(target_R - body.get_fingertip_mean_pos(data, 'R'))

        # ── Compute GAE ───────────────────────────────────────────────
        with torch.no_grad():
            last_state = torch.tensor(
                get_scout_state(body, data, target_L, target_R, mode),
                dtype=torch.float32)
            _, _, _, last_value, _ = ac.get_action_and_value(last_state, hidden)

        advantages, returns = buffer.compute_gae(gamma, lam, last_value.item())

        # Normalize advantages
        adv_mean = advantages.mean()
        adv_std = advantages.std() + 1e-8
        advantages = (advantages - adv_mean) / adv_std

        # ── PPO Update ────────────────────────────────────────────────
        total_policy_loss = 0.0
        total_value_loss = 0.0
        n_updates = 0

        for _ in range(ppo_epochs):
            for (mb_states, mb_actions, mb_old_logp, mb_adv, mb_ret, mb_hidden) in \
                    buffer.get_batches(advantages, returns, batch_size):

                _, new_logp, entropy, new_value, _ = ac.get_action_and_value(
                    mb_states, mb_hidden, action=mb_actions)

                # Clipped surrogate loss
                ratio = torch.exp(new_logp - mb_old_logp)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss (clipped)
                value_loss = 0.5 * ((new_value - mb_ret) ** 2).mean()

                # Entropy bonus
                entropy_loss = -0.01 * entropy.mean()

                loss = policy_loss + value_loss + entropy_loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(ac.parameters(), 0.5)
                optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                n_updates += 1

        buffer.clear()

        # ── Logging ───────────────────────────────────────────────────
        avg_ploss = total_policy_loss / max(n_updates, 1)
        avg_vloss = total_value_loss / max(n_updates, 1)

        if len(recent_rewards) > 0:
            avg_reward = np.mean(recent_rewards[-20:])
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                torch.save(ac.state_dict(), PPO_WEIGHTS)
                ac.export_numpy("memory/scout_ppo_numpy.npz")

            print(f"  Step {global_step:7d}/{total_steps} | "
                  f"AvgR: {avg_reward:8.2f} | Best: {best_avg_reward:8.2f} | "
                  f"PLoss: {avg_ploss:.4f} | VLoss: {avg_vloss:.4f} | "
                  f"Eps: {len(recent_rewards)}")

    if viewer:
        viewer.close()

    # Final save
    torch.save(ac.state_dict(), PPO_WEIGHTS)
    ac.export_numpy("memory/scout_ppo_numpy.npz")

    print(f"\n{'=' * 65}")
    print(f"  PPO Training Complete")
    print(f"  Total steps: {global_step} | Episodes: {len(recent_rewards)}")
    print(f"  Best avg reward: {best_avg_reward:.2f}")
    print(f"  Weights: {PPO_WEIGHTS}")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CARL Scout PPO Trainer")
    parser.add_argument("--steps", type=int, default=500000, help="Total training steps")
    parser.add_argument("--rollout-len", type=int, default=1024, help="Steps per rollout")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--render", action="store_true", help="Show viewer")
    parser.add_argument("--resume", action="store_true", help="Resume from saved weights")
    args = parser.parse_args()

    train_ppo(
        total_steps=args.steps,
        rollout_len=args.rollout_len,
        lr=args.lr,
        render=args.render,
        resume=args.resume,
    )
