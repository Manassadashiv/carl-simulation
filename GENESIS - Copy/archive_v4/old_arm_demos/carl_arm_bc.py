"""
carl_arm_bc.py — CARL Behavioral Cloning (BC) Demonstrator & Supervised Trainer (PyTorch Vectorized version)

Features:
  1. Heuristic Auto-Demonstrator:
     Uses MuJoCo's analytical Jacobian (mj_jacSite) to solve Inverse Kinematics (IK)
     and generate clean, noise-free reach trajectories toward target objects.
  2. Supervised Learning via PyTorch:
     Optimizes the LTC network (ArmPolicy) parameters directly on the demonstration
     dataset by minimizing the Mean Squared Error (MSE) loss with Adam.
"""

import os
import argparse
import numpy as np
import mujoco
import torch
import torch.nn as nn
import torch.optim as optim
from carl_arm_train import ArmPolicy, build_cache, get_state, compute_reward, reset_episode, ARM_CTRL_LOW, ARM_CTRL_HIGH, ARM_CTRL_SLICE

class LTCPolicy(nn.Module):
    def __init__(self, state_dim=29, action_dim=10, hidden_dim=40):
        super().__init__()
        self.W_in = nn.Parameter(torch.randn(hidden_dim, state_dim) * 0.1)
        self.W_rec = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.1)
        self.b = nn.Parameter(torch.zeros(hidden_dim))
        
        self.Tau_in = nn.Parameter(torch.randn(hidden_dim, state_dim) * 0.1)
        self.Tau_rec = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.1)
        self.tau_b = nn.Parameter(torch.ones(hidden_dim) * 1.5)
        
        self.W_out = nn.Parameter(torch.randn(action_dim, hidden_dim) * 0.1)
        self.b_out = nn.Parameter(torch.zeros(action_dim))
        
        self.hidden_dim = hidden_dim

    def forward(self, x_padded, lengths, max_len, dt=0.01):
        batch_size = x_padded.shape[0]
        preds = torch.zeros(batch_size, max_len, self.W_out.shape[0])
        h = torch.zeros(batch_size, self.hidden_dim)
        
        for t in range(max_len):
            tau_mod = torch.tanh(torch.matmul(x_padded[:, t], self.Tau_in.t()) + torch.matmul(h, self.Tau_rec.t()) + self.tau_b)
            tau = torch.exp(tau_mod)
            
            dx = -h + torch.tanh(torch.matmul(x_padded[:, t], self.W_in.t()) + torch.matmul(h, self.W_rec.t()) + self.b)
            h = h + (dt / tau) * dx
            
            preds[:, t] = torch.tanh(torch.matmul(h, self.W_out.t()) + self.b_out)
            
        return preds

def get_ik_action(model, data, cache, target_pos):
    """
    Solve 1-step Inverse Kinematics using numerical site Jacobian and Damped Least Squares (DLS).
    Returns the target control vector in range [-1, 1].
    """
    # 1. Compute site Jacobian
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    tip_sid = cache.get('tip_L_sid')
    if tip_sid is None:
        tip_sid = cache['touch_L']
    mujoco.mj_jacSite(model, data, jacp, jacr, tip_sid)
    
    # Slice Jacobian for the 5 left arm joints (now 5-DOF!)
    left_arm_dofs = cache['arm_dof'][:5]
    J = jacp[:, left_arm_dofs]
    
    # 2. Compute error vector to target
    tip_pos = data.site_xpos[tip_sid]
    dx = target_pos - tip_pos
    
    # 3. Damped Least Squares (DLS) Inverse
    lambda_sq = 0.015
    J_dls = J.T @ np.linalg.inv(J @ J.T + lambda_sq * np.eye(3))
    dq = J_dls @ dx
    
    # Scale dq to prevent large jumps (tuned for high kp = 350 actuators)
    step_scale = 0.15
    dq = np.clip(dq * step_scale, -0.08, 0.08)
    
    # 4. Update command target in joint radians
    current_q = np.array([data.qpos[a] for a in cache['arm_qpos'][:5]])
    cmd_q = current_q + dq
    
    # Clip to physical joint limits
    cmd_q = np.clip(cmd_q, ARM_CTRL_LOW[:5], ARM_CTRL_HIGH[:5])
    
    # 5. Mirror right arm with default center pose
    full_cmd = np.zeros(10)
    full_cmd[:5] = cmd_q
    full_cmd[5:] = (ARM_CTRL_HIGH[5:] + ARM_CTRL_LOW[5:]) / 2.0
    
    # 6. Normalize to [-1, 1] range for network output
    center = (ARM_CTRL_HIGH + ARM_CTRL_LOW) / 2.0
    scale = (ARM_CTRL_HIGH - ARM_CTRL_LOW) / 2.0 + 1e-8
    action_norm = (full_cmd - center) / scale
    
    return np.clip(action_norm, -1.0, 1.0)


def generate_demonstrations(num_episodes=50, max_steps=150):
    """Generate and save successful reach trajectories using the IK controller."""
    print(f"[DEMO] Generating {num_episodes} successful reach demonstrations...")
    model = mujoco.MjModel.from_xml_path("vessel_kinetic.xml")
    data = mujoco.MjData(model)
    cache = build_cache(model)
    rng = np.random.default_rng(seed=42)
    
    states_list = []
    actions_list = []
    lengths_list = []
    
    successes = 0
    while successes < num_episodes:
        reset_episode(model, data, cache, "reach", rng)
        
        ep_states = []
        ep_actions = []
        
        success = False
        for _ in range(max_steps):
            # Lock base position and velocity to prevent floor friction sliding
            data.qpos[cache['Q_CARL'] : cache['Q_CARL']+3] = [0.0, 0.0, 0.04]
            data.qpos[cache['Q_CARL']+3 : cache['Q_CARL']+7] = [1.0, 0.0, 0.0, 0.0]
            data.qvel[cache['V_CARL'] : cache['V_CARL']+6] = 0.0
            
            # Target object position
            obj_pos = data.xpos[cache['obj_bid']].copy()
            
            # Get current observation state
            s = get_state(model, data, cache)
            
            # Solve Inverse Kinematics command
            act = get_ik_action(model, data, cache, obj_pos)
            
            # Apply to environment
            # Map [-1, 1] target to actual control values
            center = (ARM_CTRL_HIGH + ARM_CTRL_LOW) / 2.0
            scale = (ARM_CTRL_HIGH - ARM_CTRL_LOW) / 2.0
            act_physical = act * scale + center
            
            data.ctrl[:2] = 0.0 # freeze wheels
            data.ctrl[ARM_CTRL_SLICE] = act_physical
            mujoco.mj_step(model, data)
            
            # Check reward
            _, done, dist = compute_reward(data, cache, "reach")
            
            # Record state and action
            ep_states.append(s)
            ep_actions.append(act)
            
            if done and dist < 0.045:
                success = True
                break
                
        if success:
            states_list.extend(ep_states)
            actions_list.extend(ep_actions)
            lengths_list.append(len(ep_states))
            successes += 1
            if successes % 10 == 0:
                print(f"  [Progress] {successes}/{num_episodes} episodes recorded.")
                
    # Save dataset
    states_arr = np.array(states_list, dtype=np.float32)
    actions_arr = np.array(actions_list, dtype=np.float32)
    lengths_arr = np.array(lengths_list, dtype=np.int32)
    
    os.makedirs("memory", exist_ok=True)
    np.savez("memory/demo_dataset.npz", states=states_arr, actions=actions_arr, lengths=lengths_arr)
    print(f"[DEMO] Saved {len(states_arr)} transition samples across {len(lengths_arr)} episodes to memory/demo_dataset.npz")


def train_cloning(episodes=1000, load_path=None):
    """
    Supervised Behavioral Cloning: uses PyTorch to train the LTC network
    parameters directly on the demonstration dataset.
    """
    dataset_path = "memory/demo_dataset.npz"
    if not os.path.exists(dataset_path):
        print(f"[ERROR] Demonstration dataset not found at {dataset_path}. Generate demos first.")
        return
        
    data_npz = np.load(dataset_path)
    if 'lengths' not in data_npz:
        print("[WARNING] Old dataset format detected. Re-generating demonstrations with episode boundaries...")
        generate_demonstrations()
        data_npz = np.load(dataset_path)
        
    X_flat = data_npz['states']
    Y_flat = data_npz['actions']
    lengths = data_npz['lengths']
    
    num_episodes = len(lengths)
    print(f"[TRAIN] Loaded {num_episodes} episodes with {len(X_flat)} total steps.")
    
    # Pad episodes to max length
    max_len = int(np.max(lengths))
    batch_x = np.zeros((num_episodes, max_len, 29), dtype=np.float32)
    batch_y = np.zeros((num_episodes, max_len, 10), dtype=np.float32)
    
    start_idx = 0
    for i, length in enumerate(lengths):
        end_idx = start_idx + length
        batch_x[i, :length] = X_flat[start_idx:end_idx]
        batch_y[i, :length] = Y_flat[start_idx:end_idx]
        start_idx = end_idx
        
    # Convert to PyTorch tensors
    x_padded = torch.tensor(batch_x)
    y_padded = torch.tensor(batch_y)
    lengths_t = torch.tensor(lengths)
    
    # Define PyTorch model
    model = LTCPolicy()
    
    # Try to load existing weights if load_path is provided
    if load_path and os.path.exists(load_path):
        try:
            saved = np.load(load_path)
            w = saved['weights']
            # Reconstruct model state dict from flat array
            # Order: W_in, W_rec, b, Tau_in, Tau_rec, tau_b, W_out, b_out
            idx = 0
            for name, param in [
                ('W_in', model.W_in), ('W_rec', model.W_rec), ('b', model.b),
                ('Tau_in', model.Tau_in), ('Tau_rec', model.Tau_rec), ('tau_b', model.tau_b),
                ('W_out', model.W_out), ('b_out', model.b_out)
            ]:
                sz = param.numel()
                param.data.copy_(torch.tensor(w[idx:idx+sz].reshape(param.shape)))
                idx += sz
            print(f"[LOAD] Loaded starting weights from {load_path}")
        except Exception as e:
            print(f"[LOAD] Starting fresh ({e})")
            
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    
    # Create mask for loss computation
    mask = torch.arange(max_len).expand(num_episodes, max_len) < lengths_t.unsqueeze(1)
    
    print(f"[TRAIN] Running {episodes} supervised cloning epochs via PyTorch...")
    best_loss = 1e9
    
    for epoch in range(episodes):
        model.train()
        optimizer.zero_grad()
        
        # Batch forward pass
        preds = model(x_padded, lengths_t, max_len)
        
        # Loss calculation
        loss = torch.mean(((preds - y_padded) ** 2)[mask])
        
        loss.backward()
        optimizer.step()
        
        current_loss = loss.item()
        if current_loss < best_loss:
            best_loss = current_loss
            # Save weights to numpy array
            theta = np.concatenate([
                model.W_in.detach().numpy().ravel(),
                model.W_rec.detach().numpy().ravel(),
                model.b.detach().numpy().ravel(),
                model.Tau_in.detach().numpy().ravel(),
                model.Tau_rec.detach().numpy().ravel(),
                model.tau_b.detach().numpy().ravel(),
                model.W_out.detach().numpy().ravel(),
                model.b_out.detach().numpy().ravel()
            ])
            np.savez("memory/carl_arm_weights.npz", weights=theta)
            
        if (epoch + 1) % 100 == 0 or epoch == 0 or epoch == episodes - 1:
            print(f"Epoch {epoch+1:04d}/{episodes} | MSE Loss: {current_loss:.6f} | Best Loss: {best_loss:.6f}")
            
    print(f"[DONE] PyTorch cloning complete. Best MSE Loss: {best_loss:.6f}. Saved to memory/carl_arm_weights.npz")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="CARL Arm Behavioral Cloning")
    ap.add_argument("--record", action="store_true", help="Generate auto-demonstrations")
    ap.add_argument("--train", action="store_true", help="Train BC network")
    ap.add_argument("--episodes", type=int, default=500)
    args = ap.parse_args()
    
    if args.record:
        generate_demonstrations()
    elif args.train:
        train_cloning(episodes=args.episodes)
    else:
        ap.print_help()
