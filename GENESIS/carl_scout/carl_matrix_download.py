import os
import glob
import numpy as np
import time
from carl_agent import CarlBrain

def main():
    print("======================================================")
    print("  CARL GENESIS: BEHAVIORAL CLONING (MATRIX DOWNLOAD)  ")
    print("======================================================")

    # 1. Load the empty/random brain
    brain = CarlBrain(n_obs=34)
    # We load whatever state it's in (possibly the random wobbling state we just left)
    brain.load("memory/carl_brain")

    # Set learning rate higher for offline supervised learning
    actor = brain.actor
    actor.l1.lr = 1e-3
    actor.l2.lr = 1e-3
    actor.l3.lr = 1e-3
    actor.mu_head.lr = 1e-3

    # 2. Gather Harvest Dataset
    data_dir = "harvest_dataset/data/chunk-000/"
    files = glob.glob(os.path.join(data_dir, "*.npz"))
    if not files:
        print("[ERROR] No harvest dataset found! Cannot initiate Matrix Download.")
        return

    print(f"[SYSTEM] Found {len(files)} trajectory episodes. Assembling memory matrix...")
    
    all_obs = []
    all_acts = []
    for f in files:
        data = np.load(f)
        all_obs.append(data['observations'])
        all_acts.append(data['actions'])
    
    obs_batch = np.concatenate(all_obs, axis=0)
    act_batch = np.concatenate(all_acts, axis=0)
    num_samples = len(obs_batch)
    print(f"[SYSTEM] Golden Dataset constructed: {num_samples} samples.")

    # SUBSAMPLE FOR SPEED: We will train on a random 15,000 memory slice
    # so we can do 'Deep Training' without waiting 2 hours.
    subsample_idx = np.random.choice(num_samples, size=min(15000, num_samples), replace=False)
    obs_batch = obs_batch[subsample_idx]
    act_batch = act_batch[subsample_idx]
    num_samples = len(obs_batch)
    print(f"[SYSTEM] Deep Training Slice: {num_samples} samples.")

    # 3. Supervised Training Loop
    epochs = 10
    indices = np.arange(num_samples)

    print("\n[CORTEX] Initiating rapid synaptic override (Stochastic Gradient Descent)...")
    start_time = time.time()
    
    # ---------------------------------------------------------
    # CRITICAL FIX: The observations in the dataset are RAW.
    # The brain expects NORMALIZED observations. If we don't normalize,
    # the network gets confused by the raw scale and gradient plateaus.
    # Because we previously harvested a "blind" dataset, the brain's 
    # stored stats are completely wrong for the new sighted data.
    # We MUST recalculate the true mean and variance here!
    # ---------------------------------------------------------
    
    # Calculate true mean and variance from the new dataset
    true_mean = np.mean(obs_batch, axis=0)
    true_var_raw = np.var(obs_batch, axis=0)
    
    # Update the Brain's internal statistics so inference uses these!
    brain._obs_mean = true_mean
    brain._obs_n = num_samples
    # CarlBrain expects _obs_var to be M2 (sum of squared diffs), which is var * n
    brain._obs_var = true_var_raw * num_samples
    
    # Now normalize the batch
    obs_std = np.sqrt(true_var_raw + 1e-4)
    obs_batch_norm = (obs_batch - true_mean) / obs_std
    
    # Temporarily boost learning rate for behavioral cloning
    brain.actor.l1.lr = 1e-3
    brain.actor.l2.lr = 1e-3
    brain.actor.l3.lr = 1e-3
    brain.actor.mu_head.lr = 1e-3

    for epoch in range(epochs):
        np.random.shuffle(indices)
        epoch_loss = 0.0

        for i in range(num_samples):
            idx = indices[i]
            obs = obs_batch_norm[idx]
            target = act_batch[idx]

            # Forward pass
            mu = brain.actor.forward(obs)
            
            # Loss
            loss = np.sum((mu - target)**2)
            epoch_loss += loss

            # Backward pass (gradient of MSE loss wrt mu is mu - target)
            grad = (mu - target)

            # Backprop through the actor
            dW_mu, db_mu, dh3 = brain.actor.mu_head.backward(grad)
            dW3, db3, dh2 = brain.actor.l3.backward(dh3)
            dW2, db2, dh1 = brain.actor.l2.backward(dh2)
            dW1, db1, _ = brain.actor.l1.backward(dh1)

            # Update weights
            brain.actor.mu_head.update(dW_mu, db_mu)
            brain.actor.l3.update(dW3, db3)
            brain.actor.l2.update(dW2, db2)
            brain.actor.l1.update(dW1, db1)

        avg_loss = epoch_loss / num_samples
        print(f"  > Epoch {epoch+1:02d}/{epochs} | MSE Loss: {avg_loss:.6f}")

    elapsed = time.time() - start_time
    print(f"\n[SYSTEM] Matrix Download Complete in {elapsed:.2f} seconds.")
    
    # Save the smart brain!
    brain.save("memory/carl_brain")
    print("[SYSTEM] Smart Brain safely written to 'memory/carl_brain'.")
    print("[SYSTEM] You may now restart carl_harvest.py. CARL is ready.")

if __name__ == "__main__":
    main()
