"""
carl_imagination.py — The Cerebral Cortex Spatiotemporal Predictor.

Architecture:
- SpatiotemporalPredictor:
  - Dual-Head Recurrent Elman Network with Softplus Variance Head in pure-NumPy.
  - Encoder: 10-step Elman RNN processing O_{t-10:t} (33-D sensory vectors) to shape (10, 32).
  - Attention: Softplus-based linear self-attention pooling of the encoder sequence.
  - Decoder: 5-step autoregressive rollout predicting future observations O_{t+1:t+5} and variances.
  - Analytical Backpropagation: Exact matrix derivatives for Mean Squared Error (MSE) mean
    and variance head predictions, allowing training updates at 15Hz with zero external frameworks.
  - Inference speed: <1.0ms, avoiding thread-blocking or priority inversions.

- ImaginationThread:
  - Runs at 15Hz to drain transitions from a thread-safe queue.
  - Computes surprise Delta_t and certainty C_t, updating the thalamic double buffer.
  - Performs online training updates on the predictor network using actual transitions.
"""

import numpy as np
import threading
import time
import queue

class SpatiotemporalPredictor:
    def __init__(self, obs_dim=33, act_dim=2, hidden_dim=32, steps_future=5, steps_history=10, lr=1e-3):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim
        self.steps_future = steps_future
        self.steps_history = steps_history
        self.lr = lr
        
        # Parameter storage
        self.params = {}
        
        # Xavier (He) Normal Initialization
        def init_w(n_out, n_in):
            return (np.random.randn(n_out, n_in).astype(np.float32) * np.sqrt(2.0 / n_in))
            
        def init_b(n_out):
            return np.zeros(n_out, dtype=np.float32)
            
        # Encoder (Elman RNN)
        self.params['W_in'] = init_w(hidden_dim, obs_dim)
        self.params['W_rec'] = init_w(hidden_dim, hidden_dim)
        self.params['b_h'] = init_b(hidden_dim)
        
        # Linear Attention
        self.params['W_q'] = init_w(hidden_dim, hidden_dim)
        self.params['b_q'] = init_b(hidden_dim)
        self.params['W_k'] = init_w(hidden_dim, hidden_dim)
        self.params['b_k'] = init_b(hidden_dim)
        self.params['W_v'] = init_w(hidden_dim, hidden_dim)
        self.params['b_v'] = init_b(hidden_dim)
        
        # Decoder RNN
        self.params['W_dec_in'] = init_w(hidden_dim, obs_dim + act_dim)
        self.params['W_dec_rec'] = init_w(hidden_dim, hidden_dim)
        self.params['W_ctx'] = init_w(hidden_dim, hidden_dim)
        self.params['b_dec'] = init_b(hidden_dim)
        
        # Mean Head
        self.params['W_mean'] = init_w(obs_dim, hidden_dim)
        self.params['b_mean'] = init_b(obs_dim)
        
        # Variance Head
        self.params['W_var'] = init_w(obs_dim, hidden_dim)
        self.params['b_var'] = init_b(obs_dim)
        
        # Adam Optimizer Moments
        self.m = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.t = 0
        
        self.lock = threading.RLock()
        self.cache = {}

    def forward(self, O_history, actions):
        self.lock.acquire()
        try:
            return self._forward_impl(O_history, actions)
        finally:
            self.lock.release()

    def _forward_impl(self, O_history, actions):
        """
        Evaluate future observations and variances (supports batch and single inputs).
        O_history: (10, 33) or (K, 10, 33)
        actions: (5, 2) or (K, 5, 2)
        """
        is_single = (O_history.ndim == 2)
        if is_single:
            O_hist = O_history[None, :, :]
            acts = actions[None, :, :]
        else:
            O_hist = O_history
            acts = actions
            
        K = O_hist.shape[0]
        
        # 1. Encoder RNN
        h = np.zeros((K, 11, self.hidden_dim), dtype=np.float32)
        W_in_T = self.params['W_in'].T
        W_rec_T = self.params['W_rec'].T
        b_h = self.params['b_h']
        
        for i in range(10):
            h[:, i+1] = np.tanh(np.dot(O_hist[:, i], W_in_T) + np.dot(h[:, i], W_rec_T) + b_h)
            
        # 2. Linear Attention over encoder steps
        q = np.dot(h[:, 10], self.params['W_q'].T) + self.params['b_q']  # (K, 32)
        k = np.dot(h[:, 1:], self.params['W_k'].T) + self.params['b_k']    # (K, 10, 32)
        v = np.dot(h[:, 1:], self.params['W_v'].T) + self.params['b_v']    # (K, 10, 32)
        
        qk = np.log1p(np.exp(np.clip(q, -20.0, 20.0)))  # (K, 32)
        kk = np.log1p(np.exp(np.clip(k, -20.0, 20.0)))  # (K, 10, 32)
        
        # scores: (K, 10)
        scores = np.sum(kk * qk[:, None, :], axis=2)
        denom = np.sum(scores, axis=1, keepdims=True) + 1e-8
        w = scores / denom  # (K, 10)
        
        c = np.sum(w[:, :, None] * v, axis=1)  # Context vector (K, 32)
        
        # 3. Decoder Autoregressive Rollout
        s = np.zeros((K, 6, self.hidden_dim), dtype=np.float32)
        s[:, 0] = c
        
        O_pred_mean = np.zeros((K, 5, self.obs_dim), dtype=np.float32)
        O_pred_var = np.zeros((K, 5, self.obs_dim), dtype=np.float32)
        
        W_dec_in_T = self.params['W_dec_in'].T
        W_dec_rec_T = self.params['W_dec_rec'].T
        W_ctx_T = self.params['W_ctx'].T
        b_dec = self.params['b_dec']
        W_mean_T = self.params['W_mean'].T
        b_mean = self.params['b_mean']
        W_var_T = self.params['W_var'].T
        b_var = self.params['b_var']
        
        current_O = O_hist[:, -1].copy()
        for t in range(5):
            u = np.concatenate([current_O, acts[:, t]], axis=1)
            s[:, t+1] = np.tanh(np.dot(u, W_dec_in_T) + 
                                 np.dot(s[:, t], W_dec_rec_T) + 
                                 np.dot(c, W_ctx_T) + 
                                 b_dec)
            
            O_pred_mean[:, t] = np.dot(s[:, t+1], W_mean_T) + b_mean
            O_pred_var_raw = np.dot(s[:, t+1], W_var_T) + b_var
            O_pred_var[:, t] = np.log1p(np.exp(np.clip(O_pred_var_raw, -20.0, 20.0)))
            
            current_O = O_pred_mean[:, t].copy()
            
        if is_single:
            # Re-cache single queries for backprop compatibility
            self.cache = {
                'O_history': O_history,
                'actions': actions,
                'h': h[0],
                'q': q[0],
                'k': k[0],
                'v': v[0],
                'qk': qk[0],
                'kk': kk[0],
                'scores': scores[0],
                'denom': denom[0, 0],
                'w': w[0],
                'c': c[0],
                's': s[0],
                'O_pred_mean': O_pred_mean[0],
                'O_pred_var': O_pred_var[0]
            }
            return O_pred_mean[0], O_pred_var[0]
        else:
            return O_pred_mean, O_pred_var

    def train_step(self, O_history, actions, O_future, lambda_var=0.5):
        self.lock.acquire()
        try:
            return self._train_step_impl(O_history, actions, O_future, lambda_var)
        finally:
            self.lock.release()

    def _train_step_impl(self, O_history, actions, O_future, lambda_var=0.5):
        """
        Updates weights using analytical gradients w.r.t mean and variance error.
        """
        # 1. Forward
        O_pred_mean, O_pred_var = self.forward(O_history, actions)
        
        # 2. Loss and Head gradients
        mean_err = O_pred_mean - O_future
        var_err = O_pred_var - (mean_err ** 2)
        
        loss_mean = np.mean(mean_err ** 2)
        loss_var = np.mean(var_err ** 2)
        total_loss = loss_mean + lambda_var * loss_var
        
        N = float(self.steps_future * self.obs_dim)
        dO_pred_var = 2.0 * lambda_var * var_err / N
        dO_pred_mean = 2.0 * mean_err / N - 4.0 * lambda_var * var_err * mean_err / N
        
        # 3. Backward Pass
        h = self.cache['h']
        q = self.cache['q']
        k = self.cache['k']
        v = self.cache['v']
        qk = self.cache['qk']
        kk = self.cache['kk']
        w = self.cache['w']
        c = self.cache['c']
        s = self.cache['s']
        denom = self.cache['denom']
        
        grads = {k: np.zeros_like(v) for k, v in self.params.items()}
        
        ds = np.zeros(self.hidden_dim, dtype=np.float32)
        dc = np.zeros(self.hidden_dim, dtype=np.float32)
        d_current_O = np.zeros(self.obs_dim, dtype=np.float32)
        
        # Backprop through Decoder stages
        for t in reversed(range(5)):
            dmean = dO_pred_mean[t] + d_current_O
            grads['W_mean'] += np.outer(dmean, s[t+1])
            grads['b_mean'] += dmean
            ds_from_mean = np.dot(self.params['W_mean'].T, dmean)
            
            dvar = dO_pred_var[t]
            O_pred_var_raw = np.dot(self.params['W_var'], s[t+1]) + self.params['b_var']
            sigmoid_raw = 1.0 / (1.0 + np.exp(-np.clip(O_pred_var_raw, -20.0, 20.0)))
            dvar_raw = dvar * sigmoid_raw
            grads['W_var'] += np.outer(dvar_raw, s[t+1])
            grads['b_var'] += dvar_raw
            ds_from_var = np.dot(self.params['W_var'].T, dvar_raw)
            
            ds_total = ds + ds_from_mean + ds_from_var
            dz_dec = ds_total * (1.0 - s[t+1]**2)
            
            prev_O = O_history[-1] if t == 0 else O_pred_mean[t-1]
            u = np.concatenate([prev_O, actions[t]])
            
            grads['W_dec_in'] += np.outer(dz_dec, u)
            grads['W_dec_rec'] += np.outer(dz_dec, s[t])
            grads['W_ctx'] += np.outer(dz_dec, c)
            grads['b_dec'] += dz_dec
            
            du = np.dot(self.params['W_dec_in'].T, dz_dec)
            d_current_O = du[:self.obs_dim]
            ds = np.dot(self.params['W_dec_rec'].T, dz_dec)
            dc += np.dot(self.params['W_ctx'].T, dz_dec)
            
        dc += ds  # decoder state s[0] was set to context c
        
        # Backprop through Attention Block
        # c = sum_i w[i] * v[i]
        dv = w[:, None] * dc[None, :]
        dw = np.dot(v, dc)
        
        # w_i = scores_i / denom
        dscores = (dw - np.dot(dw, w)) / denom
        
        # scores_i = dot(kk_i, qk)
        dqk = np.dot(dscores, kk)
        dkk = dscores[:, None] * qk[None, :]
        
        # Softplus mappings
        sigmoid_q = 1.0 / (1.0 + np.exp(-np.clip(q, -20.0, 20.0)))
        dq = dqk * sigmoid_q
        
        sigmoid_k = 1.0 / (1.0 + np.exp(-np.clip(k, -20.0, 20.0)))
        dk = dkk * sigmoid_k
        
        grads['W_q'] += np.outer(dq, h[10])
        grads['b_q'] += dq
        
        for i in range(10):
            grads['W_k'] += np.outer(dk[i], h[i+1])
            grads['b_k'] += dk[i]
            grads['W_v'] += np.outer(dv[i], h[i+1])
            grads['b_v'] += dv[i]
            
        dh_enc = np.zeros((11, self.hidden_dim), dtype=np.float32)
        dh_enc[10] += np.dot(self.params['W_q'].T, dq)
        for i in range(10):
            dh_enc[i+1] += np.dot(self.params['W_k'].T, dk[i]) + np.dot(self.params['W_v'].T, dv[i])
            
        # Backprop through Encoder RNN
        for i in reversed(range(10)):
            dz_enc = dh_enc[i+1] * (1.0 - h[i+1]**2)
            grads['W_in'] += np.outer(dz_enc, O_history[i])
            grads['W_rec'] += np.outer(dz_enc, h[i])
            grads['b_h'] += dz_enc
            dh_enc[i] += np.dot(self.params['W_rec'].T, dz_enc)
            
        # 4. Adam Updates
        self.update(grads, self.lr)
        return total_loss

    def update(self, grads, lr):
        self.t += 1
        beta1, beta2 = 0.9, 0.999
        eps = 1e-8
        for k in self.params:
            self.m[k] = beta1 * self.m[k] + (1.0 - beta1) * grads[k]
            self.v[k] = beta2 * self.v[k] + (1.0 - beta2) * (grads[k] ** 2)
            m_hat = self.m[k] / (1.0 - beta1 ** self.t)
            v_hat = self.v[k] / (1.0 - beta2 ** self.t)
            self.params[k] -= lr * m_hat / (np.sqrt(v_hat) + eps)


class ImaginationThread(threading.Thread):
    def __init__(self, shared_sensor_buffer, transition_queue, obs_dim=33, act_dim=2):
        super().__init__(name="ImaginationLobe", daemon=True)
        self.buffer = shared_sensor_buffer
        self.queue = transition_queue
        self.predictor = SpatiotemporalPredictor(obs_dim=obs_dim, act_dim=act_dim)
        
        self._running = False
        self.transitions = []
        self.max_transitions = 100
        
        # State tracking for surprise
        self.next_step_pred = np.zeros(obs_dim, dtype=np.float32)
        self.certainty = 1.0
        self.surprise_ema = 0.0

    def run(self):
        self._running = True
        print("[IMAGINATION] 15Hz background thread started.")
        period = 1.0 / 15.0  # 66.67ms
        
        while self._running:
            t_start = time.perf_counter()
            
            # Drain queue of transitions: (obs, action)
            new_transitions = []
            while not self.queue.empty():
                try:
                    new_transitions.append(self.queue.get_nowait())
                except queue.Empty:
                    break
            
            for trans in new_transitions:
                obs, action = trans
                # 1. Compute Surprise Delta_t compared to last step's prediction
                # (Ignore early stages where prediction is unpopulated)
                if np.any(self.next_step_pred):
                    err = np.mean((obs - self.next_step_pred) ** 2)
                    self.surprise_ema = 0.92 * self.surprise_ema + 0.08 * err
                    # Certainty maps surprise to exp decay
                    curr_certainty = np.exp(-8.0 * self.surprise_ema)
                    self.certainty = 0.95 * self.certainty + 0.05 * curr_certainty
                else:
                    self.surprise_ema = 0.0
                    self.certainty = 1.0
                
                # Write surprise/certainty to double buffer
                self.buffer.write_prediction_error(np.array([self.surprise_ema, self.certainty]))
                
                # Record to local history list
                self.transitions.append((obs.copy(), action.copy()))
                
            # Keep history bound
            if len(self.transitions) > self.max_transitions:
                self.transitions = self.transitions[-self.max_transitions:]
                
            # 2. Run online training step and future rollout prediction
            if len(self.transitions) >= 15:
                # O_history: transitions[-15:-5] (obs) -> shape (10, 33)
                # actions: transitions[-6:-1] (action) -> shape (5, 2)
                # O_future: transitions[-5:] (obs) -> shape (5, 33)
                hist_obs = np.array([t[0] for t in self.transitions[-15:-5]], dtype=np.float32)
                hist_acts = np.array([t[1] for t in self.transitions[-6:-1]], dtype=np.float32)
                fut_obs = np.array([t[0] for t in self.transitions[-5:]], dtype=np.float32)
                
                # Run optimization step
                self.predictor.train_step(hist_obs, hist_acts, fut_obs)
                
                # Run rollout prediction for the next 5 steps based on the current history
                pred_hist_obs = np.array([t[0] for t in self.transitions[-10:]], dtype=np.float32)
                pred_hist_acts = np.array([t[1] for t in self.transitions[-5:]], dtype=np.float32)
                
                pred_mean, pred_var = self.predictor.forward(pred_hist_obs, pred_hist_acts)
                
                # Store the next step's prediction for the upcoming surprise calculation
                self.next_step_pred = pred_mean[0].copy()
                
                # Write predictions to double buffers
                self.buffer.write_imagined(pred_mean)
                self.buffer.write_predicted_variances(pred_var)
                
            # Sleep remainder of the cycle
            elapsed = time.perf_counter() - t_start
            rem = period - elapsed
            if rem > 0:
                time.sleep(rem)

    def stop(self):
        self._running = False


# -- SELF TEST & BENCHMARK -------------------------------------------------------
if __name__ == "__main__":
    print("=" * 72)
    print("  SpatiotemporalPredictor — NumPy Self-Test & Benchmark")
    print("=" * 72)
    
    # Set seed before instantiation to ensure deterministic weight initialization
    np.random.seed(42)
    predictor = SpatiotemporalPredictor(obs_dim=33, act_dim=2, hidden_dim=32)
    
    # Generate random transitions: 15 steps
    history_obs = np.random.randn(10, 33).astype(np.float32)
    future_obs = np.random.randn(5, 33).astype(np.float32)
    actions = np.random.uniform(-1.0, 1.0, (5, 2)).astype(np.float32)
    
    # 1. Forward Pass Timing
    t0 = time.perf_counter()
    pred_mean, pred_var = predictor.forward(history_obs, actions)
    t1 = time.perf_counter()
    forward_ms = (t1 - t0) * 1000.0
    print(f"  Forward Pass execution:  {forward_ms:.3f} ms")
    
    # 2. Training Step Timing (Forward + Loss + Backward + Weight Update)
    t0 = time.perf_counter()
    loss = predictor.train_step(history_obs, actions, future_obs)
    t1 = time.perf_counter()
    train_ms = (t1 - t0) * 1000.0
    print(f"  Train Step execution:    {train_ms:.3f} ms")
    
    # 3. Loss Convergence Check (over 100 gradient steps)
    print("\n  Running 150 optimization cycles on static sequence...")
    initial_loss = loss
    final_loss = 0.0
    for cycle in range(150):
        final_loss = predictor.train_step(history_obs, actions, future_obs)
        if cycle % 30 == 0:
            print(f"    Cycle {cycle:3d} | Loss: {final_loss:.6f}")
            
    print(f"\n  Initial Loss:            {initial_loss:.6f}")
    print(f"  Final Loss:              {final_loss:.6f}")
    
    if final_loss < initial_loss * 0.1:
        print("  [PASS] Training converged successfully (Loss reduced by >90%).")
    else:
        print("  [FAIL] Training did not converge as expected.")
        import sys
        sys.exit(1)
        
    if train_ms < 1.0:
        print("  [PASS] Train step executes in < 1.0 ms (timing margin verified).")
    else:
        print(f"  [WARNING] Train step took {train_ms:.3f} ms (target is <1.0ms).")
        
    print("=" * 72)
