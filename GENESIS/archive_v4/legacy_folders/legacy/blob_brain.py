import numpy as np
import os

class BlobBrain:
    def __init__(self, n_neurons=32, n_inputs=20, n_outputs=2):
        self.n_neurons = n_neurons
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        
        # W_in: Maps sensors into the reservoir (initialized very weak to let reflexes dominate at birth)
        self.W_in = np.random.randn(n_neurons, n_inputs) * 0.01
        
        # Clear any initial random noise on pre-seeded rows to make them mathematically pure
        self.W_in[0, :] = 0.0
        self.W_in[1, :] = 0.0
        self.W_in[2, :] = 0.0
        self.W_in[3, :] = 0.0
        
        # Pre-seed a weak genetic navigation attraction path:
        # Input 8 (cos(theta)) -> Neuron 0 (Throttle drive)
        # Input 9 (sin(theta)) -> Neuron 1 (Steering torque)
        self.W_in[0, 8] = 0.8
        self.W_in[1, 9] = 0.8
        
        # Innate Cortical Lidar steering traits (ego-centric):
        # Input 0 (Front Proximity) -> Neuron 2 (Wall Braking)
        self.W_in[2, 0] = -1.2
        # Input 2 (Left Proximity) -> Neuron 3 (Wall Steer-Away)
        self.W_in[3, 2] = -1.0
        # Input 6 (Right Proximity) -> Neuron 3 (Wall Steer-Away)
        self.W_in[3, 6] = 1.0
        
        # W: Recurrent connections (The reservoir where short-term memory resides)
        self.W = np.random.randn(n_neurons, n_neurons) * 0.01
        
        # W_out: Maps brain activity to motor actions (Throttle, Steering)
        self.W_out = np.random.randn(n_outputs, n_neurons) * 0.01
        
        # Clear any initial random noise on motor channels for pre-seeded neurons
        self.W_out[:, 0] = 0.0
        self.W_out[:, 1] = 0.0
        self.W_out[:, 2] = 0.0
        self.W_out[:, 3] = 0.0
        
        # Pre-seed Neuron 0 -> Motor 0 (Throttle), Neuron 1 -> Motor 1 (Steering)
        self.W_out[0, 0] = 0.8
        self.W_out[1, 1] = 0.8
        # Pre-seed Neuron 2 (Wall Braking) -> Motor 0 (Throttle)
        self.W_out[0, 2] = 1.0
        # Pre-seed Neuron 3 (Wall Steer-Away) -> Motor 1 (Steering)
        self.W_out[1, 3] = 1.0
        
        # Neural state (firing rates)
        self.state = np.zeros(n_neurons)
        
        # Eligibility traces for credit assignment
        self.eligibility = np.zeros((n_neurons, n_neurons))
        self.eligibility_in = np.zeros((n_neurons, n_inputs))
        self.eligibility_out = np.zeros((n_outputs, n_neurons))

        # Critic (Basal Ganglia / Striatum)
        self.W_critic = np.random.randn(n_neurons) * 0.01
        self.gamma = 0.98
        self.prev_value = 0.0
        self.prev_state = np.zeros(n_neurons)
        self.prev_action = np.zeros(n_outputs)
        self.prev_mean_action = np.zeros(n_outputs)
        
        # Policy standard deviation (internal synaptic motor noise)
        self.sigma = 0.20
        
        self.current_action = np.zeros(n_outputs)
        self.current_mean_action = np.zeros(n_outputs)

    def step(self, inputs, dt=0.01):
        """Step the neural dynamics forward in time."""
        pre_state = self.state.copy()
        
        # Euler integration of continuous-time dynamics
        dstate = -self.state + np.tanh(self.W @ self.state + self.W_in @ inputs)
        self.state += dstate * dt
        
        # Actor computes mean motor intention
        mean_action = np.tanh(self.W_out @ self.state)
        
        # Sample stochastic action internally (synaptic motor noise)
        action = mean_action + np.random.randn(self.n_outputs) * self.sigma
        action = np.clip(action, -1.0, 1.0)
        
        # Update eligibility traces: tracks pre-synaptic and post-synaptic co-activation
        self.eligibility = 0.95 * self.eligibility + np.outer(self.state, pre_state)
        self.eligibility_in = 0.95 * self.eligibility_in + np.outer(self.state, inputs)
        self.eligibility_out = 0.95 * self.eligibility_out + np.outer(action, pre_state)
        
        # Cache for next TD update
        self.current_action = action.copy()
        self.current_mean_action = mean_action.copy()
        
        return action
        
    def apply_td_feedback(self, external_reward, learning_rate=0.01):
        """
        Dopaminergic TD Learning (Basal Ganglia & Cortical Actor-Critic).
        Computes true Reward Prediction Error (RPE) and updates Critic, Actor, and Recurrent weights.
        """
        # Calculate Critic value of current state
        value = float(np.dot(self.W_critic, self.state))
        
        # TD-Error (Dopaminergic RPE spike)
        td_error = external_reward + self.gamma * value - self.prev_value
        
        # Update Critic weights
        dW_critic = learning_rate * 2.0 * td_error * self.prev_state
        self.W_critic = np.clip(self.W_critic + dW_critic, -2.0, 2.0)
        
        # Update Actor weights (Corticostriatal Policy-Gradient Plasticity)
        # We reinforce actions that are better than predicted mean (td_error > 0)
        action_diff = self.prev_action - self.prev_mean_action
        dW_actor = learning_rate * td_error * np.outer(action_diff, self.prev_state)
        self.W_out = np.clip(self.W_out + dW_actor, -2.0, 2.0)
        
        # Update Recurrent weights (RPE-gated eligibility traces)
        dW_recurrent = learning_rate * 0.1 * td_error * self.eligibility
        self.W = np.clip(self.W + dW_recurrent, -2.0, 2.0)
        
        # Update projection weights (W_in)
        dW_in = learning_rate * 0.1 * td_error * self.eligibility_in
        self.W_in = np.clip(self.W_in + dW_in, -2.0, 2.0)
        
        # Store transition variables for next step
        self.prev_value = value
        self.prev_state = self.state.copy()
        self.prev_action = self.current_action.copy()
        self.prev_mean_action = self.current_mean_action.copy()
        
        # Enforce genetic uncorruptible reflex constraints
        self.W_in[0, :] = 0.0
        self.W_in[1, :] = 0.0
        self.W_in[2, :] = 0.0
        self.W_in[3, :] = 0.0
        
        self.W_in[0, 8] = 0.8
        self.W_in[1, 9] = 0.8
        self.W_in[2, 0] = -1.2
        self.W_in[3, 2] = -1.0
        self.W_in[3, 6] = 1.0
        
        self.W_out[:, 0] = 0.0
        self.W_out[:, 1] = 0.0
        self.W_out[:, 2] = 0.0
        self.W_out[:, 3] = 0.0
        
        self.W_out[0, 0] = 0.8
        self.W_out[1, 1] = 0.8
        self.W_out[0, 2] = 1.0
        self.W_out[1, 3] = 1.0
        
        # Decay exploration noise slowly over time as we learn
        self.sigma = max(0.02, self.sigma * 0.99995)
        
        return float(td_error)

    def apply_dopamine(self, reward_signal, learning_rate=0.01):
        """Compatibility wrapper mapping to TD feedback loop."""
        return self.apply_td_feedback(reward_signal, learning_rate)

    def startle(self, magnitude=1.0):
        """Inject random noise into the neural state to break out of corners/local minima."""
        self.state = np.random.randn(self.n_neurons) * magnitude
        self.eligibility.fill(0.0)
        self.eligibility_in.fill(0.0)
        self.eligibility_out.fill(0.0)
        self.prev_value = 0.0

    def save(self, filepath="memory/blob_brain.npz"):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        np.savez(filepath, W=self.W, W_in=self.W_in, W_out=self.W_out, W_critic=self.W_critic)
        
    def load(self, filepath="memory/blob_brain.npz"):
        if os.path.exists(filepath):
            try:
                data = np.load(filepath)
                if self.W.shape == data['W'].shape and self.W_in.shape == data['W_in'].shape and self.W_out.shape == data['W_out'].shape:
                    self.W = data['W']
                    self.W_in = data['W_in']
                    self.W_out = data['W_out']
                    if 'W_critic' in data and self.W_critic.shape == data['W_critic'].shape:
                        self.W_critic = data['W_critic']
                    print(f"[MEMORY] Successfully loaded existing brain from {filepath}")
                else:
                    print("[MEMORY] Brain architecture changed. Starting fresh.")
            except Exception as e:
                print(f"[MEMORY] Error loading brain: {e}. Starting fresh.")
        else:
            print("[MEMORY] No existing brain found. Starting from scratch.")


class ReflexLayer:
    """
    Hebbian Reflex Layer (The Spinal Cord).
    A direct sensory->motor mapping that starts pre-wired and learns Hebbian wall avoidance.
    Bypasses deliberation when confidence is high.
    """
    def __init__(self, n_sensors=20, n_motors=2, threshold=0.85):
        self.W = np.zeros((n_motors, n_sensors), dtype=np.float32)
        
        # Innate Steering Avoidance Reflexes (ego-centric Lidar index matched):
        # index 0 (Front) -> Brake/Reverse
        self.W[0, 0] = -1.2
        
        # index 1 (Front-Left) -> Steer Right (negative torque), slow down
        self.W[1, 1] = -0.8
        self.W[0, 1] = -0.4
        
        # index 2 (Left) -> Steer Right (negative torque)
        self.W[1, 2] = -1.0
        
        # index 6 (Right) -> Steer Left (positive torque)
        self.W[1, 6] = 1.0
        
        # index 7 (Front-Right) -> Steer Left (positive torque), slow down
        self.W[1, 7] = 0.8
        self.W[0, 7] = -0.4

        self.threshold = threshold
        self.n_motors = n_motors
        self.n_sensors = n_sensors
        self.eligibility = np.zeros_like(self.W)
        self.total_count = 0
        self.fired_count = 0
        self.confidence = 0.0

    def step(self, sensors, reservoir_drive, reward_signal, ne_signal):
        sensors = np.asarray(sensors, dtype=np.float32).copy()
        # Reflexes strictly react to wall proximity (first 8 inputs)
        sensors[8:] = 0.0
        
        self.total_count += 1
        self.W *= 0.9995  # Slow decay

        raw = self.W @ sensors
        reflex_out = np.tanh(raw)
        self.confidence = float(np.max(np.abs(raw)))

        # Update eligibility trace
        self.eligibility = 0.8 * self.eligibility + np.outer(reservoir_drive, sensors)

        # Hebbian learning gated by Norepinephrine and Dopamine
        ne_gate = float(np.clip(ne_signal, 0.0, 1.0))
        reward_gate = float(np.clip(abs(reward_signal), 0.0, 1.0))
        gate = max(ne_gate, reward_gate * 0.5)

        # Spinal cord weights are genetically fixed and uncorruptible
        pass

        # Fire and override if confidence exceeds threshold
        if self.confidence > self.threshold and self.total_count > 100:
            self.fired_count += 1
            return reflex_out, True

        return reservoir_drive, False

    def reflex_ratio(self):
        if self.total_count == 0: return 0.0
        return self.fired_count / self.total_count

    def save(self, filepath="memory/blob_reflex.npy"):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        np.save(filepath, self.W)

    def load(self, filepath="memory/blob_reflex.npy"):
        if os.path.exists(filepath):
            try:
                self.W = np.load(filepath)
                print(f"[MEMORY] Successfully loaded reflex weights from {filepath}")
            except Exception as e:
                print(f"[MEMORY] Error loading reflex weights: {e}. Starting empty.")
        else:
            print("[MEMORY] No reflex memory found. Spinal cord starting empty.")

if __name__ == "__main__":
    print("Testing BlobBrain (CTRNN + Dopaminergic STDP) in isolation...")
    
    brain = BlobBrain(n_neurons=16, n_inputs=1, n_outputs=2)
    
    # 1. Feed it hunger (input = 1.0)
    hunger_input = np.array([1.0])
    
    out1 = brain.step(hunger_input)
    print(f"Initial Output Action: {out1}")
    
    # 2. Simulate 50 steps of movement
    print("\nSimulating 50 steps of random drift...")
    for _ in range(50):
        brain.step(hunger_input)
        
    out2 = brain.step(hunger_input)
    print(f"Pre-Reward Output Action: {out2}")
    
    # 3. Simulate hitting food -> massive dopamine spike
    print("\nFood Touched! Applying Dopamine (+1.0)")
    weight_change = brain.apply_dopamine(1.0, learning_rate=0.1)
    print(f"Total absolute weight change in recurrent network: {weight_change:.4f}")
    
    # 4. Check outputs immediately after reward
    out3 = brain.step(hunger_input)
    print(f"Post-Reward Output Action: {out3}")
    
    print("\nNotice how the Post-Reward action shifted. The synapses that caused the recent behavior just got permanently strengthened.")
