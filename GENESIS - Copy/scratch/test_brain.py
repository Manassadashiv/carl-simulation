import numpy as np
import sys
import os
import math

sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('..'))

from blob_brain import BlobBrain, ReflexLayer

def test():
    print("=== Brain & Reflex Diagnostic Test ===")
    
    # Load model
    brain = BlobBrain(n_neurons=32, n_inputs=20, n_outputs=2)
    brain.load("memory/blob_brain.npz")
    
    reflex = ReflexLayer(n_sensors=20, n_motors=2, threshold=0.85)
    reflex.load("memory/blob_reflex.npy")
    
    # Case 1: Target is directly ahead (relative angle = 0)
    # cos_target = 1.0, sin_target = 0.0
    sensors = np.zeros(20, dtype=np.float32)
    sensors[8] = 1.0  # cos_target
    sensors[9] = 0.0  # sin_target
    
    brain.state.fill(0.0) # Reset state
    for _ in range(100):
        brain.step(sensors, dt=0.01)
    print(f"Target Ahead (Steady State) -> Mean Intention: {brain.current_mean_action}")
    
    # Case 2: Target is to the LEFT (relative angle = +pi/2)
    # cos_target = 0.0, sin_target = 1.0
    sensors = np.zeros(20, dtype=np.float32)
    sensors[8] = 0.0
    sensors[9] = 1.0
    
    brain.state.fill(0.0)
    for _ in range(100):
        brain.step(sensors, dt=0.01)
    print(f"Target Left (Steady State) -> Mean Intention: {brain.current_mean_action}")
    
    # Case 3: Target is to the RIGHT (relative angle = -pi/2)
    # cos_target = 0.0, sin_target = -1.0
    sensors = np.zeros(20, dtype=np.float32)
    sensors[8] = 0.0
    sensors[9] = -1.0
    
    brain.state.fill(0.0)
    for _ in range(100):
        brain.step(sensors, dt=0.01)
    print(f"Target Right (Steady State) -> Mean Intention: {brain.current_mean_action}")
    
    # Case 4: Wall directly in FRONT (proximity[0] = 0.9)
    sensors = np.zeros(20, dtype=np.float32)
    sensors[0] = 0.9  # Front Lidar proximity
    sensors[8] = 1.0  # cos_target (still trying to go forward)
    sensors[9] = 0.0
    
    brain.state.fill(0.0)
    for _ in range(100):
        brain.step(sensors, dt=0.01)
    print(f"Wall Front -> Mean Intention: {brain.current_mean_action}")
    
    # Case 5: Wall on the LEFT (proximity[2] = 0.9)
    sensors = np.zeros(20, dtype=np.float32)
    sensors[2] = 0.9  # Left Lidar proximity
    sensors[8] = 1.0
    sensors[9] = 0.0
    
    brain.state.fill(0.0)
    for _ in range(100):
        brain.step(sensors, dt=0.01)
    print(f"Wall Left -> Mean Intention: {brain.current_mean_action}")

    # Case 6: Wall on the RIGHT (proximity[6] = 0.9)
    sensors = np.zeros(20, dtype=np.float32)
    sensors[6] = 0.9  # Right Lidar proximity
    sensors[8] = 1.0
    sensors[9] = 0.0
    
    brain.state.fill(0.0)
    for _ in range(100):
        brain.step(sensors, dt=0.01)
    print(f"Wall Right -> Mean Intention: {brain.current_mean_action}")

if __name__ == "__main__":
    test()
