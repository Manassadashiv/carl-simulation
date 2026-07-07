from carl_agent import CarlBrain
import numpy as np

def fast_forward():
    print("Fast-forwarding infant exploration phase...")
    brain = CarlBrain(n_obs=34)
    brain.load("memory/carl_brain")
    
    # Fast forward the training steps so he skips the random "infant" exploration
    brain.actor._training_steps = 300000
    brain.actor.log_std = np.array([-2.0, -2.0])
    
    brain.save("memory/carl_brain")
    print("Done. log_std is now -2.0 (std=0.13).")

if __name__ == "__main__":
    fast_forward()
