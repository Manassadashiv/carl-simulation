import numpy as np
import sys
import os

# Ensure the GENESIS directory is in path
sys.path.append(os.path.abspath('.'))

from blob_brain import BlobBrain, ReflexLayer

def main():
    print("Generating pre-seeded, index-matched Lidar traits...")
    
    # 1. Instantiate the Brain and ReflexLayer
    brain = BlobBrain(n_neurons=32, n_inputs=20, n_outputs=2)
    reflex = ReflexLayer(n_sensors=20, n_motors=2, threshold=0.85)
    
    # 2. Force creation of memory directory
    os.makedirs("memory", exist_ok=True)
    
    # 3. Save files, overwriting any previous versions
    brain.save("memory/blob_brain.npz")
    reflex.save("memory/blob_reflex.npy")
    
    print("Successfully saved preinstalled Lidar traits to memory/blob_brain.npz and memory/blob_reflex.npy!")

if __name__ == "__main__":
    main()
