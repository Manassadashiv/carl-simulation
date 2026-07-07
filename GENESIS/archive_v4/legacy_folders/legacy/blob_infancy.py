import time
import math
import numpy as np
import mujoco
import mujoco.viewer
from blob_brain import BlobBrain

# ── 1. Load the minimal Blob world ─────────────────────────────────────────────
model = mujoco.MjModel.from_xml_path("blob_world.xml")
data = mujoco.MjData(model)

# ── 2. Initialize the Blob's Brain (Infancy) ───────────────────────────────────
# We now have 11 inputs: 8 distance sensors (Lidar), 2 Smell sensors (food dx, dy), 1 Hunger sensor.
brain = BlobBrain(n_neurons=32, n_inputs=11, n_outputs=2)
brain.load("memory/blob_brain.npz")

hunger = 0.8  # Start hungry

def get_lidar(n_rays=8):
    """Cast 8 rays around the blob to see the world."""
    blob_pos = data.geom("blob_geom").xpos
    dists = []
    
    # 8 evenly spaced angles
    for i in range(n_rays):
        angle = i * (2.0 * math.pi / n_rays)
        vec = np.array([math.cos(angle), math.sin(angle), 0.0])
        pnt = blob_pos + np.array([0, 0, 0.05])
        
        geom_id = np.array([-1], dtype=np.int32)
        dist = mujoco.mj_ray(model, data, pnt, vec, None, 1, -1, geom_id)
        
        if dist < 0: dist = 5.0  # Max range
        dists.append(dist)
        
    return dists

def check_collisions():
    """Did we touch food (joy) or a wall (pain)?"""
    food_pos = data.geom("food").xpos
    blob_pos = data.geom("blob_geom").xpos
    
    dist_to_food = math.hypot(blob_pos[0] - food_pos[0], blob_pos[1] - food_pos[1])
    
    # 0.15 (food radius) + 0.2 (blob radius) = 0.35 eat threshold
    if dist_to_food < 0.35:
        return 'food'
        
    # Check if touching walls (simplified bounding box check)
    if abs(blob_pos[0]) > 4.7 or abs(blob_pos[1]) > 4.7:
        return 'wall'
        
    # Check center obstacle
    dist_to_obs = math.hypot(blob_pos[0], blob_pos[1])
    if dist_to_obs < 0.7:  # 0.5 (obs) + 0.2 (blob)
        return 'wall'
        
    return None

# ── 3. The Infancy Loop (Pavlovian Observation) ────────────────────────────────
print("Starting The Void (Infancy Phase)...")
print("The Blob cannot move itself. The Parent is guiding it.")
print("Watch the terminal to see it associate sight with feeling.")
print("Press Ctrl+C at any time to stop training and save the brain.")

food_count = 0

try:
    with mujoco.viewer.launch_passive(model, data) as viewer:
        
        step = 0
        while viewer.is_running():
            # -- A. The Senses (Seeing and Feeling) --
            lidars = get_lidar()
            
            # Invert lidars to proximity (1.0 = touching, 0.0 = far)
            proximity = [max(0.0, (1.0 - d)) for d in lidars]
            
            # Olfactory Sensor (Smell)
            blob_pos = data.geom("blob_geom").xpos
            food_pos = data.geom("food").xpos
            food_dx = food_pos[0] - blob_pos[0]
            food_dy = food_pos[1] - blob_pos[1]
            dist_to_food = math.hypot(food_dx, food_dy)
            if dist_to_food > 0.01:
                food_dx /= dist_to_food
                food_dy /= dist_to_food
            
            # Compile sensory input for the brain
            sensors = np.array(proximity + [food_dx, food_dy, hunger], dtype=np.float32)
            
            # -- B. The Brain (Processing) --
            # The brain thinks and generates a motor output, but we ignore it.
            # Its muscles are disconnected in infancy.
            motor_intention = brain.step(sensors, dt=0.01)
            
            # -- C. The Parent (Passive Guided Movement) --
            # We manually push the blob around the room using sine waves
            # to guarantee it sweeps across food and walls randomly.
            t = step * 0.01
            guided_vx = math.sin(t * 0.5) * 1.5 + math.cos(t * 0.2) * 0.5
            guided_vy = math.cos(t * 0.7) * 1.5 + math.sin(t * 0.3) * 0.5
            
            # Override the physics velocities
            data.qvel[0] = guided_vx
            data.qvel[1] = guided_vy
            
            # -- D. The Reward (Dopamine and Pain) --
            event = check_collisions()
            
            if event == 'food':
                # Joy! Dopamine spike! Hunger drops!
                weight_change = brain.apply_dopamine(reward_signal=1.0, learning_rate=0.05)
                hunger = max(0.0, hunger - 0.5)
                food_count += 1
                print(f"[JOY] Food eaten! Dopamine flooded the synapses. Neural weight delta: +{weight_change:.4f}. Hunger: {hunger:.2f}")
                brain.save("memory/blob_brain.npz")
                    
                # Respawn food randomly
                food_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "food")
                model.geom_pos[food_id][0] = np.random.uniform(-4.0, 4.0)
                model.geom_pos[food_id][1] = np.random.uniform(-4.0, 4.0)
                
            elif event == 'wall':
                # Pain! Norepinephrine spike! 
                # Negative dopamine acts as punishment (LTD)
                weight_change = brain.apply_dopamine(reward_signal=-1.0, learning_rate=0.05)
                if step % 50 == 0:
                    print(f"[PAIN] Hit a wall! Norepinephrine released. Neural weight delta: {weight_change:.4f}")
            
            # Hunger slowly returns
            hunger = min(1.0, hunger + 0.0005)
            
            # -- E. Physics Step --
            mujoco.mj_step(model, data)
            viewer.sync()
            
            step += 1
            time.sleep(0.01)

except KeyboardInterrupt:
    print(f"\n[INFO] Training stopped manually. Total food eaten: {food_count}")
    brain.save("memory/blob_brain.npz")
    print("[INFO] Brain state saved. Ready for Phase 2: Toddlerhood.")
