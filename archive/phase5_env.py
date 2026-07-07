import pybullet as p
import pybullet_data
import time
import numpy as np
import scipy.linalg

# --- The Professor's Mathematical Model ---
g = 9.81
L = 0.25
M_body = 1.55
m_wheels = 0.2

A = np.array([
    [0, 1, 0, 0],
    [0, -0.1, -g*m_wheels/M_body, 0],
    [0, 0, 0, 1],
    [0, 0.1, g*(M_body+m_wheels)/(M_body*L), 0]
])

B = np.array([
    [0],
    [1.0/M_body],
    [0],
    [-1.0/(M_body*L)]
])

Q = np.diag([2.0, 2.0, 50.0, 5.0])
R = np.array([[1.0]])

P = scipy.linalg.solve_continuous_are(A, B, Q, R)
K = np.linalg.inv(R) @ B.T @ P
K = K[0]

if K[2] > 0:
    K = -K

dt = 1.0 / 240.0
Ad = np.eye(4) + A * dt
Bd = B * dt

# --- The Guardian's Normalized Math ---
theta_max = 0.5     
d_theta_max = 2.0   
e_max = 0.05        

# Updated weights: Prioritizing angular velocity
w1, w2, w3, w4 = 1.0, 1.0, 0.5, 2.0  

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def calculate_guardian_probability(theta, d_theta, trust_M, error_theta):
    H = (w1 * abs(theta) / theta_max) + \
        (w2 * abs(d_theta) / d_theta_max) + \
        (w3 * (1.0 - trust_M)) + \
        (w4 * abs(error_theta) / e_max)
    k = 5.0
    H_crit = 1.2 
    return sigmoid(k * (H - H_crit))

# --- Phase 5: Daughter Minds (MPC) ---
def spawn_daughter_minds(x_current):
    """
    Simulates 3 distinct futures using the Professor's internal model.
    Returns the optimal emergency control action.
    """
    u_candidates = [5.0, -5.0, 0.0]
    best_u = 0.0
    best_tilt = float('inf')
    
    simulation_horizon = 20 # Simulate ~80ms into the future
    
    for u_test in u_candidates:
        x_sim = x_current.copy()
        max_tilt = 0.0
        
        for _ in range(simulation_horizon):
            x_sim = Ad @ x_sim + Bd.flatten() * u_test
            if abs(x_sim[2]) > max_tilt:
                max_tilt = abs(x_sim[2])
                
        if max_tilt < best_tilt:
            best_tilt = max_tilt
            best_u = u_test
            
    return best_u

def get_state(robot_id):
    pos, quat = p.getBasePositionAndOrientation(robot_id)
    vel, ang_vel = p.getBaseVelocity(robot_id)
    euler = p.getEulerFromQuaternion(quat)
    return np.array([pos[0], vel[0], euler[1], ang_vel[1]])

def main():
    try:
        physicsClient = p.connect(p.GUI, options="--width=1280 --height=720")
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        
        planeId = p.loadURDF("plane.urdf")
        p.changeDynamics(planeId, -1, lateralFriction=1.0)
        
        startPos = [0, 0, 0.08]
        startOrientation = p.getQuaternionFromEuler([0, 0.05, 0])
        robotId = p.loadURDF("carl.urdf", startPos, startOrientation)
        
        p.setJointMotorControl2(robotId, 2, p.POSITION_CONTROL, targetPosition=0, force=20.0)
        p.setJointMotorControl2(robotId, 0, p.VELOCITY_CONTROL, force=0)
        p.setJointMotorControl2(robotId, 1, p.VELOCITY_CONTROL, force=0)
        
        p.resetDebugVisualizerCamera(cameraDistance=1.2, cameraYaw=45, cameraPitch=-20, cameraTargetPosition=[0,0,0.3])
        
        print("\n--- CARL Phase 5: Daughter Minds MPC ---")
        time.sleep(2)
        
        x_k = get_state(robotId)
        x_pred = x_k.copy()
        
        alpha_ema = 0.05
        e_filtered = np.zeros(4)
        trust_M = 1.0 
        
        trigger_step = -1
        emergency_counter = 0
        emergency_u = 0.0
        saves_count = 0
        
        for i in range(100000):
            x_k = get_state(robotId)
            
            e_raw = x_k - x_pred
            e_filtered = alpha_ema * e_raw + (1 - alpha_ema) * e_filtered
            
            # --- The Guardian ---
            p_fail = calculate_guardian_probability(x_k[2], x_k[3], trust_M, e_filtered[2])
            
            if emergency_counter > 0:
                # Execute Daughter Mind's chosen action
                u_k = emergency_u
                emergency_counter -= 1
            else:
                if p_fail > 0.70:
                    if trigger_step == -1:
                        trigger_step = i # Record first trigger for metrics
                        
                    print(f"\n[GUARDIAN TRIGGERED] at step {i} | P(Fail)={p_fail*100:.1f}%")
                    
                    # --- The Daughter Minds ---
                    print("--> Spawning Daughter Minds...")
                    emergency_u = spawn_daughter_minds(x_k)
                    print(f"--> Future Selected: Applying {emergency_u} torque for recovery.")
                    
                    # Hold the emergency action for 15 steps (~60ms) to ensure recovery
                    emergency_counter = 15
                    u_k = emergency_u
                    saves_count += 1
                else:
                    # The Professor (LQR Baseline)
                    u_k = -np.dot(K, x_k)
                    
            u_k = np.clip(u_k, -5.0, 5.0)
            
            p.setJointMotorControl2(robotId, 0, p.TORQUE_CONTROL, force=u_k)
            p.setJointMotorControl2(robotId, 1, p.TORQUE_CONTROL, force=u_k)
            
            x_pred = Ad @ x_k + Bd.flatten() * u_k
            
            p.resetDebugVisualizerCamera(cameraDistance=1.2, cameraYaw=45, cameraPitch=-20, cameraTargetPosition=[x_k[0], 0, 0.3])
            p.stepSimulation()
            time.sleep(dt)
            
            if abs(x_k[2]) > 0.6:
                print(f"\n[!] Catastrophic Failure at step {i}!")
                if trigger_step != -1:
                    reaction_time_ms = (i - trigger_step) * (1000.0 / 240.0)
                    print(f"Reaction Delay Margin: {reaction_time_ms:.1f} ms")
                print(f"Total Saves Executed Before Fall: {saves_count}")
                time.sleep(3)
                break

    except p.error as e:
        pass
    finally:
        try:
            if p.getConnectionInfo(physicsClient)['isConnected']:
                p.disconnect()
        except:
            pass

if __name__ == "__main__":
    main()
