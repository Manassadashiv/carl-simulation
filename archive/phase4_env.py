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

A_nom = np.array([
    [0, 1, 0, 0],
    [0, -0.1, -g*m_wheels/M_body, 0],
    [0, 0, 0, 1],
    [0, 0.1, g*(M_body+m_wheels)/(M_body*L), 0]
])

B_nom = np.array([
    [0],
    [1.0/M_body],
    [0],
    [-1.0/(M_body*L)]
])

Q = np.diag([2.0, 2.0, 50.0, 5.0])
R = np.array([[1.0]])

P_are = scipy.linalg.solve_continuous_are(A_nom, B_nom, Q, R)
K = np.linalg.inv(R) @ B_nom.T @ P_are
K = K[0]
if K[2] > 0:
    K = -K

dt = 1.0 / 240.0
Ad_nom = np.eye(4) + A_nom * dt
Bd_nom = B_nom * dt

# --- The Mechanic: Multi-Output RLS Init ---
# Theta is 5x4 matrix. Top 4x4 is A_d. Bottom 1x4 is B_d.
Theta = np.vstack((Ad_nom, Bd_nom.flatten())) 
P_rls = 100.0 * np.eye(5)
lambda_forget = 0.995
Theta_max_drift = np.abs(Theta) * 2.0 + 0.1

# --- The Mapper: Confidence Grid Init ---
grid_res = 0.1
M_grid = np.ones((200, 200)) # 20x20m area

def update_mapper(x_pos, y_pos, error_norm):
    i = int((x_pos + 10.0) / grid_res)
    j = int((y_pos + 10.0) / grid_res)
    i = np.clip(i, 0, 199)
    j = np.clip(j, 0, 199)
    
    beta = 0.2
    lam = 0.5
    gamma = 0.01
    
    M_old = M_grid[i, j]
    M_update = M_old - lam * error_norm + gamma
    M_new = (1 - beta) * M_old + beta * M_update
    M_grid[i, j] = np.clip(M_new, 0.0, 1.0)
    
    return M_grid[i, j]

# --- The Guardian ---
theta_max = 0.5     
d_theta_max = 2.0   
e_max = 0.05        
w1, w2, w3, w4 = 1.0, 1.0, 0.5, 2.0  

def calculate_guardian_probability(theta, d_theta, trust_M, error_theta):
    H = (w1 * abs(theta) / theta_max) + \
        (w2 * abs(d_theta) / d_theta_max) + \
        (w3 * (1.0 - trust_M)) + \
        (w4 * abs(error_theta) / e_max)
    k = 5.0
    H_crit = 1.2 
    return 1.0 / (1.0 + np.exp(-k * (H - H_crit)))

def spawn_daughter_minds(x_current, Theta_current):
    u_candidates = [5.0, -5.0, 0.0]
    best_u = 0.0
    best_tilt = float('inf')
    
    # Extract updated Ad and Bd from Theta for MPC simulation
    Ad_est = Theta_current[0:4, :].T
    Bd_est = Theta_current[4, :].T
    
    for u_test in u_candidates:
        x_sim = x_current.copy()
        max_tilt = 0.0
        for _ in range(15):
            x_sim = Ad_est @ x_sim + Bd_est * u_test
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
    return np.array([pos[0], vel[0], euler[1], ang_vel[1]]), pos[1]

def main():
    global Theta, P_rls, M_grid
    
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
        
        print("\n--- CARL Phase 4: Mechanic & Mapper (Learning Active) ---")
        time.sleep(2)
        
        x_k, y_pos = get_state(robotId)
        u_k = 0.0
        
        alpha_ema = 0.05
        e_filtered = np.zeros(4)
        
        emergency_counter = 0
        emergency_u = 0.0
        
        trigger_count = 0
        mpc_usage = 0
        
        for i in range(100000):
            # 1. Sense Reality (x_{k+1})
            x_next, y_pos = get_state(robotId)
            
            # 2. Mechanic (Multi-Output RLS)
            Phi_k = np.append(x_k, u_k).reshape(5, 1) # 5x1
            
            # Prediction
            x_pred = Theta.T @ Phi_k # 4x5 @ 5x1 -> 4x1
            x_pred = x_pred.flatten()
            
            # Error Calculation
            e_k = x_next - x_pred
            e_filtered = alpha_ema * e_k + (1 - alpha_ema) * e_filtered
            
            # RLS Gain & Update
            P_Phi = P_rls @ Phi_k
            L_k = P_Phi / (lambda_forget + Phi_k.T @ P_Phi) # 5x1
            
            Theta = Theta + L_k @ e_k.reshape(1, 4)
            Theta = np.clip(Theta, -Theta_max_drift, Theta_max_drift)
            P_rls = (P_rls - L_k @ Phi_k.T @ P_rls) / lambda_forget
            
            # 3. Mapper Update
            trust_M = update_mapper(x_next[0], y_pos, abs(e_filtered[2])/e_max)
            
            # 4. Guardian
            p_fail = calculate_guardian_probability(x_next[2], x_next[3], trust_M, e_filtered[2])
            
            if emergency_counter > 0:
                u_next = emergency_u
                emergency_counter -= 1
                mpc_usage += 1
            else:
                if p_fail > 0.70:
                    trigger_count += 1
                    emergency_u = spawn_daughter_minds(x_next, Theta)
                    emergency_counter = 15
                    u_next = emergency_u
                    mpc_usage += 1
                else:
                    # LQR Control using standard K (Fixed)
                    u_next = -np.dot(K, x_next)
                    
            u_next = np.clip(u_next, -5.0, 5.0)
            
            p.setJointMotorControl2(robotId, 0, p.TORQUE_CONTROL, force=u_next)
            p.setJointMotorControl2(robotId, 1, p.TORQUE_CONTROL, force=u_next)
            
            x_k = x_next
            u_k = u_next
            
            p.resetDebugVisualizerCamera(cameraDistance=1.2, cameraYaw=45, cameraPitch=-20, cameraTargetPosition=[x_k[0], 0, 0.3])
            p.stepSimulation()
            time.sleep(dt)
            
            if i % 240 == 0:
                print(f"Step {i} | Error: {abs(e_filtered[2]):.5f} | Guardian Triggers: {trigger_count} | MPC Ticks: {mpc_usage}")
                
            if abs(x_next[2]) > 0.6:
                print(f"\n[!] Catastrophic Failure at step {i}!")
                print(f"Final Stats -> Triggers: {trigger_count}, MPC Usage: {mpc_usage}")
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
