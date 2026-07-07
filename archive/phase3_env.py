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
theta_max = 0.5     # radians (failure threshold)
d_theta_max = 2.0   # rad/s
e_max = 0.05        # max filtered error expected before fall

w1, w2, w3, w4 = 1.0, 0.5, 0.5, 2.0  # High weight on prediction error!

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def calculate_guardian_probability(theta, d_theta, trust_M, error_theta):
    # Normalized Hazard Score
    H = (w1 * abs(theta) / theta_max) + \
        (w2 * abs(d_theta) / d_theta_max) + \
        (w3 * (1.0 - trust_M)) + \
        (w4 * abs(error_theta) / e_max)
    
    # We want H=1.0 to roughly equal 50% probability, H=1.5 to be 85%
    k = 5.0
    H_crit = 1.2 
    return sigmoid(k * (H - H_crit))

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
        
        print("\n--- CARL Phase 3: The Guardian ---")
        time.sleep(2)
        
        x_k = get_state(robotId)
        x_pred = x_k.copy()
        
        alpha_ema = 0.05
        e_filtered = np.zeros(4)
        trust_M = 1.0 # Assume perfect trust on the starting tile floor
        
        guardian_triggered = False
        
        for i in range(100000):
            x_k = get_state(robotId)
            
            e_raw = x_k - x_pred
            e_filtered = alpha_ema * e_raw + (1 - alpha_ema) * e_filtered
            
            # Guardian calculates Failure Probability BEFORE action
            p_fail = calculate_guardian_probability(x_k[2], x_k[3], trust_M, e_filtered[2])
            
            if p_fail > 0.70 and not guardian_triggered:
                print(f"\n[GUARDIAN TRIGGERED] at step {i}!")
                print(f"P(Failure) = {p_fail*100:.1f}%")
                print(f"Metrics: Pitch={x_k[2]:.3f}, Filtered Error={e_filtered[2]:.4f}")
                print("--> DAUGHTER MINDS WOULD SPAWN HERE <--")
                guardian_triggered = True
                
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
                print(f"Did Guardian predict this? {'YES' if guardian_triggered else 'NO'}")
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
