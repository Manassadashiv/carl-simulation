import pybullet as p
import pybullet_data
import time
import numpy as np
import scipy.linalg

# --- 1. The Professor's Mathematical Model ---
# State vector: x = [position, velocity, pitch, pitch_velocity]
g = 9.81
L = 0.25  # Approximate CoM height
M_body = 1.55
m_wheels = 0.2

# Linearized continuous-time dynamics (Rigid Body Assumption)
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

# LQR weights
Q = np.diag([2.0, 2.0, 50.0, 5.0]) # High penalty on pitch error
R = np.array([[1.0]])

# Solve Continuous Algebraic Riccati Equation
P = scipy.linalg.solve_continuous_are(A, B, Q, R)
K = np.linalg.inv(R) @ B.T @ P
K = K[0]

# Enforce PyBullet Sign Convention
# If Pitch > 0 (falling forward), we need Torque > 0 (driving forward to catch it)
# Since u = -Kx, we need -K[2] to be positive, so K[2] must be negative.
if K[2] > 0:
    K = -K

print(f"[Professor] Computed LQR Gain K: {K}")

# Discretize for Prediction (Euler integration)
dt = 1.0 / 240.0
Ad = np.eye(4) + A * dt
Bd = B * dt

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
        startOrientation = p.getQuaternionFromEuler([0, 0.05, 0]) # Initial tilt
        robotId = p.loadURDF("carl.urdf", startPos, startOrientation)
        
        neck_joint_idx = 2
        left_wheel_idx = 0
        right_wheel_idx = 1
        
        # Setup Flexible Neck (The source of model mismatch)
        neck_stiffness = 20.0
        p.setJointMotorControl2(robotId, neck_joint_idx, p.POSITION_CONTROL, targetPosition=0, force=neck_stiffness)
        
        # Free wheels for torque control
        p.setJointMotorControl2(robotId, left_wheel_idx, p.VELOCITY_CONTROL, force=0)
        p.setJointMotorControl2(robotId, right_wheel_idx, p.VELOCITY_CONTROL, force=0)
        
        p.resetDebugVisualizerCamera(cameraDistance=1.2, cameraYaw=45, cameraPitch=-20, cameraTargetPosition=[0,0,0.3])
        
        print("\n--- CARL Phase 2 Started ---")
        print("1. LQR Baseline Controller Active")
        print("2. Computing Prediction Error (e_k)")
        print("3. Filtering Error with EMA")
        print("----------------------------\n")
        time.sleep(3)
        
        # Initialize Prediction
        x_k = get_state(robotId)
        x_pred = x_k.copy()
        
        # Error Filtering Setup
        alpha_ema = 0.1
        e_filtered = np.zeros(4)
        
        for i in range(100000):
            # 1. Sense Reality
            x_k = get_state(robotId)
            
            # 2. Compute Raw Error
            e_raw = x_k - x_pred
            
            # 3. Filter Error (EMA)
            e_filtered = alpha_ema * e_raw + (1 - alpha_ema) * e_filtered
            
            # 4. Professor's Control Law (LQR)
            u_k = -np.dot(K, x_k)
            u_k = np.clip(u_k, -5.0, 5.0)
            
            # 5. Apply Control
            p.setJointMotorControl2(robotId, left_wheel_idx, p.TORQUE_CONTROL, force=u_k)
            p.setJointMotorControl2(robotId, right_wheel_idx, p.TORQUE_CONTROL, force=u_k)
            
            # 6. Predict Next State (The Professor's internal rigid model)
            x_pred = Ad @ x_k + Bd.flatten() * u_k
            
            # Update Camera
            p.resetDebugVisualizerCamera(cameraDistance=1.2, cameraYaw=45, cameraPitch=-20, cameraTargetPosition=[x_k[0], 0, 0.3])
            
            p.stepSimulation()
            time.sleep(dt)
            
            # Logging the Error
            if i % 60 == 0:
                print(f"Step {i} | Pitch: {x_k[2]:.3f} rad | Pitch Error (Filtered): {e_filtered[2]:.5f}")
                
            if abs(x_k[2]) > 0.6:
                print(f"\n[!] Catastrophic Failure at step {i}! Guardian intervention required.")
                time.sleep(3)
                break

    except p.error as e:
        print(f"PyBullet Error: {e}")
    finally:
        try:
            if p.getConnectionInfo(physicsClient)['isConnected']:
                p.disconnect()
        except:
            pass

if __name__ == "__main__":
    main()
