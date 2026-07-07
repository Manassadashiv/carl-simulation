import pybullet as p
import pybullet_data
import time
import numpy as np

def get_state(robot_id):
    pos, quat = p.getBasePositionAndOrientation(robot_id)
    vel, ang_vel = p.getBaseVelocity(robot_id)
    euler = p.getEulerFromQuaternion(quat)
    pitch = euler[1]
    pitch_vel = ang_vel[1]
    neck_state = p.getJointState(robot_id, 2)
    alpha = neck_state[0]
    alpha_vel = neck_state[1]
    return pitch, pitch_vel, pos[0], vel[0], alpha, alpha_vel

def main():
    try:
        # Connect to GUI with a large window
        physicsClient = p.connect(p.GUI, options="--width=1280 --height=720")
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        
        # Configure the camera to look right at the spawn point
        p.resetDebugVisualizerCamera(cameraDistance=1.2, cameraYaw=45, cameraPitch=-20, cameraTargetPosition=[0,0,0.3])
        
        # Load environment
        planeId = p.loadURDF("plane.urdf")
        p.changeDynamics(planeId, -1, lateralFriction=1.0)
        
        # Load CARL
        startPos = [0, 0, 0.08]
        startOrientation = p.getQuaternionFromEuler([0, 0.05, 0])
        robotId = p.loadURDF("carl.urdf", startPos, startOrientation)
        
        neck_joint_idx = 2
        left_wheel_idx = 0
        right_wheel_idx = 1
        
        # Setup Flexible Neck
        neck_stiffness = 20.0
        p.setJointMotorControl2(robotId, neck_joint_idx, p.POSITION_CONTROL, targetPosition=0, force=neck_stiffness)
        
        # Free wheels
        p.setJointMotorControl2(robotId, left_wheel_idx, p.VELOCITY_CONTROL, force=0)
        p.setJointMotorControl2(robotId, right_wheel_idx, p.VELOCITY_CONTROL, force=0)
        
        Kp_pitch = 20.0
        Kd_pitch = 2.0
        Kp_pos = 1.0
        Kd_pos = 1.0
        
        print("CARL Elite spawned. Waiting 3 seconds before activating the baseline controller...")
        time.sleep(3) # Give user time to see the URDF!
        
        for i in range(100000):
            pitch, pitch_vel, x, x_vel, alpha, alpha_vel = get_state(robotId)
            
            torque = (Kp_pitch * pitch) + (Kd_pitch * pitch_vel) + (Kp_pos * x) + (Kd_pos * x_vel)
            torque = np.clip(torque, -3.0, 3.0)
            
            p.setJointMotorControl2(robotId, left_wheel_idx, p.TORQUE_CONTROL, force=torque)
            p.setJointMotorControl2(robotId, right_wheel_idx, p.TORQUE_CONTROL, force=torque)
            
            # Follow the robot with the camera
            p.resetDebugVisualizerCamera(cameraDistance=1.2, cameraYaw=45, cameraPitch=-20, cameraTargetPosition=[x, 0, 0.3])
            
            p.stepSimulation()
            time.sleep(1./240.)
            
            if abs(pitch) > 0.6: # Tipped over completely
                print(f"Catastrophic Failure at step {i}! Guardian intervention required.")
                time.sleep(3)
                break

    except p.error as e:
        print(f"PyBullet Error (GUI closed?): {e}")
    finally:
        try:
            if p.getConnectionInfo(physicsClient)['isConnected']:
                p.disconnect()
        except:
            pass

if __name__ == "__main__":
    main()
