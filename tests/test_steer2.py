import pybullet as p
import pybullet_data
import time

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, 0) # No gravity
plane = p.loadURDF("plane.urdf")
rid = p.loadURDF("carl.urdf", [0, 0, 0.2], p.getQuaternionFromEuler([0, 0, 0]))

# Create fixed constraint so it doesn't fall, but allow rotation around Z axis?
# Actually, just disable gravity and see which way it turns!

for step in range(100):
    p.setJointMotorControl2(rid, 0, p.TORQUE_CONTROL, force=2.0)
    p.setJointMotorControl2(rid, 1, p.TORQUE_CONTROL, force=-2.0)
    p.stepSimulation()
    time.sleep(0.01)

pos, quat = p.getBasePositionAndOrientation(rid)
euler = p.getEulerFromQuaternion(quat)
print(f"Yaw after steering: {euler[2]}")

p.disconnect()
