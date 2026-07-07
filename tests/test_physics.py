import pybullet as p
import pybullet_data
import time

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
plane = p.loadURDF("plane.urdf")
rid = p.loadURDF("carl.urdf", [0, 0, 0.1], p.getQuaternionFromEuler([0, 0.1, 0]))

p.setJointMotorControl2(rid, 2, p.POSITION_CONTROL, targetPosition=0, force=12.)
p.setJointMotorControl2(rid, 0, p.VELOCITY_CONTROL, force=0)
p.setJointMotorControl2(rid, 1, p.VELOCITY_CONTROL, force=0)

for step in range(500):
    p.setJointMotorControl2(rid, 0, p.TORQUE_CONTROL, force=2.0)
    p.setJointMotorControl2(rid, 1, p.TORQUE_CONTROL, force=2.0)
    p.stepSimulation()
    time.sleep(0.01)

pos, _ = p.getBasePositionAndOrientation(rid)
print(f"Final Pos after fwd torque: {pos}")

p.disconnect()
