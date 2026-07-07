import numpy as np

class ObstacleController:
    def __init__(self, amplitude=0.8, speed=0.03):
        self.amplitude = amplitude
        self.speed = speed
        self.time = 0.0

    def update(self, model, data):
        self.time += self.speed
        # Oscillate cube along the Y-axis (crossing CARL's path)
        y_pos = np.sin(self.time) * self.amplitude
        try:
            data.joint('hazard_joint').qpos[0] = y_pos
        except Exception as e:
            # Fallback if accessed via direct index
            try:
                import mujoco
                jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "hazard_joint")
                if jid >= 0:
                    qadr = model.jnt_qposadr[jid]
                    data.qpos[qadr] = y_pos
            except Exception:
                pass
