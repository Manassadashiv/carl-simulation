import sys
sys.path.insert(0, '.')
import mujoco
import numpy as np
from carl_brainstem import BrainstemController

def fresh_sim():
    model = mujoco.MjModel.from_xml_path('vessel_kinetic.xml')
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    data.qpos[0] = -3.0
    for _ in range(200):
        mujoco.mj_step(model, data)
    carl_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "spatial_identity")
    Q_CARL = model.jnt_qposadr[carl_jid]
    V_CARL = model.jnt_dofadr[carl_jid]
    return model, data, Q_CARL, V_CARL

class FeedforwardBrainstem(BrainstemController):
    def compute_torques(self, lidar_24, encoder_vel_2, target_linear, target_angular, actual_omega=0.0, bypass_reflex=False, stress=0.0, stiffness=1.0):
        min_lidar = np.min(lidar_24)
        approach_vel = target_linear if target_linear > 0 else 0.0
        cbf_linear = self.cbf.filter(target_linear, min_lidar, approach_vel)
        
        if bypass_reflex:
            safe_linear, safe_angular, reflex_active = cbf_linear, target_angular, False
        else:
            safe_linear, safe_angular, reflex_active = self.reflex.check(
                lidar_24, cbf_linear, target_angular, stress=stress, stiffness=stiffness
            )
        
        half_track = self.track_width / 2.0
        desired_L = (safe_linear - safe_angular * half_track) / self.wheel_radius
        desired_R = (safe_linear + safe_angular * half_track) / self.wheel_radius
        
        desired_L = np.clip(desired_L, -35.0, 35.0)
        desired_R = np.clip(desired_R, -35.0, 35.0)
        
        return np.array([desired_L, desired_R], dtype=np.float32), {'reflex_active': reflex_active, 'psi': np.zeros(3)}

model, data, Q_CARL, V_CARL = fresh_sim()
bs_ff = FeedforwardBrainstem()
lidar = [5.0] * 24
last_v = 0.0
for step in range(500):
    last_v = 0.5 * last_v + 0.5 * 1.0
    wL = data.joint('w_left').qvel[0]
    wR = data.joint('w_right').qvel[0]
    cmds, diag = bs_ff.compute_torques(lidar, np.array([wL, wR]), last_v, 0.0, data.qvel[V_CARL+5])
    data.ctrl[0] = cmds[0]
    data.ctrl[1] = cmds[1]
    data.ctrl[2:8] = 0.0
    mujoco.mj_step(model, data)
    if step % 50 == 0:
        print(f"Step {step:3d} | Pos X: {data.qpos[Q_CARL]:.4f} | Vel X: {data.qvel[V_CARL]:.4f} | Cmd: {cmds[0]:.2f}")
print(f"Final Pos X: {data.qpos[Q_CARL]:.4f} | Final Vel X: {data.qvel[V_CARL]:.4f}")
