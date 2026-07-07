import sys
sys.path.insert(0, '.')
import mujoco
import numpy as np
from carl_brainstem import BrainstemController, DiscretePIDController

def fresh_sim():
    model = mujoco.MjModel.from_xml_path('vessel_kinetic.xml')
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    carl_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "spatial_identity")
    Q_CARL = model.jnt_qposadr[carl_jid]
    V_CARL = model.jnt_dofadr[carl_jid]
    data.qpos[Q_CARL] = -3.0
    mujoco.mj_forward(model, data)
    for _ in range(200):
        mujoco.mj_step(model, data)
    return model, data, Q_CARL, V_CARL

class CustomPIDBrainstem(BrainstemController):
    def __init__(self, Kp, Ki, Kd, dt=0.01, track_width=0.22, wheel_radius=0.04, max_torque=0.12):
        super().__init__(dt, track_width, wheel_radius, max_torque)
        self.left_pid = DiscretePIDController(Kp=Kp, Ki=Ki, Kd=Kd, dt=dt)
        self.right_pid = DiscretePIDController(Kp=Kp, Ki=Ki, Kd=Kd, dt=dt)

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
        
        self.vel_L_ema = self.ema_alpha * encoder_vel_2[0] + (1.0 - self.ema_alpha) * self.vel_L_ema
        self.vel_R_ema = self.ema_alpha * encoder_vel_2[1] + (1.0 - self.ema_alpha) * self.vel_R_ema
        
        corr_L = self.left_pid.compute(desired_L, self.vel_L_ema)
        corr_R = self.right_pid.compute(desired_R, self.vel_R_ema)
        
        corr_L = np.clip(corr_L, -2.0, 2.0)
        corr_R = np.clip(corr_R, -2.0, 2.0)
        
        cmd_L = desired_L + corr_L
        cmd_R = desired_R + corr_R
        
        torque_L = np.clip(cmd_L, -35.0, 35.0)
        torque_R = np.clip(cmd_R, -35.0, 35.0)
        
        diagnostics = {
            'reflex_active':    bool(reflex_active),
            'left_error':       float(desired_L - self.vel_L_ema),
            'right_error':      float(desired_R - self.vel_R_ema),
            'torque_saturated': False,
            'psi':              np.array([self.psi_L, self.psi_R, self.psi_S], dtype=np.float32)
        }
        return np.array([torque_L, torque_R], dtype=np.float32), diagnostics

bs = CustomPIDBrainstem(0.1, 0.05, 0.0)
lidar = [5.0] * 24
model, data, Q_CARL, V_CARL = fresh_sim()

print("Step | Pos X  | Pos Y  | Vel X  | wL    | wR    | cmdL  | cmdR  | integralL")
for step in range(500):
    last_v = 0.5 * last_v + 0.5 * 1.0 if step > 0 else 0.5
    wL = data.joint('w_left').qvel[0]
    wR = data.joint('w_right').qvel[0]
    cmds, _ = bs.compute_torques(lidar, np.array([wL, wR]), last_v, 0.0, data.qvel[V_CARL+5])
    data.ctrl[0] = cmds[0]
    data.ctrl[1] = cmds[1]
    data.ctrl[2:8] = 0.0
    mujoco.mj_step(model, data)
    if step % 50 == 0 or step > 450:
        print(f"{step:4d} | {data.qpos[Q_CARL]:.4f} | {data.qpos[Q_CARL+1]:.4f} | {data.qvel[V_CARL]:.4f} | {wL:.2f} | {wR:.2f} | {cmds[0]:.2f} | {cmds[1]:.2f} | {bs.left_pid.integral:.2f}")
