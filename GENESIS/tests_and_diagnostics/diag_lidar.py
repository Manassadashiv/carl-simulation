"""
Quick diagnostic: simulate 500 steps with full debug output.
"""
import mujoco
import math
import numpy as np
import sys, os

sys.path.append(os.path.abspath('.'))
from blob_brain import BlobBrain, ReflexLayer

model = mujoco.MjModel.from_xml_path("blob_world.xml")
data  = mujoco.MjData(model)

mujoco.mj_resetData(model, data)
mujoco.mj_forward(model, data)

brain  = BlobBrain(n_neurons=32, n_inputs=20, n_outputs=2)
brain.load("memory/blob_brain.npz")
reflex = ReflexLayer(n_sensors=20, n_motors=2, threshold=0.5)
reflex.load("memory/blob_reflex.npy")

init_pos = data.geom("blob_geom").xpos.copy()
init_x, init_y = float(init_pos[0]), float(init_pos[1])
print(f"Init pos: ({init_x:.2f}, {init_y:.2f})")

def get_lidar(n_rays=8):
    blob_pos = data.geom("blob_geom").xpos
    yaw = float(data.qpos[2])
    dists = []
    for i in range(n_rays):
        angle = yaw + i * (2.0 * math.pi / n_rays)
        vec   = np.array([math.cos(angle), math.sin(angle), 0.0])
        pnt   = blob_pos + np.array([0, 0, 0.05])
        geom_id = np.array([-1], dtype=np.int32)
        dist  = mujoco.mj_ray(model, data, pnt, vec, None, 1, -1, geom_id)
        if dist < 0: dist = 5.0
        dists.append(dist)
    return dists

for step in range(500):
    blob_pos = data.geom("blob_geom").xpos.copy()
    lidars = get_lidar()
    proximity = [max(0.0, (2.0 - d) / 2.0) for d in lidars]
    
    yaw = float(data.qpos[2])
    sensors = np.array(proximity + [1.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                       dtype=np.float32)
    
    motor_intention = brain.step(sensors, dt=0.01)
    final_drive, reflex_fired = reflex.step(sensors, motor_intention, 0.0, 0.0)

    # Front-arc danger
    front_danger = max(proximity[0], proximity[7], proximity[1])
    safe_speed = 0.3 if front_danger > 0.7 else 1.0
    speed_mult = min(1.4, 1.0 + 0.8*0.4) * safe_speed

    throttle = final_drive[0] * speed_mult
    # Hard brainstem override
    if proximity[0] > 0.8:
        throttle = -0.6
    if proximity[7] > 0.75 or proximity[1] > 0.75:
        throttle = min(throttle, 0.0)

    steering = final_drive[1]
    force_x  = throttle * math.cos(yaw) * 5.0
    force_y  = throttle * math.sin(yaw) * 5.0
    torque_z = steering * 4.0

    data.ctrl[0] = force_x
    data.ctrl[1] = force_y
    data.ctrl[2] = torque_z
    mujoco.mj_step(model, data)

    abs_x = init_x + data.qpos[0]
    abs_y = init_y + data.qpos[1]

    if step % 50 == 0:
        print(f"Step {step:4d} | pos=({abs_x:6.2f},{abs_y:6.2f}) | lidar={[f'{d:.1f}' for d in lidars]} | "
              f"prox=[{max(proximity):.2f}max] | throttle={throttle:+.2f} | steer={steering:+.2f} | "
              f"reflex={reflex_fired}")
    
    # Check if hitting outer wall
    if abs(abs_x) > 4.5 or abs(abs_y) > 4.5:
        print(f"*** WALL PROXIMITY at step {step}: pos=({abs_x:.2f},{abs_y:.2f}) ***")
        print(f"    Lidar: {[round(d,2) for d in lidars]}")
        print(f"    Proximity: {[round(p,2) for p in proximity]}")
        print(f"    front_danger={front_danger:.2f}, safe_speed={safe_speed}")
        print(f"    throttle={throttle:.2f}, steering={steering:.2f}")
        break

print("\nDiagnostic complete.")
