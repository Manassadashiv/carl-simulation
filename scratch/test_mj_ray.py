import mujoco
import numpy as np

m = mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom type="plane" size="1 1 0.1"/></worldbody></mujoco>')
d = mujoco.MjData(m)
geomid = np.array([-1], dtype=np.int32)
dist = mujoco.mj_ray(m, d, np.array([0,0,1],dtype=np.float64), np.array([0,0,-1],dtype=np.float64), None, 0, -1, geomid)
print("dist:", dist)
print("geomid:", geomid)
