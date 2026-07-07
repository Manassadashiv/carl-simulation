import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))

import numpy as np
import mujoco
from carl_arm_train import build_cache, reset_episode, ARM_CTRL_LOW, ARM_CTRL_HIGH, ARM_CTRL_SLICE
from scratch.test_ik import get_ik_action_dls
from carl_arm_bc import get_ik_action

# Run 1: test_ik
model1 = mujoco.MjModel.from_xml_path("vessel_kinetic.xml")
data1 = mujoco.MjData(model1)
cache1 = build_cache(model1)
rng1 = np.random.default_rng(seed=42)
reset_episode(model1, data1, cache1, "reach", rng1)

# Run 2: standalone
model2 = mujoco.MjModel.from_xml_path("vessel_kinetic.xml")
data2 = mujoco.MjData(model2)
cache2 = build_cache(model2)
rng2 = np.random.default_rng(seed=42)
reset_episode(model2, data2, cache2, "reach", rng2)

obj_pos1 = data1.xpos[cache1['obj_bid']].copy()

print(f"Initial Object Pos 1: {obj_pos1}")
print(f"Initial Object Pos 2: {data2.xpos[cache2['obj_bid']].copy()}")

for step in range(20):
    # Run 1 step
    data1.qpos[cache1['Q_CARL'] : cache1['Q_CARL']+3] = [0.0, 0.0, 0.04]
    data1.qpos[cache1['Q_CARL']+3 : cache1['Q_CARL']+7] = [1.0, 0.0, 0.0, 0.0]
    data1.qvel[cache1['V_CARL'] : cache1['V_CARL']+6] = 0.0
    
    act1 = get_ik_action_dls(model1, data1, cache1, obj_pos1)
    center1 = (ARM_CTRL_HIGH + ARM_CTRL_LOW) / 2.0
    scale1 = (ARM_CTRL_HIGH - ARM_CTRL_LOW) / 2.0
    act_phys1 = act1 * scale1 + center1
    data1.ctrl[:2] = 0.0
    data1.ctrl[ARM_CTRL_SLICE] = act_phys1
    mujoco.mj_step(model1, data1)
    dist1 = np.linalg.norm(obj_pos1 - data1.site_xpos[cache1['tip_L_sid']])
    
    # Run 2 step
    data2.qpos[cache2['Q_CARL'] : cache2['Q_CARL']+3] = [0.0, 0.0, 0.04]
    data2.qpos[cache2['Q_CARL']+3 : cache2['Q_CARL']+7] = [1.0, 0.0, 0.0, 0.0]
    data2.qvel[cache2['V_CARL'] : cache2['V_CARL']+6] = 0.0
    
    obj_pos2 = data2.xpos[cache2['obj_bid']].copy()
    act2 = get_ik_action(model2, data2, cache2, obj_pos2)
    center2 = (ARM_CTRL_HIGH + ARM_CTRL_LOW) / 2.0
    scale2 = (ARM_CTRL_HIGH - ARM_CTRL_LOW) / 2.0
    act_phys2 = act2 * scale2 + center2
    data2.ctrl[:2] = 0.0
    data2.ctrl[ARM_CTRL_SLICE] = act_phys2
    mujoco.mj_step(model2, data2)
    dist2 = np.linalg.norm(obj_pos2 - data2.site_xpos[cache2['tip_L_sid']])
    
    print(f"Step {step:02d} | Dist1: {dist1:.4f}m, Dist2: {dist2:.4f}m | Obj2 Pos: {obj_pos2.round(4)} | Act1 L: {act_phys1[:4].round(3)}, Act2 L: {act_phys2[:4].round(3)}")
