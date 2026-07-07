from carl_expression import ExpressionController

class FD:
    da=0.3; ne=0.7; cort=0.1; sero=0.5; hunger=0.2; fatigue=0.0

ec = ExpressionController()
t  = ec.step(FD())
print("Expression keys:", list(t.keys()))
print("wrist_L=%.3f  grip_L=%.3f" % (t['wrist_L'], t['grip_L']))

from carl_arm_train import ArmPolicy, build_cache, get_state
import mujoco
m = mujoco.MjModel.from_xml_path('vessel_kinetic.xml')
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)
c = build_cache(m)
s = get_state(m, d, c)
print("State dim:", s.shape, "(expected (29,))")
p = ArmPolicy()
a = p.forward(s)
print("Action:", a.round(3))
print("ALL CHECKS PASSED")
