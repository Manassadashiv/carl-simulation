import numpy as np

SDIM = 5
T_ltm = np.load("checkpoint_ltm_T.npy")
Ad = T_ltm[:SDIM, :].T
Bd = T_ltm[SDIM, :]
xs = np.zeros(SDIM)
gx, gy = 5.4, 5.4
xw, yw = 0.5, 0.5
yaw = 0.0
dop_w = 0.6
DT = 1.0/240.0

def eval_action(u, s):
    F = 0
    _xs = xs.copy()
    _xw, _yw = xw, yw
    _yaw = yaw
    for h in range(15):
        xs_n = Ad @ _xs + Bd * u
        _yaw -= s * 6.0 * DT
        spd = xs_n[1] * DT * 25.0
        _xw += spd * np.cos(_yaw)
        _yw += spd * np.sin(_yaw)
        F += (dop_w * np.sqrt((_xw - gx)**2 + (_yw - gy)**2) * 1.5 + float(np.linalg.norm(xs_n - _xs)) * 0.02 + abs(xs_n[2]) * 0.1)
        _xs = xs_n
    print(f"u={u}, s={s} -> F={F:.4f}, final pitch={_xs[2]:.4f}")

eval_action(0, 0)
eval_action(1, -2)
eval_action(3, -2)
