import numpy as np
import math

SDIM = 5
T_ltm = np.load("checkpoint_ltm_T.npy")
Ad = T_ltm[:SDIM, :].T
Bd = T_ltm[SDIM, :]
ACTIONS = [-8., -5., -2., -1., 0., 1., 2., 5., 8.]
STEER_ACTIONS = [-2., 0., 2.]
STEER_GAIN = 6.0
DT = 1.0/240.0
gx, gy = 5.4, 5.4

base_dop_w = 0.6
base_danger_w = 2.0

def danger_at(pitch):
    return 1.0 if abs(pitch) > 0.40 else 0.0

def eval_daughter_minds():
    best_F, best_u, best_s = float('inf'), 0., 0.
    
    for u in ACTIONS:
        for s in STEER_ACTIONS:
            xs, F = np.zeros(SDIM), 0.
            xw, yw = 0.5, 0.5
            heading = 0.0
            
            # Leg 1
            for h in range(20):
                xs_n = Ad @ xs + Bd * u
                heading -= s * STEER_GAIN * DT
                spd = xs_n[1] * DT * 25.
                xw += spd * math.cos(heading)
                yw += spd * math.sin(heading)
                d2 = math.sqrt((xw-gx)**2 + (yw-gy)**2)
                kin_d = danger_at(xs_n[2]) * base_danger_w
                pred_err = float(np.linalg.norm(xs_n - xs)) * 0.02
                F += (kin_d + base_dop_w * d2 * 1.5 + abs(xs_n[2])*0.1 + pred_err) * 1.0
                xs = xs_n
                
            branch_best = float('inf')
            for u2 in ACTIONS:
                xs2, F2 = xs.copy(), 0.
                xw2, yw2 = xw, yw
                h2_heading = heading
                for h2 in range(10):
                    xs2_n = Ad @ xs2 + Bd * u2
                    spd2 = xs2_n[1] * DT * 25.
                    xw2 += spd2 * math.cos(h2_heading)
                    yw2 += spd2 * math.sin(h2_heading)
                    d3 = math.sqrt((xw2-gx)**2 + (yw2-gy)**2)
                    kin_d2 = danger_at(xs2_n[2]) * base_danger_w
                    pred_err2 = float(np.linalg.norm(xs2_n - xs2)) * 0.02
                    F2 += (kin_d2 + base_dop_w * d3 * 1.5 + abs(xs2_n[2])*0.1 + pred_err2) * 1.0
                    xs2 = xs2_n
                branch_best = min(branch_best, F2)
                
            F += branch_best
            print(f"u={u:>4}, s={s:>4} -> F={F:.4f}, final Leg1 pitch={xs[2]:.4f}")
            if F < best_F:
                best_F, best_u, best_s = F, u, s
                
    print(f"\nWINNER: u={best_u}, s={best_s} with F={best_F:.4f}")

eval_daughter_minds()
