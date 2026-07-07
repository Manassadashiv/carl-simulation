import re

with open('phase14_maze.py', 'r', encoding='utf-8') as f:
    content = f.read()

target = """    for strat in strategies:
        dop_w    = strat["dop_w"]
        danger_w = strat["d_w"] * ne_amp * grief_w * soc_mod
        for u in REDUCED_ACTIONS:
            for s in STEER_ACTIONS:
                xs, F   = x.copy(), 0.
                xw, yw  = pos_2d
                heading = yaw
                # Leg 1
                for h in range(20):
                    xs_n     = Ad @ xs + Bd * u
                    conf     = max(alpha, 0.1) * (0.95**h)
                    heading -= s * STEER_GAIN * DT
                    spd      = xs_n[1] * DT * 25.
                    xw      += spd * math.cos(heading)
                    yw      += spd * math.sin(heading)
                    d2       = math.sqrt((xw-gx)**2 + (yw-gy)**2)
                    kin_d    = danger_at(D, xs_n[2], xs_n[3]) * danger_w
                    sp_d     = cognitive_map_danger(CM, xw, yw) * danger_w * 0.5
                    F       += (kin_d + sp_d + dop_w*d2*1.5
                               + abs(homeostatic.correction_signal(xs_n[2]))*0.1) * conf
                    xs = xs_n
                # Leg 2 — branch (straight steer only for speed; 5×10 rollout)
                branch_best = float('inf')
                for u2 in REDUCED_ACTIONS:
                    xs2, F2    = xs.copy(), 0.
                    xw2, yw2   = xw, yw
                    h2_heading = heading
                    for h2 in range(10):
                        xs2_n      = Ad @ xs2 + Bd * u2
                        conf2      = max(alpha, 0.1) * (0.95**(15+h2))
                        h2_heading -= 0 * STEER_GAIN * DT
                        spd2       = xs2_n[1] * DT * 25.
                        xw2       += spd2 * math.cos(h2_heading)
                        yw2       += spd2 * math.sin(h2_heading)
                        d3         = math.sqrt((xw2-gx)**2 + (yw2-gy)**2)
                        F2        += (danger_at(D,xs2_n[2],xs2_n[3])*danger_w
                                     + cognitive_map_danger(CM,xw2,yw2)*danger_w*0.5
                                     + dop_w*d3*1.5) * conf2
                        xs2 = xs2_n
                    branch_best = min(branch_best, F2)
                F += branch_best
                if F < best_F:
                    best_F, best_u, best_s, best_name = F, u, s, strat["name"]

    return best_u, best_s, best_name"""

replacement = """    proposals = []
    for strat in strategies:
        dop_w    = strat["dop_w"]
        danger_w = strat["d_w"] * ne_amp * grief_w * soc_mod
        strat_best_F, strat_best_u, strat_best_s = float('inf'), 0., 0.
        for u in REDUCED_ACTIONS:
            for s in STEER_ACTIONS:
                xs, F   = x.copy(), 0.
                xw, yw  = pos_2d
                heading = yaw
                # Leg 1
                for h in range(20):
                    xs_n     = Ad @ xs + Bd * u
                    conf     = max(alpha, 0.1) * (0.95**h)
                    heading -= s * STEER_GAIN * DT
                    spd      = xs_n[1] * DT * 25.
                    xw      += spd * math.cos(heading)
                    yw      += spd * math.sin(heading)
                    d2       = math.sqrt((xw-gx)**2 + (yw-gy)**2)
                    kin_d    = danger_at(D, xs_n[2], xs_n[3]) * danger_w
                    sp_d     = cognitive_map_danger(CM, xw, yw) * danger_w * 0.5
                    F       += (kin_d + sp_d + dop_w*d2*1.5
                               + abs(homeostatic.correction_signal(xs_n[2]))*0.1) * conf
                    xs = xs_n
                # Leg 2 — branch (straight steer only for speed; 5×10 rollout)
                branch_best = float('inf')
                for u2 in REDUCED_ACTIONS:
                    xs2, F2    = xs.copy(), 0.
                    xw2, yw2   = xw, yw
                    h2_heading = heading
                    for h2 in range(10):
                        xs2_n      = Ad @ xs2 + Bd * u2
                        conf2      = max(alpha, 0.1) * (0.95**(15+h2))
                        h2_heading -= 0 * STEER_GAIN * DT
                        spd2       = xs2_n[1] * DT * 25.
                        xw2       += spd2 * math.cos(h2_heading)
                        yw2       += spd2 * math.sin(h2_heading)
                        d3         = math.sqrt((xw2-gx)**2 + (yw2-gy)**2)
                        F2        += (danger_at(D,xs2_n[2],xs2_n[3])*danger_w
                                     + cognitive_map_danger(CM,xw2,yw2)*danger_w*0.5
                                     + dop_w*d3*1.5) * conf2
                        xs2 = xs2_n
                    branch_best = min(branch_best, F2)
                F += branch_best
                if F < strat_best_F:
                    strat_best_F, strat_best_u, strat_best_s = F, u, s
        proposals.append((strat_best_u, strat_best_s, strat["name"]))

    # Baseline evaluation of the 3 proposals
    base_dop_w = nm.effective_dopamine_weight(base=0.6)
    base_danger_w = 1.0 * ne_amp * grief_w * soc_mod
    best_F, best_u, best_s, best_name = float('inf'), 0., 0., "BALANCED"

    for (u, s, name) in proposals:
        xs, F   = x.copy(), 0.
        xw, yw  = pos_2d
        heading = yaw
        for h in range(20):
            xs_n     = Ad @ xs + Bd * u
            conf     = max(alpha, 0.1) * (0.95**h)
            heading -= s * STEER_GAIN * DT
            spd      = xs_n[1] * DT * 25.
            xw      += spd * math.cos(heading)
            yw      += spd * math.sin(heading)
            d2       = math.sqrt((xw-gx)**2 + (yw-gy)**2)
            kin_d    = danger_at(D, xs_n[2], xs_n[3]) * base_danger_w
            sp_d     = cognitive_map_danger(CM, xw, yw) * base_danger_w * 0.5
            F       += (kin_d + sp_d + base_dop_w*d2*1.5
                       + abs(homeostatic.correction_signal(xs_n[2]))*0.1) * conf
            xs = xs_n
        if F < best_F:
            best_F, best_u, best_s, best_name = F, u, s, name

    return best_u, best_s, best_name"""

content = content.replace(target, replacement)
with open('phase14_maze.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Patch applied.")
