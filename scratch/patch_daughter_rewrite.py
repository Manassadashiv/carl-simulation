import re

with open('phase14_maze.py', 'r', encoding='utf-8') as f:
    content = f.read()

correct_daughter_minds = """
def daughter_minds_maze(x, pos_2d, yaw, T_wm, T_ltm, P_wm, D, CM,
                        nm, homeostatic, swarm_hebbian, hebbian,
                        predictive, social, goal_pos, grief):
    alpha   = wm_conf(P_wm)
    T_use   = alpha * T_wm + (1. - alpha) * T_ltm
    Ad, Bd  = T_use[:SDIM,:].T, T_use[SDIM,:]
    ne_amp  = 1.0 + nm.NE * 1.5
    grief_w = 1.0 + grief * 0.5
    soc_mod = social.social_risk_modifier()
    gx, gy  = goal_pos
    strategies = [
        {"d_w": 2.0 + nm.NE*1.5,         "dop_w": 0.0,                            "name": "SAFE"},
        {"d_w": max(0.2,0.5-nm.DA*0.5),   "dop_w": nm.effective_dopamine_weight(1.0), "name": "BOLD"},
        {"d_w": 1.0,                       "dop_w": nm.effective_dopamine_weight(0.6), "name": "BALANCED"},
    ]
    
    proposals = []
    
    for strat in strategies:
        dop_w    = strat["dop_w"]
        danger_w = strat["d_w"] * ne_amp * grief_w * soc_mod
        strat_best_F, strat_best_u, strat_best_s = float('inf'), 0., 0.
        
        for u in ACTIONS:
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
                    olf1 = get_olfactory_reward(CM, xw, yw)
                    kin_d    = danger_at(D, xs_n[2], xs_n[3]) * danger_w
                    sp_d     = cognitive_map_danger(CM, xw, yw) * danger_w * 0.5
                    F       += (kin_d + sp_d - dop_w*olf1
                               + abs(homeostatic.correction_signal(xs_n[2]))*0.1) * conf
                    xs = xs_n
                # Leg 2
                branch_best = float('inf')
                for u2 in ACTIONS:
                    xs2, F2    = xs.copy(), 0.
                    xw2, yw2   = xw, yw
                    h2_heading = heading
                    for h2 in range(10):
                        xs2_n      = Ad @ xs2 + Bd * u2
                        conf2      = max(alpha, 0.1) * (0.95**(15+h2))
                        spd2       = xs2_n[1] * DT * 25.
                        xw2       += spd2 * math.cos(h2_heading)
                        yw2       += spd2 * math.sin(h2_heading)
                        olf2 = get_olfactory_reward(CM, xw2, yw2)
                        F2        += (danger_at(D,xs2_n[2],xs2_n[3])*danger_w
                                     + cognitive_map_danger(CM,xw2,yw2)*danger_w*0.5
                                     - dop_w*olf2) * conf2
                        xs2 = xs2_n
                    branch_best = min(branch_best, F2)
                F += branch_best
                if F < strat_best_F:
                    strat_best_F, strat_best_u, strat_best_s = F, u, s
        proposals.append((strat_best_u, strat_best_s, strat["name"]))

    # BASELINE EVALUATION TO PREVENT LAZY 'SAFE' FROM ALWAYS WINNING
    base_dop_w = nm.effective_dopamine_weight(base=0.6)
    base_danger_w = 1.0 * ne_amp * grief_w * soc_mod
    best_F, best_u, best_s, best_name = float('inf'), 0., 0., "BALANCED"

    for (u, s, name) in proposals:
        xs, F   = x.copy(), 0.
        xw, yw  = pos_2d
        heading = yaw
        # Full 30 step evaluation of the proposed u and s
        # Wait, the proposal only gives u and s for Leg 1.
        # We must re-evaluate Leg 1, and then assume optimal Leg 2 under baseline.
        # To keep it fast, we just evaluate Leg 1 using baseline.
        for h in range(20):
            xs_n     = Ad @ xs + Bd * u
            conf     = max(alpha, 0.1) * (0.95**h)
            heading -= s * STEER_GAIN * DT
            spd      = xs_n[1] * DT * 25.
            xw      += spd * math.cos(heading)
            yw      += spd * math.sin(heading)
            olf1 = get_olfactory_reward(CM, xw, yw)
            kin_d    = danger_at(D, xs_n[2], xs_n[3]) * base_danger_w
            sp_d     = cognitive_map_danger(CM, xw, yw) * base_danger_w * 0.5
            F       += (kin_d + sp_d - base_dop_w*olf1
                       + abs(homeostatic.correction_signal(xs_n[2]))*0.1) * conf
            xs = xs_n
            
        # Add a quick Leg 2 baseline
        branch_best = float('inf')
        for u2 in ACTIONS:
            xs2, F2 = xs.copy(), 0.
            xw2, yw2 = xw, yw
            h2_heading = heading
            for h2 in range(10):
                xs2_n = Ad @ xs2 + Bd * u2
                conf2 = max(alpha, 0.1) * (0.95**(15+h2))
                spd2 = xs2_n[1] * DT * 25.
                xw2 += spd2 * math.cos(h2_heading)
                yw2 += spd2 * math.sin(h2_heading)
                olf2 = get_olfactory_reward(CM, xw2, yw2)
                F2 += (danger_at(D,xs2_n[2],xs2_n[3])*base_danger_w
                       + cognitive_map_danger(CM,xw2,yw2)*base_danger_w*0.5
                       - base_dop_w*olf2) * conf2
                xs2 = xs2_n
            branch_best = min(branch_best, F2)
        F += branch_best

        if F < best_F:
            best_F, best_u, best_s, best_name = F, u, s, name

    return best_u, best_s, best_name
"""

# We need to replace the entire daughter_minds_maze function.
start_idx = content.find("def daughter_minds_maze")
end_idx = content.find("def directed_explore_maze", start_idx)

if start_idx != -1 and end_idx != -1:
    content = content[:start_idx] + correct_daughter_minds + "\n\n" + content[end_idx:]
    with open('phase14_maze.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("daughter_minds_maze totally replaced!")
else:
    print("Could not find function bounds!")
