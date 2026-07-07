
# ── ACTION SELECTION ─────────────────────────────────────────
def pick_action_maze(x, pos_2d, yaw, T_wm, T_ltm, P_wm, D, CM,
                     nm, homeostatic, swarm_hebb, hebbian, predictive,
                     social, goal_pos, grief=0.):
    alpha = wm_conf(P_wm); T_use = alpha*T_wm + (1.-alpha)*T_ltm
    Ad, Bd = T_use[:SDIM,:].T, T_use[SDIM,:]
    dop_w = nm.effective_dopamine_weight(0.6)
    horizon = nm.effective_horizon(HORIZON)
    ne_amp = 1.0+nm.NE*1.5; grief_w = 1.0+grief*0.5
    soc_mod = social.social_risk_modifier()
    best_u, best_s, best_F = 0., 0., float('inf')
    for u in ACTIONS:
        for s in STEER_ACTIONS:
            xs, F = x.copy(), 0.; xw, yw = pos_2d; heading = yaw
            for h in range(horizon):
                xs_n = Ad@xs + Bd*u; conf = max(alpha,0.1)*(0.95**h)
                heading -= s*STEER_GAIN*DT; spd = xs_n[1]*DT*25.
                xw += spd*math.cos(heading); yw += spd*math.sin(heading)
                kin_d = danger_at(D,xs_n[2],xs_n[3])*ne_amp*grief_w
                sp_d  = cognitive_map_danger(CM,xw,yw)*ne_amp*0.5
                olf   = get_olfactory_reward(CM,xw,yw)*dop_w
                hc = (0.7*swarm_hebb.association_cost(xs_n[2],xs_n[3],u)
                     +0.3*hebbian.association_cost(xs_n[2],xs_n[3],u))
                hom = abs(homeostatic.correction_signal(xs_n[2]))*0.1
                F += (float(np.linalg.norm(xs_n-xs))*0.02
                      +(kin_d+sp_d)*soc_mod
                      +0.1*(1.-cognitive_map_trust(CM,xw,yw))
                      -olf+hc*0.1+hom)*conf
                xs = xs_n
            if F < best_F: best_F,best_u,best_s = F,u,s
    return best_u, best_s


def daughter_minds_maze(x, pos_2d, yaw, T_wm, T_ltm, P_wm, D, CM,
                        nm, homeostatic, swarm_hebb, hebbian,
                        predictive, social, goal_pos, grief):
    alpha = wm_conf(P_wm); T_use = alpha*T_wm+(1.-alpha)*T_ltm
    Ad, Bd = T_use[:SDIM,:].T, T_use[SDIM,:]
    ne_amp = 1.0+nm.NE*1.5; grief_w = 1.0+grief*0.5
    soc_mod = social.social_risk_modifier()
    strategies = [
        {'d_w': 2.0+nm.NE*1.5,           'dop_w': 0.0,                             'name': 'SAFE'},
        {'d_w': max(0.2,0.5-nm.DA*0.5),  'dop_w': nm.effective_dopamine_weight(1.0),'name': 'BOLD'},
        {'d_w': 1.0,                      'dop_w': nm.effective_dopamine_weight(0.6),'name': 'BALANCED'},
    ]
    proposals = []
    for strat in strategies:
        dw = strat['d_w']*ne_amp*grief_w*soc_mod; dpw = strat['dop_w']
        bf, bu, bs = float('inf'), 0., 0.
        for u in ACTIONS:
            for s in STEER_ACTIONS:
                xs, F = x.copy(), 0.; xw, yw = pos_2d; heading = yaw
                for h in range(20):
                    xs_n = Ad@xs+Bd*u; conf = max(alpha,0.1)*(0.95**h)
                    heading -= s*STEER_GAIN*DT; spd = xs_n[1]*DT*25.
                    xw += spd*math.cos(heading); yw += spd*math.sin(heading)
                    F += (danger_at(D,xs_n[2],xs_n[3])*dw
                          +cognitive_map_danger(CM,xw,yw)*dw*0.5
                          -dpw*get_olfactory_reward(CM,xw,yw)
                          +abs(homeostatic.correction_signal(xs_n[2]))*0.1)*conf
                    xs = xs_n
                bb = float('inf')
                for u2 in ACTIONS:
                    xs2, F2 = xs.copy(), 0.; xw2, yw2 = xw, yw; h2h = heading
                    for h2 in range(10):
                        xs2_n = Ad@xs2+Bd*u2; conf2 = max(alpha,0.1)*(0.95**(15+h2))
                        xw2 += xs2_n[1]*DT*25.*math.cos(h2h)
                        yw2 += xs2_n[1]*DT*25.*math.sin(h2h)
                        F2 += (danger_at(D,xs2_n[2],xs2_n[3])*dw
                               +cognitive_map_danger(CM,xw2,yw2)*dw*0.5
                               -dpw*get_olfactory_reward(CM,xw2,yw2))*conf2
                        xs2 = xs2_n
                    bb = min(bb, F2)
                F += bb
                if F < bf: bf,bu,bs = F,u,s
        proposals.append((bu, bs, strat['name']))
    bdw = 1.0*ne_amp*grief_w*soc_mod; bdpw = nm.effective_dopamine_weight(0.6)
    best_F, best_u, best_s, best_name = float('inf'), 0., 0., 'BALANCED'
    for (u, s, name) in proposals:
        xs, F = x.copy(), 0.; xw, yw = pos_2d; heading = yaw
        for h in range(20):
            xs_n = Ad@xs+Bd*u; conf = max(alpha,0.1)*(0.95**h)
            heading -= s*STEER_GAIN*DT; spd = xs_n[1]*DT*25.
            xw += spd*math.cos(heading); yw += spd*math.sin(heading)
            F += (danger_at(D,xs_n[2],xs_n[3])*bdw
                  +cognitive_map_danger(CM,xw,yw)*bdw*0.5
                  -bdpw*get_olfactory_reward(CM,xw,yw)
                  +abs(homeostatic.correction_signal(xs_n[2]))*0.1)*conf
            xs = xs_n
        bb = float('inf')
        for u2 in ACTIONS:
            xs2, F2 = xs.copy(), 0.; xw2, yw2 = xw, yw; h2h = heading
            for h2 in range(10):
                xs2_n = Ad@xs2+Bd*u2; conf2 = max(alpha,0.1)*(0.95**(15+h2))
                xw2 += xs2_n[1]*DT*25.*math.cos(h2h)
                yw2 += xs2_n[1]*DT*25.*math.sin(h2h)
                F2 += (danger_at(D,xs2_n[2],xs2_n[3])*bdw
                       +cognitive_map_danger(CM,xw2,yw2)*bdw*0.5
                       -bdpw*get_olfactory_reward(CM,xw2,yw2))*conf2
                xs2 = xs2_n
            bb = min(bb, F2)
        F += bb
        if F < best_F: best_F,best_u,best_s,best_name = F,u,s,name
    return best_u, best_s, best_name


def directed_explore_maze(x, pos_2d, yaw, T_wm, P_wm, D, CM, nm, goal_pos):
    Ad, Bd = T_wm[:SDIM,:].T, T_wm[SDIM,:]
    if float(np.linalg.norm(Bd)) < 0.1:
        return float(np.random.choice(ACTIONS)), 0.
    ach_amp = 1.+nm.ACh*0.5; gx, gy = goal_pos; xw, yw = pos_2d
    best_u, best_s, best_info = 0., 0., -float('inf')
    for u in REDUCED_ACTIONS:
        for s in STEER_ACTIONS:
            xs = x.copy(); xw2, yw2 = xw, yw; heading = yaw
            for _ in range(10):
                xs = Ad@xs+Bd*u; heading -= s*STEER_GAIN*DT
                xw2 += xs[1]*DT*25.*math.cos(heading)
                yw2 += xs[1]*DT*25.*math.sin(heading)
            Phi_r = np.append(xs, u).reshape(SDIM+1, 1)
            unc = float((Phi_r.T@P_wm@Phi_r).squeeze())
            unexplored = 1.-cognitive_map_trust(CM, xw2, yw2)
            goal_pull = max(0., 1.-math.sqrt((xw2-gx)**2+(yw2-gy)**2)/8.)
            info = unc*(1.+unexplored)*ach_amp + goal_pull*0.3
            if info > best_info: best_info,best_u,best_s = info,u,s
    return best_u, best_s


def spawn_brain(T_ltm, P_ltm, idx):
    T_inst, P_inst = fresh_wm()
    return {'T_wm': T_inst.copy(), 'P_wm': P_inst.copy()+0.1*np.eye(SDIM+1),
            'xk': None, 'uk': 0., 'sk': 0., 'yaw': 0.,
            's_wm_ema': 0., 's_slow': 0.01, 'fatigue': 0.,
            'buf': [], 'near_miss_buf': [], 't0': time.time(),
            'alive': True, 'sv': 0., 'cfg': BCFG[idx], 'grief': 0.,
            'nm': NeuromodulatorSystem(), 'homeostatic': HomeostaticRegulator(),
            'hebbian': HebbianAssociator(), 'predictive': PredictiveCoder(),
            'social': SocialCognition(idx, N_BODIES), 'mode': 'BALANCED',
            'was_at_target': False, 'pos_2d': (0., 0.), 'wall_time': 0,
            'astar_path': [], 'target_u': 0., 'target_s': 0., 'curio': 0.4}


CM_global = None
goal_pos  = (5.4, 5.4)
brains    = []
pfc_active = True


def pfc_worker(b_idx):
    global brains, CM_global, goal_pos
    while pfc_active:
        try:
            if not brains or b_idx >= len(brains): time.sleep(0.1); continue
            b = brains[b_idx]
            if not b['alive']: time.sleep(0.05); continue
            path = astar_path(CM_global, b['pos_2d'], goal_pos)
            b['astar_path'] = path
            if path and len(path) > 1:
                dx = path[1][0]-b['pos_2d'][0]; dy = path[1][1]-b['pos_2d'][1]
                ideal_yaw = math.atan2(dy, dx)
                yaw_err = (ideal_yaw-b['yaw']+math.pi) % (2*math.pi) - math.pi
                b['target_s'] = -1.5 if yaw_err > 0.2 else (1.5 if yaw_err < -0.2 else 0.)
                b['target_u'] = 3.; b['mode'] = 'AUTOPILOT'
            else:
                b['target_u'] = 0.; b['target_s'] = 0.; b['mode'] = 'SEARCHING'
        except Exception as e:
            print(f'[PFC-{b_idx}] {e}')
        time.sleep(0.05)
