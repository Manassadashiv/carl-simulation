

def main():
    global CM_global, goal_pos, brains, pfc_active

    model, data = init_physics('carl_mujoco.xml')
    viewer = launch_viewer()

    T_ltm, P_ltm   = fresh_ltm()
    D_global        = fresh_danger()
    CM_global       = fresh_cognitive_map()
    L2_legacy       = fresh_legacy_2d()
    swarm_hebb      = HebbianAssociator(capacity=2000)

    episode = 0; best = 0.; best_dist_ever = 99.
    target_reached_count = 0; ghost_count = 0
    mourning_events = 0; junction_decisions = 0
    nrem1 = nrem3 = rem_count = 0
    goal_idx = 0; goal_pos = GOAL_POSITIONS[goal_idx]

    import csv
    LOG = 'carl_mj_log.csv'
    with open(LOG,'w',newline='',encoding='utf-8') as f:
        csv.writer(f).writerow(['ep','best_sv','ep_max','reached','ghosts','rem','stage'])

    print('\n== CARL Phase 15 — MuJoCo Edition ==')

    while viewer_running():
        episode += 1
        brain['episode'] = episode

        if episode % 200 == 1 and episode > 1:
            goal_idx = (goal_idx+1) % len(GOAL_POSITIONS)
            goal_pos = GOAL_POSITIONS[goal_idx]
            print(f'\n  *** GOAL MOVED to {goal_pos} ***\n')

        stage = 1
        if best > 15.:  stage = 2
        if best > 35.:  stage = 3
        if best > 60.:  stage = 4
        if best > 100.: stage = 5
        brain['curriculum_stage'] = stage

        init_pitch = float(np.random.uniform(-0.10,0.10)) if stage >= 2 else 0.

        # Reset simulation state for new episode
        for i, cfg in enumerate(BCFG):
            col = i % 4; row = i // 4
            x_off = 0.3 + col*0.25; y_off = 0.3 + row*0.25
            reset_robot(cfg, x_off, y_off, init_pitch)
        forward_physics()

        # Diffuse scent from goal
        CM_global = diffuse_scent(CM_global, goal_pos)

        brains = [spawn_brain(T_ltm, P_ltm, i) for i in range(N_BODIES)]
        for b, cfg in zip(brains, BCFG):
            b['xk'], _, _, _ = get_state(cfg)
            b['pos_2d'] = get_pos_2d(cfg)
            b['yaw'] = get_yaw(cfg)

        pfc_active = True
        pfc_threads = []
        for i in range(N_BODIES):
            t = threading.Thread(target=pfc_worker, args=(i,), daemon=True)
            t.start(); pfc_threads.append(t)
        print(f'  [PFC] {N_BODIES} threads launched')

        D = D_global.copy(); CM = CM_global.copy()
        P_ltm_ep = P_ltm.copy(); T_ltm_ep = T_ltm.copy()
        ep_max_sv = 0.0; next_wind = np.random.randint(20*240, 40*240)

        lc = ltm_conf(P_ltm_ep)
        print(f'[Ep {episode:3d}] Stage:{stage}  LTM:{lc*100:.0f}%  Best:{best:.1f}s  Goal:{goal_pos}')

        for step in range(500_000):
            if not viewer_running(): break
            sibling_died_this_step = False
            dead_this_step = []

            # Grief decay
            for b in brains:
                if b['alive'] and b.get('grief',0.) > 0.:
                    b['grief'] *= 0.9998

            if step % 240 == 0:
                swarm_hebb.wire_together_decay()
                CM = diffuse_scent(CM, goal_pos)

            # Wind (stage 3+)
            wind_active = False
            if stage >= 3 and step >= next_wind:
                fx = float(np.random.uniform(-2.5, 2.5))
                for b, cfg in zip(brains, BCFG):
                    if b['alive']: apply_wind(cfg, fx)
                next_wind = step + np.random.randint(20*240, 40*240)
                wind_active = True
            else:
                for cfg in BCFG: clear_wind(cfg)

            alive_count = sum(1 for b in brains if b['alive'])
            last_s_ltm = 0.0

            for i, (rb, cfg) in enumerate(zip(brains, BCFG)):
                if not rb['alive']: continue

                xn, y_pos, z_pos, roll = get_state(cfg)
                xw, yw = get_pos_2d(cfg)
                rb['pos_2d'] = (xw, yw)
                rb['yaw'] = get_yaw(cfg)

                nm = rb['nm']; homeostatic = rb['homeostatic']
                hebbian = rb['hebbian']; predictive = rb['predictive']
                social = rb['social']

                _ = predictive.predict(rb['xk'], rb['uk'], rb['T_wm'], T_ltm_ep, rb['P_wm'])

                ach_lam = nm.effective_learning_rate(0.990)
                rb['T_wm'], rb['P_wm'], s_wm = rls_update(
                    rb['T_wm'], rb['P_wm'], rb['xk'], rb['uk'], xn,
                    lam=ach_lam, p_floor=0.001)
                T_ltm_ep, P_ltm_ep, s_ltm = rls_update(
                    T_ltm_ep, P_ltm_ep, rb['xk'], rb['uk'], xn,
                    lam=0.99995, p_floor=0.002)
                last_s_ltm = s_ltm

                pred_mag, _ = predictive.compute_error(xn)
                rb['s_wm_ema'] = 0.1*s_wm + 0.9*rb['s_wm_ema']
                rb['s_slow']   = 0.005*s_wm + 0.995*rb['s_slow']

                D  = danger_update(D, xn[2], xn[3], rb['s_wm_ema'])
                CM = cognitive_map_update(CM, xw, yw, rb['s_wm_ema'], +0.005)

                dist_now = math.sqrt((xw-goal_pos[0])**2+(yw-goal_pos[1])**2)

                _at_tgt  = dist_now < 0.35
                _was_tgt = rb.get('was_at_target', False)
                nm.update(rb['s_wm_ema'], danger_at(D,xn[2],xn[3]),
                          dist_now, _at_tgt and not _was_tgt,
                          sibling_died_this_step, homeostatic.allostatic_load)

                h_error = homeostatic.update(xn[2], xn[1], nm, rb['fatigue'],
                                             alive_count-1, dist_now)
                rb['last_h_error'] = h_error

                social.update(brains, xw)
                swarm_hebb.fire(xn[2], xn[3], rb['uk'], rb['s_wm_ema'])
                hebbian.fire(xn[2], xn[3], rb['uk'], rb['s_wm_ema'])

                danger_here = nm.effective_danger_sensitivity(danger_at(D,xn[2],xn[3]))
                if danger_here > 0.20 or s_wm > 0.25:
                    rb['buf'].append((rb['xk'].copy(), rb['uk'], xn.copy(), s_wm))
                    if len(rb['buf']) > 200: rb['buf'].pop(0)
                if dist_now < 1.5:
                    rb['near_miss_buf'].append((rb['xk'].copy(), rb['uk'], xn.copy(), s_wm))
                    if len(rb['near_miss_buf']) > 100: rb['near_miss_buf'].pop(0)

                rb['fatigue'] = 0.999*rb['fatigue'] + 0.001*abs(rb['uk'])
                base_curio = float(np.clip(np.trace(rb['P_wm'])/P0_WM+0.08, 0., 1.))
                grief = rb.get('grief', 0.)
                curio = float(np.clip(base_curio*(1.-grief*0.5)*(1.+nm.ACh*0.3), 0.05, 1.))
                rb['curio'] = curio

                if _at_tgt and not _was_tgt:
                    target_reached_count += 1
                    brain['target_reached_count'] = target_reached_count
                    nm.DA = min(1., nm.DA+0.4)
                    print(f'  *** GOAL REACHED Body {i+1}! Total:{target_reached_count} ***')
                rb['was_at_target'] = _at_tgt

                if dist_now < best_dist_ever:
                    best_dist_ever = dist_now; brain['best_dist_ever'] = round(best_dist_ever,3)

                # Action selection
                dl = danger_here*(1.+grief*0.5)*social.social_risk_modifier()
                pfc_path  = rb.get('astar_path', [])
                pfc_steer = rb.get('target_s', 0.)

                if pfc_path:
                    Ad, Bd = rb['T_wm'][:SDIM,:].T, rb['T_wm'][SDIM,:]
                    best_u, best_cost = 0., float('inf')
                    for test_u in [-8.,-5.,-3.,0.,3.,5.,8.]:
                        xs_sim = xn.copy(); cost = 0.
                        for _ in range(15):
                            xs_sim = Ad@xs_sim + Bd*test_u
                            cost += (xs_sim[2]**2)*8.0
                            if abs(xs_sim[2]) > 0.38: cost += 500.; break
                        if cost < best_cost: best_cost = cost; best_u = test_u
                    un = best_u; sn = pfc_steer; mode = 'AUTOPILOT'
                    junction_decisions += 1; brain['junction_decisions'] = junction_decisions
                elif dl > 0.5:
                    un, sn, d_name = daughter_minds_maze(
                        xn,(xw,yw),rb['yaw'],rb['T_wm'],T_ltm_ep,rb['P_wm'],
                        D,CM,nm,homeostatic,swarm_hebb,hebbian,predictive,
                        social,goal_pos,grief)
                    mode = f'DELIBERATING-{d_name}'
                    junction_decisions += 1; brain['junction_decisions'] = junction_decisions
                elif curio > np.random.random() and dl < 0.4:
                    un, sn = directed_explore_maze(xn,(xw,yw),rb['yaw'],
                        rb['T_wm'],rb['P_wm'],D,CM,nm,goal_pos)
                    mode = 'CURIOUS'
                else:
                    un, sn = pick_action_maze(xn,(xw,yw),rb['yaw'],
                        rb['T_wm'],T_ltm_ep,rb['P_wm'],D,CM,nm,
                        homeostatic,swarm_hebb,hebbian,predictive,social,goal_pos,grief)
                    mode = ('FEARFUL' if dl>0.4 else 'GRIEVING' if grief>0.3 else 'EXPLORING')

                rb['mode'] = mode
                apply_torque(cfg, un, sn)
                rb['xk'] = xn; rb['uk'] = un; rb['sk'] = sn
                rb['sv'] = time.time()-rb['t0']
                ep_max_sv = max(ep_max_sv, rb['sv'])

                # Wall contact
                if check_wall_contact(cfg):
                    rb['wall_time'] = rb.get('wall_time',0)+1
                    nm.NE = min(1., nm.NE+0.15)
                    CM = cognitive_map_update(CM,xw,yw,0.8,-0.05)
                    D  = danger_update(D,xn[2],xn[3],0.6,rate=0.2)
                else:
                    rb['wall_time'] = 0

                # Death check
                if z_pos < 0.065 or abs(xn[2]) > 0.40 or abs(roll) > 0.40 or rb.get('wall_time',0) > 1200:
                    sv = rb['sv']; best = max(best, sv)
                    brain['episode_history'].append(round(sv,2))
                    brain['best_survival'] = round(best,2)
                    rb['alive'] = False
                    sibling_died_this_step = True
                    dead_this_step.append(rb)

                    for _ in range(4): D = danger_update(D,xn[2],xn[3],10.,rate=0.3)
                    CM = cognitive_map_update(CM,xw,yw,1.0,-0.2)

                    if dist_now < 0.8:
                        intensity = 1.-(dist_now/0.8); mourning_events += 1
                        for b in brains:
                            if b['alive']:
                                b['grief'] = min(1.,b.get('grief',0.)+intensity*0.5)
                                b['nm'].NE = min(1.,b['nm'].NE+intensity*0.3)
                                b['nm'].ACh = min(1.,b['nm'].ACh+0.2)

                    gi = min(1., 0.3+sv/60.)
                    L2_legacy = legacy_write_2d(L2_legacy,xw,yw,gi,goal_pos[0],goal_pos[1])
                    CM = cognitive_map_update(CM,xw,yw,0.,0.,ghost=gi*0.5)
                    ghost_count += 1
                    park_robot(cfg)
                    print(f'  Body {i+1} died {sv:.2f}s | {mode} | ({xw:.1f},{yw:.1f})')

            # Sleep on death
            if dead_this_step:
                all_trauma = []; all_near = []
                for b in dead_this_step:
                    all_trauma.extend(b['buf']); all_near.extend(b['near_miss_buf'])
                max_allo = max(b['homeostatic'].allostatic_load for b in dead_this_step)
                brain['sleep_phase'] = 'NREM-1'
                T_ltm_ep, P_ltm_ep, D = biological_sleep(
                    T_ltm_ep,P_ltm_ep,D,all_trauma,all_near,swarm_hebb,max_allo)
                for b in dead_this_step: b['homeostatic'].allostatic_load *= 0.98
                nrem1 += 1; nrem3 += 1; rem_count += 1
                brain['sleep_phase'] = 'AWAKE'
                brain['nrem1_count']=nrem1; brain['nrem3_count']=nrem3; brain['rem_count']=rem_count

            if sibling_died_this_step:
                for b in brains:
                    if b['alive']:
                        b['nm'].NE = min(1.0, b['nm'].NE+0.3)
                        b['nm'].DA = max(0.0, b['nm'].DA-0.1)

            CM_global[:] = CM[:]

            # Dashboard update
            xA  = brains[0]['xk'] if brains[0]['xk'] is not None else np.zeros(SDIM)
            nm0 = brains[0]['nm']; h0 = brains[0]['homeostatic']
            alive_modes  = [b['mode'] for b in brains if b['alive']]
            mode_display = alive_modes[0] if alive_modes else 'DEAD'
            avg_allo = float(np.mean([b['homeostatic'].allostatic_load for b in brains]))
            dmax = float(np.max(D))+1e-6
            cm_danger_flat = np.round(CM[:,:,0].T.flatten()/(float(np.max(CM[:,:,0]))+1e-6),3).tolist()
            brain.update({
                'survivals':     [round(b['sv'],2) for b in brains],
                'pitches':       [round(float(b['xk'][2]) if b['xk'] is not None else 0.,3) for b in brains],
                'distances':     [round(float(math.sqrt((b['pos_2d'][0]-goal_pos[0])**2+(b['pos_2d'][1]-goal_pos[1])**2)) if b['xk'] is not None else 99.,2) for b in brains],
                'alive_flags':   [b['alive'] for b in brains],
                'best_survival': round(best,2),
                'mode':          mode_display,
                'surprise_wm':   float(brains[0]['s_wm_ema']),
                'surprise_ltm':  float(last_s_ltm),
                'ltm_confidence':float(ltm_conf(P_ltm_ep)),
                'wm_confidence_A':float(wm_conf(brains[0]['P_wm'])),
                'wm_confidence_B':float(wm_conf(brains[1]['P_wm'])),
                'curiosity':     float(brains[0].get('curio',0.4)),
                'danger_level':  float(danger_at(D,xA[2],xA[3])) if brains[0]['alive'] else 0.,
                'wind_active':   wind_active,
                'quake_amp':     0.,
                'danger_grid':   (D/dmax).flatten().round(3).tolist(),
                'cognitive_map': cm_danger_flat,
                'cm_scent':      np.round(CM[:,:,3].T.flatten(),3).tolist(),
                'cm_phero':      np.round(CM[:,:,4].T.flatten(),3).tolist(),
                'curriculum_stage':stage,
                'DA':  round(nm0.DA,3),  '5HT': round(nm0.SHT,3),
                'NE':  round(nm0.NE,3),  'ACh': round(nm0.ACh,3),
                'allostatic_load':   round(avg_allo,4),
                'homeostatic_error': round(brains[0].get('last_h_error',0.) if brains[0]['alive'] else 0.,3),
                'sleep_phase':   brain['sleep_phase'],
                'social_comfort':round(float(np.mean([b['social'].social_comfort for b in brains if b['alive']])) if any(b['alive'] for b in brains) else 0.,3),
                'alive_count':   alive_count,
                'hebbian_strength':  round(swarm_hebb.total_strength(),3),
                'prediction_error':  round(float(brains[0]['predictive'].precision_weight()) if brains[0]['alive'] else 0.,3),
                'goal_x': goal_pos[0], 'goal_y': goal_pos[1],
                'ghost_count': ghost_count, 'mourning_events': mourning_events,
                'target_reached_count': target_reached_count,
                'best_dist_ever': round(best_dist_ever,3),
                'junction_decisions': junction_decisions,
                'robot_x': [b['pos_2d'][0] for b in brains if b['xk'] is not None],
                'robot_y': [b['pos_2d'][1] for b in brains if b['xk'] is not None],
                'astar_path': brains[0].get('astar_path',[]),
            })

            step_physics()
            viewer_sync()
            time.sleep(DT)

            if not any(b['alive'] for b in brains):
                pfc_active = False
                T_ltm, P_ltm = T_ltm_ep.copy(), P_ltm_ep.copy()
                D_global[:] = D; CM_global[:] = CM

                if episode % 50 == 0:
                    np.save('mj_ltm_T.npy', T_ltm); np.save('mj_ltm_P.npy', P_ltm)
                    np.save('mj_danger.npy', D_global); np.save('mj_cm.npy', CM_global)
                    np.save('mj_legacy.npy', L2_legacy); print(f'  [CHECKPOINT ep {episode}]')

                allo_avg = float(np.mean([b['homeostatic'].allostatic_load for b in brains]))
                print(f'  Ep {episode} done. Best:{best:.2f}s  EpMax:{ep_max_sv:.1f}s  Reached:{target_reached_count}x')

                with open(LOG,'a',newline='',encoding='utf-8') as f:
                    csv.writer(f).writerow([episode,round(best,2),round(ep_max_sv,2),
                                            target_reached_count,ghost_count,rem_count,stage])
                pfc_active = True
                time.sleep(0.5)
                break


if __name__ == '__main__':
    import sys
    # Append part2 contents to this file at runtime (parts merged)
    import importlib.util, os
    p2 = os.path.join(os.path.dirname(__file__), 'phase15_part2.py')
    if os.path.exists(p2):
        with open(p2, encoding='utf-8') as f:
            exec(f.read(), globals())
    main()
