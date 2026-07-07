# -*- coding: utf-8 -*-
# CARL Phase 15 â€” MuJoCo Edition
# Full biological brain + MuJoCo physics + milestone innovations incoming
import sys, time, math, numpy as np, asyncio, websockets, json, threading, random
from collections import deque
from carl_mj_physics import (BCFG, init_physics, launch_viewer, viewer_sync,
                              viewer_running, get_state, get_yaw, get_pos_2d,
                              apply_torque, reset_robot, park_robot,
                              check_wall_contact, apply_wind, clear_wind,
                              step_physics, forward_physics, get_lidar)
import carl_mj_physics as mj
from astar import astar_path
from carl_grid_cells import HippocampalNavigator
from carl_stdp import STDPActionEvaluator, QuantumDeliberator
from carl_omega_extensions import MirrorNeuronSystem, PredictiveAllostasis, ThetaGate
from carl_physarum import PhysarumMaze

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# â”€â”€ CONSTANTS â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
DT        = 1.0 / 240.0
ACTIONS   = [-8.,-5.,-2.,-1.,0.,1.,2.,5.,8.]
REDUCED_ACTIONS = [-8.,-3.,0.,3.,8.]
STEER_ACTIONS   = [-2.,0.,2.]
HORIZON   = 40
DRES      = 20
SDIM      = 5
P0_WM     = 500.0  * (SDIM+1)
P0_LTM    = 2000.0 * (SDIM+1)
N_BODIES  = 2
STEER_GAIN= 6.0
MAP_RES   = 25
MAP_XMIN,MAP_XMAX = -0.5, 7.7
MAP_YMIN,MAP_YMAX = -0.5, 6.5
MAZE_CELL = 1.2
MAZE_WALLS_GRID = [
    (0,0,6,0),(0,5,6,5),(0,0,0,5),(6,0,6,5),
    (1,0,1,1),(0,2,2,2),(1,3,1,4),(2,1,2,4),
    (3,0,3,2),(2,4,4,4),(4,1,4,3),(3,3,5,3),(5,1,6,1),
]
GOAL_POSITIONS=[(5.4,5.4),(1.8,5.4),(6.6,0.6),(4.2,1.8),(0.6,5.4)]
NM_BASELINE={"DA":0.5,"SHT":0.6,"NE":0.2,"ACh":0.4}
HOMEOSTATIC_SETPOINTS={"pitch":0.0,"velocity":0.0,"arousal":0.3,
                       "fatigue":0.0,"curiosity_drive":0.4,"social_comfort":0.7}

# â”€â”€ DASHBOARD STATE â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
brain={"episode":0,"best_survival":0.0,"mode":"BOOTING",
       "DA":0.5,"5HT":0.6,"NE":0.2,"ACh":0.4,
       "allostatic_load":0.0,"homeostatic_error":0.0,"social_comfort":1.0,
       "alive_count":N_BODIES,"hebbian_strength":0.0,"prediction_error":0.5,
       "sleep_phase":"AWAKE","nrem1_count":0,"nrem3_count":0,"rem_count":0,
       "survivals":[0.]*N_BODIES,"pitches":[0.]*N_BODIES,
       "distances":[99.]*N_BODIES,"alive_flags":[True]*N_BODIES,
       "goal_x":4.8,"goal_y":2.4,"goal_episode":0,
       "ghost_count":0,"mourning_events":0,"target_reached_count":0,
       "best_dist_ever":99.,"curriculum_stage":1,"junction_decisions":0,
       "danger_grid":[0.0]*400,"cognitive_map":[0.0]*625,
       "cm_scent":[0.0]*625,"cm_phero":[0.0]*625,
       "wind_active":False,"slope_deg":0.0,"quake_amp":0.0,
       "episode_history":[],"surprise_wm":0.0,"surprise_ltm":0.0,
       "ltm_confidence":0.0,"curiosity":0.5,"danger_level":0.0,
       "wm_confidence_A":0.0,"wm_confidence_B":0.0,
       "robot_x":[],"robot_y":[],"astar_path":[]}

async def _ws_handler(ws):
    try:
        while True:
            await ws.send(json.dumps(brain)); await asyncio.sleep(1/30)
    except: pass

def _run_ws():
    async def _s():
        print("[WS] ws://localhost:8765")
        async with websockets.serve(_ws_handler,"localhost",8765):
            await asyncio.Future()
    asyncio.run(_s())

threading.Thread(target=_run_ws,daemon=True).start()

# â”€â”€ MEMORY â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def fresh_wm():
    T=np.zeros((SDIM+1,SDIM)); T[:SDIM]=np.eye(SDIM)
    return T,500.0*np.eye(SDIM+1)

def fresh_ltm():
    T=np.zeros((SDIM+1,SDIM)); T[:SDIM]=np.eye(SDIM); P=np.eye(SDIM+1)*1000.0
    import os
    if os.path.exists("maze_ltm_T.npy"):
        try: T=np.load("maze_ltm_T.npy"); P=np.load("maze_ltm_P.npy"); print("  [LTM] Loaded checkpoint")
        except: pass
    return T,P

def rls_update(T,P,xk,uk,xn,lam,p_floor=0.001):
    Phi=np.append(xk,float(uk)).reshape(SDIM+1,1)
    e=xn-(T.T@Phi).flatten()
    PPhi=P@Phi; denom=lam+float((Phi.T@PPhi).squeeze())
    gain=PPhi/denom; T=T+gain@e.reshape(1,SDIM)
    P=(P-gain@(Phi.T@P))/lam; P=np.maximum(P,p_floor*np.eye(SDIM+1))
    return T,P,float(np.linalg.norm(e))

def wm_conf(P): return float(np.clip(1.0-np.trace(P)/P0_WM,0.,1.))
def ltm_conf(P): return float(np.clip(1.0-np.trace(P)/P0_LTM,0.,1.))

# â”€â”€ NEUROMODULATORS â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
class NeuromodulatorSystem:
    def __init__(self):
        self.DA=0.5; self.SHT=0.6; self.NE=0.2; self.ACh=0.4
    def update(self,surprise,danger,dist_to_target,target_reached,sibling_died,allostatic_load):
        prog=max(0.,1.-dist_to_target/8.)
        if target_reached: self.DA=min(1.,self.DA+0.3)
        else: self.DA=0.92*self.DA+0.08*(0.3+0.7*prog)
        self.DA=float(np.clip(self.DA,0.25,1.0))
        thr=max(danger,surprise*0.5)
        self.NE=(0.80*self.NE+0.20*thr) if thr>self.NE else (0.92*self.NE+0.08*thr)
        self.NE=float(np.clip(self.NE,0.,1.))
        self.SHT=0.990*self.SHT+0.010*max(0.,1.-danger)-self.NE*0.25*0.005
        self.SHT=float(np.clip(self.SHT,0.15,1.))
        if sibling_died: self.ACh=min(1.,self.ACh+0.2)
        self.ACh=0.99*self.ACh+0.01*min(1.,surprise*2.)
        self.ACh=float(np.clip(self.ACh,0.1,1.))
        self.SHT=max(0.15,self.SHT-allostatic_load*0.0005)
    def effective_learning_rate(self,base=0.990):
        return float(np.clip(base-(self.ACh-0.4)*0.008,0.970,0.999))
    def effective_horizon(self,base=40):
        return max(10,int(base*(0.5+self.SHT*1.0)))
    def effective_dopamine_weight(self,base=0.6):
        return float(np.clip(base*(0.5+self.DA),0.3,1.2))
    def effective_danger_sensitivity(self,base):
        return float(base*(1.0+self.NE*1.5))

# â”€â”€ HEBBIAN â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
class HebbianAssociator:
    def __init__(self,capacity=500):
        self.associations=deque(maxlen=capacity); self.strength_map={}
    def fire(self,pitch,vel,action,surprise,threshold=0.3):
        if surprise<threshold: return
        pb=int(np.clip((pitch+.65)/1.30*10,0,9))
        vb=int(np.clip((vel+3.)/6.*10,0,9))
        ab=int(np.clip((action+8.)/16.*5,0,4))
        key=(pb,vb,ab)
        self.strength_map[key]=min(1.0,self.strength_map.get(key,0.)+surprise*0.1)
    def wire_together_decay(self):
        for k in list(self.strength_map):
            self.strength_map[k]*=0.9999
            if self.strength_map[k]<0.01: del self.strength_map[k]
    def association_cost(self,pitch,vel,action):
        pb=int(np.clip((pitch+.65)/1.30*10,0,9))
        vb=int(np.clip((vel+3.)/6.*10,0,9))
        ab=int(np.clip((action+8.)/16.*5,0,4))
        return self.strength_map.get((pb,vb,ab),0.)*2.0
    def total_strength(self):
        if not self.strength_map: return 0.
        return float(np.mean(list(self.strength_map.values())))

# â”€â”€ PREDICTIVE CODER â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
class PredictiveCoder:
    def __init__(self):
        self.prediction=None; self.error_history=deque(maxlen=100)
        self.directional_bias=np.zeros(SDIM)
    def predict(self,x,u,T_wm,T_ltm,P_wm):
        alpha=wm_conf(P_wm); T_use=alpha*T_wm+(1.-alpha)*T_ltm
        Phi=np.append(x,float(u)).reshape(SDIM+1,1)
        self.prediction=(T_use.T@Phi).flatten(); return self.prediction
    def compute_error(self,x_actual):
        if self.prediction is None: return 0.,np.zeros(SDIM)
        ev=x_actual-self.prediction; m=float(np.linalg.norm(ev))
        self.error_history.append(m)
        self.directional_bias=0.99*self.directional_bias+0.01*np.abs(ev)
        return m,ev
    def precision_weight(self):
        if len(self.error_history)<10: return 0.5
        return float(np.clip(1./(1.+np.var(list(self.error_history))*10.),0.1,0.9))

# â”€â”€ HOMEOSTATIC REGULATOR â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
class HomeostaticRegulator:
    def __init__(self):
        self.variables={k:v for k,v in HOMEOSTATIC_SETPOINTS.items()}
        self.allostatic_load=0.0
    def update(self,pitch,velocity,nm_system,fatigue,alive_siblings,dist_to_target):
        self.variables.update({"pitch":float(pitch),"velocity":float(velocity),
            "arousal":float(nm_system.NE),"fatigue":float(fatigue),
            "curiosity_drive":float(nm_system.ACh),
            "social_comfort":float(alive_siblings/N_BODIES)})
        total_error=0.
        for key,sp in HOMEOSTATIC_SETPOINTS.items():
            if key=="velocity" and dist_to_target>0.3: sp=0.3
            total_error+=abs(self.variables[key]-sp)
        self.allostatic_load=min(1.,self.allostatic_load+total_error*0.0001)
        return total_error
    def correction_signal(self,pitch):
        return float(np.clip(-(pitch-HOMEOSTATIC_SETPOINTS["pitch"])*2.0,-2.,2.))

# â”€â”€ SOCIAL COGNITION â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
class SocialCognition:
    def __init__(self,idx,n): self.idx=idx; self.n=n; self.social_comfort=1.0
    def update(self,brains,my_x):
        alive=[(i,b) for i,b in enumerate(brains) if b["alive"] and i!=self.idx and b["xk"] is not None]
        if alive:
            self.social_comfort=float(np.clip(1.-np.mean([abs(b["xk"][0]-my_x) for _,b in alive])/5.,0.,1.))
        else: self.social_comfort=0.0
        return self.social_comfort
    def social_risk_modifier(self): return float(1.0+(self.social_comfort-0.5)*0.3)

# â”€â”€ COGNITIVE MAP â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def _map_cell(xw,yw):
    i=int(np.clip((xw-MAP_XMIN)/(MAP_XMAX-MAP_XMIN)*MAP_RES,0,MAP_RES-1))
    j=int(np.clip((yw-MAP_YMIN)/(MAP_YMAX-MAP_YMIN)*MAP_RES,0,MAP_RES-1))
    return i,j

def fresh_cognitive_map(): return np.zeros((MAP_RES,MAP_RES,5))

def diffuse_scent(CM,goal_pos):
    gi,gj=_map_cell(goal_pos[0],goal_pos[1])
    s=CM[:,:,3]; d=CM[:,:,0]
    ns=np.zeros_like(s)
    ns[1:-1,1:-1]=(s[:-2,1:-1]+s[2:,1:-1]+s[1:-1,:-2]+s[1:-1,2:])
    ns2=s.copy(); ns2[1:-1,1:-1]+=0.25*(ns[1:-1,1:-1]-4*s[1:-1,1:-1])
    ns2[d>0.5]=0.0; ns2*=0.999; ns2[gi,gj]=1.0
    CM[:,:,3]=ns2; CM[:,:,4]*=0.995
    return CM

def get_olfactory_reward(CM,xw,yw):
    xp=(xw-MAP_XMIN)/(MAP_XMAX-MAP_XMIN)*MAP_RES
    yp=(yw-MAP_YMIN)/(MAP_YMAX-MAP_YMIN)*MAP_RES
    x0=int(np.clip(math.floor(xp),0,MAP_RES-1)); x1=int(np.clip(math.ceil(xp),0,MAP_RES-1))
    y0=int(np.clip(math.floor(yp),0,MAP_RES-1)); y1=int(np.clip(math.ceil(yp),0,MAP_RES-1))
    dx=xp-x0; dy=yp-y0
    def bi(c):
        return (CM[x0,y0,c]*(1-dx)*(1-dy)+CM[x1,y0,c]*dx*(1-dy)+
                CM[x0,y1,c]*(1-dx)*dy+CM[x1,y1,c]*dx*dy)
    return bi(3)*2.0+bi(4)*0.5   # scaled to match danger cost range (~0-2.5)

def cognitive_map_update(CM,xw,yw,danger,trust_delta,ghost=0.):
    i,j=_map_cell(xw,yw)
    CM[i,j,0]=0.9*CM[i,j,0]+0.1*danger
    CM[i,j,1]=np.clip(CM[i,j,1]+trust_delta,0.,1.)
    if ghost>0.: CM[i,j,2]=min(1.,CM[i,j,2]+ghost)
    return CM

def cognitive_map_danger(CM,xw,yw):
    i,j=_map_cell(xw,yw); return float(CM[i,j,0])

def cognitive_map_trust(CM,xw,yw):
    i,j=_map_cell(xw,yw); return float(CM[i,j,1])

def fresh_danger(): return np.zeros((DRES,DRES))

def _cell(pitch,vel):
    i=int(np.clip((pitch+.65)/1.30*DRES,0,DRES-1))
    j=int(np.clip((vel+3.)/6.*DRES,0,DRES-1))
    return i,j

def danger_update(D,pitch,vel,surprise,rate=0.15):
    i,j=_cell(pitch,vel); D[i,j]=(1-rate)*D[i,j]+rate*surprise; return D

def danger_at(D,pitch,vel): return float(D[_cell(pitch,vel)])

def fresh_legacy_2d(): return np.zeros((MAP_RES,MAP_RES))

def legacy_write_2d(L2,xw,yw,intensity=1.0,goal_xw=None,goal_yw=None):
    if goal_xw is not None:
        dist=math.sqrt((xw-goal_xw)**2+(yw-goal_yw)**2)
        if dist<0.4: intensity*=0.15
    i,j=_map_cell(xw,yw)
    for di in [-1,0,1]:
        for dj in [-1,0,1]:
            ii=int(np.clip(i+di,0,MAP_RES-1)); jj=int(np.clip(j+dj,0,MAP_RES-1))
            w=1.0 if (di==0 and dj==0) else 0.4
            L2[ii,jj]=min(1.,L2[ii,jj]+intensity*w)
    return L2

def biological_sleep(T_ltm,P_ltm,D,buf,near_miss_buf,hebbian,allostatic_load):
    print("  [NREM-1]")
    if buf:
        for (xk_m,uk_m,xn_m,surp) in sorted(buf,key=lambda m:m[3],reverse=True)[:20]:
            T_ltm,P_ltm,_=rls_update(T_ltm,P_ltm,xk_m,uk_m,xn_m,lam=0.9999,p_floor=0.005)
    print("  [NREM-3]")
    if buf:
        for (xk_m,uk_m,xn_m,surp) in sorted(buf,key=lambda m:m[3],reverse=True)[:30]:
            for _ in range(3): D=danger_update(D,xn_m[2],xn_m[3],surp,rate=0.25)
    hebbian.wire_together_decay()
    print("  [REM]")
    if near_miss_buf:
        for (xk_m,uk_m,xn_m,surp) in sorted(near_miss_buf,key=lambda m:m[3])[:15]:
            D=danger_update(D,xn_m[2],xn_m[3],max(0.,surp*0.2),rate=0.05)
            T_ltm,P_ltm,_=rls_update(T_ltm,P_ltm,xk_m,uk_m,xn_m,lam=0.9999,p_floor=0.005)
    return T_ltm,P_ltm,D

# ── ACTION SELECTION ─────────────────────────────────────────────────────
def pick_action_maze(x, pos_2d, yaw, T_wm, T_ltm, P_wm, D, CM,
                     nm, homeostatic, swarm_hebb, hebbian, predictive,
                     social, goal_pos, grief=0., min_lidar=5.0):
    alpha = wm_conf(P_wm); T_use = alpha*T_wm + (1.-alpha)*T_ltm
    Ad, Bd = T_use[:SDIM,:].T, T_use[SDIM,:]
    dop_w = nm.effective_dopamine_weight(0.6)
    horizon = nm.effective_horizon(HORIZON)
    ne_amp = 1.0+nm.NE*1.5; grief_w = 1.0+grief*0.5
    soc_mod = social.social_risk_modifier()
    best_u, best_s, best_F = 0., 0., float('inf')
    best_traj, all_trajs = [], []
    for u in ACTIONS:
        for s in STEER_ACTIONS:
            xs, F = x.copy(), 0.; xw, yw = pos_2d; heading = yaw
            traj = [(xw, yw)]
            for h in range(horizon):
                xs_n = Ad@xs + Bd*u; conf = max(alpha,0.1)*(0.95**h)
                heading -= s*STEER_GAIN*DT; spd = xs_n[1]*DT*25.
                xw += spd*math.cos(heading); yw += spd*math.sin(heading)
                traj.append((xw, yw))
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
            all_trajs.append(traj)
            if F < best_F: best_F,best_u,best_s,best_traj = F,u,s,traj
    return best_u, best_s, all_trajs


def daughter_minds_maze(x, pos_2d, yaw, T_wm, T_ltm, P_wm, D, CM,
                        nm, homeostatic, swarm_hebb, hebbian,
                        predictive, social, goal_pos, grief, min_lidar=5.0):
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
    all_trajs = []
    for (u, s, name) in proposals:
        xs, F = x.copy(), 0.; xw, yw = pos_2d; heading = yaw
        traj = [(xw, yw)]
        for h in range(20):
            xs_n = Ad@xs+Bd*u; conf = max(alpha,0.1)*(0.95**h)
            heading -= s*STEER_GAIN*DT; spd = xs_n[1]*DT*25.
            xw += spd*math.cos(heading); yw += spd*math.sin(heading)
            traj.append((xw, yw))
            F += (danger_at(D,xs_n[2],xs_n[3])*bdw
                  +cognitive_map_danger(CM,xw,yw)*bdw*0.5
                  -bdpw*get_olfactory_reward(CM,xw,yw)
                  +abs(homeostatic.correction_signal(xs_n[2]))*0.1)*conf
            xs = xs_n
        bb = float('inf')
        best_traj_2 = []
        for u2 in ACTIONS:
            xs2, F2 = xs.copy(), 0.; xw2, yw2 = xw, yw; h2h = heading
            traj_2 = []
            for h2 in range(10):
                xs2_n = Ad@xs2+Bd*u2; conf2 = max(alpha,0.1)*(0.95**(15+h2))
                spd2 = xs2_n[1]*DT*25.
                xw2 += spd2*math.cos(h2h)
                yw2 += spd2*math.sin(h2h)
                traj_2.append((xw2, yw2))
                F2 += (danger_at(D,xs2_n[2],xs2_n[3])*bdw
                       +cognitive_map_danger(CM,xw2,yw2)*bdw*0.5
                       -bdpw*get_olfactory_reward(CM,xw2,yw2))*conf2
                xs2 = xs2_n
            if F2 < bb:
                bb = F2
                best_traj_2 = traj_2
        F += bb
        traj.extend(best_traj_2)
        all_trajs.append(traj)
        if F < best_F: best_F,best_u,best_s,best_name = F,u,s,name
    return best_u, best_s, best_name, all_trajs


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
            'energy': 100.0, 'sleeping': False, # ALife: Metabolism
            'nm': NeuromodulatorSystem(), 'homeostatic': HomeostaticRegulator(),
            'hebbian': HebbianAssociator(), 'predictive': PredictiveCoder(),
            'social': SocialCognition(idx, N_BODIES), 'mode': 'BALANCED',
            'was_at_target': False, 'pos_2d': (0., 0.), 'wall_time': 0,
            'astar_path': [], 'target_u': 0., 'target_s': 0., 'curio': 0.4,
            # ── CARL-Ω Biological Innovations ─────────────────
            'hippo':      HippocampalNavigator(),          # Grid + Place cells
            'stdp':       STDPActionEvaluator(),           # R-STDP plasticity
            'quantum':    QuantumDeliberator(),            # Quantum deliberation
            'mirror':     MirrorNeuronSystem(),            # Mirror neurons
            'pred_allo':  PredictiveAllostasis(),          # Predictive allostasis
            'theta':      ThetaGate(freq_hz=6.0),          # Theta oscillation gate
            'place_danger': np.zeros(200),                 # Place-cell danger memory
            'nav': {'uncertainty': 1.0, 'novelty': 1.0, 'theta': 0.5},
            }


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


def main():
    global CM_global, goal_pos, brains, pfc_active

    model, data = init_physics('carl_mujoco.xml')
    viewer = launch_viewer()

    T_ltm, P_ltm   = fresh_ltm()
    D_global        = fresh_danger()
    CM_global       = fresh_cognitive_map()
    # Pre-seed wall geometry so A* avoids walls from episode 1
    for (gx1,gy1,gx2,gy2) in MAZE_WALLS_GRID:
        x1,y1=gx1*MAZE_CELL,gy1*MAZE_CELL; x2,y2=gx2*MAZE_CELL,gy2*MAZE_CELL
        n_pts=max(2,int(math.sqrt((x2-x1)**2+(y2-y1)**2)/0.15))
        for k in range(n_pts+1):
            t=k/max(1,n_pts); xw=x1+t*(x2-x1); yw=y1+t*(y2-y1)
            for ddx in [-0.1,0.,0.1]:
                for ddy in [-0.1,0.,0.1]:
                    cognitive_map_update(CM_global,xw+ddx,yw+ddy,danger=0.9,trust_delta=0.)
    L2_legacy       = fresh_legacy_2d()
    swarm_hebb      = HebbianAssociator(capacity=2000)
    physarum        = PhysarumMaze(rows=MAP_RES, cols=MAP_RES)   # Slime mold navigator

    episode = 0; best = 0.; best_dist_ever = 99.
    target_reached_count = 0; ghost_count = 0
    mourning_events = 0; junction_decisions = 0
    nrem1 = nrem3 = rem_count = 0
    goal_idx = 0; goal_pos = GOAL_POSITIONS[goal_idx]

    import csv
    LOG = 'carl_mj_log.csv'
    with open(LOG,'w',newline='',encoding='utf-8') as f:
        csv.writer(f).writerow(['ep','best_sv','ep_max','reached','ghosts','rem','stage'])

    print('\n== Phase 17: ROBOT SCHOOL (Pre-Training) ==')
    print('  Biology DISABLED. Fast episodic learning ENABLED.')

    # Use only 1 robot for maximum simulation speed
    BCFG_SCHOOL = [BCFG[0]]
    brains = [spawn_brain(T_ltm, P_ltm, 0)]
    
    pfc_active = True
    pfc_threads = [threading.Thread(target=pfc_worker, args=(0,), daemon=True)]
    pfc_threads[0].start()

    episode = 1
    success_count = 0
    
    T_ltm_ep = T_ltm.copy()
    P_ltm_ep = P_ltm.copy()
    D = D_global.copy()
    CM = CM_global.copy()
    
    while viewer_running() and episode <= 300:
        print(f'\n--- EPISODE {episode} ---')
        reset_robot(BCFG_SCHOOL[0], 0.3, 0.3, 0.)
        forward_physics()
        
        brains[0]['xk'], _, _, _ = get_state(BCFG_SCHOOL[0])
        brains[0]['pos_2d'] = get_pos_2d(BCFG_SCHOOL[0])
        brains[0]['yaw'] = get_yaw(BCFG_SCHOOL[0])
        brains[0]['alive'] = True
        brains[0]['t0'] = time.time()
        brains[0]['wall_time'] = 0
        
        step = 0
        sibling_died_this_step = False
        alive_count = 1
        
        while viewer_running():
            step += 1
            if step % 200 == 0: CM_global = diffuse_scent(CM_global, goal_pos)
            
            rb = brains[0]
            cfg = BCFG_SCHOOL[0]

            xn, y_pos, z_pos, roll = get_state(cfg)
            xw, yw = get_pos_2d(cfg)
            rb['pos_2d'] = (xw, yw)
            rb['yaw'] = get_yaw(cfg)
            
            lidar_dists, hit_points = get_lidar(cfg)
            min_lidar = min(lidar_dists)
            
            # Visual SLAM: Map seen obstacles into the global Cognitive Map and Danger Grid
            for (hx, hy) in hit_points:
                gi = int(np.clip((hx - MAP_XMIN) / (MAP_XMAX - MAP_XMIN) * MAP_RES, 0, MAP_RES - 1))
                gj = int(np.clip((hy - MAP_YMIN) / (MAP_YMAX - MAP_YMIN) * MAP_RES, 0, MAP_RES - 1))
                CM_global[gi, gj, 0] = 1.0  # Max Danger (Wall)
                CM_global[gi, gj, 1] = 1.0  # Max Trust (We literally saw it)
                
                di = int(np.clip((hx - MAP_XMIN) / (MAP_XMAX - MAP_XMIN) * DRES, 0, DRES - 1))
                dj = int(np.clip((hy - MAP_YMIN) / (MAP_YMAX - MAP_YMIN) * DRES, 0, DRES - 1))
                D_global[di, dj] = 50.0

            # ALife Innovation: Visual Population Coding (15 neurons)
            prev_lidar = rb.get('prev_lidar', [5.0]*5)
            v_prox = [np.clip(1.0 - d/2.0, 0.0, 1.0) for d in lidar_dists] # Proximity (fires when close)
            v_void = [1.0 if d > 4.5 else 0.0 for d in lidar_dists]        # Void (fires down open corridors)
            v_loom = [np.clip((prev_lidar[j] - lidar_dists[j])*10.0, 0.0, 1.0) for j in range(5)] # Looming (Optical flow approach velocity)
            v_pop = np.array(v_prox + v_void + v_loom, dtype=float)
            rb['prev_lidar'] = lidar_dists
            rb['v_pop'] = [round(float(v), 3) for v in v_pop]

            nm = rb['nm']; homeostatic = rb['homeostatic']
            hebbian = rb['hebbian']; predictive = rb['predictive']
            social = rb['social']
            theta  = rb['theta'];  hippo  = rb['hippo']
            stdp   = rb['stdp'];   mirror = rb['mirror']
            pred_allo = rb['pred_allo']

            # ── INNOVATION 3: Theta gate ──────────────────
            theta.step(DT)
            theta_lam = theta.learning_lambda(0.990)

            _ = predictive.predict(rb['xk'], rb['uk'], rb['T_wm'], T_ltm_ep, rb['P_wm'])

            # ── Theta-gated RLS learning ──────────────────
            rb['T_wm'], rb['P_wm'], s_wm = rls_update(
                rb['T_wm'], rb['P_wm'], rb['xk'], rb['uk'], xn,
                lam=theta_lam, p_floor=0.001)
            T_ltm_ep, P_ltm_ep, s_ltm = rls_update(
                T_ltm_ep, P_ltm_ep, rb['xk'], rb['uk'], xn,
                lam=0.99995, p_floor=0.002)
            last_s_ltm = s_ltm

            pred_mag, _ = predictive.compute_error(xn)
            rb['s_wm_ema'] = 0.1*s_wm + 0.9*rb['s_wm_ema']
            rb['s_slow']   = 0.005*s_wm + 0.995*rb['s_slow']

            D  = danger_update(D, xn[2], xn[3], rb['s_wm_ema'])
            CM = cognitive_map_update(CM, xw, yw, rb['s_wm_ema'], +0.005)

            # ── INNOVATION 1: Grid cells + place cells ────
            vx = float(xn[1]); vy = 0.0   # use forward velocity
            nav = hippo.step(xw, yw, vx, vy, DT, learn=(theta.gate > 0.7))
            rb['nav'] = nav
            rb['place_danger'] = hippo.update_place_danger(
                rb['place_danger'], xw, yw, danger_at(D, xn[2], xn[3]))

            # ── INNOVATION 2: R-STDP update ───────────────
            stdp.update(rb['xk'], v_pop, rb['uk'], rb['s_wm_ema'], nm.DA)

            # ── INNOVATION 7: Predictive allostasis ───────
            if step % 20 == 0:
                pred_allo.predict_future_load(xn, T_ltm_ep, D, danger_at)
            pred_allo.update(homeostatic.allostatic_load)

            # ── INNOVATION 4: Physarum update (every 30 steps) ──
            if step % 30 == 0:
                gi_rob = int(np.clip((xw-MAP_XMIN)/(MAP_XMAX-MAP_XMIN)*MAP_RES, 0, MAP_RES-1))
                gj_rob = int(np.clip((yw-MAP_YMIN)/(MAP_YMAX-MAP_YMIN)*MAP_RES, 0, MAP_RES-1))
                gi_goal= int(np.clip((goal_pos[0]-MAP_XMIN)/(MAP_XMAX-MAP_XMIN)*MAP_RES, 0, MAP_RES-1))
                gj_goal= int(np.clip((goal_pos[1]-MAP_YMIN)/(MAP_YMAX-MAP_YMIN)*MAP_RES, 0, MAP_RES-1))
                physarum.step((gi_rob, gj_rob), (gi_goal, gj_goal),
                              CM_global[:,:,0], every=15)

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
            # INNOVATION 1: Novelty boosts curiosity via hippocampal signal
            novelty_boost = 1.0 + nav['novelty'] * 0.8 * theta.explore_boost(nm.ACh)
            curio = float(np.clip(base_curio*(1.-grief*0.5)*(1.+nm.ACh*0.3)*novelty_boost, 0.05, 1.))
            rb['curio'] = curio

            if _at_tgt and not _was_tgt:
                target_reached_count += 1
                brain['target_reached_count'] = target_reached_count
                nm.DA = min(1., nm.DA+0.4)
                print(f'  *** GOAL REACHED Body {i+1}! Total:{target_reached_count} ***')
            rb['was_at_target'] = _at_tgt

            if dist_now < best_dist_ever:
                best_dist_ever = dist_now; brain['best_dist_ever'] = round(best_dist_ever,3)

            # SCHOOL: Simple PID Controller
            # Balance + Drive Forward + Follow A* Path
            
            # 1. BALANCE: PID on pitch
            pitch_err = float(xn[2])
            pitch_vel = float(xn[3])
            balance_u = -(15.0 * pitch_err + 2.0 * pitch_vel)
            
            # 2. DRIVE FORWARD: Gentle, ramped forward bias
            # Start with zero bias, ramp up to 0.8 after the PID has stabilized
            ramp = float(np.clip(step / 500.0, 0.0, 1.0))
            forward_bias = 0.8 * ramp
            
            # 3. STEER: Follow A* path waypoint
            pfc_path = rb.get('astar_path', [])
            if pfc_path and len(pfc_path) > 1:
                dx = pfc_path[1][0] - xw
                dy = pfc_path[1][1] - yw
                ideal_yaw = math.atan2(dy, dx)
                yaw_err = (ideal_yaw - rb['yaw'] + math.pi) % (2*math.pi) - math.pi
                path_steer = float(np.clip(-yaw_err * 2.0, -2.5, 2.5))
            else:
                dx = goal_pos[0] - xw
                dy = goal_pos[1] - yw
                ideal_yaw = math.atan2(dy, dx)
                yaw_err = (ideal_yaw - rb['yaw'] + math.pi) % (2*math.pi) - math.pi
                path_steer = float(np.clip(-yaw_err * 2.0, -2.5, 2.5))
            
            # 4. LIDAR OVERRIDE: If wall is very close, override steer to dodge
            left_open = lidar_dists[0] + lidar_dists[1]*0.5
            right_open = lidar_dists[4] + lidar_dists[3]*0.5
            lidar_steer = float(np.clip((right_open - left_open) * 2.5, -3.0, 3.0))
            radar_dominance = float(np.clip(1.0 - (min_lidar / 0.6), 0.0, 1.0))
            
            if min_lidar < 0.3:
                forward_bias = -2.0  # Reverse!
            elif min_lidar < 0.5:
                forward_bias = 0.0   # Stop
            
            sn = (radar_dominance * lidar_steer) + ((1.0 - radar_dominance) * path_steer)
            un = float(np.clip(balance_u + forward_bias, -8., 8.))
            mode = 'SCHOOL-PID'
            apply_torque(cfg, un, sn)
            rb['xk'] = xn; rb['uk'] = un; rb['sk'] = sn
            rb['sv'] = time.time()-rb['t0']
            
            # Metabolism completely disabled for Robot School. No exhaustion.
            
            # Wall contact
            if check_wall_contact(cfg):
                rb['wall_time'] += 1
                nm.NE = min(1., nm.NE+0.15)
                # Pain/Fear stimulus for hitting walls (crucial for STDP)
                for _ in range(4): D = danger_update(D,xn[2],xn[3],10.,rate=0.3)
            else:
                rb['wall_time'] = 0

            # Collapse check — RESCUE instead of ending episode
            if z_pos < 0.065 or abs(xn[2]) > 0.50 or abs(roll) > 0.50:
                # Robot fell — rescue it at its current position
                rescue_x = float(max(0.3, min(6.5, xw)))
                rescue_y = float(max(0.3, min(5.5, yw)))
                reset_robot(cfg, rescue_x, rescue_y, 0.)
                forward_physics()
                rb['xk'], _, _, _ = get_state(cfg)
                rb['pos_2d'] = get_pos_2d(cfg)
                rb['yaw'] = get_yaw(cfg)
                rb['wall_time'] = 0
                print(f'  [RESCUE] Step {step}: Robot fell, rescued at ({rescue_x:.1f},{rescue_y:.1f})')
                # Pain stamp so it learns from the fall
                for _ in range(4): D = danger_update(D, xn[2], xn[3], 10., rate=0.3)
                continue
                
            # Wall stuck — rescue
            if rb['wall_time'] > 200:
                rescue_x = float(max(0.3, min(6.5, xw - 0.3*math.cos(rb['yaw']))))
                rescue_y = float(max(0.3, min(5.5, yw - 0.3*math.sin(rb['yaw']))))
                reset_robot(cfg, rescue_x, rescue_y, 0.)
                forward_physics()
                rb['xk'], _, _, _ = get_state(cfg)
                rb['pos_2d'] = get_pos_2d(cfg)
                rb['yaw'] = get_yaw(cfg)
                rb['wall_time'] = 0
                print(f'  [RESCUE] Step {step}: Wall stuck, backed up to ({rescue_x:.1f},{rescue_y:.1f})')
                continue
            
            # Timeout — end episode as failure
            if step > 12000:
                print(f'  [TIMEOUT] Episode {episode} timed out at step {step}.')
                break
                
            if _at_tgt:
                success_count += 1
                print(f'  [SUCCESS] Target reached in {step} steps! Streak: {success_count}/5')
                break

            CM_global[:] = CM[:]
            step_physics()
            viewer_sync()
            # Fast forward learning (no sleep)
            
            if not viewer_running():
                break
                
        # --- END OF INNER LOOP (End of episode) ---
        
        # Save the brain after the episode ends
        np.save('school_T_wm.npy', brains[0]['T_wm'])
        np.save('school_W_stdp.npy', brains[0]['stdp'].synapse.W)
        
        if success_count >= 5:
            print("\n*** GRADUATION: Robot has successfully crossed the maze 5 times in a row! ***")
            print("*** BRAIN SAVED TO school_T_wm.npy and school_W_stdp.npy ***")
            break
            
        episode += 1

if __name__ == '__main__':
    main()
