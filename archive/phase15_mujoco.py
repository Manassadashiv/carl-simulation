# -*- coding: utf-8 -*-
# CARL Phase 15 — MuJoCo Edition
# Full biological brain + MuJoCo physics + milestone innovations incoming
import sys, time, math, numpy as np, asyncio, websockets, json, threading, random
from collections import deque
from carl_mj_physics import (BCFG, init_physics, launch_viewer, viewer_sync,
                              viewer_running, get_state, get_yaw, get_pos_2d,
                              apply_torque, reset_robot, park_robot,
                              check_wall_contact, apply_wind, clear_wind,
                              step_physics, forward_physics)
from astar import astar_path

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ── CONSTANTS ─────────────────────────────────────────────────
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

# ── DASHBOARD STATE ───────────────────────────────────────────
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

# ── MEMORY ────────────────────────────────────────────────────
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

# ── NEUROMODULATORS ───────────────────────────────────────────
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

# ── HEBBIAN ───────────────────────────────────────────────────
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

# ── PREDICTIVE CODER ──────────────────────────────────────────
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

# ── HOMEOSTATIC REGULATOR ─────────────────────────────────────
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

# ── SOCIAL COGNITION ──────────────────────────────────────────
class SocialCognition:
    def __init__(self,idx,n): self.idx=idx; self.n=n; self.social_comfort=1.0
    def update(self,brains,my_x):
        alive=[(i,b) for i,b in enumerate(brains) if b["alive"] and i!=self.idx and b["xk"] is not None]
        if alive:
            self.social_comfort=float(np.clip(1.-np.mean([abs(b["xk"][0]-my_x) for _,b in alive])/5.,0.,1.))
        else: self.social_comfort=0.0
        return self.social_comfort
    def social_risk_modifier(self): return float(1.0+(self.social_comfort-0.5)*0.3)

# ── COGNITIVE MAP ─────────────────────────────────────────────
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
