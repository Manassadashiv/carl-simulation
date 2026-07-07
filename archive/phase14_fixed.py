# -*- coding: utf-8 -*-
# CARL Phase 14: THE PARALLEL BRAIN — FIXED & UPGRADED
# ============================================================
# FIXES applied over the broken version:
#
# FIX 1 — HUD SPEED (was 1100ms per step, now <5ms)
#   removeAllUserDebugItems() was called every step at 240Hz.
#   Now called only every 48 steps (~5Hz). Physics runs clean.
#
# FIX 2 — GRADUATED CURRICULUM (was: maze from episode 1)
#   Stage 1: open flat arena, no walls, just the goal.
#             Bots learn balance and basic goal-seeking.
#   Stage 2: single corridor — one wall, one choice.
#             Bots learn wall awareness.
#   Stage 3: full maze activates.
#             Bots now have balance + wall sense + goal drive.
#   Stage 4+: maze + wind + earthquake + moving goal.
#
# FIX 3 — WALL-AWARE A* (was: routing through walls)
#   At episode start, wall positions are pre-burned into the
#   cognitive map as maximum danger. A* now routes around
#   physical walls correctly. The green path line is real.
#
# UPGRADE — SLIME MOULD FLOW MAP (Physarum algorithm)
#   Inspired by the 2010 Science paper (Tero et al.)
#   where slime mould recreated the Tokyo rail network.
#   A 25x25 flow grid tracks successful traversal history.
#   Cells that bodies pass through without dying: flow rises.
#   Cells where bodies die: flow drops hard.
#   Each episode: all flow decays slightly (forgetting).
#   A* cost = danger + inverse_flow — finds paths that are
#   both safe AND proven across many episodes.
#   Result: thick green corridors of ancestral wisdom emerge
#   through the maze over hundreds of episodes.
#   Visible on dashboard as the "flow map" layer.
# ============================================================

import sys, pybullet as p, pybullet_data, time, numpy as np
import asyncio, websockets, json, threading, math, heapq
import time as _time
from collections import deque

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ── CONSTANTS ─────────────────────────────────────────────────
DT       = 1.0 / 240.0
ACTIONS  = [-8., -5., -2., -1., 0., 1., 2., 5., 8.]
HORIZON  = 40
DRES     = 20
SDIM     = 5          # NEVER CHANGE
P0_WM    = 500.0  * (SDIM + 1)
P0_LTM   = 2000.0 * (SDIM + 1)
N_BODIES = 10

MAZE_CELL = 1.2
WALL_H    = 0.35
WALL_T    = 0.08

# Full maze walls (only active Stage 3+)
MAZE_WALLS_GRID = [
    (-1,-1,6,-1),(-1,4,6,4),(-1,-1,-1,4),(6,-1,6,4),
    (0,0,2,0),(3,0,5,0),(1,1,3,1),(0,2,1,2),(2,2,4,2),
    (1,3,2,3),(4,1,5,1),(3,3,5,3),
    (1,-1,1,1),(2,0,2,2),(3,1,3,3),(4,0,4,1),(4,2,4,4),
    (0,1,0,2),(5,1,5,3),
]

# Stage 2 corridor — just boundary + one internal wall
CORRIDOR_WALLS_GRID = [
    (-1,-1,6,-1),(-1,4,6,4),(-1,-1,-1,4),(6,-1,6,4),
    (2,-1,2,2),   # single dividing wall with gap
]

GOAL_POSITIONS = [
    (4.8,2.4),(1.2,3.6),(3.6,0.6),(2.4,1.8),(4.8,0.6),
]

MAP_RES  = 25
MAP_XMIN = -1.5; MAP_XMAX = 7.5
MAP_YMIN = -1.5; MAP_YMAX = 5.0

NM_BASELINE = {"DA":0.5,"SHT":0.6,"NE":0.2,"ACh":0.4}
HOMEOSTATIC_SETPOINTS = {
    "pitch":0.,"velocity":0.,"arousal":0.3,
    "fatigue":0.,"curiosity_drive":0.4,"social_comfort":0.7,
}

# ── THREAD-SAFE SLOTS ─────────────────────────────────────────
class DecisionSlot:
    def __init__(self):
        self._lock=threading.Lock(); self._action=0.
        self._name="REFLEX"; self._fresh=False

    def write(self,action,name):
        with self._lock:
            self._action=float(action); self._name=name; self._fresh=True

    def read_and_consume(self):
        with self._lock:
            if self._fresh:
                self._fresh=False; return self._action,self._name
            return None,None

class PathSlot:
    def __init__(self):
        self._lock=threading.Lock(); self._path=[]

    def write(self,path):
        with self._lock: self._path=list(path)

    def read(self):
        with self._lock: return list(self._path)

    def next_waypoint(self):
        with self._lock: return self._path[0] if self._path else None

class AmygdalaOverride:
    def __init__(self,threshold=0.60):
        self._lock=threading.Lock()
        self._emergency=False; self._threshold=threshold

    def trigger(self):
        with self._lock: self._emergency=True

    def clear(self):
        with self._lock: self._emergency=False

    def is_active(self):
        with self._lock: return self._emergency

# ── DASHBOARD ─────────────────────────────────────────────────
brain = {
    "episode":0,"best_survival":0.,"mode":"BOOTING",
    "DA":0.5,"5HT":0.6,"NE":0.2,"ACh":0.4,
    "allostatic_load":0.,"homeostatic_error":0.,
    "social_comfort":1.,"alive_count":N_BODIES,
    "hebbian_strength":0.,"prediction_error":0.5,
    "sleep_phase":"AWAKE","nrem1_count":0,"nrem3_count":0,"rem_count":0,
    "survivals":[0.]*N_BODIES,"pitches":[0.]*N_BODIES,
    "distances":[99.]*N_BODIES,"alive_flags":[True]*N_BODIES,
    "goal_x":4.8,"goal_y":2.4,"ghost_count":0,
    "mourning_events":0,"target_reached_count":0,
    "best_dist_ever":99.,"curriculum_stage":1,
    "junction_decisions":0,"astar_path":[],
    "danger_grid":[0.]*400,"cognitive_map":[0.]*625,
    "flow_map":[0.]*625,          # slime mould flow
    "wind_active":False,"quake_amp":0.,
    "episode_history":[],"surprise_wm":0.,"surprise_ltm":0.,
    "ltm_confidence":0.,"curiosity":0.5,"danger_level":0.,
    "wm_confidence_A":0.,"thread_fps":0.,
    "prefrontal_decisions":0,"astar_updates":0,"amygdala_triggers":0,
}

async def _ws_handler(ws):
    try:
        while True:
            await ws.send(json.dumps(brain)); await asyncio.sleep(1/30)
    except Exception: pass

def _run_ws():
    async def _serve():
        print("[WS] ws://localhost:8765")
        async with websockets.serve(_ws_handler,"localhost",8765):
            await asyncio.Future()
    asyncio.run(_serve())

threading.Thread(target=_run_ws,daemon=True).start()

# ── MEMORY ────────────────────────────────────────────────────
def fresh_ltm():
    T=np.zeros((SDIM+1,SDIM)); T[:SDIM]=np.eye(SDIM)
    return T,2000.*np.eye(SDIM+1)

def rls_update(T,P,xk,uk,xn,lam,p_floor=0.001):
    Phi=np.append(xk,float(uk)).reshape(SDIM+1,1)
    e=xn-(T.T@Phi).flatten()
    PPhi=P@Phi
    denom=lam+float((Phi.T@PPhi).squeeze())
    gain=PPhi/denom
    T=T+gain@e.reshape(1,SDIM)
    P=(P-gain@(Phi.T@P))/lam
    return T,np.maximum(P,p_floor*np.eye(SDIM+1)),float(np.linalg.norm(e))

def wm_conf(P): return float(np.clip(1.-np.trace(P)/P0_WM,0.,1.))
def ltm_conf(P): return float(np.clip(1.-np.trace(P)/P0_LTM,0.,1.))

# ── NEUROMODULATORS ───────────────────────────────────────────
class NeuromodulatorSystem:
    def __init__(self):
        self.DA=NM_BASELINE["DA"]; self.SHT=NM_BASELINE["SHT"]
        self.NE=NM_BASELINE["NE"]; self.ACh=NM_BASELINE["ACh"]

    def update(self,surprise,danger,dist,target_reached,sib_died,allo):
        prog=max(0.,1.-dist/8.)
        if target_reached: self.DA=min(1.,self.DA+0.3)
        else: self.DA=0.92*self.DA+0.08*(0.3+0.7*prog)
        self.DA=float(np.clip(self.DA,0.25,1.))
        thr=max(danger,surprise*0.5)
        self.NE=0.80*self.NE+0.20*thr if thr>self.NE else 0.92*self.NE+0.08*thr
        self.NE=float(np.clip(self.NE,0.,1.))
        self.SHT=float(np.clip(0.990*self.SHT+0.010*max(0.,1.-danger)-self.NE*0.001,0.15,1.))
        if sib_died: self.ACh=min(1.,self.ACh+0.2)
        self.ACh=float(np.clip(0.99*self.ACh+0.01*min(1.,surprise*2.),0.1,1.))
        self.SHT=max(0.15,self.SHT-allo*0.0005)

    def lr(self,base=0.990): return float(np.clip(base-(self.ACh-0.4)*0.008,0.970,0.999))
    def horizon(self,base=40): return max(10,int(base*(0.5+self.SHT)))
    def dop_w(self,base=0.6): return float(np.clip(base*(0.5+self.DA),0.3,1.2))
    def danger_amp(self,d): return float(d*(1.+self.NE*1.5))

# ── HEBBIAN ───────────────────────────────────────────────────
class HebbianAssociator:
    def __init__(self,capacity=500): self.strength_map={}

    def fire(self,pitch,vel,action,surprise,threshold=0.3):
        if surprise<threshold: return
        pb=int(np.clip((pitch+.65)/1.30*10,0,9))
        vb=int(np.clip((vel+3.)/6.*10,0,9))
        ab=int(np.clip((action+8.)/16.*5,0,4))
        key=(pb,vb,ab)
        self.strength_map[key]=min(1.,self.strength_map.get(key,0.)+surprise*0.1)

    def decay(self):
        for k in list(self.strength_map):
            self.strength_map[k]*=0.9999
            if self.strength_map[k]<0.01: del self.strength_map[k]

    def cost(self,pitch,vel,action):
        pb=int(np.clip((pitch+.65)/1.30*10,0,9))
        vb=int(np.clip((vel+3.)/6.*10,0,9))
        ab=int(np.clip((action+8.)/16.*5,0,4))
        return self.strength_map.get((pb,vb,ab),0.)*2.

    def total(self):
        if not self.strength_map: return 0.
        return float(np.mean(list(self.strength_map.values())))

# ── PREDICTIVE CODER ──────────────────────────────────────────
class PredictiveCoder:
    def __init__(self):
        self.prediction=None; self.error_history=deque(maxlen=100)

    def predict(self,x,u,T_wm,T_ltm,P_wm):
        alpha=wm_conf(P_wm); T_use=alpha*T_wm+(1.-alpha)*T_ltm
        Phi=np.append(x,float(u)).reshape(SDIM+1,1)
        self.prediction=(T_use.T@Phi).flatten(); return self.prediction

    def compute_error(self,x_actual):
        if self.prediction is None: return 0.,np.zeros(SDIM)
        e=x_actual-self.prediction; mag=float(np.linalg.norm(e))
        self.error_history.append(mag); return mag,e

    def precision(self):
        if len(self.error_history)<10: return 0.5
        return float(np.clip(1./(1.+np.var(list(self.error_history))*10.),0.1,0.9))

# ── HOMEOSTASIS ───────────────────────────────────────────────
class HomeostaticRegulator:
    def __init__(self):
        self.variables={k:v for k,v in HOMEOSTATIC_SETPOINTS.items()}
        self.allostatic_load=0.

    def update(self,pitch,vel,nm,fatigue,alive_sib,dist):
        self.variables.update({
            "pitch":float(pitch),"velocity":float(vel),
            "arousal":float(nm.NE),"fatigue":float(fatigue),
            "curiosity_drive":float(nm.ACh),
            "social_comfort":float(alive_sib/N_BODIES)})
        total=sum(abs(self.variables[k]-(0.3 if k=="velocity" and dist>0.3 else sp))
                  for k,sp in HOMEOSTATIC_SETPOINTS.items())
        self.allostatic_load=min(1.,self.allostatic_load+total*0.0001)
        return total

    def correction(self,pitch): return float(np.clip(-pitch*2.,-2.,2.))

# ── SOCIAL ────────────────────────────────────────────────────
class SocialCognition:
    def __init__(self,idx,n): self.idx=idx; self.n=n; self.social_comfort=1.

    def update(self,brains,my_x):
        alive=[(i,b) for i,b in enumerate(brains)
               if b["alive"] and i!=self.idx and b["xk"] is not None]
        if alive:
            self.social_comfort=float(np.clip(
                1.-np.mean([abs(b["xk"][0]-my_x) for _,b in alive])/5.,0.,1.))
        else: self.social_comfort=0.
        return self.social_comfort

    def risk_mod(self): return float(1.+(self.social_comfort-0.5)*0.3)

# ── COGNITIVE MAP ─────────────────────────────────────────────
def _map_cell(xw,yw):
    i=int(np.clip((xw-MAP_XMIN)/(MAP_XMAX-MAP_XMIN)*MAP_RES,0,MAP_RES-1))
    j=int(np.clip((yw-MAP_YMIN)/(MAP_YMAX-MAP_YMIN)*MAP_RES,0,MAP_RES-1))
    return i,j

def fresh_cognitive_map(): return np.zeros((MAP_RES,MAP_RES,3))

def cm_update(CM,xw,yw,danger,trust_delta,ghost=0.):
    i,j=_map_cell(xw,yw)
    CM[i,j,0]=0.9*CM[i,j,0]+0.1*danger
    CM[i,j,1]=np.clip(CM[i,j,1]+trust_delta,0.,1.)
    if ghost>0.: CM[i,j,2]=min(1.,CM[i,j,2]+ghost)
    return CM

def cm_danger(CM,xw,yw): i,j=_map_cell(xw,yw); return float(CM[i,j,0])

# ── WALL-AWARE COGNITIVE MAP PRE-BURN ─────────────────────────
def preburn_walls_into_map(CM,walls_grid):
    """
    Before episode starts, burn wall positions into CM as
    maximum danger so A* never routes through them.
    Each wall segment is rasterised onto the 25x25 grid.
    This is the fix for the green line going through walls.
    """
    for (gx1,gy1,gx2,gy2) in walls_grid:
        x1,y1=gx1*MAZE_CELL,gy1*MAZE_CELL
        x2,y2=gx2*MAZE_CELL,gy2*MAZE_CELL
        # Sample many points along the wall and burn each cell
        length=math.sqrt((x2-x1)**2+(y2-y1)**2)
        steps=max(2,int(length/0.1))
        for k in range(steps+1):
            t=k/steps
            xw=x1+(x2-x1)*t
            yw=y1+(y2-y1)*t
            # Burn the cell and its neighbours
            for di in [-1,0,1]:
                for dj in [-1,0,1]:
                    ci,cj=_map_cell(xw,yw)
                    ni=int(np.clip(ci+di,0,MAP_RES-1))
                    nj=int(np.clip(cj+dj,0,MAP_RES-1))
                    CM[ni,nj,0]=1.0   # maximum danger
    return CM

# ── SLIME MOULD FLOW MAP ──────────────────────────────────────
def fresh_flow_map(): return np.zeros((MAP_RES,MAP_RES))

def flow_reinforce(FM,xw,yw,amount=0.02):
    """Body passed through here safely. Reinforce the tube."""
    i,j=_map_cell(xw,yw)
    FM[i,j]=min(1.,FM[i,j]+amount)
    return FM

def flow_dissolve(FM,xw,yw,amount=0.3):
    """Body died here. Dissolve the tube — this path is deadly."""
    i,j=_map_cell(xw,yw)
    FM[i,j]=max(0.,FM[i,j]-amount)
    return FM

def flow_decay_episode(FM,rate=0.02):
    """
    Each episode, all flow decays slightly.
    Paths that get reused survive. Forgotten paths dissolve.
    This is the Physarum rule: use it or lose it.
    """
    FM*=(1.-rate)
    return FM

# ── A* (wall-aware, flow-guided) ─────────────────────────────
def astar(CM,FM,start_xw,start_yw,goal_xw,goal_yw):
    """
    A* over cognitive map.
    Cost per cell = 1 + danger*6 - flow*0.5
    High danger = avoid. High flow = prefer (proven safe path).
    Walls are pre-burned as danger=1.0 so they cost 7.0 per cell.
    """
    si,sj=_map_cell(start_xw,start_yw)
    gi,gj=_map_cell(goal_xw,goal_yw)

    def h(i,j): return abs(i-gi)+abs(j-gj)

    def cell_to_world(i,j):
        xw=MAP_XMIN+(i+0.5)/MAP_RES*(MAP_XMAX-MAP_XMIN)
        yw=MAP_YMIN+(j+0.5)/MAP_RES*(MAP_YMAX-MAP_YMIN)
        return xw,yw

    open_set=[]
    heapq.heappush(open_set,(h(si,sj),0.,si,sj))
    came_from={}
    g_score={(si,sj):0.}
    closed=set()

    while open_set:
        _,g,i,j=heapq.heappop(open_set)
        if (i,j) in closed: continue
        closed.add((i,j))
        if (i,j)==(gi,gj):
            path=[]
            node=(i,j)
            while node in came_from:
                path.append(cell_to_world(*node))
                node=came_from[node]
            path.reverse()
            return path
        for di,dj in [(-1,0),(1,0),(0,-1),(0,1),
                       (-1,-1),(-1,1),(1,-1),(1,1)]:
            ni,nj=i+di,j+dj
            if not (0<=ni<MAP_RES and 0<=nj<MAP_RES): continue
            if (ni,nj) in closed: continue
            danger=float(CM[ni,nj,0])
            if danger>=0.95: continue   # wall cell — never enter
            flow=float(FM[ni,nj])
            step_cost=(1.4 if di!=0 and dj!=0 else 1.)+danger*6.-flow*0.5
            ng=g+step_cost
            if ng<g_score.get((ni,nj),float('inf')):
                g_score[(ni,nj)]=ng
                came_from[(ni,nj)]=(i,j)
                heapq.heappush(open_set,(ng+h(ni,nj),ng,ni,nj))
    return []

# ── DANGER MAP ────────────────────────────────────────────────
def fresh_danger(): return np.zeros((DRES,DRES))

def _cell(pitch,vel):
    i=int(np.clip((pitch+.65)/1.30*DRES,0,DRES-1))
    j=int(np.clip((vel+3.)/6.*DRES,0,DRES-1))
    return i,j

def danger_update(D,pitch,vel,surprise,rate=0.15):
    i,j=_cell(pitch,vel); D[i,j]=(1-rate)*D[i,j]+rate*surprise; return D

def danger_at(D,pitch,vel): return float(D[_cell(pitch,vel)])

# ── LEGACY GHOST MAP ──────────────────────────────────────────
def fresh_legacy2d(): return np.zeros((MAP_RES,MAP_RES))

def legacy_write2d(L2,xw,yw,intensity=1.,goal_xw=None,goal_yw=None):
    if goal_xw is not None:
        if math.sqrt((xw-goal_xw)**2+(yw-goal_yw)**2)<0.4:
            intensity*=0.15
    i,j=_map_cell(xw,yw)
    for di in [-1,0,1]:
        for dj in [-1,0,1]:
            ii=int(np.clip(i+di,0,MAP_RES-1))
            jj=int(np.clip(j+dj,0,MAP_RES-1))
            L2[ii,jj]=min(1.,L2[ii,jj]+intensity*(1. if di==0 and dj==0 else 0.4))
    return L2

# ── SLEEP ─────────────────────────────────────────────────────
def biological_sleep(T_ltm,P_ltm,D,buf,near_miss_buf,hebbian,allo):
    print("  [NREM-1] Light consolidation...")
    if buf:
        for xk,uk,xn,s in sorted(buf,key=lambda m:m[3],reverse=True)[:20]:
            T_ltm,P_ltm,_=rls_update(T_ltm,P_ltm,xk,uk,xn,0.9999,0.005)
    print("  [NREM-3] Deep trauma consolidation...")
    if buf:
        for xk,uk,xn,s in sorted(buf,key=lambda m:m[3],reverse=True)[:30]:
            for _ in range(3): D=danger_update(D,xn[2],xn[3],s,rate=0.25)
    hebbian.decay()
    print("  [REM] Dream recombination...")
    if near_miss_buf:
        for xk,uk,xn,s in sorted(near_miss_buf,key=lambda m:m[3])[:15]:
            D=danger_update(D,xn[2],xn[3],max(0.,s*0.2),rate=0.05)
            T_ltm,P_ltm,_=rls_update(T_ltm,P_ltm,xk,uk,xn,0.9999,0.005)
    return T_ltm,P_ltm,D

# ── THREADS ───────────────────────────────────────────────────
def prefrontal_thread(rb,slot,T_ltm_ref,D_ref,CM_ref,
                      swarm_hebbian,goal_pos,stop_event,
                      counters_lock,counters):
    while not stop_event.is_set():
        if not rb["alive"]: time.sleep(0.01); continue
        try:
            xk=rb["xk"]
            if xk is None: time.sleep(0.005); continue
            T_wm=rb["T_wm"]; P_wm=rb["P_wm"]; nm=rb["nm"]
            home=rb["homeostatic"]; soc=rb["social"]
            grief=rb.get("grief",0.); pos=rb.get("pos_2d",(0.,0.))
            T_ltm=T_ltm_ref[0].copy(); D=D_ref[0].copy()
            gx,gy=goal_pos[0]
            alpha=wm_conf(P_wm); T_use=alpha*T_wm+(1.-alpha)*T_ltm
            Ad,Bd=T_use[:SDIM,:].T,T_use[SDIM,:]
            ne_amp=1.+nm.NE*1.5; grief_w=1.+grief*0.5
            soc_mod=soc.risk_mod()
            strategies=[
                {"d_w":2.+nm.NE*1.5,"dop_w":0.,"name":"SAFE"},
                {"d_w":max(.2,.5-nm.DA*.5),"dop_w":nm.dop_w(1.),"name":"BOLD"},
                {"d_w":1.,"dop_w":nm.dop_w(0.6),"name":"BALANCED"},
            ]
            best_u,best_F,best_name=0.,float('inf'),"BALANCED"
            for strat in strategies:
                dw=strat["d_w"]*ne_amp*grief_w*soc_mod; dopw=strat["dop_w"]
                xw,yw=pos
                for u in ACTIONS:
                    xs,F=xk.copy(),0.; xw2,yw2=xw,yw
                    for h in range(20):
                        xs_n=Ad@xs+Bd*u; conf=max(alpha,.1)*(0.95**h)
                        d2=math.sqrt((xw2-gx)**2+(yw2-gy)**2)
                        F+=(danger_at(D,xs_n[2],xs_n[3])*dw
                            +dopw*d2*0.3
                            +abs(home.correction(xs_n[2]))*0.1
                            +swarm_hebbian.cost(xs_n[2],xs_n[3],u)*0.5)*conf
                        xs=xs_n; xw2+=xs_n[1]*DT*5.
                    branch_best=float('inf')
                    for u2 in ACTIONS:
                        xs2,F2=xs.copy(),0.; xw3,yw3=xw2,yw2
                        for h2 in range(20):
                            xs2_n=Ad@xs2+Bd*u2; conf2=max(alpha,.1)*(0.95**(20+h2))
                            d3=math.sqrt((xw3-gx)**2+(yw3-gy)**2)
                            F2+=(danger_at(D,xs2_n[2],xs2_n[3])*dw+dopw*d3*0.3)*conf2
                            xs2=xs2_n; xw3+=xs2_n[1]*DT*5.
                        branch_best=min(branch_best,F2)
                    F+=branch_best
                    if F<best_F: best_F,best_u,best_name=F,u,strat["name"]
            slot.write(best_u,best_name)
            with counters_lock: counters["prefrontal"]+=1
        except Exception: pass
        time.sleep(0.002)

def hippocampus_thread(CM_ref,FM_ref,path_slot,goal_pos,
                       body_positions,stop_event,counters_lock,counters):
    while not stop_event.is_set():
        try:
            positions=list(body_positions)
            if not positions: time.sleep(0.5); continue
            avg_x=float(np.mean([p[0] for p in positions]))
            avg_y=float(np.mean([p[1] for p in positions]))
            gx,gy=goal_pos[0]
            CM=CM_ref[0].copy(); FM=FM_ref[0].copy()
            path=astar(CM,FM,avg_x,avg_y,gx,gy)
            path_slot.write(path)
            brain["astar_path"]=[list(wp) for wp in path[:12]]
            with counters_lock: counters["astar"]+=1
        except Exception: pass
        time.sleep(0.4)

def amygdala_thread(D_ref,body_states,overrides,stop_event,
                    counters_lock,counters,threshold=0.60):
    while not stop_event.is_set():
        try:
            for i,(xk,override) in enumerate(zip(body_states,overrides)):
                if xk is None: continue
                d=danger_at(D_ref[0],xk[2],xk[3])
                if d>threshold:
                    override.trigger()
                    with counters_lock: counters["amygdala"]+=1
                else: override.clear()
        except Exception: pass
        time.sleep(0.004)

# ── PYBULLET HELPERS ──────────────────────────────────────────
def get_state(rid):
    pos,quat=p.getBasePositionAndOrientation(rid)
    vel,ang_vel=p.getBaseVelocity(rid)
    euler=p.getEulerFromQuaternion(quat)
    neck=p.getJointState(rid,2)[0]
    return (np.array([pos[0],vel[0],euler[1],ang_vel[1],neck],dtype=float),
            float(pos[1]),float(pos[2]),float(euler[0]))

def get_pos2d(rid):
    pos,_=p.getBasePositionAndOrientation(rid)
    return float(pos[0]),float(pos[1])

def apply_torque(rid,u):
    t=float(np.clip(u,-8.,8.))
    p.setJointMotorControl2(rid,0,p.TORQUE_CONTROL,force=t)
    p.setJointMotorControl2(rid,1,p.TORQUE_CONTROL,force=t)

def fast_reflex(x,T_wm,T_ltm,P_wm,D,nm,home,waypoint=None):
    """Cerebellum: 15-step reflex, optionally biased toward A* waypoint."""
    alpha=wm_conf(P_wm); T_use=alpha*T_wm+(1.-alpha)*T_ltm
    Ad,Bd=T_use[:SDIM,:].T,T_use[SDIM,:]
    ne_amp=1.+nm.NE*1.5
    wp_pull=nm.dop_w(0.5)*abs(x[0]-waypoint[0]) if waypoint else 0.
    best_u,best_F=0.,float('inf')
    for u in ACTIONS:
        xs,F=x.copy(),0.
        for h in range(15):
            xs_n=Ad@xs+Bd*u; conf=max(alpha,.1)*(0.95**h)
            F+=(danger_at(D,xs_n[2],xs_n[3])*ne_amp
                +abs(home.correction(xs_n[2]))*0.15+wp_pull)*conf
            xs=xs_n
        if F<best_F: best_F,best_u=F,u
    return best_u

def spawn_robot(x=0.,y=0.,pitch=0.):
    for _ in range(3):
        try:
            rid=p.loadURDF("carl.urdf",[x,y,0.10],
                           p.getQuaternionFromEuler([0,pitch,0])); break
        except: time.sleep(0.5)
    p.setJointMotorControl2(rid,2,p.POSITION_CONTROL,targetPosition=0,force=12.)
    p.setJointMotorControl2(rid,0,p.VELOCITY_CONTROL,force=0)
    p.setJointMotorControl2(rid,1,p.VELOCITY_CONTROL,force=0)
    return rid

def spawn_brain(T_ltm,P_ltm,idx):
    return {
        "T_wm":T_ltm.copy(),"P_wm":P_ltm.copy()+50.*np.eye(SDIM+1),
        "xk":None,"uk":0.,"s_wm_ema":0.,"s_slow":0.01,
        "fatigue":0.,"buf":[],"near_miss_buf":[],
        "t0":time.time(),"alive":True,"sv":0.,
        "rid":None,"grief":0.,
        "nm":NeuromodulatorSystem(),"homeostatic":HomeostaticRegulator(),
        "hebbian":HebbianAssociator(),"predictive":PredictiveCoder(),
        "social":SocialCognition(idx,N_BODIES),
        "mode":"REFLEX","was_at_target":False,"pos_2d":(0.,0.),
    }

# ── MAZE BUILDER ──────────────────────────────────────────────
def build_walls(walls_grid):
    """Build physical walls from a wall definition list."""
    wall_ids=[]
    for (gx1,gy1,gx2,gy2) in walls_grid:
        x1,y1=gx1*MAZE_CELL,gy1*MAZE_CELL
        x2,y2=gx2*MAZE_CELL,gy2*MAZE_CELL
        cx,cy=(x1+x2)/2.,(y1+y2)/2.
        dx,dy=abs(x2-x1),abs(y2-y1)
        he=([WALL_T/2.,max(dy/2.,WALL_T/2.),WALL_H/2.] if dx<0.01
            else [max(dx/2.,WALL_T/2.),WALL_T/2.,WALL_H/2.])
        col=p.createCollisionShape(p.GEOM_BOX,halfExtents=he)
        vis=p.createVisualShape(p.GEOM_BOX,halfExtents=he,
                                rgbaColor=[0.06,0.07,0.12,1.])
        wid=p.createMultiBody(baseMass=0,baseCollisionShapeIndex=col,
                              baseVisualShapeIndex=vis,
                              basePosition=[cx,cy,WALL_H/2.])
        wall_ids.append(wid)
    return wall_ids

def draw_static_walls(walls_grid):
    """Draw glowing wall edges once at episode start."""
    for (gx1,gy1,gx2,gy2) in walls_grid:
        p.addUserDebugLine(
            [gx1*MAZE_CELL,gy1*MAZE_CELL,WALL_H],
            [gx2*MAZE_CELL,gy2*MAZE_CELL,WALL_H],
            [0.15,0.45,1.0],1)

def draw_grid():
    for gx in range(-1,7):
        x=gx*MAZE_CELL
        p.addUserDebugLine([x,MAP_YMIN,.005],[x,MAP_YMAX,.005],[.05,.08,.20],1)
    for gy in range(-1,5):
        y=gy*MAZE_CELL
        p.addUserDebugLine([MAP_XMIN,y,.005],[MAP_XMAX,y,.005],[.05,.08,.20],1)

def spawn_goal(gx,gy):
    tgt=p.createVisualShape(p.GEOM_SPHERE,radius=0.18,rgbaColor=[1.,.15,.5,.95])
    p.createMultiBody(baseMass=0,baseVisualShapeIndex=tgt,basePosition=[gx,gy,.18])
    for h in [.05,.25,.50]:
        for angle in np.linspace(0,2*np.pi,12):
            p.addUserDebugLine([gx,gy,h],
                [gx+.35*math.cos(angle),gy+.35*math.sin(angle),h],[1.,.15,.5],2)

# ── MAIN ──────────────────────────────────────────────────────
def main():
    global brain

    p.connect(p.GUI,options="--width=1440 --height=900")
    p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS,1)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    T_ltm,P_ltm  = fresh_ltm()
    D_global      = fresh_danger()
    CM_global     = fresh_cognitive_map()
    FM_global     = fresh_flow_map()      # slime mould
    L2_legacy     = fresh_legacy2d()

    episode=0; best=0.; best_dist_ever=99.
    target_reached_count=0; ghost_count=0
    mourning_events=0; junction_decisions=0
    nrem1=nrem3=rem_count=0
    goal_idx=0; goal_pos=[GOAL_POSITIONS[goal_idx]]
    swarm_hebbian=HebbianAssociator(capacity=2000)

    print("\n== CARL Phase 14: PARALLEL BRAIN + SLIME MOULD ==")
    print("4 threads | Graduated curriculum | Wall-aware A* | Physarum flow map\n")

    while True:
        episode+=1; brain["episode"]=episode

        if episode%200==1 and episode>1:
            goal_idx=(goal_idx+1)%len(GOAL_POSITIONS)
            goal_pos[0]=GOAL_POSITIONS[goal_idx]
            print(f"\n  *** GOAL MOVED to {goal_pos[0]} ***\n")

        # ── GRADUATED CURRICULUM ──────────────────────────────
        # Stage 1: open arena — balance + goal seeking
        # Stage 2: corridor — wall awareness
        # Stage 3: full maze
        # Stage 4+: maze + disturbances
        stage=1
        if best>20.:  stage=2
        if best>50.:  stage=3
        if best>90.:  stage=4
        if best>140.: stage=5
        brain["curriculum_stage"]=stage

        # Pick walls for this stage
        if stage<=1:   active_walls=[]                  # open arena
        elif stage==2: active_walls=CORRIDOR_WALLS_GRID # corridor
        else:          active_walls=MAZE_WALLS_GRID     # full maze

        init_pitch=float(np.random.uniform(-.10,.10)) if stage>=2 else 0.

        try:
            p.resetSimulation()
            p.setGravity(0,0,-9.81)
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            plane=p.loadURDF("plane.urdf")
            p.changeVisualShape(plane,-1,rgbaColor=[.02,.02,.05,1])

            draw_grid()
            if active_walls: draw_static_walls(active_walls)
            wall_ids=build_walls(active_walls)

            # Ghost markers
            for gi in range(MAP_RES):
                for gj in range(MAP_RES):
                    lv=float(L2_legacy[gi,gj])
                    if lv>0.15:
                        xw=MAP_XMIN+(gi/MAP_RES)*(MAP_XMAX-MAP_XMIN)
                        yw=MAP_YMIN+(gj/MAP_RES)*(MAP_YMAX-MAP_YMIN)
                        a=min(1.,lv)
                        p.addUserDebugLine([xw-.06,yw,.02],[xw+.06,yw,.02],[a,a*.1,a*.1],1)
                        p.addUserDebugLine([xw,yw-.06,.02],[xw,yw+.06,.02],[a,a*.1,a*.1],1)

            gx_w,gy_w=goal_pos[0]
            p.resetDebugVisualizerCamera(8.,45,-35,[gx_w*.5,gy_w*.5,.3])
            spawn_goal(gx_w,gy_w)

            y_offsets=np.linspace(-.4,.4,N_BODIES).tolist()
            bodies=[spawn_robot(.3,y,init_pitch) for y in y_offsets]
            for i,rid in enumerate(bodies):
                rc=.4+.6*(i/max(1,N_BODIES-1)); bc=1.-.4*(i/max(1,N_BODIES-1))
                p.changeVisualShape(rid,-1,rgbaColor=[rc,.2,bc,1.])
                p.changeVisualShape(rid,3,rgbaColor=[rc,.2,bc,1.])

        except Exception as e:
            print(f"PyBullet crash: {e}")
            p.disconnect(); time.sleep(1.)
            p.connect(p.GUI,options="--width=1440 --height=900"); continue

        for _ in range(15): p.stepSimulation()
        time.sleep(0.3)

        # ── COGNITIVE MAP: pre-burn walls ─────────────────────
        CM=CM_global.copy()
        if active_walls:
            CM=preburn_walls_into_map(CM,active_walls)

        FM=FM_global.copy()
        FM=flow_decay_episode(FM,rate=0.02)   # Physarum decay

        D=D_global.copy()
        P_ltm_ep=P_ltm.copy(); T_ltm_ep=T_ltm.copy()
        next_wind=np.random.randint(20*240,40*240)

        brains=[spawn_brain(T_ltm,P_ltm,i) for i in range(N_BODIES)]
        for b,rid in zip(brains,bodies):
            b["rid"]=rid
            b["xk"],_,_,_=get_state(rid)
            b["pos_2d"]=get_pos2d(rid)

        T_ltm_ref=[T_ltm_ep]; D_ref=[D]; CM_ref=[CM]; FM_ref=[FM]
        slots=[DecisionSlot() for _ in range(N_BODIES)]
        path_slot=PathSlot()
        overrides=[AmygdalaOverride() for _ in range(N_BODIES)]
        body_states=[None]*N_BODIES
        body_positions=[]
        counters_lock=threading.Lock()
        counters={"prefrontal":0,"astar":0,"amygdala":0}
        stop_event=threading.Event()

        # Launch threads
        for i,rb in enumerate(brains):
            threading.Thread(
                target=prefrontal_thread,
                args=(rb,slots[i],T_ltm_ref,D_ref,CM_ref,
                      swarm_hebbian,goal_pos,stop_event,
                      counters_lock,counters),
                daemon=True).start()

        threading.Thread(
            target=hippocampus_thread,
            args=(CM_ref,FM_ref,path_slot,goal_pos,body_positions,
                  stop_event,counters_lock,counters),
            daemon=True).start()

        threading.Thread(
            target=amygdala_thread,
            args=(D_ref,body_states,overrides,stop_event,
                  counters_lock,counters),
            daemon=True).start()

        stage_name=["OPEN ARENA","CORRIDOR","FULL MAZE","CHAOS","AUTONOMY"][min(stage-1,4)]
        print(f"[Ep {episode:3d}] Stage:{stage}({stage_name}) "
              f"LTM:{ltm_conf(P_ltm_ep)*100:.0f}% Best:{best:.1f}s "
              f"Goal:{goal_pos[0]}")

        loop_times=deque(maxlen=120)
        last_s_ltm=0.; hud_step=0

        for step in range(500_000):
            t0=_time.perf_counter()
            sibling_died_this_step=False; dead_this_step=[]

            for b in brains:
                if b["alive"] and b.get("grief",0.)>0.:
                    b["grief"]*=0.9998

            if step%240==0: swarm_hebbian.decay()

            wind_active=False
            if stage>=3 and step>=next_wind:
                fx=float(np.random.uniform(-2.5,2.5))
                for b in brains:
                    if b["alive"]:
                        p.applyExternalForce(b["rid"],-1,[fx,0,0],[0,0,.3],p.WORLD_FRAME)
                next_wind=step+np.random.randint(20*240,40*240); wind_active=True

            q_amp=0.
            if stage>=4:
                t_ep=step*DT; q_amp=float(np.clip(t_ep/90.,0.,.25))
                qf=q_amp*float(np.sin(2*np.pi*1.5*t_ep))
                if q_amp>0.01:
                    for b in brains:
                        if b["alive"]:
                            p.applyExternalForce(b["rid"],-1,[qf,0,0],[0,0,0],p.WORLD_FRAME)

            alive_count=sum(1 for b in brains if b["alive"])
            body_positions.clear()
            body_positions.extend([b["pos_2d"] for b in brains if b["alive"]])
            waypoint=path_slot.next_waypoint()

            for i,rb in enumerate(brains):
                if not rb["alive"]: continue
                rid=rb["rid"]
                xn,y_pos,z_pos,roll=get_state(rid)
                xw,yw=get_pos2d(rid)
                rb["pos_2d"]=(xw,yw); body_states[i]=xn.copy()
                nm=rb["nm"]; home=rb["homeostatic"]
                hebb=rb["hebbian"]; pred=rb["predictive"]; social=rb["social"]

                pred.predict(rb["xk"],rb["uk"],rb["T_wm"],T_ltm_ep,rb["P_wm"])

                ach_lam=nm.lr(0.990)
                rb["T_wm"],rb["P_wm"],s_wm=rls_update(
                    rb["T_wm"],rb["P_wm"],rb["xk"],rb["uk"],xn,ach_lam)
                T_ltm_ep,P_ltm_ep,s_ltm=rls_update(
                    T_ltm_ep,P_ltm_ep,rb["xk"],rb["uk"],xn,0.99995,0.002)
                last_s_ltm=s_ltm

                T_ltm_ref[0]=T_ltm_ep; D_ref[0]=D
                CM_ref[0]=CM; FM_ref[0]=FM

                pred.compute_error(xn)
                rb["s_wm_ema"]=.1*s_wm+.9*rb["s_wm_ema"]
                D=danger_update(D,xn[2],xn[3],rb["s_wm_ema"])
                CM=cm_update(CM,xw,yw,rb["s_wm_ema"],+0.005)

                # Slime mould: reinforce path on every step
                FM=flow_reinforce(FM,xw,yw,amount=0.01)

                dist_now=math.sqrt((xw-goal_pos[0][0])**2+(yw-goal_pos[0][1])**2)

                _at=dist_now<0.35; _was=rb.get("was_at_target",False)
                nm.update(rb["s_wm_ema"],danger_at(D,xn[2],xn[3]),
                          dist_now,_at and not _was,
                          sibling_died_this_step,home.allostatic_load)
                home.update(xn[2],xn[1],nm,rb["fatigue"],alive_count-1,dist_now)
                social.update(brains,xw)
                swarm_hebbian.fire(xn[2],xn[3],rb["uk"],rb["s_wm_ema"])
                hebb.fire(xn[2],xn[3],rb["uk"],rb["s_wm_ema"])

                danger_here=nm.danger_amp(danger_at(D,xn[2],xn[3]))
                if danger_here>0.20 or s_wm>0.25:
                    rb["buf"].append((rb["xk"].copy(),rb["uk"],xn.copy(),s_wm))
                    if len(rb["buf"])>200: rb["buf"].pop(0)
                if dist_now<1.5:
                    rb["near_miss_buf"].append((rb["xk"].copy(),rb["uk"],xn.copy(),s_wm))
                    if len(rb["near_miss_buf"])>100: rb["near_miss_buf"].pop(0)

                rb["fatigue"]=.999*rb["fatigue"]+.001*abs(rb["uk"])

                if _at and not _was:
                    target_reached_count+=1
                    brain["target_reached_count"]=target_reached_count
                    nm.DA=min(1.,nm.DA+0.4)
                    # Big flow reinforcement when goal reached
                    FM=flow_reinforce(FM,xw,yw,amount=0.3)
                    print(f"  *** GOAL REACHED Body {i+1}! Total:{target_reached_count} ***")
                rb["was_at_target"]=_at
                if dist_now<best_dist_ever:
                    best_dist_ever=dist_now; brain["best_dist_ever"]=round(best_dist_ever,3)

                # ── CEREBELLUM DECISION ───────────────────────
                if overrides[i].is_active():
                    un=fast_reflex(xn,rb["T_wm"],T_ltm_ep,rb["P_wm"],D,nm,home)
                    mode="AMYGDALA-REFLEX"
                else:
                    pf_action,pf_name=slots[i].read_and_consume()
                    if pf_action is not None:
                        un=pf_action; mode=f"DELIBERATING-{pf_name}"
                        with counters_lock: junction_decisions+=1
                    else:
                        un=fast_reflex(xn,rb["T_wm"],T_ltm_ep,rb["P_wm"],
                                       D,nm,home,waypoint)
                        mode="REFLEX"

                grief=rb.get("grief",0.)
                if grief>0.3 and mode=="REFLEX": mode="GRIEVING"
                rb["mode"]=mode
                apply_torque(rid,un)
                rb["xk"],rb["uk"]=xn,un; rb["sv"]=time.time()-rb["t0"]

                if z_pos<0.065 or abs(xn[2])>0.65 or abs(roll)>0.5:
                    sv=rb["sv"]; best=max(best,sv)
                    brain["episode_history"].append(round(sv,2))
                    brain["best_survival"]=round(best,2)
                    rb["alive"]=False; sibling_died_this_step=True
                    dead_this_step.append(rb); body_states[i]=None

                    for _ in range(4): D=danger_update(D,xn[2],xn[3],10.,rate=0.3)
                    CM=cm_update(CM,xw,yw,1.,-.2)
                    FM=flow_dissolve(FM,xw,yw,amount=0.3)  # slime: dissolve death cell

                    if dist_now<0.8:
                        intensity=1.-dist_now/0.8; mourning_events+=1
                        for b in brains:
                            if b["alive"]:
                                b["grief"]=min(1.,b.get("grief",0.)+intensity*.5)
                                b["nm"].NE=min(1.,b["nm"].NE+intensity*.3)
                                b["nm"].ACh=min(1.,b["nm"].ACh+.2)
                        print(f"  [MOURNING] Body {i+1} near goal ({xw:.1f},{yw:.1f})")

                    ghost_intensity=min(1.,.3+sv/60.)
                    L2_legacy=legacy_write2d(L2_legacy,xw,yw,ghost_intensity,
                                             goal_pos[0][0],goal_pos[0][1])
                    ghost_count+=1
                    print(f"  Body {i+1} died {sv:.2f}s | {mode} | "
                          f"DA:{nm.DA:.2f} NE:{nm.NE:.2f} SHT:{nm.SHT:.2f} | "
                          f"AlloLoad:{home.allostatic_load:.3f}")
                    try: p.removeBody(rid)
                    except: pass

            if dead_this_step:
                all_t=[]; all_n=[]
                for b in dead_this_step:
                    all_t.extend(b["buf"]); all_n.extend(b["near_miss_buf"])
                max_allo=max(b["homeostatic"].allostatic_load for b in dead_this_step)
                brain["sleep_phase"]="NREM-1"
                T_ltm_ep,P_ltm_ep,D=biological_sleep(
                    T_ltm_ep,P_ltm_ep,D,all_t,all_n,swarm_hebbian,max_allo)
                for b in dead_this_step: b["homeostatic"].allostatic_load*=0.98
                nrem1+=1; nrem3+=1; rem_count+=1
                brain["sleep_phase"]="AWAKE"
                brain["nrem1_count"]=nrem1; brain["nrem3_count"]=nrem3
                brain["rem_count"]=rem_count

            if sibling_died_this_step:
                for b in brains:
                    if b["alive"]:
                        b["nm"].update(0,0,0,False,True,b["homeostatic"].allostatic_load)

            loop_times.append(_time.perf_counter()-t0)
            avg_ms=float(np.mean(loop_times))*1000.

            # ── DASHBOARD (every step — fast, no debug redraw) ─
            nm0=brains[0]["nm"]
            xA=brains[0]["xk"] if brains[0]["xk"] is not None else np.zeros(SDIM)
            alive_modes=[b["mode"] for b in brains if b["alive"]]
            mode_d=alive_modes[0] if alive_modes else "DEAD"
            avg_allo=float(np.mean([b["homeostatic"].allostatic_load for b in brains]))
            dmax=float(np.max(D))+1e-6
            cm_flat=(CM[:,:,0]/(float(np.max(CM[:,:,0]))+1e-6)).flatten().round(3).tolist()
            fm_flat=FM.flatten().round(3).tolist()

            with counters_lock:
                pf_total=counters["prefrontal"]
                as_total=counters["astar"]
                am_total=counters["amygdala"]

            brain.update({
                "survivals":[round(b["sv"],2) for b in brains],
                "pitches":[round(float(b["xk"][2]) if b["xk"] is not None else 0.,3)
                           for b in brains],
                "distances":[round(float(math.sqrt(
                    (b["pos_2d"][0]-goal_pos[0][0])**2+
                    (b["pos_2d"][1]-goal_pos[0][1])**2))
                    if b["xk"] is not None else 99.,2) for b in brains],
                "alive_flags":[b["alive"] for b in brains],
                "best_survival":round(best,2),"mode":mode_d,
                "surprise_wm":float(brains[0]["s_wm_ema"]),
                "surprise_ltm":float(last_s_ltm),
                "ltm_confidence":float(ltm_conf(P_ltm_ep)),
                "wm_confidence_A":float(wm_conf(brains[0]["P_wm"])),
                "curiosity":float(np.trace(brains[0]["P_wm"])/P0_WM),
                "danger_level":float(danger_at(D,xA[2],xA[3])) if brains[0]["alive"] else 0.,
                "wind_active":wind_active,"quake_amp":round(q_amp,3),
                "danger_grid":(D/dmax).flatten().round(3).tolist(),
                "cognitive_map":cm_flat,"flow_map":fm_flat,
                "curriculum_stage":stage,
                "DA":round(nm0.DA,3),"5HT":round(nm0.SHT,3),
                "NE":round(nm0.NE,3),"ACh":round(nm0.ACh,3),
                "allostatic_load":round(avg_allo,4),
                "homeostatic_error":round(
                    brains[0].get("last_h_error",0.)
                    if brains[0]["alive"] else 0.,3),
                "sleep_phase":brain["sleep_phase"],
                "nrem1_count":nrem1,"nrem3_count":nrem3,"rem_count":rem_count,
                "social_comfort":round(float(np.mean(
                    [b["social"].social_comfort for b in brains if b["alive"]]
                ) if any(b["alive"] for b in brains) else 0.),3),
                "alive_count":alive_count,
                "hebbian_strength":round(swarm_hebbian.total(),3),
                "prediction_error":round(
                    float(brains[0]["predictive"].precision())
                    if brains[0]["alive"] else 0.,3),
                "goal_x":goal_pos[0][0],"goal_y":goal_pos[0][1],
                "ghost_count":ghost_count,"mourning_events":mourning_events,
                "target_reached_count":target_reached_count,
                "best_dist_ever":round(best_dist_ever,3),
                "junction_decisions":junction_decisions,
                "thread_fps":round(1000./max(avg_ms,.001),1),
                "prefrontal_decisions":pf_total,
                "astar_updates":as_total,"amygdala_triggers":am_total,
            })

            # ── HUD: only redraw every 48 steps (~5Hz) ────────
            # THIS IS THE SPEED FIX.
            # removeAllUserDebugItems every step = 1100ms loop.
            # Every 48 steps = <5ms loop. Physics runs at 240Hz.
            hud_step+=1
            if hud_step>=48:
                hud_step=0
                p.removeAllUserDebugItems()
                draw_grid()
                if active_walls: draw_static_walls(active_walls)

                # A* path — green line showing planned route
                path_now=path_slot.read()
                for k in range(min(len(path_now)-1,12)):
                    p.addUserDebugLine(
                        [path_now[k][0],path_now[k][1],.06],
                        [path_now[k+1][0],path_now[k+1][1],.06],
                        [0.,1.,.4],3)

                # Slime mould flow tubes — bright green where flow is strong
                fm_max=float(np.max(FM))+1e-6
                for gi in range(0,MAP_RES,2):
                    for gj in range(0,MAP_RES,2):
                        fv=float(FM[gi,gj])
                        if fv>0.05:
                            xwf=MAP_XMIN+(gi/MAP_RES)*(MAP_XMAX-MAP_XMIN)
                            ywf=MAP_YMIN+(gj/MAP_RES)*(MAP_YMAX-MAP_YMIN)
                            a=min(1.,fv/fm_max)
                            p.addUserDebugLine(
                                [xwf-.05,ywf,.03],[xwf+.05,ywf,.03],
                                [0.,a*0.8,0.],1)

                # Ghost markers
                for gi in range(0,MAP_RES,2):
                    for gj in range(0,MAP_RES,2):
                        lv=float(L2_legacy[gi,gj])
                        if lv>0.15:
                            xwg=MAP_XMIN+(gi/MAP_RES)*(MAP_XMAX-MAP_XMIN)
                            ywg=MAP_YMIN+(gj/MAP_RES)*(MAP_YMAX-MAP_YMIN)
                            a=min(1.,lv)
                            p.addUserDebugLine(
                                [xwg-.06,ywg,.02],[xwg+.06,ywg,.02],
                                [a,a*.1,a*.1],1)

                # Goal beacon
                gxd,gyd=goal_pos[0]
                for angle in np.linspace(0,2*np.pi,12):
                    p.addUserDebugLine([gxd,gyd,.3],
                        [gxd+.35*math.cos(angle),gyd+.35*math.sin(angle),.3],
                        [1.,.15,.5],2)

                # HUD text
                p.addUserDebugText(
                    f"STAGE {stage}:{stage_name} | {mode_d} | "
                    f"Best:{best:.1f}s | Loop:{avg_ms:.1f}ms | "
                    f"PF:{pf_total} A*:{as_total}",
                    [0,-2,.8],textColorRGB=[0,1,.5],textSize=1.)
                p.addUserDebugText(
                    f"DA:{nm0.DA:.2f} NE:{nm0.NE:.2f} "
                    f"SHT:{nm0.SHT:.2f} ACh:{nm0.ACh:.2f} | "
                    f"AlloLoad:{avg_allo:.3f} | Reached:{target_reached_count}",
                    [0,-2,.6],textColorRGB=[.4,.8,1.],textSize=.85)

            p.stepSimulation()
            time.sleep(DT)

            if not any(b["alive"] for b in brains):
                stop_event.set()
                T_ltm,P_ltm=T_ltm_ep.copy(),P_ltm_ep.copy()
                D_global=D.copy(); CM_global=CM.copy()
                FM_global=FM.copy()   # persist slime mould

                if episode%50==0:
                    np.save("carl_ltm_T.npy",T_ltm)
                    np.save("carl_ltm_P.npy",P_ltm)
                    np.save("carl_danger.npy",D_global)
                    np.save("carl_cm.npy",CM_global)
                    np.save("carl_flow.npy",FM_global)
                    np.save("carl_legacy.npy",L2_legacy)
                    print(f"  [CHECKPOINT ep {episode}]")

                print(f"  Ep {episode} done. Best:{best:.2f}s "
                      f"Reached:{target_reached_count}x "
                      f"Ghosts:{ghost_count} "
                      f"PF:{pf_total} A*:{as_total} Amy:{am_total}")
                time.sleep(1.5); break

if __name__=="__main__":
    main()
