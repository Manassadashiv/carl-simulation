# genesis_run.py
# The Heartbeat of CARL Genesis.
# The main simulation loop tying physics, brain, world, and visualization together.

import time, math, os, sys
import numpy as np

# Suppress pygame welcome message
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

import genesis_physics as phy
import genesis_world   as world
from genesis_reservoir import LiquidStateMachine, ReflexLayer, Neurotransmitters, SelfModel
import genesis_telemetry as telemetry

sys.path.insert(0, os.path.abspath('../brain'))
try:
    from carl_grid_cells import HippocampalNavigator
    from carl_physarum import PhysarumMaze
    from carl_stdp import QuantumDeliberator
except ImportError:
    sys.path.insert(0, os.path.abspath('brain'))
    from carl_grid_cells import HippocampalNavigator
    from carl_physarum import PhysarumMaze
    from carl_stdp import QuantumDeliberator

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

DT         = 0.00417   # 240Hz physics
STEPS_PER_FRAME = 4    # 60Hz brain update
MAX_STEPS  = 3000000

# We will run just CARL-A for V1 to ensure stability.
# The body for B exists in the world, but its brain is asleep.
ACTIVE_BODY = 'A'

# ─────────────────────────────────────────────────────────────────────────────
# INITIALIZATION
# ─────────────────────────────────────────────────────────────────────────────

print("==================================================")
print("             CARL GENESIS - V1.0                  ")
print("==================================================")

# 1. Physics & World
phy.init_physics('genesis_body.xml')
world.init_sound()
cmd_channel = world.CommandChannel()

# Define some default safe targets for mission mode
MISSION_TARGETS = [(0.6, 1.8), (3.6, 5.4), (5.4, 5.4), (6.6, 1.8), (0.6, 5.4)]

# 2. The Brain (Reservoir + Reflexes + Chemistry + Metacognition)
# 128 neurons for stability in V1
print("\n[SPAWNING MINDS]")
brains = {}
for b_name in ['A', 'B']:
    brains[b_name] = {
        'lsm': LiquidStateMachine(n_reservoir=128, n_inputs=20, n_outputs=2),
        'reflex': ReflexLayer(n_sensors=20, n_motors=2, threshold=0.6),
        'nm': Neurotransmitters(),
        'self_model': SelfModel(),
        'energy': 100.0,
        'fatigue': 0.0,
        'grief': 0.0,
        'hippocampus': HippocampalNavigator(),
        'physarum': PhysarumMaze(rows=60, cols=60),
        'deliberator': QuantumDeliberator(),
        'mission_idx': 0 if b_name == 'A' else 1,
        'prev_pos': None,
        'cfg': phy.GCFG[0] if b_name == 'A' else phy.GCFG[1]
    }
    brains[b_name]['primary_goal'] = MISSION_TARGETS[brains[b_name]['mission_idx']]

# 3. Load Inheritance (Phase 18 LTM)
print("\n[INHERITANCE]")
if os.path.exists('memory/mj_ltm_T.npy'):
    try:
        ltm_T = np.load('memory/mj_ltm_T.npy')
        print(f"  Loaded LTM traces: {ltm_T.shape[0]} memories")
        for b in brains.values():
            b['lsm'].pretrain_from_ltm(ltm_T, None, n_samples=2000)
    except Exception as e:
        print(f"  Could not load LTM: {e}")

# 4. Cognitive Map
# ── COGNITIVE MAP INITIALIZATION ─────────────────────────────────────────────
visit_map   = world.load_visit_map('memory/genesis_visits.npy')
danger_grid = world.load_danger_grid('memory/genesis_danger.npy')
CM          = world.load_cognitive_map('memory/genesis_CM.npy')

# Seed the maze walls into the Cognitive Map Obstacle Channel (CM[:,:,2])
# This gives the Slime Mold "instinctive" knowledge of the geometry.
wall_positions = phy.get_wall_positions()
CM = world.seed_walls_into_map(CM, wall_positions)

# Vitals (Now inside the brains dictionary, keeping is_sleeping flag here if needed)
is_sleeping  = False


# ─────────────────────────────────────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────────────────────────────────────

telemetry.start()

print("\n[STARTING HEARTBEAT]")
viewer = phy.launch_viewer()

step = 0
last_time = time.time()

try:
    while phy.viewer_running() and step < MAX_STEPS:
        
        # ── 1. Perception (60Hz) ─────────────────────────────────────────────
        if step % STEPS_PER_FRAME == 0:
            for b_name, b in brains.items():
                cfg = b['cfg']
                lsm = b['lsm']
                reflex = b['reflex']
                nm = b['nm']
                self_model = b['self_model']
                
                x, y = phy.get_pos_2d(cfg)
                
                # Calculate velocity for Hippocampus
                vx, vy = 0.0, 0.0
                if b['prev_pos'] is not None:
                    px, py = b['prev_pos']
                    dt_frame = DT * STEPS_PER_FRAME
                    vx, vy = (x - px) / dt_frame, (y - py) / dt_frame
                b['prev_pos'] = (x, y)
                
                # Update Hippocampal Grid Cells
                hippo = b['hippocampus']
                nav_out = hippo.step(x, y, vx, vy, DT * STEPS_PER_FRAME, learn=(not reflex_fired if 'reflex_fired' in locals() else True))
                
                lidar_dists, _ = phy.get_lidar_360(cfg)
                danger_here    = world.danger_at(danger_grid, x, y)
                
                # Map tracking
                visit_map = world.mark_visited(visit_map, x, y)
                CM = world.update_cognitive_map(CM, x, y, nm.ACh, 0.001)

                # Day/Night cycle & Biological Curiosity
                day_factor = world.get_day_factor(step)
                world.apply_day_night(nm, day_factor)
                nm.ACh = hippo.curiosity_boost(nm.ACh) # Novelty drives curiosity


                # Process Commands
                cmd_wp = cmd_channel.process(b_name, nm, self_model)

                # Food / Hunger dynamics
                b['energy'] = max(0.0, b['energy'] - 0.005) # Metabolic cost
                nearest_food = phy.nearest_food(x, y)
                eaten_idx    = phy.check_food_contact(cfg, step)
                
                if eaten_idx >= 0:
                    phy.eat_food(eaten_idx, step)
                    b['energy'] = 100.0
                    nm.DA = min(1.0, nm.DA + 0.5)
                    world.play_eat_sound()
                    print(f"[{step}] CARL-{b_name} ate food! Energy restored.")
                
                phy.update_food_respawn(step)

                # ── 2. Metacognition & Goal Selection ────────────────────────────
                lsm_uncertainty = 1.0 - (float(np.mean(np.abs(lsm.state))) * 2.0)
                # Combine Hippocampal uncertainty with LSM uncertainty
                total_uncertainty = float(np.clip((lsm_uncertainty + nav_out['uncertainty']) / 2.0, 0.0, 1.0))
                self_model.update(nm, b['energy'], b['grief'], b['fatigue'], 10.0, total_uncertainty)
                
                # For social dynamics, pass the sibling's position
                sib_name = 'B' if b_name == 'A' else 'A'
                sib_pos = phy.get_pos_2d(brains[sib_name]['cfg'])
                
                target, mode = world.select_goal(
                    b_name, x, y, nm, self_model, sib_pos,
                    b['primary_goal'], visit_map, world.MAP_XMIN, world.MAP_XMAX,
                    world.MAP_YMIN, world.MAP_YMAX, world.MAP_RES, nearest_food,
                    b['deliberator'], cmd_waypoint=cmd_wp
                )

                if mode == 'MISSION':
                    dist_to_goal = math.hypot(target[0] - x, target[1] - y)
                    if dist_to_goal < 0.3:
                        b['mission_idx'] = (b['mission_idx'] + 1) % len(MISSION_TARGETS)
                        b['primary_goal'] = MISSION_TARGETS[b['mission_idx']]
                        nm.DA = min(1.0, nm.DA + 0.4)
                        world.play_goal_sound()
                        print(f"[{step}] CARL-{b_name} reached waypoint! Changing goal.")
                else:
                    dist_to_goal = math.hypot(target[0] - x, target[1] - y)

                # ── 3. Neurotransmitter Update & Sound ───────────────────────────
                hit_wall = phy.check_wall_contact(cfg)
                surprise = 0.0
                if hit_wall:
                    surprise = 0.5
                    danger_grid = world.update_danger_grid(danger_grid, x, y, 1.0)
                    b['energy'] -= 2.0 # Pain costs energy

                nm.update(surprise, danger_here, dist_to_goal, False, False, b['fatigue'])
                world.update_sound(b_name, nm, self_model, step)

                # ── Echolocation / Sonar Navigation (Guided by Physarum) ─────────────
                physarum = b['physarum']
                
                def w2g(xw, yw):
                    gi = int(np.clip((xw - world.MAP_XMIN)/(world.MAP_XMAX - world.MAP_XMIN)*physarum.R, 0, physarum.R-1))
                    gj = int(np.clip((yw - world.MAP_YMIN)/(world.MAP_YMAX - world.MAP_YMIN)*physarum.C, 0, physarum.C-1))
                    return gi, gj
                
                src_cell = w2g(x, y)
                snk_cell = w2g(target[0], target[1])
                
                # Step physarum network using Cognitive Map's Obstacle channel (index 2)
                # This prevents conflating learned Danger (index 0) with innate geometry
                physarum.step(src_cell, snk_cell, CM_danger=CM[:, :, 2], every=20)
                
                # Get optimal heading
                heading = physarum.get_heading(x, y, target[0], target[1], world.MAP_XMIN, world.MAP_XMAX, world.MAP_YMIN, world.MAP_YMAX)
                dx, dy = math.cos(heading), math.sin(heading)
                b['sonar_vector'] = (float(dx), float(dy))
                
                # ── The Bicameral Architecture: Precision Body + Biological Mind ─────────────
                
                # 1. Biological State Awareness (LSM still runs to generate mood/thoughts)
                sensors = np.array(lidar_dists + [dx, dy, self_model.state['hunger'], nm.NE], dtype=np.float32)
                lsm.step(sensors)
                res_out = lsm.readout()
                l_delib, r_delib = res_out[0], res_out[1]

                # Reflex layer runs to learn, but does NOT override the Precision Body
                reward_sig = (nm.DA - 0.5) * 2.0 
                if hit_wall: reward_sig = -1.0
                reflex_out, reflex_fired = reflex.step(sensors, (l_delib, r_delib), reward_sig, nm.NE)
                
                # 2. Precision Robotic Controller (Flawless Path Following)
                yaw = phy.get_yaw(cfg)
                target_yaw = math.atan2(dy, dx)
                yaw_err = target_yaw - yaw
                
                # Normalize yaw error to [-pi, pi]
                while yaw_err > math.pi:  yaw_err -= 2.0 * math.pi
                while yaw_err < -math.pi: yaw_err += 2.0 * math.pi
                
                # Biological Speed Modulation (The "Endocrine" signal)
                # Max speed is extremely fast (e.g. 15.0 rad/s), modulated by fear, hunger, and grief.
                base_speed = 10.0
                mood_multiplier = 1.0 + (nm.DA * 0.4) + (nm.NE * 0.6) - (self_model.state['grief'] * 0.5)
                target_speed = base_speed * max(0.2, mood_multiplier)
                
                # Proportional Steering
                if abs(yaw_err) > 0.4:
                    # Point turn if error is large
                    l_final = -yaw_err * 5.0
                    r_final = yaw_err * 5.0
                else:
                    # Drive forward while steering
                    l_final = target_speed - (yaw_err * 8.0)
                    r_final = target_speed + (yaw_err * 8.0)
                    
                # 3. Final Command Channel Override
                speed_mod = cmd_channel.get_speed_modifier(b_name)
                l_final *= speed_mod
                r_final *= speed_mod

                phy.apply_drive(cfg, l_final * 2.0, r_final * 2.0)

                phy.set_neck(cfg, self_model.neck_target(step, DT))
                phy.set_head_pan(cfg, self_model.head_pan_target(lidar_dists, step, DT))

                # ── 5. Continuous Online Learning ────────────────────────────────
                if not reflex_fired and nm.DA > 0.3:
                    if hit_wall:
                        td = -res_out
                    else:
                        yaw = phy.get_yaw(cfg)
                        angle_to_target = math.atan2(dy, dx)
                        yaw_err = angle_to_target - yaw
                        yaw_err = (yaw_err + math.pi) % (2 * math.pi) - math.pi
                        
                        target_l = max(-1.0, min(1.0, math.cos(yaw_err) - math.sin(yaw_err)))
                        target_r = max(-1.0, min(1.0, math.cos(yaw_err) + math.sin(yaw_err)))
                        td = np.array([target_l, target_r]) - res_out
                    
                    lsm.update_readout(td, nm.DA)

                # ── 6. Telemetry Update ──────────────────────────────────────────
                telemetry.update_bot_state(b_name,
                    pos=[x, y],
                    yaw=phy.get_yaw(cfg),
                    lidar=lidar_dists,
                    nm={"DA": nm.DA, "NE": nm.NE, "ACh": nm.ACh, "SHT": nm.SHT},
                    state=self_model.state,
                    mood=nm.mood,
                    energy=b['energy'],
                    mode=mode,
                    reflex_fired=reflex_fired,
                    reflex_ratio=reflex.reflex_ratio(),
                    neck=self_model.neck_target(step, DT),
                    head_pan=self_model.head_pan_target(lidar_dists, step, DT),
                    step=step,
                    target=list(target),
                    sonar_vector=b['sonar_vector']
                )
            
            # Update world telemetry once per perception frame
            telemetry.update_world_state(
                food=phy.get_food_positions(),
                danger_grid=danger_grid.tolist(),
                day_factor=world.get_day_factor(step)
            )

        # ── 7. Physics Step (240Hz) ──────────────────────────────────────────
        phy.step_physics()
        step += 1

        if step % 240 == 0:
            phy.viewer_sync()
            
            if step % 2400 == 0:
                for b_name, b in brains.items():
                    print(f"[{step}] CARL-{b_name} State: {b['self_model'].state}")
                    print(f"        CARL-{b_name} Mood: {b['nm'].mood:.2f} | Reflex Ratio: {b['reflex'].reflex_ratio():.2f}")

except KeyboardInterrupt:
    print("\n[SHUTDOWN] Interrupted by user.")

print("[SHUTDOWN] Saving state...")
os.makedirs('memory', exist_ok=True)
brains['A']['reflex'].save('memory/genesis_reflex.npy')
np.save('memory/genesis_cm.npy', CM)
np.save('memory/genesis_danger.npy', danger_grid)
print("[SHUTDOWN] Complete.")
