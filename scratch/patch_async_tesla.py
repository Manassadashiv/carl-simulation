import re

with open('phase14_maze.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Imports and N_BODIES
content = content.replace("import asyncio, websockets, json, threading, math, random",
                          "import asyncio, websockets, json, threading, math, random\nfrom astar import astar_path")
content = content.replace("N_BODIES      = 5", "N_BODIES      = 2")

# 2. Add PFC Thread and Global variables
pfc_logic = """
# ── ASYNCHRONOUS PREFRONTAL CORTEX (TESLA FSD) ───────────────
pfc_active = True
def pfc_worker(b_idx):
    while pfc_active:
        b = brains[b_idx]
        if not b["alive"]:
            time._sleep(0.1)
            continue
            
        try:
            # 1. A* Pathfinding (Hippocampus)
            path = astar_path(CM_global, b["pos_2d"], goal_pos)
            b["astar_path"] = path
            
            # 2. Physics Rollout (Prefrontal Cortex)
            # Evaluate paths that align with A* direction
            best_u, best_s, best_F = 0., 0., float('inf')
            
            if path and len(path) > 1:
                target_pt = path[1] if len(path) > 1 else path[0]
                # Vector to target
                dx = target_pt[0] - b["pos_2d"][0]
                dy = target_pt[1] - b["pos_2d"][1]
                ideal_yaw = math.atan2(dy, dx)
                
                # Yaw diff
                yaw_err = (ideal_yaw - b["yaw"] + math.pi) % (2*math.pi) - math.pi
                
                # Pick best steer
                if yaw_err > 0.2: best_s = -2.
                elif yaw_err < -0.2: best_s = 2.
                else: best_s = 0.
                
                # Pick best u
                best_u = 5. # move forward if aligned
                
                b["target_u"] = best_u
                b["target_s"] = best_s
                b["mode"] = "AUTOPILOT"
            else:
                b["target_u"] = 0.
                b["target_s"] = 0.
                b["mode"] = "SEARCHING"
                
        except Exception as e:
            pass
            
        time._sleep(0.05) # 20Hz update rate for deep planning
# ────────────────────────────────────────────────────────────
"""

# Insert right before main
content = content.replace("def main():", pfc_logic + "\ndef main():")

# 3. Spawning robots further apart
content = content.replace("xw = MAP_XMIN + 0.9 + (i%3)*0.15", "xw = MAP_XMIN + 0.9 + i*2.0")
content = content.replace("yw = MAP_YMIN + 0.9 + (i//3)*0.15", "yw = MAP_YMIN + 0.9 + i*1.0")

# 4. PyBullet Shadows
content = content.replace("p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)",
                          "p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)\n    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)")

# 5. Start PFC Threads in main
thread_start = """
    # Start PFC Threads
    global pfc_active
    pfc_active = True
    pfc_threads = []
    for i in range(N_BODIES):
        t = threading.Thread(target=pfc_worker, args=(i,), daemon=True)
        t.start()
        pfc_threads.append(t)
"""
content = content.replace("for ep in range(1, 10000):", thread_start + "\n    for ep in range(1, 10000):")

# 6. Delete Slime Mold from Main loop
content = re.sub(r'# --- Slime Mold Diffusion.*?(?=if step % 4 == 0:)', '', content, flags=re.DOTALL)

# 7. Replace Main Loop Daughter Minds with Subconscious MPC
old_decision = """                for i, rb in enumerate(p_robots):
                    if brains[i]["alive"]:
                        u, s, name = daughter_minds_maze(
                            brains[i]["xk"], brains[i]["pos_2d"], brains[i]["yaw"],
                            brains[i]["T_wm"], T_ltm, brains[i]["P_wm"],
                            D_global, CM_global, brains[i]["nm"],
                            brains[i]["homeostatic"], swarm_hebbian,
                            brains[i]["hebbian"], brains[i]["predictive"],
                            brains[i]["social"], goal_pos, brains[i].get("grief", 0.)
                        )
                        brains[i]["u"] = u
                        brains[i]["s"] = s
                        brains[i]["mode"] = name"""

new_decision = """                for i, rb in enumerate(p_robots):
                    if brains[i]["alive"]:
                        # Subconscious Cerebellum MPC (Instant Reflex)
                        # Reads target_u and target_s from the Asynchronous PFC
                        target_u = brains[i].get("target_u", 0.)
                        target_s = brains[i].get("target_s", 0.)
                        
                        Ad, Bd = brains[i]["T_wm"][:SDIM,:].T, brains[i]["T_wm"][SDIM,:]
                        x_curr = brains[i]["xk"]
                        
                        best_u, best_cost = 0., float('inf')
                        for test_u in [-8., -5., -2., 0., 2., 5., 8.]:
                            xs_sim = x_curr.copy()
                            cost = 0.
                            for _ in range(10): # Lightning fast 10-step MPC
                                xs_sim = Ad @ xs_sim + Bd * test_u
                                cost += (xs_sim[2]**2)*10.0 + (test_u - target_u)**2
                                if abs(xs_sim[2]) > 0.4: cost += 1000.; break
                            if cost < best_cost:
                                best_cost = cost
                                best_u = test_u
                                
                        brains[i]["u"] = best_u
                        brains[i]["s"] = target_s"""

content = content.replace(old_decision, new_decision)

# 8. Update WS Payload with A* Path
ws_payload_old = """                "cm_scent": cm_scent,
                "cm_phero": cm_phero,"""
ws_payload_new = """                "cm_scent": cm_scent,
                "cm_phero": cm_phero,
                "astar_path": brains[0].get("astar_path", []),
                "robot_x": [b["pos_2d"][0] for b in brains if b["xk"] is not None],
                "robot_y": [b["pos_2d"][1] for b in brains if b["xk"] is not None],"""

content = content.replace(ws_payload_old, ws_payload_new)

# Stop threads on exit
content = content.replace("if __name__ == \"__main__\":", "if __name__ == \"__main__\":\n    pfc_active = False")

with open('phase14_maze.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Asynchronous Tesla PFC applied.")
