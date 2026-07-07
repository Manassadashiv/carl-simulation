import re

with open('phase14_maze.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Update CM initialization
content = content.replace("return np.zeros((MAP_RES, MAP_RES, 3))", "return np.zeros((MAP_RES, MAP_RES, 5))")

# 2. Add Slime Mold functions right after `cognitive_map_update`
diffusion_logic = """
def diffuse_scent(CM, goal_pos):
    gi, gj = _map_cell(goal_pos[0], goal_pos[1])
    scent = CM[:, :, 3]
    danger = CM[:, :, 0]
    
    # Vectorised Laplacian
    ns = np.zeros_like(scent)
    ns[1:-1, 1:-1] = (scent[:-2, 1:-1] + scent[2:, 1:-1] + scent[1:-1, :-2] + scent[1:-1, 2:])
    
    new_scent = scent.copy()
    new_scent[1:-1, 1:-1] += 0.25 * (ns[1:-1, 1:-1] - 4*scent[1:-1, 1:-1])
    
    # Apply danger block (pruning dead ends)
    new_scent[danger > 0.5] = 0.0
    
    # Decay
    new_scent *= 0.99
    
    # Clamp goal (The Oat Flake)
    new_scent[gi, gj] = 1.0
    
    CM[:, :, 3] = new_scent
    CM[:, :, 4] *= 0.995 # decay agent pheromone
    return CM

def get_olfactory_reward(CM, xw, yw):
    i, j = _map_cell(xw, yw)
    goal_s = CM[i, j, 3]
    agent_p = CM[i, j, 4]
    return goal_s * 20.0 + agent_p * 8.0
"""
if "def diffuse_scent" not in content:
    content = content.replace("def cognitive_map_update(CM, xw, yw, danger, trust_delta, ghost=0.):", diffusion_logic + "\ndef cognitive_map_update(CM, xw, yw, danger, trust_delta, ghost=0.):")


# 3. Replace Euclidean dist_cost in pick_action_maze
def replace_pick_action():
    old = """                spatial_t  = 1. - cognitive_map_trust(CM, xw, yw)
                dist_2d    = math.sqrt((xw - gx)**2 + (yw - gy)**2)
                dist_cost  = dop_w * dist_2d * 1.5
                pred_err   = float(np.linalg.norm(xs_n - xs)) * 0.02
                hebb_cost  = (0.7 * swarm_hebbian.association_cost(xs_n[2], xs_n[3], u)
                            + 0.3 * hebbian.association_cost(xs_n[2], xs_n[3], u))
                homeo_corr = abs(homeostatic.correction_signal(xs_n[2])) * 0.1
                raw_cost   = (pred_err
                            + (kin_danger + spatial_d) * soc_mod
                            + 0.1 * spatial_t
                            + dist_cost + hebb_cost * 0.1 + homeo_corr)"""
    new = """                spatial_t  = 1. - cognitive_map_trust(CM, xw, yw)
                olfactory  = get_olfactory_reward(CM, xw, yw) * dop_w
                pred_err   = float(np.linalg.norm(xs_n - xs)) * 0.02
                hebb_cost  = (0.7 * swarm_hebbian.association_cost(xs_n[2], xs_n[3], u)
                            + 0.3 * hebbian.association_cost(xs_n[2], xs_n[3], u))
                homeo_corr = abs(homeostatic.correction_signal(xs_n[2])) * 0.1
                raw_cost   = (pred_err
                            + (kin_danger + spatial_d) * soc_mod
                            + 0.1 * spatial_t
                            - olfactory + hebb_cost * 0.1 + homeo_corr)"""
    return old, new

old1, new1 = replace_pick_action()
content = content.replace(old1, new1)

# 4. Replace Euclidean dist_cost in daughter_minds_maze (Leg 1 & Leg 2)
# Using regex for robust replacement since it might differ slightly
content = re.sub(r'd2\s*=\s*math\.sqrt\(\(xw-gx\)\*\*2 \+ \(yw-gy\)\*\*2\)', r'olf1 = get_olfactory_reward(CM, xw, yw)', content)
content = re.sub(r'd3\s*=\s*math\.sqrt\(\(xw2-gx\)\*\*2 \+ \(yw2-gy\)\*\*2\)', r'olf2 = get_olfactory_reward(CM, xw2, yw2)', content)

content = re.sub(r'dop_w\*d2\*1\.5', r'-dop_w*olf1', content)
content = re.sub(r'dop_w\*d3\*1\.5', r'-dop_w*olf2', content)

content = re.sub(r'base_dop_w\*d2\*1\.5', r'-base_dop_w*olf1', content)
content = re.sub(r'base_dop_w\*d3\*1\.5', r'-base_dop_w*olf2', content)

# 5. Add Main Loop Pheromone and Diffusion
main_loop_injection = """            # --- Slime Mold Diffusion & Stigmergy ---
            if step % 24 == 0:
                for b in brains:
                    if b["alive"] and b["nm"].DA > 0.4:
                        ix, iy = _map_cell(b["pos_2d"][0], b["pos_2d"][1])
                        CM_global[ix, iy, 4] = min(1.0, CM_global[ix, iy, 4] + 0.1)
            
            if step % 12 == 0:
                CM_global = diffuse_scent(CM_global, goal_pos)
            # ----------------------------------------
            """

if "Slime Mold Diffusion" not in content:
    content = content.replace("for step in range(500_000):  # ~300s max per episode", "for step in range(500_000):  # ~300s max per episode\n" + main_loop_injection)

# Write back
with open('phase14_maze.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Patch applied successfully.")
