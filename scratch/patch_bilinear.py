import re

with open('phase14_maze.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace get_olfactory_reward with a bilinearly interpolated version!
old_olf = """def get_olfactory_reward(CM, xw, yw):
    i, j = _map_cell(xw, yw)
    goal_s = CM[i, j, 3]
    agent_p = CM[i, j, 4]
    return goal_s * 500.0 + agent_p * 20.0"""

new_olf = """def get_olfactory_reward(CM, xw, yw):
    # Bilinear interpolation of the grid so the reward is continuous!
    x_pct = (xw - MAP_XMIN) / (MAP_XMAX - MAP_XMIN) * MAP_RES
    y_pct = (yw - MAP_YMIN) / (MAP_YMAX - MAP_YMIN) * MAP_RES
    
    x0 = int(np.clip(math.floor(x_pct), 0, MAP_RES-1))
    x1 = int(np.clip(math.ceil(x_pct), 0, MAP_RES-1))
    y0 = int(np.clip(math.floor(y_pct), 0, MAP_RES-1))
    y1 = int(np.clip(math.ceil(y_pct), 0, MAP_RES-1))
    
    dx = x_pct - x0
    dy = y_pct - y0
    
    def b_interp(c):
        c00 = CM[x0, y0, c]
        c10 = CM[x1, y0, c]
        c01 = CM[x0, y1, c]
        c11 = CM[x1, y1, c]
        return c00*(1-dx)*(1-dy) + c10*dx*(1-dy) + c01*(1-dx)*dy + c11*dx*dy

    goal_s = b_interp(3)
    agent_p = b_interp(4)
    
    return goal_s * 500.0 + agent_p * 20.0"""

content = content.replace(old_olf, new_olf)

with open('phase14_maze.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Bilinear interpolation patched.")
