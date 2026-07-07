import numpy as np
import heapq

def astar_path(CM, start_pos, goal_pos):
    # CM: 25x25xN, CM[:,:,0] is danger
    MAP_RES = 25
    MAP_XMIN = -0.5
    MAP_XMAX = 7.7
    MAP_YMIN = -0.5
    MAP_YMAX = 6.5
    
    def _map_cell(xw, yw):
        ix = int(np.clip((xw - MAP_XMIN)/(MAP_XMAX - MAP_XMIN)*MAP_RES, 0, MAP_RES-1))
        iy = int(np.clip((yw - MAP_YMIN)/(MAP_YMAX - MAP_YMIN)*MAP_RES, 0, MAP_RES-1))
        return ix, iy

    sx, sy = _map_cell(start_pos[0], start_pos[1])
    gx, gy = _map_cell(goal_pos[0], goal_pos[1])
    
    # 0 = free, 1 = wall (only pre-seeded walls > 0.85 are absolute)
    grid = (CM[:, :, 0] > 0.85).astype(int)
    
    if grid[gx, gy] == 1:
        # Goal is in a wall (shouldn't happen, but just in case)
        pass
        
    open_set = []
    heapq.heappush(open_set, (0, sx, sy))
    came_from = {}
    
    g_score = {(x,y): float('inf') for x in range(MAP_RES) for y in range(MAP_RES)}
    g_score[(sx, sy)] = 0
    
    f_score = {(x,y): float('inf') for x in range(MAP_RES) for y in range(MAP_RES)}
    f_score[(sx, sy)] = abs(sx - gx) + abs(sy - gy)
    
    while open_set:
        _, cx, cy = heapq.heappop(open_set)
        
        if cx == gx and cy == gy:
            # Reconstruct path
            path = []
            curr = (cx, cy)
            while curr in came_from:
                path.append(curr)
                curr = came_from[curr]
            path.reverse()
            
            # Convert grid path back to continuous coordinates
            cont_path = []
            for (px, py) in path:
                c_x = MAP_XMIN + (px + 0.5) / MAP_RES * (MAP_XMAX - MAP_XMIN)
                c_y = MAP_YMIN + (py + 0.5) / MAP_RES * (MAP_YMAX - MAP_YMIN)
                cont_path.append([c_x, c_y])
            return cont_path
            
        for dx, dy in [(0,1), (1,0), (0,-1), (-1,0), (1,1), (-1,1), (1,-1), (-1,-1)]:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < MAP_RES and 0 <= ny < MAP_RES:
                # No absolute walls - everything is just a cost
                danger_cost = CM[nx, ny, 0] * 50.0 # High penalty for pre-seeded walls (0.9) and death spots (1.0)
                step_cost = (1.414 if dx != 0 and dy != 0 else 1.0)
                tentative_g = g_score[(cx, cy)] + step_cost + danger_cost
                
                if tentative_g < g_score[(nx, ny)]:
                    came_from[(nx, ny)] = (cx, cy)
                    g_score[(nx, ny)] = tentative_g
                    f = tentative_g + abs(nx - gx) + abs(ny - gy)
                    f_score[(nx, ny)] = f
                    heapq.heappush(open_set, (f, nx, ny))
                    
    return [] # No path found
