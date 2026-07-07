import random

# Grid dimensions
rows = 5
cols = 5

# Cell coordinates
# x = -3.6, -1.8, 0.0, 1.8, 3.6 (columns 0 to 4)
# y = 3.6, 1.8, 0.0, -1.8, -3.6 (rows 0 to 4)
grid_x = [-3.6, -1.8, 0.0, 1.8, 3.6]
grid_y = [3.6, 1.8, 0.0, -1.8, -3.6]

# Initialize all walls as present
# Horizontal walls exist between (r, c) and (r+1, c) for r in range(4), c in range(5)
# Vertical walls exist between (r, c) and (r, c+1) for r in range(5), c in range(4)
h_walls = {(r, c): True for r in range(4) for c in range(5)}
v_walls = {(r, c): True for r in range(5) for c in range(4)}

# Randomized DFS to generate a spanning tree (no loops initially, all cells connected)
visited = set()
def dfs(r, c):
    visited.add((r, c))
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    random.shuffle(directions)
    for dr, dc in directions:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited:
            # Remove the wall between them
            if dr == -1: # Move up, remove horizontal wall below nr
                h_walls[(nr, nc)] = False
            elif dr == 1: # Move down, remove horizontal wall below r
                h_walls[(r, c)] = False
            elif dr == 0 and dc == -1: # Move left, remove vertical wall right of nc
                v_walls[(r, nc)] = False
            elif dr == 0 and dc == 1: # Move right, remove vertical wall right of c
                v_walls[(r, c)] = False
            dfs(nr, nc)

# Start DFS from center cell (2, 2) i.e. C3 (0,0)
dfs(2, 2)

# Now, we "relax" the maze to turn it into a braid maze (introducing loops/alternative paths)
# We randomly remove 35% of the remaining walls
all_remaining_h = [k for k, v in h_walls.items() if v]
all_remaining_v = [k for k, v in v_walls.items() if v]

random.seed(42)  # Seed for deterministic generation of the maze
random.shuffle(all_remaining_h)
random.shuffle(all_remaining_v)

# Remove 35% of horizontal remaining walls
num_to_remove_h = int(len(all_remaining_h) * 0.35)
for k in all_remaining_h[:num_to_remove_h]:
    h_walls[k] = False

# Remove 35% of vertical remaining walls
num_to_remove_v = int(len(all_remaining_v) * 0.35)
for k in all_remaining_v[:num_to_remove_v]:
    v_walls[k] = False

# Print XML geoms
print("<!-- SIGMA LABYRINTH INTERNAL WALLS -->")
wall_count = 0

# Vertical walls between col c and c+1
# Pos x is halfway between grid_x[c] and grid_x[c+1]
# Pos y is grid_y[r]
# Size: 0.05 (x half-width) x 0.9 (y half-length, since grid spacing is 1.8m) x 0.25 (z half-height)
for (r, c), present in v_walls.items():
    if present:
        wx = (grid_x[c] + grid_x[c+1]) / 2.0
        wy = grid_y[r]
        name = f"lwall_v_{r}_{c}"
        print(f'<geom name="{name}" type="box" size="0.05 0.9 0.25" pos="{wx:.2f} {wy:.2f} 0.25" material="wall_mat"/>')
        wall_count += 1

# Horizontal walls between row r and r+1
# Pos x is grid_x[c]
# Pos y is halfway between grid_y[r] and grid_y[r+1]
# Size: 0.9 (x half-length) x 0.05 (y half-width) x 0.25 (z half-height)
for (r, c), present in h_walls.items():
    if present:
        wx = grid_x[c]
        wy = (grid_y[r] + grid_y[r+1]) / 2.0
        name = f"lwall_h_{r}_{c}"
        print(f'<geom name="{name}" type="box" size="0.9 0.05 0.25" pos="{wx:.2f} {wy:.2f} 0.25" material="wall_mat"/>')
        wall_count += 1

print(f"\nGenerated {wall_count} wall segments.")
