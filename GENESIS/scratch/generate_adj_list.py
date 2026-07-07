# Grid cells
grid_x = [-3.6, -1.8, 0.0, 1.8, 3.6]
grid_y = [3.6, 1.8, 0.0, -1.8, -3.6]

blocked = set()

def block_v(c, r_start, r_end):
    for r in range(r_start, r_end + 1):
        blocked.add(((r, c), (r, c+1)))
        blocked.add(((r, c+1), (r, c)))

def block_h(r, c_start, c_end):
    for c in range(c_start, c_end + 1):
        blocked.add(((r, c), (r+1, c)))
        blocked.add(((r+1, c), (r, c)))

# Exact vertical walls from generator output:
block_v(2, 0, 0)
block_v(0, 2, 2)
block_v(2, 2, 2)
block_v(1, 3, 3)

# Exact horizontal walls from generator output:
block_h(0, 1, 1)
block_h(1, 1, 1)
block_h(2, 0, 0)
block_h(2, 2, 2)
block_h(2, 4, 4)
block_h(3, 2, 3)

valid_nodes = {}
for r in range(5):
    for c in range(5):
        node = (grid_x[c], grid_y[r])
        neighbors = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < 5 and 0 <= nc < 5:
                if ((r, c), (nr, nc)) not in blocked:
                    neighbors.append((grid_x[nc], grid_y[nr]))
        valid_nodes[node] = neighbors

print("self.valid_nodes = {")
for k, v in valid_nodes.items():
    print(f"    {k}: {v},")
print("}")
