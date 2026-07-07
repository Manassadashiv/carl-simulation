import numpy as np

# Grid cells: 5x5
cells = [(r, c) for r in range(5) for c in range(5)]
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
# lwall_v_0_2 -> c=2, r=0
# lwall_v_2_0 -> c=0, r=2
# lwall_v_2_2 -> c=2, r=2
# lwall_v_3_1 -> c=1, r=3
block_v(2, 0, 0)
block_v(0, 2, 2)
block_v(2, 2, 2)
block_v(1, 3, 3)

# Exact horizontal walls from generator output:
# lwall_h_0_1 -> r=0, c=1
# lwall_h_1_1 -> r=1, c=1
# lwall_h_2_0 -> r=2, c=0
# lwall_h_2_2 -> r=2, c=2
# lwall_h_2_4 -> r=2, c=4
# lwall_h_3_2 -> r=3, c=2
# lwall_h_3_3 -> r=3, c=3
block_h(0, 1, 1)
block_h(1, 1, 1)
block_h(2, 0, 0)
block_h(2, 2, 2)
block_h(2, 4, 4)
block_h(3, 2, 3)

# Perform BFS starting from center cell (2, 2)
visited = set([(2, 2)])
queue = [(2, 2)]

while queue:
    curr = queue.pop(0)
    r, c = curr
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < 5 and 0 <= nc < 5:
            neighbor = (nr, nc)
            if (curr, neighbor) not in blocked:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

print(f"Total cells: {len(cells)}")
print(f"Reachable cells: {len(visited)}")
unreachable = set(cells) - visited
if unreachable:
    print(f"Unreachable: {unreachable}")
else:
    print("ALL CELLS ARE REACHABLE! Graph is fully connected.")
