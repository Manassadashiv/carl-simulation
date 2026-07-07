import numpy as np

with open('phase14_maze.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if "for (gx1, gy1, gx2, gy2) in MAZE_WALLS_GRID:" in line:
        if "for (gx1, gy1, gx2, gy2) in MAZE_WALLS_GRID:" in lines[i+1]:
            lines[i+1] = "" # remove duplicate
            
    if "px = 0.3 + (i % 4) * 0.25" in line:
        lines[i] = line.replace("0.25", "0.15")
    if "py = 0.3 + (i // 4) * 0.25" in line:
        lines[i] = line.replace("0.25", "0.15")

with open('phase14_maze.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print("Patch applied.")
