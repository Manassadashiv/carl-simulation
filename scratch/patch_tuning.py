import re

with open('dashboard/maze_dashboard.html', 'r', encoding='utf-8') as f:
    html = f.read()

# Make the Scent and Pheromones visible even at low concentrations
html = html.replace('if (s > 0.01) {', 'if (s > 0.0005) {')
# Scale visual brightness for scent
html = html.replace('Math.round(s * 200);', 'Math.round(s * 5000);')
html = html.replace('Math.round(s * 255);', 'Math.round(s * 6000);')

with open('dashboard/maze_dashboard.html', 'w', encoding='utf-8') as f:
    f.write(html)

with open('phase14_maze.py', 'r', encoding='utf-8') as f:
    py = f.read()

# Decrease scent decay from 0.99 to 0.999
py = py.replace('new_scent *= 0.99', 'new_scent *= 0.999')

# Increase scent reward multiplier massively so the gradient is felt far away
py = py.replace('goal_s * 20.0 + agent_p * 8.0', 'goal_s * 500.0 + agent_p * 20.0')

with open('phase14_maze.py', 'w', encoding='utf-8') as f:
    f.write(py)

print("Patch applied.")
