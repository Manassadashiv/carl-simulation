import traceback

with open('phase14_maze.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Make brains global so the thread can see it!
content = content.replace("brains = [spawn_robot_brain(T_ltm_ep, P_ltm_ep, i) for i in range(N_BODIES)]",
                          "global brains\n    brains = [spawn_robot_brain(T_ltm_ep, P_ltm_ep, i) for i in range(N_BODIES)]")

with open('phase14_maze.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("brains globalized.")
