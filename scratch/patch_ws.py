import re

with open('phase14_maze.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Update the payload generation
payload_logic_old = """            cm_danger_flat = CM[:,:,0].T.flatten() / (float(np.max(CM[:,:,0]))+1e-6)
            cm_danger_flat = np.round(cm_danger_flat, 3).tolist()

            brain.update({"""

payload_logic_new = """            cm_danger_flat = CM[:,:,0].T.flatten() / (float(np.max(CM[:,:,0]))+1e-6)
            cm_danger_flat = np.round(cm_danger_flat, 3).tolist()
            
            # SLIME MOLD CHANNELS
            cm_scent = np.round(CM[:,:,3].T.flatten(), 3).tolist()
            cm_phero = np.round(CM[:,:,4].T.flatten(), 3).tolist()

            brain.update({
                "cm_scent": cm_scent,
                "cm_phero": cm_phero,"""

content = content.replace(payload_logic_old, payload_logic_new)

# 2. Add cm_scent and cm_phero default empty arrays to brain initialization
brain_init_old = """    "cognitive_map": [0.0]*625,   # 25x25 spatial map"""
brain_init_new = """    "cognitive_map": [0.0]*625,   # 25x25 spatial map
    "cm_scent": [0.0]*625,
    "cm_phero": [0.0]*625,"""

content = content.replace(brain_init_old, brain_init_new)

with open('phase14_maze.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("WS payload patched.")
