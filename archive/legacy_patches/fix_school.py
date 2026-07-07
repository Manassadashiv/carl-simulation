#!/usr/bin/env python3
"""Script to replace the complex action selection block in phase17_school.py 
with a simple PID controller."""
import re

with open('phase17_school.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find the start line (BUG-FIX 2+3+4)
start_idx = None
for i, line in enumerate(lines):
    if 'BUG-FIX 2+3+4' in line:
        start_idx = i - 1  # Include the dashed line before it
        break

# Find the end line (rb['mode'] = mode)
end_idx = None
for i in range(start_idx, len(lines)):
    if "rb['mode'] = mode" in lines[i]:
        end_idx = i
        break

print(f"Replacing lines {start_idx+1} to {end_idx+1}")
print(f"Old block: {end_idx - start_idx + 1} lines (skipping print due to unicode)")

replacement = '''            # SCHOOL: Simple PID Controller
            # Balance + Drive Forward + Follow A* Path
            
            # 1. BALANCE: PID on pitch
            pitch_err = float(xn[2])
            pitch_vel = float(xn[3])
            balance_u = -(15.0 * pitch_err + 2.0 * pitch_vel)
            
            # 2. DRIVE FORWARD: Constant forward bias 
            forward_bias = 3.0
            
            # 3. STEER: Follow A* path waypoint
            pfc_path = rb.get('astar_path', [])
            if pfc_path and len(pfc_path) > 1:
                dx = pfc_path[1][0] - xw
                dy = pfc_path[1][1] - yw
                ideal_yaw = math.atan2(dy, dx)
                yaw_err = (ideal_yaw - rb['yaw'] + math.pi) % (2*math.pi) - math.pi
                path_steer = float(np.clip(-yaw_err * 2.0, -2.5, 2.5))
            else:
                dx = goal_pos[0] - xw
                dy = goal_pos[1] - yw
                ideal_yaw = math.atan2(dy, dx)
                yaw_err = (ideal_yaw - rb['yaw'] + math.pi) % (2*math.pi) - math.pi
                path_steer = float(np.clip(-yaw_err * 2.0, -2.5, 2.5))
            
            # 4. LIDAR OVERRIDE: If wall is very close, override steer to dodge
            left_open = lidar_dists[0] + lidar_dists[1]*0.5
            right_open = lidar_dists[4] + lidar_dists[3]*0.5
            lidar_steer = float(np.clip((right_open - left_open) * 2.5, -3.0, 3.0))
            radar_dominance = float(np.clip(1.0 - (min_lidar / 0.6), 0.0, 1.0))
            
            if min_lidar < 0.3:
                forward_bias = -2.0  # Reverse!
            elif min_lidar < 0.5:
                forward_bias = 0.0   # Stop
            
            sn = (radar_dominance * lidar_steer) + ((1.0 - radar_dominance) * path_steer)
            un = float(np.clip(balance_u + forward_bias, -8., 8.))
            mode = 'SCHOOL-PID'
'''

new_lines = lines[:start_idx] + [replacement] + lines[end_idx+1:]

with open('phase17_school.py', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print(f"\nDone! Replaced {end_idx - start_idx + 1} lines with PID controller.")
