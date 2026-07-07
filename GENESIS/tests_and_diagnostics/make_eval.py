import re

with open('carl_train.py', 'r') as f:
    train_content = f.read()

header = re.split(r'^try:\s+while True:', train_content, flags=re.MULTILINE)[0]

eval_code = """
print("\\n============================================================")
print("  CARL v4 STRESS TEST")
print("  100 Episodes. Pure Exploitation (No Noise). Fully Random Spawns.")
print("============================================================\\n")

test_episodes = 100
ep_count = 0
success_count = 0
wall_count = 0
timeout_count = 0
step_count = 0

while ep_count < test_episodes:
    # ── Respawning ────────────────────────────────────────────────────────
    if step_count == 0:
        while True:
            cx = np.random.uniform(-4.0, 4.0)
            cy = np.random.uniform(-4.0, 4.0)
            if math.hypot(cx, cy) > 1.2:
                break
        data.qpos[Q] = cx
        data.qpos[Q+1] = cy
        data.qpos[Q+2] = 0.15
        rand_yaw = np.random.uniform(-math.pi, math.pi)
        data.qpos[Q+3:Q+7] = set_yaw(rand_yaw)
        data.qvel[:] = 0.0

        while True:
            fx = np.random.uniform(-4.0, 4.0)
            fy = np.random.uniform(-4.0, 4.0)
            if math.hypot(fx, fy) > 1.0:
                break
        data.qpos[7:10] = [fx, fy, 0.12]
        data.qpos[10:14] = [1, 0, 0, 0]
        data.qvel[6:12] = 0.0
        mujoco.mj_forward(model, data)
        print(f"Episode {ep_count+1:03d} starting...", flush=True)

    # ── Sensing ───────────────────────────────────────────────────────────
    lidars    = get_lidar(8)
    proximity = lidar_to_proximity(lidars)
    yaw      = get_yaw(data.qpos[Q+3:Q+7])
    vel_x    = float(data.qvel[V])
    vel_y    = float(data.qvel[V+1])
    vel_fwd  =  vel_x * math.cos(yaw) + vel_y * math.sin(yaw)
    vel_lat  = -vel_x * math.sin(yaw) + vel_y * math.cos(yaw)
    omega    = float(data.qvel[V+5])

    food_cos, food_sin, dist = get_food_direction()

    obs = build_observation(
        min(lidars), food_cos, food_sin, vel_fwd, vel_lat, omega,
        brain.drives, 0.0, 0.0
    )

    # ── Thinking (No noise) ───────────────────────────────────────────────
    import torch
    state_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
    with torch.no_grad():
        action = brain.actor(state_tensor).cpu().numpy()[0]
    throttle, steering = float(action[0]), float(action[1])

    # ── Acting ────────────────────────────────────────────────────────────
    left_tread  = throttle - steering
    right_tread = throttle + steering
    data.ctrl[0] = np.clip(left_tread * 2.0, -2.0, 2.0)
    data.ctrl[1] = np.clip(right_tread * 2.0, -2.0, 2.0)

    wall_proximity_override(yaw)

    # ── Physics Step ──────────────────────────────────────────────────────
    mujoco.mj_step(model, data)
    step_count += 1

    # ── Event Checking ────────────────────────────────────────────────────
    event = check_event()
    if event == 'food':
        success_count += 1
        ep_count += 1
        print(f"[SUCCESS] Ep {ep_count:03d}/{test_episodes} | Found food in {step_count} steps.")
        step_count = 0
    elif event == 'wall':
        wall_count += 1
        ep_count += 1
        print(f"[FAIL] Ep {ep_count:03d}/{test_episodes} | Hit wall in {step_count} steps.")
        step_count = 0
    elif step_count > 2000:
        timeout_count += 1
        ep_count += 1
        print(f"[TIMEOUT] Ep {ep_count:03d}/{test_episodes} | Timed out after 2000 steps.")
        step_count = 0

print("\\n============================================================")
print("  STRESS TEST COMPLETE")
print(f"  Total Episodes : {test_episodes}")
print(f"  Successes      : {success_count} ({(success_count/test_episodes)*100:.1f}%)")
print(f"  Wall Hits      : {wall_count} ({(wall_count/test_episodes)*100:.1f}%)")
print(f"  Timeouts       : {timeout_count} ({(timeout_count/test_episodes)*100:.1f}%)")
print("============================================================\\n")
"""

with open('carl_eval.py', 'w', encoding='utf-8') as f:
    f.write(header + eval_code)
