import re

with open('phase14_maze.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace everything from "# Action selection" to "mode = "EXPLOITING""
action_select_regex = r'# Action selection\n.*?(?=rb\["uk"\] = un)'

new_action_select = """# Action selection (Fast Subconscious Reflex)
                target_u = brain.get("target_u", 0.)
                target_s = brain.get("target_s", 0.)
                
                Ad, Bd = rb["T_wm"][:SDIM,:].T, rb["T_wm"][SDIM,:]
                
                best_u, best_cost = 0., float('inf')
                for test_u in [-8., -5., -2., 0., 2., 5., 8.]:
                    xs_sim = xn.copy()
                    cost = 0.
                    for _ in range(10): # Lightning fast 10-step MPC
                        xs_sim = Ad @ xs_sim + Bd * test_u
                        cost += (xs_sim[2]**2)*10.0 + (test_u - target_u)**2
                        if abs(xs_sim[2]) > 0.4: cost += 1000.; break
                    if cost < best_cost:
                        best_cost = cost
                        best_u = test_u
                        
                un = best_u
                sn = target_s
                mode = brain.get("mode", "AUTOPILOT")
                """

content = re.sub(action_select_regex, new_action_select, content, flags=re.DOTALL)

with open('phase14_maze.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Action selection replaced with Reflex MPC.")
