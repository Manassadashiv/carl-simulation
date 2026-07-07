import traceback

with open('phase14_maze.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Fix CM_global
content = content.replace("CM = np.zeros", "global CM_global\n    CM_global = np.zeros")
content = content.replace("cognitive_map_danger(CM,", "cognitive_map_danger(CM_global,")

# 2. Fix the swallowed exception in pfc_worker
old_pfc_except = """        except Exception as e:
            pass"""
new_pfc_except = """        except Exception as e:
            print("PFC ERROR:", e)
            traceback.print_exc()"""
content = content.replace(old_pfc_except, new_pfc_except)
content = "import traceback\n" + content

with open('phase14_maze.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("PFC exception fixed.")
