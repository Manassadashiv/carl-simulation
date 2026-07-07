import math
import random
import numpy as np
import mujoco
from mujoco import viewer

def generate_maze(rows=5, cols=5, difficulty=3, seed=None):
    """
    Generates a random maze layout using Randomized Prim's algorithm.
    Difficulty levels:
        1: Open arena, 2-3 walls
        2: Simple corridors, 5-7 walls
        3: Full maze, 10-15 walls
        4: Dense maze (perfect maze, ~16 walls)
    Returns: dict with 'walls' and 'open_cells'
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        
    # Grid of cells. True means part of the maze.
    maze = [[False for _ in range(cols)] for _ in range(rows)]
    
    # Possible walls. Format: ((r1, c1), (r2, c2))
    # Horizontal walls: ((r, c), (r+1, c)) for r in 0..rows-2, c in 0..cols-1
    # Vertical walls: ((r, c), (r, c+1)) for r in 0..rows-1, c in 0..cols-2
    all_walls = []
    for r in range(rows - 1):
        for c in range(cols):
            all_walls.append(((r, c), (r + 1, c)))
    for r in range(rows):
        for c in range(cols - 1):
            all_walls.append(((r, c), (r, c + 1)))
            
    # List of walls in the generated perfect maze
    # We start with all walls present (solid grid) and remove them to make passages
    active_walls = list(all_walls)
    
    # Pick a random starting cell
    start_r = random.randint(0, rows - 1)
    start_c = random.randint(0, cols - 1)
    maze[start_r][start_c] = True
    
    # Find walls connected to the starting cell
    wall_list = []
    def add_walls(r, c):
        for w in all_walls:
            if w[0] == (r, c) or w[1] == (r, c):
                if w not in wall_list:
                    wall_list.append(w)
                    
    add_walls(start_r, start_c)
    
    while wall_list:
        # Pick a random wall
        idx = random.randint(0, len(wall_list) - 1)
        w = wall_list.pop(idx)
        
        cell1, cell2 = w
        # If exactly one of the two cells divided by the wall is visited
        if maze[cell1[0]][cell1[1]] ^ maze[cell2[0]][cell2[1]]:
            # Make it a passage (remove from active_walls)
            if w in active_walls:
                active_walls.remove(w)
            
            # Mark the unvisited cell as part of the maze
            unvisited = cell2 if maze[cell1[0]][cell1[1]] else cell1
            maze[unvisited[0]][unvisited[1]] = True
            
            # Add the neighboring walls of the cell to the wall list
            add_walls(unvisited[0], unvisited[1])
            
    # active_walls now contains a perfect maze (16 walls for 5x5)
    
    # Adjust difficulty by removing walls
    if difficulty == 1:
        target_walls = random.randint(2, 3)
    elif difficulty == 2:
        target_walls = random.randint(5, 7)
    elif difficulty == 3:
        target_walls = random.randint(10, 15)
    else:  # 4
        target_walls = len(active_walls)
        
    while len(active_walls) > target_walls:
        active_walls.remove(random.choice(active_walls))
        
    # Convert to physical properties
    # Grid spacing 1.8m. Center at 0,0.
    # r=0 is +Y, r=4 is -Y. c=0 is -X, c=4 is +X
    # Center of cell (r, c): X = (c - (cols-1)/2) * 1.8, Y = ((rows-1)/2 - r) * 1.8
    physical_walls = []
    for w in active_walls:
        (r1, c1), (r2, c2) = w
        if r1 == r2:  # Vertical wall (same row, different cols)
            c = min(c1, c2)
            # Center of wall is between c and c+1
            x = (c + 0.5 - (cols - 1) / 2) * 1.8
            y = ((rows - 1) / 2 - r1) * 1.8
            physical_walls.append({
                'name': f"dwall_v_{r1}_{c}",
                'pos': [x, y, 0.25],
                'size': [0.05, 0.9, 0.25],
                'type': 'box'
            })
        else:  # Horizontal wall (different rows, same col)
            r = min(r1, r2)
            # Center of wall is between r and r+1
            x = (c1 - (cols - 1) / 2) * 1.8
            y = ((rows - 1) / 2 - (r + 0.5)) * 1.8
            physical_walls.append({
                'name': f"dwall_h_{r}_{c1}",
                'pos': [x, y, 0.25],
                'size': [0.9, 0.05, 0.25],
                'type': 'box'
            })
            
    open_cells = []
    for r in range(rows):
        for c in range(cols):
            x = (c - (cols - 1) / 2) * 1.8
            y = ((rows - 1) / 2 - r) * 1.8
            open_cells.append((x, y))
            
    return {'walls': physical_walls, 'open_cells': open_cells}

def apply_maze_to_model(xml_path, maze_data):
    """
    Generates a new MuJoCo model from the original XML by adding the dynamic walls.
    Because we cannot add geoms dynamically in Python easily, we parse the XML string,
    insert the walls, and compile it.
    """
    with open(xml_path, 'r') as f:
        xml_content = f.read()
        
    # Find the closing </worldbody> tag
    insert_pos = xml_content.rfind('</worldbody>')
    if insert_pos == -1:
        raise ValueError("Could not find </worldbody> in XML.")
        
    walls_xml = "\n    <!-- PROCEDURAL MAZE WALLS -->\n"
    for w in maze_data['walls']:
        pos_str = f"{w['pos'][0]:.2f} {w['pos'][1]:.2f} {w['pos'][2]:.2f}"
        size_str = f"{w['size'][0]:.2f} {w['size'][1]:.2f} {w['size'][2]:.2f}"
        walls_xml += f'    <geom name="{w["name"]}" type="{w["type"]}" size="{size_str}" pos="{pos_str}" material="wall_mat"/>\n'
        
    new_xml = xml_content[:insert_pos] + walls_xml + xml_content[insert_pos:]
    return mujoco.MjModel.from_xml_string(new_xml)

def get_valid_spawn_positions(maze_data):
    """Returns list of (x,y) positions that are inside open cells."""
    return maze_data['open_cells']

def scatter_objects(model, data, maze_data, object_names=None):
    """Randomly places free-body objects in reachable maze cells."""
    if object_names is None:
        object_names = ['obj_cube_0', 'obj_cube_1', 'obj_cube_2', 'obj_ball_0', 'obj_ball_1', 'obj_toy']
        
    spawn_points = get_valid_spawn_positions(maze_data)
    # Don't spawn objects too close to center (where CARL starts)
    spawn_points = [p for p in spawn_points if math.hypot(p[0], p[1]) > 1.5]
    
    for obj_name in object_names:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, obj_name)
        if body_id == -1:
            continue
            
        # Freejoints are stored in qpos. Find the qpos address.
        # But we can also set it indirectly via qpos if we know the joint.
        jnt_id = model.body_jntadr[body_id]
        if jnt_id != -1 and model.jnt_type[jnt_id] == mujoco.mjtJoint.mjJNT_FREE:
            qpos_adr = model.jnt_qposadr[jnt_id]
            pt = random.choice(spawn_points)
            # Add some jitter
            x = pt[0] + random.uniform(-0.5, 0.5)
            y = pt[1] + random.uniform(-0.5, 0.5)
            
            data.qpos[qpos_adr] = x
            data.qpos[qpos_adr+1] = y
            data.qpos[qpos_adr+2] = 0.05 # Drop from slightly above ground
            # Reset velocities
            dof_adr = model.jnt_dofadr[jnt_id]
            data.qvel[dof_adr:dof_adr+6] = 0.0
            
if __name__ == '__main__':
    print("Generating Level 3 Maze...")
    maze_data = generate_maze(difficulty=3, seed=42)
    
    print(f"Generated {len(maze_data['walls'])} walls.")
    
    xml_path = "vessel_kinetic.xml"
    
    try:
        model = apply_maze_to_model(xml_path, maze_data)
        data = mujoco.MjData(model)
        
        # Strip original labyrinth walls to avoid overlap for visualization
        # In a real run, you'd want to load a clean XML without the static labyrinth
        # But here they might overlap.
        
        scatter_objects(model, data, maze_data)
        
        # Run viewer
        mujoco.mj_forward(model, data)
        print("Launching viewer. Close window to exit.")
        viewer.launch(model, data)
    except Exception as e:
        print(f"Failed to load generated maze into MuJoCo: {e}")
