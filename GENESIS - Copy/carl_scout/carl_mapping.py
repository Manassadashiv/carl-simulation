import numpy as np
import os
import math

class OccupancyGrid:
    def __init__(self, resolution=0.20, size=50):
        self.res = resolution
        self.size = size
        self.grid = np.zeros((size, size), dtype=np.float32)
        self.visited = np.zeros((size, size), dtype=bool)
        self.filepath = 'memory/map_cache.npz'

    def world_to_grid(self, x, y):
        gx = int((x + (self.size * self.res) / 2) / self.res)
        gy = int((y + (self.size * self.res) / 2) / self.res)
        return gx, gy

    def update(self, pose, lidar_data):
        # pose is (x, y, theta)
        # lidar_data is 24 rays
        px, py, pth = pose
        
        # Mark current position cell as visited
        gx, gy = self.world_to_grid(px, py)
        if 0 <= gx < self.size and 0 <= gy < self.size:
            self.visited[gx, gy] = True
            
        for i, dist in enumerate(lidar_data):
            angle = pth + (i * (2 * math.pi / 24))
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)
            
            # Ray casting for the hit
            hx = px + dist * cos_a
            hy = py + dist * sin_a
            
            # [FIXED] Only write a wall boundary if the ray actually hit an object
            if dist < 4.8:
                hgx, hgy = self.world_to_grid(hx, hy)
                if 0 <= hgx < self.size and 0 <= hgy < self.size:
                    self.grid[hgx, hgy] = 1.0 # Wall
                
            # Mark all cells along the ray as visited (observed) and clear walls along the free path
            # Step in increments of self.res
            steps = int(dist / self.res)
            res_cos = self.res * cos_a
            res_sin = self.res * sin_a
            for s in range(steps + 1):
                rx = px + s * res_cos
                ry = py + s * res_sin
                rgx, rgy = self.world_to_grid(rx, ry)
                if 0 <= rgx < self.size and 0 <= rgy < self.size:
                    self.visited[rgx, rgy] = True
                    # Clear wall state along the ray before the hit point
                    if s < steps:
                        self.grid[rgx, rgy] = 0.0

    @property
    def confidence(self):
        # Calculate confidence as fraction of visited walkable cells
        # For simplicity, we can define it relative to total cells in size x size
        return np.sum(self.visited) / (self.size * self.size)

    def save(self):
        # Make sure directory exists
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        np.savez_compressed(self.filepath, grid=self.grid, visited=self.visited)

    def load(self):
        if os.path.exists(self.filepath):
            try:
                data = np.load(self.filepath)
                self.grid = data['grid']
                self.visited = data['visited']
                print("[MEMORY] Map cache loaded successfully.")
                return True
            except Exception as e:
                print(f"[MEMORY] Error loading map cache: {e}")
        return False
