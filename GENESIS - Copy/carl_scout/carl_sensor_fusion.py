"""
carl_sensor_fusion.py — Observation Builder for CARL Genesis.

Biological analogue: Sensory integration pathways in the thalamus and association cortex.
Aggregates heterogeneous sensory inputs (LiDAR, proprioception, internal drive state,
social visual signals) into a unified spatiotemporal representation.
"""

import numpy as np
import math

class ObservationBuilder:
    """Constructs the 34-D observation vector from raw sensor measurements and drives."""
    
    @staticmethod
    def lidar_to_proximity(dists, max_range=2.5):
        """Maps raw distances to [0, 1] proximity index where 1 is closest."""
        return [max(0.0, (max_range - d) / max_range) for d in dists]

    def build(self, lidar_data, vel_fwd, vel_lat, omega, f_cos, f_sin, 
              drives, prev_throttle, prev_steering, face_expr, ego_vec, surprise=0.0) -> np.ndarray:
        """
        Synthesizes the 34-D observation vector.
        
        Vector Layout:
          - [0:8]   Proximity (8 sectors, downsampled from 24-ray LiDAR)
          - [8:11]  Body Velocities (Forward, Lateral, Z-Angular)
          - [11:13] Target direction (Cosine, Sine of angle to target)
          - [13:18] Internal Drives (DA, NE, CORT, Sero, Hunger)
          - [18:22] Discrete Contact Zones (Front, Back, Left, Right)
          - [22:24] Previous Actions (Throttle, Steering)
          - [24:29] Social Face Expression Features
          - [29:33] Proprioceptive Ego State (PSI_L, PSI_R, PSI_S, 0.0)
          - [33]    Spatiotemporal Prediction Surprise
        """
        # 1. LiDAR Proximity (8 rays downsampled from 24)
        proximity_8 = lidar_data[::3]
        proximity = self.lidar_to_proximity(proximity_8)
        
        # 2. Discrete tactile/touch zones
        touch_front = 1.0 if max(proximity[0], proximity[7], proximity[1]) > 0.6 else 0.0
        touch_back  = 1.0 if max(proximity[3], proximity[4], proximity[5]) > 0.6 else 0.0
        touch_left  = 1.0 if proximity[6] > 0.6 else 0.0
        touch_right = 1.0 if proximity[2] > 0.6 else 0.0
        touch_zones = [touch_front, touch_back, touch_left, touch_right]
        
        # 3. Extract drives vector (first 5 drives: DA, NE, CORT, Sero, Hunger)
        drives_vec = list(drives.to_vec())[:5]
        
        # 4. Concatenate to 34-D observation vector
        obs_raw = np.array(
            list(proximity) +
            [vel_fwd, vel_lat, omega] +
            [f_cos, f_sin] +
            drives_vec +
            touch_zones +
            [prev_throttle, prev_steering] +
            list(face_expr) +
            list(ego_vec) +
            [surprise],
            dtype=np.float32
        )
        
        return obs_raw
