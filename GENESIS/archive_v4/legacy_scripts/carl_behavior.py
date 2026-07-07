"""
carl_behavior.py — Behavioral Arbitration Engine for CARL Genesis.

Biological analogue: The basal ganglia, specifically the striatum, which selects
actions based on a priority queue of competing cortical and subcortical signals,
heavily modulated by dopaminergic reinforcement and allostatic drives.
"""

import numpy as np
import math

class ActionContext:
    """Dataclass or container for all inputs needed by the BehavioralArbitrator."""
    def __init__(self, pose, velocity_fwd, drives, face_expr, ego_vec,
                 planner_step, f_dist, path, frontier_path, follower,
                 crucible, crucible_state, carl_pos_cr, carl_yaw_cr,
                 favorite_object_id, obj_names, obj_body_ids,
                 nearest_pos, nearest_dist, nearest_bid,
                 fav_pos=None, fav_dist=0.0,
                 follow_primary_path=False, habit_fired=False):
        self.pose = pose
        self.velocity_fwd = velocity_fwd
        self.drives = drives
        self.face_expr = face_expr
        self.ego_vec = ego_vec
        self.planner_step = planner_step
        self.f_dist = f_dist
        self.path = path
        self.frontier_path = frontier_path
        self.follower = follower
        self.crucible = crucible
        self.crucible_state = crucible_state
        self.carl_pos_cr = carl_pos_cr
        self.carl_yaw_cr = carl_yaw_cr
        self.favorite_object_id = favorite_object_id
        self.obj_names = obj_names
        self.obj_body_ids = obj_body_ids
        self.nearest_pos = nearest_pos
        self.nearest_dist = nearest_dist
        self.nearest_bid = nearest_bid
        self.fav_pos = fav_pos
        self.fav_dist = fav_dist
        self.follow_primary_path = follow_primary_path
        self.habit_fired = habit_fired


class BehavioralArbitrator:
    """
    Priority-cascaded action selection engine.
    
    Replaces the monolithic if/elif overrides chain in carl_harvest.py
    with structured, priority-ordered behavior evaluations.
    """
    
    def select_action(self, base_action: np.ndarray, ctx: ActionContext):
        """
        Arbitrate between base action and active overrides.
        
        Returns
        -------
        exec_action : np.ndarray
            The selected [linear_v, angular_w] action.
        fired_behavior : str
            Name of the behavior that took control (e.g. 'flee_predator').
        """
        exec_action = base_action.copy()
        fired_behavior = "base"
        override_active = False

        # --- PRIMARY PATH FOLLOW ---
        if ctx.follow_primary_path:
            v_goal, w_goal = ctx.follower.get_control(ctx.pose, ctx.path)
            exec_action[0] = np.clip(v_goal, -0.30, 0.60)
            exec_action[1] = np.clip(w_goal, -1.5, 1.5)
            override_active = True
        
        # 2. Crucible overrides (highest priority survival behaviors)
        if ctx.crucible_state is not None:
            # Priority 1: MPC DODGE — incoming projectile detected by DVS
            if ctx.crucible_state.get('mpc_dodge') is not None:
                mpc = ctx.crucible_state['mpc_dodge']
                exec_action[0] = mpc['v_dodge']
                exec_action[1] = mpc['w_dodge']
                override_active = True
                fired_behavior = "mpc_dodge"
                if ctx.planner_step % 10 == 0:
                    print(f"[MPC DODGE] Evading! Impact at ({mpc['impact_point'][0]:.1f}, {mpc['impact_point'][1]:.1f})")

            # Priority 2: FLEE PREDATOR
            elif ctx.crucible_state.get('predator', {}).get('state') in ('HUNT', 'LUNGE'):
                flee = ctx.crucible.get_flee_direction(ctx.carl_pos_cr, ctx.carl_yaw_cr)
                if flee is not None:
                    exec_action[0] = flee[0]
                    exec_action[1] = flee[1]
                    override_active = True
                    fired_behavior = "flee_predator"
                    if ctx.planner_step % 25 == 0:
                        print(f"[CRUCIBLE] FLEEING PREDATOR! dist={ctx.crucible_state['predator']['dist_to_carl']:.1f}m")

            # Priority 3: VISUAL SERVOING (social fixation on detected face)
            elif ctx.crucible_state.get('servo_w') is not None:
                exec_action[0] = 0.15  # Slow approach
                exec_action[1] = ctx.crucible_state['servo_w']
                override_active = True
                fired_behavior = "visual_servo"
                if ctx.planner_step % 50 == 0:
                    print(f"[VISUAL SERVO] Tracking face — w_servo={ctx.crucible_state['servo_w']:.2f}")

            # Priority 4: Navigate to remembered (occluded) object
            elif not override_active and ctx.favorite_object_id is not None:
                fav_name = None
                for _on, _bid in zip(ctx.obj_names, ctx.obj_body_ids):
                    if _bid == ctx.favorite_object_id:
                        fav_name = _on
                        break
                if fav_name:
                    mem_pos, confidence = ctx.crucible.object_memory.recall(fav_name)
                    if mem_pos is not None and confidence > 0.3:
                        mem_dist = math.hypot(mem_pos[0] - ctx.pose[0], mem_pos[1] - ctx.pose[1])
                        if 0.5 < mem_dist < 4.0:
                            angle_to_mem = math.atan2(mem_pos[1] - ctx.pose[1], mem_pos[0] - ctx.pose[0])
                            rel_angle = angle_to_mem - ctx.pose[2]
                            rel_angle = (rel_angle + math.pi) % (2 * math.pi) - math.pi
                            exec_action[0] = 0.2
                            exec_action[1] = np.clip(rel_angle * 2.0, -1.0, 1.0)
                            override_active = True
                            fired_behavior = "object_permanence"
                            if ctx.planner_step % 100 == 0:
                                print(f"[OBJECT PERMANENCE] Navigating to REMEMBERED {fav_name} (conf={confidence:.1%})")

        # 3. Emotional / Frustration Overrides
        # Naughtiness / Tantrum (frustrated/bored)
        if not override_active and ctx.drives.cort > 0.5 and ctx.drives.da < 0.2 and ctx.drives.energy > 0.4:
            exec_action[0] = 0.0
            exec_action[1] = 1.0  # Fast spin
            override_active = True
            fired_behavior = "tantrum"
            if ctx.planner_step % 50 == 0:
                print("[BEHAVIOR] TANTRUM!")

        # Joy / Playfulness
        elif ctx.drives.da > 0.7:
            exec_action[0] = 0.1
            exec_action[1] = 0.8 * math.sin(ctx.planner_step * 0.2)  # Happy wiggle
            override_active = True
            fired_behavior = "joy_wiggle"
            if ctx.planner_step % 50 == 0:
                print("[BEHAVIOR] JOY WIGGLE!")

        # Social Approach / Hide
        elif ctx.face_expr[0] > 0.5:
            smile_ratio = ctx.face_expr[1]
            eyebrow = ctx.face_expr[3]
            if smile_ratio > 0.5:
                exec_action[0] = 0.2
                exec_action[1] = 0.0  # Approach smile
                override_active = True
                fired_behavior = "social_approach"
                if ctx.planner_step % 50 == 0:
                    print("[BEHAVIOR] SOCIAL APPROACH!")
            elif eyebrow > 0.6 and smile_ratio < 0.2:
                exec_action[0] = -0.3
                exec_action[1] = 0.5  # Hide from angry face
                override_active = True
                fired_behavior = "social_hide"
                if ctx.planner_step % 50 == 0:
                    print("[BEHAVIOR] SOCIAL HIDE!")

        # Play / Object Chasing
        elif ctx.drives.M_t > 0.1 and ctx.nearest_pos is not None and ctx.nearest_dist < 2.0:
            angle_to_obj = math.atan2(ctx.nearest_pos[1] - ctx.pose[1], ctx.nearest_pos[0] - ctx.pose[0])
            rel_angle = angle_to_obj - ctx.pose[2]
            rel_angle = (rel_angle + math.pi) % (2 * math.pi) - math.pi
            exec_action[0] = 0.3
            exec_action[1] = np.clip(rel_angle * 2.0, -1.0, 1.0)
            override_active = True
            fired_behavior = "play_chase"
            if ctx.planner_step % 50 == 0:
                print("[BEHAVIOR] PLAYING WITH TOY!")

        # Attachment (Seek favorite object)
        elif ctx.favorite_object_id is not None and ctx.drives.energy > 0.5 and ctx.drives.M_t > 0.0 and ctx.fav_pos is not None:
            fav_dist = ctx.fav_dist
            if fav_dist > 0.5 and fav_dist < 3.0:
                angle_to_fav = math.atan2(ctx.fav_pos[1] - ctx.pose[1], ctx.fav_pos[0] - ctx.pose[0])
                rel_angle = angle_to_fav - ctx.pose[2]
                rel_angle = (rel_angle + math.pi) % (2 * math.pi) - math.pi
                exec_action[0] = 0.2
                exec_action[1] = np.clip(rel_angle * 2.0, -1.0, 1.0)
                override_active = True
                fired_behavior = "seek_favorite_toy"
                if ctx.planner_step % 50 == 0:
                    print("[BEHAVIOR] SEEKING FAVORITE TOY!")

        # Allostatic low energy throttle cap
        if ctx.drives.energy < 0.20 and not override_active:
            exec_action[0] = np.clip(exec_action[0], -0.25, 0.25)
            
        return exec_action, fired_behavior
