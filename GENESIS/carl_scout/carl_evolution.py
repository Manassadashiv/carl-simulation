"""
carl_evolution.py — Generational Lifecycle and Evolutionary Manager for CARL.

Biological analogue: Natural selection, genetic inheritance, and mutation.
Maintains an elite pool of genomes based on lifetime fitness scores, applies
clonal reproduction with mutation to drive the emergence of robust navigational
and homeostatic parameters across generations.
"""

import os
import numpy as np
import mujoco

class EvolutionManager:
    """Manages the lifetime assessment, elite pool selection, breeding, and reset lifecycle of CARL."""
    
    def __init__(self, elite_capacity=5, fossil_log_path="evolutionary_fossil_record.csv"):
        self.elite_capacity = elite_capacity
        self.fossil_log_path = fossil_log_path
        self.elite_pool = []  # List of tuples: (score, genome_dict)

    def check_death(self, drives) -> bool:
        """Returns True if the agent is metabolically dead (energy depleted or fatal damage)."""
        return (drives.energy <= 0.0) or (drives.damage >= 1.0)

    def calculate_habit_index(self, action_history) -> float:
        """Computes Shannon entropy of the action history to measure behavioral diversity/habits."""
        if len(action_history) >= 10:
            bins_x = np.linspace(-1.0, 1.0, 6)
            bins_y = np.linspace(-1.0, 1.0, 6)
            actions = np.array(action_history)
            hist, _, _ = np.histogram2d(actions[:, 0], actions[:, 1], bins=(bins_x, bins_y))
            probs = hist.flatten() / len(action_history)
            probs = probs[probs > 0.0]
            return float(-np.sum(probs * np.log2(probs)))
        return 0.0

    def log_fossil_record(self, generation_idx, lifespan, food_harvested, damage, score, 
                           goal_count, habit_index, exploration_diversity,
                           procedural_inheritance, ecological_persistence, culture_imitation, goal_crystallization):
        """Appends the lifetime performance and configurations of the deceased generation to the fossil record."""
        if not os.path.exists(self.fossil_log_path):
            with open(self.fossil_log_path, 'w') as f_fos:
                f_fos.write("generation,lifespan,food_harvested,damage_accumulated,elite_score,goal_count,habit_index,exploration_diversity,procedural_inheritance_on,ecological_persistence_on,culture_on,goal_crystallization_on\n")
        with open(self.fossil_log_path, 'a') as f_fos:
            f_fos.write(f"{generation_idx},{lifespan},{food_harvested},{damage:.4f},{score:.4f},{goal_count},{habit_index:.4f},{exploration_diversity:.2f},{procedural_inheritance},{ecological_persistence},{culture_imitation},{goal_crystallization}\n")

    def update_elite_pool(self, score, brain, goal_crystallizer=None):
        """Extracts the current genome from the brain and attempts to add it to the elite pool."""
        current_genome = {
            'CLUTCH_THRESHOLD': brain.drives.CLUTCH_THRESHOLD,
            'CLUTCH_DA_SUPPRESSION': brain.drives.CLUTCH_DA_SUPPRESSION,
            'boredom_accumulation_rate': brain.drives.boredom_accumulation_rate,
            'boredom_decay_rate': brain.drives.boredom_decay_rate,
            'sleep_replay_fraction': brain.drives.sleep_replay_fraction,
            'sleep_pruning_decay': brain.drives.sleep_pruning_decay,
            'curiosity_surprise_gain': brain.drives.curiosity_surprise_gain,
            'curiosity_hdc_gain': brain.drives.curiosity_hdc_gain,
            'cpg_frequency_multiplier': brain.drives.cpg_frequency_multiplier,
            'cpg_amplitude_multiplier': brain.drives.cpg_amplitude_multiplier,
            'brainstem_reflex_stiffness': brain.drives.brainstem_reflex_stiffness,
            # Subconsciousness genetic parameters
            'somatic_activation_threshold': brain.somatic.activation_threshold,
            'somatic_marker_strength': brain.somatic.marker_strength,
            'workspace_ignition_threshold': brain.workspace.ignition_threshold,
            'workspace_hysteresis': brain.workspace.hysteresis,
            'circadian_period': brain.circadian.circadian_period,
            'ultradian_period': brain.circadian.ultradian_period,
            'circadian_amplitude': brain.circadian.amplitude,
            'habit_preference': brain.habits.habit_preference,
            'dmn_activation_threshold': brain.dmn.activation_threshold,
            'allostatic_prediction_horizon': brain.allostatic.prediction_horizon,
            'allostatic_anticipation_gain': brain.allostatic.anticipation_gain,
            'goal_persistence_threshold': goal_crystallizer.persistence_threshold if goal_crystallizer else 30,
        }
        
        if len(self.elite_pool) < self.elite_capacity:
            self.elite_pool.append((score, current_genome))
        else:
            min_idx = int(np.argmin([x[0] for x in self.elite_pool]))
            if score > self.elite_pool[min_idx][0]:
                self.elite_pool[min_idx] = (score, current_genome)

    def breed_and_reset(self, brain, model, data, world_map, goal_crystallizer,
                        goal_lifetimes, expired_lifetimes, action_history, event_queue,
                        mujoco_lock, default_forcerange, default_qpos, default_qvel, default_food_pos,
                        food_geom_id, carl_joint_id, agent_joints, respawn_target_fn,
                        procedural_inheritance=True, ecological_persistence=True, goal_crystallization_on=True):
        """Breeds mutated parameters from elite pool and resets CARL and the environment for the next generation."""
        # 1. Inherit from a random elite parent
        if self.elite_pool:
            parent_score, parent_genome = self.elite_pool[np.random.randint(len(self.elite_pool))]
            brain.drives.CLUTCH_THRESHOLD = parent_genome['CLUTCH_THRESHOLD']
            brain.drives.CLUTCH_DA_SUPPRESSION = parent_genome['CLUTCH_DA_SUPPRESSION']
            brain.drives.boredom_accumulation_rate = parent_genome['boredom_accumulation_rate']
            brain.drives.boredom_decay_rate = parent_genome['boredom_decay_rate']
            brain.drives.sleep_replay_fraction = parent_genome['sleep_replay_fraction']
            brain.drives.sleep_pruning_decay = parent_genome['sleep_pruning_decay']
            brain.drives.curiosity_surprise_gain = parent_genome['curiosity_surprise_gain']
            brain.drives.curiosity_hdc_gain = parent_genome['curiosity_hdc_gain']
            brain.drives.cpg_frequency_multiplier = parent_genome['cpg_frequency_multiplier']
            brain.drives.cpg_amplitude_multiplier = parent_genome['cpg_amplitude_multiplier']
            brain.drives.brainstem_reflex_stiffness = parent_genome['brainstem_reflex_stiffness']
            
            # Inherit subconsciousness genome
            if 'somatic_activation_threshold' in parent_genome:
                brain.somatic.activation_threshold = parent_genome['somatic_activation_threshold']
                brain.somatic.marker_strength = parent_genome['somatic_marker_strength']
                brain.workspace.ignition_threshold = parent_genome['workspace_ignition_threshold']
                brain.workspace.hysteresis = parent_genome['workspace_hysteresis']
                brain.circadian.circadian_period = parent_genome['circadian_period']
                brain.circadian.ultradian_period = parent_genome['circadian_period'] # Wait, in harvest it was circadian_period or ultradian_period? Wait, let's verify.
                # In carl_harvest.py:
                # brain.circadian.ultradian_period = parent_genome['ultradian_period']
                brain.circadian.ultradian_period = parent_genome['ultradian_period']
                brain.circadian.amplitude = parent_genome['circadian_amplitude']
                brain.habits.habit_preference = parent_genome['habit_preference']
                brain.dmn.activation_threshold = parent_genome['dmn_activation_threshold']
                brain.allostatic.prediction_horizon = parent_genome['allostatic_prediction_horizon']
                brain.allostatic.anticipation_gain = parent_genome['allostatic_anticipation_gain']
                if 'goal_persistence_threshold' in parent_genome:
                    goal_crystallizer.persistence_threshold = parent_genome['goal_persistence_threshold']

        # 2. Mutate parameters
        brain.drives.mutate(rate=0.15)
        brain.somatic.mutate(rate=0.15)
        brain.workspace.mutate(rate=0.15)
        brain.circadian.mutate(rate=0.15)
        brain.allostatic.mutate(rate=0.15)
        brain.habits.mutate(rate=0.15)
        brain.dmn.mutate(rate=0.15)
        
        # Mutate goal persistence threshold (evolvable parameter)
        if np.random.rand() < 0.15:
            mutation = int(np.random.normal(0, 5))
            goal_crystallizer.persistence_threshold = int(np.clip(
                goal_crystallizer.persistence_threshold + mutation, 10, 100
            ))
        
        # 3. Reset internal drives
        brain.drives.energy = 1.0
        brain.drives.damage = 0.0
        brain.drives.fatigue = 0.0
        brain.drives.hunger = 0.0
        brain.drives.da = 0.0
        brain.drives.ne = 0.0
        brain.drives.cort = 0.0
        brain.drives.ach = 0.0
        brain.drives.sero = 1.0
        
        with mujoco_lock:
            model.actuator_forcerange[:] = default_forcerange.copy()
            
        # 4. Wipe HDC registers
        brain.hdc.M_spatial.fill(0.0)
        brain.hdc.M_affective.fill(0.0)
        brain.hdc.M_episodic.fill(0.0)
        brain.hdc.episodic_trace.fill(0.0)
        brain.hdc._familiarity_ema = 0.0
        brain.hdc._salience_buffer.clear()
        if not procedural_inheritance:
            brain.hdc.M_procedural.fill(0.0)
            
        # 5. Reset subconsciousness systems for new lifetime
        brain.somatic.markers.clear()
        brain.somatic.approach_bias = 0.0
        brain.somatic.arousal_injection = 0.0
        brain.workspace.broadcast_history.clear()
        brain.workspace.current_broadcast = None
        brain.workspace._suppression.clear()
        brain.dmn.replay_buffer.clear()
        brain.dmn.imagined_goals.clear()
        brain.dmn.activation_level = 0.0
        brain.habits.chunks.clear()
        brain.habits.active_chunk = None
        brain.habits.current_recording.clear()
        brain.circadian.circadian_phase = 0.0
        brain.circadian.ultradian_phase = 0.0
        brain.allostatic.energy_history.clear()
        brain.allostatic.drain_history.clear()
        brain.allostatic.urgency = 0.0
        
        # 6. Reset map
        world_map.grid.fill(0.0)
        world_map.visited.fill(0.0)
        world_map.save()
        
        # 7. Reset crystallization
        if goal_crystallization_on:
            goal_crystallizer.goals.clear()
            goal_crystallizer.active_goal_idx = None
            goal_crystallizer.candidate_pos = None
            goal_crystallizer.persistence_counter = 0
            
        # 8. Reset metrics collections
        goal_lifetimes.clear()
        expired_lifetimes.clear()
        action_history.clear()
        
        # 9. Flush transition event queues
        while not event_queue.empty():
            try:
                event_queue.get_nowait()
            except Exception:
                break
                
        # 10. Reset MuJoCo physics state
        with mujoco_lock:
            if not ecological_persistence:
                data.qpos[:] = default_qpos.copy()
                data.qvel[:] = default_qvel.copy()
                model.geom_pos[food_geom_id] = default_food_pos.copy()
            else:
                for j in agent_joints:
                    qp_adr = model.jnt_qposadr[j]
                    qv_adr = model.jnt_dofadr[j]
                    j_type = model.jnt_type[j]
                    if j_type == mujoco.mjtJoint.mjJNT_FREE:
                        data.qpos[qp_adr:qp_adr+7] = default_qpos[qp_adr:qp_adr+7].copy()
                        data.qvel[qv_adr:qv_adr+6] = default_qvel[qv_adr:qv_adr+6].copy()
                    elif j_type == mujoco.mjtJoint.mjJNT_BALL:
                        data.qpos[qp_adr:qp_adr+4] = default_qpos[qp_adr:qp_adr+4].copy()
                        data.qvel[qv_adr:qv_adr+3] = default_qvel[qv_adr:qv_adr+3].copy()
                    else:
                        data.qpos[qp_adr] = default_qpos[qp_adr]
                        data.qvel[qv_adr] = default_qvel[qv_adr]
            
            qp_adr = model.jnt_qposadr[carl_joint_id]
            data.qpos[qp_adr:qp_adr+3] = [0.0, 0.0, 0.15]
            data.qpos[qp_adr+3:qp_adr+7] = [1.0, 0.0, 0.0, 0.0]
            
            qv_adr = model.jnt_dofadr[carl_joint_id]
            data.qvel[qv_adr:qv_adr+6] = 0.0
            
            respawn_target_fn()
            mujoco.mj_forward(model, data)
