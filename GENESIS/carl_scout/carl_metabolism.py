import numpy as np

class CarlMetabolism:
    def __init__(self, initial_energy=100.0, consumption_rate=0.0005, recovery_bonus=30.0):
        self.energy = initial_energy
        self.consumption_rate = consumption_rate
        self.recovery_bonus = recovery_bonus
        self.economy_mode = False

    def update(self, torques):
        """ Consumes energy proportional to physical work done by the motors """
        # torques is an array/list of [torque_L, torque_R]
        u_L, u_R = torques
        
        # Metabolic cost of motion: absolute sum of effort
        effort = abs(u_L) + abs(u_R)
        delta_e = effort * self.consumption_rate
        
        self.energy = max(0.0, self.energy - delta_e)
        
        # Trigger Economy Mode thresholds
        if self.energy < 20.0 and not self.economy_mode:
            self.economy_mode = True
            print("[METABOLISM] Battery low! Engaging Economy Mode. High-speed systems locked.")
        elif self.energy >= 20.0 and self.economy_mode:
            self.economy_mode = False
            print("[METABOLISM] Battery restored. Nominal power systems active.")
            
        return self.energy

    def consume_food(self):
        """ Restores energy when a food pellet is successfully harvested """
        self.energy = min(100.0, self.energy + self.recovery_bonus)
        print(f"[METABOLISM] Pellet harvested! Energy restored to {self.energy:.1f}%")
