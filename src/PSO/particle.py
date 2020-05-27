import numpy as np
import random as rand


class Particle(object):
    best_pos = None

    def __init__(self, init_position, init_velocity, init_fitness):
        super().__init__()
        self.position = init_position
        self.velocity = init_velocity
        self.fitness = init_fitness
        self.best_pos = init_position

    def update_velocity(self, g_best_position, personal_coeff, global_coeff, inertia_weight):
        r_p = rand.random()
        r_g = rand.random()

        personal_part = r_p * personal_coeff * (self.best_pos - self.position)
        global_part = r_g * global_coeff * (g_best_position - self.position)
        result = inertia_weight * self.velocity + personal_part + global_part

        self.velocity = np.around(result, decimals=4)

    def update_position(self, lower_b, upper_b):
        result = self.position * self.velocity

        for i, res in enumerate(result):
            if res < lower_b[i]:
                self.velocity *= -1.0
                result[i] = lower_b[i]
            elif res > upper_b[i]:
                self.velocity *= -1.0
                result[i] = upper_b[i]

        self.position = np.around(result, decimals=4)

    def __repr__(self):
        return f'Particle (pos: {self.position} vel: {self.velocity} fit: {self.fitness}'