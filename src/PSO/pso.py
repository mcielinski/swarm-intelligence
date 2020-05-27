import numpy as np
import random
from operator import attrgetter

from PSO.particle import Particle


class PSO(object):
    def __init__(self, num_particles, num_iterations, test_func, 
                 personal_coeff=1, global_coeff=3, inertia_weight=0.6, 
                 upper_bounds=[5, 5], lower_bounds=[-5, -5], verbose_flag=False):
        super().__init__()
        self.num_particles = num_particles
        self.num_iterations = num_iterations
        self.test_func = test_func
        self.personal_coeff = personal_coeff
        self.global_coeff = global_coeff
        self.inertia_weight = inertia_weight
        self.upper_bounds = np.array(upper_bounds)
        self.lower_bounds = np.array(lower_bounds)
        self.g_best = None
        self.particles = []
        self.verbose = verbose_flag


    def get_gobal_best(self, instance=False):
        best_particle = min(self.particles, key=attrgetter('fitness'))

        if instance: return best_particle

        return best_particle.position

    def fitness(self, position):
        fitness = self.test_func(position)

        return fitness

    def update_fitness(self):
        for p in self.particles:
            p.fitness = self.fitness(p.position)

            if p.fitness < self.fitness(p.best_pos):
                p.best_pos = p.position

            if p.fitness < self.fitness(self.g_best):
                self.g_best = p.position

    def random_vector(self, ub, lb):
        ''' alternative
        r = random.random()
        
        return lb + (ub - lb) * r
        '''
        r_1 = random.random()
        r_2 = random.random()

        x_rand = lb[0] + (ub[0] - lb[0]) * r_1
        y_rand = lb[1] + (ub[1] - lb[1]) * r_2

        return np.array([x_rand, y_rand])

    def random_particle(self):
        r_position = self.random_vector(self.upper_bounds, self.lower_bounds)
        r_velocity = self.random_vector(np.array([1, 1]), np.array([-1, -1]))

        return Particle(r_position, r_velocity, self.fitness(r_position))

    def initialize_particles(self):
        self.particles = [self.random_particle() 
                            for _ in range(self.num_particles)]
        self.g_best = self.get_gobal_best()

    def optimize(self):
        self.initialize_particles()

        for _ in range(self.num_iterations):
            for p in self.particles:
                p.update_velocity(self.g_best, self.personal_coeff, 
                                    self.global_coeff, self.inertia_weight)
                p.update_position(self.upper_bounds, self.lower_bounds)

            self.update_fitness()

            if self.verbose: print(self.get_gobal_best(instance=True).__repr__())

        return self.g_best