import numpy as np
import random as rand
import random

from abc import ABCMeta


class ArtificialBee(object):
    TRIAL_INITIAL_DEFAULT_VALUE = 0
    INITIAL_DEFAULT_PROBABILITY = 0.0

    def __init__(self, test_func, max_trials, lower_b, upper_b):
        self.max_trials = max_trials
        self.test_func = test_func
        self.lower_b = lower_b
        self.upper_b = upper_b
        self.__reset_bee()

    def evaluate_boundaries(self, pos):
        return np.array([
            min([
                max([
                    pos[i],
                    self.lower_b[i]
                ]), 
                self.upper_b[i]
            ])
            for i in range(0, len(self.pos))
        ])

    def random_position(self):
        return np.array([
            self.lower_b[i] + (self.upper_b[i] - self.lower_b[i]) * random.random()
            for i in range(0, len(self.upper_b))
        ])

    def fitness(self):
        return self.fitness_for_pos(self.pos)

    def fitness_for_pos(self, pos):
        return self.test_func(self.pos)

    def explore_pos(self, old_pos):
        if self.trial <= self.max_trials:
            component = np.random.choice(old_pos)
            phi = np.random.uniform(low=-0.5, high=0.5, size=len(old_pos))
            new_pos = old_pos + (old_pos - component) * phi
            new_pos = self.evaluate_boundaries(new_pos)
            self.update_bee(old_pos, new_pos)
            
    def update_bee(self, old_pos, new_pos):
        if self.fitness_for_pos(new_pos) <= self.fitness_for_pos(old_pos):
            self.pos = new_pos
            self.trial = 0
        else:
            self.trial += 1

    def reset_bee(self):
        if self.trial >= self.max_trials:
            self.__reset_bee()

    def __reset_bee(self):
        self.pos = self.random_position()
        self.trial = ArtificialBee.TRIAL_INITIAL_DEFAULT_VALUE
        self.prob = ArtificialBee.INITIAL_DEFAULT_PROBABILITY

##########EmployeeBee###############


class EmployeeBee(ArtificialBee):
    def explore(self):
        self.explore_pos(self.pos)

    def get_fitness(self):
        return 1 / (1 + self.fitness()) if self.fitness() >= 0 else 1 + np.abs(self.fitness())

    def compute_prob(self, max_fitness):
        self.prob = self.get_fitness() / max_fitness


##########OnLookerBee###############


class OnLookerBee(ArtificialBee):

    def onlook(self, best_food_sources, max_trials):
        candidate = np.random.choice(best_food_sources)
        self.explore_pos(candidate.pos)
