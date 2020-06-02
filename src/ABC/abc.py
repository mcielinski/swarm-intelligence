import numpy as np
import random
from operator import attrgetter
from copy import deepcopy
import pandas as pd 
import matplotlib.pyplot as plt

from ABC.bees import EmployeeBee, OnLookerBee


class ABC(object):
    def __init__(self, colony_size, num_iterations, test_func, max_trials = 10, lower_bounds=[-5, -5], upper_bounds=[5, 5], verbose_flag=False):
        super().__init__()
        self.colony_size = colony_size
        self.num_iterations = num_iterations
        self.lower_bounds = np.array(lower_bounds)
        self.upper_bounds = np.array(upper_bounds)
        self.max_trials = max_trials 
        self.test_func = test_func
        self.verbose = verbose_flag

        self.optimal_solution = None
        self.optimality_tracking = []

    def optimize(self):
        self.__reset_algorithm()
        self.__initialize_employees()
        self.__initialize_onlookers()
        for itr in range(self.num_iterations):
            self.__employee_bees_phase()
            self.__update_optimal_solution()

            self.__calculate_probabilities()
            self.__select_best_food_sources()

            self.__onlooker_bees_phase()

            self.__scout_bees_phase()

            self.__update_optimal_solution()
            self.__update_optimality_tracking()

            if self.verbose: print("iter: {} = cost: {}, x={}, y={}".format(itr, "%04.03e" % self.optimal_solution.fitness(), self.optimal_solution.pos[0], self.optimal_solution.pos[1]))
            # if self.verbose: print([bee.pos for bee in self.employee_bees + self.onlokeer_bees])

            # df_employee = pd.DataFrame([bee.pos for bee in self.employee_bees], columns=['a', 'b'])
            # df_employee.plot(x='a', y='b', style='ob')
            # df_onlook = pd.DataFrame([bee.pos for bee in self.onlokeer_bees], columns=['a', 'b'])
            # df_onlook.plot(x='a', y='b', style='or')
            # plt.show()

        return self.optimal_solution.pos

    def __reset_algorithm(self):
        self.optimal_solution = None
        self.optimality_tracking = []

    def __initialize_employees(self):
        self.employee_bees = []
        for _ in range(self.colony_size // 2):
            self.employee_bees.append(EmployeeBee(self.test_func, self.max_trials, self.lower_bounds, self.upper_bounds))

    def __initialize_onlookers(self):
        self.onlokeer_bees = []
        for _ in range(self.colony_size // 2):
            self.onlokeer_bees.append(OnLookerBee(self.test_func, self.max_trials, self.lower_bounds, self.upper_bounds))

    def __update_optimal_solution(self):
        n_optimal_solution = min(self.onlokeer_bees + self.employee_bees, key=lambda bee: bee.fitness())
        if not self.optimal_solution:
            self.optimal_solution = deepcopy(n_optimal_solution)
        else:
            if n_optimal_solution.fitness() < self.optimal_solution.fitness():
                self.optimal_solution = deepcopy(n_optimal_solution)

    def __update_optimality_tracking(self):
        self.optimality_tracking.append(self.optimal_solution.fitness())
        
    def __select_best_food_sources(self):
        self.best_food_sources = list(filter(lambda bee: bee.prob > np.random.uniform(low=0, high=1), self.employee_bees))
        
        while not self.best_food_sources:
            self.best_food_sources = list(filter(lambda bee: bee.prob > np.random.uniform(low=0, high=1), self.employee_bees))

    def __calculate_probabilities(self):
        sum_fitness = sum(map(lambda bee: bee.get_fitness(), self.employee_bees))

        for bee in self.employee_bees:
            bee.compute_prob(sum_fitness)

    def __employee_bees_phase(self):
        for bee in self.employee_bees:
            bee.explore()

    def __onlooker_bees_phase(self):
        for bee in self.onlokeer_bees:
            bee.onlook(self.best_food_sources)

    def __scout_bees_phase(self):
        for bee in self.employee_bees + self.onlokeer_bees:
            bee.reset_bee()
