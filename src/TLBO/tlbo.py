import numpy as np
import random
from operator import attrgetter

from TLBO.learner import Learner


class TLBO(object):
    def __init__(self, num_learners, num_iterations, test_func, lower_bounds, upper_bounds, verbose_flag=False):
        super().__init__()
        self.num_learners = num_learners
        self.num_iterations = num_iterations
        self.test_func = test_func
        self.lower_bounds = np.array(lower_bounds)
        self.upper_bounds = np.array(upper_bounds)
        self.learners = []
        self.verbose = verbose_flag


    def fitness(self, solution):
        fitness = self.test_func(solution)

        return fitness

    def random_vector(self, lower_b, upper_b):
        ''' alternative - better results ofc
        r = random.random()
        
        return lower_b + (upper_b - lower_b) * r
        '''
        r_1 = random.random()
        r_2 = random.random()

        x_rand = lower_b[0] + (upper_b[0] - lower_b[0]) * r_1
        y_rand = lower_b[1] + (upper_b[1] - lower_b[1]) * r_2

        return np.array([x_rand, y_rand])

    def create_learner(self):
        solution = self.random_vector(self.lower_bounds, self.upper_bounds)
        fitness = self.fitness(solution)

        return Learner(solution, fitness)

    def get_teacher(self):
        best_learner = min(self.learners, key=attrgetter('fitness'))

        return best_learner
    
    def select_best(self, subjects, c_subjects):
        s_fitness = self.fitness(subjects)
        c_fitness = self.fitness(c_subjects)

        if s_fitness > c_fitness:
            best_subjects = c_subjects
            best_fitness = c_fitness
        else:
            best_subjects = subjects
            best_fitness = s_fitness

        return (best_subjects, best_fitness)
    
    def random_learner(self, excluded_id):
        available_ids = list(range(self.num_learners))
        available_ids.remove(excluded_id)
        chosen_id = random.choice(available_ids)

        return chosen_id

    def learner_phase(self, learner, learner_id):
        k_id = self.random_learner(excluded_id=learner_id)
        k_learner = self.learners[k_id]
        k_subjets = k_learner.subjects
        subjects_prim = np.zeros(len(learner.subjects))

        for i, subject in enumerate(learner.subjects):
            if learner.fitness < k_learner.fitness:
                diff = subject - k_subjets[i]
            else:
                diff = k_subjets[i] - subject

            r = random.random()
            subjects_prim[i] = subject + (r * diff)

        #rounded_subjects_prim = np.around(subjects_prim, decimals=4)
        best_subjects, best_fitness = self.select_best(learner.subjects, subjects_prim)

        return (best_subjects, best_fitness)

    def teacher_phase(self, learner, learner_id):
        teacher = self.get_teacher()
        tf = random.randint(1, 2)
        subjects_prim = np.zeros(len(teacher.subjects))

        for i, subject in enumerate(learner.subjects):
            s_mean = np.mean([s.subjects[i] for s in self.learners])
            r = random.random()
            diff_mean = teacher.subjects[i] - (tf * s_mean)
            subjects_prim[i] = subject + (r * diff_mean)

        #rounded_subjects_prim = np.around(subjects_prim, decimals=4)
        best_subjects, best_fitness = self.select_best(learner.subjects, subjects_prim)

        return (best_subjects, best_fitness)

    def initialize_learners(self):
        self.learners = [self.create_learner() for _ in range(self.num_learners)]

    def optimize(self):
        self.initialize_learners()

        for _ in range(self.num_iterations):
            for i, learner in enumerate(self.learners):
                learner.subjects, learner.fitness = self.teacher_phase(learner, i)
                learner.subjects, learner.fitness = self.learner_phase(learner, i)

            teacher = self.get_teacher()
            
            if self.verbose: print(teacher.__repr__())

        return teacher.subjects