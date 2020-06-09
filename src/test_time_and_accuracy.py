import random
import test_functions
from PSO.pso import PSO
from TLBO.tlbo import TLBO
from ABC.abc import ABC
from helpers import plot
import pandas as pd 
import time
from tqdm import tqdm

NUM_TEST = 100
PARTICLES = 10
ITERATIONS = 50
MAX_TRAILS = 10

x = 0
y = 0

TEST_NAME = "{}_test_{}_particles_{}_iterations".format(NUM_TEST, PARTICLES, ITERATIONS)

# PSO params
NUM_PARTICLES = PARTICLES
NUM_ITERATIONS_PSO = ITERATIONS

# TLBO params
NUM_LEARNERS = PARTICLES
NUM_ITERATIONS_TLBO = ITERATIONS

#ABC
COLONY_SIZE = PARTICLES
NUM_ITERATIONS_ABC = ITERATIONS
MAX_TRAILS = 10

# other params
BOUND = 10
VERBOSE = False

def round_assert(min_x, min_y):
    return 1 if round(min_x, 4) == 0 and round(min_y, 4) == 0 else 0  

test_data = []
for i in tqdm(range(NUM_TEST)):
    row = []

    start = time.time()
    pso_ackley = PSO(num_particles=NUM_PARTICLES, num_iterations=NUM_ITERATIONS_PSO, test_func=test_functions.ackley, 
                        lower_bounds=[-1 * BOUND, -1 * BOUND], upper_bounds=[BOUND, BOUND], verbose_flag=VERBOSE)
    end = time.time()
    min_x, min_y = pso_ackley.optimize()
    row.append(round_assert(min_x, min_y))
    row.append(end-start)

    start = time.time()
    pso_griewank = PSO(num_particles=NUM_PARTICLES, num_iterations=NUM_ITERATIONS_PSO, test_func=test_functions.griewank, 
                        lower_bounds=[-1 * BOUND, -1 * BOUND], upper_bounds=[BOUND, BOUND], verbose_flag=VERBOSE)
    end = time.time()
    min_x, min_y = pso_griewank.optimize()
    row.append(round_assert(min_x, min_y))
    row.append(end-start)

    start = time.time()
    pso_rastrigin = PSO(num_particles=NUM_PARTICLES, num_iterations=NUM_ITERATIONS_PSO, test_func=test_functions.rastrigin, 
                        lower_bounds=[-1 * BOUND, -1 * BOUND], upper_bounds=[BOUND, BOUND], verbose_flag=VERBOSE)
    end = time.time()
    min_x, min_y = pso_rastrigin.optimize()
    row.append(round_assert(min_x, min_y))
    row.append(end-start)

    start = time.time()
    pso_sphere = PSO(num_particles=NUM_PARTICLES, num_iterations=NUM_ITERATIONS_PSO, test_func=test_functions.sphere, 
                        lower_bounds=[-1 * BOUND, -1 * BOUND], upper_bounds=[BOUND, BOUND], verbose_flag=VERBOSE)
    end = time.time()
    min_x, min_y = pso_sphere.optimize()
    row.append(round_assert(min_x, min_y))
    row.append(end-start)

    start = time.time()
    abc_ackley = ABC(colony_size=COLONY_SIZE, max_trials=MAX_TRAILS, num_iterations=NUM_ITERATIONS_ABC, test_func=test_functions.ackley, 
                        lower_bounds=[-1 * BOUND, -1 * BOUND], upper_bounds=[BOUND, BOUND], verbose_flag=VERBOSE)
    end = time.time()
    min_x, min_y = abc_ackley.optimize()
    row.append(round_assert(min_x, min_y))
    row.append(end-start)

    start = time.time()
    abc_griewank = ABC(colony_size=COLONY_SIZE, max_trials=MAX_TRAILS, num_iterations=NUM_ITERATIONS_ABC, test_func=test_functions.griewank, 
                        lower_bounds=[-1 * BOUND, -1 * BOUND], upper_bounds=[BOUND, BOUND], verbose_flag=VERBOSE)
    end = time.time()
    min_x, min_y = abc_griewank.optimize()
    row.append(round_assert(min_x, min_y))
    row.append(end-start)

    start = time.time()
    abc_rastrigin = ABC(colony_size=COLONY_SIZE, max_trials=MAX_TRAILS, num_iterations=NUM_ITERATIONS_ABC, test_func=test_functions.rastrigin, 
                        lower_bounds=[-1 * BOUND, -1 * BOUND], upper_bounds=[BOUND, BOUND], verbose_flag=VERBOSE)
    end = time.time()
    min_x, min_y = abc_rastrigin.optimize()
    row.append(round_assert(min_x, min_y))
    row.append(end-start)

    start = time.time()
    abc_sphere = ABC(colony_size=COLONY_SIZE, max_trials=MAX_TRAILS, num_iterations=NUM_ITERATIONS_ABC, test_func=test_functions.sphere, 
                        lower_bounds=[-1 * BOUND, -1 * BOUND], upper_bounds=[BOUND, BOUND], verbose_flag=VERBOSE)
    end = time.time()
    min_x, min_y = abc_sphere.optimize()
    row.append(round_assert(min_x, min_y))
    row.append(end-start)

    start = time.time()
    tlbo_ackley = TLBO(num_learners=NUM_LEARNERS, num_iterations=NUM_ITERATIONS_TLBO, test_func=test_functions.ackley, 
                        lower_bounds=[-1 * BOUND, -1 * BOUND], upper_bounds=[BOUND, BOUND], verbose_flag=VERBOSE)
    end = time.time()
    min_x, min_y = tlbo_ackley.optimize()
    row.append(round_assert(min_x, min_y))
    row.append(end-start)

    start = time.time()
    tlbo_griewank = TLBO(num_learners=NUM_LEARNERS, num_iterations=NUM_ITERATIONS_TLBO, test_func=test_functions.griewank, 
                        lower_bounds=[-1 * BOUND, -1 * BOUND], upper_bounds=[BOUND, BOUND], verbose_flag=VERBOSE)
    end = time.time()
    min_x, min_y = tlbo_griewank.optimize()
    row.append(round_assert(min_x, min_y))
    row.append(end-start)

    start = time.time()
    tlbo_rastrigin = TLBO(num_learners=NUM_LEARNERS, num_iterations=NUM_ITERATIONS_TLBO, test_func=test_functions.rastrigin, 
                        lower_bounds=[-1 * BOUND, -1 * BOUND], upper_bounds=[BOUND, BOUND], verbose_flag=VERBOSE)
    end = time.time()
    min_x, min_y = tlbo_rastrigin.optimize()
    row.append(round_assert(min_x, min_y))
    row.append(end-start)

    start = time.time()
    tlbo_sphere = TLBO(num_learners=NUM_LEARNERS, num_iterations=NUM_ITERATIONS_TLBO, test_func=test_functions.sphere, 
                        lower_bounds=[-1 * BOUND, -1 * BOUND], upper_bounds=[BOUND, BOUND], verbose_flag=VERBOSE)
    end = time.time()
    min_x, min_y = tlbo_sphere.optimize()   
    row.append(round_assert(min_x, min_y))
    row.append(end-start)

    test_data.append(row)

result_data_df = pd.DataFrame(test_data, columns=[
    'pso_ackley_outcome', 'pso_ackley_time',
    'pso_griewank_outcome', 'pso_griewank_time',
    'pso_rastrigin_outcome', 'pso_rastrigin_time',
    'pso_sphere_outcome', 'pso_sphere_time',

    'abc_ackley_outcome', 'abc_ackley_time',
    'abc_griewank_outcome', 'abc_griewank_time',
    'abc_rastrigin_outcome', 'abc_rastrigin_time',
    'abc_sphere_outcome', 'abc_sphere_time',


    'tlbo_ackley_outcome', 'tlbo_ackley_time',
    'tlbo_griewank_outcome', 'tlbo_griewank_time',
    'tlbo_rastrigin_outcome', 'tlbo_rastrigin_time',
    'tlbo_sphere_outcome', 'tlbo_sphere_time',
])

result_data_df.to_csv("results/{}.csv".format(TEST_NAME))

print(result_data_df)