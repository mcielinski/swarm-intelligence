import random

import test_functions
from PSO.pso import PSO
from TLBO.tlbo import TLBO
from ABC.abc import ABC


# PSO params
NUM_PARTICLES = 50
NUM_ITERATIONS_PSO = 200

# TLBO params
NUM_LEARNERS = 50
NUM_ITERATIONS_TLBO = 200

#ABC
COLONY_SIZE = 50
NUM_ITERATIONS_ABC = 200
MAX_TRAILS = 10

# other params
BOUND = 10
VERBOSE = False


def evaluate_pso():
    pso_ackley = PSO(num_particles=NUM_PARTICLES, num_iterations=NUM_ITERATIONS_PSO, test_func=test_functions.ackley, 
                        lower_bounds=[-1 * BOUND, -1 * BOUND], upper_bounds=[BOUND, BOUND], verbose_flag=VERBOSE)
    min_x, min_y = pso_ackley.optimize()
    print(f'Ackley min: \tx={min_x}, y={min_y}')

    pso_griewank = PSO(num_particles=NUM_PARTICLES, num_iterations=NUM_ITERATIONS_PSO, test_func=test_functions.griewank, 
                        lower_bounds=[-1 * BOUND, -1 * BOUND], upper_bounds=[BOUND, BOUND], verbose_flag=VERBOSE)
    min_x, min_y = pso_griewank.optimize()
    print(f'Griewank min: \tx={min_x}, y={min_y}')

    pso_rastrigin = PSO(num_particles=NUM_PARTICLES, num_iterations=NUM_ITERATIONS_PSO, test_func=test_functions.rastrigin, 
                        lower_bounds=[-1 * BOUND, -1 * BOUND], upper_bounds=[BOUND, BOUND], verbose_flag=VERBOSE)
    min_x, min_y = pso_rastrigin.optimize()
    print(f'Rastrigin min: \tx={min_x}, y={min_y}')

    pso_sphere = PSO(num_particles=NUM_PARTICLES, num_iterations=NUM_ITERATIONS_PSO, test_func=test_functions.sphere, 
                        lower_bounds=[-1 * BOUND, -1 * BOUND], upper_bounds=[BOUND, BOUND], verbose_flag=VERBOSE)
    min_x, min_y = pso_sphere.optimize()
    print(f'Sphere min: \tx={min_x}, y={min_y}')


def evaluate_abc():
    abc_ackley = ABC(colony_size=COLONY_SIZE, max_trials=MAX_TRAILS, num_iterations=NUM_ITERATIONS_ABC, test_func=test_functions.ackley, 
                        lower_bounds=[-1 * BOUND, -1 * BOUND], upper_bounds=[BOUND, BOUND], verbose_flag=VERBOSE)
    min_x, min_y = abc_ackley.optimize()
    print(f'Ackley min: \tx={min_x}, y={min_y}')

    abc_griewank = ABC(colony_size=COLONY_SIZE, max_trials=MAX_TRAILS, num_iterations=NUM_ITERATIONS_ABC, test_func=test_functions.griewank, 
                        lower_bounds=[-1 * BOUND, -1 * BOUND], upper_bounds=[BOUND, BOUND], verbose_flag=VERBOSE)
    min_x, min_y = abc_griewank.optimize()
    print(f'Griewank min: \tx={min_x}, y={min_y}')

    abc_rastrigin = ABC(colony_size=COLONY_SIZE, max_trials=MAX_TRAILS, num_iterations=NUM_ITERATIONS_ABC, test_func=test_functions.rastrigin, 
                        lower_bounds=[-1 * BOUND, -1 * BOUND], upper_bounds=[BOUND, BOUND], verbose_flag=VERBOSE)
    min_x, min_y = abc_rastrigin.optimize()
    print(f'Rastrigin min: \tx={min_x}, y={min_y}')

    abc_sphere = ABC(colony_size=COLONY_SIZE, max_trials=MAX_TRAILS, num_iterations=NUM_ITERATIONS_ABC, test_func=test_functions.sphere, 
                        lower_bounds=[-1 * BOUND, -1 * BOUND], upper_bounds=[BOUND, BOUND], verbose_flag=VERBOSE)
    min_x, min_y = abc_sphere.optimize()
    print(f'Sphere min: \tx={min_x}, y={min_y}')


def evaluate_tlbo():
    tlbo_ackley = TLBO(num_learners=NUM_LEARNERS, num_iterations=NUM_ITERATIONS_TLBO, test_func=test_functions.ackley, 
                        lower_bounds=[-1 * BOUND, -1 * BOUND], upper_bounds=[BOUND, BOUND], verbose_flag=VERBOSE)
    min_x, min_y = tlbo_ackley.optimize()
    print(f'Ackley min: \tx={min_x}, y={min_y}')

    tlbo_griewank = TLBO(num_learners=NUM_LEARNERS, num_iterations=NUM_ITERATIONS_TLBO, test_func=test_functions.griewank, 
                        lower_bounds=[-1 * BOUND, -1 * BOUND], upper_bounds=[BOUND, BOUND], verbose_flag=VERBOSE)
    min_x, min_y = tlbo_griewank.optimize()
    print(f'Griewank min: \tx={min_x}, y={min_y}')

    tlbo_rastrigin = TLBO(num_learners=NUM_LEARNERS, num_iterations=NUM_ITERATIONS_TLBO, test_func=test_functions.rastrigin, 
                        lower_bounds=[-1 * BOUND, -1 * BOUND], upper_bounds=[BOUND, BOUND], verbose_flag=VERBOSE)
    min_x, min_y = tlbo_rastrigin.optimize()
    print(f'Rastrigin min: \tx={min_x}, y={min_y}')

    tlbo_sphere = TLBO(num_learners=NUM_LEARNERS, num_iterations=NUM_ITERATIONS_TLBO, test_func=test_functions.sphere, 
                        lower_bounds=[-1 * BOUND, -1 * BOUND], upper_bounds=[BOUND, BOUND], verbose_flag=VERBOSE)
    min_x, min_y = tlbo_sphere.optimize()
    print(f'Sphere min: \tx={min_x}, y={min_y}')


if __name__ == '__main__':
    print('========== EVALUATE PSO ==========')
    evaluate_pso()
    print('==================================')
    print('========== EVALUATE ABC ==========')
    evaluate_abc()
    print('==================================')
    print('========== EVALUATE TLBO ==========')
    evaluate_tlbo()
    print('==================================')
