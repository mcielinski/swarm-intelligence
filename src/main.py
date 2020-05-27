import random

import test_functions
from PSO.pso import PSO


NUM_PARTICLES = 50
NUM_ITERATIONS = 10
BOUND = 20
VERBOSE = False


def evaluate_pso():
    pso_ackley = PSO(num_particles=NUM_PARTICLES, num_iterations=NUM_ITERATIONS, test_func=test_functions.ackley, 
                        lower_bounds=[-1 * BOUND, -1 * BOUND], upper_bounds=[BOUND, BOUND], verbose_flag=VERBOSE)
    min_x, min_y = pso_ackley.optimize()
    print(f'Ackley min: \tx={min_x}, y={min_y}')

    pso_griewank = PSO(num_particles=NUM_PARTICLES, num_iterations=NUM_ITERATIONS, test_func=test_functions.griewank, 
                        lower_bounds=[-1 * BOUND, -1 * BOUND], upper_bounds=[BOUND, BOUND], verbose_flag=VERBOSE)
    min_x, min_y = pso_griewank.optimize()
    print(f'Griewank min: \tx={min_x}, y={min_y}')

    pso_rastrigin = PSO(num_particles=NUM_PARTICLES, num_iterations=NUM_ITERATIONS, test_func=test_functions.rastrigin, 
                        lower_bounds=[-1 * BOUND, -1 * BOUND], upper_bounds=[BOUND, BOUND], verbose_flag=VERBOSE)
    min_x, min_y = pso_rastrigin.optimize()
    print(f'Rastrigin min: \tx={min_x}, y={min_y}')

    pso_sphere = PSO(num_particles=NUM_PARTICLES, num_iterations=NUM_ITERATIONS, test_func=test_functions.sphere, 
                        lower_bounds=[-1 * BOUND, -1 * BOUND], upper_bounds=[BOUND, BOUND], verbose_flag=VERBOSE)
    min_x, min_y = pso_sphere.optimize()
    print(f'Sphere min: \tx={min_x}, y={min_y}')


def evaluate_abc():
    for _ in range(100):
        lb = [-20, -20]
        ub = [20, 20]
        r_1 = random.random()
        r_2 = random.random()

        x_rand = lb[0] + (ub[0] - lb[0]) * r_1
        y_rand = lb[1] + (ub[1] - lb[1]) * r_2

        print([x_rand, y_rand])


def evaluate_tlbo():
    pass


if __name__ == '__main__':
    print('========== EVALUATE PSO ==========')
    evaluate_pso()
    print('==================================')
    # print('========== EVALUATE ABC ==========')
    # evaluate_abc()
    # print('==================================')
    # print('========== EVALUATE TLBO ==========')
    # evaluate_tlbo()
    # print('==================================')
