import random

import test_functions
from PSO.pso import PSO
from TLBO.tlbo import TLBO
from ABC.abc import ABC
from helpers import plot


# PSO params
NUM_PARTICLES = 100
NUM_ITERATIONS_PSO = 100

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
ROUND_PREC = 4


def evaluate_pso():
    pso_ackley = PSO(num_particles=NUM_PARTICLES, num_iterations=NUM_ITERATIONS_PSO, test_func=test_functions.ackley, 
                        lower_bounds=[-1 * BOUND, -1 * BOUND], upper_bounds=[BOUND, BOUND], verbose_flag=VERBOSE)
    min_x, min_y = pso_ackley.optimize()
    print(f'Ackley min: \tx={round(min_x, ROUND_PREC)}, y={round(min_y, ROUND_PREC)}')

    pso_griewank = PSO(num_particles=NUM_PARTICLES, num_iterations=NUM_ITERATIONS_PSO, test_func=test_functions.griewank, 
                        lower_bounds=[-1 * BOUND, -1 * BOUND], upper_bounds=[BOUND, BOUND], verbose_flag=VERBOSE)
    min_x, min_y = pso_griewank.optimize()
    print(f'Griewank min: \tx={round(min_x, ROUND_PREC)}, y={round(min_y, ROUND_PREC)}')

    pso_rastrigin = PSO(num_particles=NUM_PARTICLES, num_iterations=NUM_ITERATIONS_PSO, test_func=test_functions.rastrigin, 
                        lower_bounds=[-1 * BOUND, -1 * BOUND], upper_bounds=[BOUND, BOUND], verbose_flag=VERBOSE)
    min_x, min_y = pso_rastrigin.optimize()
    print(f'Rastrigin min: \tx={round(min_x, ROUND_PREC)}, y={round(min_y, ROUND_PREC)}')

    pso_sphere = PSO(num_particles=NUM_PARTICLES, num_iterations=NUM_ITERATIONS_PSO, test_func=test_functions.sphere, 
                        lower_bounds=[-1 * BOUND, -1 * BOUND], upper_bounds=[BOUND, BOUND], verbose_flag=VERBOSE)
    min_x, min_y = pso_sphere.optimize()
    print(f'Sphere min: \tx={round(min_x, ROUND_PREC)}, y={round(min_y, ROUND_PREC)}')


def evaluate_abc():
    abc_ackley = ABC(colony_size=COLONY_SIZE, max_trials=MAX_TRAILS, num_iterations=NUM_ITERATIONS_ABC, test_func=test_functions.ackley, 
                        lower_bounds=[-1 * BOUND, -1 * BOUND], upper_bounds=[BOUND, BOUND], verbose_flag=VERBOSE)
    min_x, min_y = abc_ackley.optimize()
    print(f'Ackley min: \tx={round(min_x, ROUND_PREC)}, y={round(min_y, ROUND_PREC)}')
    # plot(abc_ackley.optimality_tracking)

    abc_griewank = ABC(colony_size=COLONY_SIZE, max_trials=MAX_TRAILS, num_iterations=NUM_ITERATIONS_ABC, test_func=test_functions.griewank, 
                        lower_bounds=[-1 * BOUND, -1 * BOUND], upper_bounds=[BOUND, BOUND], verbose_flag=VERBOSE)
    min_x, min_y = abc_griewank.optimize()
    print(f'Griewank min: \tx={round(min_x, ROUND_PREC)}, y={round(min_y, ROUND_PREC)}')

    abc_rastrigin = ABC(colony_size=COLONY_SIZE, max_trials=MAX_TRAILS, num_iterations=NUM_ITERATIONS_ABC, test_func=test_functions.rastrigin, 
                        lower_bounds=[-1 * BOUND, -1 * BOUND], upper_bounds=[BOUND, BOUND], verbose_flag=VERBOSE)
    min_x, min_y = abc_rastrigin.optimize()
    print(f'Rastrigin min: \tx={round(min_x, ROUND_PREC)}, y={round(min_y, ROUND_PREC)}')

    abc_sphere = ABC(colony_size=COLONY_SIZE, max_trials=MAX_TRAILS, num_iterations=NUM_ITERATIONS_ABC, test_func=test_functions.sphere, 
                        lower_bounds=[-1 * BOUND, -1 * BOUND], upper_bounds=[BOUND, BOUND], verbose_flag=VERBOSE)
    min_x, min_y = abc_sphere.optimize()
    print(f'Sphere min: \tx={round(min_x, ROUND_PREC)}, y={round(min_y, ROUND_PREC)}')


def evaluate_tlbo():
    tlbo_ackley = TLBO(num_learners=NUM_LEARNERS, num_iterations=NUM_ITERATIONS_TLBO, test_func=test_functions.ackley, 
                        lower_bounds=[-1 * BOUND, -1 * BOUND], upper_bounds=[BOUND, BOUND], verbose_flag=VERBOSE)
    min_x, min_y = tlbo_ackley.optimize()
    print(f'Ackley min: \tx={round(min_x, ROUND_PREC)}, y={round(min_y, ROUND_PREC)}')

    tlbo_griewank = TLBO(num_learners=NUM_LEARNERS, num_iterations=NUM_ITERATIONS_TLBO, test_func=test_functions.griewank, 
                        lower_bounds=[-1 * BOUND, -1 * BOUND], upper_bounds=[BOUND, BOUND], verbose_flag=VERBOSE)
    min_x, min_y = tlbo_griewank.optimize()
    print(f'Griewank min: \tx={round(min_x, ROUND_PREC)}, y={round(min_y, ROUND_PREC)}')

    tlbo_rastrigin = TLBO(num_learners=NUM_LEARNERS, num_iterations=NUM_ITERATIONS_TLBO, test_func=test_functions.rastrigin, 
                        lower_bounds=[-1 * BOUND, -1 * BOUND], upper_bounds=[BOUND, BOUND], verbose_flag=VERBOSE)
    min_x, min_y = tlbo_rastrigin.optimize()
    print(f'Rastrigin min: \tx={round(min_x, ROUND_PREC)}, y={round(min_y, ROUND_PREC)}')

    tlbo_sphere = TLBO(num_learners=NUM_LEARNERS, num_iterations=NUM_ITERATIONS_TLBO, test_func=test_functions.sphere, 
                        lower_bounds=[-1 * BOUND, -1 * BOUND], upper_bounds=[BOUND, BOUND], verbose_flag=VERBOSE)
    min_x, min_y = tlbo_sphere.optimize()
    print(f'Sphere min: \tx={round(min_x, ROUND_PREC)}, y={round(min_y, ROUND_PREC)}')


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



# Plot 3D
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import numpy as np

# def f(x, y):
#     print(x)
#     print(y)
#     return x**2 + y**2

# x = np.linspace(-6, 6, 30)
# y = np.linspace(-6, 6, 30)

# X, Y = np.meshgrid(x, y)
# Z = f(X, Y)

# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.contour3D(X, Y, Z, 50, cmap='viridis')
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')

# plt.show()