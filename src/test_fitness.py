import random
import test_functions
from PSO.pso import PSO
from TLBO.tlbo import TLBO
from ABC.abc import ABC
import pandas as pd 
import matplotlib.pyplot as plt

# PSO params
NUM_PARTICLES = 50
NUM_ITERATIONS_PSO = 100

# TLBO params
NUM_LEARNERS = 50
NUM_ITERATIONS_TLBO = 100

#ABC
COLONY_SIZE = 50
NUM_ITERATIONS_ABC = 100
MAX_TRAILS = 10

# other params
BOUND = 10
VERBOSE = False



if __name__ == '__main__':
    pso_ackley = PSO(num_particles=NUM_PARTICLES, num_iterations=NUM_ITERATIONS_PSO, test_func=test_functions.ackley, 
                        lower_bounds=[-1 * BOUND, -1 * BOUND], upper_bounds=[BOUND, BOUND], verbose_flag=VERBOSE)
    min_x, min_y = pso_ackley.optimize()
    print(f'Ackley min: \tx={round(min_x, 4)}, y={round(min_y, 4)}')

    pso_griewank = PSO(num_particles=NUM_PARTICLES, num_iterations=NUM_ITERATIONS_PSO, test_func=test_functions.griewank, 
                        lower_bounds=[-1 * BOUND, -1 * BOUND], upper_bounds=[BOUND, BOUND], verbose_flag=VERBOSE)
    min_x, min_y = pso_griewank.optimize()
    print(f'Griewank min: \tx={round(min_x, 4)}, y={round(min_y, 4)}')

    pso_rastrigin = PSO(num_particles=NUM_PARTICLES, num_iterations=NUM_ITERATIONS_PSO, test_func=test_functions.rastrigin, 
                        lower_bounds=[-1 * BOUND, -1 * BOUND], upper_bounds=[BOUND, BOUND], verbose_flag=VERBOSE)
    min_x, min_y = pso_rastrigin.optimize()
    print(f'Rastrigin min: \tx={round(min_x, 4)}, y={round(min_y, 4)}')

    pso_sphere = PSO(num_particles=NUM_PARTICLES, num_iterations=NUM_ITERATIONS_PSO, test_func=test_functions.sphere, 
                        lower_bounds=[-1 * BOUND, -1 * BOUND], upper_bounds=[BOUND, BOUND], verbose_flag=VERBOSE)
    min_x, min_y = pso_sphere.optimize()
    print(f'Sphere min: \tx={round(min_x, 4)}, y={round(min_y, 4)}')

    abc_ackley = ABC(colony_size=COLONY_SIZE, max_trials=MAX_TRAILS, num_iterations=NUM_ITERATIONS_ABC, test_func=test_functions.ackley, 
                        lower_bounds=[-1 * BOUND, -1 * BOUND], upper_bounds=[BOUND, BOUND], verbose_flag=VERBOSE)
    min_x, min_y = abc_ackley.optimize()
    print(f'Ackley min: \tx={round(min_x, 4)}, y={round(min_y, 4)}')

    abc_griewank = ABC(colony_size=COLONY_SIZE, max_trials=MAX_TRAILS, num_iterations=NUM_ITERATIONS_ABC, test_func=test_functions.griewank, 
                        lower_bounds=[-1 * BOUND, -1 * BOUND], upper_bounds=[BOUND, BOUND], verbose_flag=VERBOSE)
    min_x, min_y = abc_griewank.optimize()
    print(f'Griewank min: \tx={round(min_x, 4)}, y={round(min_y, 4)}')

    abc_rastrigin = ABC(colony_size=COLONY_SIZE, max_trials=MAX_TRAILS, num_iterations=NUM_ITERATIONS_ABC, test_func=test_functions.rastrigin, 
                        lower_bounds=[-1 * BOUND, -1 * BOUND], upper_bounds=[BOUND, BOUND], verbose_flag=VERBOSE)
    min_x, min_y = abc_rastrigin.optimize()
    print(f'Rastrigin min: \tx={round(min_x, 4)}, y={round(min_y, 4)}')

    abc_sphere = ABC(colony_size=COLONY_SIZE, max_trials=MAX_TRAILS, num_iterations=NUM_ITERATIONS_ABC, test_func=test_functions.sphere, 
                        lower_bounds=[-1 * BOUND, -1 * BOUND], upper_bounds=[BOUND, BOUND], verbose_flag=VERBOSE)
    min_x, min_y = abc_sphere.optimize()
    print(f'Sphere min: \tx={round(min_x, 4)}, y={round(min_y, 4)}')

    tlbo_ackley = TLBO(num_learners=NUM_LEARNERS, num_iterations=NUM_ITERATIONS_TLBO, test_func=test_functions.ackley, 
                        lower_bounds=[-1 * BOUND, -1 * BOUND], upper_bounds=[BOUND, BOUND], verbose_flag=VERBOSE)
    min_x, min_y = tlbo_ackley.optimize()
    print(f'Ackley min: \tx={round(min_x, 4)}, y={round(min_y, 4)}')

    tlbo_griewank = TLBO(num_learners=NUM_LEARNERS, num_iterations=NUM_ITERATIONS_TLBO, test_func=test_functions.griewank, 
                        lower_bounds=[-1 * BOUND, -1 * BOUND], upper_bounds=[BOUND, BOUND], verbose_flag=VERBOSE)
    min_x, min_y = tlbo_griewank.optimize()
    print(f'Griewank min: \tx={round(min_x, 4)}, y={round(min_y, 4)}')

    tlbo_rastrigin = TLBO(num_learners=NUM_LEARNERS, num_iterations=NUM_ITERATIONS_TLBO, test_func=test_functions.rastrigin, 
                        lower_bounds=[-1 * BOUND, -1 * BOUND], upper_bounds=[BOUND, BOUND], verbose_flag=VERBOSE)
    min_x, min_y = tlbo_rastrigin.optimize()
    print(f'Rastrigin min: \tx={round(min_x, 4)}, y={round(min_y, 4)}')

    tlbo_sphere = TLBO(num_learners=NUM_LEARNERS, num_iterations=NUM_ITERATIONS_TLBO, test_func=test_functions.sphere, 
                        lower_bounds=[-1 * BOUND, -1 * BOUND], upper_bounds=[BOUND, BOUND], verbose_flag=VERBOSE)
    min_x, min_y = tlbo_sphere.optimize()
    print(f'Sphere min: \tx={round(min_x, 4)}, y={round(min_y, 4)}')
    
    pd.DataFrame({
        "pso": pso_ackley.optimality_tracking,
        "abc": abc_ackley.optimality_tracking,
        "tlbo": tlbo_ackley.optimality_tracking,
    }).plot(title="ackley fitness")
    plt.savefig("ackley_fitnes.png")
    plt.clf()

    pd.DataFrame({
        "pso": pso_griewank.optimality_tracking,
        "abc": abc_griewank.optimality_tracking,
        "tlbo": tlbo_griewank.optimality_tracking,
    }).plot(title="griewank fitness")
    plt.savefig("griewank_fitnes.png")
    plt.clf()

    pd.DataFrame({
        "pso": pso_sphere.optimality_tracking,
        "abc": abc_sphere.optimality_tracking,
        "tlbo": tlbo_sphere.optimality_tracking,
    }).plot(title="sphere fitness")
    plt.savefig("sphere_fitnes.png")
    plt.clf()

