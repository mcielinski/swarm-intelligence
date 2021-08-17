import pandas as pd 
import matplotlib.pyplot as plt

TEST_NUM = 100
TEST_NAME = "100_test_50_particles_200_iterations"
ALGORITHMS = ["pso", "abc", "tlbo"]
TEST_FUNCTIONS = ["ackley", "griewank", "rastrigin", "sphere"]
test = pd.read_csv("{}.csv".format(TEST_NAME))

for func in TEST_FUNCTIONS:
    name = []
    time = []
    outcome = [] 
    
    for alg in ALGORITHMS:
        name.append(alg)
        time.append(test["{}_{}_time".format(alg, func)].mean())
        outcome.append(test["{}_{}_outcome".format(alg, func)].sum()/TEST_NUM)

        
    test_result_df = pd.DataFrame({
        "name": name,
        "time": time,
        "accuracy": outcome,
    })
    
    print(test_result_df)

    test_result_df.plot.bar(x='name', y="time", title="Time {}".format(func))
    plt.savefig("./{}_{}_time".format(TEST_NAME, func))
    plt.clf()
    test_result_df.plot.bar(x='name', y="accuracy", title="Accuracy {}".format(func))
    plt.savefig("./{}_{}_accuracy".format(TEST_NAME, func))
    plt.clf()