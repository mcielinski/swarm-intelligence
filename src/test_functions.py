import numpy as np


def ackley(d, a=20, b=0.2, c=2*np.pi):
    # https://www.sfu.ca/~ssurjano/ackley.html
    # Global minimum: f(X)=0, at X=(0,...,0)

    sum_part1 = np.sum([x**2 for x in d])
    sum_part2 = np.sum([np.cos(c * x) for x in d])

    part1 = -1.0 * a * np.exp(-1.0 * b * np.sqrt((1.0/len(d)) * sum_part1))
    part2 = -1.0 * np.exp((1.0 / len(d)) * sum_part2)

    return part1 + part2 + a + np.exp(1)


def griewank(d):
    # https://www.sfu.ca/~ssurjano/griewank.html
    # Global minimum: f(X)=0, at X=(0,...,0)

    sum_part = 1.0 / 4000 * np.sum([(x**2) for x in d])
    product_part = np.prod([np.cos(x / np.sqrt(i+1.0)) for i, x in enumerate(d)])

    return sum_part - product_part + 1


def rastrigin(d):
    # https://www.sfu.ca/~ssurjano/rastr.html
    # Global minimum: f(X)=0, at X=(0,...,0)

    sum_part = np.sum([x**2 - 10*np.cos(2 * np.pi * x) for x in d])
    return 10 * len(d) + sum_part


def sphere(d):
    # https://www.sfu.ca/~ssurjano/spheref.html
    # Global minimum: f(X)=0, at X=(0,...,0)

    return np.sum([x**2 for x in d])