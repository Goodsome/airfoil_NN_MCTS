from mcts import point_index
import numpy as np


def n_11():
    n = 11
    airfoil = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0,
                        0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0,
                        0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
                        1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
                        0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0,
                        0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0,
                        0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).reshape([n, n])

    c_n2 = n * (n - 1) // 2
    air_input = np.zeros([n - 2, n, n])
    point_true = np.zeros([n - 2, c_n2])
    for i in range(n - 2):
        air_input[i, :, :i + 1] = airfoil[:, :i + 1]
        point_true[i, point_index(airfoil[:, i + 1].reshape(-1), n)] = 1

    air_input[:, 5, 10] = 1

    return air_input, point_true


def naca0012():
    tmp = np.loadtxt('dian0.csv', dtype=np.str, delimiter=',')
