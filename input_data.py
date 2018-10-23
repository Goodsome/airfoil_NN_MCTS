from mcts import point_index, index_point
import numpy as np

ERROR = 0.00001

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
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).reshape([11, 11])

air1 = np.array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., ],
                 [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., ],
                 [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., ],
                 [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., ],
                 [0., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., ],
                 [1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., ],
                 [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., ],
                 [0., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., ],
                 [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., ],
                 [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., ],
                 [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., ], ]).reshape(11, 11)


def inputs_data(airfoil):
    n = airfoil.shape[0]

    c_n2 = n * (n - 1) // 2
    air_input = np.zeros([n - 2, n, n])
    point_true = np.zeros([n - 2, c_n2])
    for i in range(n - 2):
        air_input[i, :, :i + 1] = airfoil[:, :i + 1]
        point_true[i, point_index(airfoil[:, i + 1].reshape(-1), n)] = 1

    air_input[:, 5, 10] = 1

    return air_input, point_true


def naca0012(n=101):
    x = np.linspace(0, 1, n)
    z = 0.1
    points = np.zeros([(n - 1) * 2, 2])
    points[:n, 0] = x[::-1]
    points[n:, 0] = x[1:-1]
    points[1:n - 1, 1] = -z
    points[n:, 1] = z

    points *= n - 1
    points[:, 1] += (n - 1) // 2
    points += ERROR

    air = np.zeros([n, n])
    for i in points:
        air[i.astype(np.int)[1], i.astype(np.int)[0]] = 1
    return air


if __name__ == '__main__':
    airfoil_shape = naca0012()

    np.set_printoptions(threshold=np.nan)
    x_input, y_true = inputs_data(airfoil_shape)
    print(x_input[4])
    print(np.argwhere(y_true == 1))
