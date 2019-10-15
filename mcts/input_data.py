import numpy as np
import pdb
from scipy import interpolate

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


def points(n):
    tmp = np.loadtxt('dian0.csv', dtype=np.str, delimiter=',')
    coordinate = tmp[1:, [5, 6, 7]].astype(np.float)
    x_z = coordinate[coordinate[:, 1] == 0][:, [0, 2]]
    z_neg = x_z[x_z[:, 1] <= 0.0001]
    z_neg = z_neg[np.argsort(z_neg[:, 0])]

    f_n = interpolate.interp1d(z_neg[:, 0], z_neg[:, 1], kind='cubic')

    x_n = np.linspace(0, 1, n)

    z_n = f_n(x_n)

    result = np.concatenate((x_n, x_n[1:-1], z_n, -z_n[1:-1])).reshape(2, -1).T
    return result


def naca0012(p=None, train=False):
    if p is None:
        n = 21
        x = np.linspace(0, 1, n)
        z = 0.1
        p = np.zeros([(n - 1) * 2, 2])
        p[:n, 0] = x[::-1]
        p[n:, 0] = x[1:-1]
        p[1:n - 1, 1] = -z
        p[n:, 1] = z

    else:
        n = p.shape[0] // 2 + 1

    p[:, 1] /= (0.1 / 0.5)
    if train:
        p[:, 1] /= 1.2
    p *= n - 1
    p = np.rint(p).astype(np.int)
    p[:, 1] += (n - 1) // 2

    air = np.zeros([n, n])
    for i in p:
        air[tuple(i)] = 1
    return air

