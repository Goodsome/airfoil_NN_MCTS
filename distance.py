import numpy as np
from scipy import interpolate


def distance():

    tmp = np.loadtxt('wing_openFoam/press0.csv', dtype=np.str, delimiter=',')
    a = [0, 6, 8]
    pressure = tmp[1:-1, 0].astype(np.float)

    tmp0 = np.loadtxt('pressure0.csv', dtype=np.str, delimiter=',')
    data0 = tmp0[1:-1, a].astype(np.float)
    data0 = data0[data0[:, 2] <= 0]
    x = data0[np.argsort(data0[:, 1])][:, 1]
    p = data0[np.argsort(data0[:, 1])][:, 0]

    f = interpolate.interp1d(x, p, kind='cubic')

    x_new = tmp[1:-1, 6].astype(np.float)
    p_new = f(x_new)

    dis = np.exp(-np.linalg.norm(pressure - p_new) / 1e5)

    return dis


if __name__ == '__main__':
    print(distance())

