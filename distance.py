import os
import numpy as np
from scipy import interpolate


def target_pressure_fn():
    tmp0 = np.loadtxt('pressure0.csv', dtype=np.str, delimiter=',')
    a = [0, 6, 8]
    data0 = tmp0[1:-1, a].astype(np.float)
    data0 = data0[data0[:, 2] <= 0]
    x = data0[np.argsort(data0[:, 1])][:, 1]
    p = data0[np.argsort(data0[:, 1])][:, 0]

    fn = interpolate.interp1d(x, p, kind='cubic')

    return fn

def distance(fn):

    tmp = np.loadtxt('wing_openFoam/cand0.csv', dtype=np.str, delimiter=',')
    pressure = tmp[1:-1, 0].astype(np.float)

    x_new = tmp[1:-1, 2].astype(np.float)
    p_new = fn(x_new)

    dis = np.exp(-np.linalg.norm(pressure - p_new) / 1e5 / np.sqrt(pressure.shape))

    return dis


if __name__ == '__main__':
    print(distance())
