import matplotlib.pyplot as plt
from math import *
from openfoam.distance import *

N_b = 4
Y_teu = 0
Y_tel = 0
TARGET = [0.17133433, 0.15213932, 0.15627633, 0.1323941, 0.14898741,
          0.17141428, 0.15177738, 0.15705247, 0.13151355, 0.14955091]


def cst(ac):
    N_b = len(ac) // 2 - 1

    def c(x, n1=0.5, n2=1.0):
        return x ** n1 * (1 - x) ** n2

    def s(x, a):
        return np.sum(a[i] * factorial(N_b)/(factorial(i)*factorial(N_b-i))*x**i*(1-x)**(N_b-i) for i in range(N_b+1))

    def y_u(x):
        return c(x) * s(x, ac[:N_b+1]) + x * Y_teu

    def y_l(x):
        return -c(x) * s(x, ac[-N_b-1:]) + x * Y_tel

    return y_u, y_l


def err(ac, x, yu, yl):
    y_u, y_l = cst(ac)
    return np.square(y_u(x) - yu) + np.square(y_l(x) - yl)


def cal_dis(state):
    y_up, y_low = cst(state)

    x_ = np.linspace(0, 1, 100)
    x_up = x_[1:-1]
    x_low = x_[::-1]

    p = np.concatenate((x_low, x_up, y_low(x_low), y_up(x_up))).reshape(2, -1).T
    write_dict(p)
    dis = distance(target_pressure_fn())

    print(dis)

def plot_cst(state):
    y_up, y_low, = cst(state)

    x_ = np.linspace(0, 1, 100)

    # tmp = np.loadtxt('dian0.csv', dtype=np.str, delimiter=',')[1:].astype(np.float)

    # plt.plot(tmp[:, 0], tmp[:, 1])
    # plt.plot(tmp[:, 0], -tmp[:, 1])
    plt.plot(x_, y_up(x_))
    plt.plot(x_, y_low(x_))
    plt.xlim(0, 1)
    plt.ylim(-0.5, 0.5)
    plt.show()
    plt.close()


cal_dis(TARGET)


