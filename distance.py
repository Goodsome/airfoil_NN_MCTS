import os
from blockMeshDict_v2 import *


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

    os.system('./wing_openFoam/Allclean')
    blockMesh = os.system('blockMesh -case "wing_openFoam" > blockMesh.log')
    if blockMesh != 0:
        dis = 0
        return dis
    print('blockMesh done!')
    rhoSimpleFoam = os.system('rhoSimpleFoam -case "wing_openFoam" > rhoSimple.log')
    if rhoSimpleFoam != 0:
        dis = 0
        return dis
    print('rhoSimpleFoam down')
    os.system('paraFoam -touch -case "wing_openFoam"')
    os.system('mv wing_openFoam/wing_openFoam.OpenFOAM wing_openFoam/wing_openFoam.foam')
    os.system('pvpython wing_openFoam/sci.py')

    tmp = np.loadtxt('wing_openFoam/cand0.csv', dtype=np.str, delimiter=',')
    pressure = tmp[1:-1, 0].astype(np.float)

    x_new = tmp[1:-1, 2].astype(np.float)
    p_new = fn(x_new)

    dis = np.exp(-np.linalg.norm(pressure - p_new) / 1e5 / np.sqrt(pressure.shape))
    dis = np.power(dis, 1)

    return dis


if __name__ == '__main__':
    wing_array = naca0012(points(61))
    wing_points = write_dict(wing_array)
    print(distance(target_pressure_fn()))
