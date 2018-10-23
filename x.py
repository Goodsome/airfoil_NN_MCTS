import os
from blockMeshDict_v2 import *
from scipy import interpolate
from input_data import *

np.set_printoptions(suppress=True)

tmp = np.loadtxt('dian0.csv', dtype=np.str, delimiter=',')
ver = tmp[1:, [5, 6, 7]].astype(np.float)
x_z = ver[ver[:, 1] == 0][:, [0, 2]]
z_plus = x_z[x_z[:, 1] >= 0.00001]
z_plus = z_plus[np.argsort(z_plus[:, 0])]
z_neg = x_z[x_z[:, 1] <= 0.0001]
z_neg = z_neg[np.argsort(z_neg[:, 0])]

f_n = interpolate.interp1d(z_neg[:, 0], z_neg[:, 1], kind='cubic')

x_n = np.linspace(0, 1, 101)

z_n = f_n(x_n)

ver1 = np.concatenate((x_n, x_n[1:-1], z_n, -z_n[1:-1])).reshape(2, -1).T

print(ver1)
print(np.sort(ver1, axis=0))

# arr = naca0012(21)
#
# write_dict(ver1)
# os.system('blockMesh -case "wing_openFoam"')
# os.system('paraFoam -case "wing_openFoam"')
