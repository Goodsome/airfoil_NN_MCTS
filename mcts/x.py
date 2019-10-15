import os
from blockMeshDict_v2 import *
from input_data import *

np.set_printoptions(suppress=True)
np.set_printoptions(threshold=np.nan)


arr = naca0012(points(21))
ws = write_dict(arr)

#ws = np.row_stack((ws, ws[0]))
#name = 'target_dis=0.92'
#fig_path  = 'pic/' + name + '.png'
#fig, ax = plt.subplots()
#ax.plot(ws[:, 0], ws[:, 1])
#plt.xlim(0, 1)
#plt.ylim(-0.5, 0.5)
#ax.set(title=name)
#fig.savefig(fig_path)
#plt.close()

# os.system('blockMesh -case "wing_openFoam"')
# os.system('paraFoam -case "wing_openFoam"')
