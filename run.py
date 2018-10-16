import os
from resnet_model import *
from nn import train2
from mcts import *
from distance import *
from blockMeshDict import *
from input_data import *


n = 11
c_n2 = n * (n - 1) // 2

pi = np.zeros([n - 2, n * (n - 1) // 2])
input_airfoil = np.zeros([n - 2, n, n])

airfoil = np.zeros([n, n])
airfoil[n // 2, [0, n - 1]] = 1

x = tf.placeholder(tf.float32)

pro, val = model(inputs=x, filters=32, blocks=8, n=11, training=False)
saver = tf.train.Saver()

z = tf.placeholder(dtype=tf.float32, shape=[None, 1])
pi_true = tf.placeholder(dtype=tf.float32, shape=[None, c_n2])

train, loss = train2(z, pi_true, val, pro)


if __name__ == '__main__':

    sess = tf.Session()

    saver.restore(sess, '/home/xie/tf_model/x')

    for _ in range(1):
        wing = Wing(airfoil)
        for i in range(9):
            pi[i], point = uct(wing, 100, pro, val, x, sess)
            wing.draw(point)

        print(wing.airfoil)

        vertices, m = cal_vertices(wing.airfoil)
        write_block_mesh_dict(vertices, m)

        # os.system('blockMesh -case "wing_openFoam"')
        # os.system('rhoSimpleFoam -case "wing_openFoam" > rhoSimpleFoam.log')
        # os.system('paraFoam -touch -case "wing_openFoam"')
        # os.system('mv wing_openFoam/wing_openFoam.OpenFOAM wing_openFoam/wing_openFoam.foam')
        # os.system('pvpython wing_openFoam/scipt.py')
        #
        # D = np.zeros(n - 2).reshape([9, 1])
        # D += np.exp(-distance())
        #
        # for i in range(3000):
        #     _, loss_value = sess.run((train, loss), {x: input_airfoil, z: D, pi_true: pi})
        #     if i % 100 == 0:
        #         print(loss_value)
        #
        # print(wing.airfoil)
