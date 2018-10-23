import os
from resnet_model import *
from mcts import *
from distance import *
from blockMeshDict_v2 import *
from input_data import *


n = 11
num = n
c_n2 = n * (n - 1) // 2

pi = np.zeros([n - 2, n * (n - 1) // 2])
air_input, _ = inputs_data(naca0012(n))

airfoil = np.zeros([n, n])
airfoil[n // 2, [0, n - 1]] = 1

x = tf.placeholder(tf.float32)

pre, pro, val = model(inputs=x, filters=32, blocks=8, n=num, training=False)
saver = tf.train.Saver()

z = tf.placeholder(dtype=tf.float32, shape=[None, 1])
pi_true = tf.placeholder(dtype=tf.float32, shape=[None, c_n2])

train, loss = mcts_train(z, val, pi_true, pre)


if __name__ == '__main__':

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # saver.restore(sess, '/home/xie/tf_model/x')

    for _ in range(10):
        wing = Wing(airfoil)
        for i in range(n - 2):
            print(i)
            pi[i], point = uct(wing, n * 10, pro, val, x, sess)
            print(pi[i])
            wing.draw(point)

        print(wing.airfoil)
        write_dict(wing.airfoil)

        os.system('./wing_openFoam/Allclean')
        blockMesh = os.system('blockMesh -case "wing_openFoam" > blockMesh.log')
        if blockMesh != 0:
            continue
        print('blockMesh: ', blockMesh)
        rhoSimpleFoam = os.system('rhoSimpleFoam -case "wing_openFoam" > rhoSimple.log')
        print('rhoSimpleFoam :', rhoSimpleFoam)
        if rhoSimpleFoam != 0:
            continue
        os.system('paraFoam -touch -case "wing_openFoam"')
        os.system('mv wing_openFoam/wing_openFoam.OpenFOAM wing_openFoam/wing_openFoam.foam')
        os.system('pvpython wing_openFoam/scipt.py')

        D = np.zeros(n - 2).reshape([-1, 1])
        dis = 2 * distance() - 1
        print(dis)
        D += dis

        for i in range(500):
            _, loss_value = sess.run((train, loss), {x: air_input, z: D, pi_true: pi})
            if i % 100 == 0:
                print(loss_value)

        print(wing.airfoil)
