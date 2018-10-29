from model_v2 import *
from mcts_v2 import *
from distance import *
from blockMeshDict_v2 import *
from input_data import *

"""
run第二版：
神经网络输出数组V，储存子节点v_i
"""

n = 21
num = n
c_n2 = n * (n - 1) // 2

pi = np.zeros([n - 2, n * (n - 1) // 2])
air_input = np.zeros([n - 2, n, n])
val_i = np.zeros([n - 2, c_n2])

airfoil = np.zeros([n, n])
airfoil[n // 2, [0, n - 1]] = 1

x = tf.placeholder(tf.float32)

pre, pro, val = model(inputs=x, filters=32, blocks=8, n=num, training=False)
saver = tf.train.Saver()

z = tf.placeholder(dtype=tf.float32, shape=[None, 1])
pi_true = tf.placeholder(dtype=tf.float32, shape=[None, c_n2])
x_index = tf.placeholder(dtype=tf.float32, shape=[None , c_n2])
ones = tf.ones([c_n2, 1], dtype=tf.float32)
train, loss = mcts_train(z, val, pi_true, pre, x_index, ones)


if __name__ == '__main__':

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # saver.restore(sess, '/home/xie/tf_model/x')

    for _ in range(100):
        wing = Wing(airfoil)
        for i in range(n - 2):
            pi[i], ind = uct(wing, (n - i) * 20, pro, val, x, sess)
            air_input[i] = wing.airfoil
            val_i[i, ind] = 1
            wing.draw(index_point(ind, n))

        print(wing.airfoil)
        write_dict(wing.airfoil)

        os.system('./wing_openFoam/Allclean')
        blockMesh = os.system('blockMesh -case "wing_openFoam" > blockMesh.log')
        if blockMesh != 0:
            continue
        print('blockMesh done!')
        rhoSimpleFoam = os.system('rhoSimpleFoam -case "wing_openFoam" > rhoSimple.log')
        if rhoSimpleFoam != 0:
            continue
        print('rhoSimpleFoam down')
        os.system('paraFoam -touch -case "wing_openFoam"')
        os.system('mv wing_openFoam/wing_openFoam.OpenFOAM wing_openFoam/wing_openFoam.foam')
        os.system('pvpython wing_openFoam/sci.py')

        func = target_pressure_fn()
        dis = distance(func)
        print(dis)
        D = np.zeros(n - 2).reshape([-1, 1])
        D += dis

        for i in range(500):
            _, loss_value = sess.run((train, loss), {x: air_input, z: D, pi_true: pi, x_index: val_i})

    print(wing.airfoil)

