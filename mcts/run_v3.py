from model_v3 import *
from mcts_v3 import *
from distance import *
from blockMeshDict_v3 import *
from input_data import *
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

"""
第三版本
给定一个初始机翼，在此基础上的变化翼型。
翼型数组为由上到下。
"""

RANG_X = [0, 1]
RANG_Z = [-0.1, 0.1]
SPLIT = 20

n = SPLIT + 1
precision = 11
points_num = n * 2 - 4

pi = np.zeros([points_num, precision])
val_i = np.zeros([points_num, precision])
air_input = np.zeros([points_num, n, n])

airfoil = naca0012(points(n), train=True)

x = tf.placeholder(tf.float32)

pre, pro, val = model(inputs=x, filters=32, blocks=16, n=n, training=False, pre=precision)
saver = tf.train.Saver()

z = tf.placeholder(dtype=tf.float32, shape=[None, 1])
pi_true = tf.placeholder(dtype=tf.float32, shape=[None, precision])
x_index = tf.placeholder(dtype=tf.float32, shape=[None, precision])
ones = tf.ones([precision, 1], dtype=tf.float32)
train, loss = mcts_train(z, val, pi_true, pre, x_index, ones)


if __name__ == '__main__':

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # saver.restore(sess, '/home/xie/tf_model/x')

    for _ in range(100):
        print(_, 'iteration')
        wing = Wing(airfoil)
        for i in range(points_num):
            pi[i], index = uct(wing, (points_num - i) * 5, pro, val, x, sess, t=2)
            air_input[i] = wing.airfoil
            val_i[i, index] = 1
            wing.draw(index)

        print(wing.airfoil.astype(np.int))

        wing_shape = write_dict(wing.airfoil)
        wing_shape = np.row_stack((wing_shape, wing_shape[0]))

        func = target_pressure_fn()
        dis = distance(func)
        print(dis)
        D = np.zeros(points_num).reshape([-1, 1])
        D += dis

        name = 'iteration_%d_distance=%f' % (_, dis)
        fig_path = 'pic/iteration_%d.png' % _
        fig, ax = plt.subplots()
        ax.plot(wing_shape[:, 0], wing_shape[:, 1])
        plt.xlim(0, 1)
        plt.ylim(-0.5, 0.5)
        ax.set(title=name)
        fig.savefig(fig_path)
        plt.close()
        print('save wing shape')

        for i in range(200):
            train_, loss_value = sess.run((train, loss), {x: air_input, z: D, pi_true: pi, x_index: val_i})
            if i == 0:
                print(i, '\n', loss_value)
            if (i + 1) % 100 == 0:
                print(i, '\n', loss_value)

        save_path = saver.save(sess, '/home/xie/tf_model/x')
