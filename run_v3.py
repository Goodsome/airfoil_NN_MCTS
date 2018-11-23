from model_v3 import *
from mcts_v3 import *
from distance import *
from blockMeshDict_v2 import *
from input_data import *
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

"""
run第二版：
神经网络输出数组V，储存子节点v_i
"""

n = 21
precision = 11
points_num = n * 2 - 4

pi = np.zeros([points_num, precision])
val_i = np.zeros([points_num, precision])
air_input = np.zeros([points_num, n, n])

airfoil = naca0012(points(21))

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

    for _ in range(1):
        print(_, 'iteration')
        wing = Wing(airfoil)
        for i in range(points_num):
            print(i)
            pi[i], ind = uct(wing, (points_num - i) * 5, pro, val, x, sess)
            air_input[i] = wing.airfoil
            val_i[i, ind] = 1
            wing.draw(ind)

        # ws = write_dict(wing.airfoil)
        # ws = np.row_stack((ws, ws[0]))
        #
        # func = target_pressure_fn()
        # dis = distance(func)
        # print(dis)
        # D = np.zeros(n - 2).reshape([-1, 1])
        # D += dis
        #
        # name = 'iteration_%d_distance=%f' % (_, dis)
        # fig_path = 'pic/iteration_%d.png' % _
        # fig, ax = plt.subplots()
        # ax.plot(ws[:, 0], ws[:, 1])
        # plt.xlim(0, 1)
        # plt.ylim(-0.5, 0.5)
        # ax.set(title=name)
        # fig.savefig(fig_path)
        # plt.close()
        # print('save wing shape')
        #
        # for i in range(100):
        #     train_, loss_value = sess.run((train, loss), {x: air_input, z: D, pi_true: pi, x_index: val_i})
        #     if i % 10 == 0:
        #         print(i, loss_value)
        #
        # save_path = saver.save(sess, '/home/xie/tf_model/x')

    # print(wing.airfoil)

