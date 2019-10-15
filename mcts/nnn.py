from nn import *
import numpy as np
from mcts import point_index

airfoil_pre = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0,
                        0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0,
                        0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
                        1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
                        0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0,
                        0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0,
                        0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).reshape([11, 11])

if __name__ == '__main__':

    n = 11
    c_n2 = n * (n - 1) // 2
    airfoil = np.zeros([n, n])
    airfoil[n // 2, [0, n - 1]] = 1

    sess = tf.Session()
    # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    x = tf.placeholder(tf.float32)

    prediction, value = model(x, n, training=True)
    z = tf.placeholder(dtype=tf.float32, shape=[None, 1])
    pi_true = tf.placeholder(dtype=tf.float32, shape=[None, c_n2])

    train, loss = train2(z, pi_true, value, prediction)

    air_input = np.zeros([n - 2, n, n])
    point_true = np.zeros([n - 2, c_n2])
    for i in range(n - 2):
        air_input[i, :, :i + 1] = airfoil_pre[:, :i + 1]
        point_true[i, point_index(airfoil_pre[:, i + 1].reshape(-1), n)] = 1

    air_input[:, 5, 10] = 1

    y_true = tf.constant(point_true, dtype=tf.float32)
    train_p, loss_p = train1(prediction, y_true)

    saver = tf.train.Saver()

    sess.run(tf.global_variables_initializer())

    for i in range(1000):
        _, loss_value = sess.run((train_p, loss_p), {x: air_input})
        if i % 100 == 0:
            print(i, loss_value)
        if loss_value < 1e-7:
            break
    saver.save(sess, save_path='/home/xie/tf_model/xx')

    # saver.restore(sess, '/home/xie/tf_model/xx')
    pro, v = sess.run((prediction, value), {x: air_input})
    print(np.argmax(pro, axis=-1))
    print(v)
