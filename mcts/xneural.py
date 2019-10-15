import numpy as np
import tensorflow as tf
from tensorflow.python import debug as tf_debug
from mcts import *
from nn import airfoil_pre


def pre_train(y_pred, y_t):
    loss = tf.losses.mean_squared_error(labels=y_t, predictions=y_pred)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
    train = optimizer.minimize(loss)

    return train, loss


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return initial


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return initial


def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def conv_layer(input_tensor, input_dim, output_dim, layer_name):
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            w = weight_variable([5, 5, input_dim, output_dim])
            variable_summaries(w)
        with tf.name_scope('biases'):
            b = bias_variable([output_dim])
            variable_summaries(b)
        with tf.name_scope('conv_layer'):
            conv = tf.nn.conv2d(input_tensor, w, strides=[1, 1, 1, 1], padding='SAME')
            tf.summary.histogram('conv', conv)
        act = tf.nn.relu(conv + b)
        tf.summary.histogram('activations', act)
        return act


def fc_layer(input_tensor, input_dim, output_dim, layer_name):
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            w = weight_variable([input_dim, output_dim])
            variable_summaries(w)
        with tf.name_scope('biases'):
            b = bias_variable([output_dim])
            variable_summaries(b)
        with tf.name_scope('Wx_plus_b'):
            preactivate = tf.matmul(input_tensor, w) + b
            tf.summary.histogram('pre_activations', preactivate)
        act = tf.nn.relu(preactivate + b)
        tf.summary.histogram('activations', act)
        return act


def mmodel(input_tensor):
    input_layer = tf.reshape(input_tensor, shape=[-1, 11, 11, 1])

    layer1 = conv_layer(input_layer, 1, 32, 'layer1')

    flat = tf.reshape(layer1, [-1, 11 * 11 * 32])

    layer2 = fc_layer(flat, 11 * 11 * 32, 55, 'layer2')

    with tf.name_scope('layer3'):
        layer3 = tf.nn.softmax(layer2)
        tf.summary.histogram('layer3', layer3)

    return layer3


def model(il, n):
    input_layer = tf.reshape(il, shape=[-1, n, n, 1])

    layer = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding='same',
        activation=tf.nn.relu
    )

    layer = tf.layers.batch_normalization(layer)

    layer = tf.reshape(layer, [-1, n * n * 32])

    layer = tf.layers.dense(
        inputs=layer,
        units=n * (n - 1) // 2,
        activation=tf.nn.relu
    )

    # layer = tf.nn.softmax(layer)

    return layer


if __name__ == '__main__':

    num = 11
    c_n2 = num * (num - 1) // 2
    airfoil = np.zeros([num, num])
    airfoil[num // 2, [0, num - 1]] = 1

    sess = tf.InteractiveSession()

    x = tf.placeholder(tf.float32)
    prediction = mmodel(x)

    air_input = np.zeros([num - 2, num, num])
    point_true = np.zeros([num - 2, c_n2])
    for i in range(num - 2):
        air_input[i, :, :i + 1] = airfoil_pre[:, :i + 1]
        point_true[i, point_index(airfoil_pre[:, i + 1].reshape(-1), num)] = 1

    y_true = tf.constant(point_true, dtype=tf.float32)
    # train_p, loss_p = pre_train(prediction, y_true)

    # saver = tf.train.Saver()

    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter('/tmp/demo/1')
    writer.add_graph(sess.graph)

    sess.run(tf.global_variables_initializer())
    # sess = tf_debug.LocalCLIDebugWrapperSession(sess)

    sess.run(prediction, {x: air_input})

    # for i in range(3000):
    #     _, loss_value = sess.run((train_p, loss_p), {x: air_input})
    #     if i % 100 == 0:
    #         print(i, loss_value)
    #     if loss_value < 1e-8:
    #         break


