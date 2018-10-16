import tensorflow as tf


def train1(y_pred, y_t):
    loss = tf.losses.mean_squared_error(labels=y_t, predictions=y_pred)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
    train = optimizer.minimize(loss)

    return train, loss


def train2(z, pi, v, p):
    loss = tf.losses.mean_squared_error(labels=z, predictions=v) + tf.losses.softmax_cross_entropy(pi, p)
    optimizer = tf.train.MomentumOptimizer(0.1, 0.9)
    train = optimizer.minimize(loss)

    return train, loss


def model(inp, n, training):
    inp = tf.reshape(inp, shape=[-1, n, n, 1])

    inp = tf.layers.conv2d(
        inputs=inp,
        filters=32,
        kernel_size=[5, 5],
        padding='same',
        activation=tf.nn.relu
    )

    inp = tf.layers.batch_normalization(inp, training=training)

    inp = tf.layers.conv2d(
        inputs=inp,
        filters=64,
        kernel_size=[5, 5],
        padding='same',
        activation=tf.nn.relu
    )

    inp = tf.layers.conv2d(
        inputs=inp,
        filters=128,
        kernel_size=[3, 3],
        padding='same',
        activation=tf.nn.relu
    )

    inp = tf.layers.conv2d(
        inputs=inp,
        filters=64,
        kernel_size=[3, 3],
        padding='same',
        activation=tf.nn.relu
    )

    inp = tf.reshape(inp, [-1, n * n * 64])

    inp = tf.layers.dense(
        inputs=inp,
        units=1024,
        activation=tf.nn.relu
    )

    inp = tf.layers.dense(
        inputs=inp,
        units=1024,
        activation=tf.nn.relu
    )

    inp = tf.layers.dense(
        inputs=inp,
        units=n * (n - 1) // 2,
        activation=tf.nn.relu
    )

    softmax = tf.nn.softmax(inp)

    inp = tf.layers.dense(
        inputs=inp,
        units=512,
    )

    dense5 = tf.layers.dense(
        inputs=inp,
        units=1,
    )

    return softmax, dense5


