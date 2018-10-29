import tensorflow as tf
import numpy as np

v = np.arange(16).reshape(4, 4)
i = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

index = tf.placeholder(dtype=tf.float32, shape=[4, 4])
val = tf.constant(v, dtype=tf.float32)
ii = tf.ones([4, 1], dtype=tf.float32)
vi = tf.matmul(val * index, ii)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(vi, {index: i}))

