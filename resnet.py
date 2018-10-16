from resnet_model import *
from input_data import n_11
from nn import train1
import numpy as np


air_input, point_true = n_11()
x = tf.placeholder(tf.float32)
pro, val = model(inputs=x, filters=32, blocks=1, n=11, training=True)

y_true = tf.constant(point_true, dtype=tf.float32)
train_p, loss_p = train1(pro, y_true)

saver = tf.train.Saver()

sess = tf.Session()

sess.run(tf.global_variables_initializer())

for i in range(5000):
    _, loss_value = sess.run((train_p, loss_p), {x: air_input})
    if i % 100 == 0:
        print(i, loss_value)
    if loss_value < 1e-7:
        break
save_path = saver.save(sess, '/home/xie/tf_model/x')

# saver.restore(sess, '/home/xie/tf_model/x')
p = sess.run(pro, {x: air_input})
print(np.argmax(p, axis=-1))
print(sess.run(val, {x: air_input}))

