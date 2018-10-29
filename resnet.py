from model_v2 import *
from input_data import *

num = 11
airfoil = naca0012(num)
air_input, point_true = inputs_data(airfoil)
x = tf.placeholder(tf.float32)
pre, pro, val = model(inputs=x, filters=32, blocks=8, n=num, training=False)

y_true = tf.constant(point_true, dtype=tf.float32)
train_p, loss_p = pre_train(y_true, pre)

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
p = sess.run(pre, {x: air_input})
print(np.argmax(p, axis=-1))
