from run import *


np.set_printoptions(suppress=True)

sess.run(tf.global_variables_initializer())

for i in range(10000):
    _, loss_value = sess.run((train_p, loss_p), {x: air_input})
    if i % 100 == 0:
        print(i, loss_value)

pred_prob = sess.run(pro, {x: air_input})
for i in pred_prob:
    print(i)
    print(np.argmax(i))
    print(index_point(np.argmax(i), 11))
