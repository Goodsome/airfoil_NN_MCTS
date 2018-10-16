import tensorflow as tf
import numpy as np

saver = tf.train.Saver()

sess = tf.Session()

saver.restore(sess, '/home/xie/tf_model/x')

