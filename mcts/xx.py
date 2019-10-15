import sys
import multiprocessing

def hello(taskq, resultq):
    import tensorflow as tf
    config = tf,ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)
    while True:
        name = taskq.get()
        res  = sess.run(tf.constant('hello ' + name))
        resultq.put(res)


if __name__ == '__main__':
    taskq = multiprocessing.Queue()
    resultq = multiprocessing.Queue()
    p = multiprocessing.Process(target=hello, args=(taskq, resultq))
    p.start()


