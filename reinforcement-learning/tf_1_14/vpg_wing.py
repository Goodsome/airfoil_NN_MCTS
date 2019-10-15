from reinforcemant_learning.tf_1_14.core import *
import scipy.signal
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


TARGET = [0.17133433, 0.15213932, 0.15627633, 0.1323941, 0.14898741]
          # 0.17141428, 0.15177738, 0.15705247, 0.13151355, 0.14955091]
TARGET = [0.5]
DIM = len(TARGET)
LOG_STD_RATE = -2


class Env:

    def __init__(self, state_dim):
        self.state_dim = state_dim
        self.state = np.random.rand(state_dim)
        self.distance = np.sqrt(np.square(self.state - TARGET).mean())
        self.done = False

    def step(self, action):

        self.state += action
        new_distance = np.sqrt(np.square(self.state - TARGET).mean())
        reward = 1 if new_distance < self.distance else -1
        self.distance = new_distance
        if self.distance < 1e-4:
            self.done = True
        return self.state, reward, self.done

    def reset(self):
        self.state = np.random.rand(self.state_dim)
        self.done = False


class VPGBuffer:

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp):
        assert self.ptr < self.max_size
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = scipy.signal.lfilter([1], [1, float(-self.gamma*self.lam)], deltas[::-1], axis=0)[::-1]
        self.ret_buf[path_slice] = scipy.signal.lfilter([1], [1, float(-self.gamma)], rews[::-1], axis=0)[::-1][:-1]

        self.path_start_idx = self.ptr

    def get(self):
        assert self.ptr == self.max_size
        self.ptr, self.path_start_idx = 0, 0
        '''adv normalization'''
        return [self.obs_buf, self.act_buf, self.adv_buf, self.ret_buf, self.logp_buf]

    def show(self):
        return


def vpg(env, epochs=1000, steps_per_epoch=1, gamma=0.99, lam=0.97, max_ep_len=1000,
        pi_lr=0.0001, v_lr=0.0001, train_v_iters=80):

    seed = 2
    tf.set_random_seed(seed)
    np.random.seed(seed)

    buf = VPGBuffer(DIM, DIM, steps_per_epoch, gamma, lam)

    x_ph = tf.placeholder(shape=[None, DIM], dtype=tf.float32)
    a_ph = tf.placeholder(shape=[None, DIM], dtype=tf.float32)
    adv_ph = tf.placeholder(shape=None, dtype=tf.float32)
    ret_ph = tf.placeholder(shape=None, dtype=tf.float32)
    logp_old_ph = tf.placeholder(shape=None, dtype=tf.float32)

    pi, logp, logp_pi, v = mlp_actor_critic(x_ph, a_ph)

    all_phs = [x_ph, a_ph, adv_ph, ret_ph, logp_old_ph]
    get_action_ops = [pi, v, logp_pi]

    pi_loss = -tf.reduce_mean(logp * adv_ph)
    v_loss = tf.reduce_mean((ret_ph - v) ** 2)

    train_pi = tf.train.AdamOptimizer(learning_rate=pi_lr).minimize(pi_loss)
    train_v = tf.train.AdamOptimizer(learning_rate=v_lr).minimize(v_loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    def update():
        inputs = {k: v for k, v in zip(all_phs, buf.get())}
        pi_l_old, v_l_old = sess.run([pi_loss, v_loss], feed_dict=inputs)
        print('pi loss: {}, v loss: {}'.format(pi_l_old, v_l_old))
        # logp_y, adv_y = sess.run([logp, adv_ph], feed_dict=inputs)

        sess.run(train_pi, feed_dict=inputs)

        for _ in range(train_v_iters):
            sess.run(train_v, feed_dict=inputs)

        pi_l_new, v_l_new = sess.run([pi_loss, v_loss], feed_dict=inputs)
        print('pi loss: {}, v loss: {}'.format(pi_l_new, v_l_new))

    env.reset()
    o, r, d = env.step(0)
    ep_ret, ep_len = 0, 0
    print(env.state)

    for epoch in range(epochs):
        for t in range(steps_per_epoch):
            a, v_t, logp_t = sess.run(get_action_ops, feed_dict={x_ph: o.reshape(1, -1)})
            buf.store(o, a, r, v_t, logp_t)

            print(a)
            o, r, d = env.step(a[0])
            if t % 1 == 0:
                print('\nepoch {}, step {}: state {} distance {}'.format(epoch, t, env.state, env.distance))
            ep_ret += r
            ep_len += 1

            terminal = d or (ep_len == max_ep_len)
            if terminal or (t == steps_per_epoch - 1):
                if not terminal:
                    print('trajectory cun off by epoch at %d step.' % ep_len)
                last_val = r if d else sess.run(v, {x_ph: o.reshape(1, -1)})

                print(last_val)
                print('reward:', buf.rew_buf)
                buf.finish_path(last_val)
                print('return:', buf.ret_buf)

                env.reset()
                o, r, d = env.step(0)
                ep_ret, ep_len = 0, 0

        update()


wing = Env(DIM)
vpg(wing)

