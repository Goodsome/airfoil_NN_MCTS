from reinforcemant_learning.tf_1_14.core import *
import numpy as np
import tensorflow as tf


class ReplayBuffer:

    def __init__(self, obs_dim, act_dim, size):
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(obs1=self.obs1_buf[idxs],
                    obs2=self.obs2_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])


def ddpg(env_fn, seed=0, replay_size=10000,
         gamma=0.99, pi_lr=1e-3, q_lr=1e-3, polyak=0.995, steps=100000, start_steps=10000,
         act_noise=0.0001, max_ep_len=1000, batch_size=100):

    np.random.seed(seed)

    obs_dim = DIM
    act_dim = DIM

    env = env_fn(obs_dim)
    env_test = env_fn(obs_dim)

    act_limit = 0.001

    x_ph, a_ph, x2_ph, r_ph, d_ph = placeholders(obs_dim, act_dim, obs_dim, None, None)

    with tf.variable_scope('main'):
        pi, q, q_pi = ddpg_actor_critic(x_ph, a_ph, hidden_sizes=(256, 256))

    with tf.variable_scope('target'):
        pi_tar, _, q_pi_tar = ddpg_actor_critic(x2_ph, a_ph, hidden_sizes=(256, 256))

    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    var_counts = tuple(count_vars(scope) for scope in ['main/pi', 'main/q', 'main'])
    print('\nNumber of parameters: \t pi: %d, \t q: %d, \t total: %d\n' % var_counts)

    backup = tf.stop_gradient(r_ph + gamma * (1 - d_ph) * q_pi_tar)

    pi_loss = -tf.reduce_mean(q_pi)
    q_loss = tf.reduce_mean((q - backup) ** 2)

    pi_optimizer = tf.train.AdamOptimizer(learning_rate=pi_lr)
    q_optimizer = tf.train.AdamOptimizer(learning_rate=q_lr)
    train_pi_op = pi_optimizer.minimize(pi_loss, var_list=get_vars('main/pi'))
    train_q_op = q_optimizer.minimize(q_loss, var_list=get_vars('main/q'))

    target_update = tf.group([tf.assign(v_tar, polyak * v_tar + (1 - polyak) * v_main)
                              for v_main, v_tar in zip(get_vars('main'), get_vars('target'))])

    target_init = tf.group([tf.assign(v_tar, v_main)
                            for v_main, v_tar in zip(get_vars('main'), get_vars('target'))])

    op = [pi, q, q_pi, q_pi_tar, pi_loss, q_loss]

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(target_init)

    def get_action(o, noise_scale):
        a = sess.run(pi, {x_ph: o.reshape(1, -1)})[0]
        a += noise_scale * np.random.randn(act_dim)
        return np.clip(a, -act_limit, act_limit)

    def test_agent(n=10):
        for j in range(n):
            env_test.reset()
            o, r, d = env_test.step(0)
            ep_ret, ep_len = 0, 0

            while not (d or ep_len == max_ep_len):
                o, r, d = env_test.step(get_action(o, 0))
                ep_ret += r
                ep_len += 1

            print('state:', o)
            print('distance:', env_test.distance)

    env.reset()
    o, r, d = env.step(0)
    ep_ret, ep_len = 0, 0

    for t in range(steps):
        if t >= start_steps:
            a = get_action(o, act_noise)
        else:
            a = np.random.randn(act_dim)

        o2, r, d = env.step(a)
        ep_ret += r
        ep_len += 1

        d = False if ep_len == max_ep_len else d

        # print(o, a, r, o2, d)

        replay_buffer.store(o, a, r, o2, d)

        o = o2

        if d or (ep_len == max_ep_len):

            for _ in range(ep_len):
                batch = replay_buffer.sample_batch(batch_size=batch_size)
                feed_dict = {x_ph: batch['obs1'],
                             x2_ph: batch['obs2'],
                             a_ph: batch['acts'],
                             r_ph: batch['rews'],
                             d_ph: batch['done']}

                sess.run(train_q_op, feed_dict)

                sess.run([train_pi_op, target_update], feed_dict)

            env.reset()
            o, r, d = env.step(0)
            ep_ret, ep_len = 0, 0

        if (t > steps * 0) and (t + 1) % max_ep_len == 0:
            print('\nstep {}\n'.format(t))
            test_agent(1)


wing = Env
ddpg(wing)
