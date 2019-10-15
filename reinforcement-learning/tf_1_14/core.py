import numpy as np
import tensorflow as tf

TARGET = [0.17133433, 0.15213932, 0.15627633, 0.1323941, 0.14898741,
          0.17141428, 0.15177738, 0.15705247, 0.13151355, 0.14955091]
DIM = len(TARGET)
EPS = 1e-8
LOG_STD_RATE = -1


class Env:

    def __init__(self, state_dim):
        self.state_dim = state_dim
        self.state = np.random.rand(state_dim)
        self.distance = np.sqrt(np.square(self.state - TARGET).mean())
        self.done = False

    def step(self, action):

        self.state += action
        new_distance = np.sqrt(np.square(self.state - TARGET).mean())
        # reward = 1 if new_distance < self.distance else -1
        reward = -new_distance * 10
        self.distance = new_distance
        if self.distance < 1e-4:
            self.done = True
        return self.state.copy(), reward, self.done

    def reset(self):
        self.state = np.random.rand(self.state_dim)
        self.done = False


def get_vars(scope):
    return [x for x in tf.global_variables() if scope in x.name]


def count_vars(scope):
    v = get_vars(scope)
    return sum([np.prod(var.shape.as_list()) for var in v])


def placeholder(dim=None):
    return tf.placeholder(dtype=tf.float32, shape=(None, dim) if dim else (None,))


def placeholders(*args):
    return [placeholder(dim) for dim in args]


def mlp(x, hidden_sizes=(32,), activation=tf.tanh, output_activation=None):
    for h in hidden_sizes[:-1]:
        x = tf.layers.dense(x, units=h, activation=activation)
    return tf.layers.dense(x, units=hidden_sizes[-1], activation=output_activation)


def gaussian_likelihood(x, mu, log_std):
    pre_sum = -0.5 * (tf.square((x - mu) / tf.exp(log_std)) + 2 * log_std + tf.log(2 * np.pi))
    return tf.reduce_sum(pre_sum, axis=1)


def mlp_gaussian_policy(x, a, hidden_sizes, activation, output_activation):
    act_dim = a.shape.as_list()[-1]
    mu = mlp(x, list(hidden_sizes) + [act_dim], activation, output_activation)
    log_std = tf.get_variable(name='log_std', initializer=LOG_STD_RATE*np.ones(act_dim, dtype=np.float32))
    std = tf.exp(log_std)
    pi = mu + tf.random_normal(tf.shape(mu)) * std
    logp = gaussian_likelihood(a, mu, log_std)
    logp_pi = gaussian_likelihood(pi, mu, log_std)
    return pi, logp, logp_pi


def mlp_actor_critic(x, a, hidden_size=(64, 64), activation=tf.tanh, output_activation=None):
    with tf.variable_scope('pi'):
        pi, logp, logp_pi = mlp_gaussian_policy(x, a, hidden_size, activation, output_activation)
    with tf.variable_scope('v'):
        v = tf.squeeze(mlp(x, list(hidden_size) + [1], activation, output_activation))

    return pi, logp, logp_pi, v


def ddpg_actor_critic(x, a, hidden_sizes=(400, 300), activation=tf.nn.relu, output_activation=tf.tanh):
    act_dim = a.shape.as_list()[-1]
    with tf.variable_scope('pi'):
        pi = 0.001 * mlp(x, list(hidden_sizes)+[act_dim], activation, output_activation)
    with tf.variable_scope('q'):
        q = tf.squeeze(mlp(tf.concat([x, a], axis=-1), list(hidden_sizes)+[1], activation, None), axis=1)
    with tf.variable_scope('q', reuse=True):
        q_pi = tf.squeeze(mlp(tf.concat([x, pi], axis=-1), list(hidden_sizes)+[1], activation, None), axis=1)
    return pi, q, q_pi

