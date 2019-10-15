import random
import sys
import time
from utilities import my_plot
import tensorflow as tf
from collections import namedtuple
from utilities.input_data import *

NA = 10
N = 11
TARGET_N = 2


def target_naca():
    x = naca_points(N//2 + 2)
    x = x[:, 1].reshape(2, -1)[1][1:]
    x /= (0.06 / (NA-1))
    x = np.around(x)
    x = np.concatenate((np.array([1]), x, x[::-1]))

    return x

class Wing:

    def __init__(self):
        self.state = np.zeros(N, dtype=np.int)
        self.state[0] = 1
        self.index = 1
        self.done = False
        self.reward = 0

    def step(self, action):
        self.state[self.index] = action
        self.index += 1
        if self.index == N:
            self.done = True
            self.reward = self.cal_reward()
        return self.state, self.reward, self.done

    def cal_reward(self):
        distance = np.sqrt(np.square(target_naca() - self.state).mean())
        r = -distance

        return r


class Estimator:

    def __init__(self, scope='estimator'):
        self.scope = scope
        with tf.variable_scope(scope):
            self._build_model()

    def _build_model(self):

        self.X_pl = tf.placeholder(tf.float32, shape=[None, N, 3], name='X')
        self.y_pl = tf.placeholder(tf.float32, shape=[None], name='y')
        self.actions_pl = tf.placeholder(tf.int32, shape=[None, 2], name='actions')

        inputs = tf.layers.dense(self.X_pl, units=1024, activation=tf.nn.relu)
        inputs = tf.layers.dense(inputs, units=512, activation=tf.nn.relu)
        inputs = tf.layers.dense(inputs, units=256, activation=tf.nn.relu)
        inputs = tf.layers.flatten(inputs)

        self.predictions = tf.layers.dense(inputs, units=NA)
        self.action_predictions = tf.gather_nd(self.predictions, self.actions_pl)

        self.loss = tf.losses.mean_squared_error(self.y_pl, self.action_predictions)

        self.optimizer = tf.train.AdamOptimizer()
        self.train_op = self.optimizer.minimize(self.loss)

    def predict(self, sess, state):

        return sess.run(self.predictions, {self.X_pl: state})

    def update(self, sess, s, a, y):

        _, loss = sess.run((self.train_op, self.loss), {self.X_pl: s, self.actions_pl: a, self.y_pl: y})

        return loss


def copy_model_parameters(sess, e1, e2):

    e1_params = [t for t in tf.trainable_variables() if t.name.startswith(e1.scope)]
    e1_params = sorted(e1_params, key=lambda v: v.name)
    e2_params = [t for t in tf.trainable_variables() if t.name.startswith(e2.scope)]
    e2_params = sorted(e2_params, key=lambda v: v.name)

    update_ops = []
    for e1_v, e2_v in zip(e1_params, e2_params):
        op = e2_v.assign(e1_v)
        update_ops.append(op)

    sess.run(update_ops)


def state_to_array(state):
    state = state[1:]
    state = state * 0.06 / NA
    state[N//2:] *= -1
    y = np.insert(state, [0, N//2, N-1], 0)
    x = np.linspace(0, 1, N//2 + 2)
    x = np.concatenate((x, x[:-1][::-1]))
    array = np.concatenate((x, y)).reshape(2, -1).T
    return array


def epsilon_greedy_policy(estimator, na):

    def policy_fn(sess, observation, epsilon):
        a = np.ones(na) * epsilon/na
        q_values = estimator.predict(sess, np.expand_dims(observation, 0))
        best_action = np.argmax(q_values)
        a[best_action] += 1 - epsilon

        return a

    return policy_fn


def q_learning(sess,
               env,
               q_estimator,
               target_estimator,
               num_episodes=100000,
               replay_memory_init_size=500,
               replay_memory_size=5000,
               discount_factor=1, epsilon=0.1, batch_size=32, update_target_estimator_every=20):

    Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done'])
    replay_memory = []

    policy = epsilon_greedy_policy(q_estimator, NA)
    print('Populating replay memory...')
    env.__init__()
    # 历史3步
    state = np.stack([env.state] * 3, axis=1)
    for i in range(replay_memory_init_size):
        action_p = policy(sess, state, epsilon)
        action = np.random.choice(np.arange(NA), p=action_p)
        next_state, reward, done = env.step(action)
        next_state = np.append(state[:, 1:], np.expand_dims(next_state, 1), axis=1)
        replay_memory.append(Transition(state, action, reward, next_state, done))
        if done:
            env.__init__()
            state = np.stack([env.state] * 3, axis=1)
        else:
            state = next_state

    epsilons = np.linspace(1.0, 0.01, num_episodes)
    res = []
    total_t = 0
    for i_episode in range(num_episodes):

        env.__init__()
        state = np.stack([env.state] * 3, axis=1)

        loss = None
        while not env.done:

            total_t += 1
            action_p = policy(sess, state, epsilons[i_episode])
            action = np.random.choice(np.arange(NA), p=action_p)
            next_state, reward, done = env.step(action)
            next_state = np.append(state[:, 1:], np.expand_dims(next_state, 1), axis=1)

            if len(replay_memory) == replay_memory_size:
                replay_memory.pop(0)

            replay_memory.append(Transition(state, action, reward, next_state, done))

            samples = random.sample(replay_memory, batch_size)
            state_batch, action_batch, reward_batch, next_state_batch, done_batch = map(np.array, zip(*samples))

            q_value_next = q_estimator.predict(sess, next_state_batch)
            best_actions = np.argmax(q_value_next, axis=1)
            q_value_next_target = target_estimator.predict(sess, next_state_batch)
            target_batch = reward_batch + np.invert(done_batch).astype(np.float32) * \
                discount_factor * q_value_next_target[np.arange(batch_size), best_actions]

            action_batch = np.stack([np.arange(batch_size), action_batch.reshape(-1)]).reshape(2, -1).T
            loss = q_estimator.update(sess, state_batch, action_batch, target_batch)

            state = next_state

        if (i_episode+1) % update_target_estimator_every == 0:
            copy_model_parameters(sess, q_estimator, target_estimator)
            print('\nCopied model parameters to target network')
            print('\rEpisode {}/{}, loss: {}'.format(i_episode, num_episodes, loss))
            sys.stdout.flush()

        if (i_episode+1) % (num_episodes // 100) == 0:
            my_plot.plot_wing_shape(state_to_array(env.state), name='{}'.format(i_episode), fig_path='pic/{}.png'.format(i_episode))
            print(env.state)
            print(-env.cal_reward())

        if (i_episode+1) % 10 == 0:
            res.append(-env.cal_reward())

    my_plot.plot_residual(res, fig_path='pic/residual_{}.png'.format(update_target_estimator_every))


def run():
    wing = Wing()
    q_estimator = Estimator(scope='q')
    target_estimator = Estimator(scope='target')

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        q_learning(sess, wing, q_estimator, target_estimator)


if __name__ == '__main__':
    start = time.time()
    # run()
    naca = state_to_array(target_naca())
    print(naca)
    my_plot.plot_wing_shape(naca, name='Target', fig_path='pic/target.png', note=True)
    print(time.time()-start)
