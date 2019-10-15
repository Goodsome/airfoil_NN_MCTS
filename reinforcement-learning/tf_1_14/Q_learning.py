import numpy as np
from collections import defaultdict


class Wing:
    def __init__(self, n=8, na=3):
        self.state = []
        self.done = False
        self.reward = 0
        self.na = na
        self.n = n

    def step(self, action):
        self.state.append(action)
        if len(self.state) == self.n:
            self.done = True
            self.reward = self.cal_reward()
        return ''.join(str(x) for x in self.state), self.reward, self.done

    def cal_reward(self):
        target = np.ones(self.n) * 1
        distance = np.sqrt(np.square(target - self.state).mean())
        r = -distance

        return r


def draw(state, na=5):
    n = len(state)
    out = np.zeros([na, n])
    for i, s in enumerate(state):
        out[int(s), i] = 1

    wing_shape = np.zeros([2*na + 1, n // 2 + 2])
    wing_shape[na, (0, -1)] = 1
    wing_shape[0:na, 1:-1] = out[:, :n//2]
    wing_shape[na+1:, 1:-1] = out[:, n//2:]
    return wing_shape


def epsilon_greedy_policy(q, epsilon, na):

    def policy_fn(observation):
        a = np.ones(na) * epsilon/na
        best_action = np.argmax(q[observation])
        a[best_action] += 1 - epsilon

        return a

    return policy_fn


def q_learning(num_episodes, discount_factor=1, alpha=0.5, epsilon=0.1):

    q = defaultdict(lambda: np.zeros(env.na))
    env = Wing()

    policy = epsilon_greedy_policy(q, epsilon, env.na)
    for i in range(num_episodes):

        env = Wing()
        state = ''.join(str(x) for x in env.state)

        while not env.done:
            action_probs = policy(state)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            next_state, reward, done = env.step(action)

            best_next_action = np.argmax(q[next_state])
            td_target = reward + discount_factor * q[next_state][best_next_action]
            q[state][action] += alpha * td_target

            state = next_state

        if (i+1) % 100 == 0:
            print('{} episodes:'.format(i+1))
            shape_array = draw(state, env.na)
            print(shape_array)

    return q


Q = q_learning(10000)
wing = Wing()

state = ''.join(str(x) for x in wing.state)
print('action_value', epsilon_greedy_policy(Q, 0.1, 5)(state))


