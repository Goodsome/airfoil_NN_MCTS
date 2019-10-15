import tensorflow as tf
from tensorflow.python.keras import layers, Sequential, Model, losses, models

import numpy as np
import gym
from gym.spaces import Discrete, Box


def mlp(inputs_shape, sizes, activation=tf.tanh, output_activation=None):
    # Build a feedforward neural network.
    inputs = layers.Input(shape=inputs_shape)
    x = inputs
    for size in sizes[:-1]:
        x = layers.Dense(units=size, activation=activation)(x)
    outputs = layers.Dense(units=sizes[-1], activation=output_activation)(x)

    return Model(inputs=inputs, outputs=outputs)


def train(env_name='CartPole-v0', hidden_sizes=(32, 64, 128, 64, 32), lr=1e-2,
          epochs=50, batch_size=5000, render=False):
    # make environment, check spaces, get obs / act dims
    env = gym.make(env_name)
    assert isinstance(env.observation_space, Box), \
        "This example only works for envs with continuous state spaces."
    assert isinstance(env.action_space, Discrete), \
        "This example only works for envs with discrete action spaces."

    obs_dim = env.observation_space.shape[0]
    n_acts = env.action_space.n

    # make core of policy network
    policy = mlp(inputs_shape=obs_dim, sizes=hidden_sizes + (n_acts,))
    policy_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    # for training policy
    def train_one_epoch():
        # make some empty lists for logging.
        batch_obs = []  # for observations
        batch_acts = []  # for actions
        batch_weights = []  # for R(tau) weighting in policy gradient
        batch_rets = []  # for measuring episode returns
        batch_lens = []  # for measuring episode lengths

        # reset episode-specific variables
        obs = env.reset()  # first obs comes from starting distribution
        done = False  # signal from environment that episode is over
        ep_rews = []  # list for rewards accrued throughout ep

        # render first episode of each epoch
        finished_rendering_this_epoch = False

        # collect experience by acting in the environment with current policy
        while True:

            # rendering
            if (not finished_rendering_this_epoch) and render:
                env.render()

            # save obs
            batch_obs.append(obs.copy())

            # act in the environment
            logits = policy(obs.reshape(1, -1))
            act = int(tf.squeeze(tf.random.categorical(logits=logits, num_samples=1)))

            obs, rew, done, _ = env.step(act)

            # save action, reward
            batch_acts.append(act)
            ep_rews.append(rew)

            if done:
                # if episode is over, record info about episode
                ep_ret, ep_len = sum(ep_rews), len(ep_rews)
                batch_rets.append(ep_ret)
                batch_lens.append(ep_len)

                # the weight for each logprob(a|s) is R(tau)
                batch_weights += [ep_ret] * ep_len

                # reset episode-specific variables
                obs, done, ep_rews = env.reset(), False, []

                # won't render again this epoch
                finished_rendering_this_epoch = True

                # end experience loop if we have enough of it
                if len(batch_obs) > batch_size:
                    break

        # take a single policy gradient update step
        with tf.GradientTape() as policy_tape:
            logits = policy(np.array(batch_obs))
            action_masks = tf.one_hot(np.array(batch_acts), n_acts)
            log_probs = tf.reduce_sum(action_masks * tf.nn.log_softmax(logits), axis=1)
            loss = -tf.reduce_mean(np.array(batch_weights) * log_probs)

        policy_gradients = policy_tape.gradient(loss, policy.trainable_variables)
        policy_optimizer.apply_gradients(zip(policy_gradients, policy.trainable_variables))

        return loss, np.array(batch_rets), np.array(batch_lens)

    # training loop
    for i in range(epochs):
        batch_loss, batch_rets, batch_lens = train_one_epoch()
        print('epoch: %3d \t loss: %.3f \t return: %.3f \t ep_len: %.3f' %
              (i, batch_loss, batch_rets.mean(), batch_lens.mean()))


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', '--env', type=str, default='CartPole-v0')
    parser.add_argument('--render', default=True)
    parser.add_argument('--lr', type=float, default=1e-2)
    args = parser.parse_args()
    print('\nUsing simplest formulation of policy gradient.\n')
    train(env_name=args.env_name, render=args.render, lr=args.lr)


if __name__ == '__main__':
    main()
