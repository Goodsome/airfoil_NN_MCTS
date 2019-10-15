from my_rl.core import *


def vpg(epochs=50, size=500, steps_per_epoch=500):
    # env = gym.make('MountainCarContinuous-v0')
    env = gym.make('CartPole-v0')
    obs_space = env.observation_space
    act_space = env.action_space
    obs_dim = env.observation_space.shape
    act_dim = act_space.shape if isinstance(act_space, Box) else act_space.n
    x = np.isscalar(act_dim)

    actor = mlp(obs_dim, act_dim[0], sizes=(32, 64, 32,))
    critic = mlp(obs_dim, 1, sizes=(32, 64, 32,))
    actor_optimizer = tf.keras.optimizers.Adam(learning_rate=3e-4)
    critic_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    vpg_buffer = Buffer(obs_dim, act_dim, size=size)

    obs, rew = env.reset(), 0
    for i in range(epochs):
        for t in range(size):
            # env.render()

            act, log_p_pi = categorical(actor, obs.reshape((1, -1)))
            v = critic(obs.reshape(1, -1))

            vpg_buffer.store(obs, act, rew, v, log_p)
            print(act)
            print(vpg_buffer.act)
            exit()

            obs, rew, done, _ = env.step(act)
            if done or t == size - 1:
                last_val = rew if done else critic(obs.reshape(1, -1))
                vpg_buffer.finish_path(last_val)
                obs, rew = env.reset(), 0

        vpg_buffer.index, vpg_buffer.start_index = 0, 0
        adv_mean = vpg_buffer.adv.mean()
        adv_std = np.sqrt(np.sum(np.square(vpg_buffer.adv - adv_mean)))
        vpg_buffer.adv = (vpg_buffer.adv - adv_mean) / adv_std
        with tf.GradientTape() as actor_tape, tf.GradientTape() as critic_tape:
            logits = actor(vpg_buffer.obs)
            v = critic(vpg_buffer.obs)
            act_mask = tf.one_hot(vpg_buffer.act, depth=act_dim)
            log_p = tf.reduce_sum(act_mask * tf.nn.log_softmax(logits), axis=1)
            pi_loss = -tf.reduce_mean(log_p * vpg_buffer.adv)
            v_loss = tf.reduce_mean(tf.square(vpg_buffer.ret - tf.squeeze(v)))

        actor_gradient = actor_tape.gradient(pi_loss, actor.trainable_variables)
        critic_gradient = critic_tape.gradient(v_loss, critic.trainable_variables)
        actor_optimizer.apply_gradients(zip(actor_gradient, actor.trainable_variables))
        critic_optimizer.apply_gradients(zip(critic_gradient, critic.trainable_variables))


if __name__ == '__main__':
    vpg()
