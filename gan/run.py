from GAN.nn import *


class Pix2Pix(ModelBase):
    def __init__(self, param):
        super().__init__(model_name='pix2pix', param=param)

        self.g = pix2pix_generator(self.c_dim, self.z_dim)
        self.d = pix2pix_discriminator()
        self.g_optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5)
        self.d_optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5)

    @tf.function
    def train_step(self, inputs, target):
        if self.z_dim:
            z = np.random.randn(self.batch_size, self.z_dim).astype(np.float32)
            inp = [inputs, z]
        else:
            inp = inputs

        with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
            g_output = self.g(inp)
            real_output = self.d(target)
            fake_output = self.d(g_output)

            l1 = l1_loss(target, g_output)
            g_loss = generator_loss(fake_output) + l1 * 100
            d_loss = discriminator_loss(real_output, fake_output)

        g_gradients = g_tape.gradient(g_loss, self.g.trainable_variables)
        d_gradients = d_tape.gradient(d_loss, self.d.trainable_variables)

        self.g_optimizer.apply_gradients(zip(g_gradients, self.g.trainable_variables))
        self.d_optimizer.apply_gradients(zip(d_gradients, self.d.trainable_variables))

        return d_loss, g_loss, l1


class Cgan(ModelBase):
    def __init__(self, param):
        super().__init__(model_name='cgan', param=param)

        self.learning_rate = 2e-4
        self.g = cgan_generator(self.c_dim, self.z_dim)
        self.d = cgan_discriminator()
        self.g_optimizer = tf.keras.optimizers.Adam(learning_rate=5 * self.learning_rate, beta_1=0.5)
        self.d_optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate, beta_1=0.5)

    @tf.function
    def train_step(self, inputs, target):
        z = np.random.randn(self.batch_size, self.z_dim).astype(np.float32)
        with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
            g_output = self.g([inputs, z])
            l_1 = l1_loss(target, g_output)

            d_real_logits = self.d(target)
            d_fake_logits = self.d(g_output)

            d_loss = d_loss_fun(d_fake_logits, d_real_logits)
            g_loss = g_loss_fun(d_fake_logits) + l_1 * 50

        d_gradients = d_tape.gradient(d_loss, self.d.trainable_variables)
        g_gradients = g_tape.gradient(g_loss, self.g.trainable_variables)
        self.d_optimizer.apply_gradients(zip(d_gradients, self.d.trainable_variables))
        self.g_optimizer.apply_gradients(zip(g_gradients, self.g.trainable_variables))

        return d_loss, g_loss, l_1


class CVAE(ModelBase):
    def __init__(self, param):
        super().__init__(model_name='cVAE', param=param)

        self.g = pix2pix_generator(self.c_dim, self.z_dim)
        self.d = pix2pix_discriminator()
        self.e = encoder(self.z_dim)
        self.g_optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5)
        self.d_optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5)
        self.e_optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5)

    @tf.function
    def train_step(self, inputs, targets):
        with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape, tf.GradientTape() as e_tape:
            e_output_mu, e_output_log_sigma = self.e(targets)
            e_output_z = e_output_mu + tf.random.normal(shape=tf.shape(e_output_mu)) * tf.exp(e_output_log_sigma)
            g_output = self.g([inputs, e_output_z])
            d_output_real = self.d(inputs)
            d_output_fake = self.d(inputs)

            l1 = l1_loss(targets, g_output)
            kl = kl_loss(e_output_mu, e_output_log_sigma)
            d_loss = discriminator_loss(d_output_real, d_output_fake)
            g_loss = generator_loss(g_output) + l1 * 10
            e_loss = generator_loss(g_output) + l1 * 10 + kl * 10

        g_gradients = g_tape.gradient(g_loss, self.g.trainable_variables)
        d_gradients = d_tape.gradient(d_loss, self.d.trainable_variables)
        e_gradients = e_tape.gradient(e_loss, self.e.trainable_variables)

        self.g_optimizer.apply_gradients(zip(g_gradients, self.g.trainable_variables))
        self.d_optimizer.apply_gradients(zip(d_gradients, self.d.trainable_variables))
        self.e_optimizer.apply_gradients(zip(e_gradients, self.e.trainable_variables))

        return d_loss, g_loss, e_loss


class LRGAN(ModelBase):
    def __init__(self, param):
        super().__init__(model_name='lrgan', param=param)

        self.g = pix2pix_generator(self.c_dim, self.z_dim)
        self.d = pix2pix_discriminator()
        self.e = encoder(self.z_dim)
        self.g_optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5)
        self.d_optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5)
        self.e_optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5)

    @tf.function
    def train_step(self, inputs, targets):
        z = np.random.randn(self.batch_size, self.z_dim).astype(np.float32)
        with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape, tf.GradientTape() as e_tape:
            g_output = self.g([inputs, z])
            d_output_fake = self.d(g_output)
            d_output_real = self.d(targets)
            e_output_mu, e_output_log_sigma = self.e(g_output)
            # e_output_z = e_output_mu + tf.random.normal(shape=tf.shape(e_output_mu)) * tf.exp(e_output_log_sigma)

            l1 = l1_loss(z, e_output_mu)
            d_loss = discriminator_loss(d_output_real, d_output_fake)
            g_loss = generator_loss(d_output_fake) + l1
            e_loss = l1 * 10

        g_gradients = g_tape.gradient(g_loss, self.g.trainable_variables)
        d_gradients = d_tape.gradient(d_loss, self.d.trainable_variables)
        e_gradients = e_tape.gradient(e_loss, self.e.trainable_variables)

        self.g_optimizer.apply_gradients(zip(g_gradients, self.g.trainable_variables))
        self.d_optimizer.apply_gradients(zip(d_gradients, self.d.trainable_variables))
        self.e_optimizer.apply_gradients(zip(e_gradients, self.e.trainable_variables))

        return d_loss, g_loss, e_loss


class BicycleGAN(ModelBase):
    def __init__(self, param):
        super().__init__(model_name='bicycle', param=param)

        self.g = pix2pix_generator(self.c_dim, self.z_dim)
        self.d = pix2pix_discriminator()
        self.e = encoder(self.z_dim)
        self.g_optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5)
        self.d_optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5)
        self.e_optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5)

    @tf.function
    def train_step(self, inputs, targets):
        z = np.random.randn(self.batch_size, self.z_dim).astype(np.float32)
        with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape, tf.GradientTape() as e_tape:
            e_output_mu1, e_output_log_sigma1 = self.e(targets)
            e_output_z1 = e_output_mu1 + tf.random.normal(tf.shape(e_output_mu1)) * tf.exp(e_output_log_sigma1)
            g_output1 = self.g([inputs, e_output_z1])
            d_output_fake1 = self.d(g_output1)
            d_output_real = self.d(targets)

            g_output2 = self.g([inputs, z])
            d_output_fake2 = self.d(g_output2)
            e_output_mu2, e_output_log_sigma2 = self.e(g_output2)
            e_output_z2 = e_output_mu2 + tf.random.normal(tf.shape(e_output_mu2)) * tf.exp(e_output_log_sigma2)

            targets_l1 = l1_loss(targets, g_output1)
            latent_l1 = l1_loss(z, e_output_z2)
            kl = kl_loss(e_output_mu1, e_output_log_sigma1)
            d_loss = discriminator_loss(d_output_real, d_output_fake1) + discriminator_loss(d_output_real,
                                                                                            d_output_fake2)
            g_loss = generator_loss(d_output_fake1) + generator_loss(d_output_fake2) + targets_l1 * 10 + latent_l1
            e_loss = generator_loss(d_output_fake1) + targets_l1 * 10 + kl * 100

        g_gradients = g_tape.gradient(g_loss, self.g.trainable_variables)
        d_gradients = d_tape.gradient(d_loss, self.d.trainable_variables)
        e_gradients = e_tape.gradient(e_loss, self.e.trainable_variables)

        self.g_optimizer.apply_gradients(zip(g_gradients, self.g.trainable_variables))
        self.d_optimizer.apply_gradients(zip(d_gradients, self.d.trainable_variables))
        self.e_optimizer.apply_gradients(zip(e_gradients, self.e.trainable_variables))

        return targets_l1, latent_l1, kl


class Pix2Pix2d(ModelBase):
    def __init__(self, param):
        super().__init__(model_name='pix2pix2d', param=param)

        self.g = generator_2d(self.c_dim, self.z_dim)
        self.d = discriminator_2d()
        self.g_optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5)
        self.d_optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5)

    @tf.function
    def train_step(self, inputs, target):
        if len(inputs.shape) == 2:
            inputs = tf.reshape(inputs, [self.batch_size, 1, 1, -1]) * tf.ones_like(target)
        if self.z_dim:
            z = np.random.randn(self.batch_size, 1, 1, self.z_dim).astype(np.float32) * tf.ones_like(target)
            inp = [inputs, z]
        else:
            inp = inputs

        with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
            g_output = self.g(inp)
            real_output = self.d(target)
            fake_output = self.d(g_output)

            l1 = l1_loss(target, g_output)
            g_loss = generator_loss(fake_output) + l1 * 100
            d_loss = discriminator_loss(real_output, fake_output)

        g_gradients = g_tape.gradient(g_loss, self.g.trainable_variables)
        d_gradients = d_tape.gradient(d_loss, self.d.trainable_variables)

        self.g_optimizer.apply_gradients(zip(g_gradients, self.g.trainable_variables))
        self.d_optimizer.apply_gradients(zip(d_gradients, self.d.trainable_variables))

        return d_loss, g_loss, l1


def main():

    data_dir = '/home/xie/Data/panair_data/20_49'
    param = Parameter(data_dir=data_dir, inputs_name='cp_20_49.npy', outputs_name='z_20_49.npy', x_name='xy_20_49.npy',
                      z_dim=8, one4all=False, epochs=20000)

    my_model = Pix2Pix2d(param)
    my_model.train()


if __name__ == '__main__':
    main()
