import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.python.keras import metrics, layers, models

from GAN.old.utils import *

buffer_size = 100000
batch_size = 64
epoch = 50
z_dim = 62

learning_rate = 2e-4

data_dir = '/home/xie/Data/cp_data'
cp_path = '{}/{}/cp.npy'.format(data_dir, buffer_size)
cl_cd_path = '{}/{}/cl_cd.npy'.format(data_dir, buffer_size)

result_dir = '/home/xie/Data/infoGAN/cp'
model_g_path = '{}/models/pix2pix_generator'.format(result_dir)
model_d_path = '{}/models/pix2pix_discriminator'.format(result_dir)
model_c_path = '{}/models/classifier'.format(result_dir)
image_dir = '{}/images/{}'.format(result_dir, buffer_size)


def generator_model():

    model = keras.Sequential()
    model.add(layers.Dense(1024, input_shape=(66,)))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.Dense(512))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.Dense(256))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.Dense(99, activation='tanh'))

    return model


def discriminator_and_classifier_model(c_dim):

    inputs = keras.Input(shape=(99,))
    x = layers.Dense(1024, input_shape=(64,))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dense(512)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dense(256)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    d_out = layers.Dense(1)(x)

    x = layers.Dense(256)(x)
    x = layers.LeakyReLU()(x)
    q_out = layers.Dense(c_dim)(x)

    return keras.Model(inputs=inputs, outputs=d_out), keras.Model(inputs=inputs, outputs=q_out)


class InfoGAN:

    def __init__(self):
        self.batch_size = batch_size
        self.z_dim = z_dim
        self.y_dim = 4
        self.len_continuous_code = 4

        self.learning_rate = learning_rate
        self.epochs = epoch
        self.cp = np.load(cp_path)
        self.cl_cd = np.load(cl_cd_path)
        self.datasets = tf.data.Dataset.from_tensor_slices(
            (self.cp, self.cl_cd)).shuffle(buffer_size).batch(batch_size, drop_remainder=True)

        self.g = generator_model()
        self.d, self.c = discriminator_and_classifier_model(self.y_dim)
        self.g_optimizer = keras.optimizers.Adam(lr=5 * self.learning_rate, beta_1=0.5)
        self.d_optimizer = keras.optimizers.Adam(lr=self.learning_rate, beta_1=0.5)
        self.q_optimizer = keras.optimizers.Adam(lr=5 * self.learning_rate, beta_1=0.5)
        self.g_loss_metric = metrics.Mean('g_loss', dtype=tf.float32)
        self.d_loss_metric = metrics.Mean('d_loss', dtype=tf.float32)
        self.q_loss_metric = metrics.Mean('q_loss', dtype=tf.float32)

    @tf.function
    def train_one_step(self, batch_cp):
        noises = tf.random.uniform(shape=(self.batch_size, self.z_dim), minval=-1, maxval=1)
        codes = tf.random.uniform(shape=(self.batch_size, self.len_continuous_code), minval=-1, maxval=1)
        batch_z = tf.concat([noises, codes], axis=1)

        with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape, tf.GradientTape() as q_tape:
            fake_cp = self.g(batch_z)
            d_fake_logits = self.d(fake_cp)
            d_real_logits = self.d(batch_cp)
            code_logits_fake = self.c(fake_cp)

            cont_code_est = code_logits_fake
            cont_code_tg = codes

            d_loss = d_loss_fun(d_fake_logits, d_real_logits)
            g_loss = g_loss_fun(d_fake_logits)
            q_loss = q_loss_fun(cont_code_est, cont_code_tg)

        q_var = self.g.trainable_variables + self.c.trainable_variables
        d_gradients = d_tape.gradient(d_loss, self.d.trainable_variables)
        g_gradients = g_tape.gradient(g_loss, self.g.trainable_variables)
        q_gradients = q_tape.gradient(q_loss, q_var)

        self.d_optimizer.apply_gradients(zip(d_gradients, self.d.trainable_variables))
        self.g_optimizer.apply_gradients(zip(g_gradients, self.g.trainable_variables))
        self.q_optimizer.apply_gradients(zip(q_gradients, q_var))

        self.d_loss_metric(d_loss)
        self.g_loss_metric(g_loss)
        self.q_loss_metric(q_loss)

    def train(self, load=False):

        sample_batch = 64
        sample_z = tf.random.uniform(shape=(sample_batch, self.z_dim), minval=-1, maxval=1, dtype=tf.float32)
        sample_codes_1 = np.linspace(-1, 1, num=sample_batch).reshape(-1, 1)
        sample_codes_2 = np.ones(sample_batch).reshape(-1, 1)
        sample_codes_3 = np.ones(sample_batch).reshape(-1, 1)
        sample_codes_4 = np.ones(sample_batch).reshape(-1, 1)
        sample_codes = np.concatenate((sample_codes_1, sample_codes_2, sample_codes_3, sample_codes_4), axis=1)

        if load:
            self.load_models()

        step = 0
        for _ in range(self.epochs):
            for batch_images, _ in self.datasets:

                self.train_one_step(batch_images)
                step += 1

                if step % 50 == 0:
                    print('step: {}, d_loss: {:.4f}, g_loss: {:.4f}, q_loss: {:.4f}'.format(
                        step, self.d_loss_metric.result(), self.g_loss_metric.result(), self.q_loss_metric.result()
                    ))
                    batch_z_to_display = tf.concat([sample_z, sample_codes], axis=1)
                    result_to_display = self.g(batch_z_to_display)

                    image_title = 'step_{:05d}'.format(step)
                    save_img(sample_batch, result_to_display, image_title)

                if step % 400 == 0:
                    self.save_models()

                    print('\n---------------Saved checkpoint for step {}: ---------------\n'.format(step))
                    self.g_loss_metric.reset_states()
                    self.d_loss_metric.reset_states()
                    self.q_loss_metric.reset_states()

    def save_models(self):
        tf.saved_model.save(self.g, model_g_path)
        tf.saved_model.save(self.d, model_d_path)
        tf.saved_model.save(self.c, model_c_path)

    def load_models(self):
        self.g = models.load_model(model_g_path)
        self.d = models.load_model(model_d_path)
        self.c = models.load_model(model_c_path)


def d_loss_fun(d_fake_logits, d_real_logits):
    d_loss_real = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_real_logits), logits=d_real_logits)
    )
    d_loss_fake = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(d_fake_logits), logits=d_fake_logits)
    )
    total_loss = d_loss_fake + d_loss_real
    return total_loss


def g_loss_fun(logits):
    g_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(logits), logits=logits)
    )
    return g_loss


def q_loss_fun(cont_code_est, cont_code_tg):
    q_cont_loss = tf.reduce_mean(
        tf.reduce_mean(tf.square(cont_code_tg - cont_code_est), axis=1)
    )
    q_loss = q_cont_loss
    return q_loss


def save_img(n, cp, title):
    a = int(np.sqrt(n))

    fig = plt.figure(figsize=(a, a))

    for i in range(n):
        plt.subplot(a, a, i + 1)
        plt.plot(cp[i])
        plt.axis('off')

    fig.suptitle(title)
    save_path = '{}/{}.png'.format(image_dir, title)
    fig.savefig(save_path)

    plt.close()


def main():
    model = InfoGAN()
    model.load_models()

    sample_batch = 64
    sample_z = tf.random.uniform(shape=(sample_batch, model.z_dim), minval=-1, maxval=1, dtype=tf.float32)
    sample_codes_1 = np.linspace(-1, 1, num=sample_batch).reshape(-1, 1)
    sample_codes_2 = np.zeros(sample_batch).reshape(-1, 1)
    sample_codes_3 = np.zeros(sample_batch).reshape(-1, 1)
    sample_codes_4 = np.zeros(sample_batch).reshape(-1, 1)
    sample_codes = np.concatenate((sample_codes_1, sample_codes_2, sample_codes_3, sample_codes_4), axis=1)

    batch_z_to_display = tf.concat([sample_z, sample_codes], axis=1)
    result_to_display = model.g(batch_z_to_display)

    image_title = 'turn_code_1_0'
    save_img(sample_batch, result_to_display, image_title)


if __name__ == '__main__':
    main()
