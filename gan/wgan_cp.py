from tensorflow.python.keras import models
import matplotlib.pyplot as plt

from GAN.old.utils import *
from GAN.ops import *

buffer_size = 100000
batch_size = 64
epoch = 20
z_dim = 40


data_dir = '/home/xie/Data/cp_data'
cp_path = '{}/{}/cp.npy'.format(data_dir, buffer_size)
cl_cd_path = '{}/{}/cl_cd.npy'.format(data_dir, buffer_size)

result_dir = '/home/xie/Data/CGAN/cp'
model_name = 'model_wgan'
model_dir = '{}/{}'.format(result_dir, model_name)
model_g_path = '{}/pix2pix_generator/'.format(model_dir)
model_d_path = '{}/pix2pix_discriminator/'.format(model_dir)
image_dir = '{}/{}/images'.format(result_dir, model_name)

loss = {'g': [], 'd': [], 'dis': []}


def check_dir(dirs):
    if not os.path.exists(dirs):
        os.mkdir(dirs)


check_dir(model_dir)
check_dir(model_g_path)
check_dir(model_d_path)
check_dir(image_dir)


def make_generator_model(input_dim):
    model = keras.Sequential()
    model.add(layers.Dense(1024, input_shape=(input_dim,), activation='tanh'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(512, activation='tanh'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(256, activation='tanh'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(99, activation='tanh'))

    return model


def make_discriminator_model(input_dim):
    model = keras.Sequential()
    model.add(layers.Dense(512, input_shape=(input_dim,)))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dense(1024))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dense(512))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dense(1))

    return model


def d_loss_fun(d_fake_logits, d_real_logits):
    total_loss = tf.reduce_mean(d_fake_logits) - tf.reduce_mean(d_real_logits)
    return total_loss


def g_loss_fun(logits):
    g_loss = -tf.reduce_mean(logits)
    return g_loss


class CGAN:
    def __init__(self):
        super(CGAN, self).__init__()
        self.batch_size = batch_size
        self.z_dim = z_dim
        self.epochs = epoch
        cp_0 = np.load(cp_path)
        cp_0[cp_0 > 0] /= 1.2
        cp_0[cp_0 < 0] /= 3
        cl_cd_0 = np.load(cl_cd_path)
        self.cp = cp_0[cl_cd_0.T[0] > 0.1]
        self.cl_cd = cl_cd_0[cl_cd_0.T[0] > 0.1]
        self.size = self.cp.shape[0]
        self.cp_dim = self.cp.shape[1]
        self.c_dim = self.cl_cd.shape[1]

        self.datasets = tf.data.Dataset.from_tensor_slices(
            (self.cp, self.cl_cd)).shuffle(self.size).batch(batch_size, drop_remainder=True)
        self.g = make_generator_model(input_dim=self.z_dim+self.c_dim)
        self.d = make_discriminator_model(input_dim=self.cp_dim + self.c_dim)
        self.g_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0, beta_2=0.9)
        self.d_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0, beta_2=0.9)

        self.lam = 10

    @tf.function
    def train_d(self, batch_cp, batch_cl_cd):
        z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)
        z_y = tf.concat([z, batch_cl_cd], 1)
        real_cp = tf.concat([batch_cp, batch_cl_cd], 1)
        with tf.GradientTape() as d_tape:
            fake_cp = self.g(z_y)
            fake_cp = tf.concat([fake_cp, batch_cl_cd], 1)
            d_fake_logits = self.d(fake_cp)
            d_real_logits = self.d(real_cp)
            d_loss = d_loss_fun(d_fake_logits, d_real_logits)
            g_loss = g_loss_fun(d_fake_logits)

            with tf.GradientTape() as penalty_tape:
                alpha = tf.random.uniform([self.batch_size], 0., 1., dtype=tf.float32)
                alpha = tf.reshape(alpha, (-1, 1))
                interpolated = alpha * real_cp + (1 - alpha) * fake_cp
                penalty_tape.watch(interpolated)
                inter_logits = self.d(interpolated)
                gradient = penalty_tape.gradient(inter_logits, interpolated)
                grad_l2 = tf.sqrt(tf.reduce_sum(tf.square(gradient), axis=1))
                gradient_penalty = tf.reduce_mean((grad_l2 - 1) ** 2)
                d_loss_gp = d_loss + self.lam * gradient_penalty

        d_gradients = d_tape.gradient(d_loss_gp, self.d.trainable_variables)
        self.d_optimizer.apply_gradients(zip(d_gradients, self.d.trainable_variables))

        return d_loss, g_loss

    @tf.function
    def train_g(self, batch_cl_cd):

        z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)
        z_y = tf.concat([z, batch_cl_cd], 1)
        with tf.GradientTape() as g_tape:
            fake_cp = self.g(z_y)
            fake_cp = tf.concat([fake_cp, batch_cl_cd], 1)
            d_fake_logits = self.d(fake_cp)
            g_loss = g_loss_fun(d_fake_logits)

        g_gradients = g_tape.gradient(g_loss, self.g.trainable_variables)
        self.g_optimizer.apply_gradients(zip(g_gradients, self.g.trainable_variables))

        return g_loss

    def train(self, load=False):
        sample_size = 64

        sample_z = np.random.uniform(-1., 1., size=(sample_size, self.z_dim)).astype(np.float32)
        sample_index = np.random.randint(0, self.size, size=sample_size)
        sample_cl_cd = self.cl_cd[sample_index]
        sample_cp = self.cp[sample_index]

        sample_zc = tf.concat([sample_z, sample_cl_cd], 1)
        if load:
            self.load_models()

        step = 0
        for _ in range(self.epochs):
            for batch_images, batch_labels in self.datasets:
                step += 1

                d_l, g_l = self.train_d(batch_images, batch_labels)
                if step % 1 == 0:
                    g_l = self.train_g(batch_labels)

                if step % 10 == 0:
                    print('step: {}, d_loss: {:.8f}, g_loss:{:.8f}'.format(
                        step, d_l.numpy(), g_l.numpy()))
                    loss['g'].append(g_l.numpy())
                    loss['d'].append(d_l.numpy())

                if step % 100 == 0:
                    result = self.g(sample_zc, training=False)
                    dis = np.mean(np.square(result.numpy() - sample_cp))
                    print('distance {}'.format(dis))
                    loss['dis'].append(dis)
                    image_title = 'step_{:05d}_dis_{:.6}'.format(step, dis)
                    image_path = '{}/step_{:05d}.png'.format(image_dir, step)
                    save_cp_difference(sample_size, result, sample_cp, image_title, image_path)

        print("\n----------Saved model for step {}: -------------\n".format(step))
        tf.saved_model.save(self.g, model_g_path)
        tf.saved_model.save(self.d, model_d_path)

    def load_models(self):
        self.g = models.load_model(model_g_path)
        self.d = models.load_model(model_d_path)


def save_cp_difference(n, cp_1, cp_2, title, path):
    a = int(np.sqrt(n))

    fig = plt.figure(figsize=(a, a))

    for i in range(n):
        plt.subplot(a, a, i+1)
        plt.plot(cp_1[i])
        plt.plot(cp_2[i])
        plt.axis('off')

    fig.suptitle(title)
    fig.savefig(path)

    plt.close()


def main():
    tf.random.set_seed(2)
    np.random.seed(2)
    model = CGAN()
    model.train()
    fig = plt.figure()
    plt.plot(loss['g'], label='g')
    plt.plot(loss['d'], label='d')
    plt.title('wgan')
    plt.legend()
    plt.show()
    fig.savefig('{}/wgan.png'.format(image_dir))

    np.save('loss/{}_loss'.format(model_name), loss)


if __name__ == '__main__':
    main()
