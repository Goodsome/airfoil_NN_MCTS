import argparse
import datetime

from abc import ABC
from tensorflow import keras
from tensorflow.python.keras import  metrics, layers

from GAN.old.utils import *
from GAN.ops import conv_cond_concat


class Discriminator(keras.Model, ABC):

    def __init__(self):
        super(Discriminator, self).__init__(name='pix2pix_discriminator')
        self.conv_1 = layers.Conv2D(64, 4, 2, padding='same', input_shape=(28, 28, 1))
        self.lrelu_1 = layers.LeakyReLU(alpha=0.2)
        self.conv_2 = layers.Conv2D(128, 4, 2, padding='same')
        self.bn_1 = layers.BatchNormalization()
        self.lrelu_2 = layers.LeakyReLU(alpha=0.2)
        self.flat = layers.Flatten()
        self.dense_1 = layers.Dense(1024)
        self.bn_2 = layers.BatchNormalization()
        self.lrelu_3 = layers.LeakyReLU(alpha=0.2)
        self.dense_2 = layers.Dense(1)

    def call(self, inputs, **kwargs):
        x = self.conv_1(inputs)
        x = self.lrelu_1(x)
        x = self.conv_2(x)
        x = self.bn_1(x)
        x = self.lrelu_2(x)
        x = self.flat(x)
        x = self.dense_1(x)
        x = self.bn_2(x)
        x = self.lrelu_3(x)
        logits = self.dense_2(x)
        out = keras.activations.sigmoid(logits)

        return out, logits, x


class Generator(keras.Model, ABC):

    def __init__(self):
        super().__init__(name='pix2pix_generator')
        self.dense_1 = layers.Dense(1024, input_shape=(74,))
        self.bn_1 = layers.BatchNormalization()
        self.relu_1 = layers.ReLU()
        self.dense_2 = layers.Dense(128*7*7)
        self.bn_2 = layers.BatchNormalization()
        self.relu_2 = layers.ReLU()
        self.reshape = layers.Reshape((7, 7, 128))
        self.convT_1 = layers.Conv2DTranspose(64, 4, 2, padding='same')
        self.bn_3 = layers.BatchNormalization()
        self.relu_3 = layers.ReLU()
        self.convT_2 = layers.Conv2DTranspose(1, 4, 2, padding='same', activation='sigmoid')

    def call(self, inputs, **kwargs):
        x = self.dense_1(inputs)
        x = self.bn_1(x)
        x = self.relu_1(x)
        x = self.dense_2(x)
        x = self.bn_2(x)
        x = self.relu_2(x)
        x = self.reshape(x)
        x = self.convT_1(x)
        x = self.bn_3(x)
        x = self.relu_3(x)
        x = self.convT_2(x)

        return x


class Classifier(keras.Model, ABC):

    def __init__(self, y_dim):
        super().__init__(name='classifier')
        self.y_dim = y_dim
        self.dense_1 = layers.Dense(64, input_shape=(1024,))
        self.bn_1 = layers.BatchNormalization()
        self.lrelu = layers.LeakyReLU(alpha=0.2)
        self.dense_2 = layers.Dense(self.y_dim)

    def call(self, inputs, **kwargs):
        x = self.dense_1(inputs)
        x = self.bn_1(x)
        x = self.lrelu(x)
        logits = self.dense_2(x)
        out = layers.Softmax(logits)

        return out, logits


class InfoGAN:

    def __init__(self, args):
        self.datasets_path = args.datasets_path
        self.image_dir = args.image_dir
        self.model_name = args.gan_type
        self.batch_size = args.batch_size
        self.SUPERVISED = args.supervised
        self.z_dim = args.z_dim
        self.y_dim = 12
        self.len_discrete_code = 10
        self.len_continuous_code = 2
        self.checkpoint_dir = os.path.join(args.checkpoint_dir, self.model_name)
        self.datasets_name = args.datasets
        self.log_dir = args.log_dir
        self.learning_rate = args.lr
        self.epochs = args.epoch

        self.sample_z = tf.random.uniform(shape=(self.batch_size, self.z_dim),
                                          minval=-1, maxval=1, dtype=tf.float32)
        self.datasets = load_mnist_data(self.batch_size, self.datasets_name, path=self.datasets_path)

        self.g = Generator()
        self.d = Discriminator()
        self.c = Classifier(y_dim=self.y_dim)
        self.g_optimizer = keras.optimizers.Adam(lr=5*self.learning_rate, beta_1=0.5)
        self.d_optimizer = keras.optimizers.Adam(lr=self.learning_rate, beta_1=0.5)
        self.q_optimizer = keras.optimizers.Adam(lr=5*self.learning_rate, beta_1=0.5)
        self.g_loss_metric = metrics.Mean('g_loss', dtype=tf.float32)
        self.d_loss_metric = metrics.Mean('d_loss', dtype=tf.float32)
        self.q_loss_metric = metrics.Mean('q_loss', dtype=tf.float32)
        self.checkpoint = tf.train.Checkpoint(step=tf.Variable(0),
                                              generator_optimizer=self.g_optimizer,
                                              discriminator_optimizer=self.g_optimizer,
                                              classifier_optimizer=self.q_optimizer,
                                              generator=self.g,
                                              discriminator=self.d,
                                              classifier=self.c)
        self.manager = tf.train.CheckpointManager(self.checkpoint, self.checkpoint_dir, max_to_keep=3)

    @property
    def model_dir(self):
        return '{}_{}_[}'.format(self.model_name, self.datasets_name,
                                 self.batch_size, self.z_dim)

    @tf.function
    def train_one_step(self, batch_labels, batch_images):
        noises = tf.random.uniform(shape=(self.batch_size, self.z_dim), minval=-1, maxval=1)
        code = tf.random.uniform(shape=(self.batch_size, self.len_continuous_code), minval=-1, maxval=1)
        batch_codes = tf.concat((batch_labels, code), axis=1)
        batch_z = tf.concat([noises, batch_codes], axis=1)
        real_images = conv_cond_concat(batch_images, batch_codes)

        with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape, tf.GradientTape() as q_tape:
            fake_imgs = self.g(batch_z)
            # fake_imgs = conv_cond_concat(fake_imgs, batch_codes)
            d_fake, d_fake_logits, input4classifier_fake = self.d(fake_imgs)
            d_real, d_real_logits, _ = self.d(batch_images)
            d_loss = d_loss_fun(d_fake_logits, d_real_logits)
            g_loss = g_loss_fun(d_fake_logits)
            _, code_logits_fake = self.c(input4classifier_fake)
            disc_code_est = code_logits_fake[:, :self.len_discrete_code]
            disc_code_tg =  batch_codes[:, :self.len_discrete_code]
            cont_code_est = code_logits_fake[:, self.len_discrete_code:]
            cont_code_tg = batch_codes[:, self.len_discrete_code:]
            q_loss = q_loss_fun(disc_code_est, disc_code_tg, cont_code_est, cont_code_tg)

        d_gradients = d_tape.gradient(d_loss, self.d.trainable_variables)
        g_gradients = g_tape.gradient(g_loss, self.g.trainable_variables)
        q_var = self.g.trainable_variables + self.d.trainable_variables[:-2] + self.c.trainable_variables
        q_gradients = q_tape.gradient(q_loss, q_var)

        self.d_optimizer.apply_gradients(zip(d_gradients, self.d.trainable_variables))
        self.g_optimizer.apply_gradients(zip(g_gradients, self.g.trainable_variables))
        self.q_optimizer.apply_gradients(zip(q_gradients, q_var))

        self.d_loss_metric(d_loss)
        self.g_loss_metric(g_loss)
        self.q_loss_metric(q_loss)

    def train(self, load=False):

        sample_label = 2 * tf.ones(shape=(self.batch_size,), dtype=tf.int32)
        sample_label = tf.one_hot(sample_label, depth=10)
        current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        train_log_dir = os.path.join(self.log_dir, self.model_name, current_time)
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)

        if load:
            self.load_ckpt()
            ckpt_step = int(self.checkpoint.step)
            start_epoch = (ckpt_step * self.batch_size) // 60000
        else:
            start_epoch = 0

        for epoch in range(start_epoch, self.epochs):
            for batch_images, batch_labels in self.datasets:
                if not self.SUPERVISED:
                    batch_labels = np.random.multinomial(1,
                                                         self.len_discrete_code *
                                                         [float(1.0 / self.len_discrete_code)],
                                                         size=[self.batch_size]).astype(np.float32)

                continuous_code = tf.random.uniform(shape=[self.batch_size, self.len_continuous_code],
                                                    minval=-1, maxval=1)
                test_codes = tf.concat([sample_label, continuous_code], 1)
                self.train_one_step(batch_labels, batch_images)
                self.checkpoint.step.assign_add(1)
                step = int(self.checkpoint.step)

                if step % 50 == 0:
                    print('step: {}, d_loss: {:.4f}, g_loss: {:.4f}, q_loss: {:.4f}'.format(
                        step, self.d_loss_metric.result(), self.g_loss_metric.result(), self.q_loss_metric.result()
                    ))
                    manifold_h = int(np.floor(np.sqrt(self.batch_size)))
                    manifold_w = int(np.floor(np.sqrt(self.batch_size)))
                    batch_z_to_display = tf.concat([self.sample_z, test_codes[:self.batch_size, :]], 1)
                    result_to_display = self.g(batch_z_to_display)
                    save_images(result_to_display[:manifold_h * manifold_w, :, :,  :],
                                [manifold_h, manifold_w],
                                '{}/{}_train_{:02d}_{:04d}.png'.format(self.image_dir, self.model_name, epoch, step))

                    with train_summary_writer.as_default():
                        tf.summary.scalar('g_loss', self.g_loss_metric.result(), step=step)
                        tf.summary.scalar('d_loss', self.d_loss_metric.result(), step=step)
                        tf.summary.scalar('q_loss', self.q_loss_metric.result(), step=step)

                if step % 400 == 0:
                    save_path = self.manager.save()
                    print('\n---------------Saved checkpoint for step {}: {}---------------\n'.format(step, save_path))
                    self.g_loss_metric.reset_states()
                    self.d_loss_metric.reset_states()
                    self.q_loss_metric.reset_states()

    def load_ckpt(self):
        self.checkpoint.restore(self.manager.latest_checkpoint)
        if self.manager.latest_checkpoint:
            print('restore model from checkpoint: {}'.format(self.manager.latest_checkpoint))
            return True
        else:
            print('Initializing from scratch.')
            return False


def d_loss_fun(d_fake_logits, d_real_logits):
    d_loss_real = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_real_logits),logits=d_real_logits)
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


def q_loss_fun(disc_code_est, disc_code_tg, cont_code_est, cont_code_tg):
    q_disc_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(labels=disc_code_tg, logits=disc_code_est)
    )
    q_cont_loss = tf.reduce_mean(
        tf.reduce_mean(tf.square(cont_code_tg - cont_code_est), axis=1)
    )
    q_loss = q_disc_loss + q_cont_loss
    return q_loss


def parse_args():

    desc = 'Tensorflow implementation of infoGAN collection'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--supervised', type=bool, default=False)
    parser.add_argument('--datasets_path', type=str, default='/home/xie/Data/mnist/mnist.npz')
    parser.add_argument('--image_dir', type=str, default='/home/xie/Data/infoGAN/image')
    parser.add_argument('--gan_type', type=str, default='infoGAN')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--z_dim', type=int, default=62)
    parser.add_argument('--checkpoint_dir', type=str, default='/home/xie/Data/infoGAN/checkpoint')
    parser.add_argument('--datasets', type=str, default='mnist')
    parser.add_argument('--log_dir', type=str, default='/home/xie/Data/infoGAN/logs')
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--epoch', type=int, default=20)

    return check_args(parser.parse_args())


def check_args(args):

    check_folder(args.datasets_path)
    check_folder(args.image_dir)
    check_folder(args.checkpoint_dir)
    check_folder(args.log_dir)
    assert args.epoch >= 1
    assert args.batch_size >= 1
    assert args.z_dim >= 1

    return args


def main():
    args = parse_args()
    if args is None:
        exit()
    model = InfoGAN(args)
    model.train(load=False)


def test():
    args = parse_args()
    model = InfoGAN(args)
    model.load_ckpt()
    save_img(model)

def save_img(model):

    batch = 64
    # sample_label = 4 * tf.ones(shape=(batch,), dtype=tf.int32)
    # sample_label = tf.one_hot(sample_label, depth=10)
    sample_label = tf.one_hot(tf.concat([tf.range(8)] * 8, axis=0), depth=10)
    # continuous_code = tf.random.uniform(shape=[model.batch_size, model.len_continuous_code], minval=-1, maxval=1)
    continuous_code = tf.reshape(tf.linspace(-1., 1., 8), [8, 1])
    continuous_code = tf.concat([continuous_code] * 8, axis=1)
    continuous_code_1 = tf.reshape(continuous_code, (64, 1))
    continuous_code_2 = continuous_code_1
    test_codes = tf.concat([sample_label, continuous_code_1, continuous_code_2], 1)
    manifold_h = int(np.floor(np.sqrt(model.batch_size)))
    manifold_w = int(np.floor(np.sqrt(model.batch_size)))
    batch_z_to_display = tf.concat([model.sample_z, test_codes[:model.batch_size, :]], 1)
    result_to_display = model.g(batch_z_to_display)
    save_images(result_to_display[:manifold_h * manifold_w, :, :,  :],
                [manifold_h, manifold_w],
                '{}/{}_test.png'.format(model.image_dir, model.model_name))

if __name__ == '__main__':
    test()
