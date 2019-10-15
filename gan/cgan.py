import argparse
import datetime
import time

from tensorflow.python.keras import metrics

from GAN.old.utils import *
from GAN.ops import *


class CGAN:
    def __init__(self, args):
        super(CGAN, self).__init__()
        self.datasets_dir = args.datasets_dir
        self.model_name = args.gan_type
        self.batch_size = args.batch_size
        self.z_dim = args.z_dim
        self.y_dim = 10
        self.checkpoint_dir = check_folder(os.path.join(args.checkpoint_dir, self.model_name))
        self.result_dir = args.result_dir
        self.datasets_name = args.datasets
        self.log_dir = args.log_dir
        self.learning_rate = args.lr
        self.epochs = args.epoch
        self.datasets = load_mnist_data(datasets=self.datasets_name, batch_size=args.batch_size, path=self.datasets_dir)
        self.g = self.make_generator_model(is_training=True)
        self.d = self.make_discriminator_model(is_training=True)
        self.g_optimizer = tf.keras.optimizers.Adam(learning_rate=5*self.learning_rate, beta_1=0.5)
        self.d_optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate, beta_1=0.5)
        self.g_loss_metric = metrics.Mean('g_loss', dtype=tf.float32)
        self.d_loss_metric = metrics.Mean('d_loss', dtype=tf.float32)
        self.checkpoint = tf.train.Checkpoint(step=tf.Variable(0),
                                              generator_optimizer=self.g_optimizer,
                                              discriminator_optimizer=self.d_optimizer,
                                              generator=self.g,
                                              discriminator=self.d)
        self.manager = tf.train.CheckpointManager(self.checkpoint, self.checkpoint_dir, max_to_keep=3)

        self.train_summary_writer = None
        self.could_load = None

    def make_generator_model(self, is_training):
        model = keras.Sequential()
        model.add(DenseLayer(1024))
        model.add(BatchNorm(is_training=is_training))
        model.add(layers.ReLU())
        model.add(DenseLayer(128*7*7))
        model.add(BatchNorm(is_training=is_training))
        model.add(layers.ReLU())
        model.add(layers.Reshape((7, 7, 128)))
        model.add(UpConv2D(64, 4, 2))
        model.add(BatchNorm(is_training=is_training))
        model.add(layers.ReLU())
        model.add(UpConv2D(1, 4, 2))
        model.add(Sigmoid())

        return model

    def make_discriminator_model(self, is_training):
        model = keras.Sequential()
        model.add(Conv2D(64, 4, 2))
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(Conv2D(128, 4, 2))
        model.add(BatchNorm(is_training=is_training))
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(layers.Flatten())
        model.add(DenseLayer(1024))
        model.add(BatchNorm(is_training=is_training))
        model.add(layers.LeakyReLU(alpha=0.3))
        model.add(DenseLayer(1))

        return model

    @property
    def model_dir(self):
        return '{}_{}_{}_{}'.format(
            self.model_name, self.datasets_name,
            self.batch_size, self.z_dim
        )

    @staticmethod
    def d_loss_fun(d_fake_logits, d_real_logits):
        d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_real_logits), logits=d_real_logits)
        )
        d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(d_fake_logits), logits=d_fake_logits)
        )
        total_loss = d_loss_real + d_loss_fake
        return total_loss

    @staticmethod
    def g_loss_fun(logits):
        g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(logits), logits=logits))

        return g_loss

    @tf.function
    def train_one_step(self, batch_images, batch_labels):
        z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)
        z_y = tf.concat([z, batch_labels], 1)
        real_images = conv_cond_concat(batch_images, batch_labels)
        with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
            fake_imgs = self.g(z_y, training=True)
            fake_imgs = conv_cond_concat(fake_imgs, batch_labels)
            d_fake_logits = self.d(fake_imgs, training=True)
            d_real_logits = self.d(real_images, training=True)
            d_loss = self.d_loss_fun(d_fake_logits, d_real_logits)
            g_loss = self.g_loss_fun(d_fake_logits)
        d_gradients = d_tape.gradient(d_loss, self.d.trainable_variables)
        g_gradients = g_tape.gradient(g_loss, self.g.trainable_variables)
        self.d_optimizer.apply_gradients(zip(d_gradients, self.d.trainable_variables))
        self.g_optimizer.apply_gradients(zip(g_gradients, self.g.trainable_variables))

        self.d_loss_metric(d_loss)
        self.g_loss_metric(g_loss)

    def train(self, load=False):
        sample_size = 100
        sample_label = [x // 10 for x in range(sample_size)]

        sample_label = tf.one_hot(sample_label, depth=10)
        current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        train_log_dir = os.path.join(self.log_dir, self.model_name, current_time)
        self.train_summary_writer = tf.summary.create_file_writer(train_log_dir)

        if load:
            self.could_load = self.load_ckpt()
            ckpt_step = int(self.checkpoint.step)
            start_epoch = int((ckpt_step*self.batch_size)//60000)
        else:
            start_epoch = 0


        for epoch in range(start_epoch, self.epochs):
            for batch_images, batch_labels in self.datasets:
                self.train_one_step(batch_images, batch_labels)
                self.checkpoint.step.assign_add(1)
                step = int(self.checkpoint.step)

                if step % 50 == 0:
                    print('step: {}, d_loss: {:.4f}, g_loss:{:.4f}'.format(
                        step, self.d_loss_metric.result(), self.g_loss_metric.result()))
                    manifold_h = int(np.floor(np.sqrt(sample_size)))
                    manifold_w = int(np.floor(np.sqrt(sample_size)))
                    sample_z = np.random.uniform(-1., 1., size=(sample_size, self.z_dim)).astype(np.float32)
                    sample_z_y = tf.concat([sample_z, sample_label], 1)
                    result_to_display = self.g(sample_z_y, training=False)
                    save_images(result_to_display[:manifold_h * manifold_w, :, :, :],
                                [manifold_h, manifold_w], check_folder(
                                    self.result_dir + '/' + self.model_dir) + '/' + self.model_name + '_train_{:02d}_{:04d}.png'.format(
                                    epoch, int(step)))

                    with self.train_summary_writer.as_default():
                        tf.summary.scalar('g_loss', self.g_loss_metric.result(), step=step)
                        tf.summary.scalar('d_loss', self.d_loss_metric.result(), step=step)

                if step % 400 == 0:
                    save_path = self.manager.save()

                    print("\n----------Saved checkpoint for step {}: {}-------------\n".format(step, save_path))

                    self.g_loss_metric.reset_states()
                    self.d_loss_metric.reset_states()

    def load_ckpt(self):
        self.checkpoint.restore(self.manager.latest_checkpoint)
        if self.manager.latest_checkpoint:
            print('restore model from checkpoint: {}'.format(self.manager.latest_checkpoint))
            return True

        else:
            print('Initializing from scratch.')
            return False


def parse_args():
    desc = "Tensorflow implementation of GAN collections"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--datasets_dir', type=str, default='/home/xie/Data/mnist/mnist.npz')
    parser.add_argument('--gan_type', type=str, default='CGAN')
    parser.add_argument('--datasets', type=str, default='mnist')
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--epoch', type=int, default=20, help='The number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=64, help='The size of batch')
    parser.add_argument('--z_dim', type=int, default=62, help='Dimension of noise vector')
    parser.add_argument('--checkpoint_dir', type=str, default='/home/xie/Data/CGAN/checkpoint',
                        help='Directory name to save the checkpoints')
    parser.add_argument('--result_dir', type=str, default='/home/xie/Data/CGAN/results',
                        help='Directory name to save the generated images')
    parser.add_argument('--log_dir', type=str, default='/home/xie/Data/CGAN/logs',
                        help='Directory name to save training logs')
    return check_args(parser.parse_args())


"""checking arguments"""
def check_args(args):
    # --checkpoint_dir
    check_folder(args.checkpoint_dir)

    # --result_dir
    check_folder(args.result_dir)

    # --result_dir
    check_folder(args.log_dir)

    # --epoch
    assert args.epoch >= 1, 'number of epochs must be larger than or equal to one'

    # --batch_size
    assert args.batch_size >= 1, 'batch size must be larger than or equal to one'

    # --z_dim
    assert args.z_dim >= 1, 'dimension of noise vector must be larger than or equal to one'

    return args


def main():
    args = parse_args()
    if args is None:
        exit()
    model = CGAN(args)
    # model.train()
    for x, y in model.datasets:
        start = time.time()
        model.train_one_step(x, y)
        model.checkpoint.step.assign_add(1)
        step = int(model.checkpoint.step)

        print(step, time.time() - start)

if __name__ ==  '__main__':
    main()
