from tensorflow.python.keras import models
import matplotlib.pyplot as plt

from GAN.old.utils import *
from GAN.ops import *

state_dim = 10
space_dim = 3
interval = '0_59049'
data_dir = '/home/xie/Data/xfoil/{}_{}/{}'.format(state_dim, space_dim, interval)

cp_path = '{}/cp_1.npy'.format(data_dir)
cl_cd_path = '{}/cl_cd_1.npy'.format(data_dir)
x_path = '{}/x.npy'.format(data_dir)
cp = np.load(cp_path)
cl_cd = np.load(cl_cd_path)
x = np.load(x_path)

buffer_size = cl_cd.shape[0]
cl_cd_dim = cl_cd.shape[1]
z_dim = 8
test_size = 16
train_size = buffer_size - test_size
test_index = np.random.randint(0, buffer_size, size=test_size)

train_cl_cd = np.delete(cl_cd, test_index, axis=0)
train_cp = np.delete(cp, test_index, axis=0)

test_cl_cd = cl_cd[test_index]
test_cp = cp[test_index]

batch_size = 64
epoch = 20

learning_rate = 2e-4


model_name = 'cgan'
if z_dim:
    model_name = '{}_noise_{}u'.format(model_name, z_dim)

result_dir = '{}/{}'.format(data_dir, model_name)
image_dir = '{}/images'.format(result_dir)
model_dir = '{}/models'.format(result_dir)
model_g_path = '{}/generator.h5'.format(model_dir)
model_d_path = '{}/discriminator.h5'.format(model_dir)


loss = {'g': [], 'd': [], 'dis': []}


def check_dir(dirs):
    if not os.path.exists(dirs):
        os.mkdir(dirs)

check_dir(result_dir)
check_dir(model_dir)
check_dir(image_dir)


def make_generator_model(input_dim):
    model = keras.Sequential()
    model.add(layers.Dense(512, input_shape=(input_dim,)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.BatchNormalization())

    model.add(layers.Dense(99, activation='tanh'))

    return model


def make_discriminator_model():
    model = keras.Sequential()
    model.add(layers.Dense(512, input_shape=(101,)))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dense(1024))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dense(512))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dense(1))

    return model


def d_loss_fun(d_fake_logits, d_real_logits):
    d_loss_real = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_real_logits), logits=d_real_logits)
    )
    d_loss_fake = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(d_fake_logits), logits=d_fake_logits)
    )
    total_loss = d_loss_real + d_loss_fake
    return total_loss


def g_loss_fun(logits):
    g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(logits), logits=logits))

    return g_loss

def mae(real, fake):
    return tf.reduce_mean(tf.abs(real - fake))

class CGAN:
    def __init__(self):
        super(CGAN, self).__init__()
        self.batch_size = batch_size
        self.z_dim = z_dim
        self.learning_rate = learning_rate
        self.epochs = epoch
        self.cp = np.load(cp_path)
        self.cl_cd = np.load(cl_cd_path)
        self.size = self.cp.shape[0]
        self.c_dim = self.cl_cd.shape[1]

        self.datasets = tf.data.Dataset.from_tensor_slices(
            (self.cp, self.cl_cd)).shuffle(self.size).batch(batch_size, drop_remainder=True)
        self.g = make_generator_model(input_dim=self.z_dim + self.c_dim)
        self.d = make_discriminator_model()
        self.g_optimizer = tf.keras.optimizers.Adam(learning_rate=5*self.learning_rate, beta_1=0.5)
        self.d_optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate, beta_1=0.5)

        self.min_dis = 1

    @tf.function
    def train_one_step(self, batch_cp, batch_cl_cd):
        z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)
        z_y = tf.concat([z, batch_cl_cd], 1)
        real_cp = tf.concat([batch_cp, batch_cl_cd], 1)
        with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
            fake_cp = self.g(z_y)
            l_1 = mae(batch_cp, fake_cp)
            fake_cp = tf.concat([fake_cp, batch_cl_cd], 1)
            d_fake_logits = self.d(fake_cp)
            d_real_logits = self.d(real_cp)
            d_loss = d_loss_fun(d_fake_logits, d_real_logits)
            g_loss = g_loss_fun(d_fake_logits) + l_1 * 50
        d_gradients = d_tape.gradient(d_loss, self.d.trainable_variables)
        g_gradients = g_tape.gradient(g_loss, self.g.trainable_variables)
        self.d_optimizer.apply_gradients(zip(d_gradients, self.d.trainable_variables))
        self.g_optimizer.apply_gradients(zip(g_gradients, self.g.trainable_variables))

        return d_loss, g_loss, l_1

    def train(self, load=False):

        if load:
            self.load_models()

        step = 0
        while step < 10000:
            for batch_images, batch_labels in self.datasets:
                step += 1
                d_l, g_l, l1_l = self.train_one_step(batch_images, batch_labels)

                if step % 10 == 0:
                    print('step: {}, d_loss: {:.4f}, g_loss:{:.4f}, l1_loss:{:.5f}'.format(
                        step, d_l.numpy(), g_l.numpy(), l1_l.numpy()))
                    loss['g'].append(g_l.numpy())
                    loss['d'].append(d_l.numpy())

                if step % 1000 == 0:
                    dis = self.predict(step)
                    if dis < self.min_dis:
                        self.min_dis = dis
                        print("\n----------Saved model for step {}: -------------\n".format(step))
                        self.g.save(model_g_path)
                        self.d.save(model_d_path)

                if step >= 10000:
                    break

    def load_models(self):
        self.g = models.load_model(model_g_path)
        self.d = models.load_model(model_d_path)

    def predict(self, step, one_cl_cd=False):
        sample_size = 16

        if one_cl_cd:
            sample_z = np.linspace(-1, 1, sample_size).reshape([sample_size, 1]) * np.ones([1, z_dim])
            sample_index = [2] * sample_size
        else:
            sample_z = np.random.uniform(-1., 1., size=(sample_size, z_dim)).astype(np.float32)
            sample_index = np.random.randint(0, self.size, size=sample_size)
        sample_cl_cd = test_cl_cd
        sample_cp = test_cp
        sample_zc = tf.concat([sample_z, sample_cl_cd], 1)

        result = self.g(sample_zc, training=False).numpy()
        dis = np.mean(np.abs(result - sample_cp), axis=1)
        t_dis = np.mean(dis)
        print('distance {}'.format(np.mean(dis)))
        loss['dis'].append(dis)
        image_title = 'step_{:05d},dis={:.6f}'.format(step, t_dis)
        image_path = '{}/step_{:05d}.png'.format(image_dir, step)
        save_cp_difference(sample_size, sample_cp, result, dis, image_title, image_path)

        return t_dis

def save_cp_difference(n, cp_1, cp_2, error, title, path):
    a = int(np.sqrt(n))

    fig = plt.figure()

    for i in range(n):
        plt.subplot(a, a, i+1)
        plt.plot(x, -cp_1[i], 'r-', label='true')
        plt.plot(x, -cp_2[i], 'b-', label='generated')
        plt.title('{:.4f}'.format(error[i]))
        plt.axis('off')

    plt.legend(loc=(1, 5))
    fig.suptitle(title)
    fig.savefig(path)

    plt.close()


def main():
    model = CGAN()
    model.train()
    fig = plt.figure()
    plt.plot(loss['g'], label='g')
    plt.plot(loss['d'], label='d')
    plt.title('cgan')
    plt.legend()
    plt.show()
    fig.savefig('{}/cgan.png'.format(image_dir))

    # np.save('loss/{}_loss'.format(model_name), loss)


if __name__ == '__main__':
    main()
