from GAN.nn import *


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
z_dim = 50
test_size = 16
train_size = buffer_size - test_size
test_index = np.random.randint(0, buffer_size, size=test_size)

train_cl_cd = np.delete(cl_cd, test_index, axis=0)
train_cp = np.delete(cp, test_index, axis=0)

test_cl_cd = cl_cd[test_index]
test_cp = cp[test_index]


model_name = 'pix2pix'
if z_dim:
    model_name = '{}_noise_{}u'.format(model_name, z_dim)

result_dir = '{}/{}'.format(data_dir, model_name)
image_dir = '{}/images'.format(result_dir)
model_dir = '{}/models'.format(result_dir)
model_g_path = '{}/pix2pix_generator.h5'.format(model_dir)
model_d_path = '{}/pix2pix_discriminator.h5'.format(model_dir)


def check_dir(dirs):
    if not os.path.exists(dirs):
        os.mkdir(dirs)


check_dir(result_dir)
check_dir(model_dir)
check_dir(image_dir)


class Pix2Pix:
    def __init__(self):
        self.batch_size = 64
        self.step = 0
        self.terminate = 1e4

        self.train_dataset = tf.data.Dataset.from_tensor_slices((train_cl_cd, train_cp)).shuffle(train_size).batch(self.batch_size, drop_remainder=True)
        self.test_dataset = tf.data.Dataset.from_tensor_slices((test_cl_cd, test_cp))
        self.g = pix2pix_generator(cl_cd_dim, z_dim)
        self.d = pix2pix_discriminator()
        self.g_optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5)
        self.d_optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5)

    @tf.function
    def train_step(self, inputs, target):
        if z_dim:
            # z = np.random.randn(self.batch_size, z_dim).astype(np.float32)
            z = np.random.uniform(-1, 1, [self.batch_size, z_dim]).astype(np.float32)
            inp = [inputs, z]
        else:
            inp = inputs
        with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
            g_output = self.g(inp, training=True)

            real_output = self.d([inputs, target], training=True)
            fake_output = self.d([inputs, g_output], training=True)

            g_loss, l1_loss = generator_loss(fake_output, g_output, target)
            d_loss = discriminator_loss(real_output, fake_output)

        g_gradients = g_tape.gradient(g_loss, self.g.trainable_variables)
        d_gradients = d_tape.gradient(d_loss, self.d.trainable_variables)

        self.g_optimizer.apply_gradients(zip(g_gradients, self.g.trainable_variables))
        self.d_optimizer.apply_gradients(zip(d_gradients, self.d.trainable_variables))

        return d_loss, g_loss, l1_loss


    def test(self):
        if z_dim:
            # test_z = np.random.randn(test_size, z_dim).astype(np.float32)
            test_z = np.random.uniform(-1, 1, [test_size, z_dim]).astype(np.float32)
            inp = [test_cl_cd, test_z]
        else:
            inp = test_cl_cd
        prediction = self.g(inp, training=True)
        dis = np.mean(np.abs(prediction - test_cp), axis=1)
        total_dis = np.mean(dis)

        fig = plt.figure(figsize=(test_size, test_size))
        a = int(np.sqrt(test_size))
        for i in range(test_size):
            plt.subplot(a, a, i+1)
            plt.plot(x, -prediction[i], 'r-', label='generated')
            plt.plot(x, -test_cp[i], 'b-', label='true')
            plt.title('mae={:.6f}'.format(dis[i]))
            plt.axis('off')

        plt.legend(loc=(1, 5))
        fig.suptitle('step={}, total mse={:.6f}'.format(self.step, total_dis))
        fig.savefig('{}/step_{:05d}'.format(image_dir, self.step))

        plt.close()

        return total_dis


    def train(self):

        self.step = 0
        mean_time = 0
        start = time.time()
        while self.step < self.terminate:

            for batch_clcd, batch_cp in self.train_dataset:
                d_loss, g_loss, l1_loss = self.train_step(batch_clcd, batch_cp)

                if self.step % 100 == 0:
                    dis = self.test()
                    print('  step: {}, test mse={:.5f}'.format(self.step, dis))

                mean_time *= self.step
                mean_time += time.time() - start
                self.step += 1
                mean_time /= self.step
                rest_time = mean_time * (self.terminate - self.step)
                hours = rest_time // 3600
                mins = rest_time % 3600 // 60
                ticks = int(rest_time % 60)
                start = time.time()
                print('\r{}/{} d loss: {:.4f}, g loss: {:.4f}, l1: {:.4f}, rest time: {}h {}m {}s'.
                      format(self.step, self.terminate, d_loss, g_loss, l1_loss, hours, mins, ticks), end='')

                if self.step >= self.terminate:
                    break

        self.save_models()

    def save_models(self):
        self.g.save(model_g_path)
        self.d.save(model_d_path)

    def load_models(self):
        self.g = models.load_model(model_g_path)
        self.d = models.load_model(model_d_path)


def main():
    model = Pix2Pix()
    model.train()


if __name__ == '__main__':
    main()
