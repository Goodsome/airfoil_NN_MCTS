import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.keras import layers, Sequential, Model, losses, models
from tensorflow.python import random_normal_initializer

from xfoil.create_data import cal_cl_cd
from pypanair.postprocess.agps_converter import write_vtk
from utilities import Timing

OUTPUT_SIZE = 99
LAMBDA = 100


def check_dir(*dirs):
    for i in dirs:
        if not os.path.exists(i):
            os.mkdir(i)

def downsample(units, input_shape=None, apply_batchnorm=True, layer_type='dense'):

    initializer = random_normal_initializer(0., 0.02)

    seq = Sequential()
    if layer_type == 'dense':
        seq.add(layers.Dense(units, input_shape=[input_shape,], kernel_initializer=initializer, use_bias=False))
    elif layer_type == 'conv':
        seq.add(layers.Conv2D(filters=units, kernel_size=3, strides=(2, 2), padding='same',
                              input_shape=input_shape,
                              kernel_initializer=initializer, use_bias=False))
    else:
        raise ValueError('wrong layer type!')
    if apply_batchnorm:
        seq.add(layers.BatchNormalization())

    seq.add(layers.LeakyReLU())
    return seq


def upsample(units, input_shape=None, apply_dropout=False, layer_type='dense', output_padding=(1, 1)):
    initializer = random_normal_initializer(0., 0.02)

    seq = Sequential()
    if layer_type == 'dense':
        seq.add(layers.Dense(units, input_shape=[input_shape,], kernel_initializer=initializer, use_bias=False))
    elif layer_type == 'conv':
        seq.add(layers.Conv2DTranspose(filters=units, kernel_size=3, strides=(2, 2), padding='same',
                                       input_shape=input_shape,
                                       kernel_initializer=initializer, use_bias=False, output_padding=output_padding))
    else:
        raise ValueError('wrong layer_type!')
    seq.add(layers.BatchNormalization())
    if apply_dropout:
        seq.add(layers.Dropout(0.5))
    seq.add(layers.ReLU())

    return seq


def pix2pix_generator(c_dim, z_dim=None):
    down_stack = [
        downsample(units=1024, input_shape=c_dim + z_dim, apply_batchnorm=False),
        downsample(units=512, input_shape=1024),
        downsample(units=256, input_shape=512),
        downsample(units=128, input_shape=256),
        downsample(units=64, input_shape=128),
        downsample(units=64, input_shape=64),
    ]

    up_stack = [
        upsample(units=64, input_shape=64, apply_dropout=False),
        upsample(units=128, input_shape=128, apply_dropout=False),
        upsample(units=256, input_shape=256),
        upsample(units=512, input_shape=512),
        upsample(units=1024, input_shape=1024),
    ]
    initializer = random_normal_initializer(0., 0.02)
    last = layers.Dense(OUTPUT_SIZE, kernel_initializer=initializer, activation='tanh')

    inputs = layers.Input(shape=[c_dim])
    if z_dim:
        z = layers.Input(shape=[z_dim])
        x = layers.concatenate([inputs, z])
        inp = [inputs, z]
    else:
        x = inputs
        inp = inputs

    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = layers.concatenate([x, skip])

    x = last(x)

    return Model(inputs=inp, outputs=x)


def pix2pix_discriminator():

    initializer = random_normal_initializer(0., 0.02)

    inputs = layers.Input(shape=[99], name='cp')

    x = downsample(64, 99, False)(inputs)
    x = downsample(128, 64)(x)
    x = downsample(256, 128)(x)
    x = downsample(512, 256)(x)
    last = layers.Dense(1, kernel_initializer=initializer)(x)

    return Model(inputs=inputs, outputs=last)


loss_object = losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss(real_output, fake_output):
    real_loss = loss_object(tf.ones_like(real_output), real_output)
    fake_loss = loss_object(tf.zeros_like(fake_output), fake_output)

    return real_loss + fake_loss


def generator_loss(fake_output):
    gan_loss = loss_object(tf.ones_like(fake_output), fake_output)

    return gan_loss


def cgan_generator(c_dim, z_dim):

    c = layers.Input(shape=[c_dim])
    z = layers.Input(shape=[z_dim])
    x = layers.concatenate([c, z])

    x = layers.Dense(512)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(99, activation='tanh')(x)

    return Model(inputs=[c, z], outputs=x)


def cgan_discriminator():
    inp = layers.Input(shape=[2], name='cl_cd')
    tar = layers.Input(shape=[99], name='cp')
    x = layers.concatenate([inp, tar])

    x = layers.Dense(512)(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dense(1024)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dense(512)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dense(1)(x)

    return Model(inputs=[inp, tar], outputs=x)


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

def l1_loss(real, fake):
    return tf.reduce_mean(tf.abs(real - fake))


def kl_loss(mu, log_sigma):
    return 0.5 * tf.reduce_mean(-1 - log_sigma + mu ** 2 + tf.exp(log_sigma))


def encoder(z_dim):
    inputs = layers.Input(shape=[99], dtype=tf.float32)

    x = layers.Dense(512)(inputs)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dense(1024)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dense(512)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    mu = layers.Dense(z_dim)(x)
    log_sigma = layers.Dense(z_dim)(x)

    return Model(inputs=inputs, outputs=[mu, log_sigma])


def generator_2d(inputs_dim, z_dim):
    down_stack = [
        downsample(32, input_shape=[20, 49, inputs_dim + z_dim], apply_batchnorm=False, layer_type='conv'),
        downsample(64, input_shape=[10, 25, 32], layer_type='conv'),
        downsample(128, input_shape=[5, 13, 64], layer_type='conv'),
        downsample(256, input_shape=[3, 7, 128], layer_type='conv'),
        downsample(256, input_shape=[2, 4, 256], layer_type='conv'),
    ]

    up_stack = [
        upsample(256, input_shape=[1, 2, 256], apply_dropout=False, layer_type='conv', output_padding=(1, 1)),
        upsample(128, input_shape=[2, 4, 512], layer_type='conv', output_padding=(0, 0)),
        upsample(64, input_shape=[3, 7, 256], layer_type='conv', output_padding=(0, 0)),
        upsample(32, input_shape=[5, 13, 128], layer_type='conv', output_padding=(1, 0)),
    ]
    initializer = random_normal_initializer(0., 0.02)
    last = layers.Conv2DTranspose(1, kernel_size=3, strides=2, padding='same', output_padding=(1, 0),
                                  kernel_initializer=initializer, activation='tanh')

    inputs = layers.Input(shape=[20, 49, inputs_dim])
    if z_dim:
        z = layers.Input(shape=[20, 49, z_dim])
        x = layers.concatenate([inputs, z])
        inp = [inputs, z]
    else:
        x = inputs
        inp = inputs

    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = layers.concatenate([x, skip])

    x = last(x)

    return Model(inputs=inp, outputs=x)


def discriminator_2d():
    initializer = random_normal_initializer(0., 0.02)
    inputs = layers.Input(shape=[20, 49, 1])

    x = downsample(32, input_shape=[20, 49, 1], layer_type='conv')(inputs)
    x = downsample(64, input_shape=[10, 25, 32], layer_type='conv')(x)
    x = downsample(128, input_shape=[5, 13, 64], layer_type='conv')(x)
    x = layers.Conv2D(256, kernel_size=3, strides=1, kernel_initializer=initializer, use_bias=False, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(1, kernel_size=3, strides=1, kernel_initializer=initializer, padding='same')(x)

    return Model(inputs, x)


class ModelBase:
    def __init__(self, model_name, param):

        self.step = 0
        self.epochs = param.epochs
        self.batch_size = param.batch_size
        self.test_size = param.test_size
        self.data_dir = param.data_dir
        self.z_dim = param.z_dim
        self.c_dim = param.c_dim
        self.one4all = param.one4all
        self.train_dataset = param.train_dataset
        self.test_dataset = param.test_dataset
        self.x = param.x
        self.test_y = param.test_y

        self.g = None
        self.d = None
        self.e = None
        self.cp2y = None
        self.name = model_name
        if self.z_dim:
            self.name_dir = '{}_noise_{}'.format(self.name, self.z_dim)
        else:
            self.name_dir = self.name
        result_dir = '{}/{}'.format(param.data_dir, self.name_dir)
        self.image_dir = '{}/images'.format(result_dir)
        self.model_dir = '{}/models'.format(result_dir)
        self.g_path = '{}/generator.h5'.format(self.model_dir)
        self.d_path = '{}/discriminator.h5'.format(self.model_dir)
        self.e_path = '{}/encode.h5'.format(self.model_dir)
        check_dir(result_dir, self.image_dir, self.model_dir)

    @tf.function
    def train_step(self, inputs, targets):
        pass

    def test(self, save_pic=False, predict_y=False):

        a = int(np.sqrt(self.test_size))
        s = 1.6
        vtk_index = np.array(list(range(1, 50)) * 20).reshape([20, -1, 1])
        vtk_dir = '{}/{}'.format(self.image_dir, self.step)
        check_dir(vtk_dir)
        for test_inputs, test_outputs in self.test_dataset:

            if test_inputs.ndim == 2:
                test_inputs = tf.reshape(test_inputs, [self.test_size, 1, 1, -1]) * tf.ones_like(test_outputs)

            index = np.random.randint(self.test_size)
            if self.one4all:
                test_inputs = test_inputs[index] * tf.ones_like(test_outputs)
                test_outputs = test_outputs[index] * tf.ones_like(test_outputs)

            if self.z_dim:
                test_z = np.random.randn(self.test_size, 1, 1, self.z_dim).astype(np.float32) * tf.ones_like(test_outputs)
                inp = [test_inputs, test_z]
            else:
                inp = test_inputs

            prediction = self.g(inp)
            loss = prediction - test_outputs
            dis = np.mean(np.abs(prediction - test_outputs).reshape(self.test_size, -1), axis=1)
            total_dis = np.mean(dis)

            vtk_data = []
            vtk_name = '{}'.format(vtk_dir)
            for i in range(self.test_size):
                data = np.concatenate([vtk_index, self.x, test_outputs[i] / 10, loss[i] / 10], axis=-1)
                data[:, :, 1] += i % 4 * s
                data[:, :, 2] += i // 4 * s
                vtk_data.append(data)

            write_vtk(outputname=vtk_name, data=vtk_data)
            if save_pic:
                fig = plt.figure(figsize=(self.test_size, self.test_size))
                for i in range(self.test_size):
                    plt.subplot(a, a, i+1)
                    plt.plot(self.x, -prediction[i], 'r-', label='generated')
                    plt.plot(self.x, -test_outputs[i], 'b-', label='true')
                    plt.ylim(-1, 1)
                    plt.title('mae={:.6f}'.format(dis[i]), fontsize=22)

                plt.legend(loc=(1, 5))
                fig.suptitle('step={}, total mae={:.6f}'.format(self.step, total_dis), fontsize=24)
                fig.savefig('{}/step_{:05d}'.format(self.image_dir, self.step))
                plt.close()

                if predict_y:
                    test_y = self.test_y.copy()
                    test_y = test_y[index] * np.ones([self.test_size, 1])
                    pred_y = self.cp2y(prediction).numpy()
                    pred_y[:, [0, -1]] = 0
                    cl_cd = []
                    for i in range(self.test_size):
                        for j in range(10):
                            if pred_y[i][j] > 0:
                                pred_y[i][j] = 0.5 * (pred_y[i][j+1] + pred_y[i][j-1])
                            if pred_y[i][-j-1] < 0:
                                pred_y[i][-j-1] = 0.5 * (pred_y[i][j+1] + pred_y[i][j-1])
                        xy = np.concatenate([self.x, pred_y[i]/10]).reshape(2, -1).T

                        cl_cd.append(cal_cl_cd(xy))

                    fig_y = plt.figure(figsize=(self.test_size, self.test_size))
                    for i in range(self.test_size):
                        plt.subplot(a, a, i+1)
                        plt.plot(self.x, pred_y[i]/10, 'r-', label='generated')
                        plt.plot(self.x, test_y[i]/10, 'b-', label='true')
                        plt.ylim(-0.2, 0.2)
                        plt.title('cl={:.4f}, cd={:.4f}'.format(cl_cd[i][0], cl_cd[i][1]), fontsize=18)

                    plt.legend(loc=(1, 5))
                    fig_y.suptitle('target: cl={:.4f}, cd={:.4f}'.format(test_inputs[0][0], test_inputs[0][1]/10), fontsize=24)
                    fig_y.savefig('{}/y_step_{:05d}'.format(self.image_dir, self.step))
                    plt.close()

            return total_dis

    def train(self, save_pic=False, predict_y=False):

        t = Timing(self.epochs)
        while self.step < self.epochs:

            for batch_inputs, batch_outputs in self.train_dataset:
                d_loss, g_loss, l1 = self.train_step(batch_inputs, batch_outputs)
                log = 'd loss: {:.4f}, g loss: {:.4f}, l1: {:.4f}'.format(d_loss, g_loss, l1)
                t.out(log=log)
                self.step += 1

                if self.step % 1000 == 0:
                    dis = self.test(save_pic, predict_y)
                    print('  step: {}, test mae={:.5f}'.format(self.step, dis))

                if self.step >= self.epochs:
                    break

        self.save_models()

    def save_models(self):
        self.g.save(self.g_path)
        self.d.save(self.d_path)
        if self.e:
            self.e.save(self.e_path)

    def load_models(self):
        self.g = models.load_model(self.g_path, compile=False)
        self.d = models.load_model(self.d_path, compile=False)
        if self.e:
            self.e = models.load_model(self.e_path, compile=False)


class Parameter:
    def __init__(self, data_dir, inputs_name, outputs_name, x_name, z_dim, one4all, epochs):
        self.data_dir = data_dir
        self.z_dim = z_dim
        self.c_dim = 0
        self.one4all = one4all
        self.epochs = epochs
        self.inputs_name = inputs_name
        self.outputs_name = outputs_name
        self.x_name = x_name
        self.test_size = 16
        self.batch_size = 64
        self.train_dataset, self.test_dataset, self.x = self.load_data()
        self.test_y = None

    def load_data(self):
        inputs_path = '{}/{}'.format(self.data_dir, self.inputs_name)
        outputs_path = '{}/{}'.format(self.data_dir, self.outputs_name)
        x_path = '{}/{}'.format(self.data_dir, self.x_name)
        inputs = np.load(inputs_path).astype(np.float32)
        outputs = np.load(outputs_path).astype(np.float32)
        x = np.load(x_path)

        buffer_size = inputs.shape[0]
        self.c_dim = inputs.shape[-1]
        train_size = buffer_size - self.test_size
        test_index = np.random.randint(0, buffer_size, size=self.test_size)
        train_inputs = np.delete(inputs, test_index, axis=0)
        train_outputs = np.delete(outputs, test_index, axis=0)
        test_inputs = inputs[test_index]
        test_outputs = outputs[test_index]

        train_dataset = tf.data.Dataset.from_tensor_slices((train_inputs, train_outputs)) \
            .shuffle(train_size).batch(self.batch_size, drop_remainder=True)
        test_dataset = tf.data.Dataset.from_tensor_slices((test_inputs, test_outputs)).batch(self.test_size)

        return train_dataset, test_dataset, x


if __name__ == '__main__':
    pass
