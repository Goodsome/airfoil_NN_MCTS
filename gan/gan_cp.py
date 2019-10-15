import time
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import layers
import matplotlib.pyplot as plt

dic = '/home/xie/Data/xfoil_dl/npy'
cp_path = '{}/cp/cp.npy'.format(dic)

cp = np.load(cp_path)

BUFFER_SIZE = 10000
BATCH_SIZE = 64

train_dataset = tf.data.Dataset.from_tensor_slices(cp).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)


def generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(1024, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Dense(512, ))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Dense(512, ))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Dense(121, activation='tanh'))

    return model


def discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(1024, input_shape=(121,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Dense(512, ))
    model.add(layers.LeakyReLU())

    model.add(layers.Dense(1))

    return model


generator = generator_model()
discriminator = discriminator_model()
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss

    return total_loss

def generator_loss(fake_output):

    return cross_entropy(tf.ones_like(fake_output), fake_output)


generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

model_save_dir = '/home/xie/Data/tf_model/DCGAN'
pic_save_dir = '/home/xie/Data/pic/dcgan'

EPOCHS = 1000
noise_dim = 100
num_examples_to_generate = 16

seed = tf.random.normal([num_examples_to_generate, noise_dim])

@tf.function
def train_step(cps):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_cps = generator(noise, training=True)

        real_output = discriminator(cps, training=True)
        fake_output = discriminator(generated_cps, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gen_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
    disc_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gen_gradients, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(disc_gradients, discriminator.trainable_variables))


def generate_and_save_images(model, epoch, test_input):

    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.plot(predictions[i])
        plt.axis('off')

    fig.suptitle('{}'.format(epoch // 10))
    fig.savefig('{}/image_at_epoch_{:04d}.png'.format(pic_save_dir, epoch))


def train(dataset, epochs):
    generate_and_save_images(generator, 0, seed)
    for epoch in range(epochs):
        start = time.time()

        for cp_batch in dataset:
            train_step(cp_batch)

        if (epoch+1) % 10 == 0:
            generate_and_save_images(generator, epoch+1, seed)

        print('time for epoch {} is  {} sec'.format(epoch+1, time.time()-start))


# train(train_dataset, EPOCHS)
if __name__ == '__main__':
    train(train_dataset, EPOCHS)
