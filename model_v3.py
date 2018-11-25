import tensorflow as tf

_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5
DEFAULT_DTYPE = tf.float32
CASTABLE_TYPES = (tf.float16,)
ALLOWED_TYPES = (DEFAULT_DTYPE,) + CASTABLE_TYPES
REGULARIZER = tf.contrib.layers.l2_regularizer(scale=0.0001)


# Convenience functions for building the ResNet model.
def batch_norm(inputs, training, data_format):
    """Performs a batch normalization using a standard set of parameters."""
    return tf.layers.batch_normalization(
        inputs=inputs, axis=1 if data_format == 'channels_first' else 3,
        momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
        scale=True, training=training, fused=True,
        beta_regularizer=REGULARIZER,
        gamma_regularizer=REGULARIZER,
    )


def fixed_padding(inputs, kernel_size, data_format):
    """
    Pads the input along the spatial dimensions independently of input size.

    :param inputs: A tensor of size [batch, channels, height_in, width_in] or
                   [batch, height_in, width_in, channels] depending on data_format.
    :param kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
                        Should be a positive integer.
    :param data_format: The input format ('channels_last' or 'channels_first').
    :return: A tensor with the same format as the input with the data either intact
             (if kernel_size == 1) or padded (if kernel_size > 1).
    """
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg

    if data_format == 'channel_first':
        padded_inputs = tf.pad(inputs, [[0, 0], [0, 0],
                                        [pad_beg, pad_end], [pad_beg, pad_end]])

    else:
        padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                        [pad_beg, pad_end], [0, 0]])
    return padded_inputs


def conv2d_fixed_padding(inputs, filters, kernel_size, strides, data_format):
    """Strided 2-D convolution with explicit padding."""
    if strides > 1:
        inputs = fixed_padding(inputs, kernel_size, data_format)

    return tf.layers.conv2d(
        inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
        padding=('SAME' if strides == 1 else 'VALID'), use_bias=False,
        kernel_initializer=tf.variance_scaling_initializer(),
        kernel_regularizer=REGULARIZER,
        bias_regularizer=REGULARIZER,
        data_format=data_format
    )


def _bottleneck_block_v2(inputs, filters, training, projection_shortcut, strides,
                         data_format):
    """
    A single block for ResNet v2, with a bottleneck.

    Similar to _building_block_v2(), except using the "bottleneck" blocks
    :param inputs: A tensor of size [batch, channels, height_in, width_in] or
                   [batch, height_in, width_in, channels].
    :param filters: The number of filters for the convolutions.
    :param training: A Boolean of whether the model is in training or inference
                     mode. Needed for batch normalization.
    :param projection_shortcut: The function to use for projection shortcuts
                                (typically a 1x1 convolution when down sampling the input).
    :param strides: The block's stride. If greater than 1, this block will ultimately
                    down sample the input.
    :param data_format: The input format ('channels_last' of 'channels_first').
    :return: The output tensor of the block; shape should match inputs.
    """
    shortcut = inputs
    inputs = batch_norm(inputs, training, data_format)
    inputs = tf.nn.relu(inputs)

    if projection_shortcut is not None:
        shortcut = projection_shortcut(inputs)

    inputs = conv2d_fixed_padding(
        inputs=inputs, filters=filters, kernel_size=1, strides=1,
        data_format=data_format
    )

    inputs = batch_norm(inputs, training, data_format)
    inputs = tf.nn.relu(inputs)
    inputs = conv2d_fixed_padding(
        inputs=inputs, filters=filters, kernel_size=3, strides=strides,
        data_format=data_format
    )

    inputs = batch_norm(inputs, training, data_format)
    inputs = tf.nn.relu(inputs)
    inputs = conv2d_fixed_padding(
        inputs=inputs, filters=4 * filters, kernel_size=1, strides=1,
        data_format=data_format
    )

    return inputs + shortcut


def block_layer(inputs, filters, bottleneck, block_fn, blocks, strides,
                training, name, data_format):
    """
    Creats one layer of blocks for the ResNet model.

    :param inputs: A tensor of size [batch, channels, height_in, width_in] or
                   [batch, height_in, width_in, channels].
    :param filters: The number of filters for the convolutions.
    :param bottleneck: Is the block created a bottleneck block.
    :param block_fn: The block to use within the model, either 'building_block' or
                     'bottleneck_block'.
    :param blocks: The number of blocks contained in the layer.
    :param strides: The stride to use for the first convolution of the layer. If
                    greater than 1, this layer will ultimately down sample the input.
    :param training: Either True or False, whether we are currently training the model.
    :param name: A string name for the tensor output of the block layer.
    :param data_format: The input format.
    :return: The output tensor of the block layer.
    """

    # Bottleneck blocks end with 4x the number of filters as they start with
    filters_out = filters * 4 if bottleneck else filters

    def projection_shortcut(inputs):
        return conv2d_fixed_padding(
            inputs=inputs, filters=filters_out, kernel_size=1, strides=strides,
            data_format=data_format
        )

    # Only the first block per block_layer uses projection_shortcut and strides
    inputs = block_fn(inputs, filters, training, projection_shortcut, strides,
                      data_format)

    for _ in range(1, blocks):
        inputs = block_fn(inputs, filters, training, None, 1, data_format)

    return tf.identity(inputs, name)


def model(inputs, filters, blocks, training, n, pre):
    inputs = tf.reshape(inputs, [-1, n, n, 1])
    inputs = block_layer(inputs, filters, True, _bottleneck_block_v2, blocks, 1, training, 'pro', 'channels_last')
    inputs = block_layer(inputs, 4, True, _bottleneck_block_v2, 1, 1, training, 'min', 'channels_last')

    inputs = tf.reshape(inputs, [-1, n * n * 4 * 4])
    inputs = tf.layers.dense(inputs, units=1024, activation=tf.nn.relu)
    inputs = tf.layers.batch_normalization(inputs, training=training)

    outputs = tf.layers.dense(inputs, units=pre)
    pro = tf.nn.softmax(outputs)

    value = tf.layers.dense(inputs, units=512, activation=tf.nn.relu)
    value = tf.layers.dense(value, units=pre, activation=tf.nn.sigmoid)

    return outputs, pro, value


def pre_train(y_true, y):
    loss = tf.losses.softmax_cross_entropy(y_true, y)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
    train = optimizer.minimize(loss)

    return train, loss


def mcts_train(z, v, pi, p, i, ones):
    v_i = tf.matmul(v * i, ones)
    l2_loss = tf.losses.get_regularization_loss()
    pi_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=pi, logits=p)
    loss = tf.losses.mean_squared_error(z, v_i) + pi_loss + l2_loss
    optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
    train = optimizer.minimize(loss)

    return train, loss
