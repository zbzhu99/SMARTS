import tensorflow as tf


def NeuralNetwork1(name, num_actions, input1_shape):
    filter_num = [32, 32, 64, 64, 128]
    kernel_size = [65, 13, 5, 2, 3]
    pool_size = [4, 2, 2, 2, 1]

    # filter_num = [16, 32, 64]
    # kernel_size = [33, 17, 9]
    # pool_size = [4, 4, 2]

    input1 = tf.keras.layers.Input(shape=input1_shape, dtype=tf.uint8)
    input1_norm = tf.cast(input1, tf.float32) / 255.0
    x_conv = input1_norm
    for ii in range(len(filter_num)):
        x_conv = tf.keras.layers.Conv2D(
            filters=filter_num[ii],
            kernel_size=kernel_size[ii],
            strides=(1, 1),
            padding="valid",
            activation=tf.keras.activations.relu,
            name=f"conv2d_{ii}",
        )(x_conv)
        x_conv = tf.keras.layers.MaxPool2D(
            pool_size=pool_size[ii], name=f"maxpool_{ii}"
        )(x_conv)

    flatten_out = tf.keras.layers.Flatten()(x_conv)

    dense1_out = tf.keras.layers.Dense(
        units=512, activation=tf.keras.activations.relu, name="dense_1"
    )(flatten_out)

    policy = tf.keras.layers.Dense(units=num_actions, name="dense_policy")(dense1_out)
    value = tf.keras.layers.Dense(units=1, name="dense_value")(dense1_out)

    model = tf.keras.Model(inputs=input1, outputs=[policy, value], name=f"NN1_{name}")

    return model


def NeuralNetwork2(name, num_output, input1_shape):
    filter_num = [32, 32, 64, 64]
    kernel_size = [32, 17, 9, 3]
    stride_size = [4, 2, 2, 2]

    input1 = tf.keras.layers.Input(shape=input1_shape, dtype=tf.uint8)
    # Scale and center
    input1_norm = (tf.cast(input1, tf.float32) / 255.0) - 0.5
    x_conv = input1_norm
    for ii in range(len(filter_num)):
        x_conv = tf.keras.layers.Conv2D(
            filters=filter_num[ii],
            kernel_size=kernel_size[ii],
            strides=stride_size[ii],
            padding="valid",
            activation=tf.keras.activations.tanh,
            name=f"conv2d_{ii}",
        )(x_conv)
        # x_conv = tf.keras.layers.MaxPool2D(
        #     pool_size=pool_size[ii], name=f"maxpool_{ii}"
        # )(x_conv)

    flatten_out = tf.keras.layers.Flatten()(x_conv)

    dense1_out = tf.keras.layers.Dense(
        units=64, activation=tf.keras.activations.tanh, name="dense_1"
    )(flatten_out)

    dense2_out = tf.keras.layers.Dense(
        units=64, activation=tf.keras.activations.tanh, name="dense_2"
    )(dense1_out)

    output = tf.keras.layers.Dense(units=num_output, name="output")(dense2_out)

    model = tf.keras.Model(inputs=input1, outputs=output, name=f"NN2_{name}")

    return model
