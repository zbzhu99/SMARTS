import os

# Set pythonhashseed
os.environ["PYTHONHASHSEED"] = "0"
# Silence the logs of TF
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# The below is necessary for starting Numpy generated random numbers
# in a well-defined initial state.
import numpy as np

np.random.seed(123)

# The below is necessary for starting core Python generated random numbers
# in a well-defined state.
import random as python_random

python_random.seed(123)

# The below set_seed() will make random number generation
# in the TensorFlow backend have a well-defined initial state.
# For further details, see:
# https://www.tensorflow.org/api_docs/python/tf/random/set_seed
import tensorflow as tf

tf.random.set_seed(123)

# --------------------------------------------------------------------------

import absl.logging
import tensorflow_probability as tfp

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Suppress warning
absl.logging.set_verbosity(absl.logging.ERROR)


def NeuralNetwork(name, num_actions, input1_shape, input2_shape):
    filter_num = [32, 32, 64, 64, 128]
    kernel_size = [65, 13, 5, 2, 3]
    pool_size = [4, 2, 2, 2, 1]

    # filter_num = [16, 32, 64]
    # kernel_size = [33, 17, 9]
    # pool_size = [4, 4, 2]

    input1 = tf.keras.layers.Input(shape=input1_shape, dtype=tf.uint8)
    input2 = tf.keras.layers.Input(shape=input2_shape, dtype=tf.float32)
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
        units=16, activation=tf.keras.activations.relu, name="dense_1"
    )(input2)

    merged_out = tf.keras.layers.concatenate([flatten_out, dense1_out], axis=1)

    dense2_out = tf.keras.layers.Dense(
        units=512, activation=tf.keras.activations.relu, name="dense_2"
    )(merged_out)

    policy = tf.keras.layers.Dense(units=num_actions, name="dense_policy")(dense2_out)
    value = tf.keras.layers.Dense(units=1, name="dense_value")(dense2_out)

    model = tf.keras.Model(
        inputs=[input1, input2], outputs=[policy, value], name=f"AutoDrive_{name}"
    )

    return model


class PPO(object):
    def __init__(self, name, config, seed):
        self.name = name
        self.config = config
        self.seed = seed
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=config["model_para"]["initial_lr_" + self.name]
        )

        # Model
        self.model = None
        if self.config["model_para"]["model_initial"]:  # Start from existing model
            print("[INFO] PPO existing model.")
            self.model = _load(self.config["model_para"]["model_" + self.name])
        else:  # Start from new model
            print("[INFO] PPO new model.")
            self.model = NeuralNetwork(
                self.name,
                self.config["model_para"]["action_dim"],
                self.config["model_para"]["observation1_dim"],
                self.config["model_para"]["observation2_dim"],
            )
        # Path for newly trained model
        self.model_path = Path(self.config["model_para"]["model_path"]).joinpath(
            f"{name}_{datetime.now().strftime('%Y_%m_%d_%H_%M')}"
        )
        # Model summary
        self.model.summary()

        # Tensorboard
        path = Path(self.config["model_para"]["tensorboard_path"]).joinpath(
            f"{name}_{datetime.now().strftime('%Y_%m_%d_%H_%M')}"
        )
        self.tb = tf.summary.create_file_writer(str(path))

    def save(self, version: int):
        save_path = self.model_path / str(version)
        tf.keras.models.save_model(
            model=self.model,
            filepath=save_path,
        )

    def act(self, obs, train: bool):
        actions = {}
        action_samples = {}
        values = {}

        ordered_obs = _dict_to_ordered_list(obs)

        # NOTE: Order of items drawn from dict may affect reproducibility of this
        # function due to order of sampling by `actions_dist_t.sample()`.
        for vehicle, state in ordered_obs:
            if (
                self.name in vehicle
            ):  # <---- VERY IMPORTANT DO NOT REMOVE @@ !! @@ !! @@ !!
                images, scalars = zip(
                    *(map(lambda x: (x["image"], x["scalar"]), [state]))
                )
                stacked_images = tf.stack(images, axis=0)
                stacked_scalars = tf.stack(scalars, axis=0)
                actions_t, values_t = self.model.predict(
                    [stacked_images, stacked_scalars]
                )
                actions[vehicle] = tf.squeeze(actions_t, axis=0)
                values[vehicle] = tf.squeeze(values_t, axis=0)

                if train:
                    actions_dist_t = tfp.distributions.Categorical(logits=actions_t)
                    action_samples[vehicle] = tf.squeeze(
                        actions_dist_t.sample(1, seed=self.seed), axis=0
                    )
                else:
                    action_samples[vehicle] = tf.math.argmax(actions_t, axis=1)

        return actions, action_samples, values

    def write_to_tb(self, records):
        with self.tb.as_default():
            for name, value, step in records:
                tf.summary.scalar(name, value, step)


def _dict_to_ordered_list(dic: Dict[str, Any]) -> List[Tuple[str, Any]]:
    li = [tuple(x) for x in dic.items()]
    li.sort(key=lambda tup: tup[0])  # Sort in-place

    return li


def _load(model_path):
    return tf.keras.models.load_model(
        model_path,
        compile=False,
    )


def train_model(
    model: tf.keras.Model,
    optimizer: tf.keras.optimizers,
    action_inds,
    old_probs,
    states: List[Dict[str, np.ndarray]],
    advantages: np.ndarray,
    discounted_rewards: np.ndarray,
    ent_discount_val: float,
    clip_value: float,
    critic_loss_weight: float,
    grad_batch=64,
):
    images, scalars = zip(*(map(lambda x: (x["image"], x["scalar"]), states)))
    stacked_image = tf.stack(images, axis=0)
    stacked_scalar = tf.stack(scalars, axis=0)

    traj_len = stacked_image.shape[0]
    assert traj_len == stacked_scalar.shape[0]
    for ind in range(0, traj_len, grad_batch):
        image_chunk = stacked_image[ind : ind + grad_batch]
        scalar_chunk = stacked_scalar[ind : ind + grad_batch]

        with tf.GradientTape() as tape:
            policy_logits, values = model([image_chunk, scalar_chunk])
            act_loss = actor_loss(
                advantages, old_probs, action_inds, policy_logits, clip_value
            )
            cri_loss = critic_loss(discounted_rewards, values, critic_loss_weight)
            ent_loss = entropy_loss(policy_logits, ent_discount_val)
            tot_loss = act_loss + cri_loss + ent_loss

        grads = tape.gradient(tot_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return tot_loss, act_loss, cri_loss, ent_loss


# Clipped objective term, to be maximized
# @tf.function(experimental_relax_shapes=True)
def actor_loss(advantages, old_probs, action_inds, policy_logits, clip_value):
    probs = tf.nn.softmax(policy_logits)
    new_probs = tf.gather_nd(probs, action_inds)
    ratio = new_probs / old_probs  # Ratio is always positive

    policy_loss = -tf.reduce_mean(  # -Expectation
        tf.math.minimum(
            ratio * advantages,
            tf.clip_by_value(ratio, 1.0 - clip_value, 1.0 + clip_value) * advantages,
        )
    )
    return policy_loss


# Entropy term to encourage exploration, to be maximized
# @tf.function(experimental_relax_shapes=True)
def entropy_loss(policy_logits, ent_discount_val):
    probs = tf.nn.softmax(policy_logits)
    entropy_loss = -tf.reduce_mean(
        tf.keras.losses.categorical_crossentropy(probs, probs)
    )
    return entropy_loss * ent_discount_val


# Error term on value estimation, to be minimized
# @tf.function(experimental_relax_shapes=True)
def critic_loss(discounted_rewards, value_est, critic_loss_weight):
    return (
        tf.reduce_mean(
            tf.keras.losses.mean_squared_error(discounted_rewards, value_est)
        )
        * critic_loss_weight
    )
