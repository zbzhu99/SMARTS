import os
from examples.gameOfTag.types import Mode

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

from datetime import datetime
from pathlib import Path

# Suppress warning
absl.logging.set_verbosity(absl.logging.ERROR)


def NeuralNetwork(name, num_output, input1_shape, input2_shape):
    # filter_num = [32, 32, 64, 64, 128]
    # kernel_size = [65, 13, 5, 2, 3]
    # pool_size = [4, 2, 2, 2, 1]
    # stride_size = [1, 1, 1, 1, 1]

    filter_num = [32, 32, 64, 64]
    kernel_size = [32, 17, 9, 3]
    stride_size = [4, 2, 2, 2]

    # filter_num = [16, 32, 64]
    # kernel_size = [33, 17, 9]
    # pool_size = [4, 4, 2]

    input1 = tf.keras.layers.Input(shape=input1_shape, dtype=tf.uint8)
    input2 = tf.keras.layers.Input(shape=input2_shape, dtype=tf.float32)
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

    model = tf.keras.Model(
        inputs=[input1, input2], outputs=output, name=f"AutoDrive_{name}"
    )

    return model


class RL:
    def __init__(self):
        pass

    def act(self):
        raise NotImplementedError

    def save(self):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError


class PPOKeras(RL):
    def __init__(self, name, config, agent_ids, seed):
        super(PPOKeras, self).__init__()

        self._name = name
        self._seed = seed
        self._agent_ids = agent_ids
        self.actor_optimizer = tf.keras.optimizers.Adam(
            learning_rate=config["model_para"][self._name + "_actor_lr"]
        )
        self.critic_optimizer = tf.keras.optimizers.Adam(
            learning_rate=config["model_para"][self._name + "_critic_lr"]
        )

        # Model
        self.actor_model = None
        self.critic_model = None
        if config["model_para"]["model_initial"]:
            # Start from existing model
            print("[INFO] PPO existing model.")
            self.actor_model = _load(config["model_para"][self._name + "_actor"])
            self.critic_model = _load(config["model_para"][self._name + "_critic"])
        else:
            # Start from new model
            print("[INFO] PPO new model.")
            self.actor_model = NeuralNetwork(
                self._name + "_actor",
                num_output=config["model_para"]["action_dim"],
                input1_shape=config["model_para"]["observation1_dim"],
                input2_shape=config["model_para"]["observation2_dim"],
            )
            self.critic_model = NeuralNetwork(
                self._name + "_critic",
                num_output=1,
                input1_shape=config["model_para"]["observation1_dim"],
                input2_shape=config["model_para"]["observation2_dim"],
            )

        # Path for newly trained model
        time = datetime.now().strftime("%Y_%m_%d_%H_%M")
        self._actor_path = Path(config["model_para"]["model_path"]).joinpath(
            f"{name}_actor_{time}"
        )
        self._critic_path = Path(config["model_para"]["model_path"]).joinpath(
            f"{name}_critic_{time}"
        )

        # Model summary
        # self.actor_model.summary()
        # self.critic_model.summary()

        # Tensorboard
        tb_path = Path(config["model_para"]["tensorboard_path"]).joinpath(
            f"{name}_actor_{time}"
        )
        self.tb = tf.summary.create_file_writer(str(tb_path))

    def close(self):
        pass

    def save(self, version: int):
        tf.keras.models.save_model(
            model=self.actor_model,
            filepath=self._actor_path / str(version),
        )
        tf.keras.models.save_model(
            model=self.critic_model,
            filepath=self._critic_path / str(version),
        )

    def actor(self, obs, train: Mode):
        states = [
            obs[agent_id] for agent_id in self._agent_ids if agent_id in obs.keys()
        ]
        vehicles = [agent_id for agent_id in self._agent_ids if agent_id in obs.keys()]

        images, scalars = zip(*(map(lambda x: (x["image"], x["scalar"]), states)))
        stacked_images = np.stack(images, axis=0)
        stacked_scalars = np.stack(scalars, axis=0)
        logits = self.actor_model.predict([stacked_images, stacked_scalars])

        logit_t = {
            vehicle: np.expand_dims(logit, axis=0)
            for vehicle, logit in zip(vehicles, logits)
        }

        action_t = {
            vehicle: tf.random.categorical([logit], 1, seed=self._seed).numpy()[0][0]
            if train == Mode.TRAIN
            else np.argmax(logit)
            for vehicle, logit in zip(vehicles, logits)
        }

        return logit_t, action_t

    def critic(self, obs):
        states = [
            obs[agent_id] for agent_id in self._agent_ids if agent_id in obs.keys()
        ]
        vehicles = [agent_id for agent_id in self._agent_ids if agent_id in obs.keys()]

        images, scalars = zip(*(map(lambda x: (x["image"], x["scalar"]), states)))
        stacked_images = np.stack(images, axis=0)
        stacked_scalars = np.stack(scalars, axis=0)
        values = self.critic_model.predict([stacked_images, stacked_scalars])

        value_t = {vehicle: value[0] for vehicle, value in zip(vehicles, values)}

        return value_t

    def write_to_tb(self, records):
        with self.tb.as_default():
            for name, value, step in records:
                tf.summary.scalar(name, value, step)


def _load(model_path):
    return tf.keras.models.load_model(
        model_path,
        compile=False,
    )


def logprobabilities(logit, action):
    # Compute the log-probabilities of taking actions a by using the logits (i.e. the output of the actor)
    logprobabilities_all = tf.nn.log_softmax(logit)
    logprobability = np.sum(
        tf.one_hot(action, logit.shape[1]) * logprobabilities_all, axis=1
    )

    return logprobability


# Train the policy by maxizing the PPO-Clip objective
def train_actor(
    policy,
    agent,
    clip_ratio,
    grad_batch,
):

    images, scalars = zip(
        *(map(lambda x: (x["image"], x["scalar"]), agent.observation_buffer))
    )
    stacked_image = tf.stack(images, axis=0)
    stacked_scalar = tf.stack(scalars, axis=0)
    traj_len = stacked_image.shape[0]
    assert traj_len == stacked_scalar.shape[0]

    for ind in range(0, traj_len, grad_batch):
        image_chunk = stacked_image[ind : ind + grad_batch]
        scalar_chunk = stacked_scalar[ind : ind + grad_batch]
        old_probs_chunk = old_probs[ind : ind + grad_batch]
        advantages_chunk = advantages[ind : ind + grad_batch]
        actions_chunk = actions[ind : ind + grad_batch]
        discounted_rewards_chunk = discounted_rewards[ind : ind + grad_batch]

        with tf.GradientTape() as tape:
            policy_logits, values = model([image_chunk, scalar_chunk])
            act_loss = actor_loss(
                advantages=advantages_chunk,
                old_probs=old_probs_chunk,
                actions=actions_chunk,
                policy_logits=policy_logits,
                clip_value=clip_value,
            )
            cri_loss = critic_loss(
                discounted_rewards=discounted_rewards_chunk,
                value_est=values,
                critic_loss_weight=critic_loss_weight,
            )
            ent_loss = entropy_loss(
                policy_logits=policy_logits, ent_discount_val=ent_discount_val
            )
            tot_loss = act_loss + cri_loss + ent_loss

        grads = tape.gradient(tot_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return tot_loss, act_loss, cri_loss, ent_loss


# Train the policy by maxizing the PPO-Clip objective
@tf.function
def train_policy(
    observation_buffer, action_buffer, logprobability_buffer, advantage_buffer
):

    with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
        ratio = tf.exp(
            logprobabilities(actor(observation_buffer), action_buffer)
            - logprobability_buffer
        )
        min_advantage = tf.where(
            advantage_buffer > 0,
            (1 + clip_ratio) * advantage_buffer,
            (1 - clip_ratio) * advantage_buffer,
        )

        policy_loss = -tf.reduce_mean(
            tf.minimum(ratio * advantage_buffer, min_advantage)
        )
    policy_grads = tape.gradient(policy_loss, actor.trainable_variables)
    policy_optimizer.apply_gradients(zip(policy_grads, actor.trainable_variables))

    kl = tf.reduce_mean(
        logprobability_buffer
        - logprobabilities(actor(observation_buffer), action_buffer)
    )
    kl = tf.reduce_sum(kl)
    return kl


# Train the value function by regression on mean-squared error
@tf.function
def train_value_function(observation_buffer, return_buffer):
    with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
        value_loss = tf.reduce_mean((return_buffer - critic(observation_buffer)) ** 2)
    value_grads = tape.gradient(value_loss, critic.trainable_variables)
    value_optimizer.apply_gradients(zip(value_grads, critic.trainable_variables))
