import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from datetime import datetime
from pathlib import Path


class NeuralNetwork(tf.keras.Model):
    def __init__(self, num_actions):
        """
        Args:
            num_actions (int): Number of continuous actions to output
        """
        super(NeuralNetwork, self).__init__()
        self.num_actions = num_actions
        self.conv1 = tf.keras.layers.Conv2D(
            filters=16,
            kernel_size=32,
            strides=(4, 4),
            padding="valid",
            activation=tf.keras.activations.relu,
        )
        self.conv2 = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=17,
            strides=(2, 2),
            padding="valid",
            activation=tf.keras.activations.relu,
        )
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(
            units=128, activation=tf.keras.activations.relu
        )
        self.dense_value = tf.keras.layers.Dense(
            units=64, activation=tf.keras.activations.relu
        )
        self.dense_policy = tf.keras.layers.Dense(
            units=64, activation=tf.keras.activations.relu
        )
        self.policy = tf.keras.layers.Dense(units=self.num_actions)
        self.value = tf.keras.layers.Dense(units=1)

    def call(self, inputs):
        """
        Args:
            inputs ([batch_size, width, height, depth]): Input images to predict actions for

        Returns:
            [type]: action
            [type]: value
        """
        conv1_out = self.conv1(inputs)
        conv2_out = self.conv2(conv1_out)
        flatten_out = self.flatten(conv2_out)
        dense1_out = self.dense1(flatten_out)
        dense_policy_out = self.dense_policy(dense1_out)
        dense_value_out = self.dense_value(dense1_out)
        policy = self.policy(dense_policy_out)
        value = self.value(dense_value_out)
        return policy, value


class PPO(object):
    def __init__(self, name, config):
        self.name = name
        self.config = config
        self.seed = config["env_para"]["seed"]
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=config["model_para"]["initial_lr_" + name]
        )

        # Model
        self.model_path = Path(self.config["model_para"]["model_path"]).joinpath(
            f"{name}_{datetime.now().strftime('%Y_%m_%d_%H_%M')}"
        )
        self.model = None
        if self.config["model_para"]["model_initial"]:
            self.model = _load(self.config["model_para"]["model_path"])
        else:
            self.model = NeuralNetwork(self.config["model_para"]["action_dim"])

        # Tensorboard
        path = Path(self.config["model_para"]["tensorboard"]).joinpath(
            f"{name}_{datetime.now().strftime('%Y_%m_%d_%H_%M')}"
        )
        self.tb = tf.summary.create_file_writer(str(path))

    def save(self):
        tf.saved_model.save(self.model, self.model_path)

    def act(self, obs):
        actions = {}
        action_samples = {}
        values = {}
        for vehicle, state in obs.items():
            if self.name in vehicle:
                actions_t, values_t = self.model.predict_on_batch(
                    np.expand_dims(state, axis=0)
                )
                actions_dist_t = tfp.distributions.Categorical(logits=actions_t)

                actions[vehicle] = tf.squeeze(actions_t, axis=0)
                action_samples[vehicle] = tf.squeeze(
                    actions_dist_t.sample(1, seed=self.seed), axis=0
                )
                values[vehicle] = tf.squeeze(values_t, axis=0)
        return actions, action_samples, values

    def write_to_tb(self, records):
        with self.tb.as_default():
            for name, value, step in records:
                tf.summary.scalar(name, value, step)


def _load(model_path):
    return tf.saved_model.load(model_path)


def train_model(
    model,
    optimizer,
    action_inds,
    old_probs,
    states,
    advantages,
    discounted_rewards,
    ent_discount_val,
    clip_value,
    critic_loss_weight,
):
    with tf.GradientTape() as tape:
        policy_logits, values = model.call(tf.stack(states))
        act_loss = actor_loss(
            advantages, old_probs, action_inds, policy_logits, clip_value
        )
        ent_loss = entropy_loss(policy_logits, ent_discount_val)
        cri_loss = critic_loss(discounted_rewards, values, critic_loss_weight)
        tot_loss = act_loss + ent_loss + cri_loss

    watched = [var.name for var in tape.watched_variables()]
    print(watched)
    raise Exception("TEST -------------------")

    grads = tape.gradient(tot_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return tot_loss, cri_loss, act_loss, ent_loss


# Clipped objective term, to be maximized
# @tf.function
def actor_loss(advantages, old_probs, action_inds, policy_logits, clip_value):
    probs = tf.nn.softmax(policy_logits)
    new_probs = tf.gather_nd(probs, action_inds)
    ratio = new_probs / old_probs  # Ratio is always positive

    print("ACTOR LOSS --------- ")
    print(ratio)
    print(advantages)
    print(tf.clip_by_value(ratio, 1.0 - clip_value, 1.0 + clip_value) * advantages)

    policy_loss = -tf.reduce_mean(  # -Expectation
        tf.math.minimum(
            ratio * advantages,
            tf.clip_by_value(ratio, 1.0 - clip_value, 1.0 + clip_value) * advantages,
        )
    )
    print("POLICY LOSS ------ ", policy_loss)
    return policy_loss


# Entropy term to encourage exploration, to be maximized
# @tf.function
def entropy_loss(policy_logits, ent_discount_val) -> tf.float32:
    probs = tf.nn.softmax(policy_logits)
    entropy_loss = -tf.reduce_mean(
        tf.keras.losses.categorical_crossentropy(probs, probs)
    )
    return entropy_loss * ent_discount_val


# Error term on value estimation, to be minimized
# @tf.function
def critic_loss(discounted_rewards, value_est, critic_loss_weight) -> tf.float32:
    return (
        tf.reduce_mean(
            tf.keras.losses.mean_squared_error(discounted_rewards, value_est)
        )
        * critic_loss_weight
    )
