import numpy as np
from pathlib import Path
import re
import shutil
import tensorflow as tf

from datetime import datetime

import tensorflow_probability as tfp


class NeuralNetwork(tf.keras.Model):
    def __init__(self, num_actions):
        """
        Args:
            num_actions (int): Number of continuous actions to output
        """
        super().__init__()
        self.num_actions = num_actions
        self.conv1 = tf.keras.layers.Conv2D(filters=16, kernel_size=32, strides=(4,4), padding='valid', activation=tf.keras.activations.relu)
        self.conv2 = tf.keras.layers.Conv2D(filters=32, kernel_size=17, strides=(4,4), padding='valid', activation=tf.keras.activations.relu)
        self.flatten = tf.keras.layers.flatten()
        self.dense1 = tf.keras.layers.Dense(units=128, activation=tf.keras.activations.relu)
        self.dense_value = tf.keras.layers.Dense(units=64, activation=tf.keras.activations.relu)
        self.dense_policy = tf.keras.layers.Dense(units=64, activation=tf.keras.activations.relu)
        self.policy = tf.keras.layers.Dense(units=self.num_actions)
        self.value = tf.keras.layers.Dense(units=1, activation=None)

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
        conv3_out = self.conv3(conv2_out)
        flatten_out = self.flatten(conv3_out)
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
        self.model = NeuralNetwork(self.config['model_para']['action_dim'])
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=config['model_para']['initial_lr_'+name])
        path = Path(self.config['model_para']['tensorboard']).joinpath(f"{name}_{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')}")        
        self.tb = tf.summary.create_file_writer(path)

    def act(self, obs):
        actions = {}
        action_samples = {}
        values = {}
        for vehicle, state in obs.items():
            if self.name in vehicle:
                actions_t, values_t = self.model(np.expand_dims(state, axis=0))
                actions_normal_t = tfp.distributions.Categorical(logits=actions_t)

                actions[vehicle] = tf.squeeze(actions_t, axis=0)
                action_samples[vehicle] = tf.squeeze(actions_normal_t.sample(1), axis=0)
                values[vehicle] = tf.squeeze(values_t, axis=-1)
        return actions, action_samples, values

        # # Get the log probability of taken actions
        # # log π(a_t | s_t; θ)
        # self.action_log_prob = tf.reduce_sum(
        #     self.action_normal.log_prob(taken_actions), axis=-1, keepdims=True
        # )

    def save(self):
        return self.model.save()

    def restore(self):
        return 

def train_model(model, optimizer, action_inds, old_probs, states, advantages, discounted_rewards, ent_discount_val):
    with tf.GradientTape() as tape:
        values, policy_logits = model(tf.stack(states))
        act_loss = actor_loss(advantages, old_probs, action_inds, policy_logits)
        ent_loss = entropy_loss(policy_logits, ent_discount_val)
        c_loss = critic_loss(discounted_rewards, values)
        tot_loss = act_loss + ent_loss + c_loss
    grads = tape.gradient(tot_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return tot_loss, c_loss, act_loss, ent_loss


def critic_loss(discounted_rewards, value_est, critic_loss_weight):
    return tf.cast(tf.reduce_mean(tf.keras.losses.mean_squared_error(discounted_rewards, value_est)) * critic_loss_weight,
                   tf.float32)

def entropy_loss(policy_logits, ent_discount_val):
    probs = tf.nn.softmax(policy_logits)
    entropy_loss = -tf.reduce_mean(tf.keras.losses.categorical_crossentropy(probs, probs))
    return entropy_loss * ent_discount_val

# Entropy loss
self.entropy_loss = (
    tf.reduce_mean(tf.reduce_sum(self.policy.action_normal.entropy(), axis=-1)) * entropy_scale
)
    






class PPO:
    """
    Proximal policy gradient model class
    """

    def __init__(
        self,
        input_shape,
        num_actions,
        action_min,
        action_max,
        epsilon=0.2,
        value_scale=0.5,
        entropy_scale=0.01,
        model_checkpoint=None,
        model_name="ppo",
        base_path="./",
    ):
        """
        input_shape [3]:
            Shape of input images as a tuple (width, height, depth)
        num_actions (int):
            Number of continuous actions to output
        action_min [num_actions]:
            Minimum possible value for the respective action
        action_max [num_actions]:
            Maximum possible value for the respective action
        epsilon (float):
            PPO clipping parameter
        value_scale (float):
            Value loss scale factor
        entropy_scale (float):
            Entropy loss scale factor
        model_checkpoint (string):
            Path of model checkpoint file to load from
        model_name (string):
            Name of the model
        """

        tf.reset_default_graph()

        self.input_states = tf.placeholder(
            shape=(None, *input_shape), dtype=tf.float32, name="input_state_placeholder"
        )
        self.taken_actions = tf.placeholder(
            shape=(None, num_actions), dtype=tf.float32, name="taken_action_placeholder"
        )
        self.policy = PolicyGraph(
            self.input_states,
            self.taken_actions,
            num_actions,
            action_min,
            action_max,
            "policy",
            clip_action_space=True,
        )
        self.policy_old = PolicyGraph(
            self.input_states,
            self.taken_actions,
            num_actions,
            action_min,
            action_max,
            "policy_old",
            clip_action_space=True,
        )

        # Create policy gradient train function
        self.returns = tf.placeholder(
            shape=(None,), dtype=tf.float32, name="returns_placeholder"
        )
        self.advantage = tf.placeholder(
            shape=(None,), dtype=tf.float32, name="advantage_placeholder"
        )

        # Calculate ratio:
        # r_t(θ) = exp( log   π(a_t | s_t; θ) - log π(a_t | s_t; θ_old)   )
        # r_t(θ) = exp( log ( π(a_t | s_t; θ) /     π(a_t | s_t; θ_old) ) )
        # r_t(θ) = π(a_t | s_t; θ) / π(a_t | s_t; θ_old)
        self.prob_ratio = tf.exp(
            self.policy.action_log_prob - self.policy_old.action_log_prob
        )

        # Policy loss
        adv = tf.expand_dims(self.advantage, axis=-1)
        self.policy_loss = tf.reduce_mean(
            tf.minimum(
                self.prob_ratio * adv,
                tf.clip_by_value(self.prob_ratio, 1.0 - epsilon, 1.0 + epsilon) * adv,
            )
        )

        # Value loss = mse(V(s_t) - R_t)
        self.value_loss = (
            tf.reduce_mean(
                tf.squared_difference(tf.squeeze(self.policy.value), self.returns)
            )
            * value_scale
        )

        # Entropy loss
        self.entropy_loss = (
            tf.reduce_mean(tf.reduce_sum(self.policy.action_normal.entropy(), axis=-1))
            * entropy_scale
        )

        # Total loss
        self.loss = -self.policy_loss + self.value_loss - self.entropy_loss

        # Policy parameters
        policy_params = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope="policy/"
        )
        policy_old_params = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope="policy_old/"
        )
        assert len(policy_params) == len(policy_old_params)
        for src, dst in zip(policy_params, policy_old_params):
            assert src.shape == dst.shape

        # Minimize loss
        self.learning_rate = tf.placeholder(
            shape=(), dtype=tf.float32, name="lr_placeholder"
        )
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_step = self.optimizer.minimize(self.loss, var_list=policy_params)

        # Update network parameters
        self.update_op = tf.group(
            [dst.assign(src) for src, dst in zip(policy_params, policy_old_params)]
        )

        # Create session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

        # Run the initializer
        self.sess.run(tf.global_variables_initializer())

        # Summaries
        tf.summary.scalar("loss_policy", self.policy_loss)
        tf.summary.scalar("loss_value", self.value_loss)
        tf.summary.scalar("loss_entropy", self.entropy_loss)
        tf.summary.scalar("loss", self.loss)
        for i in range(num_actions):
            tf.summary.scalar(
                "taken_actions_{}".format(i), tf.reduce_mean(self.taken_actions[:, i])
            )
            tf.summary.scalar(
                "policy.action_mean_{}".format(i),
                tf.reduce_mean(self.policy.action_mean[:, i]),
            )
            tf.summary.scalar(
                "policy.action_std_{}".format(i),
                tf.reduce_mean(tf.exp(self.policy.action_logstd[i])),
            )
        tf.summary.scalar("prob_ratio", tf.reduce_mean(self.prob_ratio))
        tf.summary.scalar("returns", tf.reduce_mean(self.returns))
        tf.summary.scalar("advantage", tf.reduce_mean(self.advantage))
        tf.summary.scalar("learning_rate", tf.reduce_mean(self.learning_rate))
        self.summary_merged = tf.summary.merge_all()

        # Load model checkpoint
        self.model_name = model_name
        self.saver = tf.train.Saver()

        dateNow = datetime.now().strftime("%Y_%m_%d_%I_%M_%S")
        self.model_dir = f"{base_path}got/models_{dateNow}/{self.model_name}"
        self.log_dir = f"{base_path}got/logs_{dateNow}/{self.model_name}"
        # if model_checkpoint is None and os.path.isdir(self.model_dir):
        #     answer = input(
        #         "{} exists. Do you wish to continue (C) or restart training (R)?".format(self.model_dir))
        #     if answer.upper() == "C":
        #         model_checkpoint = tf.train.latest_checkpoint(self.model_dir)
        #     elif answer.upper() != "R":
        #         raise Exception(
        #             "There is already a model directory {}. Please delete it or change model_name and try again".format(self.model_dir))

        if model_checkpoint:
            self.step_idx = int(
                re.findall(r"[/\\]step\d+", model_checkpoint)[0][len("/step") :]
            )
            self.saver.restore(self.sess, model_checkpoint)
            print("[INFO] Model checkpoint restored from {}".format(model_checkpoint))
        else:
            self.step_idx = 0
            for d in [self.model_dir, self.log_dir]:
                if os.path.isdir(d):
                    shutil.rmtree(d)
                os.makedirs(d)

        self.train_writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)

    def save(self):
        model_checkpoint = os.path.join(
            self.model_dir, "step{}.ckpt".format(self.step_idx)
        )
        self.saver.save(self.sess, model_checkpoint)
        print("[INFO] Model checkpoint saved to {}".format(model_checkpoint))

    def train(
        self, input_states, taken_actions, returns, advantage, learning_rate=1e-4
    ):
        r = self.sess.run(
            [
                self.summary_merged,
                self.train_step,
                self.loss,
                self.policy_loss,
                self.value_loss,
                self.entropy_loss,
            ],
            feed_dict={
                self.input_states: input_states,
                self.taken_actions: taken_actions,
                self.returns: returns,
                self.advantage: advantage,
                self.learning_rate: learning_rate(self.step_idx)
                if callable(learning_rate)
                else learning_rate,
            },
        )
        self.train_writer.add_summary(r[0], self.step_idx)
        self.step_idx += 1
        return r[2:]

    def predict(self, input_states, use_old_policy=False, greedy=False):
        policy = self.policy_old if use_old_policy else self.policy
        action = policy.action_mean if greedy else policy.sampled_action
        return self.sess.run(
            [action, policy.value], feed_dict={self.input_states: input_states}
        )

    def write_to_summary(self, name, value):
        summary = tf.Summary()
        summary.value.add(tag=name, simple_value=value)
        self.train_writer.add_summary(summary, self.step_idx)

    def update_old_policy(self):
        self.sess.run(self.update_op)
