from typing import Any, Dict, List, Tuple

import absl.logging
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from examples.auto_drive.nn import cnn
from examples.auto_drive.rl import mode, rl

# Suppress warning
absl.logging.set_verbosity(absl.logging.ERROR)


class PPO(rl.RL):
    def __init__(self, name, config, seed, modeldir, logdir):
        super(PPO, self).__init__()

        self._name = name
        self._seed = seed
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=config["initial_lr"])

        # Model
        self.model = None
        if config["model_initial"]:  # Start from existing model
            print("[INFO] PPO existing model.")
            self.model = _load(config["path_old_model"])
        else:  # Start from new model
            print("[INFO] PPO new model.")
            self.model = getattr(cnn, config["nn"])(
                self._name, config["action_dim"], config["observation_dim"]
            )

        # Path for newly trained model
        self.path_new_model = modeldir.joinpath(f"{self._name}")

        # Model summary
        self.model.summary()

        # Tensorboard
        path_tensorboard = logdir.joinpath(f"{self._name}")
        self.tb = tf.summary.create_file_writer(str(path_tensorboard))

    def close(self):
        pass

    def save(self, version: int):
        tf.keras.models.save_model(
            model=self.model,
            filepath=self.path_new_model / str(version),
        )

    def act(self, obs, train: mode.Mode):
        actions = {}
        action_samples = {}
        values = {}

        ordered_obs = _dict_to_ordered_list(obs)

        for vehicle, state in ordered_obs:
            stacked_image = tf.stack([state], axis=0)
            actions_t, values_t = self.model.predict(stacked_image)
            actions[vehicle] = tf.squeeze(actions_t, axis=0)
            values[vehicle] = tf.squeeze(values_t)

            if train == mode.Mode.TRAIN:
                actions_dist_t = tfp.distributions.Categorical(logits=actions[vehicle])
                action_samples[vehicle] = actions_dist_t.sample(seed=self._seed)
            else:
                action_samples[vehicle] = tf.math.argmax(actions[vehicle])

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
    actions: List,
    old_probs: List,
    states: List[np.ndarray],
    advantages: np.ndarray,
    discounted_rewards: np.ndarray,
    clip_ratio: float,
    critic_loss_weight: float,
    grad_batch=64,
):
    stacked_image = tf.stack(states, axis=0)
    traj_len = stacked_image.shape[0]
    for ind in range(0, traj_len, grad_batch):
        image_chunk = stacked_image[ind : ind + grad_batch]
        old_probs_chunk = old_probs[ind : ind + grad_batch]
        advantages_chunk = advantages[ind : ind + grad_batch]
        actions_chunk = actions[ind : ind + grad_batch]
        discounted_rewards_chunk = discounted_rewards[ind : ind + grad_batch]

        with tf.GradientTape() as tape:
            policy_logits, values = model(image_chunk)
            act_loss = actor_loss(
                advantages=advantages_chunk,
                old_probs=old_probs_chunk,
                actions=actions_chunk,
                policy_logits=policy_logits,
                clip_ratio=clip_ratio,
            )
            cri_loss = critic_loss(
                discounted_rewards=discounted_rewards_chunk,
                value_est=values,
                critic_loss_weight=critic_loss_weight,
            )
            tot_loss = act_loss + cri_loss

        grads = tape.gradient(tot_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return tot_loss, act_loss, cri_loss


# Clipped objective term, to be maximized
def actor_loss(advantages, old_probs, actions, policy_logits, clip_ratio):
    action_inds = tf.stack(
        [tf.range(0, len(actions)), tf.cast(actions, tf.int32)], axis=1
    )
    probs = tf.nn.softmax(policy_logits)
    new_probs = tf.gather_nd(probs, action_inds)
    ratio = new_probs / old_probs  # Ratio is always positive

    policy_loss = -tf.reduce_mean(  # -Expectation
        tf.math.minimum(
            ratio * advantages,
            tf.clip_by_value(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantages,
        )
    )
    return policy_loss


# Error term on value estimation, to be minimized
def critic_loss(discounted_rewards, value_est, critic_loss_weight):
    return (
        tf.reduce_mean(
            tf.keras.losses.mean_squared_error(discounted_rewards, value_est)
        )
        * critic_loss_weight
    )
