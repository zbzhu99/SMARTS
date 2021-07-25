import numpy as np
import utils
from examples.gameOfTag import model as got_model
from typing import Sequence, Union


class TagAgent:
    def __init__(self, name, config):
        # Verify name
        if "predator" in name or "prey" in name:
            self.name = name
        else:
            raise Exception(f"Expected predator or prey, but got {name}.")
        self._config = config
        self.reset()
        self._gamma = config['model_para']['gamma']

    def reset(self):
        self._states = []
        self._actions = []
        self._values = []
        self._rewards = []
        self._dones = []
        self._probs = []
        self._advantages = None
        self._returns = None
        self._last_value = None

    @property
    def actions(self):
        return self._actions

    @property
    def probs(self):
        return self._probs

    def add_trajectory(self, action, value, state, done, prob, reward):
        self._states.append(state)
        self._actions.append(action)
        self._values.append(value)
        self._probs.append(prob)
        self._dones.append(done)
        self._rewards.append(reward)

    def add_last_transition(self, value):
        self._last_value = value


    def compute_gae(self):
        advantages = utils.compute_gae(
            rewards=self.rewards,
            values=self.values,
            bootstrap_values=self.last_value,
            terminals=self.dones,
            gamma=self.config["model_para"]["discount_factor"],
            lam=self.config["model_para"]["gae_lambda"],
        )
        self.advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        self.returns = advantages + self.values

    def compute_returns(self):
        returns = utils.compute_returns(
            rewards=np.transpose([self.rewards], [1, 0]),
            bootstrap_value=[0],
            terminals=np.transpose([self.dones], [1, 0]),
            gamma=self.config["model_para"]["discount_factor"],
        )
        return returns

    def compute_advantages(self):
        discounted_rewards = np.array(self._rewards + [self._last_value])

        for t in reversed(range(len(self._rewards))):
            discounted_rewards[t] = self._rewards[t] + self._gamma * discounted_rewards[t+1] * (1-self._dones[t])

        discounted_rewards = discounted_rewards[:-1]
        # advantages are bootstrapped discounted rewards - values, using Bellman's equation
        advantages = discounted_rewards - np.stack(self._values)[:, 0]
        # standardise advantages
        advantages -= np.mean(advantages)
        advantages /= (np.std(advantages) + 1e-10)
        # standardise rewards too
        discounted_rewards -= np.mean(discounted_rewards)
        discounted_rewards /= (np.std(discounted_rewards) + 1e-8)

        return discounted_rewards, advantages