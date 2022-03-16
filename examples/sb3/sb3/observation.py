from collections import deque

import gym
import numpy as np


class Observation(gym.Wrapper):
    def __init__(self, env: gym.Env, num_stack: int):
        super().__init__(env)
        self._num_stack = num_stack
        self._frames = deque(maxlen=self._num_stack)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(256, 256, 3), dtype=np.uint8
        )

    def _stack_obs(self, obs: np.ndarray):
        self._frames.appendleft(obs)
        stacked_obs = np.dstack(self._frames)

        return stacked_obs

    def step(self, action):
        """Steps the environment by one step.

        Args:
            actions (Any): Agent's action.

        Returns:
            Tuple[ np.ndarray, float, bool, Dict[str, Any] ]:
                Observation, reward, done, info, of the agent.
        """
        obs, rewards, dones, infos = self.env.step(action)
        converted = normalize_img(obs.rgb)
        stacked = self._stack_obs(converted)

        # plotter(obs.rgb, rgb_gray=3, name="Original RGB")
        # plotter(stacked, rgb_gray=1, name="Stacked Grayscale")

        return stacked, rewards, dones, infos

    def reset(self):
        """Resets the environment.

        Returns:
            np.ndarray: Agent's observation after reset.
        """
        obs = self.env.reset()
        converted = normalize_img(obs.rgb)
        for _ in range(self._num_stack - 1):
            self._frames.appendleft(converted)

        return self._stack_obs(converted)


def normalize_img(img: np.ndarray) -> np.ndarray:
    from smarts.core import colors as smarts_colors

    # Repaint self
    clr = (122, 140, 153)
    repainted = img.copy()
    repainted[123:132, 126:130, 0] = clr[0]
    repainted[123:132, 126:130, 1] = clr[1]
    repainted[123:132, 126:130, 2] = clr[2]

    # Convert to grayscale
    R, G, B = repainted[:, :, 0], repainted[:, :, 1], repainted[:, :, 2]
    gray = 0.2989 * R + 0.587 * G + 0.114 * B

    # Expand dims
    expanded = np.expand_dims(gray, -1)

    return np.uint8(expanded)


def plotter(obs: np.ndarray, rgb_gray=1, name: str = "Graph"):
    """Plot images

    Args:
        obs (np.ndarray): Image.
        rgb_gray (int, optional): 3 for rgb and 1 for grayscale. Defaults to 1.
    """

    import matplotlib.pyplot as plt

    rows = 1
    columns = obs.shape[2] // rgb_gray
    # cmap = 'gray' if rgb_gray == 1 else 'viridis'
    fig, axs = plt.subplots(nrows=rows, ncols=columns, squeeze=False)
    fig.suptitle("Observation")

    for row in range(0, rows):
        for col in range(0, columns):
            img = obs[:, :, col * rgb_gray : col * rgb_gray + rgb_gray]
            axs[row, col].imshow(img)
            axs[row, col].set_title(f"{name}")
    plt.show()
    # plt.pause(3)
    plt.close()
