import numpy as np
import os
import yaml
from examples.gameOfTag import env as got_env
from examples.gameOfTag import utils
from pathlib import Path


def evaluate(model_predator, model_prey, config, discount_factor, frame_stack_size):
    total_reward = 0
    env = got_env.TagEnv(config, 42)
    obs = env.reset()
    values, rewards, dones = [], [], []

    while True:
        # Predict action given state: π(a_t | s_t; θ)
        state = frame_stack.get_state()
        action, value = model.predict(
            np.expand_dims(state, axis=0), greedy=False)
        frame, reward, done, _ = env.step(action[0])
        total_reward += reward
        dones.append(done)
        values.append(value)
        rewards.append(reward)
        frame_stack.add_frame(frame)
        if done:
            break

    returns = utils.compute_returns(np.transpose([rewards], [1, 0]), [
                              0], np.transpose([dones], [1, 0]), discount_factor)
    value_error = np.mean(np.square(np.array(values) - returns))
    return total_reward, value_error


if __name__ == "__main__":
    config_yaml=(Path(__file__).absolute().parent).joinpath("smarts.yaml")
    with open(config_yaml, "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # Silence the logs of TF
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    evaluate()