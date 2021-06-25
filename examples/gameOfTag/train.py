import numpy as np
import os
import signal
import yaml
from examples.gameOfTag import env as got_env
from examples.gameOfTag import model as got_model
from pathlib import Path
from utils import compute_gae


def train(config, model_name, save_interval=1000, eval_interval=200):

    # Traning parameters
    discount_factor = config['model_para']["discount_factor"]
    gae_lambda = config['model_para']["gae_lambda"]
    ppo_epsilon = config['model_para']["ppo_epsilon"]
    value_scale = config['model_para']["value_scale"]
    entropy_scale = config['model_para']["entropy_scale"]
    horizon = config['model_para']["horizon"]
    num_epochs = config['model_para']["num_epochs"]
    batch_size = config['model_para']["batch_size"]

    print("[INFO] Creating environments")
    env = got_env.TagEnv(config)

    # Environment constants
    input_shape = env.observation_space.shape
    num_actions = env.action_space.shape[0]
    action_min = env.action_space.low
    action_max = env.action_space.high

    # Create model
    print("[INFO] Creating model")
    model_predator = got_model.PPO(input_shape, 
        num_actions, action_min, action_max,
        epsilon=ppo_epsilon,
        value_scale=value_scale, 
        entropy_scale=entropy_scale,
        model_name='predator')
    model_prey = got_model.PPO(input_shape, 
        num_actions, action_min, action_max,
        epsilon=ppo_epsilon,
        value_scale=value_scale, 
        entropy_scale=entropy_scale,
        model_name='prey')

    def interrupt(*args):
        model_predator.save()
        model_prey.save()
        print("Interrupt key detected.") 

    # Catch keyboard interrupt and terminate signal
    signal.signal(signal.SIGINT, interrupt)

    print("[INFO] Training loop")
    obs = env.reset()
    while True:
        # While there are running environments
        states, taken_actions, values, rewards, dones = [], [], [], [], []

        # Simulate game for some number of steps
        for _ in range(horizon):
            # Predict and value action given state
            # π(a_t | s_t; θ_old)
            actions_t_predator, values_t_predator = model_predator.predict(obs)
            actions_t_prey, values_t_prey = model_prey.predict(obs)

            # Sample action from a Gaussian distribution
            envs.step_async(actions_t)
            frames, rewards_t, dones_t, _ = envs.step_wait()
            envs.get_images()  # render

            # Store state, action and reward
            # [T, N, 84, 84, 4]
            states.append(states_t)
            taken_actions.append(actions_t)              # [T, N, 3]
            values.append(np.squeeze(values_t, axis=-1))  # [T, N]
            rewards.append(rewards_t)                    # [T, N]
            dones.append(dones_t)                        # [T, N]

            # Get new state
            for i in range(num_envs):
                # Reset environment's frame stack if done
                if dones_t[i]:
                    for _ in range(frame_stack_size):
                        frame_stacks[i].add_frame(frames[i])
                else:
                    frame_stacks[i].add_frame(frames[i])

        # Calculate last values (bootstrap values)
        states_last = [frame_stacks[i].get_state()
                        for i in range(num_envs)]
        last_values = np.squeeze(model.predict(
            states_last)[1], axis=-1)  # [N]

        advantages = compute_gae(
            rewards, values, last_values, dones, discount_factor, gae_lambda)
        advantages = (advantages - advantages.mean()) / \
            (advantages.std() + 1e-8)  # Move down one line?
        returns = advantages + values
        # Flatten arrays
        states = np.array(states).reshape(
            (-1, *input_shape))       # [T x N, 84, 84, 4]
        taken_actions = np.array(taken_actions).reshape(
            (-1, num_actions))  # [T x N, 3]
        # [T x N]
        returns = returns.flatten()
        # [T X N]
        advantages = advantages.flatten()

        T = len(rewards)
        N = num_envs
        assert states.shape == (
            T * N, input_shape[0], input_shape[1], frame_stack_size)
        assert taken_actions.shape == (T * N, num_actions)
        assert returns.shape == (T * N,)
        assert advantages.shape == (T * N,)

        # Train for some number of epochs
        model.update_old_policy()  # θ_old <- θ
        for _ in range(num_epochs):
            num_samples = len(states)
            indices = np.arange(num_samples)
            np.random.shuffle(indices)
            for i in range(int(np.ceil(num_samples / batch_size))):
                # Evaluate model
                if model.step_idx % eval_interval == 0:
                    print("[INFO] Running evaluation...")
                    avg_reward, value_error = evaluate(
                        model, test_env, discount_factor, frame_stack_size, make_video=True)
                    model.write_to_summary("eval_avg_reward", avg_reward)
                    model.write_to_summary("eval_value_error", value_error)

                # Save model
                if model.step_idx % save_interval == 0:
                    model.save()

                # Sample mini-batch randomly
                begin = i * batch_size
                end = begin + batch_size
                if end > num_samples:
                    end = None
                mb_idx = indices[begin:end]

                # Optimize network
                model.train(states[mb_idx], taken_actions[mb_idx],
                            returns[mb_idx], advantages[mb_idx])


if __name__ == "__main__":
    config_yaml=(Path(__file__).absolute().parent).joinpath("smarts.yaml")
    with open(config_yaml, "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # Silence the logs of TF
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    train(
        config=config, 
        model_name=config['env_para']['env_name'], 
        save_interval=config['model_para']['save_interval'],
        eval_interval=config['model_para']['eval_interval'],
    )
