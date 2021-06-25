import os
import argparse
import gym
import yaml
from examples import gameOfTag as got 

import cv2
import numpy as np
from utils import FrameStack, compute_gae, compute_returns

from ppo import PPO
from vec_env.subproc_vec_env import SubprocVecEnv


def make_env():
    return gym.make(env_name)

def train(config, model_name, save_interval=1000, eval_interval=200):
    try:
        test_env = gym.make(env_name)

        # Traning parameters
        initial_lr = config['model_para']['model_config']["initial_lr"]
        discount_factor = config['model_para']['model_config']["discount_factor"]
        gae_lambda = config['model_para']['model_config']["gae_lambda"]
        ppo_epsilon = config['model_para']['model_config']["ppo_epsilon"]
        value_scale = config['model_para']['model_config']["value_scale"]
        entropy_scale = config['model_para']['model_config']["entropy_scale"]
        horizon = config['model_para']['model_config']["horizon"]
        num_epochs = config['model_para']['model_config']["num_epochs"]
        batch_size = config['model_para']['model_config']["batch_size"]

        # Training parameters
        def lr_scheduler(step_idx): return initial_lr * \
            0.85 ** (step_idx // 10000)

        # Environment constants
        frame_stack_size = 4
        input_shape = (84, 84, frame_stack_size)
        num_actions = test_env.action_space.shape[0]
        action_min = test_env.action_space.low
        action_max = test_env.action_space.high

        # Create model
        print("[INFO] Creating model")
        model = PPO(input_shape, num_actions, action_min, action_max,
                    epsilon=ppo_epsilon,
                    value_scale=value_scale, entropy_scale=entropy_scale,
                    model_name=model_name)

        print("[INFO] Creating environments")
        envs = SubprocVecEnv([make_env for _ in range(num_envs)])

        initial_frames = envs.reset()
        envs.get_images()
        frame_stacks = [FrameStack(initial_frames[i], stack_size=frame_stack_size,
                                   preprocess_fn=preprocess_frame) for i in range(num_envs)]

        print("[INFO] Training loop")
        while True:
            # While there are running environments
            states, taken_actions, values, rewards, dones = [], [], [], [], []

            # Simulate game for some number of steps
            for _ in range(horizon):
                # Predict and value action given state
                # π(a_t | s_t; θ_old)
                states_t = [frame_stacks[i].get_state()
                            for i in range(num_envs)]
                actions_t, values_t = model.predict(states_t)

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
    except KeyboardInterrupt:
        model.save()


if __name__ == "__main__":
    with open(r"./smarts.yaml") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
        print(config)

    # Silence the logs of TF
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    train(
        config=config, 
        model_name=config['env_para']['env_name'], 
        save_interval=config['model_para']['model_config']['save_interval'],
        eval_interval=config['model_para']['model_config']['eval_interval'],
        record_episodes=config['model_para']['model_config']['record_episodes'],
        restart=config['model_para']['model_config']['restart']
    )
