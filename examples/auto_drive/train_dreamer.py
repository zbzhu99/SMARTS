import os

from smarts.env.wrappers import single_agent_frame_stack

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

import argparse
import signal
import sys
import warnings
import yaml

from examples.auto_drive.env import traffic
from examples.auto_drive.agent import behaviour
from examples.auto_drive.agent import vehicle
from examples.auto_drive.rl import ppo, mode
from pathlib import Path


def main(config):

    print("[INFO] Train")
    save_interval = config["model_para"].get("save_interval", 20)
    run_mode = mode.Mode(config["model_para"]["mode"])  # Mode: Evaluation or Testing

    # Traning parameters
    num_train_epochs = config["model_para"]["num_train_epochs"]
    n_steps = config["model_para"]["n_steps"]
    max_traj = config["model_para"]["max_traj"]
    clip_ratio = config["model_para"]["clip_ratio"]
    critic_loss_weight = config["model_para"]["critic_loss_weight"]

    # Create env
    print("[INFO] Creating environments")
    env_frame_stack = traffic.Traffic(config, config["env_para"]["seed"])
    env = single_agent_frame_stack.SingleAgentFrameStack(env_frame_stack)

    # Create agent
    print("[INFO] Creating agents")
    all_agents = {name: vehicle.Vehicle(name, config) for name in env.agent_ids}

    # Create model
    print("[INFO] Creating model")
    policy = ppo.PPO(
        behaviour.Behaviour.CRUISER, config, config["env_para"]["seed"] + 1
    )

    def interrupt(*args):
        nonlocal run_mode
        nonlocal policy
        if run_mode == mode.Mode.TRAIN:
            policy.save(-1)
        env.close()
        print("Interrupt key detected.")
        sys.exit(0)

    # Catch keyboard interrupt and terminate signal
    signal.signal(signal.SIGINT, interrupt)

    print("[INFO] Batch loop")
    states_t = env.reset()
    episode = 0
    steps_t = 0
    episode_reward = 0
    flag_crash = False
    for traj_num in range(max_traj):
        [agent.reset() for _, agent in all_agents.items()]
        active_agents = {}

        print(f"[INFO] New batch data collection {traj_num}/{max_traj}")
        for cur_step in range(n_steps):

            # Update all agents which were active in this batch
            active_agents.update({agent_id: True for agent_id, _ in states_t.items()})

            # Predict and value action given state
            actions_t = {}
            action_samples_t = {}
            values_t = {}
            (
                actions_t,
                action_samples_t,
                values_t,
            ) = policy.act(obs=states_t, train=run_mode)
            actions_t.update(actions_t)
            action_samples_t.update(action_samples_t)
            values_t.update(values_t)

            # Sample action from a distribution
            action_numpy_t = {
                vehicle: action_sample_t.numpy()
                for vehicle, action_sample_t in action_samples_t.items()
            }
            try:
                next_states_t, rewards_t, dones_t, _ = env.step(action_numpy_t)
            except:
                # To counter tracii error
                print(
                    f"   Simulation crashed and reset. Cur_Step: {cur_step}. Step: {steps_t}."
                )
                step = traj_num * n_steps + cur_step
                policy.save(-1 * step)
                new_env = traffic.Traffic(config, config["env_para"]["seed"] + step)
                env = new_env
                next_states_t = env.reset()
                states_t = next_states_t
                flag_crash = True
                break

            steps_t += 1

            # Store state, action and reward
            for agent_id, _ in states_t.items():
                all_agents[agent_id].add_trajectory(
                    action=action_samples_t[agent_id],
                    value=values_t[agent_id].numpy(),
                    state=states_t[agent_id],
                    done=int(dones_t[agent_id]),
                    prob=actions_t[agent_id],
                    reward=rewards_t[agent_id],
                )
                episode_reward += rewards_t[agent_id]
                if dones_t[agent_id] == 1:
                    # Remove done agents
                    del next_states_t[agent_id]
                    # Print done agents
                    print(
                        f"   Done: {agent_id}. Cur_Step: {cur_step}. Step: {steps_t}."
                    )

            # Reset when episode completes
            if dones_t["__all__"]:
                # Next episode
                next_states_t = env.reset()
                episode += 1

                # Log rewards
                print(
                    f"   Episode: {episode}. Cur_Step: {cur_step}. "
                    f"Episode reward: {episode_reward}."
                )
                policy.write_to_tb([("episode_reward", episode_reward, episode)])

                # Reset counters
                episode_reward = 0
                steps_t = 0

            # Assign next_states to states
            states_t = next_states_t

        # If env crash due to tracii error, reset env and skip to next trajectory.
        if flag_crash == True:
            flag_crash = False
            continue

        # Skip the remainder if evaluating
        if run_mode == mode.Mode.EVALUATE:
            continue

        # Compute and store last state value
        for agent_id in active_agents.keys():
            if dones_t.get(agent_id, None) == 0:  # Agent not done yet
                _, _, next_values_t = policy.act(
                    {agent_id: next_states_t[agent_id]}, train=run_mode
                )
                all_agents[agent_id].add_last_transition(
                    value=next_values_t[agent_id].numpy()
                )
            else:  # Agent is done
                all_agents[agent_id].add_last_transition(value=np.float32(0))

        for agent_id in active_agents.keys():
            # Compute generalised advantages
            all_agents[agent_id].compute_advantages()

            actions = all_agents[agent_id].actions
            action_inds = tf.stack(
                [tf.range(0, len(actions)), tf.cast(actions, tf.int32)], axis=1
            )

            # Compute old probabilities
            probs_softmax = tf.nn.softmax(all_agents[agent_id].probs)
            all_agents[agent_id].old_probs = tf.gather_nd(probs_softmax, action_inds)

        total_loss = np.zeros((num_train_epochs))
        actor_loss = np.zeros((num_train_epochs))
        critic_loss = np.zeros(((num_train_epochs)))

        # Elapsed steps
        step = (traj_num + 1) * n_steps

        print("[INFO] Training")
        # Run multiple gradient ascent on the samples.
        for epoch in range(num_train_epochs):
            for agent_id in active_agents.keys():
                agent = all_agents[agent_id]

                loss_tuple = ppo.train_model(
                    model=policy.model,
                    optimizer=policy.optimizer,
                    actions=agent.actions,
                    old_probs=agent.old_probs,
                    states=agent.states,
                    advantages=agent.advantages,
                    discounted_rewards=agent.discounted_rewards,
                    clip_ratio=clip_ratio,
                    critic_loss_weight=critic_loss_weight,
                    grad_batch=config["model_para"]["grad_batch"],
                )
                total_loss[epoch] += loss_tuple[0]
                actor_loss[epoch] += loss_tuple[1]
                critic_loss[epoch] += loss_tuple[2]

        print("[INFO] Record metrics")
        records = []
        records.append(("tot_loss", np.mean(total_loss), step))
        records.append(("critic_loss", np.mean(critic_loss), step))
        records.append(("actor_loss", np.mean(actor_loss), step))
        policy.write_to_tb(records)

        # Save model
        if traj_num % save_interval == 0:
            print("[INFO] Saving model")
            policy.save(step)

    # Close env
    env.close()


def replace_args(config):
    parser = argparse.ArgumentParser("train")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--mode",
        default=None,
    )
    parser.add_argument("--model_initial", action="store_true")
    parser.add_argument(
        "--path_tensorboard",
        default=None,
    )
    parser.add_argument(
        "--path_new_model",
        default=None,
    )
    parser.add_argument(
        "--path_old_model",
        default=None,
    )
    args = parser.parse_args()

    if args.headless == False:
        config["env_para"]["headless"] = False
    config["model_para"]["mode"] = args.mode or config["model_para"]["mode"]
    config["model_para"]["model_initial"] = (
        args.model_initial or config["model_para"]["model_initial"]
    )
    config["model_para"]["path_tensorboard"] = (
        args.path_tensorboard or config["model_para"]["path_tensorboard"]
    )
    config["model_para"]["path_new_model"] = (
        args.path_new_model or config["model_para"]["path_new_model"]
    )
    if config["model_para"]["model_initial"]:
        config["model_para"]["path_old_model"] = (
            args.path_old_model or config["model_para"]["path_old_model"]
        )

    return config


if __name__ == "__main__":
    config_yaml = (Path(__file__).absolute().parent).joinpath("autodrive.yaml")
    with open(config_yaml, "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # Setup GPU
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices("GPU")
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
    else:
        warnings.warn(
            f"Not configured to use GPU or GPU not available.",
            ResourceWarning,
        )
        # raise SystemError("GPU device not found")

    config = replace_args(config)

    main(config=config)
