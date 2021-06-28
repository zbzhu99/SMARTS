import numpy as np
import os
import random
import signal
import sys
import yaml
from examples.gameOfTag import env as got_env
from examples.gameOfTag import evaluate
from examples.gameOfTag import agent as got_agent
from pathlib import Path


def train(config, save_interval=50, eval_interval=50):

    # Traning parameters
    discount_factor = config['model_para']["discount_factor"]
    num_epochs = config['model_para']["num_epochs"]
    batch_size = config['model_para']["batch_size"]
    num_episodes = config['model_para']['num_episodes']

    # Create env
    print("[INFO] Creating environments")
    seed = random.randint(0, 4294967295) # [0, 2^32 -1)
    env = got_env.TagEnv(config, seed)
    input_shape = env.observation_space.shape
    num_actions = env.action_space.shape[0]

    # Create agent
    print("[INFO] Creating agents")
    all_agents = {name:got_agent.TagAgent(name, config) for name in config["env_para"]["agent_ids"]}
    all_predators_id = env.predators
    all_preys_id = env.preys

    # Create model
    print("[INFO] Creating model")
    model_predator = got_agent.TagModel(config, "predator")
    model_prey = got_agent.TagModel(config, "prey")

    def interrupt(*args):
        model_predator.save()
        model_prey.save()
        print("Interrupt key detected.") 
        sys.exit(0)

    # Catch keyboard interrupt and terminate signal
    signal.signal(signal.SIGINT, interrupt)

    print("[INFO] Training loop")
    for episode in num_episodes:
        # While there are running environments
        
        obs = env.reset()
        [agent.reset() for _, agent in all_agents.items()]
        active_agents = set([agent_id for agent_id, _ in all_agents.items()])

        # Simulate for one episode
        while True:
            # Predict and value action given state
            # π(a_t | s_t; θ_old)
            actions_t={}
            values_t={}
            obs = env.step()
            actions_t_predator, values_t_predator = model_predator.act(obs)
            actions_t_prey, values_t_prey = model_prey.act(obs)
            actions_t.update(actions_t_predator)
            actions_t.update(actions_t_prey)
            values_t.update(values_t_predator)
            values_t.update(values_t_prey)

            # Sample action from a Gaussian distribution
            states_t, rewards_t, dones_t, _ = env.step(actions_t)

            # Store state, action and reward
            for agent in active_agents:
                agent.add_trajectory(
                    state=states_t[agent.name],
                    action=actions_t[agent.name],
                    value=np.squeeze(values_t[agent.name], axis=-1),
                    reward=rewards_t[agent.name],
                    done=dones_t[agent.name],
                )

            for agent_id, done in dones_t:
                if done:
                    # Remove done agents
                    active_agents.remove(agent_id)
                    # Calculate last values (bootstrap values)
                    if 'predator' in agent_id:
                        _, values_t = model_predator.act(states_t)
                    elif 'prey' in agent_id:
                        _, values_t = model_prey.act(states_t)
                    else:
                        raise Exception(f"Unknown {agent_id}.")
                    # Store last values    
                    all_agents[agent_id].store_last_values(np.squeeze(values_t, axis=-1))

            # Break when episode completes
            if dones_t['__all__']:
                break

        # Compute generalised advantage
        [agent.compute_gae() for _, agent in all_agents.items()]

        states_predator = [np.array(all_agents[agent_id].states) for agent_id in all_predators_id]
        states_predator = got_agent.stack_vars(states_predator)
        states_prey = [np.array(all_agents[agent_id].states) for agent_id in all_preys_id]
        states_prey = got_agent.stack_vars(states_prey)

        actions_predator = [np.array(all_agents[agent_id].actions) for agent_id in all_predators_id]
        actions_predator = got_agent.stack_vars(actions_predator)
        actions_prey = [np.array(all_agents[agent_id].actions) for agent_id in all_preys_id]
        actions_prey = got_agent.stack_vars(actions_prey)

        returns_predator = [all_agents[agent_id].returns for agent_id in all_predators_id]
        returns_predator = got_agent.stack_vars(returns_predator)
        returns_prey = [all_agents[agent_id].returns for agent_id in all_preys_id]
        returns_prey = got_agent.stack_vars(returns_prey)

        advantages_predator = [all_agents[agent_id].advantages for agent_id in all_predators_id]
        advantages_predator = got_agent.stack_vars(advantages_predator)
        advantages_prey = [all_agents[agent_id].advantages for agent_id in all_preys_id]
        advantages_prey = got_agent.stack_vars(advantages_prey)


        T = len(returns_prey)
        N = 1
        assert states_prey.shape == (T * N, *input_shape)
        assert actions_prey.shape == (T * N, num_actions)
        assert returns_prey.shape == (T * N,)
        assert advantages_prey.shape == (T * N,)

        # Train for some number of epochs
        model_predator.update_old_policy()  # θ_old <- θ
        model_prey.update_old_policy()  # θ_old <- θ

        # Train predator
        for _ in range(num_epochs):
            num_samples = len(states_predator)
            indices = np.arange(num_samples)
            np.random.shuffle(indices)
            for i in range(int(np.ceil(num_samples / batch_size))):
                # Sample mini-batch randomly
                begin = i * batch_size
                end = begin + batch_size
                if end > num_samples:
                    end = None
                mb_idx = indices[begin:end]

                # Optimize network
                model_predator.train(states_predator[mb_idx], actions_predator[mb_idx],
                            returns_predator[mb_idx], advantages_predator[mb_idx], 
                            learning_rate=config['model_para']['initial_lr_predator'])

        # Train prey
        for _ in range(num_epochs):
            num_samples = len(states_prey)
            indices = np.arange(num_samples)
            np.random.shuffle(indices)
            for i in range(int(np.ceil(num_samples / batch_size))):
                # Sample mini-batch randomly
                begin = i * batch_size
                end = begin + batch_size
                if end > num_samples:
                    end = None
                mb_idx = indices[begin:end]

                # Optimize network
                model_prey.train(states_prey[mb_idx], actions_prey[mb_idx],
                            returns_prey[mb_idx], advantages_prey[mb_idx], 
                            learning_rate=config['model_para']['initial_lr_prey'])

        # Evaluate model
        if episode % eval_interval == 0:
            print("[INFO] Running evaluation...")
            avg_reward_predator, avg_reward_prey, value_error_predator, value_error_prey = evaluate.evaluate(
                model_predator, model_prey, config, discount_factor)
            model_predator.write_to_summary("eval_avg_reward", avg_reward_predator)
            model_predator.write_to_summary("eval_value_error", value_error_predator)
            model_prey.write_to_summary("eval_avg_reward", avg_reward_prey)
            model_prey.write_to_summary("eval_value_error", value_error_prey)

        # Save model
        if episode % save_interval == 0:
            model_predator.save()
            model_prey.save()


if __name__ == "__main__":
    config_yaml=(Path(__file__).absolute().parent).joinpath("smarts.yaml")
    with open(config_yaml, "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # Silence the logs of TF
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    train(
        config=config, 
        save_interval=config['model_para']['save_interval'],
        eval_interval=config['model_para']['eval_interval'],
    )
