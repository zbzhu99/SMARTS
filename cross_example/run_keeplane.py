# modified from examples/multi_agent.py
import gym
import argparse

from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.agent import AgentSpec, Agent
from smarts.core.utils.episodes import episodes

class KeepLaneAgent(Agent):
    def act(self, obs):
        return "keep_lane"


def main(scenarios, sim_name, headless, num_episodes, seed, max_episode_steps=None):
    # define agent spec
    agent_specs = {
        "AGENT-007": AgentSpec(
            interface=AgentInterface.from_type(
                AgentType.Laner, max_episode_steps=max_episode_steps
            ),
            agent_builder=KeepLaneAgent,
        )
    }

    # define env
    env = gym.make(
        "smarts.env:hiway-v0",
        scenarios=scenarios,
        agent_specs=agent_specs,
        sim_name=sim_name,
        headless=headless,
        seed=seed,
    )

    # build agent
    agents = {
        agent_id: agent_spec.build_agent()
        for agent_id, agent_spec in agent_specs.items()
    }

    # run simulation
    for episode in episodes(n=num_episodes):
        observations = env.reset()
        # episode.record_scenario(env.scenario_log)

        dones = {"__all__": False}
        while not dones["__all__"]:
            actions = {
                agent_id: agents[agent_id].act(agent_obs)
                for agent_id, agent_obs in observations.items()
            }

            observations, rewards, dones, infos = env.step(actions)
            # episode.record_step(observations, rewards, dones, infos)

    env.close()

def default_argument_parser(program: str):
    """This factory method returns a vanilla `argparse.ArgumentParser` with the
    minimum subset of arguments that should be supported.

    You can extend it with more `parser.add_argument(...)` calls or obtain the
    arguments via `parser.parse_args()`.
    """
    parser = argparse.ArgumentParser(program)
    parser.add_argument(
        "scenarios",
        help="A list of scenarios. Each element can be either the scenario to run "
        "(see scenarios/ for some samples you can use) OR a directory of scenarios "
        "to sample from.",
        type=str,
        nargs="+",
    )
    parser.add_argument(
        "--sim-name",
        help="a string that gives this simulation a name.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--headless", help="Run the simulation in headless mode.", action="store_true"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sumo-port", help="Run SUMO with a specified port.", type=int, default=None
    )
    parser.add_argument(
        "--episodes",
        help="The number of episodes to run the simulation for.",
        type=int,
        default=10,
    )
    return parser

if __name__ == "__main__":
    parser = default_argument_parser("multi-agent-example")
    args = parser.parse_args()

    main(
        scenarios=args.scenarios,
        sim_name=args.sim_name,
        headless=args.headless,
        num_episodes=args.episodes,
        seed=args.seed,
    )
