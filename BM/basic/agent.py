from smarts.core.controllers import ActionSpaceType
from checker_cutin import CutinChecker
from check_uturn import UTurnChecker
from checker_host import CheckerConfig, CheckerHost
from examples.single_agent import UTurnAgent
import logging
import os
import gym

from examples import default_argument_parser
from smarts.core.agent import Agent, AgentSpec
from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.sensors import Observation
from smarts.core.utils.episodes import episodes

logging.basicConfig(level=logging.INFO)

AGENT_ID = "Agent-007"


class ChaseViaPointsAgent(Agent):
    def act(self, obs: Observation):
        if (
            len(obs.via_data.near_via_points) < 1
            or obs.ego_vehicle_state.edge_id != obs.via_data.near_via_points[0].edge_id
        ):
            return (obs.waypoint_paths[0][0].speed_limit, 0)

        nearest = obs.via_data.near_via_points[0]
        if nearest.lane_index == obs.ego_vehicle_state.lane_index:
            return (nearest.required_speed, 0)

        return (
            nearest.required_speed,
            1 if nearest.lane_index > obs.ego_vehicle_state.lane_index else -1,
        )


def main(scenarios, sim_name, headless, num_episodes, seed, max_episode_steps=None):
    # agent_spec = AgentSpec(
    #     interface=AgentInterface(
    #         max_episode_steps=max_episode_steps,
    #         waypoints=True,
    #         action=ActionSpaceType.LaneWithContinuousSpeed,
    #         neighborhood_vehicles=True,
    #     ),
    #     agent_builder=ChaseViaPointsAgent,
    # )
    agent_spec = AgentSpec(
        interface=AgentInterface.from_type(
            AgentType.StandardWithAbsoluteSteering,
            max_episode_steps=max_episode_steps
        ),
        agent_builder=UTurnAgent,
    )
    env = gym.make(
        "smarts.env:hiway-v0",
        scenarios=scenarios,
        agent_specs={AGENT_ID: agent_spec},
        sim_name=sim_name,
        headless=headless,
        visdom=False,
        timestep_sec=0.1,
        sumo_headless=False,
        sumo_auto_start=False,
        seed=seed,
    )
    UTurnAgent.sim = env._smarts
    for episode in episodes(n=num_episodes):
        agent = agent_spec.build_agent()
        observations = env.reset()
        episode.record_scenario(env.scenario_log)
        logger = logging.getLogger(
            # env.scenario_log["scenario_map"]
        )
        ch: CheckerHost = CheckerHost(env._smarts, logger)
        checker = UTurnChecker(bm_id=AGENT_ID, target_id="target")
        # checker = CutinChecker(bm_id=AGENT_ID, target_id="target")
        ch.add_checkers(CheckerConfig(checker))

        dones = {"__all__": False}
        while not dones["__all__"]:  # and not ch.done:
            agent_obs = observations[AGENT_ID]
            agent_action = agent.act(agent_obs)
            observations, rewards, dones, infos = env.step({AGENT_ID: agent_action})
            episode.record_step(observations, rewards, dones, infos)
            ch.record_step(observations, rewards, dones, infos)

        ch.conclude()

    env.close()


if __name__ == "__main__":
    parser = default_argument_parser("single-agent-example")
    args = parser.parse_args()

    main(
        scenarios=args.scenarios,
        sim_name=args.sim_name,
        headless=args.headless,
        num_episodes=args.episodes,
        seed=args.seed,
    )
