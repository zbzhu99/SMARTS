from pathlib import Path
from typing import Dict

import examples.auto_drive.env.adapter as adapter
from smarts.core import agent as smarts_agent
from smarts.core import agent_interface as smarts_agent_interface
from smarts.core import controllers as smarts_controllers
from smarts.env import hiway_env as smarts_hiway_env
from smarts.env.wrappers import action as smarts_action
from smarts.env.wrappers import rgb_image as smarts_rgb_image
from smarts.env.wrappers import single_agent as smarts_single_agent


def make_single_agent_env(config: Dict, seed: int):

    vehicle_interface = smarts_agent_interface.AgentInterface(
        max_episode_steps=config["max_episode_steps"],
        neighborhood_vehicles=smarts_agent_interface.NeighborhoodVehicles(
            radius=config["neighborhood_radius"]
        ),
        rgb=smarts_agent_interface.RGB(
            width=config["rgb_pixels"], height=config["rgb_pixels"], resolution=config["rgb_meters"] / config["rgb_pixels"]
        ),
        vehicle_color="BrightRed",
        action=getattr(
            smarts_controllers.ActionSpaceType,
            config["action_space_type"],
        ),
        done_criteria=smarts_agent_interface.DoneCriteria(
            collision=True,
            off_road=True,
            off_route=False,
            on_shoulder=False,
            wrong_way=False,
            not_moving=False,
        ),
    )

    agent_specs = {
        agent_id: smarts_agent.AgentSpec(
            interface=vehicle_interface,
            agent_builder=None,
            reward_adapter=adapter.reward_adapter,
            info_adapter=adapter.info_adapter,
        )
        for agent_id in config["agent_ids"]
    }

    base = (Path(__file__).absolute().parents[3]).joinpath("scenarios")
    scenarios = [str(base.joinpath(scenario)) for scenario in config["scenarios"]]
    env = smarts_hiway_env.HiWayEnv(
        scenarios=scenarios,
        agent_specs=agent_specs,
        headless=config["headless"],
        visdom=config["visdom"],
        seed=seed,
    )
    # Wrap env with ActionWrapper
    env = smarts_action.Action(env=env, wrapper=config["action_adapter"])
    # Wrap env with RGBImage wrapper to only get rgb images in observation
    env = smarts_rgb_image.RGBImage(env=env, num_stack=1)
    # Wrap env with SingleAgent wrapper to be Gym compliant
    env = smarts_single_agent.SingleAgent(env=env)

    return env
