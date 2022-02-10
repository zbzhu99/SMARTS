# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import pathlib
from typing import Optional

from smarts.core.agent import AgentSpec
from smarts.core.agent_interface import (
    OGM,
    RGB,
    AgentInterface,
    DoneCriteria,
    DrivableAreaGridMap,
    NeighborhoodVehicles,
    Waypoints,
)
from smarts.core.controllers import ActionSpaceType
from smarts.env import build_scenario
from smarts.env.hiway_env import HiWayEnv
from smarts.env.wrappers.format_obs import FormatObs
from smarts.env.wrappers.single_agent import SingleAgent


def intersection_env(
    headless: bool = True,
    visdom: bool = False,
    sumo_headless: bool = True,
    envision_record_data_replay_path: Optional[str] = None,
):
    """An intersection environment where a single agent needs to make an
    unprotected left turn in the presence of traffic and without traffic
    lights. Traffic vehicles stop before entering the junction.

    Observation:
        A `smarts.env.wrappers.format_obs.StdObs` dataclass, with all fields
        enabled, is returned as observation.

    Actions:
        Type: gym.spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )

        Action     Value range
        Throttle   [ 0, 1]
        Brake      [ 0, 1]
        Steering   [-1, 1]

    Reward:
        Reward is distance travelled (in meters) in each step, including the
        termination step.

    Episode termination:
        Episode is terminated if any of the following is fulfilled.
        Steps per episode exceed 3000.
        Agent collides, drives off road, drives off route, or drives on shoulder.

    Solved requirement:
        Considered solved when the average `info["score"]` is greater than or
        equal to 90.0 over 100 consecutive trials.

    Args:
        headless (bool, optional): If True, disables visualization in
            Envision. Defaults to False.
        visdom (bool, optional): If True, enables visualization of observed
            RGB images in Visdom. Defaults to False.
        sumo_headless (bool, optional): If True, disables visualization in
            SUMO GUI. Defaults to True.
        envision_record_data_replay_path (Optional[str], optional):
            Envision's data replay output directory. Defaults to None.

    Returns:
        A single-agent unprotected left turn intersection environment.
    """

    scenario = [
        str(
            pathlib.Path(__file__).absolute().parents[2]
            / "scenarios"
            / "intersections"
            / "2lane_left_turn"
        )
    ]
    build_scenario(scenario)

    done_criteria = DoneCriteria(
        collision=False,  # Temporary for testing purpose ********
        off_road=True,
        off_route=True,
        on_shoulder=True,
        wrong_way=False,
        not_moving=False,
        agents_alive=None,
    )
    max_episode_steps = 3000
    img_meters = 100
    img_pixels = 256
    agent_specs = {
        "LeftTurnAgent": AgentSpec(
            interface=AgentInterface(
                accelerometer=True,
                # action=ActionSpaceType.Continuous, # Temporary for testing purpose ********
                action=ActionSpaceType.LaneWithContinuousSpeed,  # Temporary for testing purpose ********
                done_criteria=done_criteria,
                drivable_area_grid_map=DrivableAreaGridMap(
                    width=img_pixels,
                    height=img_pixels,
                    resolution=img_meters / img_pixels,
                ),
                lidar=True,
                max_episode_steps=max_episode_steps,
                neighborhood_vehicles=NeighborhoodVehicles(img_meters),
                ogm=OGM(
                    width=img_pixels,
                    height=img_pixels,
                    resolution=img_meters / img_pixels,
                ),
                rgb=RGB(
                    width=img_pixels,
                    height=img_pixels,
                    resolution=img_meters / img_pixels,
                ),
                road_waypoints=False,
                waypoints=Waypoints(lookahead=img_meters),
            ),
        )
    }

    env = HiWayEnv(
        scenarios=scenario,
        agent_specs=agent_specs,
        sim_name="LeftTurn",
        headless=headless,
        visdom=visdom,
        sumo_headless=sumo_headless,
        endless_traffic=False,
        envision_record_data_replay_path=envision_record_data_replay_path,
    )
    env = FormatObs(env=env)
    env = SingleAgent(env=env)

    return env
