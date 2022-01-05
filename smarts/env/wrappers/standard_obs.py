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

from smarts.core.events import Events
from smarts.core.sensors import DrivableAreaGridMap, Observation, EgoVehicleObservation, OccupancyGridMap, TopDownRGB
from typing import Any, DefaultDict, Dict, NamedTuple, Sequence

import gym
import numpy as np


class StandardObs(gym.ObservationWrapper):
    """Filters SMARTS environment observation and returns standard observations
    only.

    Observations


    """

    def __init__(self, env: gym.Env):
        """
        Args:
            env (gym.Env): SMARTS environment to be wrapped.
        """
        super().__init__(env)

        agent_id = next(iter(self.agent_specs.keys()))
        self.intrfcs = set()
        for intrfc in {
            "accelerometer",
            "drivable_area_grid_map",
            "rgb",
            "lidar",
            "neighborhood_vehicles",
            "ogm",
            "road_waypoints",
            "waypoints",
        }:
            val = getattr(self.agent_specs[agent_id].interface, intrfc)
            if val:
                self._comp_intrfc(intrfc, val)
                self.intrfcs.add(intrfc)

        self.std_obs = {
            'drivable_area_grid_map', 
            'ego_vehicle_state',
            'events',
            'lidar_point_cloud',
            'neighborhood_vehicle_states',
            'occupancy_grid_map',       
            'road_waypoints',
            'top_down_rgb',          
            'waypoint_paths',
        }

        self.observation_space = gym.spaces.Dict(
            {
                agent_id: gym.spaces.Dict({
                    'drivable_area_grid_map': gym.spaces.Box(low=0, high=255,shape=(self.agent_specs[agent_id].interface.drivable_area_grid_map.width, self.agent_specs[agent_id].interface.drivable_area_grid_map.height, 1), dtype=np.uint8),
                    'events': gym.spaces.Dict({
                        'agents_alive_done': gym.spaces.MultiBinary(1),
                        'collisions': gym.spaces.MultiBinary(1),
                        'not_moving': gym.spaces.MultiBinary(1),
                        'off_road': gym.spaces.MultiBinary(1),
                        'off_route': gym.spaces.MultiBinary(1),
                        'on_shoulder': gym.spaces.MultiBinary(1),
                        'reached_goal': gym.spaces.MultiBinary(1),
                        'reached_max_episode_steps': gym.spaces.MultiBinary(1),
                        'wrong_way': gym.spaces.MultiBinary(1),
                    }),
                    'occupancy_grid_map':gym.spaces.Box(low=0, high=255,shape=(self.agent_specs[agent_id].interface.ogm.width, self.agent_specs[agent_id].interface.ogm.height, 1), dtype=np.uint8),
                    'top_down_rgb':gym.spaces.Box(low=0, high=255,shape=(self.agent_specs[agent_id].interface.rgb.width, self.agent_specs[agent_id].interface.rgb.height, 3), dtype=np.uint8),
                })
                for agent_id in self.agent_specs.keys()
            }
        )

    def _comp_intrfc(self, intrfc: str, val: Any):
        assert all(
            getattr(self.agent_specs[agent_id].interface, intrfc) == val
            for agent_id in self.agent_specs.keys()
        ), f"To use StandardObs wrapper, all agents must have the same "
        f"AgentInterface.{intrfc} attribute."

    def observation(self, obs: Dict[str, Any]):
        from collections import defaultdict
        wrapped_obs = defaultdict(Observation)
        for agent_id, agent_obs in obs.items():
            for std_ob in self.std_obs:
                func = globals()[f"_std_{std_ob}"]
                func(getattr(agent_obs,std_ob))

        return obs


# def _make_std():


def _std_drivable_area_grid_map(val:DrivableAreaGridMap)->np.ndarray:
    return val.data.astype(np.uint8)

def _std_ego_vehicle_state(val)->NamedTuple: 
    return

def _std_events(val: Events)->Dict[str,int]: 
    return {"agents_alive_done": int(val.agents_alive_done),
        "collisions":int(len(val.collisions)>0),
        "not_moving":int(val.not_moving),
        "off_road":int(val.off_road),
        "off_route":int(val.off_route),
        "on_shoulder":int(val.on_shoulder),
        "reached_goal":int(val.reached_goal),
        "reached_max_episode_steps":int(val.reached_max_episode_steps),
        "wrong_way":int(val.wrong_way),
    }

def _std_lidar_point_cloud(val): 
    return val

def _std_neighborhood_vehicle_states(val):
    return val

def _std_occupancy_grid_map(val:OccupancyGridMap)->np.ndarray:
    return val.data.astype(np.uint8)

def _std_road_waypoints(val):
    return val
    
def _std_top_down_rgb(val:TopDownRGB)->np.ndarray: 
    return val.data.astype(np.uint8)

def _std_waypoint_paths(val):
    return val
