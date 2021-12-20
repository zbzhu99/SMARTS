# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
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
"""This script provides a Python interface to generate scenario artifacts. This includes
route files (sumo \\*.rou.xml), missions, and bubbles.
"""

import collections
import itertools
import logging
import os
import pickle
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence, Tuple, Union

from . import types
from .generators import TrafficGenerator

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__file__)


def gen_scenario(
    scenario: types.Scenario,
    output_dir: Path,
    seed: int = 42,
    overwrite: bool = False,
):
    """This is now the preferred way to generate a scenario. Instead of calling the
    gen_* methods directly, we provide this higher-level abstraction that takes care
    of the sub-calls.
    """
    # XXX: For now this simply coalesces the sub-calls but in the future this allows
    #      us to simplify our serialization between SStudio and SMARTS.

    output_dir = str(output_dir)

    if scenario.traffic:
        for name, traffic in scenario.traffic.items():
            gen_traffic(
                scenario=output_dir,
                traffic=traffic,
                name=name,
                seed=seed,
                overwrite=overwrite,
            )

    if scenario.ego_missions:
        missions = []
        for mission in scenario.ego_missions:
            if isinstance(mission, types.GroupedLapMission):
                gen_group_laps(
                    scenario=output_dir,
                    begin=mission.route.begin,
                    end=mission.route.end,
                    grid_offset=mission.offset,
                    used_lanes=mission.lanes,
                    vehicle_count=mission.actor_count,
                    num_laps=mission.num_laps,
                    seed=seed,
                    overwrite=overwrite,
                )
            else:
                missions.append(mission)

        if missions:
            gen_missions(
                scenario=output_dir,
                missions=missions,
                seed=seed,
                overwrite=overwrite,
            )

    if scenario.social_agents:
        for name, actor_mission_pairs in scenario.social_agents.items():
            if not (isinstance(actor_mission_pairs, collections.abc.Sequence)):
                raise ValueError("Actor-mission pairs must be sequences")

            gen_social_agents(
                name=name,
                scenario=output_dir,
                social_actor_mission_pairs=actor_mission_pairs,
            )

    if scenario.bubbles:
        gen_bubbles(scenario=output_dir, bubbles=scenario.bubbles)

    if scenario.friction_maps:
        gen_friction_map(scenario=output_dir, surface_patches=scenario.friction_maps)

    if scenario.traffic_histories:
        # TODO:  pass in Sumo graph offset and use to offset history coordinates
        #    if sumo_road_network._graph._shifted_by_smarts: sumo_road_network._graph.getLocationOffset()
        gen_traffic_histories(
            scenario=output_dir,
            histories_datasets=scenario.traffic_histories,
            overwrite=overwrite,
        )


def gen_traffic(
    scenario: str,
    traffic: types.Traffic,
    name: str = None,
    output_dir: str = None,
    seed: int = 42,
    overwrite: bool = False,
):
    """Generates the traffic routes for the given scenario. If the output directory is
    not provided, the scenario directory is used. If name is not provided the default is
    "routes".
    """
    assert name != "missions", "The name 'missions' is reserved for missions!"

    output_dir = os.path.join(output_dir or scenario, "traffic")
    os.makedirs(output_dir, exist_ok=True)

    generator = TrafficGenerator(scenario, overwrite=overwrite)
    saved_path = generator.plan_and_save(traffic, name, output_dir, seed=seed)

    if saved_path:
        logger.debug(f"Generated traffic for scenario={scenario}")


def gen_social_agents(
    scenario: str,
    social_actor_mission_pairs: Sequence[Tuple[types.SocialAgentActor, types.Mission]],
    name: str,
    seed: int = 42,
    overwrite: bool = False,
):
    """Generates the social agent missions for the given scenario.

    Args:
        scenario:
            The scenario directory
        missions:
            A sequence of missions for social agents to perform
        social_agent_actor(s):
            The actor(s) to use
        name:
            A short name for this grouping of social agents. Is also used as the name
            of the social agent traffic file
        seed:
            The random seed to use when generating behaviour
        overwrite:
            If to forcefully write over the previous existing output file
    """

    # For backwards compatibility we support both a single value and a sequence
    social_actor_mission_pairs = social_actor_mission_pairs
    if not isinstance(social_actor_mission_pairs, collections.abc.Sequence):
        raise ValueError(f"{list(social_actor_mission_pairs)}")

    actors: Sequence[types.SocialAgentActor] = []
    missions: Sequence[types.Mission] = []
    for a, m in social_actor_mission_pairs:
        actors.append(a)
        missions.append(m)

    # This doesn't support BoidAgentActor. Here we make that explicit
    if any(isinstance(actor, types.BoidAgentActor) for actor in actors):
        raise ValueError(
            f"{gen_social_agents.__name__}(...) can't be called with BoidAgentActor, got:"
            f"{social_actor_mission_pairs}"
        )

    actor_names = [a.name for a in actors]
    if len(actor_names) != len(set(actor_names)):
        raise ValueError(f"Actor names={actor_names} must not contain duplicates")
    if len(missions) != len(set(missions)):
        raise ValueError(f"Supplied Missions must be unique.")

    output_dir = os.path.join(scenario, "social_agents")
    saved = _gen_missions(
        scenario=scenario,
        missions=missions,
        actors=actors,
        name=name,
        output_dir=output_dir,
        seed=seed,
        overwrite=overwrite,
    )

    if saved:
        logger.debug(f"Generated social agent missions for scenario={scenario}")


def gen_social_agent_missions(
    scenario: str,
    missions: Sequence[types.Mission],
    social_agent_actor: Union[types.SocialAgentActor, Sequence[types.SocialAgentActor]],
    name: str,
    seed: int = 42,
    overwrite: bool = False,
):
    """[DEPRECATED] Generates the social agent missions for the given scenario.

    Args:
        scenario:
            The scenario directory
        missions:
            A sequence of missions for social agents to perform
        social_agent_actor(s):
            The actor(s) to use
        name:
            A short name for this grouping of social agents. Is also used as the name
            of the social agent traffic file
        seed:
            The random seed to use when generating behaviour
        overwrite:
            If to forcefully write over the previous existing output file
    """

    logger.warn(
        DeprecationWarning(
            f"`{gen_social_agent_missions.__name__}` has been deprecated. Please use `{gen_social_agents.__name__}`"
        )
    )
    # For backwards compatibility we support both a single value and a sequence
    actors = social_agent_actor
    if not isinstance(actors, collections.abc.Sequence):
        actors = [actors]

    # This doesn't support BoidAgentActor. Here we make that explicit
    if any(isinstance(actor, types.BoidAgentActor) for actor in actors):
        raise ValueError(
            f"{gen_social_agent_missions.__name__}(...) can't be called with {types.BoidAgentActor.__name__}, got:"
            f"{actors}"
        )

    actor_names = [a.name for a in actors]
    if len(actor_names) != len(set(actor_names)):
        raise ValueError(f"Actor names={actor_names} must not contain duplicates")

    output_dir = os.path.join(scenario, "social_agents")
    saved = _gen_missions(
        scenario=scenario,
        missions=missions,
        actors=actors,
        name=name,
        output_dir=output_dir,
        seed=seed,
        overwrite=overwrite,
    )

    if saved:
        logger.debug(f"Generated social agent missions for scenario={scenario}")


def gen_missions(
    scenario: str,
    missions: Sequence,
    seed: int = 42,
    overwrite: bool = False,
):
    """Generates a route file to represent missions (a route per mission). Will create
    the output_dir if it doesn't exist already. The ouput file will be named `missions`.

    Args:
        scenario:
            The scenario directory
        missions:
            A sequence of missions for social agents to perform
        seed:
            The random seed to use when generating behaviour
        overwrite:
            If to forcefully write over the previous existing output file
    """

    saved = _gen_missions(
        scenario=scenario,
        missions=missions,
        actors=[types.TrafficActor(name="car")],
        name="missions",
        output_dir=scenario,
        seed=seed,
        overwrite=overwrite,
    )

    if saved:
        logger.debug(f"Generated missions for scenario={scenario}")


def gen_group_laps(
    scenario: str,
    begin: Tuple[str, int, Any],
    end: Tuple[str, int, Any],
    grid_offset: int,
    used_lanes: int,
    vehicle_count: int,
    num_laps: int = 3,
    seed: int = 42,
    overwrite: bool = False,
):
    """Generates missions that start with a grid offset at the startline and do a number
    of laps until finishing.

    Args:
        scenario:
            The scenario directory
        begin:
            The edge and offset of the first vehicle
        end:
            The edge and offset of the finish-line
        grid_offset:
            The F1 starting line staggered with offset disadvantage imposed per vehicle
        used_lanes:
            The number of lanes used for the starting-line from the innermost lane
        vehicle_count:
            The number of vehicles to use
        num_laps:
            The amount of laps before finishing
    """

    start_road_id, start_lane, start_offset = begin
    end_road_id, end_lane, end_offset = end

    missions = []
    for i in range(vehicle_count):
        s_lane = (start_lane + i) % used_lanes
        missions.append(
            types.LapMission(
                types.Route(
                    begin=(
                        start_road_id,
                        s_lane,
                        start_offset - grid_offset * i,
                    ),
                    end=(end_road_id, (end_lane + i) % used_lanes, end_offset),
                ),
                num_laps=num_laps,
                # route_length=route_length,
            )
        )

    saved = gen_missions(
        scenario=scenario, missions=missions, seed=seed, overwrite=overwrite
    )

    if saved:
        logger.debug(f"Generated grouped lap missions for scenario={scenario}")


def gen_bubbles(scenario: str, bubbles: Sequence[types.Bubble]):
    output_path = os.path.join(scenario, "bubbles.pkl")
    with open(output_path, "wb") as f:
        pickle.dump(bubbles, f)


def gen_friction_map(scenario: str, surface_patches: Sequence[types.RoadSurfacePatch]):
    """Generates friction map file according to the surface patches defined in
    scenario file.
    """
    output_path = os.path.join(scenario, "friction_map.pkl")
    with open(output_path, "wb") as f:
        pickle.dump(surface_patches, f)


def _gen_missions(
    scenario: str,
    missions: Sequence[types.Mission],
    actors: Sequence[types.Actor],
    name: str,
    output_dir: str,
    seed: int = 42,
    overwrite: bool = False,
):
    """Generates a route file to represent missions (a route per mission). Will
    create the output_dir if it doesn't exist already.
    """

    generator = TrafficGenerator(scenario)

    def resolve_mission(mission):
        route = getattr(mission, "route", None)
        kwargs = {}
        if route:
            kwargs["route"] = generator.resolve_route(route)

        via = getattr(mission, "via", ())
        if via is not ():
            kwargs["via"] = _resolve_vias(via, generator=generator)

        mission = replace(mission, **kwargs)

        return mission

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, name + ".pkl")

    if os.path.exists(output_path) and not overwrite:
        return False

    _validate_missions(missions)

    with open(output_path, "wb") as f:
        pickle.dump((actors, [resolve_mission(mission) for mission in missions]), f)

    return True


def _resolve_vias(via: Tuple[types.Via], generator):
    vias = [*via]
    for i in range(len(vias)):
        v = vias[i]
        if isinstance(v.road_id, types.JunctionEdgeIDResolver):
            vias[i] = replace(v, road_id=v.road_id.to_edge(generator.road_network))
    return tuple(vias)


def _validate_missions(missions):
    for mission in missions:
        _validate_entry_tactic(mission)


def _validate_entry_tactic(mission):
    if not mission.entry_tactic:
        return

    if isinstance(mission.entry_tactic, types.TrapEntryTactic):
        if not mission.entry_tactic.zone and not isinstance(
            mission.entry_tactic.zone, types.MapZone
        ):
            return

        z_edge, _, _ = mission.entry_tactic.zone.start
        if isinstance(mission, types.EndlessMission):
            edge, _, _ = mission.start
            assert (
                edge == z_edge
            ), f"Zone edge `{z_edge}` is not the same edge as `types.EndlessMission` start edge `{edge}`"

        elif isinstance(mission, (types.Mission, types.LapMission)):
            edge, _, _ = mission.route.begin
            assert (
                edge == z_edge
            ), f"Zone edge `{z_edge}` is not the same edge as `types.Mission` route begin edge `{edge}`"


def gen_traffic_histories(scenario: str, histories_datasets, overwrite: bool):
    # For SUMO maps, we need to check if the map was shifted and translate the vehicle positions if so
    xy_offset = None
    road_network_path = os.path.join(scenario, "map.net.xml")
    if os.path.exists(road_network_path):
        from smarts.core.sumo_road_network import SumoRoadNetwork

        road_network = SumoRoadNetwork.from_file(road_network_path)
        if road_network._graph and getattr(
            road_network._graph, "_shifted_by_smarts", False
        ):
            xy_offset = road_network._graph.getLocationOffset()

    genhistories_py = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "genhistories.py"
    )
    for hdsr in histories_datasets:
        hds = os.path.join(scenario, hdsr)
        if not os.path.exists(hds):
            raise ValueError(f"Traffic history dataset file missing: {hds}")
        cmd = [sys.executable, genhistories_py, hdsr]
        base, ext = os.path.splitext(os.path.basename(hds))
        if ext == ".json":
            logger.warn(
                """
                Converting old smarts JSON format history file.
                scenario.py should be updated with new YAML dataset spec.
                See SMARTS Issue #732."""
            )
            cmd += ["--old"]
        th_file = f"{base}.shf"
        if overwrite:
            cmd += ["-f"]
        if xy_offset:
            cmd += ["--x_offset", str(xy_offset[0])]
            cmd += ["--y_offset", str(xy_offset[1])]
        elif os.path.exists(os.path.join(scenario, th_file)):
            continue
        cmd += [th_file]
        subprocess.check_call(cmd, cwd=scenario)
