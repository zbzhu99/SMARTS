import logging
from pathlib import Path

from smarts.sstudio import gen_scenario
from smarts.sstudio import types as t

from sys import path

path.append(str(Path(__file__).parent.parent))
from copy_scenario import copy_to_dir

scenario_map_file = "scenarios/curve_ahead_long_2e"

logger = logging.getLogger(str(Path(__file__)))

s_dir = str(Path(__file__).parent)

try:
    copy_to_dir(scenario_map_file, s_dir)
except Exception as e:
    logger.error(f"Scenario {scenario_map_file} failed to copy")
    raise e

base_ego_position = 200
ego_missions = [
    t.Mission(
        t.Route(
            begin=("curve_ahead", 0, base_ego_position),
            end=("curve_ahead", 1, "max"),
        ),
        via=[
            t.Via("curve_ahead", 0, 425, 20),
            t.Via("curve_ahead", 1, 440, 15),
        ],
    )
]

traffic = t.Traffic(
    flows=[
        t.Flow(
            route=t.Route(
                begin=("curve_ahead", 1, 240),
                end=("curve_ahead", 1, "max"),
            ),
            rate=1,
            actors={
                t.TrafficActor(
                    "target",
                    speed=t.Distribution(mean=1, sigma=0),
                    lane_changing_model=t.LaneChangingModel(
                        strategic=0, cooperative=0, keepRight=0
                    ),
                ): 1
            },
        )
    ]
)

scenario = t.Scenario(
    traffic={"all": traffic},
    ego_missions=ego_missions,
)

gen_scenario(scenario, output_dir=s_dir)
