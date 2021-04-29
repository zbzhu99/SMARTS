import logging
from pathlib import Path

from smarts.sstudio import gen_scenario
from smarts.sstudio import types as t

from sys import path

path.append(str(Path(__file__).parent.parent))
from copy_scenario import copy_to_dir

scenario_map_file = "scenarios/straightaway_2lane"

logger = logging.getLogger(str(Path(__file__)))

s_dir = str(Path(__file__).parent)

try:
    copy_to_dir(scenario_map_file, s_dir)
except Exception as e:
    logger.error(f"Scenario {scenario_map_file} failed to copy")
    raise e


base_offset = 40
ego_missions = [
    t.Mission(
        route=t.Route(
            begin=("straightaway", 1, 1),
            end=("straightaway", 0, "max"),
        ),
        via=[
            t.Via("straightaway", 1, 120 + base_offset, 20),
            t.Via("straightaway", 0, 140 + base_offset, 15),
            t.Via("straightaway", 0, 160 + base_offset, 7.5),
        ],
    )
]

traffic = t.Traffic(
    flows=[
        t.Flow(
            route=t.Route(
                begin=("straightaway", 0, 0 + base_offset),
                end=("straightaway", 0, "max"),
            ),
            rate=1,
            actors={t.TrafficActor("car", speed=t.Distribution(mean=1, sigma=0)): 1},
        ),
        t.Flow(
            route=t.Route(
                begin=("straightaway", 1, 50 + base_offset),
                end=("straightaway", 1, "max"),
            ),
            rate=1,
            actors={
                t.TrafficActor(
                    "left_lane_hog",
                    speed=t.Distribution(mean=0.80, sigma=0),
                    lane_changing_model=t.LaneChangingModel(
                        strategic=0, cooperative=0, keepRight=0
                    ),
                ): 1
            },
        ),
        t.Flow(
            route=t.Route(
                begin=("straightaway", 0, 40 + base_offset),
                end=("straightaway", 0, "max"),
            ),
            rate=1,
            actors={
                t.TrafficActor(
                    "forward_car",
                    speed=t.Distribution(mean=1, sigma=0),
                    lane_changing_model=t.LaneChangingModel(
                        strategic=0, cooperative=0, keepRight=0
                    ),
                ): 1
            },
        ),
    ]
)

scenario = t.Scenario(
    traffic={"all": traffic},
    ego_missions=ego_missions,
)

gen_scenario(scenario, output_dir=s_dir)
