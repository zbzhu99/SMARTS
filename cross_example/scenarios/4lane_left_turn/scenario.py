import os
from pathlib import Path

from smarts.sstudio.genscenario import gen_scenario
from smarts.sstudio.types import (
    CutIn,
    EndlessMission,
    Flow,
    JunctionEdgeIDResolver,
    Mission,
    RandomRoute,
    Route,
    Scenario,
    Traffic,
    TrafficActor,
    Via,
)

social_vehicle_num = 10

ego_missions = [
    Mission(
        route=Route(begin=("edge-south-SN", 1, 20), end=("edge-west-EW", 1, "max")),
    ),
]

scenario = Scenario(
    traffic={
        "basic": Traffic(
            flows=[
                Flow(
                    route=RandomRoute(),
                    rate=1,
                    actors={TrafficActor(name="car"): 1.0},
                )
                for i in range(social_vehicle_num)
            ]
        )
    },
    ego_missions=ego_missions,
)

gen_scenario(
    scenario=scenario, output_dir=Path(__file__).parent, ovewrite=True
)
