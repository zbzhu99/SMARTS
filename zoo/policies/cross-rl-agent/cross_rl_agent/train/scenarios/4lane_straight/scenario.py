from pathlib import Path

from smarts.sstudio.genscenario import gen_scenario
from smarts.sstudio.types import (
    Distribution,
    Flow,
    JunctionModel,
    LaneChangingModel,
    Mission,
    RandomRoute,
    Route,
    Scenario,
    Traffic,
    TrafficActor,
)

social_vehicle_num = 100

ego_missions = [
    Mission(
        route=Route(begin=("edge-south-SN", 1, 10), end=("edge-north-SN", 1, 8)),
    ),
]

stright_traffic_actor = TrafficActor(
    name="car",
    speed=Distribution(sigma=0.2, mean=1),
    lane_changing_model=LaneChangingModel(impatience=0),
    junction_model=JunctionModel(
        drive_after_red_time=1.5, drive_after_yellow_time=1.0, impatience=0.5
    ),
)
scenario = Scenario(
    traffic={
        "basic": Traffic(
            flows=[
                Flow(
                    route=RandomRoute(),
                    rate=1,
                    actors={stright_traffic_actor: 1.0},
                )
                for i in range(social_vehicle_num)
            ]
        )
    },
    ego_missions=ego_missions,
)

gen_scenario(scenario=scenario, output_dir=Path(__file__).parent)
