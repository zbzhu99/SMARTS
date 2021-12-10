from pathlib import Path

from smarts.sstudio.genscenario import gen_scenario
from smarts.sstudio.types import (
    Distribution,
    Flow,
    JunctionModel,
    LaneChangingModel,
    Mission,
    Route,
    Scenario,
    Traffic,
    TrafficActor,
)

impatient_car = TrafficActor(
    name="car",
    speed=Distribution(sigma=0.2, mean=1.0),
    lane_changing_model=LaneChangingModel(impatience=1, cooperative=0.25),
    junction_model=JunctionModel(
        drive_after_red_time=1.5, drive_after_yellow_time=1.0, impatience=1.0
    ),
)

patient_car = TrafficActor(
    name="car",
    speed=Distribution(sigma=0.2, mean=0.8),
    lane_changing_model=LaneChangingModel(impatience=0, cooperative=0.5),
    junction_model=JunctionModel(drive_after_yellow_time=1.0, impatience=0.5),
)

vertical_routes = [
    ("north-NS", "south-NS"),
    ("south-SN", "north-SN"),
]

horizontal_routes = [
    ("west-WE", "east-WE"),
    ("east-EW", "west-EW"),
]

turn_left_routes = [
    ("south-SN", "west-EW"),
    ("west-WE", "north-SN"),
    ("north-NS", "east-WE"),
    ("east-EW", "south-NS"),
]

turn_right_routes = [
    ("south-SN", "east-WE"),
    ("west-WE", "south-NS"),
    ("north-NS", "west-EW"),
    ("east-EW", "north-SN"),
]


traffic = {
    name: Traffic(
        flows=[
            Flow(
                route=Route(
                    begin=(f"edge-{r[0]}", 0, "random"),
                    end=(f"edge-{r[1]}", 0, "random"),
                ),
                rate=60 * 60,
                end=1e8 * 60 * 60,
                actors={impatient_car: 0.5, patient_car: 0.5},
            )
            for r in routes
        ]
    )
    for name, routes in {
        "vertical": vertical_routes,
        "horizontal": horizontal_routes,
        "unprotected_left": turn_left_routes,
        "turns": turn_left_routes + turn_right_routes,
        "all": vertical_routes
        + horizontal_routes
        + turn_left_routes
        + turn_right_routes,
    }.items()
}

ego_missions = [
    Mission(
        route=Route(begin=("edge-south-SN", 0, 10), end=("edge-west-EW", 0, "max")),
    ),
]

scnr_path = str(Path(__file__).parent)
scnr = Scenario(traffic=traffic, ego_missions=ego_missions)
gen_scenario(scenario=scnr, output_dir=scnr_path, seed=42, overwrite=True)
