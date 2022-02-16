from pathlib import Path

from smarts.sstudio.genscenario import gen_scenario
from smarts.sstudio.types import (
    Flow,
    MapZone,
    Mission,
    Route,
    Scenario,
    Traffic,
    TrafficActor,
    TrapEntryTactic,
)

scnr_path = str(Path(__file__).parent)

intersection_car = TrafficActor(
    name="car",
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

traffic = {}
for name, routes in {
    "vertical": vertical_routes,
    "horizontal": horizontal_routes,
    "turn_left": turn_left_routes,
    "turn_right": turn_right_routes,
    "turns": turn_left_routes + turn_right_routes,
    "all": vertical_routes + horizontal_routes + turn_left_routes + turn_right_routes,
}.items():
    traffic[name] = Traffic(
        flows=[
            Flow(
                route=Route(
                    begin=(f"edge-{r[0]}", 0, "random"),
                    end=(f"edge-{r[1]}", 0, "max"),
                ),
                rate=60 * 4,
                begin=i * 5,
                end=60 * 30,
                # Note: For an episode with maximum_episode_steps=3000 and step
                # time=0.1s, maximum episode time=300s. Hence, traffic set to
                # end at 1800s, which is greater than maximum episode time of
                # 300s.
                actors={intersection_car: 1},
            )
            for i, r in enumerate(routes)
        ]
    )

route = Route(begin=("edge-west-WE", 0, 55), end=("edge-north-SN", 0, 40))
ego_missions = [
    Mission(
        route=route,
        entry_tactic=TrapEntryTactic(
            wait_to_hijack_limit_s=3,
            zone=MapZone(
                start=(
                    route.begin[0],
                    route.begin[1],
                    route.begin[2] - 5,
                ),
                length=10,
                n_lanes=1,
            ),
            default_entry_speed=5,
        ),
    ),
]

gen_scenario(
    scenario=Scenario(
        traffic=traffic,
        ego_missions=ego_missions,
    ),
    output_dir=scnr_path,
)
