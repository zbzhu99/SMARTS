from pathlib import Path

from smarts.sstudio import gen_scenario
from smarts.sstudio import types as t

flow1 = [t.Flow(
            route=t.Route(
                begin=("-E5", lane, "random"), end=("-E10", lane, "max")
            ),
            rate=1,
            actors={t.TrafficActor("car"): 1},
            
        ) for lane in range(2)
        for _ in range(2)]

flow2 = [t.Flow(
            route=t.Route(
                begin=("-E6", lane, "random"), end=("E11", lane, "max")
            ),
            rate=1,
            actors={t.TrafficActor("car"): 1},
            
        ) for lane in range(2)
        for _ in range(1)]

flow3 = [t.Flow(
            route=t.Route(
                begin=("-E11", lane, "random"), end=("E6", lane, "max")
            ),
            rate=1,
            actors={t.TrafficActor("car"): 1},
            
        ) for lane in range(2)
        for _ in range(1)]

flow4 = [t.Flow(
            route=t.Route(
                begin=("E9", lane, "random"), end=("E4", lane, "max")
            ),
            rate=1,
            actors={t.TrafficActor("car"): 1},
            
        ) for lane in range(2)
        for _ in range(1)]

flow5 = [t.Flow(
            route=t.Route(
                begin=("-E4", lane, "random"), end=("-E9", lane, "max")
            ),
            rate=1,
            actors={t.TrafficActor("car"): 1},
            
        ) for lane in range(2)
        for _ in range(1)]

flow6 = [t.Flow(
            route=t.Route(
                begin=("E10", lane, "random"), end=("E5", lane, "max")
            ),
            rate=1,
            actors={t.TrafficActor("car"): 1},
            
        ) for lane in range(2)
        for _ in range(1)]

flow7 = [t.Flow(
            route=t.Route(
                begin=("-E8", lane, "random"), end=("-E7", lane, "max")
            ),
            rate=1,
            actors={t.TrafficActor("car"): 1},
            
        ) for lane in range(2)
        for _ in range(1)]

flow8 = [t.Flow(
            route=t.Route(
                begin=("E7", lane, "random"), end=("E8", lane, "max")
            ),
            rate=1,
            actors={t.TrafficActor("car"): 1},
            
        ) for lane in range(2)
        for _ in range(1)]

traffic = t.Traffic(
    flows=flow1
)

gen_scenario(
    scenario=t.Scenario(
        traffic={"basic": traffic},
    ),
    output_dir=Path(__file__).parent,
)
