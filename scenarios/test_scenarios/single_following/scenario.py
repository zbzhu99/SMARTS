from pathlib import Path
from random import randrange

from smarts.core import seed
from smarts.sstudio import gen_scenario
from smarts.sstudio import types as t

seed(42)

start_offset = 0
route1 = t.Route(begin=('-E5',0,start_offset), end=('-E10',0,'max'))

SECS_PER_HOUR = 60.0 * 60.0
secs_btw_adding = 10
actor = t.TrafficActor(name="car",vehicle_type='passenger')

traffic = t.Traffic(
    flows=[
        t.Flow(
            route=route1,
            actors={actor:1},
            rate=SECS_PER_HOUR,
            begin=randrange(secs_btw_adding)
        )
    ]
)

gen_scenario(
    t.Scenario(
        traffic={'basic':traffic},
    ),
    output_dir=Path(__file__).parent,
)