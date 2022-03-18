# MIT License
#
# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
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
from pathlib import Path

import smarts.sstudio.types as t
from smarts.sstudio import gen_scenario

missions = [
    t.Mission(t.Route(begin=("-gneE1", 0, 5), end=("-gneE1", 0, 97))),
    t.Mission(t.Route(begin=("gneE1", 0, 5), end=("gneE1", 0, 97))),
    t.Mission(t.Route(begin=("-gneE1", 0, 20), end=("-gneE1", 0, 97))),
    t.Mission(t.Route(begin=("gneE1", 0, 20), end=("gneE1", 0, 97))),
]


impatient_car = t.TrafficActor(
    name="car",
    speed=t.Distribution(sigma=0.2, mean=1.0),
    model_overrides=(
        t.SumoVTypeOverride(
            lcImpatience=1,
            lcCooperative=0.25,
            jmDriveAfterRedTime=1.5,
            jmDriveAfterYellowTime=1.0,
            impatience=1.0,
        ),
    ),
)

patient_car = t.TrafficActor(
    name="car",
    speed=t.Distribution(sigma=0.2, mean=0.8),
    model_overrides=(
        t.SumoVTypeOverride(
            lcImpatience=0,
            lcCooperative=0.5,
            jmDriveAfterYellowTime=1.0,
            impatience=0.5,
        ),
    ),
)

traffic = {
    "1": t.Traffic(
        flows=[
            t.Flow(
                route=t.Route(
                    begin=(f"-gneE1", 0, 35),
                    end=(f"-gneE1", 0, 100),
                ),
                rate=1,
                actors={impatient_car: 0.5, patient_car: 0.5},
            )
        ]
    ),
    "2": t.Traffic(
        flows=[
            t.Flow(
                route=t.Route(
                    begin=(f"gneE1", 0, 35),
                    end=(f"gneE1", 0, 100),
                ),
                rate=1,
                actors={impatient_car: 0.5, patient_car: 0.5},
            )
        ]
    ),
}


gen_scenario(
    t.Scenario(ego_missions=missions, traffic=traffic),
    output_dir=Path(__file__).parent,
)
