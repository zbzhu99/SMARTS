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
import logging
import os
from functools import lru_cache
from subprocess import check_output
from typing import List, Optional, Sequence, Set, Tuple
from waymo_open_dataset.protos import scenario_pb2
from enum import Enum
import numpy as np
from cached_property import cached_property

from smarts.sstudio.types import MapSpec

from .coordinates import BoundingBox, Heading, Point, Pose, RefLinePoint
from .lanepoints import LanePoints, LinkedLanePoint
from .road_map import RoadMap, Waypoint
from .utils.file import read_tfrecord_file


def convert_polyline(polyline) -> Tuple[List[float], List[float]]:
    xs, ys = [], []
    for p in polyline:
        xs.append(p.x)
        ys.append(p.y)
    return xs, ys


class RoadLineType(Enum):
    UNKNOWN = 0
    BROKEN_SINGLE_WHITE = 1
    SOLID_SINGLE_WHITE = 2
    SOLID_DOUBLE_WHITE = 3
    BROKEN_SINGLE_YELLOW = 4
    BROKEN_DOUBLE_YELLOW = 5
    SOLID_SINGLE_YELLOW = 6
    SOLID_DOUBLE_YELLOW = 7
    PASSING_DOUBLE_YELLOW = 8

    @staticmethod
    def is_road_line(line):
        return True if line.__class__ == RoadLineType else False

    @staticmethod
    def is_yellow(line):
        return True if line in [
            RoadLineType.SOLID_DOUBLE_YELLOW, RoadLineType.PASSING_DOUBLE_YELLOW, RoadLineType.SOLID_SINGLE_YELLOW,
            RoadLineType.BROKEN_DOUBLE_YELLOW, RoadLineType.BROKEN_SINGLE_YELLOW
        ] else False

    @staticmethod
    def is_broken(line):
        return True if line in [
            RoadLineType.BROKEN_DOUBLE_YELLOW, RoadLineType.BROKEN_SINGLE_YELLOW, RoadLineType.BROKEN_SINGLE_WHITE
        ] else False


class RoadEdgeType(Enum):
    UNKNOWN = 0
    # Physical road boundary that doesn't have traffic on the other side (e.g., a curb or the k-rail on the right side of a freeway).
    BOUNDARY = 1
    # Physical road boundary that separates the car from other traffic (e.g. a k-rail or an island).
    MEDIAN = 2

    @staticmethod
    def is_road_edge(edge):
        return True if edge.__class__ == RoadEdgeType else False

    @staticmethod
    def is_sidewalk(edge):
        return True if edge == RoadEdgeType.BOUNDARY else False


class WaymoMap(RoadMap):
    """A map associated with a Waymo dataset"""

    def __init__(self, map_spec: MapSpec):
        self._log = logging.getLogger(self.__class__.__name__)
        self._map_spec = map_spec
        self._surfaces = {}
        self._lanes = {}
        self._roads = {}
        # self._waypoints_cache = TODO
        self._lanepoints = None
        if map_spec.lanepoint_spacing is not None:
            assert map_spec.lanepoint_spacing > 0
            # XXX: this should be last here since LanePoints() calls road_network methods immediately
            self._lanepoints = LanePoints.from_waymo(
                self, spacing=map_spec.lanepoint_spacing
            )

    @staticmethod
    def _load(path, scenario_id):
        scenario = None
        dataset = read_tfrecord_file(path)
        for record in dataset:
            parsed_scenario = scenario_pb2.Scenario()
            parsed_scenario.ParseFromString(bytearray(record))
            if parsed_scenario.scenario_id == scenario_id:
                scenario = parsed_scenario
                break

        if scenario is None:
            errmsg = f"Dataset file does not contain scenario with id: {scenario_id}"
            raise ValueError(errmsg)

        features = {"lane": [], "road_line": [], "road_edge": [], "stop_sign": [], "crosswalk": [],
                    "speed_bump": []}

        lanes = []
        for i in range(len(scenario.map_features)):
            map_feature = scenario.map_features[i]
            key = map_feature.WhichOneof("feature_data")
            if key is not None:
                features[key].append(getattr(map_feature, key))

        return scenario.scenario_id, features, lanes

    @classmethod
    def from_spec(cls, map_spec: MapSpec):
        """Generate a road network from the given map specification."""
        pass  # TODO

    @property
    def source(self) -> str:
        pass  # TODO

    def is_same_map(self, map_spec: MapSpec) -> bool:
        """Test if the road network is identical to the given map specification."""
        pass  # TODO

    @cached_property
    def bounding_box(self) -> BoundingBox:
        """Get the minimal axis aligned bounding box that contains all map geometry."""
        pass  # TODO

    @property
    def scale_factor(self) -> float:
        """Get the scale factor between the default lane width and the default lane width."""
        pass  # TODO

    def to_glb(self, at_path):
        """Build a glb file for camera rendering and envision"""
        pass  # TODO (or not!)

    class Surface(RoadMap.Surface):
        """Describes a surface."""

        def __init__(self, surface_id: str, road_map):
            self._surface_id = surface_id
            self._map = road_map

        @property
        def surface_id(self) -> str:
            """The identifier for this surface."""
            return self._surface_id

        @property
        def is_drivable(self) -> bool:
            """If it is possible to drive on this surface."""
            # TODO
            return True

    def surface_by_id(self, surface_id: str) -> RoadMap.Surface:
        """Find a surface by its identifier."""
        return self._surfaces.get(surface_id)

    # TODO, etc.
