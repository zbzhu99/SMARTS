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
import logging
import time
from typing import Dict, List, Tuple
from dataclasses import dataclass
import math
import numpy as np
from lxml import etree
from opendrive2lanelet.opendriveparser.elements.opendrive import (
    OpenDrive as OpenDriveElement,
)
from opendrive2lanelet.opendriveparser.elements.geometry import Line as LineGeometry
from opendrive2lanelet.opendriveparser.elements.road import Road as RoadElement
from opendrive2lanelet.opendriveparser.elements.roadLanes import Lane as LaneElement
from opendrive2lanelet.opendriveparser.elements.roadLanes import (
    LaneSection as LaneSectionElement,
)
from opendrive2lanelet.opendriveparser.elements.roadPlanView import (
    PlanView as PlanViewElement,
)
from opendrive2lanelet.opendriveparser.parser import parse_opendrive

from smarts.core.road_map import RoadMap
from smarts.core.utils.math import CubicPolynomial, constrain_angle


@dataclass
class LaneBoundary:
    refline: PlanViewElement
    inner: "LaneBoundary"
    poly: CubicPolynomial

    def refline_to_linear_segments(self, s_start: float, s_end: float) -> List[float]:
        s_vals = []
        for geom in self.refline._geometries:
            if type(geom) == LineGeometry:
                s_vals.extend([s_start, s_end])
            else:
                num_segments = int((s_end - s_start) / 0.1)
                for seg in range(num_segments):
                    s_vals.append(seg * 0.1)
        return sorted(s_vals)

    def calc_t(self, ds: float) -> float:
        if not self.inner:
            return 0
        return self.poly.eval(ds) + self.inner.calc_t(ds)

    def to_linear_segments(self, s_start: float, s_end: float):
        if self.inner:
            inner_s_vals = self.inner.to_linear_segments(s_start, s_end)
        else:
            return self.refline_to_linear_segments(s_start, s_end)

        if self.poly.c == 0 and self.poly.d == 0:
            outer_s_vals = [s_start, s_end]
        else:
            num_segments = int((s_end - s_start) / 0.1)
            outer_s_vals = []
            for seg in range(num_segments):
                outer_s_vals.append(seg * 0.1)
        return sorted(set(inner_s_vals + outer_s_vals))


class OpenDriveRoadNetwork(RoadMap):
    def __init__(self, xodr_file: str):
        self._log = logging.getLogger(self.__class__.__name__)
        self._log.setLevel(logging.INFO)
        self._xodr_file = xodr_file
        self._roads: Dict[str, OpenDriveRoadNetwork.Road] = {}
        self._lanes: Dict[str, OpenDriveRoadNetwork.Lane] = {}
        self._lanepoints = None

    @classmethod
    def from_file(
        cls,
        xodr_file,
    ):
        od_map = cls(xodr_file)
        od_map.load()
        return od_map

    @staticmethod
    def _elem_id(elem):
        if type(elem) == LaneSectionElement:
            return f"{elem.parentRoad.id}_{elem.idx}"
        elif type(elem) == LaneElement:
            return f"{elem.parentRoad.id}_{elem.lane_section.idx}_{elem.id}"
        else:
            return None

    def load(self):
        # Parse the xml definition into an initial representation
        start = time.time()
        od: OpenDriveElement = None
        with open(self._xodr_file, "r") as f:
            od = parse_opendrive(etree.parse(f).getroot())
        end = time.time()
        elapsed = round((end - start) * 1000.0, 3)
        self._log.info(f"Parsing .xodr file: {elapsed} ms")

        # First pass: create all Road and Lane objects
        start = time.time()
        for road_elem in od.roads:
            road_elem: RoadElement = road_elem
            for section_elem in road_elem.lanes.lane_sections:
                section_elem: LaneSectionElement = section_elem
                road_id = OpenDriveRoadNetwork._elem_id(section_elem)
                road = OpenDriveRoadNetwork.Road(
                    road_id,
                    section_elem.parentRoad.junction is not None,
                    section_elem.length,
                )
                self._roads[road_id] = road
                for lane_elem in section_elem.leftLanes + section_elem.rightLanes:
                    lane_id = OpenDriveRoadNetwork._elem_id(lane_elem)
                    lane = OpenDriveRoadNetwork.Lane(
                        lane_id, road, lane_elem.id, section_elem.length
                    )
                    self._lanes[lane_id] = lane
                    road.lanes.append(lane)
        end = time.time()
        elapsed = round((end - start) * 1000.0, 3)
        self._log.info(f"First pass: {elapsed} ms")

        # Second pass: compute road and lane connections, compute lane boundaries and polygon
        start = time.time()
        self._precompute_junction_connections(od)
        for road_elem in od.roads:
            for section_elem in road_elem.lanes.lane_sections:
                road_id = OpenDriveRoadNetwork._elem_id(section_elem)
                road = self._roads[road_id]

                # Compute road's incoming and outgoing roads
                self._compute_road_connections(od, road, road_elem)

                road_plan_view = road_elem.planView

                # Compute left lanes connections and polygon
                inner_boundary = LaneBoundary(road_plan_view, None, None)
                right_lanes_iteration = False
                for lane_elem in section_elem.leftLanes + section_elem.rightLanes:
                    lane_id = OpenDriveRoadNetwork._elem_id(lane_elem)
                    lane = self._lanes[lane_id]

                    # Compute lane's incoming and outgoing lanes
                    self._compute_lane_connections(od, lane, lane_elem, road_elem)

                    # Reset inner_boundary if rightLane iteration begins
                    if (
                        lane_elem in section_elem.rightLanes
                        and not right_lanes_iteration
                    ):
                        inner_boundary = LaneBoundary(road_plan_view, None, None)
                        right_lanes_iteration = True

                    # Computer lane's boundary
                    for width in lane_elem.widths:
                        poly = CubicPolynomial.from_list(width.polynomial_coefficients)
                        outer_boundary = LaneBoundary(None, inner_boundary, poly)
                        lane.lane_boundaries.append((inner_boundary, outer_boundary))
                        inner_boundary = outer_boundary

                    # Compute lane's polygon
                    OpenDriveRoadNetwork._compute_lane_polygon(
                        lane, lane_elem, road_plan_view
                    )

        end = time.time()
        elapsed = round((end - start) * 1000.0, 3)
        self._log.info(f"Second pass: {elapsed} ms")

        # Third pass: fill in remaining properties
        start = time.time()
        for road_elem in od.roads:
            for section_elem in road_elem.lanes.lane_sections:
                road_id = OpenDriveRoadNetwork._elem_id(section_elem)
                road = self._roads[road_id]

                for lane_elem in section_elem.leftLanes + section_elem.rightLanes:
                    lane_id = OpenDriveRoadNetwork._elem_id(lane_elem)
                    lane = self._lanes[lane_id]

                    # Compute lanes in same direction
                    sign = np.sign(lane.index)
                    elems = [
                        elem
                        for elem in section_elem.allLanes
                        if np.sign(elem.id) == sign and elem.id != lane.index
                    ]
                    same_dir_lanes = [
                        self._lanes[OpenDriveRoadNetwork._elem_id(elem)]
                        for elem in elems
                    ]
                    lane.lanes_in_same_direction = same_dir_lanes

                    # Compute lane to the left
                    result = None
                    if lane.index > 0:
                        for other in lane.lanes_in_same_direction:
                            if lane.index - other.index == 1:
                                result = other
                                break
                    elif lane.index < 0:
                        for other in lane.lanes_in_same_direction:
                            if lane.index - other.index == -1:
                                result = other
                                break
                    lane.lane_to_left = result, True

                    # Compute lane to right
                    result = None
                    if lane.index > 0:
                        for other in lane.lanes_in_same_direction:
                            if lane.index - other.index == -1:
                                result = other
                                break
                    elif lane.index < 0:
                        for other in lane.lanes_in_same_direction:
                            if lane.index - other.index == 1:
                                result = other
                                break
                    lane.lane_to_right = result, True

                    # Compute lane foes
                    result = [
                        incoming
                        for outgoing in lane.outgoing_lanes
                        for incoming in outgoing.incoming_lanes
                        if incoming != lane
                    ]
                    if lane.in_junction:
                        in_roads = set(il.road for il in lane.incoming_lanes)
                        for foe in lane.road.lanes:
                            foe_in_roads = set(il.road for il in foe.incoming_lanes)
                            if not bool(in_roads & foe_in_roads):
                                result.append(foe)
                    lane.foes = list(set(result))

        end = time.time()
        elapsed = round((end - start) * 1000.0, 3)
        self._log.info(f"Third pass: {elapsed} ms")

    def _precompute_junction_connections(self, od: OpenDriveElement):
        for road_elem in od.roads:
            if road_elem.junction:
                # TODO: handle multiple lane sections in connecting roads?
                assert (
                    len(road_elem.lanes.lane_sections) == 1
                ), "Junction connecting roads must have a single lane section"
                # precompute junction road connections
                road_id = OpenDriveRoadNetwork._elem_id(
                    road_elem.lanes.lane_sections[0]
                )
                road = self.road_by_id(road_id)
                pred_road_id = None
                succ_road_id = None

                if road_elem.link.predecessor:
                    road_predecessor = road_elem.link.predecessor
                    if road_predecessor.contactPoint == "end":
                        pred_road_elem = od.getRoad(road_predecessor.element_id)
                        pred_ls_index = pred_road_elem.lanes.getLastLaneSectionIdx()
                    else:
                        pred_ls_index = 0
                    pred_road_id = f"{road_predecessor.element_id}_{pred_ls_index}"
                    pred_road = self.road_by_id(pred_road_id)
                    pred_road.outgoing_roads.append(road)
                    road.incoming_roads.append(pred_road)

                if road_elem.link.successor:
                    road_successor = road_elem.link.successor
                    if road_successor.contactPoint == "end":
                        succ_road_elem = od.getRoad(road_successor.element_id)
                        succ_ls_index = succ_road_elem.lanes.getLastLaneSectionIdx()
                    else:
                        succ_ls_index = 0
                    succ_road_id = f"{road_successor.element_id}_{succ_ls_index}"
                    succ_road = self.road_by_id(succ_road_id)
                    succ_road.incoming_roads.append(road)
                    road.outgoing_roads.append(succ_road)

                # precompute junction lane connections
                for lane_elem in (
                    road_elem.lanes.lane_sections[0].leftLanes
                    + road_elem.lanes.lane_sections[0].rightLanes
                ):
                    lane_id = OpenDriveRoadNetwork._elem_id(lane_elem)
                    lane = self.lane_by_id(lane_id)

                    if lane_elem.link.predecessorId:
                        assert pred_road_id
                        pred_lane_id = f"{pred_road_id}_{lane_elem.link.predecessorId}"
                        pred_lane = self.lane_by_id(pred_lane_id)

                        pred_lane.outgoing_lanes.append(lane)
                        lane.incoming_lanes.append(pred_lane)

                    if lane_elem.link.successorId:
                        assert succ_road_id
                        succ_lane_id = f"{succ_road_id}_{lane_elem.link.successorId}"
                        succ_lane = self.lane_by_id(succ_lane_id)

                        succ_lane.incoming_lanes.append(lane)
                        lane.outgoing_lanes.append(succ_lane)

    def _compute_road_connections(self, od, road, road_elem):
        if road.is_junction:
            return

        lane_section_idx = int(road.road_id.split("_")[1])
        # Incoming roads
        # For OpenDRIVE lane sections with idx = 0
        if lane_section_idx == 0:
            # Incoming roads - simple case
            predecessor = road_elem.link.predecessor
            if predecessor and predecessor.elementType == "road":
                pred_road_elem = od.getRoad(predecessor.element_id)
                section_index = (
                    pred_road_elem.lanes.getLastLaneSectionIdx()
                    if predecessor.contactPoint == "end"
                    else 0
                )
                in_road = self.road_by_id(
                    f"{road_elem.link.predecessor.element_id}_{section_index}"
                )
                road.incoming_roads.append(in_road)
        else:
            pred_road_id = f"{road_elem.id}_{lane_section_idx - 1}"
            in_road = self.road_by_id(pred_road_id)
            road.incoming_roads.append(in_road)

        # Outgoing roads
        # For OpenDRIVE lane sections with last idx
        if lane_section_idx == road_elem.lanes.getLastLaneSectionIdx():
            # Outgoing roads - simple case
            successor = road_elem.link.successor
            if successor and successor.elementType == "road":
                succ_road_elem = od.getRoad(successor.element_id)
                section_index = (
                    succ_road_elem.lanes.getLastLaneSectionIdx()
                    if successor.contactPoint == "end"
                    else 0
                )
                out_road = self.road_by_id(f"{successor.element_id}_{section_index}")
                road.outgoing_roads.append(out_road)

        else:
            succ_road_id = f"{road_elem.id}_{lane_section_idx + 1}"
            out_road = self.road_by_id(succ_road_id)
            road.outgoing_roads.append(out_road)

    def _compute_lane_connections(self, od, lane, lane_elem, road_elem):
        if lane.in_junction:
            return

        lane_link = lane_elem.link
        if lane_link.predecessorId:
            ls_index = lane_elem.lane_section.idx
            if ls_index == 0:
                # This is the first lane section, so get the last lane section of the incoming road
                road_predecessor = road_elem.link.predecessor
                if road_predecessor and road_predecessor.elementType == "road":
                    pred_road_elem = od.getRoad(road_predecessor.element_id)
                    section_index = (
                        pred_road_elem.lanes.getLastLaneSectionIdx()
                        if road_predecessor.contactPoint == "end"
                        else 0
                    )
                    pred_lane_id = f"{road_predecessor.element_id}_{section_index}_{lane_link.predecessorId}"
                    lane.incoming_lanes.append(self.lane_by_id(pred_lane_id))
            else:
                # Otherwise, get the previous lane section of the current road
                pred_lane_id = (
                    f"{road_elem.id}_{ls_index - 1}_{lane_link.predecessorId}"
                )
                lane.incoming_lanes.append(self.lane_by_id(pred_lane_id))

        if lane_link.successorId:
            ls_index = lane_elem.lane_section.idx
            if ls_index == len(road_elem.lanes.lane_sections) - 1:
                # This is the last lane section, so get the first lane section of the outgoing road
                road_successor = road_elem.link.successor
                if road_successor and road_successor.elementType == "road":
                    succ_road_elem = od.getRoad(road_successor.element_id)
                    section_index = (
                        succ_road_elem.lanes.getLastLaneSectionIdx()
                        if road_successor.contactPoint == "end"
                        else 0
                    )
                    succ_lane_id = f"{road_successor.element_id}_{section_index}_{lane_link.successorId}"
                    lane.outgoing_lanes.append(self.lane_by_id(succ_lane_id))
            else:
                # Otherwise, get the next lane section in the current road
                succ_lane_id = f"{road_elem.id}_{ls_index + 1}_{lane_link.successorId}"
                lane.outgoing_lanes.append(self.lane_by_id(succ_lane_id))

    @staticmethod
    def _compute_lane_polygon(
        lane, lane_elem: LaneElement, road_planview: PlanViewElement
    ):
        xs, ys = [], []
        section: LaneSectionElement = lane_elem.lane_section
        section_len = section.length
        section_s_start = section.sPos
        section_s_end = section_s_start + section_len

        for i in range(len(lane_elem.widths)):
            (inner_boundary, outer_boundary) = lane.lane_boundaries[i]
            inner_s_vals = inner_boundary.to_linear_segments(
                section_s_start, section_s_end
            )
            outer_s_vals = outer_boundary.to_linear_segments(
                section_s_start, section_s_end
            )
            s_vals = sorted(set(inner_s_vals + outer_s_vals))

            xs_inner, ys_inner = [], []
            xs_outer, ys_outer = [], []
            for s in s_vals:
                ds = s - lane_elem.widths[i].start_offset
                t_inner = inner_boundary.calc_t(ds)
                t_outer = outer_boundary.calc_t(ds)
                (x_ref, y_ref), heading = road_planview.calc(s)
                angle = lane.t_angle(heading)
                xs_inner.append(x_ref + t_inner * math.cos(angle))
                ys_inner.append(y_ref + t_inner * math.sin(angle))
                xs_outer.append(x_ref + t_outer * math.cos(angle))
                ys_outer.append(y_ref + t_outer * math.sin(angle))
            inner_boundary = outer_boundary
            xs.extend(xs_inner + xs_outer[::-1] + [xs_inner[0]])
            ys.extend(ys_inner + ys_outer[::-1] + [ys_inner[0]])

        assert len(xs) == len(ys)
        for i in range(len(xs)):
            lane.lane_polygon.append((xs[i], ys[i]))

    @property
    def source(self) -> str:
        """This is the .xodr file of the OpenDRIVE map."""
        return self._xodr_file

    class Lane(RoadMap.Lane):
        def __init__(self, lane_id: str, road: RoadMap.Road, index: int, length: float):
            self._lane_id = lane_id
            self._road = road
            self._index = index
            self._length = length
            self._incoming_lanes = []
            self._outgoing_lanes = []
            self._lanes_in_same_dir = []
            self._foes = []
            self._lane_boundaries = []
            self._lane_polygon = []
            self._lane_to_left = None, True
            self._lane_to_right = None, True
            self._in_junction = None

        @property
        def lane_id(self) -> str:
            return self._lane_id

        @property
        def road(self) -> RoadMap.Road:
            return self._road

        @property
        def length(self) -> float:
            return self._length

        @property
        def in_junction(self) -> bool:
            return self.road.is_junction

        @property
        def index(self) -> int:
            # TODO: convert to expected convention?
            return self._index

        @property
        def incoming_lanes(self) -> List[RoadMap.Lane]:
            return self._incoming_lanes

        @incoming_lanes.setter
        def incoming_lanes(self, value):
            self._incoming_lanes = value

        @property
        def outgoing_lanes(self) -> List[RoadMap.Lane]:
            return self._outgoing_lanes

        @outgoing_lanes.setter
        def outgoing_lanes(self, value):
            self._outgoing_lanes = value

        @property
        def lanes_in_same_direction(self) -> List[RoadMap.Lane]:
            return self._lanes_in_same_dir

        @lanes_in_same_direction.setter
        def lanes_in_same_direction(self, lanes):
            self._lanes_in_same_dir = lanes

        @property
        def lane_to_left(self) -> Tuple[RoadMap.Lane, bool]:
            return self._lane_to_left

        @lane_to_left.setter
        def lane_to_left(self, value):
            self._lane_to_left = value

        @property
        def lane_to_right(self) -> Tuple[RoadMap.Lane, bool]:
            return self._lane_to_right

        @lane_to_right.setter
        def lane_to_right(self, value):
            self._lane_to_right = value

        @property
        def foes(self) -> List[RoadMap.Lane]:
            return self._foes

        @foes.setter
        def foes(self, value):
            self._foes = value

        @property
        def lane_boundaries(self) -> List[Tuple[LaneBoundary, LaneBoundary]]:
            return self._lane_boundaries

        @lane_boundaries.setter
        def lane_boundaries(self, value):
            self._lane_boundaries = value

        @property
        def lane_polygon(self) -> List[Tuple[float, float]]:
            return self._lane_polygon

        @lane_polygon.setter
        def lane_polygon(self, value):
            self._lane_polygon = value

        def t_angle(self, s_heading: float):
            lane_elem_id = int(self.lane_id.split("_")[2])
            angle = (
                (s_heading - math.pi / 2)
                if lane_elem_id < 0
                else (s_heading + math.pi / 2)
            )
            return constrain_angle(angle)

    def lane_by_id(self, lane_id: str) -> RoadMap.Lane:
        lane = self._lanes.get(lane_id)
        if not lane:
            self._log.warning(
                f"OpenDriveRoadNetwork got request for unknown lane_id '{lane_id}'"
            )
        return lane

    class Road(RoadMap.Road):
        def __init__(self, road_id: str, is_junction: bool, length: float):
            self._log = logging.getLogger(self.__class__.__name__)
            self._road_id = road_id
            self._is_junction = is_junction
            self._length = length
            self._lanes = []
            self._incoming_roads = []
            self._outgoing_roads = []
            self._parallel_roads = []

        @property
        def road_id(self) -> str:
            return self._road_id

        @property
        def is_junction(self) -> bool:
            return self._is_junction

        @property
        def length(self) -> float:
            return self._length

        @property
        def incoming_roads(self) -> List[RoadMap.Road]:
            return self._incoming_roads

        @incoming_roads.setter
        def incoming_roads(self, value):
            self._incoming_roads = value

        @property
        def outgoing_roads(self) -> List[RoadMap.Road]:
            return self._outgoing_roads

        @outgoing_roads.setter
        def outgoing_roads(self, value):
            self._outgoing_roads = value

        @property
        def parallel_roads(self) -> List[RoadMap.Road]:
            return self._parallel_roads

        @parallel_roads.setter
        def parallel_roads(self, value):
            self._parallel_roads = value

        @property
        def lanes(self) -> List[RoadMap.Lane]:
            return self._lanes

        @lanes.setter
        def lanes(self, value):
            self._lanes = value

        def lane_at_index(self, index: int) -> RoadMap.Lane:
            lanes_with_index = [lane for lane in self.lanes if lane.index == index]
            if len(lanes_with_index) == 0:
                self._log.warning(
                    f"Road with id {self.road_id} has no lane at index {index}"
                )
                return None
            assert len(lanes_with_index) == 1
            return lanes_with_index[0]

    def road_by_id(self, road_id: str) -> RoadMap.Road:
        road = self._roads.get(road_id)
        if not road:
            self._log.warning(
                f"OpenDriveRoadNetwork got request for unknown road_id '{road_id}'"
            )
        return road
