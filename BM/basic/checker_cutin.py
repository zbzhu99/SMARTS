import math
from typing import Dict
from checker_utils import (
    get_lane_vector_at_offset,
    get_offset_into_lane,
    get_position_from_lane_offset,
    is_further_along_route,
)
from smarts.core.utils.math import squared_dist, vec_to_radians
from smarts.core.smarts import SMARTS
from smarts.core.sensors import EgoVehicleObservation, Observation, VehicleObservation
from checker import Checker, CheckerFrameResult, Result


class CutinChecker(Checker):
    @staticmethod
    def _get_index(l: list, index, default=None):
        try:
            return l[index]
        except IndexError:
            return default

    @staticmethod
    def _get_item(l: list, id, default=None):
        index = 0
        for i in range(len(l) + 1):
            index = i
            try:
                if id in l[i].id:
                    break
            except IndexError:
                return default
        return l[index]

    def __init__(self, bm_id, target_id=None) -> None:
        super().__init__(bm_id)

        self._uturn_started = False
        self._target_id = target_id

    def evaluate(
        self, sim: SMARTS, observations: Dict[str, Observation], rewards, dones, infos
    ) -> CheckerFrameResult:
        done = dones.get(self._bm_id, None)
        if done:
            return CheckerFrameResult("Done before performed", Result.FAIL)
        elif done is None:
            return Checker("Vehicle is no longer active", Result.INVALID)

        observation = observations.get(self._bm_id, None)
        if observation is None:
            return CheckerFrameResult(f"Vehicle `{self._bm_id}` not found", Result.TBD)

        if self._target_id:
            near: VehicleObservation = self._get_item(
                observation.neighborhood_vehicle_states, self._target_id
            )
        else:
            near: VehicleObservation = self._get_index(
                observation.neighborhood_vehicle_states, 0
            )
        ego: EgoVehicleObservation = observation.ego_vehicle_state

        if near:
            rn = sim.road_network

            further = is_further_along_route(
                rn, ego.lane_id, ego.position, near.lane_id, near.position
            )
            eoff = get_offset_into_lane(rn, ego.lane_id, ego.position[:2])

            ep_on_lane = get_position_from_lane_offset(rn, ego.lane_id, eoff)
            e_d_of_lane = get_lane_vector_at_offset(rn, ego.lane_id, eoff)

            e_lane_angle = vec_to_radians(e_d_of_lane)
            e_h_relative_to_lane = ego.heading.relative_to(e_lane_angle)

            if (
                further
                and squared_dist(ep_on_lane, ego.position[:2]) < 0.5
                and abs(e_h_relative_to_lane) < math.pi / 50
            ):
                return CheckerFrameResult("Successful cut-in", Result.PASS)

        return CheckerFrameResult("Episode continues", Result.TBD)

    def timeout(self):
        return CheckerFrameResult("Episode ended without cut-in", Result.FAIL)
