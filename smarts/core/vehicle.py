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
# FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
from __future__ import annotations

import importlib.resources as pkg_resources
import logging
import os
from dataclasses import dataclass
from functools import lru_cache, partial
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from smarts.core.agent_interface import AgentInterface
from smarts.core.plan import Mission, Plan

from . import config, models
from .actor import ActorRole
from .chassis import AckermannChassis, BoxChassis, Chassis
from .colors import SceneColors
from .coordinates import Dimensions, Heading, Pose
from .sensors import (
    AccelerometerSensor,
    DrivableAreaGridMapSensor,
    DrivenPathSensor,
    LanePositionSensor,
    LidarSensor,
    NeighborhoodVehiclesSensor,
    OGMSensor,
    RGBSensor,
    RoadWaypointsSensor,
    Sensor,
    SignalsSensor,
    TripMeterSensor,
    ViaSensor,
    WaypointsSensor,
)
from .utils.core_math import rotate_cw_around_point
from .utils.custom_exceptions import RendererException
from .vehicle_state import VEHICLE_CONFIGS, VehicleConfig, VehicleState


class Vehicle:
    """Represents a single vehicle."""

    _HAS_DYNAMIC_ATTRIBUTES = True  # dynamic pytype attribute
    _sensor_names = (
        "ogm_sensor",
        "rgb_sensor",
        "lidar_sensor",
        "driven_path_sensor",
        "trip_meter_sensor",
        "drivable_area_grid_map_sensor",
        "neighborhood_vehicle_states_sensor",
        "waypoints_sensor",
        "road_waypoints_sensor",
        "accelerometer_sensor",
        "lane_position_sensor",
        "via_sensor",
        "signals_sensor",
    )

    def __init__(
        self,
        id: str,
        chassis: Chassis,
        vehicle_config_type: str = "passenger",
        color: Optional[SceneColors] = None,
        action_space=None,
    ):
        self._log = logging.getLogger(self.__class__.__name__)
        self._id = id

        self._chassis: Chassis = chassis
        if vehicle_config_type == "sedan":
            vehicle_config_type = "passenger"
        self._vehicle_config_type = vehicle_config_type
        self._action_space = action_space

        self._meta_create_sensor_functions()
        self._sensors = {}

        # Color override
        self._color: Optional[SceneColors] = color
        if self._color is None:
            config = VEHICLE_CONFIGS[vehicle_config_type]
            self._color = config.color

        self._initialized = True
        self._has_stepped = False

    def _assert_initialized(self):
        assert self._initialized, f"Vehicle({self.id}) is not initialized"

    def __eq__(self, __o: object) -> bool:
        if self is __o:
            return True
        if isinstance(__o, self.__class__) and self.state == __o.state:
            return True
        return False

    def __repr__(self):
        return f"""Vehicle({self.id},
  pose={self.pose},
  speed={self.speed},
  type={self.vehicle_type},
  w={self.width},
  l={self.length},
  h={self.height}
)"""

    @property
    def id(self):
        """The id of this vehicle."""
        return self._id

    @property
    def length(self) -> float:
        """The length of this vehicle."""
        self._assert_initialized()
        return self._chassis.dimensions.length

    @property
    def max_steering_wheel(self) -> Optional[float]:
        """The max steering value the chassis steering wheel can turn to.

        Some chassis types do not support this.
        """
        self._assert_initialized()
        return getattr(self._chassis, "max_steering_wheel", None)

    @property
    def width(self) -> float:
        """The width of this vehicle."""
        self._assert_initialized()
        return self._chassis.dimensions.width

    @property
    def height(self) -> float:
        """The height of this vehicle."""
        self._assert_initialized()
        return self._chassis.dimensions.height

    @property
    def speed(self) -> float:
        """The current speed of this vehicle."""
        self._assert_initialized()
        return self._chassis.speed

    @property
    def sensors(self) -> Dict[str, Sensor]:
        """The sensors attached to this vehicle."""
        self._assert_initialized()
        return self._sensors

    # # TODO: See issue #898 This is a currently a no-op
    # @speed.setter
    # def speed(self, speed):
    #     self._chassis.speed = speed

    @property
    def vehicle_color(self) -> Union[SceneColors, None]:
        """The color of this vehicle (generally used for rendering purposes.)"""
        self._assert_initialized()
        return self._color

    @property
    def state(self) -> VehicleState:
        """The current state of this vehicle."""
        self._assert_initialized()
        return VehicleState(
            actor_id=self.id,
            actor_type=self.vehicle_type,
            source="SMARTS",  # this is the "ground truth" state
            vehicle_config_type=self._vehicle_config_type,
            pose=self.pose,
            dimensions=self._chassis.dimensions,
            speed=self.speed,
            # pytype: disable=attribute-error
            steering=self._chassis.steering,
            # pytype: enable=attribute-error
            yaw_rate=self._chassis.yaw_rate,
            linear_velocity=self._chassis.velocity_vectors[0],
            angular_velocity=self._chassis.velocity_vectors[1],
        )

    @property
    def action_space(self):
        """The action space this vehicle uses."""
        self._assert_initialized()
        return self._action_space

    @property
    def pose(self) -> Pose:
        """The pose of this vehicle. Pose is defined as position and orientation."""
        self._assert_initialized()
        return self._chassis.pose

    @property
    def chassis(self) -> Chassis:
        """The underlying chassis of this vehicle."""
        self._assert_initialized()
        return self._chassis

    @property
    def heading(self) -> Heading:
        """The heading of this vehicle.

        Note: Heading rotates counterclockwise with north as 0.
        """
        self._assert_initialized()
        return self._chassis.pose.heading

    @property
    def position(self) -> np.ndarray:
        """The position of this vehicle."""
        self._assert_initialized()
        return self._chassis.pose.position

    @property
    def bounding_box(self) -> List[np.ndarray]:
        """The minimum fitting heading aligned bounding box. Four 2D points representing the minimum fitting box."""
        # XXX: this doesn't return a smarts.core.coordinates.BoundingBox!
        self._assert_initialized()
        # Assuming the position is the center,
        # calculate the corner coordinates of the bounding_box
        origin = self.position[:2]
        dimensions = np.array([self.width, self.length])
        corners = np.array([(-1, 1), (1, 1), (1, -1), (-1, -1)]) / 2
        heading = self.heading
        return [
            rotate_cw_around_point(
                point=origin + corner * dimensions,
                radians=Heading.flip_clockwise(heading),
                origin=origin,
            )
            for corner in corners
        ]

    @property
    def vehicle_type(self) -> str:
        """Get the vehicle type identifier."""
        return VEHICLE_CONFIGS[self._vehicle_config_type].vehicle_type

    @property
    def valid(self) -> bool:
        """Check if the vehicle still `exists` and is still operable."""
        return self._initialized

    @classmethod
    @property
    def sensor_names(cls) -> Tuple[str]:
        """The names of the sensors that are potentially available to this vehicle."""
        return cls._sensor_names

    @staticmethod
    @lru_cache(maxsize=None)
    def vehicle_urdf_path(vehicle_type: str, override_path: Optional[str]) -> str:
        """Resolve the physics model filepath.

        Args:
            vehicle_type (str): The type of the vehicle.
            override_path (Optional[str]): The override.

        Raises:
            ValueError: The vehicle type is valid.

        Returns:
            str: The path to the model `.urdf`.
        """
        if (override_path is not None) and os.path.exists(override_path):
            return override_path

        if vehicle_type == "sedan":
            vehicle_type = "passenger"

        if vehicle_type == "passenger":
            urdf_name = "vehicle"
        elif vehicle_type in {
            "bus",
            "coach",
            "motorcycle",
            "pedestrian",
            "trailer",
            "truck",
        }:
            urdf_name = vehicle_type
        else:
            raise ValueError(f"Vehicle type `{vehicle_type}` does not exist!!!")

        with pkg_resources.path(models, urdf_name + ".urdf") as path:
            vehicle_filepath = str(path.absolute())

        return vehicle_filepath

    @staticmethod
    def agent_vehicle_dims(
        mission: Mission, default: Optional[str] = None
    ) -> Dimensions:
        """Get the vehicle dimensions from the mission requirements.
        Args:
            A mission for the agent.
        Returns:
            The mission vehicle spec dimensions XOR the default "passenger" vehicle dimensions.
        """
        if default == "sedan":
            default = "passenger"
        default_type = default or config().get_setting(
            "resources", "default_agent_vehicle", default="passenger"
        )
        if mission.vehicle_spec:
            # mission.vehicle_spec.veh_config_type will always be "passenger" for now,
            # but we use that value here in case we ever expand our history functionality.
            vehicle_config_type = mission.vehicle_spec.veh_config_type
            return Dimensions.copy_with_defaults(
                mission.vehicle_spec.dimensions,
                VEHICLE_CONFIGS[vehicle_config_type or default_type].dimensions,
            )
        return VEHICLE_CONFIGS[default_type].dimensions

    @classmethod
    def build_agent_vehicle(
        cls,
        sim,
        vehicle_id: str,
        agent_interface: AgentInterface,
        plan: Plan,
        vehicle_filepath: Optional[str],
        tire_filepath: str,
        trainable: bool,
        surface_patches: List[Dict[str, Any]],
        initial_speed: Optional[float] = None,
    ) -> "Vehicle":
        """Create a new vehicle and set up sensors and planning information as required by the
        ego agent.
        """
        urdf_file = cls.vehicle_urdf_path(
            vehicle_type=agent_interface.vehicle_type, override_path=vehicle_filepath
        )

        mission = plan.mission
        chassis_dims = cls.agent_vehicle_dims(
            mission, default=agent_interface.vehicle_type
        )

        start = mission.start
        if start.from_front_bumper:
            start_pose = Pose.from_front_bumper(
                front_bumper_position=np.array(start.position[:2]),
                heading=start.heading,
                length=chassis_dims.length,
            )
        else:
            start_pose = Pose.from_center(start.position, start.heading)

        vehicle_color = SceneColors.Agent if trainable else SceneColors.SocialAgent
        controller_parameters = sim.vehicle_index.controller_params_for_vehicle_type(
            agent_interface.vehicle_type
        )

        chassis = None
        if agent_interface and agent_interface.action in sim.dynamic_action_spaces:
            if mission.vehicle_spec:
                logger = logging.getLogger(cls.__name__)
                logger.warning(
                    "setting vehicle dimensions on a AckermannChassis not yet supported"
                )
            chassis = AckermannChassis(
                pose=start_pose,
                bullet_client=sim.bc,
                vehicle_filepath=vehicle_filepath,
                tire_parameters_filepath=tire_filepath,
                friction_map=surface_patches,
                controller_parameters=controller_parameters,
                initial_speed=initial_speed,
            )
        else:
            chassis = BoxChassis(
                pose=start_pose,
                speed=initial_speed,
                dimensions=chassis_dims,
                bullet_client=sim.bc,
            )

        vehicle = Vehicle(
            id=vehicle_id,
            chassis=chassis,
            color=vehicle_color,
            vehicle_config_type=agent_interface.vehicle_type,
        )

        return vehicle

    @staticmethod
    def build_social_vehicle(sim, vehicle_id, vehicle_state: VehicleState) -> "Vehicle":
        """Create a new unassociated vehicle."""
        dims = Dimensions.copy_with_defaults(
            vehicle_state.dimensions,
            VEHICLE_CONFIGS[vehicle_state.vehicle_config_type].dimensions,
        )
        chassis = BoxChassis(
            pose=vehicle_state.pose,
            speed=vehicle_state.speed,
            dimensions=dims,
            bullet_client=sim.bc,
        )
        vehicle = Vehicle(
            id=vehicle_id,
            chassis=chassis,
            vehicle_config_type=vehicle_state.vehicle_config_type,
        )
        return vehicle

    @classmethod
    def attach_sensors_to_vehicle(
        cls,
        sensor_manager,
        sim,
        vehicle: "Vehicle",
        agent_interface: AgentInterface,
        replace=True,
        reset_sensors=False,
    ):
        """Attach sensors as required to satisfy the agent interface's requirements"""
        # The distance travelled sensor is not optional b/c it is used for the score
        # and reward calculation
        vehicle_state = vehicle.state
        has_no_sensors = len(vehicle.sensors) == 0
        added_sensors: List[Tuple[str, Sensor]] = []

        if reset_sensors:
            sensor_manager.remove_actor_sensors_by_actor_id(vehicle.id)
            # pytype: disable=attribute-error
            Vehicle.detach_all_sensors_from_vehicle(vehicle)
            # pytype: enable=attribute-error

        def add_sensor_if_needed(
            sensor_type,
            sensor_name: str,
            condition: bool = True,
            **kwargs,
        ):
            assert sensor_name in cls._sensor_names
            if (
                replace
                or has_no_sensors
                or (condition and not vehicle.subscribed_to(sensor_name))
            ):
                sensor = sensor_type(**kwargs)
                vehicle.attach_sensor(sensor, sensor_name)
                added_sensors.append((sensor_name, sensor))

        add_sensor_if_needed(TripMeterSensor, sensor_name="trip_meter_sensor")
        add_sensor_if_needed(DrivenPathSensor, sensor_name="driven_path_sensor")
        if agent_interface.neighborhood_vehicle_states:
            add_sensor_if_needed(
                NeighborhoodVehiclesSensor,
                sensor_name="neighborhood_vehicle_states_sensor",
                radius=agent_interface.neighborhood_vehicle_states.radius,
            )

        add_sensor_if_needed(
            AccelerometerSensor,
            sensor_name="accelerometer_sensor",
            condition=agent_interface.accelerometer,
        )
        add_sensor_if_needed(
            WaypointsSensor,
            sensor_name="waypoints_sensor",
            condition=agent_interface.waypoint_paths,
        )
        if agent_interface.road_waypoints:
            add_sensor_if_needed(
                RoadWaypointsSensor,
                "road_waypoints_sensor",
                horizon=agent_interface.road_waypoints.horizon,
            )
        add_sensor_if_needed(
            LanePositionSensor,
            "lane_position_sensor",
            condition=agent_interface.lane_positions,
        )
        # DrivableAreaGridMapSensor
        if agent_interface.drivable_area_grid_map:
            if not sim.renderer:
                raise RendererException.required_to("add a drivable_area_grid_map")
            add_sensor_if_needed(
                DrivableAreaGridMapSensor,
                "drivable_area_grid_map_sensor",
                True,  # Always add this sensor
                vehicle_state=vehicle_state,
                width=agent_interface.drivable_area_grid_map.width,
                height=agent_interface.drivable_area_grid_map.height,
                resolution=agent_interface.drivable_area_grid_map.resolution,
                renderer=sim.renderer,
            )
        # OGMSensor
        if agent_interface.occupancy_grid_map:
            if not sim.renderer:
                raise RendererException.required_to("add an OGM")
            add_sensor_if_needed(
                OGMSensor,
                "ogm_sensor",
                True,  # Always add this sensor
                vehicle_state=vehicle_state,
                width=agent_interface.occupancy_grid_map.width,
                height=agent_interface.occupancy_grid_map.height,
                resolution=agent_interface.occupancy_grid_map.resolution,
                renderer=sim.renderer,
            )
        # RGBSensor
        if agent_interface.top_down_rgb:
            if not sim.renderer:
                raise RendererException.required_to("add an RGB camera")
            add_sensor_if_needed(
                RGBSensor,
                "rgb_sensor",
                True,  # Always add this sensor
                vehicle_state=vehicle_state,
                width=agent_interface.top_down_rgb.width,
                height=agent_interface.top_down_rgb.height,
                resolution=agent_interface.top_down_rgb.resolution,
                renderer=sim.renderer,
            )
        if agent_interface.lidar_point_cloud:
            add_sensor_if_needed(
                LidarSensor,
                "lidar_sensor",
                vehicle_state=vehicle_state,
                sensor_params=agent_interface.lidar_point_cloud.sensor_params,
            )
        add_sensor_if_needed(
            ViaSensor, "via_sensor", True, lane_acquisition_range=80, speed_accuracy=1.5
        )
        if agent_interface.signals:
            add_sensor_if_needed(
                SignalsSensor,
                "signals_sensor",
                lookahead=agent_interface.signals.lookahead,
            )

        for sensor_name, sensor in added_sensors:
            if not sensor:
                continue
            sensor_manager.add_sensor_for_actor(vehicle.id, sensor_name, sensor)

    def step(self, current_simulation_time):
        """Update internal state."""
        self._has_stepped = True
        self._chassis.step(current_simulation_time)

    def control(self, *args, **kwargs):
        """Apply control values to this vehicle.

        Forwards control to the chassis.
        """
        self._chassis.control(*args, **kwargs)

    def update_state(self, state: VehicleState, dt: float):
        """Update the vehicle's state"""
        state.updated = True
        if state.role != ActorRole.External:
            assert isinstance(self._chassis, BoxChassis)
            self.control(pose=state.pose, speed=state.speed, dt=dt)
            return
        # External actors are "privileged", which means they work directly (bypass force application).
        # Conceptually, this is playing 'god' with physics and should only be used
        # to defer to a co-simulator's states.
        linear_velocity, angular_velocity = None, None
        if not np.allclose(
            self._chassis.velocity_vectors[0], state.linear_velocity
        ) or not np.allclose(self._chassis.velocity_vectors[1], state.angular_velocity):
            linear_velocity = state.linear_velocity
            angular_velocity = state.angular_velocity
        if not state.dimensions.equal_if_defined(self.length, self.width, self.height):
            self._log.warning(
                "Unable to change a vehicle's dimensions via external_state_update()."
            )
        # XXX:  any way to update acceleration in pybullet?
        self._chassis.state_override(dt, state.pose, linear_velocity, angular_velocity)

    def create_renderer_node(self, renderer):
        """Create the vehicle's rendering node in the renderer."""
        config = VEHICLE_CONFIGS[self._vehicle_config_type]
        return renderer.create_vehicle_node(
            config.glb_model, self._id, self.vehicle_color, self.pose
        )

    # @lru_cache(maxsize=1)
    def _warn_AckermannChassis_set_pose(self):
        if self._has_stepped and isinstance(self._chassis, AckermannChassis):
            logging.warning(
                f"Agent `{self._id}` has called set pose after step."
                "This may cause collision problems"
            )

    # TODO: Merge this w/ speed setter as a set GCD call?
    def set_pose(self, pose: Pose):
        """Use with caution. This will directly set the pose of the chassis.

        This may disrupt physics simulation of the chassis physics body for a few steps after use.
        """
        self._warn_AckermannChassis_set_pose()
        self._chassis.set_pose(pose)

    def swap_chassis(self, chassis: Chassis):
        """Swap the current chassis with the given chassis. Apply the GCD of the previous chassis
        to the new chassis ("greatest common denominator state" from front-end to back-end)
        """
        chassis.inherit_physical_values(self._chassis)
        self._chassis.teardown()
        self._chassis = chassis

    def teardown(self, renderer, exclude_chassis=False):
        """Clean up internal resources"""
        if not exclude_chassis:
            self._chassis.teardown()
        if renderer:
            renderer.remove_vehicle_node(self._id)
        self._initialized = False

    def attach_sensor(self, sensor, sensor_name):
        """replace previously-attached sensor with this one
        (to allow updating its parameters).
        Sensors might have been attached to a non-agent vehicle
        (for example, for observation collection from history vehicles),
        but if that vehicle gets hijacked, we want to use the sensors
        specified by the hijacking agent's interface."""
        detach = getattr(self, f"detach_{sensor_name}")
        if detach:
            self.detach_sensor(sensor_name)
        self._log.debug("Replaced existing %s on vehicle %s", sensor_name, self.id)
        setattr(self, f"_{sensor_name}", sensor)
        self._sensors[sensor_name] = sensor

    def detach_sensor(self, sensor_name):
        """Detach a sensor by name."""
        self._log.debug("Removed existing %s on vehicle %s", sensor_name, self.id)
        sensor = getattr(self, f"_{sensor_name}", None)
        if sensor is not None:
            setattr(self, f"_{sensor_name}", None)
            del self._sensors[sensor_name]
        return sensor

    def subscribed_to(self, sensor_name):
        """Confirm if the sensor is subscribed."""
        sensor = getattr(self, f"_{sensor_name}", None)
        return sensor is not None

    def sensor_property(self, sensor_name):
        """Call a sensor by name."""
        sensor = getattr(self, f"_{sensor_name}", None)
        assert sensor is not None, f"'{sensor_name}' is not attached to '{self.id}'"
        return sensor

    def _meta_create_instance_sensor_functions(self):
        for sensor_name in Vehicle._sensor_names:
            setattr(self, f"_{sensor_name}", None)
            setattr(
                self,
                f"attach_{sensor_name}",
                partial(
                    self.__class__.attach_sensor, self=self, sensor_name=sensor_name
                ),
            )
            setattr(
                self,
                f"detach_{sensor_name}",
                partial(
                    self.__class__.detach_sensor, self=self, sensor_name=sensor_name
                ),
            )

    @classmethod
    @lru_cache(1)
    def _meta_create_class_sensor_functions(cls):
        for sensor_name in cls._sensor_names:
            setattr(
                cls,
                f"subscribed_to_{sensor_name}",
                property(
                    partial(cls.subscribed_to, sensor_name=sensor_name)
                ),
            )
            setattr(
                Vehicle,
                f"{sensor_name}",
                property(
                    partial(cls.sensor_property, sensor_name=sensor_name)
                ),
            )

        def detach_all_sensors_from_vehicle(vehicle):
            sensors = []
            for sensor_name in cls._sensor_names:
                detach_sensor_func = getattr(vehicle, f"detach_{sensor_name}")
                sensors.append(detach_sensor_func())
            return sensors

        setattr(
            cls,
            "detach_all_sensors_from_vehicle",
            staticmethod(detach_all_sensors_from_vehicle),
        )

    def _meta_create_sensor_functions(self):
        # Bit of metaprogramming to make sensor creation more DRY
        self._meta_create_instance_sensor_functions()
        self._meta_create_class_sensor_functions()

