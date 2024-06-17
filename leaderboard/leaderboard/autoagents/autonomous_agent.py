#!/usr/bin/env python

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides the base class for all autonomous agents
"""

from __future__ import print_function

from enum import Enum
import numpy as np
import carla
import torch.nn as nn
from srunner.scenariomanager.timer import GameTime
import os
from leaderboard.utils.route_manipulation import downsample_route

from pathlib import Path
import math
import cv2


from filelock import FileLock
from leaderboard.envs.sensor_interface import SensorInterface

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

from utilities.velocity import *
from leaderboard.utils.transforms import loc_global_to_ref_numpy

from leaderboard.envs.bev_utils.producer import BirdViewProducer

MAXIMUM_VELOCITY = 120/3.6
MAXIMUM_BLOCK_TIME = 20

class Track(Enum):

    """
    This enum represents the different tracks of the CARLA AD leaderboard.
    """
    SENSORS = 'SENSORS'
    MAP = 'MAP'


class AutonomousAgent(nn.Module):

    """
    Autonomous agent base class. All user agents have to be derived from this class
    """

    def __init__(self):
        super(AutonomousAgent, self).__init__()
        self.reset_vars()
        

    def reset_vars(self):
        self.track = Track.SENSORS
        #  current global plans to reach a destination
        self._global_plan = None
        self._global_plan_world_coord = None
        self._global_route = None

        self.vehicle = None

        # this data structure will contain all sensor data
        self.sensor_interface = SensorInterface()

        self.wallclock_t0 = None

        self._last_route_location = None

        self.block_time = 0.0

        # vars to compute velocities as in the logs.
        self.current_positions = {}
        self.old_positions = {}
        self.velocities = {}

    def setup(self, configs):
        """
        Initialize everything needed by your agent and set the track attribute to the right type:
            Track.SENSORS : CAMERAS, LIDAR, RADAR, GPS and IMU sensors are allowed
            Track.MAP : OpenDRIVE map is also allowed
        """
        
        self.fps = configs['experiment']['fps']

        ppm = 10

        if configs['experiment']['routes'].split('/')[-1] == 'routes_training.xml':
            cache_path = Path(f"{os.getenv('PRIBOOT_ROOT')}/birdview_cache/Carla/Maps/Town12/Town12_ppm{ppm}.dat")
            shape = (3, 88654, 99980)
        
        else:
            cache_path = (f"{os.getenv('PRIBOOT_ROOT')}/birdview_cache/Carla/Maps/Town13/Town13_ppm10.dat")
            shape = (3, 105382, 136662)

        with FileLock(f"{cache_path}.lock"):
            if Path(cache_path).is_file():
                print(f"Loading cache from {cache_path}")
                # static_cache = np.load(cache_path)
                static_cache = np.memmap(cache_path, dtype=np.uint8, mode='r', shape=shape)
                full_road_cache = static_cache[0]
                full_lanes_cache = static_cache[1]
                full_centerlines_cache = static_cache[2]
                print(f"Loaded static layers from cache file: {cache_path}")

        self.cache_dict = {'full_road_cache':full_road_cache,
                            'full_lanes_cache':full_lanes_cache,
                            'full_centerlines_cache':full_centerlines_cache}


        self.block_time = 0.0
        self.tl_node = None
        
        

    def sensors(self):  # pylint: disable=no-self-use
        """
        Define the sensor suite required by the agent

        :return: a list containing the required sensors in the following format:

        [
            {'type': 'sensor.camera.rgb', 'x': 0.7, 'y': -0.4, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                      'width': 300, 'height': 200, 'fov': 100, 'id': 'Left'},

            {'type': 'sensor.camera.rgb', 'x': 0.7, 'y': 0.4, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                      'width': 300, 'height': 200, 'fov': 100, 'id': 'Right'},

            {'type': 'sensor.lidar.ray_cast', 'x': 0.7, 'y': 0.0, 'z': 1.60, 'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0,
             'id': 'LIDAR'}
        ]

        """
        sensors = []

        return sensors

    def run_step(self, input_data, timestamp):
        """
        Execute one step of navigation.
        :return: control
        """
        control = carla.VehicleControl()
        control.steer = 0.0
        control.throttle = 0.0
        control.brake = 0.0
        control.hand_brake = False

        return control

    def destroy(self):
        """
        Destroy (clean-up) the agent
        :return:
        """
        pass
    
    def update_input_data(self):
        input_data = self.sensor_interface.get_data(GameTime.get_frame())
        input_data['control'] = self.get_control()
        input_data['speed'] = self.get_velocity_squared()
        input_data['block_time'] = np.array([self.block_time / MAXIMUM_BLOCK_TIME], dtype=np.float32)
        input_data['birdview'] = self.process_masks(input_data['birdview'][1]['masks'])
        input_data['target_point'] = self.get_target_point()
        input_data['command'] = self.get_command()
        
        self.input_data = input_data
    
    def __call__(self):
        """
        Execute the agent call, e.g. agent()
        Returns the next vehicle controls
        """
        self.old_positions = self.current_positions
        self.current_positions = get_positions(self.world)
        self.velocities = compute_velocities(self.current_positions, self.old_positions)
        
        self._truncate_global_route_till_target()
        
        input_data = self.sensor_interface.get_data(GameTime.get_frame())
        input_data['control'] = self.get_control()
        input_data['speed'] = self.get_velocity_squared()
        input_data['block_time'] = np.array([self.block_time / MAXIMUM_BLOCK_TIME], dtype=np.float32)        
        input_data['birdview'] = self.process_masks(input_data['birdview'][1]['masks'])
        input_data['target_point'] = self.get_target_point()
        input_data['command'] = self.get_command()
        
        self.update_block_time(input_data['speed'])
                
        control, waypoints, metadata = self.run_step(input_data)
        control.manual_gear_shift = False
        
        return control, waypoints, metadata

    def set_global_plan(self, global_plan_gps, global_plan_world_coord):
        """
        Set the plan (route) for the agent
        """
        ds_ids = downsample_route(global_plan_world_coord, 200)
        self._global_plan_world_coord = [(global_plan_world_coord[x][0], global_plan_world_coord[x][1]) for x in ds_ids]
        self._global_plan = [global_plan_gps[x] for x in ds_ids]
        
        self._global_route = global_plan_world_coord
        
        self.vehicle = CarlaDataProvider.get_hero_actor()
        self.world = CarlaDataProvider.get_world()
        
    def set_tl_node(self, tl_node):
        self.tl_node = tl_node
        
    def set_om(self, om_vehicle, om_pedestrian, om_construction):
        self.om_vehicle = om_vehicle
        self.om_pedestrian = om_pedestrian
        self.om_construction = om_construction  
    
    def _truncate_global_route_till_target(self, windows_size=5):
        
        ev_location = self.vehicle.get_location()
        closest_idx = 0

        for i in range(len(self._global_route)-1):
            if i > windows_size:
                break
            
            loc0 = self._global_route[i][0].location
            loc1 = self._global_route[i+1][0].location

            wp_dir = loc1 - loc0
            wp_veh = ev_location - loc0
            dot_ve_wp = wp_veh.x * wp_dir.x + wp_veh.y * wp_dir.y + wp_veh.z * wp_dir.z

            if dot_ve_wp > 0:
                closest_idx = i+1
                
        if closest_idx > 0:
            self._last_route_location = carla.Location(self._global_route[0][0].location)

        self._global_route = self._global_route[closest_idx:]
        

    def get_route_transform(self):
        
        if self._last_route_location == None:
            self._last_route_location = self.vehicle.get_location()
        
        loc0 = self._last_route_location
        loc1 = self._global_route[0][0].location

        if loc1.distance(loc0) < 0.1:
            yaw = self._global_route[0][0].rotation.yaw
        else:
            f_vec = loc1 - loc0
            yaw = np.rad2deg(np.arctan2(f_vec.y, f_vec.x))
        rot = carla.Rotation(yaw=yaw)
        return carla.Transform(location=loc0, rotation=rot)
    

    def get_control(self):
        
        control = self.vehicle.get_control()
        speed_limit = (self.vehicle.get_speed_limit() / 3.6) / MAXIMUM_VELOCITY
        control = {
            'throttle': np.array([control.throttle], dtype=np.float32),
            'steer': np.array([control.steer], dtype=np.float32),
            'brake': np.array([control.brake], dtype=np.float32),
            'gear': np.array([control.gear], dtype=np.float32),
            'speed_limit': np.array([speed_limit], dtype=np.float32),
        }
        return control

    def get_velocity_squared(self):
        
        velocity = self.velocities[self.vehicle.id]
        velocity_squared = velocity.x**2
        velocity_squared += velocity.y**2
        velocity_squared = math.sqrt(velocity_squared)
        
    
        return np.array([velocity_squared/MAXIMUM_VELOCITY], dtype=np.float32)

    def update_block_time(self, velocity):
        if (velocity * MAXIMUM_VELOCITY) < VELOCITY_THRESHOLD:
            self.block_time += 1 / self.fps
        else:
            self.block_time = 0.0

    def get_target_point(self):
        target_point, _ = self._global_route[30] if len(self._global_route) > 30 else self._global_route[-1]
        target_point_x = target_point.location.x
        target_point_y = target_point.location.y
        target_point_z = target_point.location.z
        
        ego_location = self.vehicle.get_location()
        ego_location_np = np.array([ego_location.x, ego_location.y, ego_location.z])
        
        ego_rotation = self.vehicle.get_transform().rotation
        ego_rotation_dict = {'roll': ego_rotation.roll, 'pitch': ego_rotation.pitch, 'yaw': ego_rotation.yaw}
        
        
        target_point = loc_global_to_ref_numpy(np.array([target_point_x, target_point_y, target_point_z]), {'location': ego_location_np, 'rotation': ego_rotation_dict})
        
        return target_point[0:2]
    
    def get_command(self):
        
        command_dict = {'VOID': -1, 'LEFT': 1, 'RIGHT': 2, 'STRAIGHT': 3, 'LANEFOLLOW': 4, 'CHANGELANELEFT': 5, 'CHANGELANERIGHT': 6}

        _, command = self._global_route[0]
        command = str(command)
        command = command.split('.')[-1]

        command = command_dict[command]

        if command < 0:
            command = 4
        command -= 1
        assert command in [0, 1, 2, 3, 4, 5]
        cmd_one_hot = [0] * 6
        cmd_one_hot[command] = 1
        
        cmd_one_hot = np.array(cmd_one_hot)
        
        return cmd_one_hot    
    
    def process_masks(self, masks):
        bev = cv2.cvtColor(BirdViewProducer.as_rgb(masks), cv2.COLOR_RGB2BGR)
        bev = cv2.resize(bev, (256, 256), interpolation=cv2.INTER_AREA)
        bev = cv2.cvtColor(bev, cv2.COLOR_BGR2RGB)  
        return bev

        