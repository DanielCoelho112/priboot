import numpy as np
import leaderboard.utils.transforms as trans_utils
import math
import carla

from leaderboard.envs.bev_utils.actors import CONSTRUCTION_TYPES


class ObsManager(object):

    def __init__(self, obs_configs):

        self._max_detection_number = obs_configs['max_detection_number']
        self._distance_threshold = obs_configs['distance_threshold']

        self._parent_actor = None
        self._world = None
       


    def attach_ego_vehicle(self, parent_actor):
        self._parent_actor = parent_actor
        self._world = self._parent_actor.get_world()
        self._map = self._world.get_map()
       
    def get_observation(self):
        ev_transform = self._parent_actor.get_transform()
        ev_location = ev_transform.location
        def dist_to_actor(w): return w.get_location().distance(ev_location)

        surrounding_objects = []
        construction_list = []
        
        for actor in self._world.get_actors():
            for construction_type in CONSTRUCTION_TYPES:
                if construction_type in actor.type_id:
                    construction_list.append(actor)
                    break
        
        for construction in construction_list:
            if dist_to_actor(construction) <= self._distance_threshold:
                surrounding_objects.append(construction)

        sorted_surrounding_objects = sorted(surrounding_objects, key=dist_to_actor)

        location, rotation, absolute_velocity, _ = trans_utils.get_loc_rot_vel_in_ev(
            sorted_surrounding_objects, ev_transform)

        binary_mask, extent, on_sidewalk, road_id, lane_id = [], [], [], [], []
        for ped in sorted_surrounding_objects[:self._max_detection_number]:
            binary_mask.append(1)

            bbox_extent = ped.bounding_box.extent
            extent.append([bbox_extent.x, bbox_extent.y, bbox_extent.z])

            loc = ped.get_location()
            wp = self._map.get_waypoint(loc, project_to_road=False, lane_type=carla.LaneType.Driving)
            if wp is None:
                on_sidewalk.append(1)
            else:
                on_sidewalk.append(0)
            wp = self._map.get_waypoint(loc)
            road_id.append(wp.road_id)
            lane_id.append(wp.lane_id)

        for i in range(self._max_detection_number - len(binary_mask)):
            binary_mask.append(0)
            location.append([0, 0, 0])
            rotation.append([0, 0, 0])
            absolute_velocity.append([0, 0, 0])
            extent.append([0, 0, 0])
            on_sidewalk.append(0)
            road_id.append(0)
            lane_id.append(0)

        obs_dict = {
            'frame': self._world.get_snapshot().frame,
            'binary_mask': np.array(binary_mask, dtype=np.int8),
            'location': np.array(location, dtype=np.float32),
            'rotation': np.array(rotation, dtype=np.float32),
            'absolute_velocity': np.array(absolute_velocity, dtype=np.float32),
            'extent': np.array(extent, dtype=np.float32),
            'on_sidewalk': np.array(on_sidewalk, dtype=np.int8),
            'road_id': np.array(road_id, dtype=np.int8),
            'lane_id': np.array(lane_id, dtype=np.int8)
        }

        return obs_dict

    def clean(self):
        self._parent_actor = None
        self._world = None
        self._map = None
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        def dist_to_actor(w):
            return math.sqrt((w.location.x - ev_transform.location.x) ** 2 +
                            (w.location.y - ev_transform.location.y) ** 2 +
                            (w.location.z - ev_transform.location.z) ** 2)

        obs_dynamic = []
        dynamic_bbox_list = self._world.get_level_bbs(carla.CityObjectLabel.Dynamic)
        
        for dynamic_bbox in dynamic_bbox_list:
            distance = dist_to_actor(dynamic_bbox)
            if distance <= self._distance_threshold:
                obs_dynamic.append(dynamic_bbox)
        
        obs_dynamic = sorted(obs_dynamic, key=dist_to_actor)[:self._max_detection_number]
        
        location, rotation = trans_utils.get_loc_rot_in_ev(
            obs_dynamic, ev_transform)
        
        obs_dynamic = {
            'frame': self._world.get_snapshot().frame,
            'binary_mask': np.array([], dtype=np.int8),
            'location': np.array(location, dtype=np.float32),
            'rotation': np.array(rotation, dtype=np.float32),
        }
        
        return obs_dynamic

    def clean(self):
        self._parent_actor = None
        self._world = None
