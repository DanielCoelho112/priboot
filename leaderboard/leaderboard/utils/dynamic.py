import numpy as np
import leaderboard.utils.transforms as trans_utils
import math
import carla


class ObsManager(object):

    def __init__(self, obs_configs):

        self._max_detection_number = obs_configs['max_detection_number']
        self._distance_threshold = obs_configs['distance_threshold']

        self._parent_actor = None
        self._world = None
       


    def attach_ego_vehicle(self, parent_actor):
        self._parent_actor = parent_actor
        self._world = self._parent_actor.get_world()
       
    def get_observation(self):
        ev_transform = self._parent_actor.get_transform()
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
