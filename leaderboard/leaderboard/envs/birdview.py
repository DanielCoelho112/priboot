import cv2 as cv 
import time

from leaderboard.envs.sensor_interface import BaseReader

from leaderboard.envs.bev_utils.producer import BirdViewProducer, BirdViewCropType
from leaderboard.envs.bev_utils.mask import PixelDimensions

from utilities.velocity import compute_velocities, get_positions

DEFAULT_HEIGHT = 1400  
DEFAULT_WIDTH = 800
DEFAULT_CROP_TYPE = BirdViewCropType.FRONT_AND_REAR_AREA


class BirdviewReader(BaseReader):
    """
    Sensor to compute the Birdview image
    """
    MAX_CONNECTION_ATTEMPTS = 10
    
    def __init__(self, vehicle, reading_frequency=1, configs=None, criteria_node=None, agent=None, client=None, cache_dict={}):
        
        self.birdview_producer = BirdViewProducer(
            client,
            agent,
            vehicle,
            PixelDimensions(width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT),
            pixels_per_meter=10,
            crop_type=BirdViewCropType.FRONT_AND_REAR_AREA,
            render_lanes_on_junctions=False,
            cache_dict=cache_dict,
            criteria_node=criteria_node,
        )
        
        # init positions and velocities.
        self.current_positions = get_positions(self.birdview_producer._world)
        self.old_positions = {}
        self.velocities = {}
        
        super().__init__(vehicle, reading_frequency)


    def __call__(self):
        self.old_positions = self.current_positions
        self.current_positions = get_positions(self.birdview_producer._world)
        self.velocities = compute_velocities(self.current_positions, self.old_positions)
        
        birdview = self.birdview_producer.produce(agent_vehicle=self._vehicle, velocities=self.velocities)
        birdview = birdview[:1000, :, :]
        rgb =  BirdViewProducer.as_rgb(birdview)

        return {'birdview': rgb, 'masks':birdview}

