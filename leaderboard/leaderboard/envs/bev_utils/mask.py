import carla
import numpy as np
import os 
import pickle


from typing import NamedTuple, List, Tuple, Optional
from cv2 import cv2 as cv

Mask = np.ndarray  # of shape (y, x), stores 0 and 1, dtype=np.int32
RoadSegmentWaypoints = List[carla.Waypoint]

COLOR_OFF = 0
COLOR_ON = 1


class Coord(NamedTuple):
    x: int
    y: int


class Dimensions(NamedTuple):
    width: int
    height: int


PixelDimensions = Dimensions
Pixels = int
Meters = float
Canvas2D = np.ndarray  # of shape (y, x)

MAP_BOUNDARY_MARGIN: Meters = 300
MAXIMUM_SPEED = 120/3.6 # m/s
MAXIMUM_ARROW_LENGTH = 90
Z_DIST_PED = 25
PEDESTRIAN_SCALE = 5.0

class MapBoundaries(NamedTuple):
    """Distances in carla.World coordinates"""

    min_x: Meters
    min_y: Meters
    max_x: Meters
    max_y: Meters


class CroppingRect(NamedTuple):
    x: int
    y: int
    width: int
    height: int

    @property
    def vslice(self) -> slice:
        return slice(self.y, self.y + self.height)

    @property
    def hslice(self) -> slice:
        return slice(self.x, self.x + self.width)


def lateral_shift(transform, shift):
    """Makes a lateral shift of the forward vector of a transform"""
    transform.rotation.yaw += 90
    return transform.location + shift * transform.get_forward_vector()


class RenderingWindow(NamedTuple):
    origin: carla.Location
    area: PixelDimensions


class MapMaskGenerator:
    """Generates 2D, top-down representations of a map.

    Each mask covers area specified by rendering window or whole map (when rendering window is disabled).
    Note that layer, mask, canvas are somewhat interchangeable terms for the same thing.

    Rendering is implemented using OpenCV, so it can be easily adjusted
    to become a regular RGB renderer (just change all `color` arguments to 3-element tuples)
    """

    def __init__(
        self, client, agent, vehicle, pixels_per_meter: int, render_lanes_on_junctions: bool
    ) -> None:
        self.client = client
        self.agent = agent
        self.vehicle = vehicle
        self.pixels_per_meter = pixels_per_meter
        self.rendering_window: Optional[RenderingWindow] = None

        self._world = client.get_world()

        self._map = self._world.get_map()
        
        town_name = self._map.name.split('/')[-1]
        
        with open(f'{os.getenv("PRIBOOT_ROOT")}/birdview_cache/Carla/Maps/{town_name}/{town_name}_boundaries.pkl', 'rb') as file:
            self._map_boundaries = pickle.load(file)  
              
        
        self._mask_size: PixelDimensions = self.calculate_mask_size()

        self._render_lanes_on_junctions = render_lanes_on_junctions

    def calculate_mask_size(self) -> PixelDimensions:
        """Convert map boundaries to pixel resolution."""
        width_in_meters = self._map_boundaries.max_x - self._map_boundaries.min_x
        height_in_meters = self._map_boundaries.max_y - self._map_boundaries.min_y
        width_in_pixels = int(width_in_meters * self.pixels_per_meter)
        height_in_pixels = int(height_in_meters * self.pixels_per_meter)
        return PixelDimensions(width=width_in_pixels, height=height_in_pixels)

    def disable_local_rendering_mode(self):
        self.rendering_window = None

    def enable_local_rendering_mode(self, rendering_window: RenderingWindow):
        self.rendering_window = rendering_window

    def location_to_pixel(self, loc: carla.Location) -> Coord:
        """Convert world coordinates to pixel coordinates.

        For example: top leftmost location will be a pixel at (0, 0).
        """
        min_x = self._map_boundaries.min_x
        min_y = self._map_boundaries.min_y

        # Pixel coordinates on full map
        x = round(self.pixels_per_meter * (loc.x - min_x))
        y = round(self.pixels_per_meter * (loc.y - min_y))

        if self.rendering_window is not None:
            # global rendering area coordinates
            origin_x = self.pixels_per_meter * (self.rendering_window.origin.x - min_x)            
            origin_y = self.pixels_per_meter * (self.rendering_window.origin.y - min_y)
            topleft_x = round(origin_x - self.rendering_window.area.width / 2)
            topleft_y = round(origin_y - self.rendering_window.area.height / 2)

            # x, y becomes local coordinates within rendering window
            x -= topleft_x
            y -= topleft_y
            
        return Coord(x=round(x), y=round(y))

    def make_empty_mask(self) -> Mask:
        if self.rendering_window is None:
            shape = (self._mask_size.height, self._mask_size.width)
        else:
            shape = (
                self.rendering_window.area.height,
                self.rendering_window.area.width,
            )
        return np.zeros(shape, np.uint8)

    def agent_vehicle_mask(self, agent: carla.Actor) -> Mask:
        canvas = self.make_empty_mask()
        bb = agent.bounding_box.extent
        corners = [
            carla.Location(x=-bb.x, y=-bb.y),
            carla.Location(x=bb.x, y=-bb.y),
            carla.Location(x=bb.x, y=bb.y),
            carla.Location(x=-bb.x, y=bb.y),
        ]

        agent.get_transform().transform(corners)
        corners = [self.location_to_pixel(loc) for loc in corners]
        cv.fillPoly(img=canvas, pts=np.int32([corners]), color=COLOR_ON)
        
        return canvas

    def arrows_mask(self, agent, segregated_actors, velocities):
        canvas = self.make_empty_mask()
        
        def draw_arrow(actor, small=False):
            location = actor.get_location()
            
            location_pix = self.location_to_pixel(location)
        
            velocity = velocities[actor.id]
            v_x = velocity.x
            v_y = velocity.y
            
            if v_x != 0 or v_y != 0:
                normalized_velocity = [v_x / MAXIMUM_SPEED, v_y / MAXIMUM_SPEED]
            else:
                normalized_velocity = [0, 0]
            

            arrow_initpoint = (round(location_pix.x), round(location_pix.y))
            arrow_endpoint = (round(location_pix.x + normalized_velocity[0] * MAXIMUM_ARROW_LENGTH),
                        round(location_pix.y + normalized_velocity[1] * MAXIMUM_ARROW_LENGTH))

            if small:
                constant_tip_size = 3.0
                thickness = 1
            else:
                constant_tip_size = 5.0
                thickness = 2
                
            line_length = np.sqrt((arrow_endpoint[0] - arrow_initpoint[0]) ** 2 + (arrow_endpoint[1] - arrow_initpoint[1]) ** 2)
            
            if line_length > 0.0:
                tip_length = constant_tip_size / line_length
            else:
                tip_length = 0.0
            
 
            cv.arrowedLine(canvas, arrow_initpoint,
                        arrow_endpoint, 1, thickness=thickness, tipLength=tip_length)

            
        draw_arrow(agent)
        for veh in segregated_actors.vehicles:
            if self.discard_underground(veh):
                continue
            draw_arrow(veh)
        for bike in segregated_actors.bikes:
            if self.discard_underground(bike):
                continue
            draw_arrow(bike, small=True)
        for emergency in segregated_actors.emergencys:
            if self.discard_underground(emergency):
                continue
            draw_arrow(emergency)
        for pedestrian in segregated_actors.pedestrians:
            if self.discard_underground(pedestrian):
                continue
            draw_arrow(pedestrian, small=True)        
      
        return canvas
        

    def vehicles_mask(self, vehicles: List[carla.Actor]) -> Mask:
        canvas = self.make_empty_mask()
        for veh in vehicles:
            if self.discard_underground(veh):
                continue
            bb = veh.bounding_box.extent
            corners = [
                carla.Location(x=-bb.x, y=-bb.y),
                carla.Location(x=bb.x, y=-bb.y),
                carla.Location(x=bb.x, y=bb.y),
                carla.Location(x=-bb.x, y=bb.y),
            ]

            veh.get_transform().transform(corners)
            corners = [self.location_to_pixel(loc) for loc in corners]

            if veh.attributes["role_name"] == "hero":
                color = COLOR_OFF
            else:
                color = COLOR_ON

            cv.fillPoly(img=canvas, pts=np.int32([corners]), color=color)
        return canvas

    def pedestrians_mask(self, pedestrians: List[carla.Actor], agent_vehicle) -> Mask:
        canvas = self.make_empty_mask()
        for ped in pedestrians:
            if self.discard_underground(ped):
                continue
            if not hasattr(ped, "bounding_box"):
                continue
                        
            bb = ped.bounding_box.extent
            corners = [
                carla.Location(x=-bb.x*PEDESTRIAN_SCALE, y=-bb.y*PEDESTRIAN_SCALE),
                carla.Location(x=bb.x*PEDESTRIAN_SCALE, y=-bb.y*PEDESTRIAN_SCALE),
                carla.Location(x=bb.x*PEDESTRIAN_SCALE, y=bb.y*PEDESTRIAN_SCALE),
                carla.Location(x=-bb.x*PEDESTRIAN_SCALE, y=bb.y*PEDESTRIAN_SCALE),
            ]
            
    
            ped.get_transform().transform(corners)
            corners = [self.location_to_pixel(loc) for loc in corners]
            cv.fillPoly(img=canvas, pts=np.int32([corners]), color=COLOR_ON)
        return canvas

    def construction_mask(self, pedestrians: List[carla.Actor], debug=False) -> Mask:
        canvas = self.make_empty_mask()
        for ped in pedestrians:
            if not hasattr(ped, "bounding_box"):
                continue

            bb = ped.bounding_box.extent
            corners = [
                carla.Location(x=-bb.x, y=-bb.y),
                carla.Location(x=bb.x, y=-bb.y),
                carla.Location(x=bb.x, y=bb.y),
                carla.Location(x=-bb.x, y=bb.y),
            ]
            
            ped.get_transform().transform(corners)
            corners = [self.location_to_pixel(loc) for loc in corners]
            cv.fillPoly(img=canvas, pts=np.int32([corners]), color=COLOR_ON)
        return canvas

    def traffic_lights_masks(self, traffic_lights: List[carla.Actor]) -> Tuple[Mask]:
        red_light_canvas = self.make_empty_mask()
        yellow_light_canvas = self.make_empty_mask()
        green_light_canvas = self.make_empty_mask()
        tls = carla.TrafficLightState
        for tl in traffic_lights:
            world_pos = tl.get_location()
            pos = self.location_to_pixel(world_pos)
            radius = int(self.pixels_per_meter * 1.2)
            if tl.state == tls.Red:
                target_canvas = red_light_canvas
            elif tl.state == tls.Yellow:
                target_canvas = yellow_light_canvas
            elif tl.state == tls.Green:
                target_canvas = green_light_canvas
            else:
                # Unknown or off traffic light
                continue

            cv.circle(
                img=target_canvas,
                center=pos,
                radius=radius,
                color=COLOR_ON,
                thickness=cv.FILLED,
            )
        return red_light_canvas, yellow_light_canvas, green_light_canvas

    def stops_mask(self, stops) -> Tuple[Mask]:
        canvas = self.make_empty_mask()

        for actor_transform, bb_loc, bb_ext in stops:

            corners = [carla.Location(x=-bb_ext.x, y=-bb_ext.y),
                       carla.Location(x=bb_ext.x, y=-bb_ext.y),
                       carla.Location(x=bb_ext.x, y=0),
                       carla.Location(x=bb_ext.x, y=bb_ext.y),
                       carla.Location(x=-bb_ext.x, y=bb_ext.y)]
            corners = [bb_loc + corner for corner in corners]

            corners = [actor_transform.transform(corner) for corner in corners]
            corners_in_pixel = np.array([[self.location_to_pixel(corner)] for corner in corners])

            cv.fillConvexPoly(canvas, np.round(corners_in_pixel).astype(np.int32), 1)
        
        return canvas
    
    def route_mask(self):        
        canvas = self.make_empty_mask()
        route_in_pixel = np.array([[self.location_to_pixel(wp.location)]
                                   for wp, _ in self.agent._global_route[0:150]])
        
        cv.polylines(canvas, [np.round(route_in_pixel).astype(np.int32)], False, 1, thickness=30)
        canvas = canvas.astype(np.bool)
        
        return canvas

    def discard_underground(self, actor):
        actor_location = actor.get_location()
        agent_location = self.vehicle.get_location()
        z_dist = np.abs(actor_location.z - agent_location.z)
        # sometimes there are pedestrians 100 meters below the ground that can be projected into the BEV.
        if z_dist >= Z_DIST_PED:
            return True 
        else:
            return False
    



