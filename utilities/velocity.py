
import carla 

FPS = 20 # 10
VELOCITY_THRESHOLD = 0.1
SPEED_TELEPORT = 35


def estimate_velocity_xyz(pos1, pos2):

    # Time difference between frames
    time_diff = 1 / FPS

    # Velocity components
    vx = (pos2[0] - pos1[0]) / time_diff
    vy = (pos2[1] - pos1[1]) / time_diff
    vz = (pos2[2] - pos1[2]) / time_diff
    
    if vx > SPEED_TELEPORT or vy > SPEED_TELEPORT or vz > SPEED_TELEPORT:
        vx = 0.0
        vy = 0.0
        vz = 0.0

    return carla.Vector3D(vx, vy, vz)

def compute_velocities(current_positions, old_positions):
    velocities = {}
    for actor_id in current_positions:
        if actor_id in old_positions:
            pos2 = current_positions[actor_id]
            pos1 = old_positions[actor_id]
            velocity = estimate_velocity_xyz(pos1, pos2)
            velocities[actor_id] = velocity
        else:
            velocities[actor_id] = carla.Vector3D(0.0, 0.0, 0.0)
    return velocities

def get_positions(world):
    vehicles = world.get_actors().filter('vehicle.*')
    pedestrians = world.get_actors().filter('walker.pedestrian.*')
    all_actors = list(vehicles) + list(pedestrians)
    
    positions = {}
    for actor in all_actors:
        location = actor.get_location()
        positions[actor.id] = (location.x, location.y, location.z)
    return positions
