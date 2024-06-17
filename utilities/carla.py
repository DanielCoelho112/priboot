import numpy as np

def get_forward_speed(vehicle):
    """ Convert the vehicle transform directly to forward speed """
    
    velocity = vehicle.get_velocity()
    transform = vehicle.get_transform()

    vel_np = np.array([velocity.x, velocity.y, velocity.z])
    pitch = np.deg2rad(transform.rotation.pitch)
    yaw = np.deg2rad(transform.rotation.yaw)
    orientation = np.array([np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), np.sin(pitch)])
    speed = np.dot(vel_np, orientation)
    return speed

def generate_routes_subset(num_routes, eval_avg_episodes):
    if eval_avg_episodes < 2 or num_routes < 2:
        return "Invalid input (both numbers should be 2 or more)"

    # Generate evenly spaced indices
    step = (num_routes - 1) / (eval_avg_episodes - 1)
    indices = [int(round(i * step)) for i in range(eval_avg_episodes)]

    # Joining indices with a comma
    return ','.join(map(str, indices))