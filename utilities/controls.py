import carla
import numpy as np
import torch

def carla_control(action):
    control = carla.VehicleControl(
        throttle= float(action[0]), steer=float(action[1]), brake=float(action[2]))
    return control


def acc_controller(action):
    """
    converting acceleration into throttle and brake.
    """
    acc = action[0]
    steer = action[1]
    
    if acc>= 0.0:
        throttle = acc 
        brake = 0.0
    else:
        throttle = 0.0
        brake = np.abs(acc)
        
    if brake < 0.05:
        brake = 0.0
    
    throttle = np.clip(throttle, 0, 1)
    brake = np.clip(brake, 0, 1)
    steer = np.clip(steer, -1, 1)
    
    return [throttle, steer, brake]
    