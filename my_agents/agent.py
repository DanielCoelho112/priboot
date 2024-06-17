import importlib
import torch
import carla
from torchvision.models import EfficientNet_B0_Weights
import numpy as np

from leaderboard.autoagents.autonomous_agent import AutonomousAgent
from leaderboard.utils.hazard_actor import lbc_hazard_walker, lbc_hazard_construction, challenge_hazard_vehicle2

MAXIMUM_VELOCITY = 120/3.6

EMERGENCY_DISTANCE_VEHICLE = 7.0
EMERGENCY_DISTANCE_PEDESTRIAN = 5.5
EMERGENCY_DISTANCE_CONSTRUCTION = 4.0

def get_entry_point():
    return 'ILAgent'

class ILAgent(AutonomousAgent):
    def __init__(self):
        super(ILAgent, self).__init__()
    
    def setup(self, configs):
        super(ILAgent, self).setup(configs)
        
        self.configs = configs
        
        agent_entrypoint = configs['agent']['entrypoint_model']
        module_agent = importlib.import_module(agent_entrypoint)
        agent_class_name = getattr(module_agent, 'get_entry_point')()
        agent_class_obj = getattr(module_agent, agent_class_name)
        self.model= agent_class_obj(self.configs)
        
        self.model.load_state_dict(torch.load(f"{self.configs['experiment']['experiment_path']}/weights/model.pt", map_location=self.configs['agent']['device']))
        self.model.eval()
        

        self.transforms_base = EfficientNet_B0_Weights.IMAGENET1K_V1.transforms()
        
        self.step = -1
        self.stuck_detector = 0
        self.forced_move = 0
    
    
    def sensors(self):
        
        sensors = [self.configs['observation']['image'], {'type': 'sensor.birdview', 'reading_frequency': 1, 'id' : 'birdview'}]

        return sensors
        
    
    def filter_obs(self, obs):
        obs_ = {}

        speed = obs['speed']

        block_time = obs['block_time']
        
        speed_limit = obs['control']['speed_limit']

        obs_['target_point'] = obs['target_point']
        
        obs_['command'] = obs['command']
        
        obs_['speed'] = np.array([speed])

        obs_['vm'] = np.concatenate([speed, speed_limit, block_time, obs_['target_point'], obs_['command']])
            

        obs_['bev'] = obs['birdview']

        obs_['bev'] = obs_['bev'].transpose(2, 0, 1)
        
        return obs_

    def preprocess_policy(self, obs):
        obs['bev'] = np.expand_dims(obs['bev'], 0)
        obs['vm'] = np.expand_dims(obs['vm'], 0)
        obs['target_point'] = np.expand_dims(obs['target_point'], 0)

        obs["bev"] = torch.from_numpy(obs["bev"]).float().div(255.0)

        obs = {k: torch.Tensor(v).to(self.configs['agent']['device']) for k, v in obs.items()}

        if self.transforms_base:
            obs["bev"] = self.transforms_base(obs["bev"])
        
        return obs
    
    @torch.inference_mode()
    def run_step(self, input_data):
        self.input_data = input_data
        
        obs = self.filter_obs(obs=input_data)
        obs = self.preprocess_policy(obs=obs)
        
        current_speed = obs['speed'].cpu().numpy()[0] * MAXIMUM_VELOCITY
        
        emergency, dist_veh, dist_ped, dist_dyn= self.is_emergency(current_speed)
        
        slow_down = False
        if current_speed >= 3.0:
            if dist_veh > 7.0 and dist_veh <= 15.0:
                slow_down = True
                multiplier_veh = 0.5 + ((1.0 - 0.5) * (dist_veh - 7.0) / (15.0 - 7.0))
            else:
                multiplier_veh = 1.0
            if dist_ped > 5.5 and dist_ped <= 12.5:
                slow_down = True
                multiplier_ped = 0.1 + ((0.6 - 0.1) * (dist_ped - 5.5) / (12.5 - 5.5))
            else:
                multiplier_ped = 1.0
            if dist_dyn > 4.0 and dist_dyn <= 5.0:
                slow_down = True
                multiplier_dyn = 0.5 + ((1.0 - 0.5) * (dist_dyn - 4.0) / (5.0 - 4.0))
            else:
                multiplier_dyn = 1.0
                
            multiplier_slow_dow = min(multiplier_veh, multiplier_ped, multiplier_dyn)
        
        pred_waypoints, _ = self.model(obs)
        
        
        is_stuck = False
        green_push = False
        if self.configs['agent']['use_stuck']:
            if(self.stuck_detector > self.configs['agent']['stuck_threshold']):
                
                if self.tl_node.in_traffic_light:
                    
                    if self.tl_node.traffic_light_color == 'GREEN':
                        is_stuck = True
                        green_push = True
                        
                    else:
                        is_stuck = False
                        emergency = True
                        
                else:
                    is_stuck = True

            
        if self.tl_node.in_traffic_light: 
            if self.tl_node.traffic_light_color != 'GREEN':                    
                emergency = True
                

        if self.configs['agent']['use_stuck'] and is_stuck:
            num_waypoints = self.configs['agent']['pred_len']
            x_pos = torch.linspace(0, self.configs['agent']['x_creep'], steps=num_waypoints).unsqueeze(1)
            y_pos = torch.zeros(num_waypoints, 1)
            pred_waypoints = torch.cat((x_pos, y_pos), dim=1).unsqueeze(0)
        
        if self.configs['agent']['use_stuck']:
            if current_speed < 0.1:
                self.stuck_detector += 1
            elif current_speed >= self.configs['agent']['velocity_creep']:
                self.stuck_detector = 0

            
        steer, throttle, brake, metadata = self.model.control_pid(waypoints=pred_waypoints, speed=current_speed, target=obs['target_point'])

        if self.configs['agent']['use_stuck']:
            metadata['is_stuck'] = is_stuck 
            metadata['green_push'] = green_push
            metadata['is_traffic_light'] = self.tl_node.in_traffic_light
            metadata['traffic_light_color'] = self.tl_node.traffic_light_color 
        
        if brake < 0.05: brake = 0.0
        if throttle > brake: brake = 0.0
        if slow_down and not is_stuck: throttle = throttle * multiplier_slow_dow

        control = carla.VehicleControl()
        control.steer = np.clip(steer, -1, 1)
        control.throttle = np.clip(throttle, 0, self.configs['agent']['max_throttle'])
        control.brake = np.clip(brake, 0, 1)


        if emergency and not is_stuck:
            control.throttle = 0.0
            control.brake = 1.0
            control.steer = 0.0
            self.forced_move = 0
            metadata['emergency'] = True
            pred_waypoints = torch.zeros_like(pred_waypoints)
        else:
            metadata['emergency'] = False

        if control.brake > 0.5:
            control.throttle = float(0)

        self.step += 1

        metadata['dist_vehicle'] = dist_veh
        metadata['dist_pedestrian'] = dist_ped
        metadata['dist_dynamic'] = dist_dyn
        metadata['slow_down'] = slow_down
        
        return control, pred_waypoints, metadata
    
    def get_obs(self):
        return self.input_data
    
    def is_emergency(self, speed):
        
        obs_vehicle = self.om_vehicle.get_observation()
        obs_pedestrian = self.om_pedestrian.get_observation()
        obs_construction = self.om_construction.get_observation()


        hazard_vehicle_loc = challenge_hazard_vehicle2(obs_vehicle, ev_speed=speed)

        hazard_ped_loc = lbc_hazard_walker(obs_pedestrian)
        hazard_construction_loc = lbc_hazard_construction(obs_construction)
    
        dist_veh = 100
        if hazard_vehicle_loc is not None:
            dist_veh = float(max(0.0, np.linalg.norm(hazard_vehicle_loc[0:2])))
        
        dist_ped = 100
        if hazard_ped_loc is not None:
            dist_ped = float(max(0.0, np.linalg.norm(hazard_ped_loc[0:2])))

            
        dist_dyn = 100
        if hazard_construction_loc is not None:
            dist_dyn = float(max(0.0, np.linalg.norm(hazard_construction_loc[0:2])))
        
        if dist_veh < EMERGENCY_DISTANCE_VEHICLE or dist_ped < EMERGENCY_DISTANCE_PEDESTRIAN or dist_dyn < EMERGENCY_DISTANCE_CONSTRUCTION:
            return True, dist_veh, dist_ped, dist_dyn
        else:  
            return False, dist_veh, dist_ped, dist_dyn
