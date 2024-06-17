import torch
import torch.nn as nn
import numpy as np 


from my_agents.models.image_encoder.efficient_net_b0 import ImageEncoder
from my_agents.models.vm_encoder.vm_encoder import VMEncoder
from my_agents.models.pid.pid import PIDControllerTCP

MAXIMUM_VELOCITY = 120/3.6

def get_entry_point():
    return 'PRIBOOT'

class PRIBOOT(nn.Module):
    def __init__(self, configs):
        super(PRIBOOT, self).__init__()

        self.configs = configs
        self.device = self.configs['agent']['device']
        self.image_encoder = ImageEncoder(latent_size=self.configs['agent']['bev']['out_dims'], pretrained=self.configs['agent']['bev']['pretrained']).to(self.device)
        self.vm_encoder = VMEncoder(num_inputs=self.configs['agent']['vm']['num_inputs'], fc_dims=self.configs['agent']['vm']['fc_dims'],
                                       out_dims=self.configs['agent']['vm']['out_dims']).to(self.device)
        
        state_size = self.configs['agent']['bev']['out_dims'] + self.configs['agent']['vm']['out_dims']
        
        # waypoints prediction
        gru_hidden_size = self.configs['agent']['gru']['hidden_size']
        speed_dims = self.configs['agent']['speed_dims']
        trajectory_dims = gru_hidden_size * 2
        
        self.trajectory_branch = nn.Sequential(
                                    nn.Linear(state_size, trajectory_dims),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(trajectory_dims, trajectory_dims),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(trajectory_dims, gru_hidden_size),
                                    nn.ReLU(inplace=True),
                                ).to(self.device)

        self.speed_branch = nn.Sequential(
                                nn.Linear(self.configs['agent']['bev']['out_dims'], speed_dims),
                                nn.ReLU(inplace=True),
                                nn.Linear(speed_dims, speed_dims),
                                nn.Dropout(p=0.5),
                                nn.ReLU(inplace=True),
                                nn.Linear(speed_dims, 1)).to(self.device)

        self.decoder_traj = nn.GRUCell(input_size=4, hidden_size=gru_hidden_size).to(self.device)
        self.output_traj = nn.Linear(gru_hidden_size, 2).to(self.device)


        # pid controllers
        self.turn_controller = PIDControllerTCP(K_P=configs['agent']['controller']['turn_KP'], K_I=configs['agent']['controller']['turn_KI'], K_D=configs['agent']['controller']['turn_KD'], n=configs['agent']['controller']['turn_n'])
        self.speed_controller = PIDControllerTCP(K_P=configs['agent']['controller']['speed_KP'], K_I=configs['agent']['controller']['speed_KI'], K_D=configs['agent']['controller']['speed_KD'], n=configs['agent']['controller']['speed_n'])


    def forward(self, data):
        image_features = self.image_encoder(data['bev'])
        
        pred_speed = self.speed_branch(image_features)
        
        vm_features = self.vm_encoder(data['vm'])
        
        state = torch.cat([image_features, vm_features], dim=1)
        
        z = self.trajectory_branch(state)
                
        output_wp = list()

        # initial input variable to GRU
        x = torch.zeros(size=(z.shape[0], 2), dtype=z.dtype).type_as(z).to(self.device)

        # autoregressive generation of output waypoints
        for _ in range(self.configs['agent']['pred_len']):
            x_in = torch.cat([x, data['target_point']], dim=1)
            z = self.decoder_traj(x_in, z)
            dx = self.output_traj(z)
            x = dx + x
            output_wp.append(x)

        pred_waypoints = torch.stack(output_wp, dim=1)
        
        
        return pred_waypoints, pred_speed


    def control_pid(self, waypoints, speed, target):

        assert(waypoints.size(0)==1)
        waypoints = waypoints[0].data.cpu().numpy()
        target = target.squeeze().data.cpu().numpy()


        num_pairs = len(waypoints) - 1

        desired_speed = 0

        for i in range(num_pairs):

            desired_speed += np.linalg.norm(
                    waypoints[i+1] - waypoints[i]) * 2.0 / num_pairs


        aim = (waypoints[1] + waypoints[0]) / 2.0



        angle = np.degrees(np.arctan2(aim[1], aim[0])) / 90
        
        angle_last = angle 
        angle_target = angle
        

        angle_final = angle

        steer = self.turn_controller.step(angle_final)
        steer = np.clip(steer, -1.0, 1.0)


        brake = desired_speed < self.configs['agent']['brake_speed'] or (speed / desired_speed) > self.configs['agent']['brake_ratio']

        delta = np.clip(desired_speed - speed, 0.0, self.configs['agent']['clip_delta'])
        throttle = self.speed_controller.step(delta)
        throttle = np.clip(throttle, 0.0, self.configs['agent']['max_throttle'])
        throttle = throttle if not brake else 0.0


        brake = float(brake)
        throttle = float(throttle)
        steer = float(steer)
        
        metadata = {
            'speed': float(speed.astype(np.float64)),
            'steer': float(steer),
            'throttle': float(throttle),
            'brake': float(brake),
            'aim': tuple(aim.astype(np.float64)),
            'target': tuple(target.astype(np.float64)),
            'desired_speed': float(desired_speed.astype(np.float64)),
            'angle': float(angle.astype(np.float64)),
            'angle_last': float(angle_last.astype(np.float64)),
            'angle_target': float(angle_target.astype(np.float64)),
            'angle_final': float(angle_final.astype(np.float64)),
            'delta': float(delta.astype(np.float64)),
            }

        wp_idx = 1
        for waypoint in waypoints:
            waypoint_tuple = tuple(waypoint.astype(np.float64))
            metadata[f'wp_{wp_idx}'] = waypoint_tuple
            wp_idx += 1

        return steer, throttle, brake, metadata

