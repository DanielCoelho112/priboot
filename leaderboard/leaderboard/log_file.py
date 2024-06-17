import os
import shutil
import json
import cv2

from utilities.visualization import save_video_from_images, save_episode_plot
from utilities.configs import save_config, convert_numpy_dict_to_list_dict, get_config
from utilities.common import recursive_format, format_number

from leaderboard.envs.bev_utils.producer import BirdViewProducer


class LogFile():
    def __init__(self, experiment_path, agent_config, experiment_config, observation_config, overwrite, fps):

        
        if os.path.exists(f'{experiment_path}/evaluation') and not overwrite:
            print(f'{experiment_path}/evaluation already exits. ')
            raise Exception(
                'Experiment name already exists. If you want to overwrite, use flag -ow')
        if os.path.exists(f'{experiment_path}/evaluation'):
            shutil.rmtree(f'{experiment_path}/evaluation')

        self.experiment_path = experiment_path
        self.experiment_name = self.experiment_path.split('/')[-1]
        
        self.log_episode_filename = f"{self.experiment_path}/evaluation/logs/$EPISODE.json"

        self.fps = fps
        self.create_folders()
        
        self.reset_episodes_logfile()

        self.save_config_files(agent_config=agent_config, experiment_config=experiment_config, observation_config=observation_config)

        self.reset_monitor_buffer()
        self.reset_bev_buffer()
        
        self.n_logfile_full = 0
            
    def create_folders(self):
            os.makedirs(f'{self.experiment_path}/evaluation')
            os.makedirs(f'{self.experiment_path}/evaluation/monitor')
            os.makedirs(f'{self.experiment_path}/evaluation/logs')
            os.makedirs(f'{self.experiment_path}/evaluation/plots')
            os.makedirs(f'{self.experiment_path}/evaluation/monitor/bev')
            os.makedirs(f'{self.experiment_path}/configs')

    def reset_episodes_logfile(self):
        self.episodes_logfile = {}

    def reset_monitor_buffer(self):
        self.monitor_buffer = {}
    def reset_bev_buffer(self):
        self.bev_buffer = {}
        
    def reset(self):
        self.reset_episodes_logfile()
        self.reset_monitor_buffer()
        self.reset_bev_buffer()

    def init_episode_log_file(self):
        return {'steps' : {}}

    def save_config_files(self, agent_config, experiment_config, observation_config):

        agent_config_filename = f'{self.experiment_path}/configs/agent.yaml'
        experiment_config_filename = f'{self.experiment_path}/configs/experiment.yaml'
        observation_config_filename = f'{self.experiment_path}/configs/observation.yaml'
        save_config(agent_config, agent_config_filename)
        save_config(experiment_config, experiment_config_filename)
        save_config(observation_config, observation_config_filename)


    def add_data(self, episode, step, observation, waypoints, metadata, control):
        
        # get the first 3 channels.
        image = observation['image'][1][:,:,0:-1]
        bev = observation['birdview']

        if bev.shape[-1] != 3:
            bev = cv2.cvtColor(BirdViewProducer.as_rgb(bev), cv2.COLOR_RGB2BGR)
            bev = cv2.resize(bev, (256, 256), interpolation=cv2.INTER_AREA)
            bev = cv2.cvtColor(bev, cv2.COLOR_BGR2RGB) 
        else:
            bev = cv2.cvtColor(bev, cv2.COLOR_RGB2BGR)
        
        
        if episode not in self.episodes_logfile.keys():
            self.episodes_logfile[episode] = self.init_episode_log_file()
        if episode not in self.monitor_buffer.keys():
            self.monitor_buffer[episode] = []
            self.bev_buffer[episode] = []
        
        filtered_obs = self.filter_obs(observation=observation)
        
        control_dict = {'throttle':float(control.throttle), 'steer':float(control.steer), 'brake':float(control.brake)}        
        
        if waypoints != None:
            waypoints = waypoints.cpu().numpy().tolist()
        
        self.episodes_logfile[episode]['steps'][step] = recursive_format({'observation': filtered_obs,
                                                                          'waypoints': waypoints,
                                                                          'metadata':metadata,
                                                                          'control': control_dict}, format_number)

        
        self.monitor_buffer[episode].append(image)
        self.bev_buffer[episode].append(bev)
        
            
    def save_episodes(self, eval_idx):
        self.save_episodes_log_file()
        self.save_episode_video_monitor()
        self.save_episode_video_bev()

        
        self.reset_episodes_logfile()
        self.reset_monitor_buffer()
        self.reset_bev_buffer()

    def save_episode_video_monitor(self):
        for episode, images in self.monitor_buffer.items():
            video_name = f"{self.experiment_path}/evaluation/monitor/{episode:04d}.mp4"
            save_video_from_images(video_name, images=images, fps=self.fps)

    def save_episode_video_bev(self):
        for episode, images in self.bev_buffer.items():
            video_name = f"{self.experiment_path}/evaluation/monitor/bev/{episode:04d}.mp4"
            save_video_from_images(video_name, images=images, fps=self.fps)
               
    def save_episodes_log_file(self):
        for episode, episode_logfile in self.episodes_logfile.items():
            self.save_logfile(logfile=episode_logfile, filename= self.log_episode_filename.replace('$EPISODE', f'{episode:04d}'))

    def save_logfile(self, logfile, filename):
        with open(filename, "w") as f:
            json.dump(logfile, f, indent=4)
     

    def filter_obs(self, observation):
        speed = observation['speed']
        control = observation['control']
        block_time = observation['block_time']
        target_point = observation['target_point']


        filtered_obs = {'speed': float(speed),
                        'control': convert_numpy_dict_to_list_dict(control),
                        'block_time': float(block_time),
                        'target_point': target_point.tolist()}

        
        return filtered_obs


    def is_full(self):
        if len(list(self.monitor_buffer.keys())) > self.n_logfile_full:
            return True
        return False
    
    def save_metrics(self, metrics, steps):
        if steps % self.log_every == 0:                     
            self.save_logfile(logfile=metrics, filename=self.metrics_step_filename.replace('$STEP', f'{steps:07d}'))
            
            if steps % (self.log_every * 10) == 0:
                self.save_plot_metrics()
                
    def save_plot_metrics(self):
        logs_dir = f"{self.experiment_path}/logs/training"
        files = sorted(filter(lambda x: os.path.isfile(os.path.join(logs_dir, x)),
                    os.listdir(logs_dir)))

        data = {'steps':[]}
        for file in files:
            metrics = get_config(f"{logs_dir}/{file}")
            for key, value in metrics.items():
                if key in data:
                    data[key].append(value)
                else:
                    data[key] = [value]
            if len(metrics.keys()) > 0: 
                data['steps'].append(int(file.split('.')[0]))
        
        for key, value in data.items():
            if key == 'steps':
                continue 
            save_episode_plot(x_array=data['steps'], y_array=value, xlabel='Steps', ylabel=key, vertical_line=None, filename=f"{logs_dir}/plots/{key}.png")
            