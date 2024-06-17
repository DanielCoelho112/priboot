# PRIBOOT

PRIBOOT is a new data-driven expert system designed to tackle CARLA Leaderboard 2.0. This repository contains the implementation details and instructions for setting up and running PRIBOOT, as described in the paper:

> [**PRIBOOT: A New Data-Driven Expert for Improved Driving Simulations**](https://arxiv.org/abs/2406.08421)

>
> Daniel Coelho, Miguel Oliveira, Vítor Santos, and Antonio M. López


If you find our work useful, please consider citing: 

```bibtex
@article{coelho2024priboot,
  title={PRIBOOT: A New Data-Driven Expert for Improved Driving Simulations},
  author={Coelho, Daniel and Oliveira, Miguel and Santos, Vitor and Lopez, Antonio M},
  journal={arXiv preprint arXiv:2406.08421},
  year={2024}
}
```
  

## Setup
1. Clone the repository with `git clone git@github.com:DanielCoelho112/priboot.git `

2. Download the folder [birdview_cache](https://uapt33090-my.sharepoint.com/:f:/g/personal/danielsilveiracoelho_ua_pt/EsLVgW8cMpdDnyslMkvurrkB2evUbXEwMJtDHzu3c_Vzdw?e=xNnk7I) and place it in the root directory of the PRIBOOT repository.

3. Create a folder to store the results. In that folder place the folder [priboot_original](https://uapt33090-my.sharepoint.com/:f:/g/personal/danielsilveiracoelho_ua_pt/EsLVgW8cMpdDnyslMkvurrkB2evUbXEwMJtDHzu3c_Vzdw?e=xNnk7I) which contains the weights of the model.

4. Download [CARLA 0.9.15](https://github.com/carla-simulator/carla/releases/tag/0.9.15).

5. Run the docker container with `docker run -it --gpus all --network=host -v results_path:/root/results/priboot -v priboot_path:/root/priboot danielc11/priboot:0.0 bash`
where `results_path` is the path where the results will be written, and `priboot_path` is the path of the PRIBOOT repository.


## Evaluating the agent
1. Start the CARLA server

2. Open the file priboot/config/priboot_original/experiment.yaml and configure the various options according to your requirements.

3. Run: `python3 priboot/leaderboard/leaderboard/leaderboard_evaluator.py -en priboot_original`
