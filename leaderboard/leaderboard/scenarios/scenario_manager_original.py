#!/usr/bin/env python

# Copyright (c) 2018-2020 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides the ScenarioManager implementations.
It must not be modified and is for reference only!
"""

from __future__ import print_function
import signal
import sys
import time

import py_trees
import carla
import threading

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.timer import GameTime
from srunner.scenariomanager.watchdog import Watchdog

from leaderboard.autoagents.agent_wrapper import AgentWrapperFactory, AgentError
from leaderboard.envs.sensor_interface import SensorReceivedNoData
from leaderboard.utils.result_writer import ResultOutputProvider

from leaderboard.utils.vehicle import ObsManager as OmVehicle
from leaderboard.utils.pedestrian import ObsManager as OmPedestrian
from leaderboard.utils.construction import ObsManager as OmConstruction


EMERGENCY_DISTANCE = 50

class ScenarioManager(object):

    """
    Basic scenario manager class. This class holds all functionality
    required to start, run and stop a scenario.

    The user must not modify this class.

    To use the ScenarioManager:
    1. Create an object via manager = ScenarioManager()
    2. Load a scenario via manager.load_scenario()
    3. Trigger the execution of the scenario manager.run_scenario()
       This function is designed to explicitly control start and end of
       the scenario execution
    4. If needed, cleanup with manager.stop_scenario()
    """

    def __init__(self, timeout, statistics_manager, debug_mode=0):
        """
        Setups up the parameters, which will be filled at load_scenario()
        """
        self.route_index = None
        self.scenario = None
        self.scenario_tree = None
        self.ego_vehicles = None
        self.other_actors = None

        self._debug_mode = debug_mode
        self._agent_wrapper = None
        self._running = False
        self._timestamp_last_run = 0.0
        self._timeout = float(timeout)

        self.scenario_duration_system = 0.0
        self.scenario_duration_game = 0.0
        self.start_system_time = 0.0
        self.start_game_time = 0.0
        self.end_system_time = 0.0
        self.end_game_time = 0.0

        self._watchdog = None
        self._agent_watchdog = None
        self._scenario_thread = None

        self._statistics_manager = statistics_manager

        # Use the callback_id inside the signal handler to allow external interrupts
        signal.signal(signal.SIGINT, self.signal_handler)

    def signal_handler(self, signum, frame):
        """
        Terminate scenario ticking when receiving a signal interrupt
        """
        if self._agent_watchdog and not self._agent_watchdog.get_status():
            raise RuntimeError("Agent took longer than {}s to send its command".format(self._timeout))
        elif self._watchdog and not self._watchdog.get_status():
            raise RuntimeError("The simulation took longer than {}s to update".format(self._timeout))
        self._running = False

    def cleanup(self):
        """
        Reset all parameters
        """
        self._timestamp_last_run = 0.0
        self.scenario_duration_system = 0.0
        self.scenario_duration_game = 0.0
        self.start_system_time = 0.0
        self.start_game_time = 0.0
        self.end_system_time = 0.0
        self.end_game_time = 0.0

        # self._spectator = None
        self._watchdog = None
        self._agent_watchdog = None

    def load_scenario(self, scenario, agent, route_index, rep_number):
        """
        Load a new scenario
        """

        GameTime.restart()
        self._agent_wrapper = AgentWrapperFactory.get_wrapper(agent)
        self.route_index = route_index
        self.scenario = scenario
        self.scenario_tree = scenario.scenario_tree
        self.ego_vehicles = scenario.ego_vehicles
        self.other_actors = scenario.other_actors
        self.repetition_number = rep_number

        # self._spectator = CarlaDataProvider.get_world().get_spectator()

        # To print the scenario tree uncomment the next line
        # py_trees.display.render_dot_tree(self.scenario_tree)


        # get criteria node
        self.criteria_node = None
        self.timeout_node = None
        self.route_percentage_node = None
        self.run_tl_node = None
        for node in self.scenario_tree.iterate():
            if node.name == "Criteria":
                self.criteria_node = node
            elif node.name == "RouteTimeoutBehavior":
                self.timeout_node = node
        
        for child in self.criteria_node.children:
            if child.name == "RouteCompletionTest":
                self.route_percentage_node = child 
            elif child.name == "RunningRedLightTest":
                self.run_tl_node = child

                
        self.om_vehicle = OmVehicle({'max_detection_number': 5, 'distance_threshold': EMERGENCY_DISTANCE})
        self.om_pedestrian = OmPedestrian({'max_detection_number': 5, 'distance_threshold': EMERGENCY_DISTANCE})
        self.om_construction = OmConstruction({'max_detection_number': 5, 'distance_threshold': EMERGENCY_DISTANCE})
        
        self.om_vehicle.attach_ego_vehicle(self.ego_vehicles[0])
        self.om_pedestrian.attach_ego_vehicle(self.ego_vehicles[0])
        self.om_construction.attach_ego_vehicle(self.ego_vehicles[0])
                

        agent.set_tl_node(self.run_tl_node)        
        agent.set_om(self.om_vehicle, self.om_pedestrian, self.om_construction)     
                
        self._agent_wrapper.setup_sensors(self.ego_vehicles[0], criteria_node=self.criteria_node)
        


    def build_scenarios_loop(self, debug):
        """
        Keep periodically trying to start the scenarios that are close to the ego vehicle
        Additionally, do the same for the spawned vehicles
        """
        while self._running:
            self.scenario.build_scenarios(self.ego_vehicles[0], debug=debug)
            self.scenario.spawn_parked_vehicles(self.ego_vehicles[0])
            time.sleep(1)

    def run_scenario(self, logfile, evaluation_vars):
        """
        Trigger the start of the scenario and wait for it to finish/fail
        """
        self.start_system_time = time.time()
        self.start_game_time = GameTime.get_time()

        # Detects if the simulation is down
        self._watchdog = Watchdog(self._timeout)
        self._watchdog.start()

        # Stop the agent from freezing the simulation
        self._agent_watchdog = Watchdog(self._timeout)
        self._agent_watchdog.start()
        
        self.logfile = logfile
        self.evaluation_vars = evaluation_vars
        # self._terminal_reward = TerminalReward()
        # self._step_reward = StepReward(vehicle=self.ego_vehicles[0], world=CarlaDataProvider.get_world())
        self.counter_frame = 0
        

        self._running = True

        # Thread for build_scenarios
        self._scenario_thread = threading.Thread(target=self.build_scenarios_loop, args=(self._debug_mode > 0, ))
        self._scenario_thread.start()

        while self._running:
            if self.counter_frame % 1200 == 0:
                print("Route Completion (%): ", self.route_percentage_node.actual_value)
            self._tick_scenario()
            self.counter_frame += 1

    def _tick_scenario(self):
        """
        Run next tick of scenario and the agent and tick the world.
        """
        if self._running and self.get_running_status():
            CarlaDataProvider.get_world().tick(self._timeout)

        timestamp = CarlaDataProvider.get_world().get_snapshot().timestamp

        if self._timestamp_last_run < timestamp.elapsed_seconds and self._running:
            self._timestamp_last_run = timestamp.elapsed_seconds

            self._watchdog.update()
            # Update game time and actor information
            GameTime.on_carla_tick(timestamp)
            CarlaDataProvider.on_carla_tick()
            self._watchdog.pause()

            try:
                self._agent_watchdog.resume()
                self._agent_watchdog.update()
                control, waypoints, metadata = self._agent_wrapper()
                self._agent_watchdog.pause()

            # Special exception inside the agent that isn't caused by the agent
            except SensorReceivedNoData as e:
                raise RuntimeError(e)

            except Exception as e:
                raise AgentError(e)


            self._watchdog.resume()
            self.ego_vehicles[0].apply_control(control)

            # Tick scenario. Add the ego control to the blackboard in case some behaviors want to change it
            py_trees.blackboard.Blackboard().set("AV_control", control, overwrite=True)
            self.scenario_tree.tick_once()

            self._agent_watchdog.resume()
            self._agent_watchdog.update()   

            obs = self._agent_wrapper.get_obs()
            
            self.logfile.add_data(episode=self.evaluation_vars['ep'], step=self.evaluation_vars['steps'], observation=obs, waypoints=waypoints, metadata=metadata, control=control)
            self.evaluation_vars['steps'] += 1


            if self._debug_mode > 1:
                self.compute_duration_time()

                # Update live statistics
                self._statistics_manager.compute_route_statistics(
                    self.route_index,
                    self.scenario_duration_system,
                    self.scenario_duration_game,
                    failure_message=""
                )
                self._statistics_manager.write_live_results(
                    self.route_index,
                    self.ego_vehicles[0].get_velocity().length(),
                    control,
                    self.ego_vehicles[0].get_location()
                )

            if self._debug_mode > 2:
                print("\n")
                py_trees.display.print_ascii_tree(self.scenario_tree, show_status=True)
                sys.stdout.flush()

            if self.scenario_tree.status != py_trees.common.Status.RUNNING:
                self._running = False

            self._agent_watchdog.pause()
            # ego_trans = self.ego_vehicles[0].get_transform()
            # self._spectator.set_transform(carla.Transform(ego_trans.location + carla.Location(z=70),
            #                                               carla.Rotation(pitch=-90)))

    def get_running_status(self):
        """
        returns:
           bool: False if watchdog exception occured, True otherwise
        """
        if self._watchdog:
            return self._watchdog.get_status()
        return True

    def stop_scenario(self):
        """
        This function triggers a proper termination of a scenario
        """
        if self._watchdog:
            self._watchdog.stop()

        if self._agent_watchdog:
            self._agent_watchdog.stop()

        self.compute_duration_time()

        if self.get_running_status():
            if self.scenario is not None:
                self.scenario.terminate()

            if self._agent_wrapper is not None:
                self._agent_wrapper.cleanup()
                self._agent_wrapper = None

            self.analyze_scenario()

        # Make sure the scenario thread finishes to avoid blocks
        self._running = False
        self._scenario_thread.join()
        self._scenario_thread = None
        
        # clean up the observation managers
        self.om_vehicle.clean()
        self.om_pedestrian.clean()
        self.om_construction.clean()

    def compute_duration_time(self):
        """
        Computes system and game duration times
        """
        self.end_system_time = time.time()
        self.end_game_time = GameTime.get_time()

        self.scenario_duration_system = self.end_system_time - self.start_system_time
        self.scenario_duration_game = self.end_game_time - self.start_game_time

    def analyze_scenario(self):
        """
        Analyzes and prints the results of the route
        """
        ResultOutputProvider(self)