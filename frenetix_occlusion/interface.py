__author__ = "Korbinian Moller,"
__copyright__ = "TUM Professorship Autonomous Vehicle Systems"
__version__ = "2.0"
__maintainer__ = "Korbinian Moller"
__email__ = "korbinian.moller@tum.de"
__status__ = "Beta"

# imports
import os
import warnings

from frenetix_occlusion.sensor_model import SensorModel
from frenetix_occlusion.occlusion_tracker import OcclusionTracker
from frenetix_occlusion.agent import FOAgentManager
from frenetix_occlusion.spawn_locator import SpawnLocator
from frenetix_occlusion.metrics.metric import Metric
from frenetix_occlusion.utils.fo_obstacle import FOObstacles
from frenetix_occlusion.utils.visualization import FOVisualization
from omegaconf import OmegaConf


class FOInterface:
    """
        The FOInterface class is the main interface of Frenetix-Occlusion.

        Attributes:
            - config (dict): Configuration settings loaded from a config file
            - scenario (Scenario): The current scenario being evaluated, including the lanelet network and obstacles.
            - lanelet_network (LaneletNetwork): The lanelet network of the current scenario.
            - ego_reference_path (list): The reference path for the ego vehicle.
            - cosy_cl (CurvilinearCoordinateSystem): The curvilinear coordinate system based on the ego vehicle's reference path.
              Defaults to None and is initialized if not provided.
            - vehicle_params (dict): Parameters of the ego vehicle, including dimensions and dynamics.
            - dt (float): The simulation time step in seconds.
            - plot (bool): Flag to enable or disable plotting for debug visualization.
            - predictions (dict): Predictions about other agents in the environment, updated during scenario evaluation.
            - ego_pos, ego_orientation, ego_pos_cl (tuple, float, tuple): The ego vehicle's position, orientation, and position
            - timestep (int): The current timestep of the simulation
            - spawn_points (list): Potential spawn points for phantom agents, identified during scenario evaluation.
            - sensor_radius, sensor_angle (float): Parameters of the sensor model defining the sensing range and angle.
            - visualization (FOVisualization): Tool for visualizing the scenario and for debugging purposes
            - fo_obstacles (FOObstacles): A representation of obstacles detected by the sensor model.
            - sensor_model (SensorModel): The sensor model used to detect obstacles and calculate visible and occluded areas.
            - agent_manager (FOAgentManager): Manages agents (both real and phantom) within the simulation.
            - spawn_locator (SpawnLocator): Identifies potential spawn points for phantom agents based on the current
              scenario state.
            - metrics (Metric): Metrics for evaluating the performance and safety of the ego vehicle in the scenario.

        Initialization Parameters:
            - scenario (Scenario): The current scenario, including lanelets and obstacles.
            - reference_path (list): The reference path for the ego vehicle.
            - vehicle_params (dict): Parameters of the ego vehicle.
            - dt (float): The simulation time step in seconds.
            - cosy_cl (CurvilinearCoordinateSystem, optional): An optional curvilinear coordinate system. If None,
              a new one is created based on the reference path.

        Methods:
            evaluate_scenario(predictions, ego_pos, ego_orientation, ego_pos_cl, ego_v, timestep):
                Evaluates the current scenario by updating the internal state based on external inputs,
                calculating visible and occluded areas, identifying spawn points for phantom agents,
                and visualizing the scenario and predictions if enabled.
                Inputs:
                    predictions (dict): Predictions about other agents in the environment.
                    ego_pos (tuple): The current position of the ego vehicle.
                    ego_orientation (float): The current orientation of the ego vehicle in radians.
                    ego_pos_cl (tuple): The current position of the ego vehicle in curvilinear coordinates.
                    ego_v (float): The current velocity of the ego vehicle.
                    timestep (int): The current timestep of the simulation.
                Outputs:

        """

    def __init__(self, scenario, reference_path, vehicle_params, dt, replanning_freq=0, config=None, cosy_cl=None):

        # Check and load frenetix occlusion configuration
        if config:
            # If a configuration is provided, use it
            self.config = config
        else:
            # Otherwise, load the default configuration
            self.config = self._load_default_config()

        # load global variables that never change
        self.scenario = scenario
        self.track_obstacles = self.config.track_obstacles
        self.omniscient_prediction = self.config.omniscient_prediction
        self.lanelet_network = scenario.lanelet_network
        self.ego_reference_path = reference_path
        self.cosy_cl = cosy_cl
        self.vehicle_params = vehicle_params
        self.dt = dt
        self.replanning_freq = replanning_freq
        self.plot = self.config.plot
        self.debug = self.config.debug

        # initialize changing variables
        self.predictions = None
        self.ego_pos = None  # position of the vehicles rear axle
        self.ego_orientation = None # orientation of the vehicles rear axle
        self.ego_pos_center = None  # position of the vehicle center
        self.ego_orientation_center = None # orientation of the vehicle center
        self.ego_pos_cl = None
        self.timestep = None
        self.spawn_points = []

        # create visualization (mainly for debugging)
        self.visualization = FOVisualization(scenario, reference_path, self.plot)

        # convert obstacles to FOObstacles
        self.fo_obstacles = FOObstacles(self.scenario.obstacles)

        # initialize agent_manager
        self.agent_manager = FOAgentManager(scenario=self.scenario,
                                            reference_path=self.ego_reference_path,
                                            config=self.config.agent_manager,
                                            visualization=self.visualization,
                                            timestep=self.timestep,
                                            dt=self.dt,
                                            debug=self.debug,
                                            fo_obstacles=self.fo_obstacles)
        
        # initialize sensor model
        self.sensor_model = SensorModel(lanelet_network=self.lanelet_network,
                                        config=self.config.sensor_model,
                                        visualization=self.visualization,
                                        debug=self.debug,
                                        replanning_freq=self.replanning_freq,
                                        dt=self.dt)

        # initiate occlusion tracker
        self.occlusion_tracker = OcclusionTracker(lanelet_network=self.scenario.lanelet_network,
                                                  road_polygon=self.sensor_model.road_polygon,
                                                  lanelet_network_polygons=self.sensor_model.lanelet_network_polygons,
                                                  config=self.config.occlusion_tracker,
                                                  ego_v_reference_path=self.ego_reference_path,
                                                  visualization=self.visualization,
                                                  scenario=self.scenario,
                                                  sensor_model=self.sensor_model)

        # initialize spawn locator
        self.spawn_locator = SpawnLocator(agent_manager=self.agent_manager,
                                          ref_path=self.ego_reference_path,
                                          config=self.config,
                                          cosy_cl=self.cosy_cl,
                                          sensor_model=self.sensor_model,
                                          occlusion_tracker=self.occlusion_tracker,
                                          fo_obstacles=self.fo_obstacles,
                                          visualization=self.visualization,
                                          debug=self.debug)

        # initialize metrics
        self.metrics = Metric(self.config.metrics, self.vehicle_params, self.agent_manager)

    @property
    def visible_area(self):
        return self.sensor_model.visible_area

    @property
    def occluded_area(self):
        return self.occlusion_tracker.total_occluded_area

    def set_coordinate_system(self, cosy_cl):
        self.cosy_cl = cosy_cl
        self.spawn_locator.cosy_cl = cosy_cl

    def _add_real_agents(self):
        if self.config.agents is None:
            return
        for agent in self.config.agents:
            try:
                self.agent_manager.add_agent(pos=agent['position'],
                                             velocity=agent['velocity'],
                                             agent_type=agent['agent_type'],
                                             add_to_scenario=True,
                                             timestep=agent['timestep'],
                                             horizon=agent['horizon'])
            except:
                self.agent_manager.real_agents.pop()
                warnings.warn("Could not add specified {} agent at position {} to scenario. Last added agent is getting removed. "
                              "It is not checked, if this agent is the one that could not be added."
                              .format(agent['agent_type'], agent['position']))

    def evaluate_scenario(self, predictions, ego_pos_center, ego_orientation_center,
                          ego_pos, ego_orientation, ego_pos_cl, ego_v, timestep, cosy_cl):
        
        # set current coordinate system 
        self.set_coordinate_system(cosy_cl)

        # update timestep
        self._update_time_step(timestep)

        # add real agents
        self._add_real_agents()

        # update variables with external inputs
        self.predictions = predictions
        self.ego_pos = ego_pos
        self.ego_orientation = ego_orientation
        self.ego_pos_center = ego_pos_center
        self.ego_orientation_center = ego_orientation_center
        self.ego_pos_cl = ego_pos_cl

        # reset agent manager
        self.agent_manager.reset()

        # clear spawn points
        self.spawn_points.clear()

        # visualize scenario if activated
        if self.visualization is not None and self.plot:
            self.visualization.draw_scenario(timestep=self.timestep)
            self.visualization.show_plot()

        # set relevant obstacles
        self.fo_obstacles.update(self.timestep)

        # calculate visible area and visible obstacles
        self.sensor_model.calc_visible_area_and_obstacles(timestep=self.timestep, ego_pos=self.ego_pos_center,
                                                          ego_orientation=self.ego_orientation_center, obstacles=self.fo_obstacles)

        # update multipolygon that stores the polygons with visible obstacles for spawn point detection
        self.fo_obstacles.update_multipolygon()

        # update occlusion tracker
        self.occlusion_tracker.update_tracker(self.sensor_model.visible_area, dt=self.dt, replanning_counter=self.replanning_freq,
                                              timestep=self.timestep)

        # find spawn points
        self.spawn_points = self.spawn_locator.find_spawn_points(ego_pos=self.ego_pos,
                                                                 ego_orientation=self.ego_orientation,
                                                                 ego_cl=self.ego_pos_cl,
                                                                 ego_pos_center=self.ego_pos_center,
                                                                 ego_orientation_center=self.ego_orientation_center,
                                                                 ego_v=ego_v)

        # iterate over spawn points and add phantom agents
        for sp in self.spawn_points:
            # determine mode
            # mode = 'lane_center' if sp.source == 'left turn' or sp.source == 'right turn' else 'ref_path'
            mode = "ref_path"

            # add agent for each spawn point
            self.agent_manager.add_agent(pos=sp.position, velocity="default", agent_type=sp.agent_type,
                                         timestep=self.timestep, horizon=3.0, mode=mode, orientation=sp.orientation)

            # print debug message
            if self.debug:
                print("Phantom agent of type {} with id {} added to scenario at position {}"
                      .format(sp.agent_type, self.agent_manager.phantom_agents[-1].agent_id, sp.position))

        # update pedestrian predictions after adding real pedestrians to the scenario
        self.agent_manager.update_real_agents(self.predictions)

        # visualize PA predictions if activated
        if self.visualization is not None and self.plot:
            self.visualization.set_focus(self.ego_pos_center, 50, 40)
            self.visualization.plot_poly_fast(self.sensor_model.visible_area, color='green', fill=True, zorder=100, opacity=0.2)
            self.visualization.plot_poly_fast(self.occlusion_tracker.total_occluded_area, color='red', fill=True, zorder=100, opacity=0.2)
            self.visualization.draw_point(self.ego_pos_center, color='blue', marker='o', zorder=101)
            self.visualization.draw_predictions(self.agent_manager.predictions, label=False)
            self.visualization.save_plot(directory="plots", timestep=self.timestep)

        # return visible area if needed
        return self.sensor_model.visible_area

    def trajectory_safety_assessment(self, trajectory):
        metrics, safety_assessment = self.metrics.evaluate_metrics(trajectory)

        return metrics, safety_assessment

    def _update_time_step(self, timestep):
        # updates the timestep in all objects
        self.timestep = timestep
        self.sensor_model.timestep = timestep
        self.agent_manager.timestep = timestep

    @staticmethod
    def _load_default_config():

        # loads the default config.yaml file
        filepath = os.path.join(os.path.dirname(__file__), "config", "config.yaml")
        print("Load default frenetix occlusion settings!")

        with open(filepath, 'r') as file:
            config = OmegaConf.load(filepath)

        return config


if __name__ == '__main__':

    from commonroad.common.file_reader import CommonRoadFileReader
    import matplotlib as mpl
    from omegaconf import OmegaConf

    mpl.use('TkAgg')

    config = OmegaConf.load("../configurations/simulation/occlusion.yaml")

    scenario, planning_problem_set = CommonRoadFileReader('../example_scenarios/ZAM_Tjunction-1_315_P--4312-no-ped.xml').open()
    sensor_model = SensorModel(scenario.lanelet_network, config=config.sensor_model)
    fo_obstacles = FOObstacles(scenario.obstacles)
    fo_obstacles.update(0)
    visualization0 = FOVisualization(scenario, None, True, title="Timestep 0")
    visualization1 = FOVisualization(scenario, None, True, title="Timestep 1")

    occlusion_tracker = OcclusionTracker(lanelet_network=scenario.lanelet_network,
                                         road_polygon=sensor_model.road_polygon,
                                         lanelet_network_polygons=sensor_model.lanelet_network_polygons,
                                         config=config.occlusion_tracker,
                                         visualization=None,
                                         ego_v_reference_path=None)

    visualization0.draw_scenario()
    visualization1.draw_scenario()

    # timestep 0
    ego_pos0 = [-30, -1]
    visualization0.ax.plot(ego_pos0[0], ego_pos0[1], 'bo', zorder=100)
    visible_area_0 = sensor_model.calc_visible_area_and_obstacles(0, ego_pos0, 0, fo_obstacles.fo_obstacles)
    visualization0.plot_poly_fast(visible_area_0, color='green', fill=True, zorder=100, opacity=0.5)
    occluded_area_0 = occlusion_tracker.update_tracker(visible_area_0, dt=0.1)
    visualization0.plot_poly_fast(occluded_area_0, color='red', fill=True, zorder=100, opacity=0.5)
    # plt.show()

    # timestep 1
    ego_pos1 = [0, 0]
    visualization1.ax.plot(ego_pos1[0], ego_pos1[1], 'ko', zorder=100)
    visible_area_1 = sensor_model.calc_visible_area_and_obstacles(0, ego_pos1, 0, fo_obstacles.fo_obstacles)
    visualization1.plot_poly_fast(visible_area_1, color='green', fill=True, zorder=100, opacity=0.5)
    occluded_area_1 = occlusion_tracker.update_tracker(visible_area_1, dt=0.1)
    visualization1.plot_poly_fast(occluded_area_1, color='red', fill=True, zorder=100, opacity=0.5)
    visualization1.plot_poly_fast(visible_area_0, color='green', fill=False, zorder=100, opacity=1.0)

    print('Done')
