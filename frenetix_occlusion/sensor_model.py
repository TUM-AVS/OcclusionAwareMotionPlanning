__author__ = "Korbinian Moller,"
__copyright__ = "TUM AVS"
__version__ = "2.0"
__maintainer__ = "Korbinian Moller"
__email__ = "korbinian.moller@tum.de"
__status__ = "Beta"

# imports
from functools import reduce
import numpy as np
from shapely.geometry import Polygon, Point
from shapely.geometry.multipolygon import MultiPolygon
import frenetix_occlusion.utils.helper_functions as hf


class SensorModel:
    def __init__(self, lanelet_network, config, debug=True, visualization=None,
                 dt=0.1, replanning_freq=0):

        # load global variables that never change
        self.lanelet_network = lanelet_network
        # combined lanelets as a shapely polygon and polygon list
        self.road_polygon, self.lanelet_network_polygons = self._convert_lanelet_network()
        self.visualization = visualization
        self.config = config

        # initialize changing variables
        self.sensor_sector = None
        self.visible_area = None
        self.visible_obstacles_occupancy = None
        self.visible_obstacles_safety_distance = None
        self.obstacle_occlusions = {}
        self.visible_objects_timestep = None
        self.timestep = None
        self.ego_pos = None
        self.ego_orientation = None
        self.dt = None

        # define configuration
        self.sensor_radius = self.config.sensor_radius
        self.sensor_angle = self.config.sensor_angle
        self.debug = debug

        self.dt = dt
        self.replanning_freq = replanning_freq

    def calc_visible_area_and_obstacles(self, timestep, ego_pos, ego_orientation, obstacles):
        """
            Calculate the visible area and visible obstacles for a given timestep.

            This function calculates the visible area based on the lanelet network geometry and obstacles.
            It updates the list of visible objects and obstacle occlusions for the given timestep.

            Args:
                timestep (int): The current timestep.
                ego_pos (tuple): The position of the ego vehicle as (x, y) coordinates.
                ego_orientation (float): The orientation of the ego vehicle in radians.
                obstacles (list): A list of obstacles to consider for occlusions.

            Returns:
                shapely.geometry.Polygon: The visible area as a shapely Polygon.
            """
        # if the visible area for the timestep has already been calculated -> return without recalculating
        # if self.timestep == timestep:
        #    return self.visible_area

        # Set variables
        self.ego_pos = ego_pos
        self.ego_orientation = ego_orientation
        self.visible_objects_timestep = []

        # reset obstacle occlusions list
        self.obstacle_occlusions.clear()

        # calculate visible area based on lanelet geometry
        visible_area_road = self._calc_visible_area_from_lanelet_geometry()

        # calculate visible area based on obstacles
        visible_area, self.visible_obstacles_occupancy, self.visible_obstacles_safety_distance = (
            self._calc_visible_area_from_obstacle_occlusions(visible_area_road, obstacles))

        # get visible obstacles and add to list (has to be done after finishing the calculation of the visible area)
        visible_area_check = visible_area.buffer(0.01, join_style=2)

        if visible_area_check.is_valid is False:
            visible_area_check = visible_area.buffer(0.01)

        for obst in obstacles:

            # if obstacle exists at the current timestep
            if obst.current_pos is not None:

                # check if obstacle intersects with visible area
                if obst.current_polygon.intersects(visible_area_check):
                    # add to list of visible objects
                    self.visible_objects_timestep.append(obst.cr_obstacle.obstacle_id)

                    # update obstacle
                    obst.current_visible = True
                    obst.last_visible_at_ts = timestep

        # remove linestrings from visible_area
        self.visible_area = hf.remove_unwanted_shapely_elements(visible_area)

        return self.visible_area
    
    def _calc_visible_area_from_lanelet_geometry(self):
        """
        Calculates the visible area at the ego position in consideration of the lanelet network geometry

        Returns:
        visible area
        """
        # at first, the visible area is the entire road polygon and a buffer at the road boundaries
        visible_area = self.road_polygon.buffer(self.config.lanelet_buffer)

        # calculate two points based on the sensor opening angle
        angle_start = self.ego_orientation - np.radians(self.sensor_angle / 2)
        angle_end = self.ego_orientation + np.radians(self.sensor_angle / 2)

        # create a "sector" with the given angles points and the ego position
        if self.sensor_angle >= 359.9:
            self.sensor_sector = Point(self.ego_pos).buffer(self.sensor_radius)
            
        else:
            self.sensor_sector = self._calc_relevant_sector(angle_start, angle_end)

        # self.visualization.fill_area(self.sensor_sector, color='r')

        # find the intersection between the "circle" and the "triangle"
        visible_area = visible_area.intersection(self.sensor_sector)

        # remove unwanted elements
        visible_area = self._remove_unwanted_shapely_elements(visible_area)
        
        return visible_area

    def _calc_visible_area_from_obstacle_occlusions(self, visible_area, obstacles):
        """
        Calculate occlusions from obstacles and subtract them from visible_area
        Args:
            visible_area: visible area
            obstacles: list of obstacles of type FOObstacle

        Returns:
        updated visible area
        multipolygon of obstacles
        """
        
        all_obstacles_polygon = Polygon([])
        all_obstacles_polygon_safety_distance = Polygon([])

        # Calculate occlusions from obstacles and subtract them from visible_area
        for obst in obstacles:

            # obstacle position is not empty, this happens if dynamic obstacle is not available at timestep
            if obst.current_pos is not None and obst.cr_obstacle.obstacle_type.value != 'bicycle':

                # check if within sensor radius or if obstacle intersects with visible area
                if obst.current_pos_point.within(visible_area) or obst.current_polygon.intersects(visible_area):

                    # Subtract obstacle shape from visible area
                    visible_area = visible_area.difference(obst.current_polygon.buffer(0.005, join_style=2))

                    # store obstacle polygons without buffer in multipolygon for later use
                    all_obstacles_polygon = all_obstacles_polygon.union(obst.current_polygon)

                    # calculate safety distance based on obstacle velocity
                    if obst.cr_obstacle.obstacle_role.name == "STATIC":
                        safety_distance = 0.1
                    else:
                        safety_distance = max(1.0, 0.036 * obst.current_v + 0.5)

                    # Store obstacle multipolygon with safety distance for use in spawn point calculation
                    obstacle_with_safety_distance = obst.current_polygon.buffer(safety_distance, join_style=2)
                    all_obstacles_polygon_safety_distance = all_obstacles_polygon_safety_distance.union(obstacle_with_safety_distance)

                    # calculate occlusion polygon (shadow) that is caused by the obstacle
                    occlusion = hf.get_polygon_from_obstacle_occlusion(self.ego_pos, obst.current_corner_points)
                    self.obstacle_occlusions[obst.cr_obstacle.obstacle_id] = occlusion.difference(obst.current_polygon)

                    # Subtract shadow caused by obstacle (everything behind obstacle) from visible area
                    if occlusion.is_valid:
                        visible_area = visible_area.difference(occlusion)

        return visible_area, all_obstacles_polygon, all_obstacles_polygon_safety_distance

    def _convert_lanelet_network(self):
        lanelet_polygons = [polygon.shapely_object for polygon in self.lanelet_network.lanelet_polygons]
        valid_polygons = []
        for poly in lanelet_polygons:
            if not poly.is_valid:
                
                poly = poly.buffer(0)  # Attempt to fix the invalid geometry
                if not poly.is_valid:
                    continue  # Skip the polygon if it is still invalid after fixing
            valid_polygons.append(poly)
        road_polygon = reduce(lambda x, y: x.union(y), valid_polygons)

        return road_polygon, lanelet_polygons

    def _calc_relevant_sector(self, angle_start, angle_end, factor=1.0):
        points = [(self.ego_pos[0] + self.sensor_radius * factor * np.cos(angle),
                   self.ego_pos[1] + self.sensor_radius * factor * np.sin(angle))
                  for angle in np.linspace(angle_start, angle_end, 100)]

        # create a "sector" with these points and the ego position
        sector = Polygon([self.ego_pos] + points + [self.ego_pos])

        return sector

    @staticmethod
    def _remove_unwanted_shapely_elements(polys) -> MultiPolygon:
        """
        This function removes every Geometry except Polygons from a GeometryCollection
        and converts the remaining Polygons to a MultiPolygon.

        Args:
            polys: GeometryCollection

        Returns: MultiPolygon

        """
        if polys.geom_type == "GeometryCollection":
            poly_list = []
            for pol in polys.geoms:
                if pol.geom_type == 'Polygon':
                    if not np.isclose(pol.area, 0, 1.e-3):
                        poly_list.append(pol)

            multipolygon = MultiPolygon(poly_list)
        else:
            multipolygon = polys

        return multipolygon.buffer(0)
