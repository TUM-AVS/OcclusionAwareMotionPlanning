__author__ = "Korbinian Moller,"
__copyright__ = "TUM Professorship Autonomous Vehicle Systems"
__version__ = "2.0"
__maintainer__ = "Korbinian Moller"
__email__ = "korbinian.moller@tum.de"
__status__ = "Beta"

# imports
import copy
import numpy as np
from commonroad_dc.geometry.util import compute_pathlength_from_polyline, compute_curvature_from_polyline
from shapely.geometry import Polygon, Point, LineString
from shapely.ops import unary_union
from commonroad_route_planner.utility.route_util import lanelet_orientation_at_position
import frenetix_occlusion.utils.helper_functions as hf


class SpawnPoint:
    """
    Simple class containing the spawn point information
    """
    def __init__(self, pos, agent_type, pos_cl=None, source=None, orientation=None):
        self.position = pos
        self.agent_type = agent_type
        self.cl_pos = pos_cl
        self.source = source
        self.orientation = orientation


class SpawnLocator:

    def __init__(self, agent_manager, ref_path, cosy_cl, sensor_model, occlusion_tracker,
                 fo_obstacles, config, visualization=None, debug=False):

        # load other modules
        self.agent_manager = agent_manager
        self.scenario = agent_manager.scenario
        self.sensor_model = sensor_model
        self.occlusion_tracker = occlusion_tracker
        self.fo_obstacles = fo_obstacles
        self.visualization = visualization
        self.config = config.spawn_locator
        self.ref_path = ref_path
        self.cosy_cl = cosy_cl

        # initialize own variables
        self.reference = None
        self.s = None
        self.reference_s = None
        self.ego_pos = None
        self.ego_orientation = None
        self.ego_intention = None
        self.ego_cl = None
        self.ego_pos_center = None
        self.ego_orientation_center = None
        self.spawn_points = []

        # define where to look for spawn points (from config)
        self.spawn_pedestrians = self.config.spawn_pedestrians
        self.spawn_vehicles_and_bicycles = self.config.spawn_vehicles_and_bicycles

        # maximum number of spawn points (from config)
        self.max_pedestrians = self.config.max_pedestrians
        self.max_vehicles_and_bicycles = self.config.max_vehicles_and_bicycles

        # initialize internal parameters
        self.ped_width = config.agent_manager.pedestrian.width
        self.ped_length = config.agent_manager.pedestrian.length
        self.s_threshold_time = 3.0  # needed for s_threshold --> calculates the distance when moving at current speed for X seconds
        self.min_s_threshold = 25  # min s_threshold, if vehicle is very slow
        self.s_threshold = None  # calculated by function - maximum allowed distance of PA
        self.tolerance_same_direction = np.radians(20)  # tolerance until a vehicle is defined as driving in same direction in °
        self.max_distance_to_other_obstacle = 30  # max allowed distance between ego vehicle and obstacle in m
        self.buffer_around_vehicle_from_side = 12  # buffer around vehicles in m
        self.min_area_threshold = 10  # min required area size in m² -> smaller areas are neglected
        self.agent_area_limits = {'Car': 9, 'Bicycle': 1.7}  # required size for a rectangle, that an agent can be spawned in m²
        self.min_distance_between_pedestrians = 5  # minimum distance between two pedestrians (behind static obstacles)
        self.debug = debug
        # spawn points behind turns (special parameters)
        self.offset_ref_path = {'left turn': 3, 'right turn': -0.6}
        self.phantom_offset_s = {'left turn': -0.5, 'right turn': 0}
        self.phantom_offset_d = {'left turn': 1, 'right turn': -1}

        # new parameters
        self.relevant_occluded_area = None
        self.relevant_occlusions = None

    def find_spawn_points(self, ego_pos, ego_orientation, ego_cl, ego_pos_center, ego_orientation_center, ego_v, debug=False):
        """
            Determines potential spawn points for phantom agents.

            This function calculates various spawn points for phantom agents considering the ego vehicle's current state
            and intended direction. It clears previous spawn points, updates internal state with the ego vehicle's
            current position and curvilinear coordinates, and calculates a threshold based on the vehicle's velocity.
            The function then determines the ego vehicle's intention (e.g., left turn, straight ahead) and searches for
            potential spawn points behind turns, dynamic obstacles, and static obstacles.

            Parameters:
            ego_pos (numpy.ndarray): The current position of the ego vehicle, as (x, y) coordinates.
            ego_orientation (float): The current orientation of the ego vehicle in radians.
            ego_cl (numpy.ndarray): The current curvilinear coordinates of the ego vehicle, as (s, d) coordinates.
            ego_v (float): The current velocity of the ego vehicle.
            ego_pos_center (numpy.ndarray): The current position of the ego vehicle's center, as (x, y) coordinates.
            ego_orientation_center (float): The current orientation of the ego vehicle's center in radians.

            Returns:
            list: A list of SpawnPoint objects, each representing a potential location for a phantom agent.
            The list may be empty if no suitable spawn points are found.

            Note:
            The function utilizes several internal properties (like `self.s_threshold_time`, `self.min_s_threshold`) and
            methods (like `_prepare_reference_path`, `_find_ego_intention`, `_find_spawn_point_behind_turn`, etc.) which
             are part of the class this function belongs to.
        """

        # preprocessing - find relevant occluded areas etc.
        self._preprocessing(ego_pos, ego_orientation, ego_cl, ego_pos_center, ego_orientation_center, ego_v)

        if debug:
            self.visualization.plot_poly_fast(self.relevant_occluded_area, fill=True, opacity=0.2, color='red', zorder=100)
            self.visualization.plot_poly_fast(self.occlusion_tracker.total_occluded_area, fill=False, opacity=1.0, color='red',
                                              zorder=100)
            self.visualization.plot_poly_fast(self.sensor_model.visible_area, fill=True, opacity=0.2, color='green', zorder=100)

        # search for pedestrian spawn points and add them to self.spawn_points
        if self.spawn_pedestrians:
            self._create_pedestrian_spawn_points()

        # search for vehicle and bicycle spawn points and add them to self.spawn_points
        if self.spawn_vehicles_and_bicycles:
            self._create_vehicle_bicycle_spawn_points()

        return self.spawn_points

    def _create_pedestrian_spawn_points(self) -> None:

        # find spawn points behind static obstacles, e.g. parked cars
        spawn_points_behind_static_obstacles = self._find_spawn_point_behind_static_obstacle()
        self._append_spawn_point(spawn_points_behind_static_obstacles)

        # find spawn points behind turns
        spawn_points_behind_turns = self._find_spawn_point_behind_turn()
        self._append_spawn_point(spawn_points_behind_turns)

    def _create_vehicle_bicycle_spawn_points(self):

        dynamic_spawn_points = self._find_dynamic_spawn_points()
        self._append_spawn_point(dynamic_spawn_points)

    #################################################################################
    ############################## Preprocessing  ###################################
    #################################################################################

    def _preprocessing(self, ego_pos, ego_orientation, ego_cl, ego_pos_center, ego_orientation_center, ego_v):
        # clear previous points
        self.spawn_points.clear()

        # save values for other functions (point at the rear axle
        self.ego_pos = ego_pos
        self.ego_orientation = ego_orientation
        self.ego_cl = ego_cl

        # center points
        self.ego_pos_center = ego_pos_center
        self.ego_orientation_center = ego_orientation_center

        # calculate threshold
        self.s_threshold = ego_cl[0] + max(ego_v * self.s_threshold_time, self.min_s_threshold)

        # calc new reference
        self.reference, self.reference_s = self._prepare_reference_path(ego_cl + 2, distance=50)

        # find ego intention within the new reference
        self.ego_intention = self._find_ego_intention(self.reference)

        # calc relevant area based on the ego reference path
        relevant_area = hf.buffer_sides_only(LineString(self.reference), 20).intersection(self.sensor_model.road_polygon)

        # calc relevant occluded area -> intersection of relevant area and occluded area
        relevant_occluded_area = relevant_area.intersection(self.occlusion_tracker.total_occluded_area)

        # subtract the vehicle occupancy from the relevant occluded area
        self.relevant_occluded_area = relevant_occluded_area.difference(self.sensor_model.visible_obstacles_safety_distance.buffer(0.1))

        # find relevant occlusion objects
        self.relevant_occlusions = self._find_and_process_relevant_occlusions(self.relevant_occluded_area)

    #################################################################################
    ########################### Dynamic spawn points  ###############################
    #################################################################################

    def _find_dynamic_spawn_points(self) -> list[SpawnPoint]:
        """
        Find spawn points behind dynamic obstacles (vehicles and bicycles) up to a maximum number.

        Args:
            max_spawn_points (int): The maximum number of spawn points to generate.

        Returns:
            list[SpawnPoint]: A list of generated spawn points.
        """
        spawn_points = []  # List to store the resulting spawn points
        used_occlusions = []  # Track processed occlusions to avoid duplicates

        for occlusion in self.relevant_occlusions:
            # Skip occlusions that significantly overlap with already processed ones
            if self._is_occlusion_overlapping(occlusion, used_occlusions):
                continue

            # Mark this occlusion as processed
            used_occlusions.append(occlusion)

            # self.visualization.plot_poly_fast(occlusion.relevant_polygon, fill=True, opacity=0.3, color='purple', zorder=100)

            # Convert and sort the center points by proximity to the ego vehicle
            sorted_center_points = self._get_sorted_center_points(occlusion, self.ego_cl)

            if sorted_center_points.size == 0:
                continue

            # Evaluate each point until a valid spawn point is found
            spawn_points_for_occlusion = self._evaluate_center_points(sorted_center_points, occlusion, spawn_points)

            # Add the new spawn points and stop if the maximum number is reached
            spawn_points.extend(spawn_points_for_occlusion)

        # Sort all collected spawn points by distance to the ego vehicle
        sorted_spawn_points = sorted(spawn_points, key=lambda sp: np.linalg.norm(np.array(sp.position) - self.ego_pos))

        return sorted_spawn_points[:self.max_vehicles_and_bicycles]  # Ensure no more than the specified maximum

    @staticmethod
    def _is_occlusion_overlapping(occlusion, used_occlusions) -> bool:
        """
        Check if the current occlusion overlaps significantly with any already processed occlusion.

        Args:
            occlusion: The current occlusion being checked.
            used_occlusions: List of previously processed occlusions.

        Returns:
            bool: True if there is significant overlap, False otherwise.
        """
        for used_occlusion in used_occlusions:
            intersection = occlusion.relevant_polygon.intersection(used_occlusion.relevant_polygon)
            overlap_ratio = intersection.area / occlusion.polygon.area
            if overlap_ratio > 0.9:  # Threshold for considering occlusions overlapping
                return True
        return False

    def _get_sorted_center_points(self, occlusion, ego_cl) -> np.ndarray:
        """
        Transform and sort the center points of an occlusion by proximity to the ego vehicle,
        filtering out points with an s-coordinate less than ego_cl[0] + 5.

        Args:
            occlusion: The occlusion containing the center points.
            ego_cl (np.array): The curvilinear coordinates of the ego vehicle.

        Returns:
            np.ndarray: Sorted center points in Cartesian coordinates.
        """
        center_points = occlusion.center_points

        # Convert to curvilinear coordinates
        center_points_cl = np.vstack(self.cosy_cl.convert_list_of_points_to_curvilinear_coords(center_points, 1))

        # Filter points with s-coordinate less than ego_cl[0] + 5 and d-coordinate between -1 and 1
        center_points_cl = center_points_cl[
            (center_points_cl[:, 0] >= ego_cl[0] + 5) &  # Filter based on s-coordinate
            (center_points_cl[:, 0] <= self.s_threshold * 1.5) &  # Filter based on s-coordinate
            ((center_points_cl[:, 1] >= -15) & (center_points_cl[:, 1] <= 15))  # Filter based on d-coordinate
            ]

        # Handle the case where no points remain after filtering
        if center_points_cl.size == 0:
            return np.array([])  # Return an empty array if no valid points exist

        # Sort by distance (s² + d²)
        sorted_points_cl = np.array(sorted(center_points_cl, key=lambda x: np.sum(x ** 2)))

        # Convert back to Cartesian coordinates
        return np.vstack(self.cosy_cl.convert_list_of_points_to_cartesian_coords(sorted_points_cl, 4))

    def _evaluate_center_points(self, center_points_sorted, occlusion, existing_spawn_points, debug=True) -> list[SpawnPoint]:
        """
        Evaluate center points to find at most one spawn point for a car and one for a bicycle in a single occlusion.

        Args:
            center_points_sorted (np.ndarray): Center points sorted by proximity to the ego vehicle.
            occlusion: The occlusion associated with the center points.

        Returns:
            list[SpawnPoint]: Valid spawn points (at most one for car and one for bicycle).
        """
        spawn_points = []
        car_found = False
        bike_found = False

        for point in center_points_sorted:
            # self.visualization.rnd.ax.plot(point[0], point[1], 'bo', zorder=100)

            # Find possible oriented rectangles for the point
            rectangles = self._find_matching_rectangle(point, occlusion.relevant_polygon, occlusion.lanelet)

            for agent_type, rect_data in rectangles.items():
                # Skip if we already found the required spawn point for this agent type
                if (agent_type == "Car" and car_found) or (agent_type == "Bicycle" and bike_found):
                    continue

                polygon = rect_data['polygon']
                metric = rect_data['jaccard_similarity']

                # Check area and similarity thresholds
                if polygon.area >= self.agent_area_limits[agent_type] and metric > 0.98:
                    phantom_pos = np.array([polygon.centroid.x, polygon.centroid.y])
                    spawn_points.append(
                        SpawnPoint(pos=phantom_pos, agent_type=agent_type, pos_cl=None, source='behind_dynamic_obstacle')
                    )

                    # Plot debug visuals
                    if debug and self.visualization:
                        self.visualization.rnd.ax.plot(point[0], point[1], 'ro', zorder=100)
                        self.visualization.plot_poly_fast(polygon, color='blue', fill=True, opacity=0.3, zorder=100)

                    # Mark as found
                    if agent_type == "Car":
                        car_found = True
                    elif agent_type == "Bicycle":
                        bike_found = True

            # Stop checking further points if both a car and a bicycle spawn point have been found
            if car_found and bike_found:
                break

        return spawn_points

    def _find_matching_rectangle(self, position, allowed_area, lanelet=None) -> dict:
        """
        Generate and evaluate oriented rectangles for potential vehicle and bicycle spawn points.

        Args:
            position (tuple): The position to evaluate.
            allowed_area: The polygon representing the area where spawn points are allowed.

        Returns:
            dict: Rectangles with their metrics for different agent types.
        """
        _, orientation = self._find_orientation_at_position(position, lanelet)

        # Oriented rectangle for vehicles
        vehicle_rect = hf.create_oriented_rectangle(position, 5.5, 2.5, orientation)
        poly_vehicle = vehicle_rect.intersection(allowed_area)
        vehicle_metrics = self._calculate_polygon_metrics(poly_vehicle)
        vehicle_metrics['polygon'] = poly_vehicle

        # Oriented rectangle for bicycles
        # bike_position = [poly_vehicle.centroid.x, poly_vehicle.centroid.y]
        bike_rect = hf.create_oriented_rectangle(position, 2, 1, orientation)
        poly_bike = bike_rect.intersection(allowed_area)
        bike_metrics = self._calculate_polygon_metrics(poly_bike)
        bike_metrics['polygon'] = poly_bike

        return {"Car": vehicle_metrics, "Bicycle": bike_metrics}

    @staticmethod
    def _calculate_polygon_metrics(polygon) -> dict:
        """
        Calculate metrics (aspect ratio, Jaccard similarity) for a polygon.

        Args:
            polygon: The polygon to evaluate.

        Returns:
            dict: Metrics including aspect ratio and Jaccard similarity.
        """
        if polygon.is_empty:
            return {"area_ratio": 0, "jaccard_similarity": 0}

        mbr = polygon.minimum_rotated_rectangle  # Minimum bounding rectangle
        area_ratio = polygon.area / mbr.area

        intersection = polygon.intersection(mbr).area
        union = unary_union([polygon, mbr]).area
        jaccard_similarity = intersection / union

        return {"area_ratio": area_ratio, "jaccard_similarity": jaccard_similarity}

    #################################################################################
    #################### Spawn points behind static obstacle  #######################
    #################################################################################

    def _find_spawn_point_behind_static_obstacle(self) -> list[SpawnPoint] or None:
        """
        Identifies potential spawn points behind visible static obstacles within a specified distance from the ego vehicle.

        The function operates in several steps:
        - Identifies currently visible static obstacles and sorts them based on their distance to the ego vehicle.
        - Skips the process if no visible static obstacle is found.
        - Calculates curvilinear coordinates of each obstacle and determines if they are within a certain distance
          from the ego vehicle.
        - Converts the corner points of the obstacle to curvilinear coordinates and determines the minimum and
          maximum coordinates.
        - Creates lines perpendicular to the reference path to find intersections with the visible area.
        - Determines potential spawn positions based on the intersection of these lines with the visible area.
        - Ensures that these spawn positions do not intersect with other obstacles and are sufficiently spaced from existing
          spawn points.

        Returns:
            list[SpawnPoint]: A list of SpawnPoint objects representing potential spawn locations behind static obstacles.
                               Each SpawnPoint contains the position, agent type, curvilinear position, and source information.
                               Returns None if no suitable spawn point is found.

        Note:
            The function relies on various attributes of the class such as `self.ego_pos`, `self.cosy_cl`, ...
        """

        # initialize lists to store results
        spawn_points = []
        s_positions = []

        # find currently visible static obstacles
        visible_stat_obst = [obst for obst in self.fo_obstacles
                             if obst.current_visible and obst.cr_obstacle.obstacle_role.name == 'STATIC' and obst.cr_obstacle.obstacle_type.value != 'building']

        visible_stat_obst = sorted(visible_stat_obst, key=lambda obstacle: np.linalg.norm(self.ego_pos - obstacle.current_pos))

        # quit if no visible obstacle is available
        if not visible_stat_obst:
            return

        for stat_obst in visible_stat_obst:

            possible_spawn_points = []

            # break if maximum number is reached
            if len(spawn_points) >= self.max_pedestrians:
                break

            # check distance between vehicles and continue, if to far away
            distance_to_obstacle = np.linalg.norm(self.ego_pos - stat_obst.current_pos)
            if distance_to_obstacle > self.max_distance_to_other_obstacle:
                continue

            # try to calculate curvilinear coordinates of obstacle [s, d] -> if outside projection domain continue
            try:
                stat_obstacle_cl = self.cosy_cl.convert_to_curvilinear_coords(stat_obst.current_pos[0], stat_obst.current_pos[1])
            except:
                continue

            # if obstacle is behind ego or too far away, continue
            if self.s_threshold < stat_obstacle_cl[0] or stat_obstacle_cl[0] < self.ego_cl[0] + 3:
                continue

            # convert corner points of vehicle to curvilinear coordinates
            list_of_corner_points = [np.array([[x], [y]]) for x, y in stat_obst.current_corner_points]
            list_of_corner_points_cl = np.array(self.cosy_cl.convert_list_of_points_to_curvilinear_coords(list_of_corner_points, 4))

            # find minimum and maximum curvilinear coordinates of obstacle ( + s offset)
            s_offset = 0.8

            # Initialize a list for lines
            lines_cl = []

            # Process each corner point to create lines with +s and -s offsets
            for point in list_of_corner_points_cl:
                # Create adjusted points for +s and -s offsets
                adjusted_point_plus_s = point.copy()
                adjusted_point_minus_s = point.copy()

                adjusted_point_plus_s[0] += s_offset  # +s offset
                adjusted_point_minus_s[0] -= s_offset  # -s offset

                # Create lines for both adjusted points
                d_value = point[1]
                line_plus_s = np.array([[adjusted_point_plus_s[0], d_value], [adjusted_point_plus_s[0], 0]])
                line_minus_s = np.array([[adjusted_point_minus_s[0], d_value], [adjusted_point_minus_s[0], 0]])

                # Add the lines to the list
                lines_cl.append(line_plus_s)
                lines_cl.append(line_minus_s)

            # try to convert lines to cartesian coordinates -> failes sometimes due to the curvilinear cosy
            for line_cl in lines_cl:
                try:
                    line = np.array([self.cosy_cl.convert_to_cartesian_coords(p[0], p[1]) for p in line_cl])
                except:
                    continue

                # convert to LineString
                line_ls = LineString(line)
                # plt.plot(line[:, 0], line[:, 1], zorder=100)

                # if possible path does not intersect with visible area and occluded area --> continue
                if not line_ls.intersects(self.relevant_occluded_area) or not line_ls.intersects(self.sensor_model.visible_area):
                    continue

                # if line intersects with another visible obstacle
                if line_ls.intersects(self.fo_obstacles.visible_obstacle_multipolygon):
                    continue

                # find intersection with visible area -> this marks the possible spawn position
                try:
                    spawn_pos_point = self.sensor_model.visible_area.buffer(self.ped_length / 2 * 1.3). \
                        exterior.intersection(line_ls)
                except:
                    continue

                # check if calculated point is a multipoint --> find point closest to the left lanelet vertices and within
                # occluded area
                if spawn_pos_point.geom_type == 'MultiPoint':
                    # select one point of lanelet
                    obst_lanelet = self._find_lanelet_by_position(stat_obst.current_pos)
                    left_vertices_point = obst_lanelet.left_vertices[0]

                    # sort points
                    points = [point for point in spawn_pos_point.geoms]
                    points = sorted(points, key=lambda point: np.linalg.norm(left_vertices_point - [point.x, point.y]))

                    # iterate over points and
                    spawn_pos_point = None
                    for point in points:
                        if point.within(self.relevant_occluded_area):
                            spawn_pos_point = point
                            break

                # if no spawn pos point could be found, continue
                if spawn_pos_point is None:
                    continue

                # if buffered position intersects with visible area --> is visible --> continue
                if spawn_pos_point.buffer(0.15).intersects(self.sensor_model.visible_area):
                    continue

                # if spawn point buffer is not within the road polygon
                if not spawn_pos_point.buffer(0.15).within(self.sensor_model.road_polygon):
                    continue

                # buffer variables
                spawn_pos = np.array([spawn_pos_point.x, spawn_pos_point.y])
                spawn_pos_cl = self.cosy_cl.convert_to_curvilinear_coords(spawn_pos[0], spawn_pos[1])
                source = 'behind static obstacle ' + str(stat_obst.cr_obstacle.obstacle_id)

                # check longitudinal distance between new spawn point and already added spawn points
                if any(abs(s - spawn_pos_cl[0]) <= self.min_distance_between_pedestrians for s in s_positions):
                    continue

                # once the code reaches this spot, a suitable spawn point is found

                # find orientation of lanelet at obstacle position --> pedestrian orientation is orientation + 90°
                _, orientation = self._find_orientation_at_position(stat_obst.current_pos)
                orientation = orientation + np.pi/2

                # append spawn point and break for loop --> only one spawn point per obstacle
                possible_spawn_points.append(SpawnPoint(pos=spawn_pos, agent_type="Pedestrian", pos_cl=spawn_pos_cl,
                                                        source=source, orientation=orientation))

                # save curvilinear spawn points
                s_positions.append(spawn_pos_cl[0])

            if possible_spawn_points:

                for spawn_point in possible_spawn_points:
                    self.visualization.rnd.ax.plot(spawn_point.position[0], spawn_point.position[1], 'ro', zorder=100)

                closest_spawn_point = min(possible_spawn_points, key=lambda sp: np.linalg.norm(stat_obst.current_pos - sp.position))
                spawn_points.append(closest_spawn_point)
                s_positions.append(closest_spawn_point.cl_pos[0])

        return spawn_points

    #################################################################################
    ######################## Spawn Points behind turns ##############################
    #################################################################################

    def _find_spawn_point_behind_turn(self) -> SpawnPoint or None:
        """
            Finds a spawn point for a phantom agent behind a corner based on the ego vehicle's intention and the
            current environment configuration.

            This function calculates a potential spawn point for a pedestrian (phantom agent) by analyzing
            the intersection of the ego vehicle's reference path with occluded areas detected by its sensor model.
            It considers the ego vehicle's intention (e.g., left turn) and uses various offsets and thresholds to
            determine the most suitable spawn point. The function also checks if the suggested spawn point is within a
            visible area and adjusts it if necessary.

            Parameters:

            Returns:
            SpawnPoint or None: A SpawnPoint object containing the position and other details of the phantom agent if a
            suitable location is found. Returns None if no suitable spawn point is identified or if the suggested point
            does not meet certain criteria (e.g., not within visible area, not within another obstacle).

            Raises:
            ValueError: If an unknown intersection type is encountered during the process.

            Note:
            This function uses several internal properties and methods (`self.reference`, `self.sensor_model`, etc.)
            which are part of the class this function belongs to.
            """

        # check if the maximum number of spawn points has already been reached
        if len(self.spawn_points) >= self.max_pedestrians or self.ego_intention == 'straight ahead':
            return

        # shift the reference path to the left or right based on the ego vehicle's intention
        curvilinear_list = np.column_stack((self.reference_s,
                                            np.full(self.reference_s.shape, self.offset_ref_path[self.ego_intention])))

        # convert curvilinear coordinates to cartesian coordinates
        ref_path_parallel = np.vstack(self.cosy_cl.convert_list_of_points_to_cartesian_coords(curvilinear_list, 1))

        # create LineString from the parallel reference path
        ref_path_ls = LineString(ref_path_parallel)

        # find intersection of linestring and occluded area
        occluded_area_intersection = ref_path_ls.intersection(self.relevant_occluded_area)

        # check intersection type and find first intersection point
        if occluded_area_intersection.is_empty:
            return
        elif occluded_area_intersection.geom_type == 'LineString':
            coords = np.array(occluded_area_intersection.coords)
        elif occluded_area_intersection.geom_type == 'MultiLineString':
            coords = np.vstack([np.array(line.coords) for line in occluded_area_intersection.geoms])
            if self.debug:
                print('MultiLineString in spawn point processing detected')
        else:
            raise ValueError('Unknown intersection type!')

        # find point with the closest distance to ego vehicle
        distances = np.linalg.norm(coords - self.ego_pos, axis=1)
        closest_idx = np.argmin(distances)
        intersection = coords[closest_idx]

        # find s coordinate of first intersection
        intersection_cl = self.cosy_cl.convert_to_curvilinear_coords(intersection[0], intersection[1])

        # calculate s coordinate of possible phantom agent crossing point
        s_phantom = intersection_cl[0]
        d_phantom = intersection_cl[1]

        # init phantom pos
        phantom_pos = intersection

        iterations = 0
        while True:
            if iterations >= 10:
                return

            # end here if s_phantom is too far away (> s_threshold) or behind ego
            if s_phantom > self.s_threshold or s_phantom < self.ego_cl[0] + 3:
                return

            # check, if position is not completely in relevant occluded area
            if Point(phantom_pos).buffer(0.5).within(self.relevant_occluded_area):
                break

            s_phantom += 0.5
            phantom_pos = self.cosy_cl.convert_to_cartesian_coords(s_phantom, d_phantom)
            iterations += 1

        # check if spawn point is too close to already existing spawn point
        if any(abs(sp.cl_pos[0] - s_phantom) <= self.min_distance_between_pedestrians for sp in self.spawn_points if sp.agent_type == 'Pedestrian'):
            return

        # find ego lanelet id, lanelet and orientation
        ego_lanelet, ego_lanelet_orientation = self._find_orientation_at_position(self.ego_pos)

        # find lanelet id, lanelet and orientation at possible pedestrian point
        phantom_lanelet, phantom_lanelet_orientation = self._find_orientation_at_position(phantom_pos)

        # calculate orientation difference of both lanelets
        orientation_diff = abs((phantom_lanelet_orientation - ego_lanelet_orientation)) % (2 * np.pi)

        # check orientation difference against threshold (45°) -> if orientation is too similar,
        # the spawn point is not behind the turn -> return
        if orientation_diff < np.radians(45):
            return

        # create spawn point and return it
        spawn_point = SpawnPoint(pos=phantom_pos, agent_type="Pedestrian",
                                 pos_cl=[s_phantom, self.phantom_offset_d[self.ego_intention]], source=self.ego_intention)

        return spawn_point

    #################################################################################
    ############################### Helper functions ################################
    #################################################################################

    def _append_spawn_point(self, spawn_point):
        """
        Simple helper function to append spawn point to list self.spawn_points if it is not None
        """
        if spawn_point is not None:
            if type(spawn_point) is list:
                self.spawn_points.extend([point for point in spawn_point if point is not None])
            else:
                self.spawn_points.append(spawn_point)

    def _find_orientation_at_position(self, pos, lanelet=None):
        """
        Find lanelet id, lanelet and orientation at given point
        """
        if lanelet is None:
            lanelet = self._find_lanelet_by_position(pos)
        lanelet_orientation = lanelet_orientation_at_position(lanelet, pos)

        return lanelet, lanelet_orientation

    def _find_lanelet_by_position(self, pos):
        """
        Find lanelet at given point
        """
        lanelet_id = self.scenario.lanelet_network.find_lanelet_by_position([pos])
        if not lanelet_id[0]:
            return None
        lanelet_id = lanelet_id[0][0]
        lanelet = self.scenario.lanelet_network.find_lanelet_by_id(lanelet_id)

        return lanelet

    def _prepare_reference_path(self, ego_cl, distance=50):
        """
        Cut reference path to "length" of interest
        """
        # compute path length of reference path
        self.s = compute_pathlength_from_polyline(self.ref_path)

        # find start and end index of reference path
        index_start = self._find_nearest_index(self.s, ego_cl[0])
        index_end = self._find_nearest_index(self.s, ego_cl[0] + distance)

        # cut reference path to interesting area
        reference = self.ref_path[index_start:index_end]
        reference_s = self.s[index_start:index_end]

        return reference, reference_s

    @staticmethod
    def _find_ego_intention(reference):
        """
        Find the ego intention of the vehicle within the reference line by analyzing the curvature
        """
        # Calculate curvature of the reference path
        curvature = compute_curvature_from_polyline(reference)
        # Determine the vehicle intention
        if max(curvature) > 0.12:
            return "left turn"
        elif min(curvature) < -0.12:
            return "right turn"
        else:
            return "straight ahead"

    @staticmethod
    def _find_nearest_index(path_s, current_s):
        """
        Find the index in the path that is closest to the current s-coordinate.
        """
        # Calculate the difference between each s-coordinate in the path and the current s-coordinate
        differences = np.abs(path_s - current_s)
        # Find the index of the smallest difference
        nearest_index = np.argmin(differences)
        return nearest_index

    def _find_and_process_relevant_occlusions(self, relevant_occluded_area):
        """
        Find occlusions that intersect with the relevant occluded area.

        Args:
            relevant_occluded_area (Polygon): The area to check for intersections.

        Returns:
            list[Occlusion]: List of occlusions that intersect with the relevant occluded area.
        """

        relevant_occlusions = []
        reference_ls = LineString(self.reference)

        for occlusion in self.occlusion_tracker.occlusions:

            # if the occlusion lane does not intersect with the ego vehicles reference path -> continue
            if not occlusion.tracked_lane.intersection_with_ego_v_reference_path:
                continue

            # if the occlusion lane does not intersect with the shortened ego vehicles reference path -> continue
            if not LineString(occlusion.lanelet.center_vertices).intersects(reference_ls):
                continue

            # find intersection of occlusion polygon and relevant occluded area
            intersection = occlusion.polygon.intersection(relevant_occluded_area)

            # if the intersection exists, and it is greater than 1.0, set the relevant polygon and calculate the center points
            if intersection.area > 2.0:
                occlusion.set_relevant_polygon(intersection)

                # if center points could not be calculated, do not consider the occlusion
                if self.spawn_vehicles_and_bicycles and occlusion.calc_center_points():
                    relevant_occlusions.append(occlusion)
        return relevant_occlusions
