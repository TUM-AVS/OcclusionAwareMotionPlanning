__author__ = "Korbinian Moller,"
__copyright__ = "TUM Professorship Autonomous Vehicle Systems"
__version__ = "1.0"
__maintainer__ = "Korbinian Moller"
__email__ = "korbinian.moller@tum.de"
__status__ = "Beta"

import os
import csv
import warnings
import numpy as np
from shapely.geometry import Polygon, MultiPolygon, LineString, Point, MultiLineString
from shapely.ops import unary_union, substring
import frenetix_occlusion.utils.helper_functions as hf
from commonroad.scenario.lanelet import LaneletType
from centerline.geometry import Centerline


class TrackedLane:
    """
    Represents a tracked lane in the road network.

    Attributes:
        lanelet (Lanelet): The CommonRoad lanelet object.
        propagation_speed (float): The calculated propagation speed for the lane.
        lane_config(dict): Configuration settings for the lane.
    """

    def __init__(self, lanelet, lane_config, ego_v_reference_path):
        """
        Initialize the Lane object with a CommonRoad lanelet.

        Args:
            lanelet (Lanelet): The CommonRoad lanelet object.
            lane_config (dict): Configuration settings for the lane.
            ego_v_reference_path (np.array): Reference path of the ego vehicle.
        """
        self.lanelet = lanelet
        self.polygon = lanelet.polygon.shapely_object
        self.lane_config = lane_config
        self.sidewalk = False
        self.propagation_speed = self._calculate_propagation_speed()
        self.intersection_with_ego_v_reference_path = self._check_if_intersects(ego_v_reference_path)

    @property
    def polygon(self):
        return self._polygon

    @polygon.setter
    def polygon(self, polygon):
        if polygon.is_valid:
            self._polygon = polygon
        else:
            self._polygon = polygon.buffer(0.001)
            warnings.warn(f"Invalid lanelet polygon: {self.lanelet.lanelet_id}! It was buffered to fix error!")

    def _calculate_propagation_speed(self):
        """
        Calculate the propagation speed for the lane based on its attributes and configuration.

        Returns:
            float: The calculated propagation speed.
        """
        if LaneletType.SIDEWALK in self.lanelet.lanelet_type:
            speed_setting = self.lane_config.propagation_speed_sidewalk
            self.sidewalk = True
        else:
            speed_setting = self.lane_config.propagation_speed_street

        if speed_setting == 'lanelet':
            return self._get_lanelet_speed_limit()
        else:
            return float(speed_setting)

    def _get_lanelet_speed_limit(self):
        """
        Retrieve the speed limit based on the lanelet type.

        Returns:
            float: Speed limit in m/s.
        """
        lanelet_type = self.lanelet.lanelet_type  # Assuming all lanelets have the same type for simplicity

        if LaneletType.URBAN in lanelet_type:
            v = 20.0 / 3.6  # 30 km/h in urban environments
        elif LaneletType.SIDEWALK in lanelet_type:
            v = 1.5  # 1.5 m/s for sidewalks
        elif LaneletType.COUNTRY in lanelet_type:
            v = 80 / 3.6  # 80 km/h on country roads (take me home, to the place, I belong ....)
        elif LaneletType.HIGHWAY in lanelet_type:
            v = 100.0 / 3.6  # 100 km/h on highways
        else:
            warnings.warn(f"Unknown lanelet type {lanelet_type}. Using default speed limit.")
            v = 20.0 / 3.6  # Default speed limit in m/s

        return v

    def _check_if_intersects(self, ego_v_reference_path):
        """
        Check if the lanelet intersects with the ego vehicle's reference path.

        Args:
            ego_v_reference_path (np.array): Reference path of the ego vehicle.

        Returns:
            bool: True if the lanelet intersects with the reference path.
        """

        if ego_v_reference_path is None or LineString(self.lanelet.center_vertices).intersects(LineString(ego_v_reference_path)):
            return True
        else:
            return False


class Occlusion:
    """
    Represents a single occluded area associated with a specific lanelet.

    Attributes:
        polygon (Polygon): The occluded area.
        tracked_lane (TrackedLane): The associated lane of the occlusion of type Lane.
        center_line (LineString): The center line of the lanelet.
        right_line (LineString): The right boundary of the lanelet.
        left_line (LineString): The left boundary of the lanelet.
    Methods:
        expand: expand the occlusion along the lanelet's geometry
    """

    def __init__(self, polygon, tracked_lane):
        self.polygon = polygon
        self.relevant_polygon = None
        self.center_points = None
        self.tracked_lane = tracked_lane  # The tracked lane associated with the occlusion
        self.center_line = LineString(self.lanelet.center_vertices)  # Center line of the lanelet
        self.right_line = LineString(self.lanelet.right_vertices)  # Right boundary of the lanelet
        self.left_line = LineString(self.lanelet.left_vertices)  # Left boundary of the lanelet

    @property
    def lanelet(self):
        return self.tracked_lane.lanelet

    def set_relevant_polygon(self, relevant_polygon):
        self.relevant_polygon = relevant_polygon

    def expand(self, dt):
        """
        Expand the occluded area along the lanelet's geometry.

        Args:
            dt (float): The timestep by which to expand the occlusion.
        """
        dist = dt * self.tracked_lane.propagation_speed
        self.polygon = self._get_next_occ(self.polygon, dist)

    def _get_next_occ(self, poly, dist):
        """
        Compute the next state of the occlusion by expanding it along the lanelet.

        Args:
            poly (Polygon): Current occlusion polygon.
            dist (float): Distance to expand.

        Returns:
            Polygon: Expanded polygon.
        """
        smallest_projection = float('inf')
        # Find the smallest projection along the center line to limit the expansion
        for edge in poly.exterior.coords:
            projection = self.center_line.project(Point(edge))
            if projection < smallest_projection:
                smallest_projection = projection
            if smallest_projection <= 0:
                break

        # Create a substring of the center line starting from the smallest projection
        sub_center_line = substring(self.center_line, smallest_projection, self.center_line.length)

        # If the lane is a sidewalk, expand the occlusion in both directions
        if self.tracked_lane.sidewalk:
            sub_center_line = self.center_line

        # Buffer the center line to create a relevant area and intersect it with the lanelet polygon
        relevant_area_lane = hf.buffer_sides_only(sub_center_line, 2.8).intersection(self.tracked_lane.polygon)

        # Expand the polygon by the specified distance
        poly = poly.buffer(dist, join_style=1)

        # Ensure expansion stays within the relevant area (-> expansion only in allowed driving direction)
        poly = poly.intersection(relevant_area_lane)

        return poly

    def calc_center_points(self):
        try:
            center_mls = Centerline(self.relevant_polygon, interpolation_distance=1.0).geometry
            self.center_points = self._extract_points_from_multilinestring(center_mls)
            return True
        except:
            self.center_points = None
            return False

    @staticmethod
    def _extract_points_from_multilinestring(multilinestring: MultiLineString) -> np.array:
        """
        Extracts unique points from a MultiLineString and returns them as a numpy array of coordinates.

        Args:
            multilinestring (MultiLineString): A shapely MultiLineString object.

        Returns:
            unique_points[np.array]: A np array of unique points (x, y) from the MultiLineString.
        """
        if not isinstance(multilinestring, MultiLineString):
            raise ValueError("Input must be a MultiLineString")

        # Extract all points from each LineString in the MultiLineString
        points = []
        for line in multilinestring.geoms:
            points.extend(line.coords)

        # Remove duplicates while preserving order
        unique_points = list(dict.fromkeys(points))

        return np.array(unique_points)


class OcclusionTracker:
    """
    The OcclusionTracker class manages the occluded areas in the environment.

    Attributes:
        lanelet_network (LaneletNetwork): The lanelet network of the current scenario.
        road_polygon (Polygon): The overall road polygon.
        lanelet_network_polygons (list): List of polygons representing the lanelet network.
        config (dict): Configuration settings.
        _occlusions (list[Occlusion]): Previously calculated occlusions.
        tracked_lanes (list): Precomputed collection of possible routes (sequences of lanelets).
        total_occluded_area (Polygon or MultiPolygon): Unified representation of all occluded areas.
        visualization (FOVisualization): Tool for visualizing occluded areas and scenarios.
    """

    def __init__(self, lanelet_network, road_polygon, lanelet_network_polygons, config, ego_v_reference_path=None,
                 visualization=None, scenario=None, sensor_model=None):
        self.lanelet_network = lanelet_network  # The lanelet network for the scenario
        self.road_polygon = road_polygon  # The overall road area as a polygon
        self.lanelet_network_polygons = lanelet_network_polygons  # Polygons representing lanelets
        self.config = config  # Configuration settings
        self.tracking_enabled = config.tracking_enabled  # Whether occlusion tracking is enabled
        self._occlusions = []  # List of individual occluded areas
        self._unique_occlusions = []  # List of unique occluded areas
        self.total_occluded_area = None  # Unified occluded area for external use
        self.summed_occluded_area = 0
        self.visualization = visualization
        self.ego_v_reference_path = ego_v_reference_path

        # evaluation
        self.scenario = scenario
        self.sensor_model = sensor_model

        # find all possible lanes through the scenario (Collection of lanelets from start to end of the scenario)
        self.tracked_lanes = self._generate_lanes()

    @property
    def occlusions(self):
        return self._occlusions

    @occlusions.setter
    def occlusions(self, value):
        raise AttributeError("Use _add_occlusion method to modify occlusions.")

    @property
    def unique_occlusions(self):
        """
        Returns a list of unique occlusions, ensuring no duplicates based on polygon.

        Returns:
            list[Occlusion]: List of unique occlusions.
        """
        if not self._unique_occlusions:
            self._unique_occlusions = self._get_unique_occlusions()

        return self._unique_occlusions

    def _get_unique_occlusions(self):
        """
        Returns a list of unique occlusions, ensuring no duplicates based on polygon.

        Returns:
            list[Occlusion]: List of unique occlusions.
        """
        unique_occlusions = []
        seen_polygons = set()

        for occlusion in self._occlusions:
            polygon_key = occlusion.polygon.wkt
            if polygon_key not in seen_polygons:
                unique_occlusions.append(occlusion)
                seen_polygons.add(polygon_key)

        return unique_occlusions

    def _generate_lanes(self):
        """
        Generate possible routes (sequences of lanelets) from start to end of the scenario.

        Returns:
            list: List of routes, where each route is a sequence of lanelets.
        """
        tracked_lanes = []

        # Find all initial lanelets (without predecessors)
        initial_lanelets = [lanelet for lanelet in self.lanelet_network.lanelets if not lanelet.predecessor]

        for initial_lanelet in initial_lanelets:
            # Generate all possible successor routes for each initial lanelet
            curr_lanes, _ = initial_lanelet.all_lanelets_by_merging_successors_from_lanelet(initial_lanelet,
                                                                                            self.lanelet_network, max_length=500)
            for lane in curr_lanes:
                tracked_lanes.append(TrackedLane(lane, self.config.lane_config, self.ego_v_reference_path))
        return tracked_lanes

    def _initialize_occlusions(self, visible_area, timestep=None):
        """
        Initialize the occlusions using the lanelet network area and the visible area.

        This function is called when there are no existing occlusions to initialize the occluded areas.

        Args:
            visible_area (Polygon or MultiPolygon): The currently visible area.

        Returns:
            list[Occlusion]: Initialized occluded areas.
        """

        # Calculate the total occluded area from the visible area
        self._calc_total_occlusion_from_visible_area(visible_area)

        self.write_csv(timestep)

        # Iterate over all lanes to find intersections with the total occluded area
        for tracked_lane in self.tracked_lanes:
            intersection = self.total_occluded_area.intersection(tracked_lane.polygon)

            # Check if the intersection is not empty
            if not intersection.is_empty:
                # If the intersection is a MultiPolygon, iterate over each polygon
                if isinstance(intersection, MultiPolygon):
                    for poly in intersection.geoms:
                        # Add occlusion if the polygon area is greater than the minimum occlusion area
                        if poly.area > self.config.min_occlusion_area:
                            occlusion = Occlusion(poly, tracked_lane)
                            self._add_occlusion(occlusion)

                # If the intersection is a single polygon, add it if the area is greater than the minimum occlusion area
                elif intersection.area > self.config.min_occlusion_area:
                    occlusion = Occlusion(intersection, tracked_lane)
                    self._add_occlusion(occlusion)

    def update_tracker(self, visible_area, dt=0.1, replanning_counter=1, timestep=None):
        """
        Update the occluded areas by propagating them along the lanes and integrating the sensor view.

        Args:
            visible_area (Polygon or MultiPolygon): The currently visible area detected by the sensor.
            dt (float): Timestep size (usually 0.1).
            replanning_counter: Indicates after how many timesteps a new trajectory is planned.

        Returns:
            Polygon or MultiPolygon: The total occluded area after updating.
        """
        # Check if tracking is enabled and if there are existing occlusions
        if not self.tracking_enabled or not self.occlusions:
            # Initialize occlusions if tracking is not enabled or no occlusions exist
            self._initialize_occlusions(visible_area, timestep=timestep)
            # Visualize the total occluded area
            # self.visualization.plot_poly_fast(self.total_occluded_area, color='red', fill=True, zorder=100, opacity=0.5)
            return self.total_occluded_area

        new_occlusions = []
        for occlusion in self.occlusions:

            # Expand the occlusion based on the propagation speed and timestep
            occlusion.expand(dt * replanning_counter)

            # Calculate the difference between the occlusion polygon and the visible area
            intersections = occlusion.polygon.difference(visible_area)

            # Check the type of intersections and extract geometries accordingly
            if isinstance(intersections, MultiPolygon):
                geoms = intersections.geoms  # Extract geometries from MultiPolygon
            elif isinstance(intersections, Polygon):
                geoms = [intersections]  # Wrap single Polygon in a list
            else:
                # Raise an error if the geometry type is unexpected
                raise TypeError(f"Unexpected geometry type: {type(intersections)}")

            # Iterate over the extracted geometries
            for geom in geoms:
                # Add valid and non-empty intersections as new occlusions
                if geom.is_valid and not geom.is_empty and geom.area > self.config.min_occlusion_area:
                    new_occlusions.append(Occlusion(geom, occlusion.tracked_lane))

        # Update the list of occlusions with the new occlusions
        # Clear the existing occlusions list
        self._occlusions.clear()
        self._unique_occlusions.clear()

        # Add new occlusions using the add_occlusion method
        for new_occlusion in new_occlusions:
            self._add_occlusion(new_occlusion)

        # Update the total occluded area by combining all individual occlusions
        self._update_total_occluded_area(timestep=timestep)

        # Return the total occluded area
        return self.total_occluded_area

    def _add_occlusion(self, occlusion):
        """
        Add a new occlusion to the list of occlusions

        Args:
            occlusion (Occlusion): The occlusion to add.
        """
        # Add the occlusion if it does not exist
        self._occlusions.append(occlusion)

    def _clear_occlusions(self):
        """
        Clear the list of occlusions.
        """
        self._occlusions.clear()

    def _update_total_occluded_area(self, timestep=None):
        """
        Combine all individual occlusions into a unified representation for external use.
        """
        # combine all occlusions to a single polygon
        total_occluded_area = unary_union([occlusion.polygon for occlusion in self.occlusions])

        # remove small artifacts
        self.total_occluded_area = total_occluded_area.buffer(0.01)

        self.write_csv(timestep)

    def write_csv(self, timestep):
        # Prepare CSV file
        csv_filename = f"{self.scenario.scenario_id}_occlusion_data.csv"
        file_exists = os.path.isfile(csv_filename)

        # sum it up for evaluation
        total_area = self.total_occluded_area.intersection(self.sensor_model.sensor_sector).area
        self.summed_occluded_area += total_area

        # Prepare the data to write
        data_row = {
            "timestep": timestep if timestep is not None else "N/A",
            "total": total_area,
            "summed": self.summed_occluded_area
        }

        # Write to CSV file
        with open(csv_filename, mode="a", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=["timestep", "total", "summed"])

            # Write the header only if the file is newly created
            if not file_exists:
                writer.writeheader()

            # Write the current data row
            writer.writerow(data_row)

        print(f"Occlusion data written to {csv_filename}")

    def _calc_total_occlusion_from_visible_area(self, visible_area):
        # calculate
        occluded_areas = self.road_polygon.difference(visible_area)
        occluded_areas = hf.remove_unwanted_shapely_elements(occluded_areas)

        self.total_occluded_area = occluded_areas

        return self.total_occluded_area

# EOF
