__author__ = "Korbinian Moller,"
__copyright__ = "TUM AVS"
__version__ = "2.0"
__maintainer__ = "Korbinian Moller"
__email__ = "korbinian.moller@tum.de"
__status__ = "Beta"

# imports
from commonroad_route_planner.utility.route_util import lanelet_orientation_at_position
from commonroad_route_planner.route_generation_strategies.default_generation_strategy import DefaultGenerationStrategy
from shapely.geometry import LineString


class FORoutePlanner:
    def __init__(self, scenario, lanelet_network, visualization, debug):
        # global variables - never change
        self.cr_scenario = scenario
        self.lanelet_network = lanelet_network
        self.debug = debug
        self.visualization = visualization

        # changing variables
        self.orientations = None
        self.start_lanelets = None
        self.route_candidates = None
        self.reference_paths = None

    def calc_possible_reference_paths(self, pos, ego_v_reference_path, add_to_scenario=False):
        # Initialize lists to store reference paths and their orientations
        self.reference_paths = []
        self.orientations = []

        # Find current lanelet ids and lanelets
        start_lanelet_ids = self.cr_scenario.lanelet_network.find_lanelet_by_position([pos])[0]

        for start_lanelet_id in start_lanelet_ids:
            start_lanelet = self.cr_scenario.lanelet_network.find_lanelet_by_id(start_lanelet_id)

            # Find lanelet orientation at initial position
            orientation = lanelet_orientation_at_position(start_lanelet, pos)

            # Calculate all routes using recursive function
            routes = self._find_all_routes(start_lanelet_id, max_depth=2)

            # Convert "graph ids" to real route using commonroad Route class
            ref_path_candidates = [
                DefaultGenerationStrategy.generate_route(
                    lanelet_network=self.lanelet_network,
                    lanelet_ids=route,
                    goal_region=None,
                    initial_state=None
                )
                for route in routes if route
            ]

            # Filter reference paths that intersect with ego_v_reference_path
            for path in ref_path_candidates:
                if LineString(path.reference_path).intersects(LineString(ego_v_reference_path)) or add_to_scenario:
                    self.reference_paths.append(path.reference_path)
                    self.orientations.append(orientation)

        if self.debug is True and self.visualization is not None:  # ToDo remove when not needed anymore
            self.visualization.draw_point(pos, color='k', zorder=20)
            for ref in self.reference_paths:
                self.visualization.draw_reference_path(ref, zorder=20)

    def _find_all_routes(self, id_lanelet_start, max_depth=2):
        all_routes = []
        self._explore_routes(id_lanelet_start, [], all_routes, 0, max_depth)
        if not all_routes:
            raise ValueError('[Frenetix Occlusion - Route Planner] Route Explorer could not find a Route')
        return all_routes

    def _explore_routes(self, id_lanelet_current, route, all_routes, depth, max_depth):
        lanelet = self.lanelet_network.find_lanelet_by_id(id_lanelet_current)

        # Add current lanelet to the route
        route.append(lanelet.lanelet_id)

        successors = []
        if lanelet.successor:
            successors.extend(lanelet.successor)
        if lanelet.adj_right and lanelet.adj_right_same_direction:
            lanelet_adj_right = self.lanelet_network.find_lanelet_by_id(lanelet.adj_right)
            if lanelet_adj_right.successor:
                successors.append(lanelet.adj_right)
        if lanelet.adj_left and lanelet.adj_left_same_direction:
            lanelet_adj_left = self.lanelet_network.find_lanelet_by_id(lanelet.adj_left)
            if lanelet_adj_left.successor:
                successors.append(lanelet.adj_left)

        if depth >= max_depth:
            successors = []
            # Max depth reached, return without exploring further

        if not successors:
            # If no successors, save route and return
            all_routes.append(route.copy())
            return

        for successor in successors:
            self._explore_routes(successor, route, all_routes, depth + 1, max_depth)
            route.pop()  # Backtrack to explore other paths



