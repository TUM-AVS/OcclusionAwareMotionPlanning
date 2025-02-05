__author__ = "Korbinian Moller,"
__copyright__ = "TUM Professorship Autonomous Vehicle Systems"
__version__ = "1.0"
__maintainer__ = "Korbinian Moller"
__email__ = "korbinian.moller@tum.de"
__status__ = "Beta"

# imports
import matplotlib as mpl
import matplotlib.pyplot as plt
import logging
from matplotlib.lines import Line2D
import os

from commonroad.visualization.mp_renderer import MPRenderer
from commonroad.visualization.draw_params import MPDrawParams

from frenetix_occlusion.utils.prediction_visualization import draw_uncertain_predictions
import frenetix_occlusion.utils.helper_functions as hf

# get logger
msg_logger = logging.getLogger("Message_logger")


class FOVisualization:
    def __init__(self, scenario, reference_path, plot, title="Occlusion Aware Motion Planning"):

        # figsize and plot limits
        x_min = -90
        x_max = 90
        y_min = -70
        y_max = 70
        self.plot_limits = [x_min, x_max, y_min, y_max]
        self.figsize = (20, 15)

        # interactive mode
        self.interactive = True
        self.backend = "TkAgg"
        self.plot = plot

        # set backend
        if self.interactive:
            mpl.use(self.backend)

        # load global variables that never change
        self.cr_scenario = scenario
        self.lanelet_network = scenario.lanelet_network
        self.ego_reference_path = reference_path
        self.fig, ax = plt.subplots(figsize=self.figsize)
        self.fig.suptitle(title)
        self.rnd = MPRenderer(ax=ax, figsize=self.figsize, plot_limits=self.plot_limits)  # plot_limits=self.plot_limits

        # init draw params
        self.params = MPDrawParams()
        self.params.time_begin = 0
        self.params.dynamic_obstacle.draw_icon = False
        self.params.dynamic_obstacle.draw_shape = True
        self.params.dynamic_obstacle.show_label = True
        self.params.dynamic_obstacle.vehicle_shape.occupancy.shape.facecolor = "#0065bd"  # E37222"
        self.params.dynamic_obstacle.vehicle_shape.occupancy.shape.edgecolor = "#003359"
        self.params.dynamic_obstacle.trajectory.draw_trajectory = False
        self.params.lanelet_network.lanelet.show_label = True

        self.params.static_obstacle.occupancy.shape.facecolor = "#a30000"
        self.params.static_obstacle.occupancy.shape.edgecolor = "#756f61"

        # init dynamic variables
        self.timestep = None
        self.frame_exits = False

    @property
    def ax(self):
        return self.rnd.ax

    def set_focus(self, focus_point, distance_x, distance_y, ax=None):
        if ax is None:
            ax = self.ax

        x, y = focus_point

        ax.set_xlim(x - distance_x, x + distance_x)
        ax.set_ylim(y - distance_y, y + distance_y)

    def draw_scenario(self, timestep=0):
        self.params.time_begin = timestep
        self.cr_scenario.draw(self.rnd, draw_params=self.params)
        self.rnd.render()

    def draw_reference_path(self, ref_path, color='y', zorder=10):
        self.rnd.ax.plot(ref_path[:, 0], ref_path[:, 1], color=color, zorder=zorder)

    def fill_area(self, poly, color='b', zorder=10, opacity=0.5):
        self.plot_poly_fast(poly, color=color, zorder=zorder, opacity=opacity, fill=True)

    def draw_point(self, point, color='b', marker='o', zorder=10):
        self.rnd.ax.plot(point[0], point[1], color=color, marker=marker, zorder=zorder)

    def draw_predictions(self, predictions, label=False, ax=None, **kwargs):

        if ax is None:
            ax = self.rnd.ax

        draw_uncertain_predictions(predictions, ax, number_of_predictions=40, draw_to=40)

        if label:
            # show label in plot
            for key in predictions.keys():
                ax.annotate(str(key), xy=(predictions[key]['pos_list'][-1]), zorder=20)

    def save_plot(self, directory, timestep):
        if not os.path.exists(directory):
            os.makedirs(directory)
        filename = os.path.join(directory, f"plot_{timestep}.png")
        self.rnd.f.savefig(filename)
        print(f"Plot saved to {filename}")

    def show_plot(self, time=0.1):
        if self.interactive is True:
            plt.show(block=False)
            plt.pause(time)

    def plot_poly_fast(self, poly, color='b', zorder=1, opacity=1.0, fill=False):

        ax = self.rnd.ax
        if fill is False:
            poly = self.plot_polygons(ax, poly, color, zorder, opacity)
        else:
            poly = self.fill_polygons(ax, poly, color, zorder, opacity)

        return poly

    @staticmethod
    def plot_polygons(ax, polys, color='b', zorder=1, opacity=1.0):
        """
        Helper function to plot polygons.
        Args:
            ax: axis
            polys: list of polygons, Polygon or MultiPolygon
            color: optional color
            zorder: optional zorder ("importance of the plot")
            opacity: opacity between 0 and 1
        """
        ret_obj = None

        try:
            if type(polys) == list:
                for pol in polys:
                    if pol.geom_type == 'Polygon':
                        ret_obj, = ax.plot(pol.exterior.xy[0], pol.exterior.xy[1], color, zorder=zorder, alpha=opacity)

            elif polys.geom_type == 'Polygon':
                ret_obj, = ax.plot(polys.exterior.xy[0], polys.exterior.xy[1], color, zorder=zorder, alpha=opacity)

            elif polys.geom_type == 'MultiPolygon' or polys.geom_type == 'GeometryCollection':
                for pol in polys.geoms:
                    if pol.geom_type == 'Polygon':
                        ret_obj, = ax.plot(pol.exterior.xy[0], pol.exterior.xy[1], color, zorder=zorder, alpha=opacity)

            return ret_obj
        except:
            msg_logger.warning('Could not plot the Polygon')

    @staticmethod
    def fill_polygons(ax, polys, color='b', zorder=1, opacity=1.0):
        """
        Helper function to fill polygons.
        Args:
            ax: axis
            polys: list of polygons, Polygon or MultiPolygon
            color: optional color
            zorder: optional zorder ("importance of the plot")
            opacity: opacity between 0 and 1
        """
        try:
            if type(polys) == list:
                for pol in polys:
                    if pol.geom_type == 'Polygon':
                        ret_obj = ax.fill(pol.exterior.xy[0], pol.exterior.xy[1], color, zorder=zorder, alpha=opacity)

            elif polys.geom_type == 'Polygon':
                ret_obj = ax.fill(polys.exterior.xy[0], polys.exterior.xy[1], color, zorder=zorder, alpha=opacity)

            elif polys.geom_type == 'MultiPolygon' or polys.geom_type == 'GeometryCollection':
                for pol in polys.geoms:
                    if pol.geom_type == 'Polygon':
                        ret_obj = ax.fill(pol.exterior.xy[0], pol.exterior.xy[1], color, zorder=zorder, alpha=opacity)

            return ret_obj
        except:
            msg_logger.warning('Could not plot the Polygon')

    def plot_trajectories(self, trajectories, color='k', zorder=15):
        if trajectories is None:
            return

        try:
            if type(trajectories) == list:
                for traj in trajectories:
                    self.rnd.ax.plot(traj.cartesian.x, traj.cartesian.y, color=color, zorder=zorder)
            else:
                obj = self.rnd.ax.plot(trajectories.cartesian.x, trajectories.cartesian.y, color=color, zorder=zorder)
                return obj[0]
        except:
            msg_logger.warning('Could not plot the Trajectory')

    def plot_metric_hr(self, trajectory, hr, min_risk=0, max_risk=1, print_risks=True, legend=False):
        if hr is None:
            return

        # plot trajectories with color according to their costs

        metric = hr['max_obst_risk_all']

        if legend:
            # create Legend
            line_c = Line2D([0], [0], label='metric 0% of max. metric', color='c')
            line_g = Line2D([0], [0], label='metric in the range of 1%-25% of max. metric', color='g')
            line_y = Line2D([0], [0], label='metric in the range of 26%-50% of max. metric', color='y')
            line_o = Line2D([0], [0], label='metric in the range of 51%-75% of max. metric', color='orange')
            line_r = Line2D([0], [0], label='metric in the range of 76%-100% of max. metric', color='r')

            # access legend objects automatically created from data
            handles, labels = self.rnd.ax.get_legend_handles_labels()
            handles.extend([line_c, line_g, line_y, line_o, line_r])
            self.rnd.ax.legend(handles=handles, loc='upper left')

        if metric == min_risk:
            self.plot_trajectories([trajectory], color='c')
        elif metric <= 0.25 * max_risk:
            self.plot_trajectories([trajectory], color='g')
        elif 0.25 * max_risk < metric <= 0.5 * max_risk:
            self.plot_trajectories([trajectory], color='y')
        elif 0.5 * max_risk < metric <= 0.75 * max_risk:
            self.plot_trajectories([trajectory], color='orange')
        else:
            self.plot_trajectories([trajectory], color='r')

        if print_risks:
            if hr is not None:
                print('trajectory: risk {}, harm {}'.format(metric, hr['max_obst_harm_all']))

    def plot_dynamic_obstacle(self, dyn_obstacle, color='r', opacity=0.5):
        # plots state list of commonroad obstacles
        for occupancy in dyn_obstacle.prediction.occupancy_set:
            self.plot_poly_fast(occupancy.shape.shapely_object, color=color, opacity=opacity, zorder=20)

    def plot_vector(self, pos, orientation, color='r', zorder=20):

        [dx, dy] = hf.vector_from_angle(orientation)

        self.rnd.ax.arrow(pos[0], pos[1], dx * 5, dy * 5, color=color, zorder=zorder, width=0.2, head_width=1)
