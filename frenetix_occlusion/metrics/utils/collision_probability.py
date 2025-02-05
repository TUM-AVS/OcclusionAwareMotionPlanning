__author__ = "Maximilian Geisslinger, Rainer Trauth, Korbinian Moller,"
__copyright__ = "TUM Professorship Autonomous Vehicle Systems"
__version__ = "2.0"
__maintainer__ = "Korbinian Moller"
__email__ = "korbinian.moller@tum.de"
__status__ = "Beta"

# imports
import numpy as np
from scipy.stats import multivariate_normal


def get_collision_probability(traj, predictions, vehicle_params, safety_margin=0.0, debug=False):
    """
    Calculates the collision probabilities of a trajectory with predicted pedestrian positions.

    Args:
        traj (FrenetTrajectory): The considered trajectory of the ego vehicle.
        predictions (dict): Predictions of visible pedestrians, including position and covariance.
        vehicle_params (VehicleParameters): Parameters of the ego vehicle (length, width, etc.).
        safety_margin (float): Additional safety margin to consider around the vehicle dimensions.
        debug (bool): Whether to validate the collision probability using Monte Carlo sampling.

    Returns:
        dict: Collision probability per time step for each visible pedestrian.
    """
    collision_prob_dict = {}
    ego_dimensions = (vehicle_params.length + safety_margin, vehicle_params.width + safety_margin)
    ego_positions = np.stack((traj.cartesian.x, traj.cartesian.y), axis=-1)
    ego_orientations = traj.cartesian.theta

    for obstacle_id, prediction in predictions.items():
        probs = []
        mean_list = prediction['pos_list']
        cov_list = prediction['cov_list']
        min_len = min(len(traj.cartesian.x), len(mean_list))

        # distance from ego vehicle
        distance_array = mean_list - ego_positions
        distance_array = np.sqrt(distance_array[:, 0] ** 2 + distance_array[:, 1] ** 2)

        # bool: whether min distance is larger than 10.0
        min_distance_array = distance_array > 10.0

        for i in range(min_len):

            if min_distance_array[i]:
                prob = 0.0
            else:
                # store values in temporary variables
                ego_orientation = ego_orientations[i]
                ego_pos = ego_positions[i]
                ego_center_pos = shift_to_vehicle_center(ego_pos, ego_orientation, vehicle_params.wb_rear_axle)
                pedestrian_mean = np.array(mean_list[i][:2])
                pedestrian_cov = np.array(cov_list[i][:2, :2])

                # Ensure that the covariance matrix is positive semi-definite
                pedestrian_cov = ensure_positive_semi_definite(pedestrian_cov)

                # Compute the collision probability using the CDF method (analytical)
                prob = compute_collision_probability(ego_center_pos, ego_orientation, ego_dimensions, pedestrian_mean, pedestrian_cov)

                if debug:
                    # Validation of the collision probability using Monte Carlo sampling
                    monte_carlo_prob = monte_carlo_collision_probability(
                        ego_center_pos, ego_orientation, ego_dimensions, pedestrian_mean, pedestrian_cov, num_samples=10000)

                    if not np.isclose(prob, monte_carlo_prob, atol=0.01):
                        print(f"Collision Probability: {prob}")
                        print(f"Monte Carlo Collision Probability: {monte_carlo_prob}")

            probs.append(prob)

        collision_prob_dict[obstacle_id] = np.array(probs)

    return collision_prob_dict


def compute_collision_probability(ego_pos, ego_orientation, ego_dimensions, obstacle_mean, obstacle_cov):
    """
    Computes the collision probability between the ego vehicle and a pedestrian at a given time step.

    Args:
        ego_pos (np.ndarray): [x, y] position of the ego vehicle.
        ego_orientation (float): Orientation angle (theta) of the ego vehicle in radians.
        ego_dimensions (tuple): (length, width) dimensions of the ego vehicle.
        obstacle_mean (np.ndarray): [x, y] expected position of the pedestrian.
        obstacle_cov (np.ndarray): 2x2 covariance matrix of the pedestrian's position.

    Returns:
        float: Collision probability between 0 and 1.
    """
    # Unpack the ego vehicle dimensions
    ego_length, ego_width = ego_dimensions

    # Rotation matrix to align the ego vehicle's coordinate frame with the global frame
    cos_theta = np.cos(-ego_orientation)  # Negative sign for inverse rotation
    sin_theta = np.sin(-ego_orientation)
    rotation_matrix = np.array([[cos_theta, -sin_theta],
                                [sin_theta,  cos_theta]])

    # Transform the pedestrian's mean position into the ego vehicle's coordinate frame
    relative_mean = obstacle_mean - ego_pos
    transformed_mean = rotation_matrix @ relative_mean

    # Transform the pedestrian's covariance matrix into the ego vehicle's coordinate frame
    transformed_cov = rotation_matrix @ obstacle_cov @ rotation_matrix.T

    # Define the collision zone (ego vehicle's bounding box in its own coordinate frame)
    half_length = ego_length / 2.0
    half_width = ego_width / 2.0
    lower = np.array([-half_length, -half_width])
    upper = np.array([half_length, half_width])

    # Compute the cumulative distribution function (CDF) at the four corners of the rectangle
    cdf_upper_upper = multivariate_normal.cdf(upper, mean=transformed_mean, cov=transformed_cov)
    cdf_lower_upper = multivariate_normal.cdf([lower[0], upper[1]], mean=transformed_mean, cov=transformed_cov)
    cdf_upper_lower = multivariate_normal.cdf([upper[0], lower[1]], mean=transformed_mean, cov=transformed_cov)
    cdf_lower_lower = multivariate_normal.cdf(lower, mean=transformed_mean, cov=transformed_cov)

    # Apply the inclusion-exclusion principle to compute the probability over the rectangle
    collision_probability = (
        cdf_upper_upper
        - cdf_lower_upper
        - cdf_upper_lower
        + cdf_lower_lower
    )

    # Ensure the probability is within [0, 1]
    if collision_probability < 0.0 or collision_probability > 1.0:
        if collision_probability > 1.0:
            print(f"Warning: Collision probability {collision_probability} is out of bounds. Correcting it.")
        # Clip the probability to the valid range
        collision_probability = max(min(collision_probability, 1.0), 0.0)

    return np.round(collision_probability, 4)


def monte_carlo_collision_probability(ego_pos, ego_orientation, ego_dimensions, obstacle_mean, obstacle_cov,
                                      num_samples=10000):
    """
    Computes the collision probability using Monte Carlo sampling (without considering obstacle size).
    Treats the obstacle as a point with positional uncertainty.

    Args:
        ego_pos (np.ndarray): [x, y] position of the ego vehicle.
        ego_orientation (float): Orientation angle (theta) of the ego vehicle in radians.
        ego_dimensions (tuple): (length, width) dimensions of the ego vehicle.
        obstacle_mean (np.ndarray): [x, y] expected position of the pedestrian's center.
        obstacle_cov (np.ndarray): 2x2 covariance matrix of the pedestrian's position.
        num_samples (int): Number of Monte Carlo samples to generate.
        plot (bool): Whether to plot the pedestrian's position and samples.
        zorder (int): Z-order for plotting the pedestrian's position and samples.

    Returns:
        float: Estimated collision probability using Monte Carlo sampling.
    """
    # Unpack dimensions
    ego_length, ego_width = ego_dimensions

    # Rotation matrix to align the ego vehicle's coordinate frame with the global frame
    cos_theta = np.cos(-ego_orientation)  # Negative sign for inverse rotation
    sin_theta = np.sin(-ego_orientation)
    rotation_matrix = np.array([[cos_theta, -sin_theta],
                                [sin_theta,  cos_theta]])

    # Define the ego vehicle's bounding box in its own coordinate frame (no obstacle size)
    half_length = ego_length / 2.0
    half_width = ego_width / 2.0
    lower_bound = np.array([-half_length, -half_width])
    upper_bound = np.array([half_length, half_width])

    # Generate samples from the pedestrian's multivariate normal distribution
    samples = np.random.multivariate_normal(mean=obstacle_mean, cov=obstacle_cov, size=num_samples)

    # Transform each sample to the ego vehicle's coordinate frame
    relative_samples = samples - ego_pos
    transformed_samples = (rotation_matrix @ relative_samples.T).T

    # Count how many samples fall inside the ego vehicle's bounding box
    inside_box = np.all((transformed_samples >= lower_bound) & (transformed_samples <= upper_bound), axis=1)
    num_inside_box = np.sum(inside_box)

    world_samples_inside = (rotation_matrix.T @ transformed_samples[inside_box].T).T + ego_pos

    # Estimate collision probability as the proportion of samples inside the box
    collision_probability = num_inside_box / num_samples

    return collision_probability


def shift_to_vehicle_center(ego_pos, ego_orientation, wheelbase_rear_axle):
    """
    Shifts the reference point of the ego vehicle from the rear axle to the center of the vehicle.

    Args:
        ego_pos (np.ndarray): [x, y] position of the ego vehicle's rear axle (from trajectory).
        ego_orientation (float): Orientation angle (theta) of the ego vehicle in radians.
        wheelbase_rear_axle (float): Distance from the rear axle to the center of the vehicle (wheelbase length).

    Returns:
        np.ndarray: The [x, y] position of the ego vehicle's center.
    """
    # Compute the forward shift along the vehicle's orientation direction (i.e., x-axis in the ego vehicle's local frame)
    dx = wheelbase_rear_axle * np.cos(ego_orientation)
    dy = wheelbase_rear_axle * np.sin(ego_orientation)

    # Shift the ego vehicle's position from rear axle to center
    center_pos = np.array([ego_pos[0] + dx, ego_pos[1] + dy])

    return center_pos


def ensure_positive_semi_definite(cov_matrix):
    """
    Ensure the covariance matrix is positive semi-definite.
    If it's not, adjust by adding a small value to the diagonal.

    Args:
        cov_matrix (np.ndarray): Covariance matrix to check and adjust.

    Returns:
        np.ndarray: Adjusted covariance matrix that is positive semi-definite.
    """
    try:
        # Attempt Cholesky decomposition
        np.linalg.cholesky(cov_matrix)
    except np.linalg.LinAlgError:
        # If decomposition fails, add small value to diagonal
        cov_matrix += np.eye(cov_matrix.shape[0]) * 1e-6
    return cov_matrix
