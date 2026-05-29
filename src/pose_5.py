import numpy as np
from helperfunctions import add_pose_from_global, add_landmark_measurement_from_global
import gtsam
from gtsam.symbol_shorthand import L, X

PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.05]))
ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.1]))
MEASUREMENT_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05, 0.1]))

def add_pose(graph, initial_estimate, pose_5):
    pose_4 = initial_estimate.atPose2(X(4))
    graph, initial_estimate = add_pose_from_global(
        graph=graph,
        initial_estimate=initial_estimate,
        prev_key=X(4),
        new_key=X(5),
        prev_pose=pose_4,
        new_pose_global=pose_5,
        odom_noise=ODOMETRY_NOISE
    )
    return graph, initial_estimate

def add_landmark_measurement(graph, result, pose_5, landmark):
    landmark_point = result.atPoint2(L(landmark))
    graph = add_landmark_measurement_from_global(
        graph=graph,
        pose_key=X(5),
        pose=pose_5,
        landmark_key=L(landmark),
        landmark_point=landmark_point,
        measurement_noise=MEASUREMENT_NOISE
    )
    return graph

def optimize(graph, initial_estimate):
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate)
    result = optimizer.optimize()
    return result

def minimize_marginals(graph, initial_estimate, pose_options):
    best_pose = None
    best_landmark = None
    best_sum = float('inf')

    for pose_key, pose_5 in pose_options.items():
        for landmark in [1, 2]:
            g = graph.clone()
            est = gtsam.Values(initial_estimate)

            g, est = add_pose(g, est, pose_5)
            result = optimize(g, est)
            g = add_landmark_measurement(g, result, pose_5, landmark)
            result = optimize(g, est)

            marginals = gtsam.Marginals(g, result)
            total = (np.trace(marginals.marginalCovariance(L(1))) +
                     np.trace(marginals.marginalCovariance(L(2))))

            if total < best_sum:
                best_sum = total
                best_pose = pose_key
                best_landmark = landmark

    g = graph.clone()
    est = gtsam.Values(initial_estimate)
    pose_5 = pose_options[best_pose]
    g, est = add_pose(g, est, pose_5)
    result = optimize(g, est)
    g = add_landmark_measurement(g, result, pose_5, best_landmark)
    result = optimize(g, est)
    marginals = gtsam.Marginals(g, result)
    sum_of_marginals = (marginals.marginalCovariance(L(1)).sum() +
                        marginals.marginalCovariance(L(2)).sum())

    return best_pose, best_landmark, sum_of_marginals

def minimize_errors(graph, initial_estimate, pose_options):

    best_pose = "d"      # chosen pose option
    best_landmark = 1    # chosen landmark (1 or 2)
    pose_5 = pose_options[best_pose]
    if initial_estimate.exists(X(5)):
        initial_estimate.erase(X(5))
    if graph.exists(X(5)):
        graph.erase(X(5))
    graph, initial_estimate = add_pose(graph, initial_estimate, pose_5)
    result = optimize(graph, initial_estimate)
    graph = add_landmark_measurement(graph, result, pose_5, best_landmark)
    result = optimize(graph, initial_estimate)

    X1_old = [0.0, 0.0, 0.0]
    X2_old = [2.0, 0.0, 0.0]
    X3_old = [4.0, 0.0, 0.0]

    X1_new = result.atPose2(X(1))
    X2_new = result.atPose2(X(2))
    X3_new = result.atPose2(X(3))

    x1, y1, theta1 = X1_new.x(), X1_new.y(), X1_new.theta()
    x2, y2, theta2 = X2_new.x(), X2_new.y(), X2_new.theta()
    x3, y3, theta3 = X3_new.x(), X3_new.y(), X3_new.theta()

    X1_new = [x1, y1, theta1]
    X2_new = [x2, y2, theta2]
    X3_new = [x3, y3, theta3]

    error_X1 = np.linalg.norm(np.array(X1_new) - np.array(X1_old))
    error_X2 = np.linalg.norm(np.array(X2_new) - np.array(X2_old))
    error_X3 = np.linalg.norm(np.array(X3_new) - np.array(X3_old))

    sum_of_errors = error_X1 + error_X2 + error_X3

    return best_pose, best_landmark, sum_of_errors