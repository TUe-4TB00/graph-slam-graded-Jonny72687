import math
import numpy as np
import gtsam
from gtsam.symbol_shorthand import L, X

PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.05]))
ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.1]))
MEASUREMENT_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05, 0.1]))

def add_landmark_measurement(graph, initial_estimate, result):
    # Get the optimized pose of X(4) and position of L(2)
    pose_4 = result.atPose2(X(4))
    landmark_2 = result.atPoint2(L(2))

    # Calculate the difference in global frame
    dx = landmark_2[0] - pose_4.x()
    dy = landmark_2[1] - pose_4.y()

    # Calculate range (distance)
    distance = math.sqrt(dx**2 + dy**2)

    # Calculate bearing (angle in global frame, then subtract robot heading)
    global_angle = math.atan2(dy, dx)
    bearing = global_angle - pose_4.theta()

    graph.add(gtsam.BearingRangeFactor2D(X(4), L(2), gtsam.Rot2(bearing), distance, MEASUREMENT_NOISE))
    return graph