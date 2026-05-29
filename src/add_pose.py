import math
import numpy as np
import gtsam
from gtsam.symbol_shorthand import L, X

PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.05]))
ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.1]))
MEASUREMENT_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05, 0.1]))

def add_pose(graph, initial_estimate):

    odometry = gtsam.Pose2(np.sqrt(2), np.sqrt(2), np.pi / 2)

    # Add BetweenFactor from X(3) to X(4)
    graph.add(gtsam.BetweenFactorPose2(X(3), X(4), odometry, ODOMETRY_NOISE))

    # Insert X(4) directly at the expected global position
    pose_4_estimate = gtsam.Pose2(4.0 + math.sqrt(2), math.sqrt(2), math.pi / 2)
    initial_estimate.insert(X(4), pose_4_estimate)

    return graph, initial_estimate
    