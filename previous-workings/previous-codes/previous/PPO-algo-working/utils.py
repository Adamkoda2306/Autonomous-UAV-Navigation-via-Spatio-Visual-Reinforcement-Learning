import math
import numpy as np

def angle_to_goal(position, goal):
    dx = goal[0] - position.x_val
    dy = goal[1] - position.y_val
    return math.atan2(dy, dx)

def distance_2d(position, goal):
    return np.linalg.norm(
        np.array([position.x_val, position.y_val]) - goal
    )
