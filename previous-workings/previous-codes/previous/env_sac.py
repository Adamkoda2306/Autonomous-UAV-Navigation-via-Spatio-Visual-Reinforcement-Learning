import gymnasium as gym
from gymnasium import spaces
import airsim
import numpy as np
import time
import math
from utils import distance_2d


class AirSimAutoNavEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()

        # -------- AirSim --------
        self.client = airsim.CarClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.controls = airsim.CarControls()

        # -------- Goal (NED) --------
        self.goal = np.array([78.98, -84.75])

        # -------- Action Space (SAC friendly) --------
        # [steering, throttle]
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.2]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32
        )

        # -------- Observation Space --------
        # [distance, rel_angle, speed, lidar_front, lidar_left, lidar_right]
        self.observation_space = spaces.Box(
            low=np.array([0, -np.pi, 0, 0, 0, 0], dtype=np.float32),
            high=np.array([300, np.pi, 30, 50, 50, 50], dtype=np.float32)
        )

        self.prev_distance = None
        self.steps = 0
        self.max_steps = 1200

    # ---------------- RESET ----------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.client.reset()
        self.client.enableApiControl(True)
        time.sleep(0.5)

        self.prev_distance = self._distance_to_goal()
        self.steps = 0

        return self._get_observation(), {}

    # ---------------- STEP ----------------
    def step(self, action):
        self.steps += 1

        self._apply_action(action)
        time.sleep(0.1)

        obs = self._get_observation()
        reward, terminated = self._compute_reward()

        return obs, reward, terminated, False, {}

    # ---------------- OBSERVATION ----------------
    def _get_observation(self):
        car_state = self.client.getCarState()
        pos = car_state.kinematics_estimated.position
        speed = car_state.speed

        dist = distance_2d(pos, self.goal)
        rel_angle = self._relative_angle_to_goal(car_state)

        lidar_front, lidar_left, lidar_right = self._process_lidar()

        return np.array(
            [dist, rel_angle, speed, lidar_front, lidar_left, lidar_right],
            dtype=np.float32
        )

    # ---------------- ACTION ----------------
    def _apply_action(self, action):
        self.controls.steering = float(np.clip(action[0], -1.0, 1.0))
        self.controls.throttle = float(np.clip(action[1], 0.2, 1.0))
        self.controls.brake = 0.0
        self.client.setCarControls(self.controls)

    # ---------------- REWARD ----------------
    def _compute_reward(self):
        car_state = self.client.getCarState()
        collision = self.client.simGetCollisionInfo().has_collided
        speed = car_state.speed

        dist = self._distance_to_goal()
        progress = self.prev_distance - dist
        self.prev_distance = dist

        # ---- Collision ----
        if collision:
            return -300.0, True

        # ---- Goal ----
        if dist < 4.0:
            return 500.0, True

        # ---- Timeout ----
        if self.steps >= self.max_steps:
            return -50.0, True

        # ---- Reward shaping (SAC-friendly) ----
        reward = 20.0 * progress            # move toward goal
        reward -= abs(self._relative_angle_to_goal(car_state)) * 1.0
        reward -= 0.05                      # time penalty

        if speed < 1.0:
            reward -= 1.0

        return reward, False

    # ---------------- LIDAR ----------------
    def _process_lidar(self):
        lidar = self.client.getLidarData("Lidar1")
        if len(lidar.point_cloud) < 6:
            return 50.0, 50.0, 50.0

        points = np.array(lidar.point_cloud, dtype=np.float32).reshape(-1, 3)
        x, y = points[:, 0], points[:, 1]

        distances = np.sqrt(x**2 + y**2)
        angles = np.arctan2(y, x)

        front = distances[np.abs(angles) < np.deg2rad(15)]
        left = distances[(angles > np.deg2rad(15)) & (angles < np.deg2rad(60))]
        right = distances[(angles < -np.deg2rad(15)) & (angles > -np.deg2rad(60))]

        return (
            np.min(front) if len(front) else 50.0,
            np.min(left) if len(left) else 50.0,
            np.min(right) if len(right) else 50.0
        )

    # ---------------- HELPERS ----------------
    def _distance_to_goal(self):
        pos = self.client.getCarState().kinematics_estimated.position
        return distance_2d(pos, self.goal)

    def _relative_angle_to_goal(self, car_state):
        pos = car_state.kinematics_estimated.position
        dx = self.goal[0] - pos.x_val
        dy = self.goal[1] - pos.y_val
        goal_angle = math.atan2(dy, dx)
        yaw = airsim.to_eularian_angles(
            car_state.kinematics_estimated.orientation
        )[2]
        rel_angle = goal_angle - yaw
        return (rel_angle + math.pi) % (2 * math.pi) - math.pi
