import gymnasium as gym
from gymnasium import spaces
import airsim
import numpy as np
import time
from utils import angle_to_goal, distance_2d


class AirSimAutoNavEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()

        # -------- AirSim Client --------
        self.client = airsim.CarClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)

        self.controls = airsim.CarControls()

        # -------- Goal (NH safe point) --------
        self.goal = np.array([80.0, 20.0])

        # -------- Action Space --------
        # 0: straight | 1: left | 2: right | 3: brake
        self.action_space = spaces.Discrete(4)

        # -------- Observation Space --------
        # [distance, angle, speed, lidar_front, lidar_left, lidar_right]
        self.observation_space = spaces.Box(
            low=np.array([0, -np.pi, 0, 0, 0, 0], dtype=np.float32),
            high=np.array([300, np.pi, 30, 50, 50, 50], dtype=np.float32)
        )

        self.prev_distance = None

    # ---------------- RESET ----------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.client.reset()
        self.client.enableApiControl(True)
        time.sleep(0.5)

        self.prev_distance = self._distance_to_goal()
        obs = self._get_observation()

        return obs, {}

    # ---------------- STEP ----------------
    def step(self, action):
        self._apply_action(action)
        time.sleep(0.1)

        obs = self._get_observation()
        reward, terminated = self._compute_reward()

        truncated = False
        info = {}

        return obs, reward, terminated, truncated, info

    # ---------------- OBSERVATION ----------------
    def _get_observation(self):
        car_state = self.client.getCarState()
        pos = car_state.kinematics_estimated.position

        dist = self._distance_to_goal()
        angle = angle_to_goal(pos, self.goal)
        speed = car_state.speed

        lidar_front, lidar_left, lidar_right = self._process_lidar()

        return np.array(
            [dist, angle, speed, lidar_front, lidar_left, lidar_right],
            dtype=np.float32
        )

    # ---------------- ACTION ----------------
    def _apply_action(self, action):
        self.controls.throttle = 0.6
        self.controls.steering = 0.0
        self.controls.brake = 0.0

        if action == 1:
            self.controls.steering = -0.5
        elif action == 2:
            self.controls.steering = 0.5
        elif action == 3:
            self.controls.throttle = 0.0
            self.controls.brake = 1.0

        self.client.setCarControls(self.controls)

    # ---------------- REWARD ----------------
    def _compute_reward(self):
        collision = self.client.simGetCollisionInfo().has_collided
        dist = self._distance_to_goal()

        # Progress-based reward
        progress = (self.prev_distance - dist) * 5.0
        self.prev_distance = dist

        # Terminal conditions
        if collision:
            return -150.0, True

        if dist < 4.0:
            return 200.0, True

        # Small time penalty to encourage efficiency
        reward = progress - 0.05
        return reward, False

    # ---------------- LIDAR PROCESSING ----------------
    def _process_lidar(self):
        lidar = self.client.getLidarData(lidar_name="Lidar1")

        if len(lidar.point_cloud) < 3:
            return 50.0, 50.0, 50.0

        points = np.array(lidar.point_cloud, dtype=np.float32).reshape(-1, 3)
        distances = np.linalg.norm(points[:, :2], axis=1)

        left = distances[points[:, 1] > 1.0]
        center = distances[np.abs(points[:, 1]) <= 1.0]
        right = distances[points[:, 1] < -1.0]

        return (
            np.min(center) if len(center) else 50.0,
            np.min(left) if len(left) else 50.0,
            np.min(right) if len(right) else 50.0
        )

    # ---------------- DISTANCE ----------------
    def _distance_to_goal(self):
        pos = self.client.getCarState().kinematics_estimated.position
        return distance_2d(pos, self.goal)
