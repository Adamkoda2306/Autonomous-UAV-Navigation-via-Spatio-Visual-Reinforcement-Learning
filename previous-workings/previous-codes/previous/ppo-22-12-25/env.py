import gymnasium as gym
from gymnasium import spaces
import airsim
import numpy as np
import time
import math
from collections import deque
from utils import distance_2d


class AirSimAutoNavEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()

        # -------- AirSim Client --------
        self.client = airsim.CarClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)

        self.controls = airsim.CarControls()

        # -------- Goal --------
        # self.goal = np.array([78.98, -84.75])
        self.goal = np.array([72.15, -1.57])

        # -------- Action Space --------
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32
        )

        # -------- Observation Space --------
        self.observation_space = spaces.Box(
            low=np.array([
                0, -np.pi, 0, 0, 0, 0,
                -50, 0, 0, 0,
                0
            ], dtype=np.float32),
            high=np.array([
                300, np.pi, 30, 50, 50, 50,
                50, 1, 1, 1,
                1
            ], dtype=np.float32)
        )

        # -------- Episode Control --------
        self.prev_distance = None
        self.steps = 0
        self.max_steps = 1500
        self.last_done_reason = "running"

        # -------- Episodic Collision Memory --------
        self.collision_memory = []
        self.collision_radius = 6.0
        self.collision_penalty = -25.0

        # -------- Temporal Memory --------
        self.front_lidar_history = deque(maxlen=6)
        self.left_open_history = deque(maxlen=20)
        self.right_open_history = deque(maxlen=20)
        self.forward_motion_steps = 0

    # ---------------- RESET ----------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.client.reset()
        self.client.enableApiControl(True)
        time.sleep(0.5)

        self.prev_distance = self._distance_to_goal()
        self.steps = 0
        self.last_done_reason = "running"

        self.front_lidar_history.clear()
        self.left_open_history.clear()
        self.right_open_history.clear()
        self.forward_motion_steps = 0

        return self._get_observation(), {}

    # ---------------- STEP ----------------
    def step(self, action):
        self.steps += 1
        self._apply_action(action)
        time.sleep(0.1)

        obs = self._get_observation()
        reward, terminated = self._compute_reward()

        return obs, reward, terminated, False, {"done_reason": self.last_done_reason}

    # ---------------- OBSERVATION ----------------
    def _get_observation(self):
        car_state = self.client.getCarState()
        speed = car_state.speed

        dist = self._distance_to_goal()
        rel_angle = self._relative_angle_to_goal(car_state)
        lidar_front, lidar_left, lidar_right = self._process_lidar()
        corridor_width = self._estimate_corridor_width()

        # ---- Temporal updates ----
        self.front_lidar_history.append(lidar_front)
        self.left_open_history.append(1.0 if lidar_left > 8.0 else 0.0)
        self.right_open_history.append(1.0 if lidar_right > 8.0 else 0.0)

        if speed > 2.0:
            self.forward_motion_steps += 1
        else:
            self.forward_motion_steps = max(0, self.forward_motion_steps - 1)

        front_trend = (
            self.front_lidar_history[-1] - self.front_lidar_history[0]
            if len(self.front_lidar_history) >= 2 else 0.0
        )

        left_open_freq = np.mean(self.left_open_history) if self.left_open_history else 0.0
        right_open_freq = np.mean(self.right_open_history) if self.right_open_history else 0.0
        forward_time_norm = min(self.forward_motion_steps / 100.0, 1.0)

        return np.array([
            dist,
            rel_angle,
            speed,
            lidar_front,
            lidar_left,
            lidar_right,
            front_trend,
            left_open_freq,
            right_open_freq,
            forward_time_norm,
            corridor_width
        ], dtype=np.float32)

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

        if collision:
            pos = car_state.kinematics_estimated.position
            self.collision_memory.append(np.array([pos.x_val, pos.y_val]))
            self.last_done_reason = "collision"
            return -200.0, True

        if dist < 4.0:
            self.last_done_reason = "goal"
            return 500.0, True

        if self.steps >= self.max_steps:
            self.last_done_reason = "timeout"
            return -100.0, True

        reward = progress * 15.0
        reward -= abs(self._relative_angle_to_goal(car_state)) * 0.5
        reward -= 0.1

        if speed < 1.0:
            reward -= 1.0

        # Dead-end anticipation
        if (
            len(self.front_lidar_history) >= 5
            and self.front_lidar_history[-1] < 6.0
            and np.mean(self.left_open_history) < 0.2
            and np.mean(self.right_open_history) < 0.2
        ):
            reward -= 5.0

        reward += self._collision_memory_penalty(
            car_state.kinematics_estimated.position
        )

        self.last_done_reason = "running"
        return reward, False

    # ---------------- CAMERA: Corridor Width ----------------
    def _estimate_corridor_width(self):
        try:
            responses = self.client.simGetImages([
                airsim.ImageRequest(
                    "0",
                    airsim.ImageType.DepthPerspective,
                    pixels_as_float=True
                )
            ])
            depth = np.array(responses[0].image_data_float)
            if depth.size == 0:
                return 0.0

            depth = depth.reshape(responses[0].height, responses[0].width)

            center_row = depth[depth.shape[0] // 2]
            free_space = center_row > 5.0
            corridor_width = np.sum(free_space) / free_space.size
            return float(np.clip(corridor_width, 0.0, 1.0))

        except Exception:
            return 0.0

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

    # ---------------- MEMORY PENALTY ----------------
    def _collision_memory_penalty(self, pos):
        p = np.array([pos.x_val, pos.y_val])
        for c in self.collision_memory:
            if np.linalg.norm(p - c) < self.collision_radius:
                return self.collision_penalty
        return 0.0

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
