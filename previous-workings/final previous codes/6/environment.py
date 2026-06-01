import gymnasium as gym
from gymnasium import spaces
import airsim
import numpy as np
import time
import cv2
import math

TARGET_POS = [79.27, -58.94, -0.15]
MAX_HEIGHT = 10.0
ALPHA = 0.005  # smoothing factor for velocity prev = 0.2

class AirSimDroneEnv(gym.Env):
    def __init__(self, unet_model, sess, input_img, unet_output):
        super(AirSimDroneEnv, self).__init__()

        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()

        self.sess = sess
        self.input_img = input_img
        self.unet_output = unet_output

        # Smooth velocity state
        self.current_vel = np.zeros(3, dtype=np.float32)

        # Action space: continuous velocities
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)

        # Observation: U-Net pooled + relative position to target
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(27,), dtype=np.float32)

        self.prev_dist = None  # for progress reward

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.client.takeoffAsync().join()

        self.current_vel = np.zeros(3, dtype=np.float32)
        self.prev_dist = self._compute_dist()
        obs = self.get_obs()
        info = {}
        return obs, info

    def get_obs(self):

        while True:
            responses = self.client.simGetImages([
                airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)
            ])

            if responses and responses[0].height > 0 and responses[0].width > 0:
                break

            print("⚠️ Waiting for valid image...")
            time.sleep(0.05)

        img1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
        img = img1d.reshape(responses[0].height, responses[0].width, 3)

        # ✅ Resize to match U-Net input
        img = cv2.resize(img, (256, 256))

        seg_map = self.sess.run(
            self.unet_output,
            feed_dict={self.input_img: [img / 255.0]}
        )[0]

        pooled = cv2.resize(seg_map, (5, 5), interpolation=cv2.INTER_AREA).flatten()

        # Add relative position
        pos = self.client.getMultirotorState().kinematics_estimated.position
        rel_pos = np.array([
            TARGET_POS[0] - pos.x_val,
            TARGET_POS[1] - pos.y_val
        ], dtype=np.float32)

        obs = np.concatenate([pooled.astype(np.float32), rel_pos])
        return obs

    def _compute_dist(self):
        pos = self.client.getMultirotorState().kinematics_estimated.position
        dx = TARGET_POS[0] - pos.x_val
        dy = TARGET_POS[1] - pos.y_val
        return math.sqrt(dx**2 + dy**2)

    def step(self, action):
        # Convert to Python float and smooth velocity
        target_vel = action.astype(np.float64) * 5.0
        self.current_vel = (1 - ALPHA) * self.current_vel + ALPHA * target_vel

        pos = self.client.getMultirotorState().kinematics_estimated.position
        vx, vy = float(self.current_vel[0]), float(self.current_vel[1])
        z_err = float(-1.0 * (pos.z_val + MAX_HEIGHT))

        self.client.moveByVelocityAsync(vx, vy, z_err, 0.2) # 0.1 is changed to 0.2

        # Distance metrics
        dist = self._compute_dist()
        progress = self.prev_dist - dist if self.prev_dist is not None else 0
        self.prev_dist = dist

        # Collision info
        collision = self.client.simGetCollisionInfo()
        terminated = False
        truncated = False

        # Reward shaping
        if collision.has_collided:
            reward = -100.0
            terminated = True
        elif dist < 3.0:
            reward = 1000.0
            terminated = True
            print("🎯 Target reached!")
        else:
            # Reward for moving toward target + progress reward + proximity
            direction_reward = (vx*(TARGET_POS[0]-pos.x_val) + vy*(TARGET_POS[1]-pos.y_val)) / (dist+1e-6)
            reward = 0.2*direction_reward + 0.5*progress + 1.0/(dist+1.0)
            # added by me
            jerk_penalty = np.linalg.norm(target_vel - self.current_vel)
            reward -= 0.05 * jerk_penalty

        obs = self.get_obs()
        info = {}

        return obs, reward, terminated, truncated, info