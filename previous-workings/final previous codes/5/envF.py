# AirSim RL Environment
# Full Updated Version
# Realistic drone control + full reward shaping

import gymnasium as gym
from gymnasium import spaces
import airsim
import numpy as np
import time
import cv2
import math

# ---------------------------------------------------
# TARGET POSITION
# ---------------------------------------------------
# TARGET_POS = [117.51, -131.96, -0.15]
TARGET_POS = [80.00, -97.22, -0.16]

# ---------------------------------------------------
# SETTINGS
# ---------------------------------------------------
MAX_HEIGHT = 10.0
ALPHA = 0.05              # smoothing factor
MAX_SPEED = 5.0
MAX_YAW_RATE = 5.0


class AirSimDroneEnv(gym.Env):

    def __init__(self, unet_model, sess, input_img, unet_output):
        super(AirSimDroneEnv, self).__init__()

        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()

        self.sess = sess
        self.input_img = input_img
        self.unet_output = unet_output

        # ---------------------------------------------------
        # ACTION SPACE
        # [forward/backward, left/right, yaw rotate]
        # ---------------------------------------------------
        self.action_space = spaces.Box(
            low=np.array([-1, -1, -1], dtype=np.float32),
            high=np.array([1, 1, 1], dtype=np.float32),
            dtype=np.float32
        )

        # ---------------------------------------------------
        # OBSERVATION SPACE
        # 25 pooled map + dx dy dz
        # ---------------------------------------------------
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(28,),
            dtype=np.float32
        )

        self.current_vel = np.zeros(2, dtype=np.float32)
        self.current_yaw = 0.0
        self.prev_dist = None
        self.step_count = 0
        self.max_steps = 350

    # =====================================================
    # RESET
    # =====================================================
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.client.takeoffAsync().join()

        time.sleep(1)

        self.current_vel[:] = 0
        self.current_yaw = 0.0
        self.step_count = 0
        self.prev_dist = self._compute_dist()

        obs, _ = self.get_obs()
        return obs, {}

    # =====================================================
    # GET OBSERVATION
    # =====================================================
    def get_obs(self):

        while True:
            responses = self.client.simGetImages([
                airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)
            ])

            if responses and responses[0].height > 0:
                break

            time.sleep(0.05)

        img1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
        img = img1d.reshape(
            responses[0].height,
            responses[0].width,
            3
        )

        img = cv2.resize(img, (256, 256))

        # U-Net output
        seg_map = self.sess.run(
            self.unet_output,
            feed_dict={self.input_img: [img / 255.0]}
        )[0]

        # 5x5 pooled obstacle map
        pooled = cv2.resize(
            seg_map,
            (5, 5),
            interpolation=cv2.INTER_AREA
        ).flatten()

        # Drone position
        pos = self.client.getMultirotorState().kinematics_estimated.position

        rel = np.array([
            TARGET_POS[0] - pos.x_val,
            TARGET_POS[1] - pos.y_val,
            TARGET_POS[2] - pos.z_val
        ], dtype=np.float32)

        obs = np.concatenate([
            pooled.astype(np.float32),
            rel
        ])

        return obs, pooled

    # =====================================================
    # DISTANCE TO TARGET
    # =====================================================
    def _compute_dist(self):
        pos = self.client.getMultirotorState().kinematics_estimated.position

        dx = TARGET_POS[0] - pos.x_val
        dy = TARGET_POS[1] - pos.y_val

        return math.sqrt(dx * dx + dy * dy)

    # =====================================================
    # STEP
    # =====================================================
    def step(self, action):
        self.step_count += 1

        # ------------------------------------------
        # ACTIONS
        # ------------------------------------------
        forward = float(action[0]) * MAX_SPEED
        side = float(action[1]) * MAX_SPEED
        yaw_cmd = float(action[2]) * MAX_YAW_RATE

        target_vel = np.array([forward, side], dtype=np.float32)

        # ------------------------------------------
        # SMOOTH MOVEMENT
        # ------------------------------------------
        self.current_vel = (
            (1 - ALPHA) * self.current_vel +
            ALPHA * target_vel
        )

        # ------------------------------------------
        # UPDATE YAW
        # ------------------------------------------
        self.current_yaw += yaw_cmd * 0.1
        self.current_yaw = (self.current_yaw + 180) % 360 - 180

        # ------------------------------------------
        # ALTITUDE HOLD
        # ------------------------------------------
        pos = self.client.getMultirotorState().kinematics_estimated.position

        vz = -1.0 * (pos.z_val + MAX_HEIGHT)

        # ------------------------------------------
        # REALISTIC BODY FRAME CONTROL
        # ------------------------------------------
        self.client.moveByVelocityBodyFrameAsync(
            vx=float(self.current_vel[0]),
            vy=float(self.current_vel[1]),
            vz=float(vz),
            duration=0.2,
            yaw_mode=airsim.YawMode(True, self.current_yaw)
        )

        terminated = False
        truncated = False

        # ------------------------------------------
        # DISTANCE / PROGRESS
        # ------------------------------------------
        dist = self._compute_dist()

        progress = self.prev_dist - dist
        self.prev_dist = dist

        # ------------------------------------------
        # COLLISION
        # ------------------------------------------
        collision = self.client.simGetCollisionInfo()

        # ------------------------------------------
        # OBSERVATION
        # ------------------------------------------
        obs, pooled = self.get_obs()

        # =====================================================
        # REWARD FUNCTION
        # =====================================================
        reward = 0.0

        # Collision
        if collision.has_collided:
            reward = -100.0
            terminated = True

        # Reached target
        elif dist < 1.0:
            reward = 2000.0
            terminated = True
            print("🎯 Target reached!")
        
        else:
        
            # --------------------------------------
            # 1. Progress reward
            # --------------------------------------
            reward += progress * (20 + 100/(dist+1))
            reward -= dist * 0.05

            # milestone rewards
            if dist < 50:
                reward += 50
            if dist < 20:
                reward += 100
            if dist < 10:
                reward += 300

            # --------------------------------------
            # FIX 1: Target-facing reward
            # --------------------------------------
            pos = self.client.getMultirotorState().kinematics_estimated.position
            dx = TARGET_POS[0] - pos.x_val
            dy = TARGET_POS[1] - pos.y_val
            target_angle = math.degrees(math.atan2(dy, dx))
            yaw_error = abs((target_angle - self.current_yaw + 180) % 360 - 180)
            reward += 0.2 * (1.0 - yaw_error / 180.0)

            # --------------------------------------
            # FIX 3: Stronger spin penalty
            # --------------------------------------
            reward -= abs(yaw_cmd) * 0.4

            # --------------------------------------
            # FIX 4: Reward forward movement
            # --------------------------------------
            if forward > 0:
                reward += forward * 0.3

            # --------------------------------------
            # Inverse distance reward
            # --------------------------------------
            reward += 1.0 / (dist + 1.0)

            # --------------------------------------
            # Jerk penalty
            # --------------------------------------
            jerk_penalty = np.linalg.norm(target_vel - self.current_vel)
            reward -= 0.05 * jerk_penalty
            
            # --------------------------------------
            # Reward actual movement speed
            # --------------------------------------
            move_mag = np.linalg.norm(self.current_vel)
            reward += move_mag * 0.4
            if np.linalg.norm(self.current_vel) < 0.3 and abs(yaw_cmd) > 2:
                reward -= 2.0

            # --------------------------------------
            # Obstacle penalties
            # --------------------------------------
            seg2d = pooled.reshape(5, 5)
            front_obs = np.mean(seg2d[0:2, 2:4])
            global_obs = np.mean(seg2d)
            reward -= 4.0 * front_obs
            reward -= 1.0 * global_obs

            # Encourage side movement near obstacle
            if front_obs > 0.3:
                reward += abs(side)

            # --------------------------------------
            # FIX 5: Idle penalty
            # --------------------------------------
            if abs(progress) < 0.01:
                reward -= 0.5
            
            reward -= 0.02

        # Timeout penalty
        if (not terminated) and self.step_count >= self.max_steps:
            reward -= 100
            truncated = True

        info = {}

        return obs, reward, terminated, truncated, info