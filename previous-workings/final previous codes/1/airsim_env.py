import airsim
import numpy as np
import cv2
import torch
import time

class AirSimNHEnv:
    def __init__(self):
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

        self.goal_xy = np.array([79.16, -89.30])

    def reset(self):
        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.client.takeoffAsync().join()
        time.sleep(1)
        return self.get_image()

    def get_position(self):
        state = self.client.getMultirotorState()
        pos = state.kinematics_estimated.position
        return np.array([pos.x_val, pos.y_val, pos.z_val])

    def get_image(self):
        response = self.client.simGetImages(
            [airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)]
        )[0]

        img = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
        img = img.reshape(response.height, response.width, 3)
        img = cv2.resize(img, (256, 256))
        img = img / 255.0
        img = torch.tensor(img).permute(2,0,1).float()
        return img.unsqueeze(0)

    def step(self, action):
        vx, vy, vz = action

        self.client.moveByVelocityAsync(
            float(vx), float(vy), float(vz), 1.0
        ).join()

        time.sleep(0.1)

        collision = self.client.simGetCollisionInfo().has_collided
        pos = self.get_position()

        distance = np.linalg.norm(self.goal_xy - pos[:2])

        reward = -0.01
        done = False

        if collision:
            reward = -50
            done = True

        reward += 5.0 / (distance + 1e-6)

        if distance < 3:
            reward += 100
            done = True

        return self.get_image(), reward, done