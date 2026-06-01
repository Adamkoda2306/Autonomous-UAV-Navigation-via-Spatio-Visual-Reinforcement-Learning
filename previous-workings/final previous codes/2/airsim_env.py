import gymnasium as gym
from gymnasium import spaces
import numpy as np
import airsim
import cv2
import time
from config import *

class AirSimDroneEnv(gym.Env):

    def __init__(self):

        super().__init__()

        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()

        self.client.enableApiControl(True)
        self.client.armDisarm(True)

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(3,IMAGE_SIZE,IMAGE_SIZE),
            dtype=np.float32
        )

        self.action_space = spaces.Box(
            low=-1,
            high=1,
            shape=(3,),
            dtype=np.float32
        )

    def reset(self,seed=None,options=None):

        self.client.reset()

        self.client.enableApiControl(True)
        self.client.armDisarm(True)

        self.client.takeoffAsync().join()

        self.prev_dist = self.get_distance()

        img = self.get_image()

        return img, {}

    def get_image(self):

        responses = self.client.simGetImages([
            airsim.ImageRequest("0",airsim.ImageType.Scene,False,False)
        ])

        img1d = np.frombuffer(responses[0].image_data_uint8,dtype=np.uint8)

        img = img1d.reshape(
            responses[0].height,
            responses[0].width,
            3
        )

        img = cv2.resize(img,(IMAGE_SIZE,IMAGE_SIZE))

        img = img.transpose(2,0,1)

        return img.astype(np.float32)

    def get_position(self):

        state = self.client.getMultirotorState()

        pos = state.kinematics_estimated.position

        return np.array([pos.x_val,pos.y_val,pos.z_val])

    def get_distance(self):

        pos = self.get_position()

        goal = np.array(GOAL)

        return np.linalg.norm(pos-goal)

    def step(self,action):

        vx = float(action[0])*3
        vy = float(action[1])*3
        vz = float(action[2])*0.3

        self.client.moveByVelocityAsync(vx,vy,vz,0.3).join()

        time.sleep(0.2)

        img = self.get_image()

        dist = self.get_distance()

        reward = self.prev_dist - dist

        self.prev_dist = dist

        collision = self.client.simGetCollisionInfo().has_collided

        terminated = False
        truncated = False

        if collision:
            reward -= 100
            terminated = True

        if dist < 2:
            reward += 200
            terminated = True

        return img,reward,terminated,truncated,{}