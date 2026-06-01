"""
environment.py
--------------
AirSim UAV Navigation Environment (AirSim v5.0 / Unreal Engine NH Suburban)

Implements the environment loop described in Section 3 of the paper:
  - Captures monocular 256×256×3 RGB from AirSim FPV camera
  - Runs U-Net inference to produce dense obstacle probability map
  - Compresses to 5×5 occupancy matrix via Spatial Area Pooling
  - Constructs 28-dim RL state [V_visual || dx || dy || dz]
  - Executes continuous body-frame velocity commands via
      moveByVelocityBodyFrameAsync() (AirSim v5 API)
  - Constant-altitude hold via altitude_correction()
  - Returns (state, reward, done, info) at each step
  - Supports reset(), step(), close()

AirSim NED frame convention:
  - X: North (forward), Y: East (right), Z: Down (negative = up)
  - Altitude hold at H_TARGET = 10 m   -> NED z ≈ -10.0
"""

import time
import math
import numpy as np
import cv2
import torch

# AirSim Python client (pip install airsim)
import airsim

from unet import load_unet
from utils import (
    spatial_area_pooling,
    build_state,
    exponential_smoothing,
    scale_actions,
    altitude_correction,
    compute_reward,
    horizontal_distance,
    target_yaw_deg,
    H_TARGET,
    ALPHA,
)


# ─────────────────────────────────────────────────────────────────────────────
# Default environment hyper-parameters
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_GOAL      = np.array([80.0, 40.0, -H_TARGET], dtype=np.float32)
SUCCESS_RADIUS    = 1.0     # metres  (eq. 28 terminal condition)
MAX_STEPS         = 350     # timeout truncation  (eq. 28)
STEP_DURATION     = 0.1     # seconds per control step
IMAGE_SHAPE       = (256, 256, 3)
CAMERA_NAME       = "0"     # AirSim front-facing camera
VEHICLE_NAME      = ""      # default multirotor


class UAVNavEnv:
    """
    OpenAI Gym-style environment wrapping AirSim for UAV navigation.

    Observation space : np.ndarray (28,)  float32
    Action space      : np.ndarray (3,)   float32  in [-1, 1]
                        [a_forward, a_lateral, a_yaw]
    """

    def __init__(
        self,
        goal: np.ndarray = DEFAULT_GOAL,
        unet_weights: str = None,
        alpha: float = ALPHA,
        device: str = "cpu",
        verbose: bool = True,
    ):
        self.goal    = goal.copy()
        self.alpha   = alpha
        self.device  = device
        self.verbose = verbose

        # Connect to AirSim
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        if self.verbose:
            print("[Env] Connected to AirSim.")

        # Load U-Net
        self.unet = load_unet(unet_weights, device=device)
        if self.verbose:
            print("[Env] U-Net loaded.")

        # Internal state
        self._v_smooth  = np.zeros(3, dtype=np.float32)
        self._step_count = 0
        self._dist_prev  = 0.0

        # Image request (uncompressed BGR)
        self._img_request = [
            airsim.ImageRequest(
                CAMERA_NAME,
                airsim.ImageType.Scene,
                pixels_as_float=False,
                compress=False,
            )
        ]

    # ──────────────────────────────────────────────────────────────────────
    # reset()
    # ──────────────────────────────────────────────────────────────────────
    def reset(self) -> np.ndarray:
        """
        Reset the AirSim environment.
        Returns initial 28-dim state.
        """
        self.client.reset()
        self.client.enableApiControl(True, VEHICLE_NAME)
        self.client.armDisarm(True, VEHICLE_NAME)

        # Take off and hover at target altitude
        self.client.takeoffAsync(vehicle_name=VEHICLE_NAME).join()
        self.client.moveToZAsync(-H_TARGET, velocity=3.0,
                                  vehicle_name=VEHICLE_NAME).join()
        time.sleep(1.0)

        self._v_smooth   = np.zeros(3, dtype=np.float32)
        self._step_count = 0
        pos = self._get_position()
        self._dist_prev  = horizontal_distance(pos, self.goal)

        state = self._get_state()
        return state

    # ──────────────────────────────────────────────────────────────────────
    # step()
    # ──────────────────────────────────────────────────────────────────────
    def step(self, action: np.ndarray):
        """
        Execute one control step.

        Args:
            action: (3,) float32 in [-1, 1]  [a_forward, a_lateral, a_yaw]

        Returns:
            next_state (28,), reward (float), done (bool), info (dict)
        """
        self._step_count += 1

        # 1. Scale action -> physical commands  (eq. 10-12)
        v_forward, v_lateral, omega_yaw = scale_actions(action)
        v_target = np.array([v_forward, v_lateral, omega_yaw], dtype=np.float32)

        # 2. Exponential velocity smoothing  (eq. 23)
        self._v_smooth = exponential_smoothing(self._v_smooth, v_target, self.alpha)
        vf_smooth, vl_smooth, yw_smooth = (
            self._v_smooth[0], self._v_smooth[1], self._v_smooth[2]
        )

        # 3. Altitude correction  (eq. 13)
        pos = self._get_position()
        vz  = altitude_correction(pos[2])

        # 4. Send velocity command to AirSim  (moveByVelocityBodyFrameAsync)
        self.client.moveByVelocityBodyFrameAsync(
            vx=vf_smooth,
            vy=vl_smooth,
            vz=vz,
            duration=STEP_DURATION,
            yaw_mode=airsim.YawMode(
                is_rate=True,
                yaw_or_rate=yw_smooth
            ),
            vehicle_name=VEHICLE_NAME,
        ).join()

        # 5. Get next state
        next_state = self._get_state()

        # 6. Compute reward
        pos_new   = self._get_position()
        dist_curr = horizontal_distance(pos_new, self.goal)

        collision = self.client.simGetCollisionInfo(
            vehicle_name=VEHICLE_NAME
        ).has_collided

        success = dist_curr < SUCCESS_RADIUS
        timeout = self._step_count >= MAX_STEPS

        G         = self._get_occupancy_matrix()
        theta_tgt = target_yaw_deg(pos_new[:2], self.goal[:2])
        kinematics = self.client.getMultirotorState(
            vehicle_name=VEHICLE_NAME
        ).kinematics_estimated
        # yaw from AirSim quaternion
        _, _, yaw = airsim.to_eularian_angles(
            kinematics.orientation
        )
        theta_yaw_deg = math.degrees(yaw)

        reward, breakdown, done = compute_reward(
            dist_prev=self._dist_prev,
            dist_curr=dist_curr,
            G=G,
            v_forward=vf_smooth,
            v_lateral=vl_smooth,
            v_smooth=self._v_smooth,
            v_target=v_target,
            theta_target_deg=theta_tgt,
            theta_yaw_deg=theta_yaw_deg,
            omega_yaw=yw_smooth,
            collision=collision,
            success=success,
            timeout=timeout,
            step=self._step_count,
        )

        self._dist_prev = dist_curr

        info = {
            "step"        : self._step_count,
            "dist"        : dist_curr,
            "collision"   : collision,
            "success"     : success,
            "timeout"     : timeout,
            "pos"         : pos_new.tolist(),
            "v_smooth"    : self._v_smooth.tolist(),
            **breakdown,
        }

        return next_state, reward, done, info

    # ──────────────────────────────────────────────────────────────────────
    # close()
    # ──────────────────────────────────────────────────────────────────────
    def close(self):
        self.client.armDisarm(False, VEHICLE_NAME)
        self.client.enableApiControl(False, VEHICLE_NAME)
        if self.verbose:
            print("[Env] Connection closed.")

    # ──────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────
    def _get_position(self) -> np.ndarray:
        """Returns UAV NED position as (3,) float32 array."""
        state = self.client.getMultirotorState(vehicle_name=VEHICLE_NAME)
        p = state.kinematics_estimated.position
        return np.array([p.x_val, p.y_val, p.z_val], dtype=np.float32)

    def _get_rgb_image(self) -> np.ndarray:
        """
        Capture 256×256×3 uint8 RGB from AirSim FPV camera.
        Returns np.ndarray (256, 256, 3) float32 in [0,1].
        """
        responses = self.client.simGetImages(
            self._img_request, vehicle_name=VEHICLE_NAME
        )
        r = responses[0]
        img = np.frombuffer(r.image_data_uint8, dtype=np.uint8)
        img = img.reshape(r.height, r.width, 3)
        img = cv2.resize(img, (256, 256))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img.astype(np.float32) / 255.0

    def _get_prob_map(self) -> np.ndarray:
        """
        Run U-Net inference on current camera frame.
        Returns M_prob (256, 256) float32 in [0, 1].
        """
        img = self._get_rgb_image()                    # (256, 256, 3)
        tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)  # (1,3,256,256)
        tensor = tensor.to(self.device)
        with torch.no_grad():
            prob = self.unet(tensor)                   # (1, 1, 256, 256)
        return prob.squeeze().cpu().numpy()            # (256, 256)

    def _get_occupancy_matrix(self) -> np.ndarray:
        """Returns the 5×5 spatial occupancy matrix G."""
        prob_map = self._get_prob_map()
        return spatial_area_pooling(prob_map)

    def _get_state(self) -> np.ndarray:
        """
        Construct the full 28-dimensional navigation state.
        """
        G   = self._get_occupancy_matrix()
        pos = self._get_position()
        return build_state(G, pos, self.goal)

    @property
    def observation_space_dim(self) -> int:
        return 28

    @property
    def action_space_dim(self) -> int:
        return 3


# ─────────────────────────────────────────────────────────────────────────────
# Quick connectivity test (run without training)
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    env = UAVNavEnv(verbose=True)
    state = env.reset()
    print(f"[Env] Initial state shape: {state.shape}")  # (28,)
    print(f"[Env] Initial state: {state}")

    action = np.array([0.5, 0.0, 0.0], dtype=np.float32)  # forward
    next_state, reward, done, info = env.step(action)
    print(f"[Env] Step result: reward={reward:.4f}, done={done}")
    print(f"[Env] Info: {info}")
    env.close()