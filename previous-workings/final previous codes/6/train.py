import warnings
warnings.filterwarnings("ignore")

import airsim
import numpy as np
import cv2
import tensorflow.compat.v1 as tf
import time
import math
import os

# Essential for compatibility with Keras 3 and older TF sessions
tf.disable_v2_behavior()

# --- Configuration & Hyperparameters ---
LEARNING_RATE = 1e-4 # [cite: 249]
MAX_HEIGHT = 10.0
# Your specific environment coordinates
TARGET_POS = [79.27, -58.94, -0.15] 
TOTAL_EPISODES = 5000
SAVE_INTERVAL = 500
ALPHA = 0.1 # Smoothing factor for continuous velocity changes [cite: 341]

class DroneTrainer:
    def __init__(self):
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        
        self.model_dir = "./drone_model/"
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        # Session Setup
        self.sess = tf.Session()
        self.input_img = tf.placeholder(tf.float32, [None, 256, 256, 3])
        self.unet_output = self.build_unet(self.input_img)
        
        self.actor_input = tf.placeholder(tf.float32, [None, 25])
        self.velocity_output = self.build_actor(self.actor_input)
        
        self.saver = tf.train.Saver(max_to_keep=10)
        self.sess.run(tf.global_variables_initializer())

    def build_unet(self, inputs):
        """U-Net architecture for visual representation [cite: 215, 246]"""
        # Encoder [cite: 213, 214]
        x = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
        p1 = tf.keras.layers.MaxPooling2D(2, 2)(x)
        x = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(p1)
        p2 = tf.keras.layers.MaxPooling2D(2, 2)(x)
        
        # Decoder [cite: 234, 239]
        u1 = tf.keras.layers.UpSampling2D(2)(p2)
        x = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(u1)
        u2 = tf.keras.layers.UpSampling2D(2)(x)
        # Output: 1D Reward Segmentation Map [cite: 247]
        return tf.keras.layers.Conv2D(1, 1, activation='sigmoid')(u2)

    def build_actor(self, inputs):
        """Actor Network for continuous velocity control [cite: 216, 278]"""
        x = tf.keras.layers.Dense(64, activation='tanh')(inputs)
        x = tf.keras.layers.Dense(64, activation='tanh')(x)
        # Output: Linear velocities for X, Y, Z [cite: 231, 340]
        return tf.keras.layers.Dense(3, activation='tanh')(x)

    def train(self):
        for ep in range(1, TOTAL_EPISODES + 1):
            self.client.reset()
            self.client.enableApiControl(True)
            self.client.armDisarm(True)
            self.client.takeoffAsync().join()
            
            total_reward = 0
            current_vel = np.array([0.0, 0.0, 0.0])
            done = False
            
            print(f"--- Starting Episode {ep} ---")
            
            while not done:
                # 1. Perception: Capture FPV Image [cite: 187]
                responses = self.client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])
                if not responses: continue
                img1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
                img = img1d.reshape(responses[0].height, responses[0].width, 3)
                
                # 2. State Information
                state = self.client.getMultirotorState().kinematics_estimated
                pos = state.position

                # 3. Calculate Target Direction (To fix the "Opposite Direction" issue)
                dx = TARGET_POS[0] - pos.x_val
                dy = TARGET_POS[1] - pos.y_val
                dist = math.sqrt(dx**2 + dy**2)
                
                # Unit vector toward goal
                target_dir = [dx/dist, dy/dist] if dist > 0 else [0, 0]

                # 4. U-Net Processing & Vectorization (5x5) [cite: 174, 259, 317]
                seg_map = self.sess.run(self.unet_output, feed_dict={self.input_img: [img/255.0]})[0]
                pooled = cv2.resize(seg_map, (5, 5), interpolation=cv2.INTER_AREA).flatten()
                
                # 5. Predict Action & Smooth Velocity [cite: 333, 341]
                nn_vel = self.sess.run(self.velocity_output, feed_dict={self.actor_input: [pooled]})[0]
                target_vel = nn_vel * 5.0 # Max 5m/s
                current_vel = (1 - ALPHA) * current_vel + ALPHA * target_vel
                
                # Height PD Control (Maintain 10m)
                z_err = -1.0 * (pos.z_val + MAX_HEIGHT)
                
                self.client.moveByVelocityAsync(float(current_vel[0]), float(current_vel[1]), float(z_err), 0.1)
                
                # 6. Reward Logic: Obstacle Avoidance + Goal Reaching [cite: 324, 325, 327, 460]
                collision = self.client.simGetCollisionInfo()
                
                if collision.has_collided:
                    reward = -100.0 # Heavy collision penalty [cite: 324]
                    done = True
                elif dist < 3.0:
                    reward = 1000.0 # Huge goal reach bonus
                    done = True
                    print(f"TARGET REACHED at Episode {ep}!")
                else:
                    # DIRECTIONAL REWARD (Dot Product)
                    # Positive reward for moving TOWARD target, negative for AWAY
                    dot_reward = (current_vel[0] * target_dir[0]) + (current_vel[1] * target_dir[1])
                    # Combined with proximity reward [cite: 460]
                    reward = dot_reward + (2.0 / (dist + 1.0))
                
                total_reward += reward

            print(f"Episode: {ep} | Total Reward: {total_reward:.4f}")

            # Save Model every 500 episodes [cite: 335]
            if ep % SAVE_INTERVAL == 0:
                self.saver.save(self.sess, f"{self.model_dir}model_ep_{ep}.ckpt")
                print(f"Checkpoint saved: Episode {ep}")

if __name__ == "__main__":
    trainer = DroneTrainer()
    trainer.train()