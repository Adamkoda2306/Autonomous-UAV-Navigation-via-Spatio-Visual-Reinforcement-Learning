import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from unet import build_unet
from env import AirSimDroneEnv

# =========================
# Load U-Net (same as training)
# =========================
sess = tf.Session()
input_img = tf.placeholder(tf.float32, [None, 256,256,3])
unet_output = build_unet(input_img)

sess.run(tf.global_variables_initializer())

# =========================
# Create Environment
# =========================
env = AirSimDroneEnv(None, sess, input_img, unet_output)
env = Monitor(env)

# =========================
# Load trained PPO model
# =========================
model = PPO.load("./present-working-codes/models/ppo/ppo_drone_800000_steps", env=env)
# or final model:
# model = PPO.load("ppo_drone_model", env=env)

# =========================
# Testing parameters
# =========================
NUM_EPISODES = 1
all_rewards = []

# =========================
# Run episodes
# =========================
for ep in range(NUM_EPISODES):
    obs, _ = env.reset()
    done = False
    total_reward = 0
    step_count = 0

    print(f"\n🚀 Episode {ep+1} started")

    while not done:
        # Deterministic = no exploration
        action, _ = model.predict(obs, deterministic=True)

        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        total_reward += reward
        step_count += 1

    print(f"✅ Episode {ep+1} finished")
    print(f"   Steps: {step_count}")
    print(f"   Total Reward: {total_reward:.2f}")

    all_rewards.append(total_reward)

# =========================
# Final Results
# =========================
print("\n========================")
print("📊 TEST RESULTS")
print("========================")
# print(f"Average Reward: {np.mean(all_rewards):.2f}")
# print(f"Max Reward: {np.max(all_rewards):.2f}")
# print(f"Min Reward: {np.min(all_rewards):.2f}")
print(f"Reward: {np.mean(all_rewards):.2f}")