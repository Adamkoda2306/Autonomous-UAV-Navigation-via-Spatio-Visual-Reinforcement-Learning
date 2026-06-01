import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from stable_baselines3 import PPO
from unet import build_unet
from envF import AirSimDroneEnv
import time

# ----------------------------------------
# TF SESSION (same as training)
# ----------------------------------------
sess = tf.Session()
input_img = tf.placeholder(tf.float32, [None, 256, 256, 3])
unet_output = build_unet(input_img)

sess.run(tf.global_variables_initializer())

# ----------------------------------------
# CREATE ENVIRONMENT
# ----------------------------------------
env = AirSimDroneEnv(None, sess, input_img, unet_output)

# ----------------------------------------
# LOAD TRAINED MODEL
# ----------------------------------------
model = PPO.load("./models/ppo-full-control/ppo_drone_950000_steps")

# ----------------------------------------
# TESTING LOOP (5 EPISODES)
# ----------------------------------------
NUM_EPISODES = 5

for ep in range(NUM_EPISODES):
    obs, _ = env.reset()
    done = False
    truncated = False
    total_reward = 0

    print(f"\n🚀 Episode {ep+1} started")

    while not (done or truncated):

        # Predict action
        action, _ = model.predict(obs, deterministic=True)

        # Step environment
        obs, reward, done, truncated, _ = env.step(action)

        total_reward += reward

        # Small delay for visualization (optional)
        time.sleep(0.05)

    print(f"✅ Episode {ep+1} finished | Total Reward: {total_reward}")