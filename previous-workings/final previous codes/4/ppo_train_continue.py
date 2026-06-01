import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from unet import build_unet
from environment import AirSimDroneEnv

# =========================
# Load U-Net (same as before)
# =========================
sess = tf.Session()
input_img = tf.placeholder(tf.float32, [None, 256,256,3])
unet_output = build_unet(input_img)

sess.run(tf.global_variables_initializer())

# =========================
# Create environment
# =========================
env = AirSimDroneEnv(None, sess, input_img, unet_output)

# =========================
# Load existing checkpoint
# =========================
model = PPO.load("./models/ppo/ppo_drone_300000_steps", env=env)

print("✅ Loaded model from 300k steps. Resuming training...")

# =========================
# Continue saving checkpoints
# =========================
checkpoint_callback = CheckpointCallback(
    save_freq=100000,
    save_path="./models/ppo/",
    name_prefix="ppo_drone"
)

# =========================
# Continue training
# =========================
# Train for additional 700k → total = 1M
model.learn(
    total_timesteps=700000,
    callback=checkpoint_callback,
    reset_num_timesteps=False   # ⭐ IMPORTANT
)

# =========================
# Save final model
# =========================
model.save("ppo_drone_model_1M")

print("🎉 Training completed up to 1,000,000 steps!")