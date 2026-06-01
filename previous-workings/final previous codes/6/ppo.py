import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from stable_baselines3 import PPO
from unet import build_unet
from environment import AirSimDroneEnv
from stable_baselines3.common.callbacks import CheckpointCallback

# TF session for U-Net
sess = tf.Session()
input_img = tf.placeholder(tf.float32, [None, 256,256,3])
unet_output = build_unet(input_img)

sess.run(tf.global_variables_initializer())

# Create environment
env = AirSimDroneEnv(None, sess, input_img, unet_output)

# ✅ Callback: save every 10,000 timesteps
checkpoint_callback = CheckpointCallback(
    save_freq=100000,
    save_path="./models/ppo/",
    name_prefix="ppo_drone"
)

# PPO model
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log="./ppo_drone/"
)

# Training loop
TOTAL_TIMESTEPS = 500000

model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=checkpoint_callback)

# Save model
model.save("ppo_drone_model")