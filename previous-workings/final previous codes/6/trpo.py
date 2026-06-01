import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from sb3_contrib import TRPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from unet import build_unet
from environment import AirSimDroneEnv

# TF session for U-Net
sess = tf.Session()
input_img = tf.placeholder(tf.float32, [None, 256,256,3])
unet_output = build_unet(input_img)

sess.run(tf.global_variables_initializer())

# Create environment
env = AirSimDroneEnv(None, sess, input_img, unet_output)
env = Monitor(env)

# ✅ Callback: save every 10,000 timesteps
checkpoint_callback = CheckpointCallback(
    save_freq=10000,
    save_path="./models/",
    name_prefix="trpo_drone"
)

# TRPO model
model = TRPO(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log="./trpo_drone/",
    learning_rate=1e-4,
    gamma=0.99,
    batch_size=1024
)

# Training
TOTAL_TIMESTEPS = 500000

model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=checkpoint_callback
)

# Final save
model.save("trpo_drone_model_final")