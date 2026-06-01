from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.noise import NormalActionNoise
import numpy as np

from env_drone import AirSimDroneAutoNavEnv

# Create environment
env = AirSimDroneAutoNavEnv()

# Add action noise for exploration
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.2 * np.ones(n_actions))

# Create TD3 model
model = TD3(
    policy="MlpPolicy",
    env=env,
    learning_rate=1e-3,
    buffer_size=500_000,
    batch_size=256,
    tau=0.005,
    gamma=0.99,
    train_freq=(1, "step"),
    gradient_steps=1,
    action_noise=action_noise,
    verbose=1,
    device="cpu"
)

# Checkpoint callback
checkpoint = CheckpointCallback(
    save_freq=10_000,
    save_path="./checkpoints",
    name_prefix="td3_airsim"
)

# Train model
model.learn(
    total_timesteps=100_000,
    callback=checkpoint
)

# Save model
model.save("airsim_autonav_drone_td3")

print("✅ TD3 training finished")
