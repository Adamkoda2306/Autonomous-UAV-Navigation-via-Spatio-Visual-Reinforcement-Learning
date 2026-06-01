from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from env_drone import AirSimDroneAutoNavEnv()

env = AirSimDroneAutoNavEnv()

model = SAC(
    policy="MlpPolicy",
    env=env,
    learning_rate=3e-4,
    buffer_size=200_000,
    batch_size=256,
    gamma=0.99,
    tau=0.005,
    ent_coef="auto",
    train_freq=1,
    gradient_steps=1,
    verbose=1,
    device="cpu"
)

checkpoint = CheckpointCallback(
    save_freq=50_000,
    save_path="./checkpoints",
    name_prefix="sac_airsim"
)

model.learn(
    total_timesteps=600_000,
    callback=checkpoint
)

# Save model
model.save("airsim_autonav_drone_sac")
print("✅ SAC training finished")
