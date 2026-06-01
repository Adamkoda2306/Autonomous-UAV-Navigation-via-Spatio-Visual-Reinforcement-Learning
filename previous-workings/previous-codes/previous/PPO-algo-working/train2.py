from stable_baselines3 import PPO
from env2 import AirSimAutoNavEnv
from episode_logger import EpisodeLoggerCallback

env = AirSimAutoNavEnv()

model = PPO(
    policy="MlpPolicy",
    env=env,
    learning_rate=3e-4,
    gamma=0.99,
    n_steps=2048,
    batch_size=64,
    verbose=0,
    device="cpu"
)

EPISODES = 100
STEPS_PER_EPISODE = 500

callback = EpisodeLoggerCallback()

model.learn(
    total_timesteps=EPISODES * STEPS_PER_EPISODE,
    callback=callback
)

model.save("airsim_autonav_ppo")
print("✅ Training finished successfully")
