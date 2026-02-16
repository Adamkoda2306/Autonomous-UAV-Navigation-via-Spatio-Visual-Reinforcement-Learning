from stable_baselines3 import PPO
from env import AirSimAutoNavEnv
from episode_logger import EpisodeLoggerCallback
from sb3_contrib import RecurrentPPO
from sb3_contrib.ppo_recurrent.policies import MlpLstmPolicy


env = AirSimAutoNavEnv()

model = RecurrentPPO(
    policy="MlpLstmPolicy",
    env=env,
    learning_rate=3e-4,
    gamma=0.99,
    gae_lambda=0.95,
    n_steps=128,
    batch_size=128,
    verbose=1,
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
