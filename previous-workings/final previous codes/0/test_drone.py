from stable_baselines3 import SAC
from env_drone import AirSimDroneAutoNavEnv
import time

# ---------------- Load Environment ----------------
env = AirSimDroneAutoNavEnv()

# ---------------- Load Trained Model ----------------
model = SAC.load("./checkpoints/sac_airsim_turned_left_200000_steps", env=env)

print("✅ Model loaded successfully. Starting test...")

# ---------------- Run One Episode ----------------
obs, _ = env.reset()
done = False

while not done:
    # Predict action using trained policy
    action, _states = model.predict(obs, deterministic=True)

    # Step the environment
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    print(f"Reward: {reward:.2f}")

    time.sleep(0.05)   # slow down for visualization

print("🛬 Test episode finished.")
