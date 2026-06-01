import torch
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from models import UNet, Actor, Critic
from ppo import PPO
from airsim_env import AirSimNHEnv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs("heatmaps", exist_ok=True)

unet = UNet().to(device)
actor = Actor().to(device)
critic = Critic().to(device)
ppo = PPO(actor, critic)

env = AirSimNHEnv()
unet_optimizer = torch.optim.Adam(unet.parameters(), lr=1e-4)

episode_rewards = []
episode_lengths = []

def compress_to_grid(map_tensor):
    map_np = map_tensor.squeeze().detach().cpu().numpy()
    grid = cv2.resize(map_np, (5,5))
    return torch.tensor(grid.flatten(), dtype=torch.float32)

for episode in range(500):

    img = env.reset().to(device)
    states, actions, log_probs, rewards = [], [], [], []
    prev_gray = None

    for t in range(200):

        reward_map = unet(img)

        # Save heatmap
        if t % 10 == 0:
            heat_np = reward_map.squeeze().detach().cpu().numpy()
            heat_np = (heat_np - heat_np.min()) / (heat_np.max() - heat_np.min() + 1e-6)
            heat_np = (heat_np * 255).astype(np.uint8)
            heat_color = cv2.applyColorMap(heat_np, cv2.COLORMAP_JET)
            cv2.imwrite(f"heatmaps/ep{episode}_step{t}.png", heat_color)

        grid_state = compress_to_grid(reward_map).to(device)

        pos = env.get_position()
        goal_vec = env.goal_xy - pos[:2]
        goal_vec = goal_vec / (np.linalg.norm(goal_vec) + 1e-6)
        goal_vec = torch.tensor(goal_vec, dtype=torch.float32).to(device)

        state = torch.cat([grid_state, goal_vec]).unsqueeze(0)

        action_mean = actor(state)
        dist = torch.distributions.Normal(action_mean, 0.3)
        action = torch.clamp(dist.sample(), -1, 1)
        scaled_action = action * 4.0
        log_prob = dist.log_prob(action).sum(dim=1)

        next_img, reward, done = env.step(
            scaled_action.squeeze().cpu().numpy()
        )

        # Optical Flow Label
        curr_img_np = img.squeeze().permute(1,2,0).cpu().numpy()
        curr_gray = cv2.cvtColor((curr_img_np*255).astype(np.uint8), cv2.COLOR_BGR2GRAY)

        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, curr_gray,
                None, 0.5, 3, 15, 3, 5, 1.2, 0
            )

            vx = np.mean(flow[...,0])
            vy = np.mean(flow[...,1])
            motion_vector = -np.array([vx, vy])

            grid = np.zeros((5,5))
            center = np.array([2,2])
            direction = motion_vector / (np.linalg.norm(motion_vector)+1e-6)

            for i in range(5):
                for j in range(5):
                    vec = np.array([i,j]) - center
                    if np.dot(vec, direction) > 0:
                        grid[i,j] = reward

            label_map = cv2.resize(grid, (256,256),
                                   interpolation=cv2.INTER_NEAREST)

            label_map = torch.tensor(label_map,
                                     dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

            loss_unet = torch.nn.functional.mse_loss(
                reward_map, label_map
            )

            unet_optimizer.zero_grad()
            loss_unet.backward()
            unet_optimizer.step()

        prev_gray = curr_gray

        states.append(state.squeeze())
        actions.append(action.squeeze())
        log_probs.append(log_prob.squeeze())
        rewards.append(reward)

        img = next_img.to(device)

        if done:
            break

    states = torch.stack(states)
    actions = torch.stack(actions)
    log_probs = torch.stack(log_probs)

    ppo.update(states, actions, log_probs, rewards)

    total_reward = sum(rewards)
    episode_rewards.append(total_reward)
    episode_lengths.append(len(rewards))

    print(f"Episode {episode} | Reward: {total_reward:.2f}")

# Plot curves
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(episode_rewards)
plt.title("Episode Reward")

plt.subplot(1,2,2)
plt.plot(episode_lengths)
plt.title("Episode Length")

plt.savefig("training_curves.png")
plt.show()