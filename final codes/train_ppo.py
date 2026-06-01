"""
train_ppo.py
------------
Proximal Policy Optimization (PPO) training for map-less UAV navigation.

Matches Section 3.4 / 5 of the paper:
  - Actor-Critic neural network operating over 28-dim spatio-visual state
  - PPO clipped surrogate objective  (eq. 14) with ε = 0.2
  - Continuous action space A ∈ [-1, 1]^3
  - Trains for up to 1M environment interaction steps
  - Saves best model checkpoint and logs rewards / episode lengths

Usage:
    python train_ppo.py --timesteps 1000000 --alpha 0.01
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from collections import deque
import json
import time

from environment import UAVNavEnv
from utils import ALPHA


# ─────────────────────────────────────────────────────────────────────────────
# Hyper-parameters (matching paper)
# ─────────────────────────────────────────────────────────────────────────────
LR_ACTOR        = 3e-4
LR_CRITIC       = 1e-3
GAMMA           = 0.99
GAE_LAMBDA      = 0.95
CLIP_EPS        = 0.2          # ε in eq. 14
ENTROPY_COEF    = 0.01
VALUE_COEF      = 0.5
MAX_GRAD_NORM   = 0.5
N_STEPS         = 2048         # rollout buffer size
BATCH_SIZE      = 64
N_EPOCHS        = 10
HIDDEN_SIZE     = 256
STATE_DIM       = 28
ACTION_DIM      = 3
LOG_INTERVAL    = 10           # episodes
SAVE_INTERVAL   = 50000        # timesteps
CHECKPOINT_DIR  = "checkpoints_ppo"


# ─────────────────────────────────────────────────────────────────────────────
# Actor-Critic Network
# ─────────────────────────────────────────────────────────────────────────────
class ActorCritic(nn.Module):
    """
    Shared-trunk Actor-Critic for continuous action space.
    State: (28,) -> Actor: mean + log_std of Normal, Critic: scalar value
    """
    def __init__(self, state_dim=STATE_DIM, action_dim=ACTION_DIM,
                 hidden=HIDDEN_SIZE):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.actor_mean  = nn.Linear(hidden, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        self.critic = nn.Linear(hidden, 1)

        # Orthogonal initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)

    def forward(self, x):
        feat = self.shared(x)
        mean = torch.tanh(self.actor_mean(feat))   # bound action to [-1,1]
        std  = self.actor_log_std.exp().expand_as(mean)
        val  = self.critic(feat).squeeze(-1)
        return mean, std, val

    def get_action(self, state: np.ndarray, deterministic=False):
        """Sample or take mean action from current policy."""
        x = torch.FloatTensor(state).unsqueeze(0)
        mean, std, _ = self.forward(x)
        if deterministic:
            action = mean
        else:
            dist   = Normal(mean, std)
            action = dist.sample()
        action = action.clamp(-1.0, 1.0)
        return action.squeeze(0).detach().numpy()

    def evaluate(self, states, actions):
        mean, std, values = self.forward(states)
        dist       = Normal(mean, std)
        log_probs  = dist.log_prob(actions).sum(-1)
        entropy    = dist.entropy().sum(-1)
        return log_probs, values, entropy


# ─────────────────────────────────────────────────────────────────────────────
# Rollout Buffer
# ─────────────────────────────────────────────────────────────────────────────
class RolloutBuffer:
    def __init__(self, n_steps, state_dim, action_dim):
        self.n     = n_steps
        self.sd    = state_dim
        self.ad    = action_dim
        self.clear()

    def clear(self):
        self.states    = np.zeros((self.n, self.sd), dtype=np.float32)
        self.actions   = np.zeros((self.n, self.ad), dtype=np.float32)
        self.rewards   = np.zeros(self.n, dtype=np.float32)
        self.dones     = np.zeros(self.n, dtype=np.float32)
        self.log_probs = np.zeros(self.n, dtype=np.float32)
        self.values    = np.zeros(self.n, dtype=np.float32)
        self.ptr       = 0

    def add(self, state, action, reward, done, log_prob, value):
        i = self.ptr
        self.states[i]    = state
        self.actions[i]   = action
        self.rewards[i]   = reward
        self.dones[i]     = done
        self.log_probs[i] = log_prob
        self.values[i]    = value
        self.ptr += 1

    def full(self):
        return self.ptr >= self.n

    def compute_gae(self, last_value, gamma=GAMMA, lam=GAE_LAMBDA):
        """Generalized Advantage Estimation."""
        advantages = np.zeros_like(self.rewards)
        last_gae   = 0.0
        for t in reversed(range(self.n)):
            if t == self.n - 1:
                next_nonterminal = 1.0 - self.dones[t]
                next_value       = last_value
            else:
                next_nonterminal = 1.0 - self.dones[t]
                next_value       = self.values[t + 1]
            delta              = (self.rewards[t]
                                  + gamma * next_value * next_nonterminal
                                  - self.values[t])
            last_gae = delta + gamma * lam * next_nonterminal * last_gae
            advantages[t] = last_gae
        returns = advantages + self.values
        return advantages, returns

    def get_batches(self, advantages, returns):
        """Yield random mini-batches of size BATCH_SIZE."""
        indices = np.random.permutation(self.n)
        for start in range(0, self.n, BATCH_SIZE):
            idx = indices[start:start + BATCH_SIZE]
            yield (
                torch.FloatTensor(self.states[idx]),
                torch.FloatTensor(self.actions[idx]),
                torch.FloatTensor(advantages[idx]),
                torch.FloatTensor(returns[idx]),
                torch.FloatTensor(self.log_probs[idx]),
            )


# ─────────────────────────────────────────────────────────────────────────────
# PPO Update  (eq. 14)
# ─────────────────────────────────────────────────────────────────────────────
def ppo_update(policy, optimizer, buffer, last_value,
               clip_eps=CLIP_EPS, n_epochs=N_EPOCHS):
    advantages, returns = buffer.compute_gae(last_value)
    # Normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    policy_losses, value_losses, entropy_losses = [], [], []

    for _ in range(n_epochs):
        for states_b, actions_b, adv_b, ret_b, old_lp_b in buffer.get_batches(
                advantages, returns):
            log_probs, values, entropy = policy.evaluate(states_b, actions_b)

            # Ratio r_t(θ)  (eq. 14)
            ratio = torch.exp(log_probs - old_lp_b)

            # Clipped surrogate objective
            surr1 = ratio * adv_b
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv_b
            actor_loss  = -torch.min(surr1, surr2).mean()

            # Critic loss
            critic_loss = VALUE_COEF * nn.functional.mse_loss(values, ret_b)

            # Entropy bonus
            entropy_loss = -ENTROPY_COEF * entropy.mean()

            loss = actor_loss + critic_loss + entropy_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), MAX_GRAD_NORM)
            optimizer.step()

            policy_losses.append(actor_loss.item())
            value_losses.append(critic_loss.item())
            entropy_losses.append(entropy_loss.item())

    return (np.mean(policy_losses), np.mean(value_losses),
            np.mean(entropy_losses))


# ─────────────────────────────────────────────────────────────────────────────
# Training Loop
# ─────────────────────────────────────────────────────────────────────────────
def train(args):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    env = UAVNavEnv(alpha=args.alpha, device=device, verbose=True)
    policy    = ActorCritic().to("cpu")   # policy runs on CPU for AirSim compat
    optimizer = optim.Adam(policy.parameters(), lr=LR_ACTOR)
    buffer    = RolloutBuffer(N_STEPS, STATE_DIM, ACTION_DIM)

    # Logging
    ep_rewards  = []
    ep_lengths  = []
    all_rewards = []
    best_reward = -np.inf
    total_steps = 0
    episode     = 0
    ep_reward   = 0.0
    ep_len      = 0

    state = env.reset()

    print(f"[PPO] Starting training for {args.timesteps} steps | α={args.alpha}")
    start_time = time.time()

    while total_steps < args.timesteps:
        # ── Collect rollout ──
        buffer.clear()
        while not buffer.full():
            with torch.no_grad():
                st_t         = torch.FloatTensor(state).unsqueeze(0)
                mean, std, v = policy(st_t)
                dist         = Normal(mean, std)
                action_t     = dist.sample().clamp(-1.0, 1.0)
                log_prob     = dist.log_prob(action_t).sum(-1)
                value        = v.item()

            action = action_t.squeeze(0).numpy()
            next_state, reward, done, info = env.step(action)

            buffer.add(state, action, reward, float(done),
                       log_prob.item(), value)

            ep_reward   += reward
            ep_len      += 1
            total_steps += 1
            state        = next_state

            if done:
                episode += 1
                ep_rewards.append(ep_reward)
                ep_lengths.append(ep_len)
                all_rewards.append({
                    "episode"    : episode,
                    "timestep"   : total_steps,
                    "reward"     : ep_reward,
                    "ep_length"  : ep_len,
                    "success"    : info.get("success", False),
                })

                if episode % LOG_INTERVAL == 0:
                    mean_r = np.mean(ep_rewards[-LOG_INTERVAL:])
                    mean_l = np.mean(ep_lengths[-LOG_INTERVAL:])
                    elapsed = time.time() - start_time
                    print(f"[PPO] Ep {episode:5d} | Steps {total_steps:7d} | "
                          f"MeanR {mean_r:8.1f} | MeanLen {mean_l:5.0f} | "
                          f"Elapsed {elapsed:.0f}s")

                if ep_reward > best_reward:
                    best_reward = ep_reward
                    torch.save(policy.state_dict(),
                               f"{CHECKPOINT_DIR}/ppo_best.pth")

                ep_reward = 0.0
                ep_len    = 0
                state     = env.reset()

            # Periodic checkpoint
            if total_steps % SAVE_INTERVAL == 0:
                torch.save(policy.state_dict(),
                           f"{CHECKPOINT_DIR}/ppo_{total_steps}.pth")

        # ── PPO Update ──
        with torch.no_grad():
            _, _, last_v = policy(torch.FloatTensor(state).unsqueeze(0))
        pl, vl, el = ppo_update(policy, optimizer, buffer, last_v.item())

    # Final save
    torch.save(policy.state_dict(), f"{CHECKPOINT_DIR}/ppo_final.pth")
    with open(f"{CHECKPOINT_DIR}/ppo_rewards.json", "w") as f:
        json.dump(all_rewards, f, indent=2)

    print(f"[PPO] Training complete. Best reward: {best_reward:.1f}")
    env.close()
    return all_rewards


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int,   default=1_000_000,
                        help="Total training timesteps (default: 1M)")
    parser.add_argument("--alpha",     type=float, default=0.05,
                        help="Velocity smoothing coefficient α (default: 0.05)")
    parser.add_argument("--unet",      type=str,   default=None,
                        help="Path to pre-trained U-Net weights (.pth)")
    args = parser.parse_args()
    train(args)