"""
train_trpo.py
-------------
Trust Region Policy Optimization (TRPO) for map-less UAV navigation.

Matches Section 3.4 / 5.4 of the paper:
  - KL-divergence trust-region constraint  (eq. 15):  D_KL(π_θ_old || π_θ) ≤ δ
  - Natural policy gradient via conjugate gradient + line search
  - Same Actor-Critic architecture as PPO
  - Continuous action space A ∈ [-1, 1]^3
  - Compared against PPO in Fig. 8 (navigation efficiency by episode length)

Note: TRPO uses a Fisher-vector-product conjugate gradient approach with
backtracking line search to satisfy the KL constraint δ.
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
import json
import time
import copy

from environment import UAVNavEnv
from train_ppo import ActorCritic, RolloutBuffer, STATE_DIM, ACTION_DIM
from utils import ALPHA


# ─────────────────────────────────────────────────────────────────────────────
# TRPO Hyper-parameters
# ─────────────────────────────────────────────────────────────────────────────
GAMMA           = 0.99
GAE_LAMBDA      = 0.95
MAX_KL          = 0.01          # δ in eq. 15  (trust region bound)
DAMPING         = 0.1           # conjugate gradient damping
CG_ITERS        = 10            # conjugate gradient iterations
BACKTRACK_ITERS = 10            # line search iterations
BACKTRACK_COEF  = 0.5
VALUE_LR        = 1e-3
VALUE_EPOCHS    = 5
N_STEPS         = 2048
BATCH_SIZE      = 64
HIDDEN_SIZE     = 256
LOG_INTERVAL    = 10
SAVE_INTERVAL   = 50000
CHECKPOINT_DIR  = "checkpoints_trpo"


# ─────────────────────────────────────────────────────────────────────────────
# Conjugate Gradient solver (for natural gradient)
# ─────────────────────────────────────────────────────────────────────────────
def flat_grad(loss, params, retain_graph=False, create_graph=False):
    grads = torch.autograd.grad(
        loss, params,
        retain_graph=retain_graph,
        create_graph=create_graph,
        allow_unused=True,
    )
    return torch.cat([
        g.view(-1) if g is not None else torch.zeros_like(p).view(-1)
        for g, p in zip(grads, params)
    ])


def get_flat_params(model):
    return torch.cat([p.data.view(-1) for p in model.parameters()])


def set_flat_params(model, flat_params):
    offset = 0
    for p in model.parameters():
        sz = p.numel()
        p.data.copy_(flat_params[offset:offset + sz].view_as(p))
        offset += sz


def fisher_vector_product(policy, states, actions, v, damping=DAMPING):
    """Compute (F + damping * I) * v using forward-mode differentiation."""
    kl = compute_kl(policy, policy, states, actions, detach_old=True)
    kl_grad = flat_grad(kl, list(policy.parameters()), create_graph=True)
    kl_grad_v = (kl_grad * v).sum()
    fvp = flat_grad(kl_grad_v, list(policy.parameters()), retain_graph=False)
    return fvp + damping * v


def compute_kl(policy_new, policy_old, states, actions, detach_old=True):
    """KL divergence D_KL(π_old || π_new) per eq. 15."""
    mean_new, std_new, _ = policy_new(states)
    with torch.no_grad() if detach_old else torch.enable_grad():
        mean_old, std_old, _ = policy_old(states)
    dist_new = Normal(mean_new, std_new)
    dist_old = Normal(mean_old.detach(), std_old.detach())
    kl = torch.distributions.kl_divergence(dist_old, dist_new).sum(-1)
    return kl.mean()


def conjugate_gradient(policy, states, actions, b, n_iters=CG_ITERS, tol=1e-10):
    """Solve Ax = b where A = Fisher info matrix."""
    x   = torch.zeros_like(b)
    r   = b.clone()
    p   = r.clone()
    rr  = r.dot(r)
    for _ in range(n_iters):
        Ap  = fisher_vector_product(policy, states, actions, p)
        alpha = rr / (p.dot(Ap) + 1e-8)
        x   = x + alpha * p
        r   = r - alpha * Ap
        rr_new = r.dot(r)
        if rr_new < tol:
            break
        p   = r + (rr_new / rr) * p
        rr  = rr_new
    return x


# ─────────────────────────────────────────────────────────────────────────────
# TRPO update (trust-region constrained natural gradient step)
# ─────────────────────────────────────────────────────────────────────────────
def trpo_update(policy, value_optimizer, buffer, last_value):
    """
    Perform one TRPO policy update with:
      1. Compute policy gradient
      2. Solve for natural gradient via conjugate gradient
      3. Backtracking line search to satisfy KL constraint δ
      4. Separate value function update via Adam
    """
    advantages, returns = buffer.compute_gae(last_value)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    states_t  = torch.FloatTensor(buffer.states)
    actions_t = torch.FloatTensor(buffer.actions)
    adv_t     = torch.FloatTensor(advantages)
    ret_t     = torch.FloatTensor(returns)

    # ── Policy gradient ──
    mean, std, _ = policy(states_t)
    dist         = Normal(mean, std)
    log_probs    = dist.log_prob(actions_t).sum(-1)

    # Old log probs (detached)
    with torch.no_grad():
        mean_old, std_old, _ = policy(states_t)
    old_log_probs = Normal(mean_old, std_old).log_prob(actions_t).sum(-1)

    ratio         = torch.exp(log_probs - old_log_probs.detach())
    surrogate     = (ratio * adv_t).mean()

    grad_g = flat_grad(surrogate, list(policy.parameters()), retain_graph=True)

    # ── Natural gradient via CG ──
    step_dir = conjugate_gradient(policy, states_t, actions_t, grad_g.detach())

    # Step size from trust-region constraint
    sHs = step_dir.dot(
        fisher_vector_product(policy, states_t, actions_t, step_dir)
    )
    step_size = torch.sqrt(2 * MAX_KL / (sHs + 1e-8))
    full_step = step_size * step_dir

    # ── Backtracking line search ──
    old_params = get_flat_params(policy).clone()
    old_policy = copy.deepcopy(policy)

    for i in range(BACKTRACK_ITERS):
        new_params = old_params + (BACKTRACK_COEF ** i) * full_step
        set_flat_params(policy, new_params)

        with torch.no_grad():
            mean_n, std_n, _ = policy(states_t)
        dist_n     = Normal(mean_n, std_n)
        lp_new     = dist_n.log_prob(actions_t).sum(-1)
        ratio_n    = torch.exp(lp_new - old_log_probs.detach())
        surr_new   = (ratio_n * adv_t).mean()

        kl_new     = compute_kl(policy, old_policy, states_t, actions_t)

        if surr_new > 0 and kl_new <= MAX_KL:
            break
        if i == BACKTRACK_ITERS - 1:
            set_flat_params(policy, old_params)   # revert if no improvement

    # ── Value function update ──
    v_losses = []
    for _ in range(VALUE_EPOCHS):
        idx = np.random.permutation(len(states_t))
        for start in range(0, len(states_t), BATCH_SIZE):
            b_idx = idx[start:start + BATCH_SIZE]
            _, _, vals = policy(states_t[b_idx])
            v_loss = nn.functional.mse_loss(vals, ret_t[b_idx])
            value_optimizer.zero_grad()
            v_loss.backward()
            value_optimizer.step()
            v_losses.append(v_loss.item())

    return np.mean(v_losses)


# ─────────────────────────────────────────────────────────────────────────────
# Training Loop
# ─────────────────────────────────────────────────────────────────────────────
def train(args):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    device = "cpu"

    env = UAVNavEnv(alpha=args.alpha, device=device, verbose=True)
    policy          = ActorCritic()
    value_optimizer = torch.optim.Adam(policy.critic.parameters(), lr=VALUE_LR)
    buffer          = RolloutBuffer(N_STEPS, STATE_DIM, ACTION_DIM)

    ep_rewards  = []
    ep_lengths  = []
    all_rewards = []
    best_reward = -np.inf
    total_steps = 0
    episode     = 0
    ep_reward   = 0.0
    ep_len      = 0

    state = env.reset()

    print(f"[TRPO] Starting training for {args.timesteps} steps | α={args.alpha} | δ={MAX_KL}")
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
                    "episode"   : episode,
                    "timestep"  : total_steps,
                    "reward"    : ep_reward,
                    "ep_length" : ep_len,
                    "success"   : info.get("success", False),
                })

                if episode % LOG_INTERVAL == 0:
                    mean_r = np.mean(ep_rewards[-LOG_INTERVAL:])
                    mean_l = np.mean(ep_lengths[-LOG_INTERVAL:])
                    elapsed = time.time() - start_time
                    print(f"[TRPO] Ep {episode:5d} | Steps {total_steps:7d} | "
                          f"MeanR {mean_r:8.1f} | MeanLen {mean_l:5.0f} | "
                          f"Elapsed {elapsed:.0f}s")

                if ep_reward > best_reward:
                    best_reward = ep_reward
                    torch.save(policy.state_dict(),
                               f"{CHECKPOINT_DIR}/trpo_best.pth")

                ep_reward = 0.0
                ep_len    = 0
                state     = env.reset()

            if total_steps % SAVE_INTERVAL == 0:
                torch.save(policy.state_dict(),
                           f"{CHECKPOINT_DIR}/trpo_{total_steps}.pth")

        # ── TRPO Update ──
        with torch.no_grad():
            _, _, last_v = policy(torch.FloatTensor(state).unsqueeze(0))
        trpo_update(policy, value_optimizer, buffer, last_v.item())

    torch.save(policy.state_dict(), f"{CHECKPOINT_DIR}/trpo_final.pth")
    with open(f"{CHECKPOINT_DIR}/trpo_rewards.json", "w") as f:
        json.dump(all_rewards, f, indent=2)

    print(f"[TRPO] Training complete. Best reward: {best_reward:.1f}")
    env.close()
    return all_rewards


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int,   default=1_000_000)
    parser.add_argument("--alpha",     type=float, default=0.05)
    parser.add_argument("--unet",      type=str,   default=None)
    args = parser.parse_args()
    train(args)