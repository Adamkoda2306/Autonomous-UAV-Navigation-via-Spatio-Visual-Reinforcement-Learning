"""
test_ppo.py
-----------
Evaluation script for the trained PPO policy.

Runs deterministic rollouts inside AirSimNH and logs:
  - 3D trajectory (x, y, z positions)
  - Velocity profiles (v_forward, v_lateral, ω_yaw, v_smooth)
  - Per-step rewards and reward components
  - Episode success / failure statistics

Generates data consumed by results.py for Fig. 3-8 reproduction.

Usage:
    python test_ppo.py --model checkpoints_ppo/ppo_best.pth --episodes 10
"""

import os
import argparse
import numpy as np
import torch
import json
from torch.distributions import Normal

from environment import UAVNavEnv
from train_ppo import ActorCritic
from utils import ALPHA, scale_actions, exponential_smoothing


RESULTS_DIR = "results"


def evaluate_ppo(args):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    device = "cpu"
    env = UAVNavEnv(alpha=args.alpha, device=device, verbose=False)

    policy = ActorCritic()
    policy.load_state_dict(torch.load(args.model, map_location="cpu"))
    policy.eval()
    print(f"[PPO-Test] Loaded model: {args.model}")

    all_episodes = []
    successes    = 0

    for ep in range(args.episodes):
        state   = env.reset()
        done    = False
        step    = 0

        ep_data = {
            "episode"     : ep,
            "trajectory"  : [],    # list of [x, y, z, progress]
            "velocities"  : [],    # list of [vf, vl, yw, v_smooth_mag]
            "rewards"     : [],
            "reward_components": [],
            "success"     : False,
            "total_reward": 0.0,
            "ep_length"   : 0,
        }

        total_reward = 0.0

        while not done:
            with torch.no_grad():
                st_t         = torch.FloatTensor(state).unsqueeze(0)
                mean, std, _ = policy(st_t)
                # Deterministic: use mean action
                action = mean.clamp(-1.0, 1.0).squeeze(0).numpy()

            next_state, reward, done, info = env.step(action)

            # Log position
            pos = info.get("pos", [0, 0, 0])
            progress = step / 350.0   # normalised progression [0,1]
            ep_data["trajectory"].append(pos + [progress])

            # Log velocities
            v_sm = info.get("v_smooth", [0, 0, 0])
            ep_data["velocities"].append([
                v_sm[0], v_sm[1], v_sm[2],
                float(np.linalg.norm(v_sm[:2]))
            ])

            # Log rewards
            ep_data["rewards"].append(reward)
            ep_data["reward_components"].append({
                k: info.get(k, 0.0) for k in
                ["R_progress","R_yaw","R_motion","R_obs","R_jerk","R_terminal"]
            })

            total_reward += reward
            step         += 1
            state         = next_state

        ep_data["success"]      = info.get("success", False)
        ep_data["total_reward"] = total_reward
        ep_data["ep_length"]    = step

        if ep_data["success"]:
            successes += 1

        all_episodes.append(ep_data)
        print(f"[PPO-Test] Ep {ep+1:3d}/{args.episodes} | "
              f"Reward {total_reward:8.1f} | "
              f"Steps {step:4d} | "
              f"Success {'✓' if ep_data['success'] else '✗'}")

    success_rate = successes / args.episodes * 100
    print(f"\n[PPO-Test] Success rate: {success_rate:.1f}% ({successes}/{args.episodes})")

    # Save results
    out_path = os.path.join(RESULTS_DIR, "ppo_test_results.json")
    with open(out_path, "w") as f:
        json.dump({"episodes": all_episodes, "success_rate": success_rate}, f, indent=2)
    print(f"[PPO-Test] Results saved to {out_path}")

    env.close()
    return all_episodes


# ─────────────────────────────────────────────────────────────────────────────
# Velocity oscillation metric (used in Fig. 4)
# ─────────────────────────────────────────────────────────────────────────────
def compute_velocity_oscillation(velocities: list) -> float:
    """
    Mean absolute successive difference of forward velocity magnitude.
    Used as the 'Velocity Oscillation' metric in Table 2 / Fig. 4.
    """
    mags = [abs(v[0]) for v in velocities]
    if len(mags) < 2:
        return 0.0
    diffs = [abs(mags[i] - mags[i-1]) for i in range(1, len(mags))]
    return float(np.mean(diffs))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",    type=str,   default="checkpoints_ppo/ppo_best.pth")
    parser.add_argument("--episodes", type=int,   default=10)
    parser.add_argument("--alpha",    type=float, default=0.01)
    args = parser.parse_args()
    evaluate_ppo(args)