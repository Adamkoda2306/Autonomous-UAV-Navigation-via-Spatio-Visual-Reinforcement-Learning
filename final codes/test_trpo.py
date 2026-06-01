"""
test_trpo.py
------------
Evaluation script for the trained TRPO policy.

Runs deterministic rollouts inside AirSimNH and logs same
metrics as test_ppo.py for comparative analysis (Fig. 8).

Usage:
    python test_trpo.py --model checkpoints_trpo/trpo_best.pth --episodes 10
"""

import os
import argparse
import numpy as np
import torch
import json
from torch.distributions import Normal

from environment import UAVNavEnv
from train_ppo import ActorCritic
from utils import ALPHA


RESULTS_DIR = "results"


def evaluate_trpo(args):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    device = "cpu"
    env = UAVNavEnv(alpha=args.alpha, device=device, verbose=False)

    policy = ActorCritic()
    policy.load_state_dict(torch.load(args.model, map_location="cpu"))
    policy.eval()
    print(f"[TRPO-Test] Loaded model: {args.model}")

    all_episodes = []
    successes    = 0

    for ep in range(args.episodes):
        state = env.reset()
        done  = False
        step  = 0

        ep_data = {
            "episode"     : ep,
            "trajectory"  : [],
            "velocities"  : [],
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
                action       = mean.clamp(-1.0, 1.0).squeeze(0).numpy()

            next_state, reward, done, info = env.step(action)

            pos      = info.get("pos", [0, 0, 0])
            progress = step / 350.0
            ep_data["trajectory"].append(pos + [progress])

            v_sm = info.get("v_smooth", [0, 0, 0])
            ep_data["velocities"].append([
                v_sm[0], v_sm[1], v_sm[2],
                float(np.linalg.norm(v_sm[:2]))
            ])

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
        print(f"[TRPO-Test] Ep {ep+1:3d}/{args.episodes} | "
              f"Reward {total_reward:8.1f} | "
              f"Steps {step:4d} | "
              f"Success {'✓' if ep_data['success'] else '✗'}")

    success_rate = successes / args.episodes * 100
    print(f"\n[TRPO-Test] Success rate: {success_rate:.1f}% ({successes}/{args.episodes})")

    out_path = os.path.join(RESULTS_DIR, "trpo_test_results.json")
    with open(out_path, "w") as f:
        json.dump({"episodes": all_episodes, "success_rate": success_rate}, f, indent=2)
    print(f"[TRPO-Test] Results saved to {out_path}")

    env.close()
    return all_episodes


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",    type=str,   default="checkpoints_trpo/trpo_best.pth")
    parser.add_argument("--episodes", type=int,   default=10)
    parser.add_argument("--alpha",    type=float, default=0.01)
    args = parser.parse_args()
    evaluate_trpo(args)