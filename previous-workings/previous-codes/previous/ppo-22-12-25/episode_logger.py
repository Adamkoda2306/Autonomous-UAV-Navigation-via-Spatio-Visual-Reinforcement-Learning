from stable_baselines3.common.callbacks import BaseCallback


class EpisodeLoggerCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episode_reward = 0.0
        self.episode_count = 0

    def _on_step(self) -> bool:
        self.episode_reward += self.locals["rewards"][0]

        done = self.locals["dones"][0]
        info = self.locals["infos"][0]

        if done:
            self.episode_count += 1
            reason = info.get("done_reason", "unknown")

            print(
                f"Episode {self.episode_count:03d} | "
                f"Total Reward: {self.episode_reward:8.2f} | "
                f"Result: {reason.upper()}"
            )

            self.episode_reward = 0.0

        return True
