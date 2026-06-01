import torch
import torch.nn.functional as F

class PPO:
    def __init__(self, actor, critic, lr=3e-4):
        self.actor = actor
        self.critic = critic
        self.optimizer = torch.optim.Adam(
            list(actor.parameters()) + list(critic.parameters()), lr=lr
        )
        self.clip = 0.2
        self.gamma = 0.99

    def compute_returns(self, rewards):
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        return torch.tensor(returns, dtype=torch.float32)

    def update(self, states, actions, old_log_probs, rewards):
        returns = self.compute_returns(rewards)
        values = self.critic(states).squeeze()

        advantages = returns - values.detach()

        action_mean = self.actor(states)
        dist = torch.distributions.Normal(action_mean, 0.3)
        new_log_probs = dist.log_prob(actions).sum(dim=1)

        ratio = (new_log_probs - old_log_probs).exp()

        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1-self.clip, 1+self.clip) * advantages

        actor_loss = -torch.min(surr1, surr2).mean()
        critic_loss = F.mse_loss(values, returns)

        loss = actor_loss + 0.5 * critic_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()