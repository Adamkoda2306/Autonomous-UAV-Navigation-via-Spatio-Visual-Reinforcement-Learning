import torch
import torch.nn as nn

class ActorCritic(nn.Module):

    def __init__(self,state_dim=6,action_dim=3):

        super().__init__()

        self.actor = nn.Sequential(

            nn.Linear(state_dim,128),
            nn.Tanh(),
            nn.Linear(128,64),
            nn.Tanh(),
            nn.Linear(64,action_dim),
            nn.Tanh()

        )

        self.critic = nn.Sequential(

            nn.Linear(state_dim,128),
            nn.Tanh(),
            nn.Linear(128,64),
            nn.Tanh(),
            nn.Linear(64,1)

        )

    def forward(self,x):

        action = self.actor(x)
        value = self.critic(x)

        return action,value