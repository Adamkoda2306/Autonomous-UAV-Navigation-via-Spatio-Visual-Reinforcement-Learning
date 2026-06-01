import torch
import torch.nn as nn

class Critic(nn.Module):

    def __init__(self):

        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(25,64),
            nn.Tanh(),
            nn.Linear(64,1)
        )

    def forward(self,x):

        return self.fc(x)