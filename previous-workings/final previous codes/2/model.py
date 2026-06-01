import torch
import torch.nn as nn
from config import *

class ActorCritic(nn.Module):

    def __init__(self):

        super().__init__()

        self.cnn = nn.Sequential(

            nn.Conv2d(3,32,8,4),
            nn.ReLU(),

            nn.Conv2d(32,64,4,2),
            nn.ReLU(),

            nn.Conv2d(64,64,3,1),
            nn.ReLU(),

            nn.Flatten()
        )

        self.fc = nn.Sequential(

            nn.Linear(3136+3,512),
            nn.ReLU()
        )

        self.actor = nn.Sequential(

            nn.Linear(512,3),
            nn.Tanh()
        )

        self.critic = nn.Sequential(

            nn.Linear(512,1)
        )

    def forward(self,img,goal):

        x = self.cnn(img)

        x = torch.cat((x,goal),dim=1)

        x = self.fc(x)

        action = self.actor(x)

        value = self.critic(x)

        return action,value