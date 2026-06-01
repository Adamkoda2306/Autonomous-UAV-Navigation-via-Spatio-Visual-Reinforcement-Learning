import torch
from airsim_env import AirSimDroneEnv
from model import ActorCritic
from config import *
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = AirSimDroneEnv()

model = ActorCritic().to(device)

model.load_state_dict(torch.load("drone_acktr_model.pth"))

model.eval()

state,_ = env.reset()

while True:

    img = torch.FloatTensor(state).unsqueeze(0).to(device)

    pos = env.get_position()

    goal_vec = np.array(GOAL)-pos

    goal = torch.FloatTensor(goal_vec).unsqueeze(0).to(device)

    action,_ = model(img,goal)

    action = action.detach().cpu().numpy()[0]

    state,_,done,_,_ = env.step(action)

    if done:

        state,_ = env.reset()