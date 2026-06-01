import torch
import numpy as np
from airsim_env import AirSimDroneEnv
from model import ActorCritic
from config import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = AirSimDroneEnv()

model = ActorCritic().to(device)

optimizer = torch.optim.Adam(model.parameters(),lr=3e-4)

gamma = 0.99

for episode in range(MAX_EPISODES):

    state,_ = env.reset()

    total_reward = 0

    for step in range(MAX_STEPS):

        img = torch.FloatTensor(state).unsqueeze(0).to(device)

        pos = env.get_position()

        goal = np.array(GOAL)

        goal_vec = goal-pos

        goal_tensor = torch.FloatTensor(goal_vec).unsqueeze(0).to(device)

        action,value = model(img,goal_tensor)

        action_np = action.detach().cpu().numpy()[0]

        next_state,reward,done,_,_ = env.step(action_np)

        next_img = torch.FloatTensor(next_state).unsqueeze(0).to(device)

        next_goal_vec = goal-env.get_position()

        next_goal = torch.FloatTensor(next_goal_vec).unsqueeze(0).to(device)

        _,next_value = model(next_img,next_goal)

        target = reward + gamma*next_value*(1-int(done))

        advantage = target-value

        actor_loss = -(advantage.detach()*action.mean())

        critic_loss = advantage.pow(2)

        loss = actor_loss+critic_loss

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        state = next_state

        total_reward += reward

        if done:
            break

    print("Episode:",episode,"Reward:",total_reward)

    if episode % SAVE_INTERVAL == 0:

        torch.save(model.state_dict(),"drone_acktr_model.pth")