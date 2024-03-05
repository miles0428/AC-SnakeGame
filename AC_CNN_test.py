import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import snake_game_for_AC_CNN as SnakeGame
import matplotlib.pyplot as plt
from typing import Union
from AC_CNN import  A2C

LR_ACTOR = 1e-3                 # learning rate for actor network
LR_CRITIC = 1e-4                # learning rate for critic network
GAMMA = 0.9                     # reward discount
EPSILON_BASE = 0.85             # ϵ-greedy base
EPSILON = EPSILON_BASE          # ϵ-greedy
TARGET_REPLACE_ITER =  12       # target update frequency
N_ACTIONS = 4                   # number of actions
N_SPACES = 12                   # number of states
HIDDEN_SIZE = 128               # number of neurons in hidden layer
WIDTH = 240                     # width of the game
GRID_SIZE = 14                  # size of the grid
GAME_NUM = 30000                # number of games
BATCH_SIZE = 100                # batch size
MEMORY_ITER = 1                 # memory iteration


#set the device to gpu if available
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print ("MPS device found.")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print ("CUDA device found.")
else:
    device = torch.device("cpu")
    print ("No GPU found.")



path_actor_net = 'actor_net_cnn_last.pth'
path_critic_net = 'critic_net_cnn_last.pth'

# if you want to train a new model, set new_model = True
new_model = False

if not new_model:
    a2c = A2C()
    a2c.actor_net.load_state_dict(torch.load(path_actor_net))
    a2c.critic_net.load_state_dict(torch.load(path_critic_net))
    a2c.target_net.load_state_dict(torch.load(path_critic_net))
else:
    a2c = A2C()

#initialize the game
game = SnakeGame.SnakeGame(speed=50,mode='machine',w=WIDTH)

rewards = []
scores = []

#test 100 games
for i in range(1000):
    game.reset()
    s = game.state().to(device)
    ep_r = 0
    time_not_die = 0
    while True:
        time_not_die += 1
        a_value = a2c.choose_action(s)     # 选择动作
        count = 0
        while True:
            if count > 10:
                a_value = torch.rand_like(a_value)
            a = torch.multinomial(a_value, 1).item()
            count += 1
            if game.action(a):
                break
        done,score , r = game.play_step()
        s_ = game.state().to(device)
        ep_r += r
        if done:
            break
        s = s_
    rewards.append(ep_r)
    scores.append(score)
    print(f'Ep: {i : 2.2f} | Ep_r: {ep_r:1.2f}', end = '\r')

#print the average reward and score
print(f'average reward: {np.mean(rewards)}')
print(f'average score: {np.mean(scores)}')

#print the best reward and score
print(f'best reward: {np.max(rewards)}')
print(f'best score: {np.max(scores)}')

#print the worst reward and score
print(f'worst reward: {np.min(rewards)}')
print(f'worst score: {np.min(scores)}')

#plot a histogram of the rewards
plt.hist(rewards)
plt.savefig('rewards_cnn.png')
plt.clf()
plt.hist(scores)
plt.savefig('scores_cnn.png')
plt.clf()
print('done')


