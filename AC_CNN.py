import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import snake_game_for_AC_CNN as SnakeGame
import matplotlib.pyplot as plt
from typing import Union

LR_ACTOR = 1e-3                 # learning rate for actor network
LR_CRITIC = 1e-4                # learning rate for critic network
GAMMA = 0.9                     # reward discount
EPSILON_BASE = 0.95             # ϵ-greedy base
EPSILON = EPSILON_BASE          # ϵ-greedy
EPSILON_AMPLITUTE = 0.05
TARGET_REPLACE_ITER =  12       # target update frequency
N_ACTIONS = 4                   # number of actions
N_SPACES = 12                   # number of states
HIDDEN_SIZE = 128               # number of neurons in hidden layer
WIDTH = 240                     # width of the game
GRID_SIZE = 14                  # size of the grid
GAME_NUM = 10000                # number of games
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


def init_weights(m:torch.nn.Module) -> None:
    """
    initialize the weights of the network
    
    arg:
        m : the network

    return:
        None
    """
    if isinstance(m, nn.Linear) :
        nn.init.normal_(m.weight, mean = 0, std = 0.1)

class MyCNN(nn.Module):
    def __init__(self):
        super(MyCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 18, 3, padding=1)
        self.fc1 = nn.Linear(18 * (((GRID_SIZE)//2)//2) * (((GRID_SIZE)//2)//2), 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 128)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 18 * (((GRID_SIZE)//2)//2) * (((GRID_SIZE)//2)//2))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 策略网络
class Actor(nn.Module) :
    def __init__(self):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(128, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, N_ACTIONS) 
        )
        self.cnn = MyCNN()

    def forward(self, s):
        s = self.cnn(s)
        s = s.view(s.size(0), -1)
        output = self.net(s)
        output = F.softmax(output, dim = -1) 
        return output

# 价值网络
class Critic(nn.Module) :
    def __init__(self):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(128, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, 1) 
        )
        self.cnn = MyCNN()

    def forward(self, s):
        s = self.cnn(s)
        s = s.view(s.size(0), -1)
        output = self.net(s)
        return output

# A2C 的主体函数
class A2C :
    def __init__(self):
        self.actor_net, self.critic_net, self.target_net = Actor().apply(init_weights), Critic().apply(init_weights), Critic().apply(init_weights)
        self.actor_net.to(device)
        self.critic_net.to(device)
        self.target_net.to(device)
        self.learn_step_counter = 0 
        self.optimizer_actor = optim.Adam(self.actor_net.parameters(), lr = LR_ACTOR)    # 策略网络优化器
        self.optimizer_critic = optim.Adam(list(self.critic_net.parameters()), lr=LR_CRITIC)
        self.criterion_critic = nn.MSELoss() 

    def choose_action(self, s):
        grid = s
        if len(s.shape) == 1:
            s = torch.tensor(s, dtype=torch.float32).unsqueeze(0)
        action_value = self.actor_net(grid)
        if np.random.uniform() < EPSILON :
            pass
        else :
            action_value = torch.rand_like(action_value)

        return action_value

    def learn(self, 
              s : torch.Tensor, 
              a : torch.Tensor, 
              r : torch.Tensor, 
              s_: torch.Tensor) -> Union[torch.Tensor, torch.Tensor]:
        """
        learn from the game following the A2C algorithm

        arg:
            s   : the state of the game
            a   : the action operate by the agent
            r   : the reward from the game by the action
            s_  : the next state of the game
        
        return:
            actor_loss 
            critic_loss
        """
        if self.learn_step_counter % (TARGET_REPLACE_ITER * BATCH_SIZE) == 0 :          # 更新目标网络
            self.target_net.load_state_dict(self.critic_net.state_dict())

        self.learn_step_counter += 1
        grid = s
        grid_ = s_

        r = torch.tensor(r, dtype=torch.float32).view(-1,1).to(device)

        q_actor = self.actor_net(grid)                  # calculate the actor value
        q_critic = self.critic_net(grid)                # calculate the critic value
        q_next = self.target_net(grid_).detach()        # calculate the target value
        q_target = r + GAMMA * q_next                   # update the TD target
        td_error = (q_critic - q_target).detach()       # TD error

        if type(a) == int:
            a = torch.tensor([[a]], dtype=torch.int64).to(device)
        else:
            a = a.view(-1,1).to(device)

        # update the critic network
        loss_critic = self.criterion_critic(q_critic, q_target)
        self.optimizer_critic.zero_grad()
        loss_critic.backward()
        self.optimizer_critic.step()

        # update the actor network
        log_q_actor = torch.log(q_actor+1e-8)           # avoid log(0)
        actor_loss = torch.mean(log_q_actor.gather(1,a) * td_error)
        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        self.optimizer_actor.step()

        return actor_loss, loss_critic


if __name__ == '__main__':
    path_actor_net = 'actor_net_cnn_score.pth'
    path_critic_net = 'critic_net_cnn_score.pth'

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
    actor_loss_l        = []
    critic_loss_l       = []
    reward_l            = []
    tensor_action       = torch.zeros((BATCH_SIZE*MEMORY_ITER), dtype=torch.int64).to(device)
    tensor_reward       = torch.zeros((BATCH_SIZE*MEMORY_ITER)).to(device)
    tensor_done         = torch.zeros((BATCH_SIZE*MEMORY_ITER)).to(device)
    tensor_state        = torch.zeros((BATCH_SIZE*MEMORY_ITER,3,GRID_SIZE,GRID_SIZE)).to(device)
    tensor_next_state   = torch.zeros((BATCH_SIZE*MEMORY_ITER,3,GRID_SIZE,GRID_SIZE)).to(device)
    best_score          = 0
    best_reward         = -1e20

    for epoch in range(GAME_NUM) :
        game.reset()
        s = game.state().to(device)
        ep_r = 0
        time_not_die = 0
        # print("reset")
        while True :
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
            # print(r)

            #add to memory replace the memory in a2c.learn_step_counter
            tensor_action[a2c.learn_step_counter % (BATCH_SIZE)] = a
            tensor_reward[a2c.learn_step_counter % (BATCH_SIZE)] = r
            tensor_done[a2c.learn_step_counter % (BATCH_SIZE)] = done
            tensor_state[a2c.learn_step_counter % (BATCH_SIZE)] = s
            tensor_next_state[a2c.learn_step_counter % (BATCH_SIZE)] = s_

            
            # 学习
            if a2c.learn_step_counter % BATCH_SIZE == 0 and a2c.learn_step_counter !=0 and BATCH_SIZE != 1:
                actor_loss, critic_loss =a2c.learn(tensor_state, tensor_action, tensor_reward, tensor_next_state)
                actor_loss_l.append(actor_loss)
                critic_loss_l.append(critic_loss)
                EPSILON = EPSILON_BASE - EPSILON_AMPLITUTE* np.cos(a2c.learn_step_counter/BATCH_SIZE * np.pi /100)
                # EPSILON = 0.8 
            else:
                # actor_loss, critic_loss =a2c.learn(s, a, r, s_)
                a2c.learn_step_counter += 1
                pass
            if done :
                break

            s = s_
            
        reward_l.append(ep_r)
        if score > best_score:
            best_score = score
            #save the model
            torch.save(a2c.actor_net.state_dict(), 'actor_net_cnn_score.pth')
            torch.save(a2c.critic_net.state_dict(), 'critic_net_cnn_score.pth')
        if ep_r > best_reward:
            best_reward = ep_r
            #save the model
            torch.save(a2c.actor_net.state_dict(), 'actor_net_cnn_reward.pth')
            torch.save(a2c.critic_net.state_dict(), 'critic_net_cnn_reward.pth')
        # print(ep_r)
        print(f'Ep: {epoch : 2.2f} | Ep_r: {ep_r:1.2f} |EPSILON:{EPSILON :.2f}', end = '\r')
    torch.save(a2c.actor_net.state_dict(), 'actor_net_cnn_last.pth')
    torch.save(a2c.critic_net.state_dict(), 'critic_net_cnn_last.pth')

    critic_loss_l = [i.item() for i in critic_loss_l]
    actor_loss_l = [i.item() for i in actor_loss_l]
    plt.plot(critic_loss_l, label = 'crtic_loss')
    plt.legend()
    plt.savefig('crtic_loss_cnn_2.png')
    plt.clf()
    plt.plot(actor_loss_l, label = 'actor_loss')
    plt.legend()
    plt.savefig('actor_loss_cnn_2.png')
    plt.clf()
    plt.plot(reward_l, label = 'reward')
    plt.legend()
    plt.savefig('reward_cnn_2.png')
    plt.clf()