import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import snake_game_for_AC
import matplotlib.pyplot as plt

LR_ACTOR = 1e-3             # 策略网络的学习率
LR_CRITIC = 1e-3            # 价值网络的学习率
GAMMA = 0.9                 # 奖励的折扣因子
EPSILON = 0.9               # ϵ-greedy 策略的概率
TARGET_REPLACE_ITER =  100  # 目标网络更新的频率
N_ACTIONS = 3               # 动作数
N_SPACES = 12               # 状态数量
HIDDEN_SIZE = 40            # 隐藏层神经元数量


# 网络参数初始化，采用均值为 0，方差为 0.1 的高斯分布
def init_weights(m) :
    if isinstance(m, nn.Linear) :
        nn.init.normal_(m.weight, mean = 0, std = 0.1)

# 策略网络
class Actor(nn.Module) :
    def __init__(self):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(N_SPACES, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, N_ACTIONS) # 输出为各个动作的概率，维度为 3
        )

    def forward(self, s):
        output = self.net(s)
        output = F.softmax(output, dim = -1) # 概率归一化
        return output

# 价值网络
class Critic(nn.Module) :
    def __init__(self):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(N_SPACES, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, 1) # 输出值是对当前状态的打分，维度为 1
        )

    def forward(self, s):
        output = self.net(s)
        return output

# A2C 的主体函数
class A2C :
    def __init__(self):
        # 初始化策略网络，价值网络和目标网络。价值网络和目标网络使用同一个网络
        self.actor_net, self.critic_net, self.target_net = Actor().apply(init_weights), Critic().apply(init_weights), Critic().apply(init_weights)
        self.learn_step_counter = 0 # 学习步数
        self.optimizer_actor = optim.Adam(self.actor_net.parameters(), lr = LR_ACTOR)    # 策略网络优化器
        self.optimizer_critic = optim.Adam(self.critic_net.parameters(), lr = LR_CRITIC) # 价值网络优化器
        self.criterion_critic = nn.MSELoss() # 价值网络损失函数

    def choose_action(self, s):
        s = torch.unsqueeze(torch.FloatTensor(s), dim = 0) # 增加维度
        if np.random.uniform() < EPSILON :                 # ϵ-greedy 策略对动作进行采取
            action_value = self.actor_net(s)
            action = torch.multinomial(action_value, 1).item()
        else :
            action = np.random.randint(0, N_ACTIONS)

        return action

    def learn(self, s, a, r, s_):
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0 :          # 更新目标网络
            self.target_net.load_state_dict(self.critic_net.state_dict())

        self.learn_step_counter += 1

        s = torch.FloatTensor(s)
        s_ = torch.FloatTensor(s_)

        q_actor = self.actor_net(s)               # 策略网络
        q_critic = self.critic_net(s)             # 价值对当前状态进行打分
        q_next = self.target_net(s_).detach()     # 目标网络对下一个状态进行打分
        q_target = r + GAMMA * q_next             # 更新 TD 目标
        td_error = (q_critic - q_target).detach() # TD 误差

        # 更新价值网络
        loss_critic = self.criterion_critic(q_critic, q_target)
        self.optimizer_critic.zero_grad()
        loss_critic.backward()
        self.optimizer_critic.step()

        # 更新策略网络
        log_q_actor = torch.log(q_actor+1e-5) # 防止 log(0) 的出现
        actor_loss = log_q_actor[a] * td_error
        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        self.optimizer_actor.step()
        return actor_loss, loss_critic

a2c = A2C()
game = snake_game_for_AC.SnakeGame(speed=50,mode='machine')
actor_loss_l = []
critic_loss_l = []
reward_l = []


for epoch in range(1000) :
    game.reset()
    s = game.state()[0]
    ep_r = 0
    time_not_die = 0
    # print("reset")
    while True :
        time_not_die += 1
        a = a2c.choose_action(s)      
        game.action(a)
        done,_ , r = game.play_step()
        s_ = game.state()[0]
        ep_r += r

        time_not_die = 0
        
        actor_loss, critic_loss =a2c.learn(s, a, r, s_)
        actor_loss_l.append(actor_loss)
        critic_loss_l.append(critic_loss)

        if done :
            break
        s = s_
        EPSILON = min(0.9, EPSILON+0.0001)
    reward_l.append(ep_r)
    print(f'Ep: {epoch : 2.2f} | Ep_r: {round(ep_r):1.2f}', end = '\r')

#test 1000 games
rewards = []
scores = []
for i in range(1000):
    game.reset()
    s = game.state()[0]
    ep_r = 0
    time_not_die = 0
    while True:
        time_not_die += 1
        a = a2c.choose_action(s)      
        game.action(a)
        done,_ , r = game.play_step()
        s_ = game.state()[0]
        ep_r += r
        if done:
            break
        s = s_
    rewards.append(ep_r)
    scores.append(game.score)
    print(f'Ep: {i : 2.2f} | Ep_r: {ep_r:1.2f}', end = '\r')

critic_loss_l = [i.item() for i in critic_loss_l]
actor_loss_l  = [i.item() for i in  actor_loss_l]
plt.plot(critic_loss_l, label = 'crtic_loss')
plt.legend()
plt.savefig('crtic_loss_AC.png')
plt.clf()
plt.plot(actor_loss_l, label = 'actor_loss')
plt.legend()
plt.savefig('actor_loss_AC.png')
plt.clf()
plt.plot(reward_l, label = 'reward')
plt.legend()
plt.savefig('reward_AC.png')
plt.clf()

# print the average reward and score
print(f'average reward: {np.mean(rewards)}')
print(f'average score: {np.mean(scores)}')

# print the best reward and score
print(f'best reward: {np.max(rewards)}')
print(f'best score: {np.max(scores)}')

# print the worst reward and score
print(f'worst reward: {np.min(rewards)}')
print(f'worst score: {np.min(scores)}')

# plot a histogram of the rewards
plt.hist(rewards)
plt.savefig('rewards_AC.png')
plt.clf()
plt.hist(scores)
plt.savefig('scores_AC.png')
plt.clf()
print('done')
