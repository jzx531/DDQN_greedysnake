from collections import deque
import random
import numpy as np
import pygame
import torch.nn
import torch.nn.functional as F
import torch
import rl_utils
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import sys
import math
import os

def setup_seed(seed=0):
    random.seed(seed)  # Python的随机性
    os.environ['PYTHONHASHSEED'] = str(seed)  # 设置Python哈希种子，为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)  # numpy的随机性
    torch.manual_seed(seed)  # torch的CPU随机性，为CPU设置随机种子
    torch.cuda.manual_seed(seed)  # torch的GPU随机性，为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.   torch的GPU随机性，为所有GPU设置随机种子


class map:
    def __init__(self):
        self.len = 10
        self.wid = 10


class snake:
    def __init__(self):
        self.x = 2
        self.y = 2
        self.direction = ['w', 'a', 's', 'd']
        self.direct = 'w'
        self.pos = deque()
        self.pos.append((self.x, self.y))
        self.pos.append((self.x, self.y - 1))
        self.score = 0
        self.done = False

    def mov(self, direct):
        if direct == 'w' and self.direct != self.direction[2]:
            self.direct = direct
        elif direct == 's' and self.direct != self.direction[0]:
            self.direct = direct
        elif direct == 'a' and self.direct != self.direction[3]:
            self.direct = direct
        elif direct == 'd' and self.direct != self.direction[1]:
            self.direct = direct
        if self.direct == self.direction[0]:
            self.pos.appendleft((self.pos[0][0], self.pos[0][1] + 1))
            self.pos.pop()
        elif self.direct == self.direction[1]:
            self.pos.appendleft((self.pos[0][0] - 1, self.pos[0][1]))
            self.pos.pop()
        elif self.direct == self.direction[2]:
            self.pos.appendleft((self.pos[0][0], self.pos[0][1] - 1))
            self.pos.pop()
        else:
            self.pos.appendleft((self.pos[0][0] + 1, self.pos[0][1])), self.pos.pop()

    def eated(self, food, map):
        length = len(self.pos)
        if self.pos[0][0] == food.x and self.pos[0][1] == food.y:
            self.pos.append((2 * self.pos[length - 1][0] - self.pos[length - 2][0],
                             2 * self.pos[length - 1][1] - self.pos[length - 2][1]))
            self.score += food.score
            food.reset(self, map)
            # #if self.direct == self.direction[0]:
            #  #   self.pos.appendleft((self.pos[0][0], self.pos[0][1] + 1))
            # elif self.direct == self.direction[1]:
            #     self.pos.appendleft((self.pos[0][0] - 1, self.pos[0][1]))
            # elif self.direct == self.direction[2]:
            #     self.pos.appendleft((self.pos[0][0], self.pos[0][1] - 1))
            # else:
            #     self.pos.appendleft((self.pos[0][0] + 1, self.pos[0][1]))

    def judgedead(self, map):
        if self.pos[0][0] >= map.len or self.pos[0][0] < 0 or self.pos[0][1] >= map.wid or self.pos[0][1] < 0:
            self.done = True
            self.score -= 1000
        for i in range(len(self.pos)):
            if i != 0:
                if self.pos[0][0] == self.pos[i][0] and self.pos[0][1] == self.pos[i][1]:
                    self.done = True
            self.score -= 1000

    def move_onestep(self, direct, food, map):
        self.mov(direct)
        self.judgedead(map)
        self.eated(food, map)

    def reset(self):
        self.x = 1
        self.y = 1
        self.direct = 'w'
        self.pos.clear()
        self.pos.append((self.x, self.y))
        self.pos.append((self.x, self.y - 1))
        self.score = 0
        self.done = False


class food:
    def __init__(self, map):
        self.x = np.random.randint(map.len - 3) + 2
        self.y = np.random.randint(map.wid - 3) + 2
        self.score = 200

    def reset(self, snake, map):
        #setup_seed()
        list = []
        for x in range(map.len):
            for y in range(map.wid):
                attach = True
                for i in range(len(snake.pos)):
                    if x == snake.pos[i][0] and y == snake.pos[i][1]:
                        attach = False
                if attach:
                    list.append((x, y))
        long = len(list)
        item = np.random.randint(long)
        self.x = list[item][0]
        self.y = list[item][1]
        # self.x = np.random.randint(map.len - 3) + 2
        # self.y = np.random.randint(map.wid - 3) + 2
        # for i in range(len(snake.pos)):
        #     while self.x == snake.pos[i][0] and self.y == snake.pos[i][1]:
        #         self.x = np.random.randint(map.len - 3) + 2
        #         self.y = np.random.randint(map.wid - 3) + 2


class gameenv:
    def __init__(self, map, snake, food):
        self.map = map
        self.snake = snake
        self.food = food
        # self.state=np.zeros([map.len,map.wid])
        # self.state = [0 for _ in range(self.map.len * 40 + self.map.wid + 1)]
        self.state = [0 for _ in range(32)]

    def newstate(self):
        self.state = [0 for _ in range(32)]
        item = self.snake.direction.index(self.snake.direct)
        self.state[item] = 1
        dir = ''
        length = len(self.snake.pos)
        if self.snake.pos[length - 1][0] - self.snake.pos[length - 2][0] == 1:
            dir = 'a'
        elif self.snake.pos[length - 1][0] - self.snake.pos[length - 2][0] == -1:
            dir = 'd'
        elif self.snake.pos[length - 1][1] - self.snake.pos[length - 2][1] == -1:
            dir = 'w'
        elif self.snake.pos[length - 1][1] - self.snake.pos[length - 2][1] == 1:
            dir = 's'
        item = self.snake.direction.index(dir)
        self.state[item + 4] = 1
        item = 8
        x = self.snake.pos[0][0]
        y = self.snake.pos[0][1]
        if (self.food.x - x) == (self.food.y - x):
            if (self.food.x - x) > 0:
                self.state[item + 1] = 1
            else:
                self.state[item + 5] = 1
        elif (self.food.x - x) == -(self.food.y - x):
            if (self.food.x - x) > 0:
                self.state[item + 3] = 1
            else:
                self.state[item + 7] = 1
        if (self.food.x - x) == 0:
            if (self.food.y - y) > 0:
                self.state[item] = 1
            else:
                self.state[item + 4] = 1
        if (self.food.y - y) == 0:
            if (self.food.x - x) > 0:
                self.state[item + 2] = 1
        else:
            self.state[item + 6] = 1
        item = 16
        for i in range(1, length):
            x_body = self.snake.pos[i][0]
            y_body = self.snake.pos[i][1]
            if (x_body - x) == (y_body - x):
                if (x_body - x) > 0:
                    self.state[item + 1] = 1
                else:
                    self.state[item + 5] = 1
            elif (x_body - x) == -(y_body - x):
                if (x_body - x) > 0:
                    self.state[item + 3] = 1
                else:
                    self.state[item + 7] = 1
            if (x_body - x) == 0:
                if (y_body - y) > 0:
                    self.state[item] = 1
                else:
                    self.state[item + 4] = 1
            if (y_body - y) == 0:
                if (x_body - x) > 0:
                    self.state[item + 2] = 1
            else:
                self.state[item + 6] = 1
        left = self.snake.pos[0][0] + 0.1
        right = self.map.len - self.snake.pos[0][0] + 0.1
        high = self.snake.pos[0][1] + 0.1
        low = self.map.wid - self.snake.pos[0][1] + 0.1
        rh = math.sqrt(math.pow(right, 2) + math.pow(high, 2)) + 0.1
        rl = math.sqrt(math.pow(right, 2) + math.pow(low, 2)) + 0.1
        ll = math.sqrt(math.pow(left, 2) + math.pow(low, 2)) + 0.1
        lh = math.sqrt(math.pow(left, 2) + math.pow(high, 2)) + 0.1
        item = 24
        self.state[item] = 1 / high
        self.state[item + 1] = 1 / rh
        self.state[item + 2] = 1 / right
        self.state[item + 3] = 1 / rl
        self.state[item + 4] = 1 / low
        self.state[item + 5] = 1 / ll
        self.state[item + 6] = 1 / left
        self.state[item + 7] = 1 / lh
        return self.state

        # self.state = [0 for _ in range(self.map.len * self.map.wid * 2 + 3)]
        # for i in range(len(self.snake.pos)):
        #     self.state[2 * i] = self.snake.pos[i][0]
        #     self.state[2 * i + 1] = self.snake.pos[i][1]
        # self.state[self.map.len * self.map.wid * 2] = self.food.x
        # self.state[self.map.len * self.map.wid * 2 + 1] = self.food.y
        # for i in range(len(self.snake.direction)):
        #     if self.snake.direct == self.snake.direction[i]:
        #         self.state[self.map.len * self.map.wid * 2 + 2] = i
        # return self.state

        # self.state = [0 for _ in range(self.map.len * 40 + self.map.wid + 1)]
        # for i in range(len(self.snake.pos)):
        #     if i == 0:
        #         self.state[self.snake.pos[i][0] * self.map.len + self.snake.pos[i][1]] = 100
        #     else:
        #         self.state[self.snake.pos[i][0] * self.map.len + self.snake.pos[i][1]] = 15
        # self.state[self.food.x * self.map.len + self.food.y] = 30
        # for i in range(len(self.snake.direction)):
        #     if self.snake.direct == self.snake.direction[i]:
        #         self.state[self.map.len * 40 + self.map.wid] = i
        # return self.state
    def distance(self,i):
        distance=math.sqrt(math.pow(self.food.x - self.snake.pos[i][0], 2) +
                           math.pow(self.food.y - self.snake.pos[i][1], 2)) + 0.1

        return distance
    def runonestep(self, action):
        rewards = 0
        distance=self.distance(0)
        self.state = self.newstate()
        lenth = len(self.snake.pos)
        self.snake.move_onestep(self.snake.direction[action], self.food, self.map)
        new_distance=self.distance(0)
        newstate = self.newstate()
        done = self.snake.done
        if self.snake.done is True:
            rewards = -10
        if self.snake.done is False:
             # if new_distance<distance:
             #    rewards = 2
             # else: rewards=1
             rewards=1
             if len(self.snake.pos) > lenth:
                rewards = self.food.score

        return newstate, rewards, done

    def reset(self):
        self.state = self.newstate()
        self.snake.reset()
        return self.state




# greedysnake.runonestep(3)
# print(greedysnake.state)
class qnet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim1,hidden_dim2, action_dim):
        super(qnet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim1)
        self.fc2 = torch.nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = torch.nn.Linear(hidden_dim2, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return self.fc3(x)


class DQN:
    def __init__(self, state_dim,
                 hidden_dim1,
                 hidden_dim2,
                 action_dim,
                 learning_rate,
                 gamma,
                 epsilon,
                 target_update):
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.target_update = target_update
        self.count = 0
        self.lr = learning_rate
        self.qnet = qnet(state_dim, hidden_dim1,hidden_dim2, action_dim)
        # self.qnet.weight.data.normal_(1.0, 0.02)
        self.target_qnet = qnet(state_dim, hidden_dim1,hidden_dim2, action_dim)
        # self.optimizer = torch.optim.SGD(self.qnet.parameters(), lr=learning_rate,momentum=0.09)
        self.optimizer = torch.optim.Adagrad(self.qnet.parameters(), lr=learning_rate)

    def take_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            state = torch.tensor([state], dtype=torch.float)
            action = self.qnet(state).argmax().item()
        return action

    def max_q_value(self, state):
        state = torch.tensor([state], dtype=torch.float)
        return self.qnet(state).max().item()

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1)
        q_values = self.qnet(states).gather(1, actions)  # Q值
        # 下个状态的最大Q值

        max_action = self.qnet(next_states).max(1)[1].view(-1, 1)
        max_next_q_values = self.target_qnet(next_states).gather(1, max_action)

        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)  # TD误差目标
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))  # 均方误差损失函数
        self.optimizer.zero_grad()  # PyTorch中默认梯度会累积,这里需要显式将梯度置为0
        dqn_loss.backward()  # 反向传播更新参数
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.target_qnet.load_state_dict(
                self.qnet.state_dict())  # 更新目标网络
        self.count += 1


def train_DQN(agent, env, num_episodes, replay_buffer, minimal_size,
              batch_size):
    return_list = []
    max_q_value_list = []
    max_q_value = 0
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                state = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    max_q_value = agent.max_q_value(
                        state) * 0.005 + max_q_value * 0.995  # 平滑处理
                    max_q_value_list.append(max_q_value)  # 保存每个状态的最大Q值
                    next_state, reward, done = env.runonestep(action)
                    replay_buffer.add(state, action, reward, next_state, done)
                    state = next_state
                    episode_return += reward
                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(
                            batch_size)
                        transition_dict = {
                            'states': b_s,
                            'actions': b_a,
                            'next_states': b_ns,
                            'rewards': b_r,
                            'dones': b_d
                        }
                        agent.update(transition_dict)
                return_list.append(episode_return)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({
                        'episode':
                            '%d' % (num_episodes / 10 * i + i_episode + 1),
                        'return':
                            '%.3f' % np.mean(return_list[-10:]),
                        'lenth':
                            '%d' % len(env.snake.pos)
                    })
                pbar.update(1)
    return return_list, max_q_value_list


def save_model(save_path, iteration, optimizer, model):
    torch.save({'iteration': iteration,
                'optimizer_dict': optimizer.state_dict(),
                'model_dict': model.state_dict()},
               save_path)
    print("model save success")


def load_model(save_name, optimizer, model):
    model_data = torch.load(save_name)
    model.load_state_dict(model_data['model_dict'])
    optimizer.load_state_dict(model_data['optimizer_dict'])
    print("model load success")

lr = 1e-2
num_episodes = 5000


path = 'target_net.pkl'
random.seed(0)
np.random.seed(0)
#env.seed(0)
torch.manual_seed(0)
#setup_seed()
gamma = 0.98
epsilon = 0.01
target_update = 50
buffer_size = 5000
minimal_size = 1000
batch_size = 1000
hidden_dim1 = 128
hidden_dim2 = 128
map = map()
snake = snake()
food = food(map)
greedysnake = gameenv(map, snake, food)
state_dim = len(greedysnake.state)
action_dim = 4
replay_buffer = rl_utils.ReplayBuffer(buffer_size)
agent = DQN(state_dim, hidden_dim1,hidden_dim2, action_dim, lr, gamma, epsilon,
            target_update)
#
load_model(path, agent.optimizer, agent.qnet)
load_model(path, agent.optimizer, agent.target_qnet)
train = False
if train:
    setup_seed()
    return_list, max_q_value_list = train_DQN(agent, greedysnake, num_episodes,
                                              replay_buffer, minimal_size,
                                              batch_size)
else:
    return_list = []
    max_q_value_list = []

if train:

    episodes_list = list(range(len(return_list)))
    mv_return = rl_utils.moving_average(return_list, 5)
    plt.plot(episodes_list, mv_return)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('DQN on {}'.format('greedy_snake'))
    plt.show()

    frames_list = list(range(len(max_q_value_list)))
    plt.plot(frames_list, max_q_value_list)
    plt.axhline(0, c='orange', ls='--')
    plt.axhline(10, c='red', ls='--')
    plt.xlabel('Frames')
    plt.ylabel('Q value')
    plt.title('DQN on {}'.format('greedy_snake'))
    plt.show()
    save_model(path, num_episodes, agent.optimizer, agent.target_qnet)



class Game:
    def __init__(self, greedysnake):
        pygame.init()
        self.screen = pygame.display.set_mode((500, 500))
        pygame.display.set_caption('AI_greedy_snake_DQN')
        self.game = greedysnake
        self.screen.fill(pygame.Color(255, 255, 255))

    def draw(self):
        self.screen.fill(pygame.Color(255, 255, 255))
        # bg_color2 = (60, 60, 60)
        bg_color3 = (120, 160, 160)
        for i in range(len(self.game.snake.pos)):
            bg_color2 = (60 + 5 * i, 60 + 5 * i, 60 + 5 * i)
            bullet_rect = pygame.Rect(self.game.snake.pos[i][0] * 50, self.game.snake.pos[i][1] * 50, 50, 50)
            pygame.draw.rect(self.screen, bg_color2, bullet_rect)
        bullet_rect = pygame.Rect(self.game.food.x * 50, self.game.food.y * 50, 50, 50)
        pygame.draw.rect(self.screen, bg_color3, bullet_rect)
        pygame.display.flip()


epsilon = 0.005
aigs = Game(greedysnake)
rectangle = 50
aigs.game.snake.reset()
#setup_seed()
while aigs.game.snake.done == False:
    aigs.draw()
    state = aigs.game.newstate()
    state = torch.tensor([state], dtype=torch.float)
    # action = agent.target_qnet(state).argmax().item()
    if np.random.random() < epsilon:
        action = np.random.randint(action_dim)
    else:
        action = agent.target_qnet(state).argmax().item()
    aigs.game.runonestep(action)
    aigs.draw()
    time.sleep(0.2)
pygame.quit()
print(len(aigs.game.snake.pos))
#sys.exit()

