"""
DQN 智能体核心逻辑 (With Loss Return)
文件名: dl_module/dqn_agent.py
"""
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import random
from collections import deque
from .model import QNetwork

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DQNAgent:
    def __init__(self, state_dim=4, action_dim=5, lr=0.001, gamma=0.9, buffer_size=2000):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.lr = lr

        self.policy_net = QNetwork(state_dim, action_dim).to(DEVICE)
        self.target_net = QNetwork(state_dim, action_dim).to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        self.memory = deque(maxlen=buffer_size)
        self.batch_size = 32
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.learn_step_counter = 0
        self.target_update_freq = 100

    def select_action(self, state, training=True):
        if training and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_dim), np.zeros(self.action_dim)

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        return torch.argmax(q_values).item(), q_values.cpu().numpy()[0]

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def learn(self):
        """训练网络并返回 Loss"""
        if len(self.memory) < self.batch_size:
            return None  # 还没开始学

        batch = random.sample(self.memory, self.batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)

        state_batch = torch.FloatTensor(np.array(state_batch)).to(DEVICE)
        action_batch = torch.LongTensor(action_batch).unsqueeze(1).to(DEVICE)
        reward_batch = torch.FloatTensor(reward_batch).unsqueeze(1).to(DEVICE)
        next_state_batch = torch.FloatTensor(np.array(next_state_batch)).to(DEVICE)
        done_batch = torch.FloatTensor(done_batch).unsqueeze(1).to(DEVICE)

        q_eval = self.policy_net(state_batch).gather(1, action_batch)

        with torch.no_grad():
            q_next = self.target_net(next_state_batch).max(1)[0].unsqueeze(1)
            q_target = reward_batch + (self.gamma * q_next * (1 - done_batch))

        loss = self.loss_fn(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.learn_step_counter += 1
        if self.learn_step_counter % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()  # [关键] 返回 loss 值



# """
# DQN 智能体核心逻辑
# 文件名: dl_module/dqn_agent.py
#
# 功能:
#     1. 管理两个网络 (Eval Net, Target Net)
#     2. 经验回放 (Replay Buffer)
#     3. Epsilon-Greedy 策略
#     4. 训练步 (Learn)
# """
# import torch
# import torch.optim as optim
# import torch.nn as nn
# import numpy as np
# import random
# from collections import deque
# from .model import QNetwork
#
# # 检测设备
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#
# class DQNAgent:
#     def __init__(self, state_dim=4, action_dim=5, lr=0.001, gamma=0.9, buffer_size=2000):
#         self.state_dim = state_dim
#         self.action_dim = action_dim
#         self.gamma = gamma
#         self.lr = lr
#
#         # 策略网络与目标网络
#         self.policy_net = QNetwork(state_dim, action_dim).to(DEVICE)
#         self.target_net = QNetwork(state_dim, action_dim).to(DEVICE)
#         self.target_net.load_state_dict(self.policy_net.state_dict())
#         self.target_net.eval()  # 目标网络不训练
#
#         self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
#         self.loss_fn = nn.MSELoss()
#
#         # 经验池
#         self.memory = deque(maxlen=buffer_size)
#         self.batch_size = 32
#
#         # 探索参数
#         self.epsilon = 1.0
#         self.epsilon_min = 0.05
#         self.epsilon_decay = 0.995
#
#         self.learn_step_counter = 0
#         self.target_update_freq = 100
#
#     def select_action(self, state, training=True):
#         """选择动作: training=True 时使用 e-greedy"""
#         if training and np.random.rand() <= self.epsilon:
#             return random.randrange(self.action_dim), np.zeros(self.action_dim)
#
#         state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
#         with torch.no_grad():
#             q_values = self.policy_net(state_tensor)
#
#         action = torch.argmax(q_values).item()
#         return action, q_values.cpu().numpy()[0]
#
#     def store_transition(self, state, action, reward, next_state, done):
#         """存储经验"""
#         self.memory.append((state, action, reward, next_state, done))
#
#     def learn(self):
#         """从经验池采样并训练网络"""
#         if len(self.memory) < self.batch_size:
#             return
#
#         # 采样
#         batch = random.sample(self.memory, self.batch_size)
#         state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)
#
#         state_batch = torch.FloatTensor(np.array(state_batch)).to(DEVICE)
#         action_batch = torch.LongTensor(action_batch).unsqueeze(1).to(DEVICE)
#         reward_batch = torch.FloatTensor(reward_batch).unsqueeze(1).to(DEVICE)
#         next_state_batch = torch.FloatTensor(np.array(next_state_batch)).to(DEVICE)
#         done_batch = torch.FloatTensor(done_batch).unsqueeze(1).to(DEVICE)
#
#         # 计算 Q(s, a)
#         q_eval = self.policy_net(state_batch).gather(1, action_batch)
#
#         # 计算 Target Q: r + gamma * max Q'(s', a')
#         with torch.no_grad():
#             q_next = self.target_net(next_state_batch).max(1)[0].unsqueeze(1)
#             q_target = reward_batch + (self.gamma * q_next * (1 - done_batch))
#
#         # 梯度下降
#         loss = self.loss_fn(q_eval, q_target)
#         self.optimizer.zero_grad()
#         loss.backward()
#         self.optimizer.step()
#
#         # 衰减 epsilon
#         if self.epsilon > self.epsilon_min:
#             self.epsilon *= self.epsilon_decay
#
#         # 定期更新目标网络
#         self.learn_step_counter += 1
#         if self.learn_step_counter % self.target_update_freq == 0:
#             self.target_net.load_state_dict(self.policy_net.state_dict())
#
#     def save(self, path):
#         torch.save(self.policy_net.state_dict(), path)
#
#     def load(self, path):
#         self.policy_net.load_state_dict(torch.load(path))
#         self.target_net.load_state_dict(self.policy_net.state_dict())