# """
# RL 控制器 (Capture Loss)
# 文件名: algorithms/rl_controller.py
# """
# import numpy as np
# from dl_module.dqn_agent import DQNAgent
#
#
# class RLController:
#     def __init__(self):
#         self.state_dim = 4
#         self.action_dim = 5
#         self.agent = DQNAgent(self.state_dim, self.action_dim)
#
#         self.action_map = {
#             0: {'name': 'Hold', 'pc': None, 'pm': None},
#             1: {'name': 'Converge', 'pc': 0.90, 'pm': 0.01},
#             2: {'name': 'Balanced', 'pc': 0.80, 'pm': 0.10},
#             3: {'name': 'Explore', 'pc': 0.60, 'pm': 0.20},
#             4: {'name': 'Emergency', 'pc': 0.40, 'pm': 0.50}
#         }
#         self.last_state = None
#         self.last_action = 0
#         self.last_best_fitness = -float('inf')
#         self.stagnation_counter = 0
#
#     def get_action(self, ga_state_dict, training=True):
#         current_best = ga_state_dict['best_fit']
#
#         if current_best > self.last_best_fitness + 1e-6:
#             self.stagnation_counter = 0
#             reward = 10.0
#         else:
#             self.stagnation_counter += 1
#             reward = -0.1 * self.stagnation_counter
#         self.last_best_fitness = current_best
#
#         diversity = min(1.0, ga_state_dict['diversity'] / 1000.0)
#         stagnation = min(1.0, self.stagnation_counter / 50.0)
#         progress = ga_state_dict['progress']
#         conv_ratio = ga_state_dict['avg_fit'] / (ga_state_dict['best_fit'] + 1e-5)
#         current_state = [diversity, stagnation, progress, conv_ratio]
#
#         loss = None  # 初始化 Loss
#         if training and self.last_state is not None:
#             if progress < 0.5 and diversity > 0.2:
#                 reward += 2.0
#             self.agent.store_transition(self.last_state, self.last_action, reward, current_state, False)
#             loss = self.agent.learn()  # [关键] 获取 Loss
#
#         action_id, q_values = self.agent.select_action(current_state, training)
#         self.last_state = current_state
#         self.last_action = action_id
#
#         params = self.action_map[action_id]
#         return {
#             'action_id': action_id,
#             'pc': params['pc'],
#             'pm': params['pm'],
#             'q_values': q_values.tolist(),
#             'state_vector': current_state,
#             'reward': reward,
#             'loss': loss  # [关键] 返回 Loss
#         }

"""
RL 控制器 (Capture Loss + Persistence)
文件名: algorithms/rl_controller.py
"""
import numpy as np
import torch
import os
from dl_module.dqn_agent import DQNAgent


class RLController:
    def __init__(self):
        self.state_dim = 4
        self.action_dim = 5
        self.agent = DQNAgent(self.state_dim, self.action_dim)

        self.action_map = {
            0: {'name': 'Hold', 'pc': None, 'pm': None},
            1: {'name': 'Converge', 'pc': 0.90, 'pm': 0.01},
            2: {'name': 'Balanced', 'pc': 0.80, 'pm': 0.10},
            3: {'name': 'Explore', 'pc': 0.60, 'pm': 0.20},
            4: {'name': 'Emergency', 'pc': 0.40, 'pm': 0.50}
        }
        self.last_state = None
        self.last_action = 0
        self.last_best_fitness = -float('inf')
        self.stagnation_counter = 0

    def get_action(self, ga_state_dict, training=True):
        current_best = ga_state_dict['best_fit']

        if current_best > self.last_best_fitness + 1e-6:
            self.stagnation_counter = 0
            reward = 10.0
        else:
            self.stagnation_counter += 1
            reward = -0.1 * self.stagnation_counter
        self.last_best_fitness = current_best

        diversity = min(1.0, ga_state_dict['diversity'] / 1000.0)
        stagnation = min(1.0, self.stagnation_counter / 50.0)
        progress = ga_state_dict['progress']
        conv_ratio = ga_state_dict['avg_fit'] / (ga_state_dict['best_fit'] + 1e-5)
        current_state = [diversity, stagnation, progress, conv_ratio]

        loss = None
        if training and self.last_state is not None:
            if progress < 0.5 and diversity > 0.2:
                reward += 2.0
            self.agent.store_transition(self.last_state, self.last_action, reward, current_state, False)
            loss = self.agent.learn()

        action_id, q_values = self.agent.select_action(current_state, training)
        self.last_state = current_state
        self.last_action = action_id

        params = self.action_map[action_id]
        return {
            'action_id': action_id,
            'pc': params['pc'],
            'pm': params['pm'],
            'q_values': q_values.tolist() if hasattr(q_values, 'tolist') else q_values,
            'state_vector': current_state,
            'reward': reward,
            'loss': loss
        }

    def save_brain(self, filename="drl_model_memory.pth"):
        """保存模型权重"""
        torch.save(self.agent.policy_net.state_dict(), filename)

    def load_brain(self, filename="drl_model_memory.pth"):
        """加载模型权重"""
        if os.path.exists(filename):
            try:
                self.agent.policy_net.load_state_dict(torch.load(filename))
                self.agent.target_net.load_state_dict(self.agent.policy_net.state_dict())
            except:
                pass

