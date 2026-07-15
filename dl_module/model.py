"""
深度Q网络模型定义 (PyTorch Implementation)
文件名: dl_module/model.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """
    输入: 状态向量 (State Dim = 4)
    输出: 动作Q值 (Action Dim = 5)
    结构: FC(64) -> ReLU -> FC(64) -> ReLU -> FC(ActionDim)
    """

    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)

        # 初始化权重 (He initialization)
        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc2.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)