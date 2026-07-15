

"""
RL 控制器 (Capture Loss + Persistence)
文件名: algorithms/rl_controller.py
功能:
1. 基于 4 维状态进行 DQN 动作选择；
2. 输出 5 类离散搜索策略，对应不同 pc / pm；
3. 记录 reward / loss / q_values；
4. 支持模型权重保存与加载；
5. 支持每次新 run 前重置 episode 状态，避免跨实验串扰。
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
        # episode 内部状态
        self.last_state = None
        self.last_action = 0
        self.last_best_fitness = -float('inf')
        self.stagnation_counter = 0

    def reset_episode(self):
        """
        每次新的 GA run 开始前调用，避免不同 run 之间状态串扰。
        注意：不会清空 DQN 网络参数，也不会清空经验池；
        这里只重置当前 episode 的轨迹状态。
        """
        self.last_state = None
        self.last_action = 0
        self.last_best_fitness = -float("inf")
        self.stagnation_counter = 0

    def _build_state(self, ga_state_dict):
        """
        将 GA 状态构造成 RL 状态向量：
        [diversity, stagnation, progress, conv_ratio]
        """
        # 当前 genetic_algorithm.py 的 diversity 已经是 [0,1] 内比例值
        diversity = float(np.clip(ga_state_dict["diversity"], 0.0, 1.0))

        # 停滞计数归一化
        stagnation = float(np.clip(self.stagnation_counter / 50.0, 0.0, 1.0))

        # 进化进度
        progress = float(np.clip(ga_state_dict["progress"], 0.0, 1.0))

        # 收敛程度近似指标，做最小稳定化
        best_fit = float(ga_state_dict["best_fit"])
        avg_fit = float(ga_state_dict["avg_fit"])
        denom = max(abs(best_fit), 1e-6)
        conv_ratio = float(np.clip(avg_fit / denom, -5.0, 5.0))

        return [diversity, stagnation, progress, conv_ratio]

    def _calc_reward(self, current_best, ga_state_dict):
        """
        奖励函数：
        1) 若当前最优适应度相对上一时刻提升，给 +10；
        2) 否则按连续停滞代数给递增惩罚；
        3) 若处于前半程且多样性较高，额外给 +2 探索奖励。
        """
        # diversity, _, progress, _ = current_state
        diversity = float(np.clip(ga_state_dict["diversity"], 0.0, 1.0))
        progress = float(np.clip(ga_state_dict["progress"], 0.0, 1.0))

        # 性能提升奖励 + 停滞惩罚
        if current_best > self.last_best_fitness + 1e-6:
            self.stagnation_counter = 0
            reward = 10.0
        else:
            self.stagnation_counter += 1
            reward = -0.1 * self.stagnation_counter

        # 早期探索奖励
        if progress < 0.5 and diversity > 0.2:
            reward += 2.0

        self.last_best_fitness = current_best
        return float(reward)

    def get_action(self, ga_state_dict, training=True):
        """
        输入:
            ga_state_dict = {
                'diversity': ...,
                'avg_fit': ...,
                'best_fit': ...,
                'progress': ...
            }

        输出:
            {
                'action_id': ...,
                'pc': ...,
                'pm': ...,
                'q_values': ...,
                'state_vector': ...,
                'reward': ...,
                'loss': ...
            }
        """
        current_best = float(ga_state_dict["best_fit"])
        # 先更新 reward / stagnation，再构造 state，消除一拍延迟
        reward = self._calc_reward(current_best, ga_state_dict)
        current_state = self._build_state(ga_state_dict)
        # current_state = self._build_state(ga_state_dict)
        # reward = self._calc_reward(current_best, current_state)

        loss = None
        if training and self.last_state is not None:
            self.agent.store_transition(
                self.last_state,
                self.last_action,
                reward,
                current_state,
                False
            )
            loss = self.agent.learn()

        action_id, q_values = self.agent.select_action(current_state, training=training)

        self.last_state = current_state
        self.last_action = int(action_id)

        params = self.action_map[action_id]
        return {
            "action_id": int(action_id),
            "action_name": params["name"],
            "pc": params["pc"],
            "pm": params["pm"],
            "q_values": q_values.tolist() if hasattr(q_values, "tolist") else list(q_values),
            "state_vector": current_state,
            "reward": reward,
            "loss": loss
        }


    def save_brain(self, filename="drl_model_memory.pth"):
        """
        保存策略网络权重。
        """
        folder = os.path.dirname(filename)
        if folder:
            os.makedirs(folder, exist_ok=True)

        torch.save(
            {
                "policy_net": self.agent.policy_net.state_dict(),
                "target_net": self.agent.target_net.state_dict(),
                "epsilon": self.agent.epsilon,
                "state_dim": self.state_dim,
                "action_dim": self.action_dim,
            },
            filename,
        )

    def load_brain(self, filename="drl_model_memory.pth"):
        """
        加载策略网络权重。
        返回 True 表示加载成功，False 表示未加载。
        """
        if not os.path.exists(filename):
            return False

        try:
            checkpoint = torch.load(filename, map_location="cpu")

            if isinstance(checkpoint, dict) and "policy_net" in checkpoint:
                self.agent.policy_net.load_state_dict(checkpoint["policy_net"])
                if "target_net" in checkpoint:
                    self.agent.target_net.load_state_dict(checkpoint["target_net"])
                else:
                    self.agent.target_net.load_state_dict(self.agent.policy_net.state_dict())

                if "epsilon" in checkpoint:
                    self.agent.epsilon = float(checkpoint["epsilon"])
            else:
                # 兼容旧版：直接保存的是 policy_net.state_dict()
                self.agent.policy_net.load_state_dict(checkpoint)
                self.agent.target_net.load_state_dict(self.agent.policy_net.state_dict())

            return True
        except Exception:
            return False

    # def save_brain(self, filename="drl_model_memory.pth"):
    #     """保存模型权重"""
    #     torch.save(self.agent.policy_net.state_dict(), filename)
    #
    # def load_brain(self, filename="drl_model_memory.pth"):
    #     """加载模型权重"""
    #     if os.path.exists(filename):
    #         try:
    #             self.agent.policy_net.load_state_dict(torch.load(filename))
    #             self.agent.target_net.load_state_dict(self.agent.policy_net.state_dict())
    #         except:
    #             pass
    #
