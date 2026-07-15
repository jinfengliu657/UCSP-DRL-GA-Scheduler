"""
算法模块初始化
文件名: algorithms/__init__.py
"""
# 1. 核心算法引擎
from .genetic_algorithm import GeneticAlgorithm

# 2. 遗传算子 (使用新类名)
from .selection import RouletteWheelSelection
from .crossover import UniformCrossover  # 修正这里
from .mutation import UniformMutation    # 修正这里

# 3. 增强模块
from .tabu_search import TabuSearch
from .constraint_solver import generate_feasible_solution
from .rl_controller import RLController
from .zhu_replica import ZhuReplicaAlgorithm
from .ifts import IFTS