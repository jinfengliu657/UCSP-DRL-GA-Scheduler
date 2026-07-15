"""
选择算子 (Selection)
文件名: algorithms/algorithms/selection.py
"""
import random


class RouletteWheelSelection:
    """轮盘赌选择策略"""

    def select(self, population):
        """从种群中选择一个个体 (适应度越高被选概率越大)"""
        if not population:
            return None

        # 1. 处理适应度 (特别是负值情况)
        # 如果有负值，将所有适应度平移到正数区间
        min_fit = min(c.fitness for c in population)
        if min_fit < 0:
            # 加上 offset 使得最小值为 1 (保持正数)
            offset = abs(min_fit) + 1.0
            fits = [c.fitness + offset for c in population]
        else:
            fits = [c.fitness for c in population]

        total_fit = sum(fits)

        # 防止除零
        if total_fit == 0:
            return random.choice(population)

        # 2. 轮盘赌逻辑
        pick = random.uniform(0, total_fit)
        current = 0
        for i, ind in enumerate(population):
            current += fits[i]
            if current > pick:
                return ind

        return population[-1]


# """
# 选择算子 - 锦标赛选择
# """
# import random
# from typing import List
#
# from models.chromosome import Chromosome
#
#
# class SelectionOperator:
#     """选择算子基类"""
#
#     def select(self, population: List[Chromosome], **kwargs) -> Chromosome:
#         """从种群中选择一个个体"""
#         raise NotImplementedError
#
#
# class TournamentSelection(SelectionOperator):
#     """
#     锦标赛选择
#
#     随机选择k个个体，返回适应度最高的
#     """
#
#     def __init__(self, tournament_size: int = 3):
#         """
#         初始化锦标赛选择
#
#         Args:
#             tournament_size: 锦标赛大小，默认3
#         """
#         self.tournament_size = tournament_size
#
#     def select(self, population: List[Chromosome]) -> Chromosome:
#         """
#         执行锦标赛选择
#
#         Args:
#             population: 种群
#
#         Returns:
#             被选中的个体
#         """
#         # 随机选择k个个体
#         competitors = random.sample(population, min(self.tournament_size, len(population)))
#         # 返回适应度最高的
#         return max(competitors, key=lambda x: x.fitness)
#
#
# class RouletteWheelSelection(SelectionOperator):
#     """轮盘赌选择"""
#
#     def select(self, population: List[Chromosome]) -> Chromosome:
#         """执行轮盘赌选择"""
#         # 计算总适应度（需要转换为正数）
#         min_fitness = min(c.fitness for c in population)
#         adjusted_fitness = [c.fitness - min_fitness + 1 for c in population]
#         total_fitness = sum(adjusted_fitness)
#
#         # 轮盘赌
#         r = random.uniform(0, total_fitness)
#         cumsum = 0
#         for i, chromo in enumerate(population):
#             cumsum += adjusted_fitness[i]
#             if cumsum >= r:
#                 return chromo
#
#         return population[-1]
#
#
# class RankSelection(SelectionOperator):
#     """排序选择 - 按适应度排序后按概率选择"""
#
#     def __init__(self, pressure: float = 1.5):
#         """
#         初始化排序选择
#
#         Args:
#             pressure: 选择压力，默认1.5
#         """
#         self.pressure = pressure
#
#     def select(self, population: List[Chromosome]) -> Chromosome:
#         """执行排序选择"""
#         # 按适应度排序
#         sorted_pop = sorted(population, key=lambda x: x.fitness, reverse=True)
#         n = len(sorted_pop)
#
#         # 计算选择概率
#         probs = []
#         for i in range(n):
#             probs.append((2 - self.pressure) / n + 2 * (self.pressure - 1) * i / (n * (n - 1)))
#
#         # 归一化
#         total = sum(probs)
#         probs = [p / total for p in probs]
#
#         # 按概率选择
#         r = random.random()
#         cumsum = 0
#         for i, prob in enumerate(probs):
#             cumsum += prob
#             if cumsum >= r:
#                 return sorted_pop[i]
#
#         return sorted_pop[-1]
