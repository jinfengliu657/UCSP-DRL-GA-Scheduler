"""
交叉算子 (Crossover)
文件名: algorithms/crossover.py
"""
import random
from models.chromosome import Chromosome


class UniformCrossover:
    def __init__(self, crossover_rate):
        self.rate = crossover_rate

    def cross(self, parent1: Chromosome, parent2: Chromosome):
        """执行均匀交叉，返回两个子代"""
        # 如果不交叉，返回克隆体
        if random.random() > self.rate:
            return parent1.clone(), parent2.clone()

        child1 = parent1.clone()
        child2 = parent2.clone()

        # 遍历所有基因位，按 50% 概率交换
        for task_id in parent1.genes:
            if random.random() < 0.5:
                # 交换基因
                gene1 = parent1.genes.get(task_id)
                gene2 = parent2.genes.get(task_id)

                if gene1 is not None and gene2 is not None:
                    child1.genes[task_id] = gene2
                    child2.genes[task_id] = gene1

        # 重置适应度
        child1.fitness = None
        child2.fitness = None

        return child1, child2

# """
# 交叉算子 - 约束感知的交叉操作
# """
# import random
# import numpy as np
# from typing import List, Tuple
#
# from models.chromosome import Chromosome
# from data.data_loader import ScheduleData
#
#
# class CrossoverOperator:
#     """交叉算子基类"""
#
#     def crossover(self, parent1: Chromosome, parent2: Chromosome,
#                  data: ScheduleData) -> Tuple[Chromosome, Chromosome]:
#         """执行交叉操作"""
#         raise NotImplementedError
#
#
# class UniformCrossover(CrossoverOperator):
#     """均匀交叉 - 每个基因独立决定来自哪个父代"""
#
#     def __init__(self, crossover_rate: float = 0.5):
#         """
#         初始化均匀交叉
#
#         Args:
#             crossover_rate: 每个基因来自parent1的概率
#         """
#         self.crossover_rate = crossover_rate
#
#     def crossover(self, parent1: Chromosome, parent2: Chromosome,
#                  data: ScheduleData) -> Tuple[Chromosome, Chromosome]:
#         """执行均匀交叉"""
#         offspring1 = parent1.clone()
#         offspring2 = parent2.clone()
#
#         # 获取所有任务ID（两个父代的并集）
#         all_task_ids = set(parent1.genes.keys()) | set(parent2.genes.keys())
#
#         for task_id in all_task_ids:
#             # 固定任务不交叉
#             if data and task_id in data.fixed_time_tasks:
#                 continue
#
#             if random.random() < self.crossover_rate:
#                 # 交换基因
#                 gene1 = parent1.genes.get(task_id)
#                 gene2 = parent2.genes.get(task_id)
#                 if gene1 is not None:
#                     offspring1.genes[task_id] = gene2
#                 if gene2 is not None:
#                     offspring2.genes[task_id] = gene1
#
#         return offspring1, offspring2
#
#
# class OnePointCrossover(CrossoverOperator):
#     """单点交叉"""
#
#     def crossover(self, parent1: Chromosome, parent2: Chromosome,
#                  data: ScheduleData) -> Tuple[Chromosome, Chromosome]:
#         """执行单点交叉"""
#         offspring1 = parent1.clone()
#         offspring2 = parent2.clone()
#
#         # 随机选择交叉点（避开固定时间任务）
#         variable_tasks = self._get_variable_tasks(parent1, data)
#         if len(variable_tasks) < 2:
#             return offspring1, offspring2
#
#         point = random.choice(variable_tasks[:-1])
#
#         # 交换交叉点后的基因
#         for task_id in variable_tasks:
#             if task_id > point:
#                 offspring1.genes[task_id] = parent2.genes.get(task_id, offspring1.genes.get(task_id))
#                 offspring2.genes[task_id] = parent1.genes.get(task_id, offspring2.genes.get(task_id))
#
#         # 修复合班约束
#         self._fix_combined_classes(offspring1, data)
#         self._fix_combined_classes(offspring2, data)
#
#         return offspring1, offspring2
#
#     def _get_variable_tasks(self, chromosome: Chromosome, data: ScheduleData) -> List[int]:
#         """获取可变任务列表（非固定时间）"""
#         if data is None:
#             return list(chromosome.genes.keys())
#         return [tid for tid in data.tasks.keys()
#                 if tid not in data.fixed_time_tasks]
#
#     def _fix_combined_classes(self, chromosome: Chromosome, data: ScheduleData):
#         """修复合班约束 - 确保同一合班的所有任务在同一时间同一教室"""
#         for course_class_id, task_ids in data.combined_classes.items():
#             if len(task_ids) <= 1:
#                 continue
#
#             # 获取第一个任务的排课
#             ref_slot, ref_room = chromosome.get_gene(task_ids[0])
#
#             # 同步其他任务
#             for task_id in task_ids[1:]:
#                 # 固定任务不修改
#                 if task_id in data.fixed_time_tasks:
#                     continue
#                 chromosome.set_gene(task_id, ref_slot, ref_room)
#
#
# class TwoPointCrossover(CrossoverOperator):
#     """两点交叉"""
#
#     def crossover(self, parent1: Chromosome, parent2: Chromosome,
#                  data: ScheduleData) -> Tuple[Chromosome, Chromosome]:
#         """执行两点交叉"""
#         offspring1 = parent1.clone()
#         offspring2 = parent2.clone()
#
#         # 选择两个交叉点
#         variable_tasks = self._get_variable_tasks(parent1, data)
#         if len(variable_tasks) < 2:
#             return offspring1, offspring2
#
#         points = sorted(random.sample(variable_tasks, 2))
#
#         # 交换两个交叉点之间的基因
#         for task_id in variable_tasks:
#             if points[0] <= task_id <= points[1]:
#                 offspring1.genes[task_id], offspring2.genes[task_id] = \
#                     parent2.genes.get(task_id, offspring1.genes.get(task_id)), \
#                     parent1.genes.get(task_id, offspring2.genes.get(task_id))
#
#         # 修复合班约束
#         self._fix_combined_classes(offspring1, data)
#         self._fix_combined_classes(offspring2, data)
#
#         return offspring1, offspring2
#
#     def _get_variable_tasks(self, chromosome: Chromosome, data: ScheduleData) -> List[int]:
#         """获取可变任务列表"""
#         if data is None:
#             return list(chromosome.genes.keys())
#         return [tid for tid in data.tasks.keys()
#                 if tid not in data.fixed_time_tasks]
#
#     def _fix_combined_classes(self, chromosome: Chromosome, data: ScheduleData):
#         """修复合班约束"""
#         for course_class_id, task_ids in data.combined_classes.items():
#             if len(task_ids) <= 1:
#                 continue
#
#             ref_slot, ref_room = chromosome.get_gene(task_ids[0])
#
#             for task_id in task_ids[1:]:
#                 if task_id in data.fixed_time_tasks:
#                     continue
#                 chromosome.set_gene(task_id, ref_slot, ref_room)
#
#
# class ConstraintAwareCrossover(CrossoverOperator):
#     """
#     约束感知交叉 - 考虑硬约束的智能交叉
#
#     特点:
#     - 保护固定时间任务
#     - 保护合班约束
#     - 交叉后修复冲突
#     """
#
#     def __init__(self, crossover_rate: float = 0.8):
#         self.crossover_rate = crossover_rate
#
#     def crossover(self, parent1: Chromosome, parent2: Chromosome,
#                  data: ScheduleData) -> Tuple[Chromosome, Chromosome]:
#         """执行约束感知交叉"""
#         # 决定是否交叉
#         if random.random() > self.crossover_rate:
#             return parent1.clone(), parent2.clone()
#
#         offspring1 = parent1.clone()
#         offspring2 = parent2.clone()
#
#         # 获取所有任务ID
#         all_task_ids = set(parent1.genes.keys()) | set(parent2.genes.keys())
#
#         for task_id in all_task_ids:
#             # 固定任务不交叉
#             if data and task_id in data.fixed_time_tasks:
#                 continue
#
#             if random.random() < 0.5:
#                 # 交换基因
#                 gene1 = parent1.genes.get(task_id)
#                 gene2 = parent2.genes.get(task_id)
#                 if gene1 is not None:
#                     offspring1.genes[task_id] = gene2
#                 if gene2 is not None:
#                     offspring2.genes[task_id] = gene1
#
#         # 后处理
#         self._post_process(offspring1, data)
#         self._post_process(offspring2, data)
#
#         return offspring1, offspring2
#
#     def _get_variable_tasks(self, data: ScheduleData) -> List[int]:
#         """获取可变任务列表"""
#         if data is None:
#             return list(range(545))
#         return [tid for tid in data.tasks.keys()
#                 if tid not in data.fixed_time_tasks]
#
#     def _post_process(self, chromosome: Chromosome, data: ScheduleData):
#         """后处理：修复约束"""
#         if data is None:
#             return
#
#         # 1. 修复合班约束
#         self._fix_combined_classes(chromosome, data)
#
#     def _fix_combined_classes(self, chromosome: Chromosome, data: ScheduleData):
#         """修复合班约束"""
#         for course_class_id, task_ids in data.combined_classes.items():
#             if len(task_ids) <= 1:
#                 continue
#
#             # 找到参考任务（非固定时间的）
#             ref_task = None
#             for tid in task_ids:
#                 if tid not in data.fixed_time_tasks:
#                     ref_task = tid
#                     break
#
#             if ref_task is None:
#                 ref_task = task_ids[0]
#
#             ref_slot, ref_room = chromosome.get_gene(ref_task)
#
#             # 同步其他任务
#             for task_id in task_ids:
#                 if task_id == ref_task:
#                     continue
#                 if task_id in data.fixed_time_tasks:
#                     continue
#                 chromosome.set_gene(task_id, ref_slot, ref_room)
