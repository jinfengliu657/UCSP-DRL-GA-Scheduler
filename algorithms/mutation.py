"""
变异算子 (Mutation)
文件名: algorithms/mutation.py
"""
import random
from models.chromosome import Chromosome


class UniformMutation:
    def __init__(self, mutation_rate, data):
        self.rate = mutation_rate
        self.data = data

    def mutate(self, chromosome: Chromosome):
        """对个体进行均匀变异"""
        if random.random() > self.rate:
            return

        # 随机选择一个任务进行基因重置
        task_ids = list(chromosome.genes.keys())
        if not task_ids:
            return

        target_task_id = random.choice(task_ids)

        # 生成新基因 (随机时间槽 0-24, 随机教室)
        new_gene = self._random_gene()

        chromosome.genes[target_task_id] = new_gene
        chromosome.fitness = None  # 标记适应度失效，需重算

    def _random_gene(self):
        # 1. 随机时间槽 (0-24)
        slot = random.randint(0, 24)

        # 2. 随机教室
        # 这里的 keys 可能是 str，需要处理
        room_keys = list(self.data.classrooms.keys())
        if not room_keys:
            room_id = 0
        else:
            r_key = random.choice(room_keys)
            # 尝试转 int，转不了就 Hash (为了存入 int 类型的 gene)
            try:
                room_id = int(r_key)
            except ValueError:
                room_id = hash(r_key) & 0xFFFF

        # 3. 组合
        return (slot << 16) | room_id

# """
# 变异算子 - 智能变异操作
# """
# import random
# from typing import List
#
# from models.chromosome import Chromosome
# from data.data_loader import ScheduleData
#
#
# class MutationOperator:
#     """变异算子基类"""
#
#     def mutate(self, chromosome: Chromosome, data: ScheduleData) -> Chromosome:
#         """执行变异操作"""
#         raise NotImplementedError
#
#
# class SingleGeneMutation(MutationOperator):
#     """单基因变异 - 随机改变一个基因"""
#
#     def __init__(self, mutation_rate: float = 0.1):
#         """
#         初始化单基因变异
#
#         Args:
#             mutation_rate: 变异概率
#         """
#         self.mutation_rate = mutation_rate
#
#     def mutate(self, chromosome: Chromosome, data: ScheduleData) -> Chromosome:
#         """执行单基因变异"""
#         offspring = chromosome.clone()
#
#         if random.random() < self.mutation_rate:
#             # 随机选择一个可变任务
#             variable_tasks = self._get_variable_tasks(data)
#             if not variable_tasks:
#                 return offspring
#
#             task_id = random.choice(variable_tasks)
#             task = data.tasks[task_id]
#
#             # 新时间槽
#             new_slot = random.randint(0, 29)
#
#             # 新教室
#             available_rooms = self._get_suitable_rooms(task, data)
#             new_room = random.choice(available_rooms) if available_rooms else 1
#
#             offspring.set_gene(task_id, new_slot, new_room)
#
#         return offspring
#
#     def _get_variable_tasks(self, data: ScheduleData) -> List[int]:
#         """获取可变任务列表"""
#         if data is None:
#             return list(range(545))
#         return [tid for tid in data.tasks.keys()
#                 if tid not in data.fixed_time_tasks]
#
#     def _get_suitable_rooms(self, task, data: ScheduleData) -> List[int]:
#         """获取合适的教室列表"""
#         if data is None:
#             return list(range(100))
#
#         suitable = []
#         for room_id, room in data.classrooms.items():
#             # 检查特殊教室要求
#             if task.course_attr == '07' and room.attr != '07':
#                 continue
#             # 检查容量
#             if room.capacity >= task.student_count:
#                 suitable.append(room_id)
#         return suitable if suitable else list(data.classrooms.keys())
#
#
# class SwapMutation(MutationOperator):
#     """交换变异 - 交换两个任务的基因"""
#
#     def __init__(self, mutation_rate: float = 0.1):
#         self.mutation_rate = mutation_rate
#
#     def mutate(self, chromosome: Chromosome, data: ScheduleData) -> Chromosome:
#         """执行交换变异"""
#         offspring = chromosome.clone()
#
#         if random.random() < self.mutation_rate:
#             variable_tasks = self._get_variable_tasks(data)
#             if len(variable_tasks) < 2:
#                 return offspring
#
#             # 选择两个任务
#             task1, task2 = random.sample(variable_tasks, 2)
#
#             # 交换它们的时间槽
#             slot1, room1 = offspring.get_gene(task1)
#             slot2, room2 = offspring.get_gene(task2)
#
#             offspring.set_gene(task1, slot2, room2)
#             offspring.set_gene(task2, slot1, room1)
#
#         return offspring
#
#     def _get_variable_tasks(self, data: ScheduleData) -> List[int]:
#         if data is None:
#             return list(range(545))
#         return [tid for tid in data.tasks.keys()
#                 if tid not in data.fixed_time_tasks]
#
#
# class CombinedClassMutation(MutationOperator):
#     """合班变异 - 同时变异整个合班"""
#
#     def __init__(self, mutation_rate: float = 0.05):
#         self.mutation_rate = mutation_rate
#
#     def mutate(self, chromosome: Chromosome, data: ScheduleData) -> Chromosome:
#         """执行合班变异"""
#         offspring = chromosome.clone()
#
#         if data is None or random.random() >= self.mutation_rate:
#             return offspring
#
#         # 随机选择一个合班
#         if not data.combined_classes:
#             return offspring
#
#         course_class_id = random.choice(list(data.combined_classes.keys()))
#         task_ids = data.combined_classes[course_class_id]
#
#         if len(task_ids) <= 1:
#             return offspring
#
#         # 找到非固定时间的任务
#         variable_tasks = [tid for tid in task_ids if tid not in data.fixed_time_tasks]
#         if not variable_tasks:
#             return offspring
#
#         # 计算总人数
#         total_students = sum(data.tasks[tid].student_count for tid in task_ids)
#
#         # 新时间槽
#         new_slot = random.randint(0, 29)
#
#         # 找容量足够的教室
#         suitable_rooms = []
#         for room_id, room in data.classrooms.items():
#             if room.capacity >= total_students:
#                 suitable_rooms.append(room_id)
#
#         new_room = random.choice(suitable_rooms) if suitable_rooms else 1
#
#         # 更新所有可变任务
#         for task_id in variable_tasks:
#             offspring.set_gene(task_id, new_slot, new_room)
#
#         return offspring
#
#
# class IntelligentMutation(MutationOperator):
#     """
#     智能变异 - 自适应选择变异策略
#
#     根据当前解的质量选择变异策略：
#     - 高质量解: 使用小幅度变异
#     - 低质量解: 使用大幅度变异
#     """
#
#     def __init__(self, mutation_rate: float = 0.2):
#         self.mutation_rate = mutation_rate
#         self.strategies = [
#             ('single', SingleGeneMutation(1.0)),  # 总是变异
#             ('swap', SwapMutation(1.0)),
#             ('combined', CombinedClassMutation(1.0)),
#         ]
#
#     def mutate(self, chromosome: Chromosome, data: ScheduleData) -> Chromosome:
#         """执行智能变异"""
#         offspring = chromosome.clone()
#
#         if random.random() >= self.mutation_rate:
#             return offspring
#
#         # 根据适应度选择变异强度
#         if chromosome.fitness < 0:
#             # 低质量解：使用更多变异
#             num_mutations = random.randint(2, 5)
#         else:
#             # 高质量解：使用少量变异
#             num_mutations = random.randint(1, 2)
#
#         for _ in range(num_mutations):
#             # 随机选择变异策略
#             strategy_name, strategy = random.choice(self.strategies)
#             offspring = strategy.mutate(offspring, data)
#
#         return offspring
