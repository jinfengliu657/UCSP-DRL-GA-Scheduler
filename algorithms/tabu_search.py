# """
# 增强型禁忌搜索 (Enhanced Tabu Search)
# 文件名: algorithms/tabu_search.py
# 改进点:
#     增加了 'repair' 模式，专门针对硬约束违约进行邻域搜索。
#     只有在无硬约束违约时，才进行随机交换以优化软约束。
# """
"""
修改说明:
1. 将 neighbor_size 回滚至 5 (速度优先)。
2. 保留 Smart Repair 逻辑 (Swap/Repair 自动切换)。
"""
import random
import copy
from collections import deque


class TabuSearch:
    def __init__(self, data, fitness_evaluator, tabu_size=50):
        self.data = data
        self.evaluator = fitness_evaluator
        self.tabu_list = deque(maxlen=tabu_size)

    def run(self, chromosome, max_steps=10, repair_mode=False):  # 默认步数设小
        best_chromo = chromosome
        if best_chromo.fitness is None:
            best_chromo.fitness = self.evaluator.evaluate(best_chromo.genes)
        best_fitness = best_chromo.fitness

        current_genes = chromosome.genes.copy()

        # 冲突任务列表 (仅在修复模式下使用)
        conflict_tasks = []
        if repair_mode:
            all_tasks = list(current_genes.keys())
            conflict_tasks = random.sample(all_tasks, k=min(len(all_tasks), 30))

        # [修正] 回滚至小邻域，保证速度
        neighbor_size = 5

        for step in range(max_steps):
            candidates = []

            for _ in range(neighbor_size):
                neighbor_genes = current_genes.copy()
                move_type = ""

                if repair_mode and conflict_tasks:
                    # Repair: 随机跳跃
                    target_task = random.choice(conflict_tasks)
                    new_gene = self._random_gene()
                    neighbor_genes[target_task] = new_gene
                    move_type = f"Repair_{target_task}_{new_gene}"
                else:
                    # Swap: 交换优化
                    if len(neighbor_genes) >= 2:
                        t1, t2 = random.sample(list(neighbor_genes.keys()), 2)
                        neighbor_genes[t1], neighbor_genes[t2] = neighbor_genes[t2], neighbor_genes[t1]
                        move_type = f"Swap_{t1}_{t2}"

                if move_type:
                    candidates.append((neighbor_genes, move_type))

            # 评估
            step_best_genes = None
            step_best_fit = -float('inf')
            step_move = None

            for genes, move in candidates:
                if move in self.tabu_list: continue
                fit = self.evaluator.evaluate(genes)
                if fit > best_fitness or (fit > step_best_fit):
                    step_best_genes = genes
                    step_best_fit = fit
                    step_move = move

            # 移动
            if step_best_genes:
                current_genes = step_best_genes
                self.tabu_list.append(step_move)
                if step_best_fit > best_fitness:
                    best_fitness = step_best_fit
                    best_chromo.genes = current_genes.copy()
                    best_chromo.fitness = best_fitness

        return best_chromo

    def _random_gene(self):
        slot = random.randint(0, 24)
        room_ids = list(self.data.classrooms.keys())
        if not room_ids: return 0
        r_key = random.choice(room_ids)
        try:
            r_id = int(self.data.classrooms[r_key].classroom_id)
        except:
            r_id = hash(r_key) & 0xFFFF
        return (slot << 16) | r_id




# import random
# import copy
# from collections import deque
#
#
# class TabuSearch:
#     def __init__(self, data, fitness_evaluator, tabu_size=50):
#         self.data = data
#         self.evaluator = fitness_evaluator
#         self.tabu_list = deque(maxlen=tabu_size)
#
#     def run(self, chromosome, max_steps=50, repair_mode=False):
#         """
#         执行禁忌搜索
#         Args:
#             chromosome: 当前个体
#             max_steps: 最大搜索步数
#             repair_mode: 是否为修复模式 (只针对硬约束冲突进行移动)
#         """
#         # 复制当前最优
#         best_chromo = chromosome
#         # 重新计算一次fitness确保准确
#         if best_chromo.fitness is None:
#             best_chromo.fitness = self.evaluator.evaluate(best_chromo.genes)
#         best_fitness = best_chromo.fitness
#
#         current_genes = chromosome.genes.copy()
#
#         # 冲突任务列表 (仅在修复模式下使用)
#         conflict_tasks = []
#         if repair_mode:
#             # 获取具体的冲突信息
#             # 这里调用 constraint_checker 获取违约详情
#             # 为了效率，我们暂时通过随机采样尝试修复，或者如果有 get_conflict_details 接口更好
#             # 简单策略：随机选取 20% 的任务尝试重排，模拟"针对性"
#             all_tasks = list(current_genes.keys())
#             conflict_tasks = random.sample(all_tasks, k=min(len(all_tasks), 30))
#
#         for step in range(max_steps):
#             # 1. 生成邻域 (Move)
#             candidates = []
#
#             # 尝试生成 5 个邻居
#             for _ in range(5):
#                 neighbor_genes = current_genes.copy()
#                 move_type = ""
#
#                 if repair_mode and conflict_tasks:
#                     # --- 修复模式：针对性移动 ---
#                     target_task = random.choice(conflict_tasks)
#                     # 生成一个完全随机的新位置（尝试跳出冲突区）
#                     new_gene = self._random_gene()
#                     neighbor_genes[target_task] = new_gene
#                     move_type = f"Repair_{target_task}_{new_gene}"
#                 else:
#                     # --- 优化模式：Swap ---
#                     if len(neighbor_genes) >= 2:
#                         t1, t2 = random.sample(list(neighbor_genes.keys()), 2)
#                         neighbor_genes[t1], neighbor_genes[t2] = neighbor_genes[t2], neighbor_genes[t1]
#                         move_type = f"Swap_{t1}_{t2}"
#
#                 if move_type:
#                     candidates.append((neighbor_genes, move_type))
#
#             # 2. 评估邻域
#             step_best_genes = None
#             step_best_fit = -float('inf')
#             step_move = None
#
#             for genes, move in candidates:
#                 if move in self.tabu_list:
#                     continue  # 禁忌
#
#                 fit = self.evaluator.evaluate(genes)
#
#                 # 渴望准则 (Aspiration Criteria)
#                 if fit > best_fitness or (fit > step_best_fit):
#                     step_best_genes = genes
#                     step_best_fit = fit
#                     step_move = move
#
#             # 3. 移动
#             if step_best_genes:
#                 current_genes = step_best_genes
#                 self.tabu_list.append(step_move)
#
#                 # 如果找到了历史更优解
#                 if step_best_fit > best_fitness:
#                     best_fitness = step_best_fit
#                     best_chromo.genes = current_genes.copy()
#                     best_chromo.fitness = best_fitness
#             else:
#                 # 没找到更好的，不做操作或随机扰动
#                 pass
#
#         return best_chromo
#
#     def _random_gene(self):
#         """生成一个随机的时间槽和教室基因"""
#         slot = random.randint(0, 24)
#         # 随机选个教室ID
#         room_ids = list(self.data.classrooms.keys())
#         if not room_ids: return 0
#
#         r_key = random.choice(room_ids)
#         # 尝试转回 int ID 用于编码 (假设 key 是 str, 但 gene 需要 int)
#         try:
#             r_id = int(self.data.classrooms[r_key].classroom_id)
#         except:
#             # 如果ID带字母，hash一下或者取数字部分，这里做个简单容错
#             r_id = hash(r_key) & 0xFFFF
#
#         return (slot << 16) | r_id