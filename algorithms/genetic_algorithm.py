# # # # """
# # # # 遗传算法 (Early Stopping + Loss Log + Detailed Console)
# # # # 遗传算法 (Enhanced Tabu Integration)
# # # # 文件名: algorithms/genetic_algorithm.py
# # # # """
# # # """
# # # 遗传算法 (Step 1: 修正起跑线)
# # # 文件名: algorithms/genetic_algorithm.py
# # # 修改核心:
# # # 将 _init_population 中的贪心比例 (n_heu) 从 50% 提升到 90%-100%。
# # # 这能让您在 Gen 0 就拿到和 IFTS 一样高的分数 (约 14000+)。
# # # """
# # """
# # 遗传算法 (Final Perfect Version - Logic Aligned)
# # 文件名: algorithms/genetic_algorithm.py
# # 功能:
# # 1. 真实计时: 记录每一代的 time_s。
# # 2. 软分必算: 每一代计算 soft_score_sum。
# # 3. 关键帧采样: 每 metric_interval 代及早停时刻计算复杂指标。
# # 4. [重点] 快照记录: 当找到历史最优解时，同步记录那一刻的 Avg_Fitness 和 Soft_Score_Sum。
# # """
# #
# # """
# # 功能:
# # 1. 真实计时: 记录每一代的 time_s。
# # 2. 软分必算: 每一代计算 soft_score_sum。
# # 3. 关键帧采样: 每 metric_interval 代及早停时刻计算复杂指标。
# # 4. 快照记录: 当找到历史最优解时，同步记录那一刻的 Avg_Fitness 和 Soft_Score_Sum。
# # """
# # """
# # 遗传算法 (Final Logic: Hard Constraints First)
# # 文件名: algorithms/genetic_algorithm.py
# # 核心修正:
# # 1. [模型对齐] 只有当硬约束冲突(Violations)为0时，才计算软约束得分。否则 Soft_Score_Sum 强制为 0。
# # 2. [快照修正] 更新 best_stats 时，同样遵循上述逻辑。
# # """
# # """
# # 1. 记录 f_daily, f_interval, f_room, f_util, f_build 原始分，用于画堆叠图。
# # 2. 保持硬约束优先逻辑。
# # 已包含所有的开关注入、Gen0 净化和 Fitness 重评估防崩）：
# # """
"""
DRL引导的混合文化算法 (DRL-Guided Block-based Memetic Algorithm)
文件名: algorithms/genetic_algorithm.py
"""
import time
import random
import numpy as np
from models.chromosome import Chromosome
from algorithms.selection import RouletteWheelSelection
from algorithms.crossover import UniformCrossover
from algorithms.mutation import UniformMutation
from algorithms.tabu_search import TabuSearch
from algorithms.constraint_solver import generate_feasible_solution

from algorithms.combined_operators import CombinedOperatorsHelper, CombinedBlockCrossover
from algorithms.combined_cdm import TrueCombinedConflictDirectedMutation
from algorithms.combined_tabu import CombinedTabuSearch
from algorithms.combined_repair import CombinedRepair

try:
    from algorithms.rl_controller import RLController
except ImportError:
    RLController = None


class GeneticAlgorithm:
    def __init__(self, data, evaluator, config):
        self.data = data
        self.evaluator = evaluator
        self.config = config

        self.use_heuristic = config.get('use_heuristic_init', False)
        self.use_rl = config.get('use_rl_adaptive', False)
        self.use_tabu = config.get('use_tabu', False)

        self.use_block = config.get('use_block_operators', False)
        self.use_repair = config.get('use_repair', False)
        self.use_cdm = config.get('use_cdm', False)

        self.metric_interval = config.get('metric_interval', 50)
        self.patience = config.get('patience', 50)
        self.min_delta = 1e-4

        self.pc = config.get('crossover_rate', 0.8)
        self.pm = config.get('mutation_rate', 0.1)

        self.selection = RouletteWheelSelection()

        # 严格的算子路由 (Block mode vs Standard mode)
        if self.use_block:
            self.helper = CombinedOperatorsHelper(data)
            self.repair = CombinedRepair(self.helper)
            self.crossover = CombinedBlockCrossover(self.pc, self.helper, enable_repair=False)
            if self.use_cdm:
                self.mutation = TrueCombinedConflictDirectedMutation(self.pm, self.helper, data, evaluator, tries=8)
            if self.use_tabu:
                self.tabu_search = CombinedTabuSearch(evaluator, self.helper)
        else:
            self.crossover = UniformCrossover(self.pc)
            self.mutation = UniformMutation(self.pm, data)
            if self.use_tabu:
                self.tabu_search = TabuSearch(data, evaluator)

        # 安全加载 RL，优雅降级
        if self.use_rl and RLController:
            self.rl_controller = RLController()
        else:
            self.rl_controller = None

        self.population = []
        self.best_chromosome = None

        self.best_stats = {
            'avg_fitness': 0.0,
            'soft_score_sum': 0.0,
            'f_daily': 0.0, 'f_interval': 0.0, 'f_room': 0.0, 'f_util': 0.0, 'f_build': 0.0
        }

    def _calc_soft_score_safe(self, genes):
        violations = self.evaluator.constraint_checker.check_all(genes)
        if violations['total'] > 0:
            zero_raw = {'f_daily': 0.0, 'f_interval': 0.0, 'f_room': 0.0, 'f_util': 0.0, 'f_build': 0.0}
            return 0.0, violations, zero_raw
        else:
            raw = self.evaluator._calculate_raw_scores(genes)
            weighted_sum = sum(raw[k] * self.evaluator.weights.get(k, 0) for k in raw)
            return weighted_sum, violations, raw

    def run(self, max_generations, pop_size, run_id):
        self._init_population(pop_size)
        self._eval_population()
        self.best_chromosome = max(self.population, key=lambda x: x.fitness)

        # 关键修复：每次独立 run 开始前重置 RL episode 状态，防止跨 run 串扰
        if self.use_rl and self.rl_controller:
            self.rl_controller.reset_episode()

        soft_init, _, raw_init = self._calc_soft_score_safe(self.best_chromosome.genes)
        self.best_stats['avg_fitness'] = np.mean([c.fitness for c in self.population])
        self.best_stats['soft_score_sum'] = soft_init
        self.best_stats.update(raw_init)

        history = []
        rl_log = []
        no_improve_count = 0
        best_fitness_history = self.best_chromosome.fitness

        start_time = time.time()
        print(f"    ▶ Run {run_id} Start (Pop={pop_size}, MaxGen={max_generations}, Patience={self.patience})")

        for gen in range(max_generations):
            rl_info = {}
            current_avg = np.mean([c.fitness for c in self.population])

            # --- A. RL 自适应参数控制 ---
            if self.use_rl and self.rl_controller:
                state = {
                    'diversity': self._calculate_diversity(self.population),
                    'avg_fit': current_avg,
                    'best_fit': self.best_chromosome.fitness,
                    'progress': gen / max_generations
                }
                rl_info = self.rl_controller.get_action(state, training=True)

                if rl_info.get('pc') is not None:
                    # ⭐ 最终装甲：RL参数钳制 (Clamp)，防止模型异常导致概率越界崩盘
                    self.pc = max(0.0, min(1.0, float(rl_info['pc'])))
                    self.crossover.rate = self.pc
                if rl_info.get('pm') is not None:
                    # ⭐ 最终装甲：RL参数钳制 (Clamp)
                    self.pm = max(0.0, min(1.0, float(rl_info['pm'])))
                    if hasattr(self, 'mutation') and self.mutation is not None:
                        self.mutation.rate = self.pm

            # --- B. Genetic Evolution ---
            self._evolve_population(pop_size)

            # --- C. 全局最优判定 ---
            current_gen_best = max(self.population, key=lambda x: x.fitness)
            current_avg_fit = np.mean([c.fitness for c in self.population])

            if current_gen_best.fitness > self.best_chromosome.fitness:
                self.best_chromosome = current_gen_best.clone()
                soft_best, _, raw_best = self._calc_soft_score_safe(self.best_chromosome.genes)
                self.best_stats['avg_fitness'] = current_avg_fit
                self.best_stats['soft_score_sum'] = soft_best
                self.best_stats.update(raw_best)

            # --- D. Memetic Local Search (Unit-based Tabu) ---
            # 仅在硬约束消除(fitness > 0)后启动 Tabu，极大节省无效算力
            if self.use_tabu and self.best_chromosome.fitness > 0:
                if self.use_block:
                    improved_best = self.best_chromosome.clone()
                    improved_best = self.tabu_search.run(improved_best, max_steps=5)
                    if improved_best.fitness > self.best_chromosome.fitness:
                        self.best_chromosome = improved_best
                        no_improve_count = 0
                        soft_tabu, _, raw_tabu = self._calc_soft_score_safe(self.best_chromosome.genes)
                        self.best_stats['soft_score_sum'] = soft_tabu
                        self.best_stats['avg_fitness'] = current_avg_fit
                        self.best_stats.update(raw_tabu)
                else:
                    need_repair = (self.best_chromosome.fitness < 0)
                    improved_best = self.tabu_search.run(self.best_chromosome, self.config.get('tabu_steps', 10),
                                                         need_repair)
                    if improved_best.fitness > self.best_chromosome.fitness:
                        self.best_chromosome = improved_best
                        no_improve_count = 0
                        soft_tabu, _, raw_tabu = self._calc_soft_score_safe(self.best_chromosome.genes)
                        self.best_stats['soft_score_sum'] = soft_tabu
                        self.best_stats['avg_fitness'] = current_avg_fit
                        self.best_stats.update(raw_tabu)

            # --- E. 早停检查 ---
            stop_triggered = False
            if self.best_chromosome.fitness > best_fitness_history + self.min_delta:
                best_fitness_history = self.best_chromosome.fitness
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count >= self.patience:
                    stop_triggered = True

            # --- F. 数据记录与日志 ---
            current_time = time.time() - start_time
            curr_soft_sum, violations, curr_raw_scores = self._calc_soft_score_safe(self.best_chromosome.genes)

            is_key_frame = (gen % self.metric_interval == 0) or (gen == max_generations - 1) or stop_triggered
            if is_key_frame:
                detailed_metrics = self.evaluator.calculate_quality_metrics(self.best_chromosome.genes) if violations[
                                                                                                               'total'] == 0 else {
                    'Daily_Var': 0.0, 'Util_Avg': 0.0, 'Interval_Rate': 0.0, 'Build_Conc_Rate': 0.0}
            else:
                detailed_metrics = {'Daily_Var': 0.0, 'Util_Avg': 0.0, 'Interval_Rate': 0.0, 'Build_Conc_Rate': 0.0}

            record = {
                'gen': gen, 'best_fitness': self.best_chromosome.fitness, 'avg_fitness': current_avg_fit,
                'hard_violations': violations['total'], 'soft_score_sum': curr_soft_sum, 'time_s': current_time,
                **curr_raw_scores, **detailed_metrics
            }
            history.append(record)

            if gen % 20 == 0 or stop_triggered:
                log_str = (f"      [Gen {gen:3d}] Best: {self.best_chromosome.fitness:10.2f} | "
                           f"Soft: {curr_soft_sum:.1f} | Daily: {curr_raw_scores.get('f_daily', 0):.1f} | "
                           f"Wait: {no_improve_count}/{self.patience}")
                if self.use_rl and rl_info:
                    log_str += f" | Act: {rl_info.get('action_id')} | Pc: {self.pc:.2f} | Pm: {self.pm:.2f}"
                print(log_str)

            if self.use_rl and rl_info:
                rl_log.append({
                    'gen': gen, 'action': rl_info.get('action_id'), 'reward': rl_info.get('reward', 0),
                    'loss': rl_info.get('loss'), 'mutation': self.pm, 'crossover': self.pc,
                    'q_values': rl_info.get('q_values')
                })

            if stop_triggered:
                print(f"      ⏹️ Early Stopping at Gen {gen}")
                break

        return self.best_chromosome, history, rl_log, self.best_stats

    def _init_population(self, pop_size):
        self.population = []
        n_heu = int(pop_size * 0.9) if self.use_heuristic else 0
        for _ in range(n_heu):
            genes = generate_feasible_solution(self.data)
            c = Chromosome(self.data)
            c.genes = genes
            if self.use_block and self.use_repair:
                self.repair.repair(c)
            self.population.append(c)
        for _ in range(pop_size - n_heu):
            c = Chromosome(self.data)
            c.initialize()
            if self.use_block and self.use_repair:
                self.repair.repair(c)
            self.population.append(c)

    def _eval_population(self):
        for c in self.population:
            if c.fitness is None:
                c.fitness = self.evaluator.evaluate(c.genes)

    def _evolve_population(self, pop_size):
        n_elites = self.config.get('elite_count', 2)
        elites = sorted(self.population, key=lambda x: x.fitness, reverse=True)[:n_elites]
        new_pop = [c.clone() for c in elites]

        while len(new_pop) < pop_size:
            p1 = self.selection.select(self.population)
            p2 = self.selection.select(self.population)

            if self.use_block:
                c1, c2 = self.crossover.cross(p1, p2)
                if self.use_cdm:
                    self.mutation.mutate(c1)
                    self.mutation.mutate(c2)
                if self.use_repair:
                    self.repair.repair(c1)
                    self.repair.repair(c2)
            else:
                c1, c2 = self.crossover.cross(p1, p2)
                if hasattr(self, 'mutation') and self.mutation is not None:
                    self.mutation.mutate(c1)
                    self.mutation.mutate(c2)

            new_pop.extend([c1, c2])

        self.population = new_pop[:pop_size]
        self._eval_population()

    def _calculate_diversity(self, population):
        sample_size = min(len(population), 20)
        samples = random.sample(population, sample_size)
        hashes = set()
        for ind in samples:
            hashes.add(hash(tuple(ind.genes.values())))
        return len(hashes) / sample_size







# import time
# import numpy as np
# from models.chromosome import Chromosome
# from algorithms.selection import RouletteWheelSelection
# from algorithms.crossover import UniformCrossover
# from algorithms.mutation import UniformMutation
# from algorithms.tabu_search import TabuSearch
# from algorithms.rl_controller import RLController
# from algorithms.constraint_solver import generate_feasible_solution
#
#
# class GeneticAlgorithm:
#     def __init__(self, data, evaluator, config):
#         self.data = data
#         self.evaluator = evaluator
#         self.config = config
#
#         self.selection = RouletteWheelSelection()
#         self.crossover = UniformCrossover(config['crossover_rate'])
#         self.mutation = UniformMutation(config['mutation_rate'], data)
#
#         self.use_heuristic = config.get('use_heuristic_init', False)
#         self.use_rl = config.get('use_rl_adaptive', False)
#         self.use_tabu = config.get('use_tabu', False)
#
#         self.metric_interval = config.get('metric_interval', 50)
#         self.patience = config.get('patience', 50)
#         self.min_delta = 1e-4
#
#         if self.use_rl:
#             self.rl_controller = RLController()
#         if self.use_tabu:
#             self.tabu_search = TabuSearch(data, evaluator)
#
#         self.population = []
#         self.best_chromosome = None
#
#         # [快照] 记录最优解"诞生时刻"的统计数据
#         self.best_stats = {
#             'avg_fitness': 0.0,
#             'soft_score_sum': 0.0,
#             # 详细分快照
#             'f_daily': 0.0, 'f_interval': 0.0, 'f_room': 0.0, 'f_util': 0.0, 'f_build': 0.0
#         }
#
#     def _calc_soft_score_safe(self, genes):
#         """
#         [辅助函数] 安全计算软分及详细子项：
#         只有当硬约束冲突为 0 时，才计算软分；否则全返回 0.0。
#         """
#         violations = self.evaluator.constraint_checker.check_all(genes)
#         if violations['total'] > 0:
#             zero_raw = {'f_daily': 0.0, 'f_interval': 0.0, 'f_room': 0.0, 'f_util': 0.0, 'f_build': 0.0}
#             return 0.0, violations, zero_raw
#         else:
#             raw = self.evaluator._calculate_raw_scores(genes)
#             # 计算加权总分 (Soft Score Sum)
#             # 注意: 这里我们假设 Soft_Score_Sum 是所有加权子项的和
#             weighted_sum = sum(raw[k] * self.evaluator.weights.get(k, 0) for k in raw)
#             return weighted_sum, violations, raw
#
#     def run(self, max_generations, pop_size, run_id):
#         # 1. 初始化
#         self._init_population(pop_size)
#         self._eval_population()
#         self.best_chromosome = max(self.population, key=lambda x: x.fitness)
#
#         # 初始化快照
#         soft_init, _, raw_init = self._calc_soft_score_safe(self.best_chromosome.genes)
#         self.best_stats['avg_fitness'] = np.mean([c.fitness for c in self.population])
#         self.best_stats['soft_score_sum'] = soft_init
#         self.best_stats.update(raw_init) # 更新 f_daily 等
#
#         history = []
#         rl_log = []
#         no_improve_count = 0
#         best_fitness_history = self.best_chromosome.fitness
#
#         start_time = time.time()
#
#         print(
#             f"    ▶ Run {run_id} Start (Pop={pop_size}, MaxGen={max_generations}, Patience={self.patience})")
#
#         for gen in range(max_generations):
#             rl_info = {}
#             curr_pc = self.crossover.rate
#             curr_pm = self.mutation.rate
#
#             # --- A. RL 自适应 ---
#             if self.use_rl:
#                 fits = [c.fitness for c in self.population]
#                 state = {
#                     'diversity': np.std(fits),
#                     'avg_fit': np.mean(fits),
#                     'best_fit': self.best_chromosome.fitness,
#                     'progress': gen / max_generations
#                 }
#                 rl_info = self.rl_controller.get_action(state, training=True)
#                 if rl_info['pc'] is not None:
#                     self.crossover.rate = rl_info['pc']
#                     self.mutation.rate = rl_info['pm']
#                     curr_pc = rl_info['pc']
#                     curr_pm = rl_info['pm']
#
#             # --- B. 进化操作 ---
#             self._evolve_population(pop_size)
#             current_gen_best = max(self.population, key=lambda x: x.fitness)
#             current_avg_fit = np.mean([c.fitness for c in self.population])
#
#             # 检查是否更新全局最优
#             if current_gen_best.fitness > self.best_chromosome.fitness:
#                 self.best_chromosome = current_gen_best
#
#                 # [修正] 更新快照
#                 soft_best, _, raw_best = self._calc_soft_score_safe(self.best_chromosome.genes)
#                 self.best_stats['avg_fitness'] = current_avg_fit
#                 self.best_stats['soft_score_sum'] = soft_best
#                 self.best_stats.update(raw_best)
#
#             # --- C. Tabu 优化 ---
#             if self.use_tabu:
#                 need_repair = (self.best_chromosome.fitness < 0)
#                 improved_best = self.tabu_search.run(self.best_chromosome, self.config.get('tabu_steps', 10),
#                                                      need_repair)
#                 if improved_best.fitness > self.best_chromosome.fitness:
#                     self.best_chromosome = improved_best
#                     no_improve_count = 0
#
#                     # [修正] Tabu 更新后更新快照
#                     soft_tabu, _, raw_tabu = self._calc_soft_score_safe(self.best_chromosome.genes)
#                     self.best_stats['soft_score_sum'] = soft_tabu
#                     self.best_stats['avg_fitness'] = current_avg_fit
#                     self.best_stats.update(raw_tabu)
#
#             # --- D. 早停检查 ---
#             stop_triggered = False
#             if self.best_chromosome.fitness > best_fitness_history + self.min_delta:
#                 best_fitness_history = self.best_chromosome.fitness
#                 no_improve_count = 0
#             else:
#                 no_improve_count += 1
#                 if no_improve_count >= self.patience:
#                     stop_triggered = True
#
#             # --- E. 数据记录 (History) ---
#             current_time = time.time() - start_time
#
#             # 实时计算当前最优解的软分详情
#             curr_soft_sum, violations, curr_raw_scores = self._calc_soft_score_safe(self.best_chromosome.genes)
#
#             # 复杂指标采样
#             is_key_frame = (gen % self.metric_interval == 0) or (gen == max_generations - 1) or stop_triggered
#             detailed_metrics = {}
#             if is_key_frame:
#                 if violations['total'] == 0:
#                     detailed_metrics = self.evaluator.calculate_quality_metrics(self.best_chromosome.genes)
#                 else:
#                     detailed_metrics = {'Daily_Var': 0.0, 'Util_Avg': 0.0, 'Interval_Rate': 0.0, 'Build_Conc_Rate': 0.0}
#             else:
#                 detailed_metrics = {'Daily_Var': 0.0, 'Util_Avg': 0.0, 'Interval_Rate': 0.0, 'Build_Conc_Rate': 0.0}
#
#             # 写入 History
#             record = {
#                 'gen': gen,
#                 'best_fitness': self.best_chromosome.fitness,
#                 'avg_fitness': current_avg_fit,
#                 'hard_violations': violations['total'],
#                 'soft_score_sum': curr_soft_sum,
#                 'time_s': current_time,
#                 # 展开详细分数 (f_daily, f_interval, ...)
#                 **curr_raw_scores,
#                 # 展开复杂指标 (Daily_Var, ...)
#                 **detailed_metrics
#             }
#             history.append(record)
#
#             # --- F. 日志输出 ---
#             if gen % 20 == 0 or stop_triggered:
#                 log_str = (f"      [Gen {gen:3d}] Best: {self.best_chromosome.fitness:10.2f} | "
#                            f"Soft: {curr_soft_sum:.1f} | "
#                            f"Daily: {curr_raw_scores.get('f_daily',0):.1f} | " # 简单看一个子项
#                            f"Wait: {no_improve_count}/{self.patience}")
#                 if self.use_rl:
#                     log_str += f" | Act: {rl_info.get('action_id')}"
#                 print(log_str)
#
#             if self.use_rl and rl_info:
#                 rl_log.append({
#                     'gen': gen, 'action': rl_info['action_id'], 'reward': rl_info.get('reward', 0),
#                     'loss': rl_info.get('loss'), 'mutation': curr_pm, 'crossover': curr_pc,
#                     'q_values': rl_info['q_values']
#                 })
#
#             if stop_triggered:
#                 print(f"      ⏹️ Early Stopping at Gen {gen}")
#                 break
#
#         return self.best_chromosome, history, rl_log, self.best_stats
#
#     def _init_population(self, pop_size):
#         self.population = []
#         n_heu = int(pop_size * 0.9) if self.use_heuristic else 0
#         for _ in range(n_heu):
#             genes = generate_feasible_solution(self.data)
#             c = Chromosome(self.data)
#             c.genes = genes
#             self.population.append(c)
#         for _ in range(pop_size - n_heu):
#             c = Chromosome(self.data)
#             c.initialize()
#             self.population.append(c)
#
#     def _eval_population(self):
#         for c in self.population:
#             if c.fitness is None: c.fitness = self.evaluator.evaluate(c.genes)
#
#     def _evolve_population(self, pop_size):
#         n_elites = self.config.get('elite_count', 2)
#         elites = sorted(self.population, key=lambda x: x.fitness, reverse=True)[:n_elites]
#         new_pop = list(elites)
#         while len(new_pop) < pop_size:
#             p1 = self.selection.select(self.population)
#             p2 = self.selection.select(self.population)
#             c1, c2 = self.crossover.cross(p1, p2)
#             self.mutation.mutate(c1)
#             self.mutation.mutate(c2)
#             new_pop.extend([c1, c2])
#         self.population = new_pop[:pop_size]
#         self._eval_population()
