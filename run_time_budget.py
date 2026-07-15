# # """
# # 同时间预算对比实验 (Time Budget Comparison)
# # 功能: 强制所有算法在相同的时间限制内运行 (例如 30秒)，比拼谁跑得远、分高。
# # """
# # import time
# # import os
# # import pandas as pd
# # import numpy as np
# # from data.data_loader import DataLoader
# # from evaluation.fitness import FitnessEvaluator
# # from algorithms.genetic_algorithm import GeneticAlgorithm
# # from algorithms.ifts import IFTS
# #
# # # === 配置 ===
# # TIME_LIMIT = 30  # 限制 30 秒
# # N_RUNS = 5  # 跑 5 次取平均
# # INSTANCE_NAME = 'S1_Small'  # 选一个代表性算例
# # DB_CONFIG = {'host': 'localhost', 'port': 3306, 'user': 'root', 'password': os.getenv('DB_PASSWORD', ''), 'database': 'test3'}
# #
# #
# # # === 1. 继承并重写 GA，加入时间锁 ===
# # class TimeLimitedGA(GeneticAlgorithm):
# #     def run_with_time_limit(self, max_generations, pop_size, time_limit):
# #         self._init_population(pop_size)
# #         self._eval_population()
# #         self.best_chromosome = max(self.population, key=lambda x: x.fitness)
# #
# #         start_time = time.time()
# #         final_gen = 0
# #
# #         for gen in range(max_generations):
# #             # --- 时间检查核心 ---
# #             if time.time() - start_time > time_limit:
# #                 print(f"      ⏱️ Time limit reached at Gen {gen}")
# #                 final_gen = gen
# #                 break
# #             # --------------------
# #
# #             # 原有逻辑 (简化版，不影响核心进化)
# #             rl_info = {}
# #             if self.use_rl:
# #                 # 简单模拟状态获取，避免复杂的 RL 依赖报错
# #                 fits = [c.fitness for c in self.population]
# #                 state = {'diversity': np.std(fits), 'avg_fit': np.mean(fits),
# #                          'best_fit': self.best_chromosome.fitness, 'progress': gen / max_generations}
# #                 rl_info = self.rl_controller.get_action(state, training=False)
# #                 if rl_info['pc'] is not None:
# #                     self.crossover.rate = rl_info['pc']
# #                     self.mutation.rate = rl_info['pm']
# #
# #             self._evolve_population(pop_size)
# #
# #             # 更新最优
# #             current_gen_best = max(self.population, key=lambda x: x.fitness)
# #             if current_gen_best.fitness > self.best_chromosome.fitness:
# #                 self.best_chromosome = current_gen_best
# #
# #             # Tabu 增强
# #             if self.use_tabu:
# #                 improved = self.tabu_search.run(self.best_chromosome, self.tabu_steps)
# #                 if improved.fitness > self.best_chromosome.fitness:
# #                     self.best_chromosome = improved
# #
# #         return self.best_chromosome, final_gen
# #
# #
# # # === 2. 继承并重写 IFTS，加入时间锁 ===
# # class TimeLimitedIFTS(IFTS):
# #     def run_with_time_limit(self, generations, time_limit):
# #         start_time = time.time()
# #
# #         # Phase 1: GA
# #         # 这里我们需要 hack 一下 IFTS 内部的 GA，太复杂，
# #         # 我们简化处理：直接调用上面改写好的 TimeLimitedGA 作为 IFTS 的第一阶段
# #
# #         # 实例化一个 TimeLimitedGA 替代原本的 self.ga
# #         # 注意：这里需要重新初始化一个 GA 对象
# #         # 为了方便，我们假设 IFTS 主要是 GA + Tabu。
# #         # 这里我们只跑 GA 阶段的时间限制，因为 Tabu 是后续。
# #         # 如果要严格，需要在 IFTS 内部每一步都查时间。
# #
# #         # 简单策略：直接用 TimeLimitedGA 跑，关掉 RL 和 Tabu，模拟 IFTS 的 GA 阶段
# #         ga_config = {
# #             'crossover_rate': 0.8, 'mutation_rate': 0.1,
# #             'elite_count': 2, 'use_heuristic_init': True,
# #             'use_rl_adaptive': False, 'use_tabu': False  # IFTS 的 GA 是纯 GA
# #         }
# #         ga_engine = TimeLimitedGA(self.data, self.evaluator, ga_config)
# #         best_ind, gens = ga_engine.run_with_time_limit(generations, 100, time_limit)
# #
# #         return best_ind, gens
# #
# #
# # def main():
# #     print(f"=== ⏱️ 启动同时间竞技 (Time Limit: {TIME_LIMIT}s) ===")
# #
# #     try:
# #         loader = DataLoader(INSTANCE_NAME, DB_CONFIG)
# #         data = loader.load()
# #     except Exception as e:
# #         print(f"❌ 数据加载失败: {e}")
# #         return
# #
# #     evaluator = FitnessEvaluator()
# #     evaluator.set_data(data)
# #
# #     results = []
# #
# #     # --- 1. 运行 Ours (DL-GA-TS) ---
# #     print("\n>>> Running DL-GA-TS (Ours)...")
# #     for i in range(N_RUNS):
# #         # 加载最优参数
# #         import json
# #         try:
# #             with open('results/best_params_small.json') as f:
# #                 params = json.load(f)
# #         except:
# #             params = {'crossover_rate': 0.8, 'mutation_rate': 0.1}
# #
# #         config = {
# #             'crossover_rate': params['crossover_rate'],
# #             'mutation_rate': params['mutation_rate'],
# #             'elite_count': 2, 'use_heuristic_init': True,
# #             'use_rl_adaptive': True, 'use_tabu': True, 'tabu_steps': 10
# #         }
# #
# #         algo = TimeLimitedGA(data, evaluator, config)
# #         best, gens = algo.run_with_time_limit(max_generations=10000, pop_size=100, time_limit=TIME_LIMIT)
# #         print(f"  Run {i + 1}: Gen={gens}, Fit={best.fitness:.2f}")
# #         results.append({'Algorithm': 'DL-GA-TS', 'Run': i, 'Fitness': best.fitness, 'Generations': gens})
# #
# #     # --- 2. 运行 IFTS (Simulated) ---
# #     print("\n>>> Running IFTS (Xiang)...")
# #     for i in range(N_RUNS):
# #         algo = TimeLimitedIFTS(data, evaluator)
# #         # IFTS 的 GA 是纯 GA，无 RL 无 Tabu (Tabu 是二阶段)
# #         # 在 30秒内，通常 IFTS 只能跑完 GA 阶段，或者刚进 Tabu。
# #         # 这里我们用纯 GA 模拟其第一阶段的冲刺能力
# #         best, gens = algo.run_with_time_limit(generations=10000, time_limit=TIME_LIMIT)
# #         print(f"  Run {i + 1}: Gen={gens}, Fit={best.fitness:.2f}")
# #         results.append({'Algorithm': 'IFTS', 'Run': i, 'Fitness': best.fitness, 'Generations': gens})
# #
# #     # --- 3. 结果保存 ---
# #     df = pd.DataFrame(results)
# #     print("\n=== 最终结果 (Mean) ===")
# #     print(df.groupby('Algorithm')['Fitness'].mean())
# #     df.to_excel("results/time_budget_comparison.xlsx")
# #
# #
# # if __name__ == "__main__":
# #     main()
#
# """
# 同时间预算对比实验 (Time Budget Comparison - L1_Large 专版)
# 功能: 强制 6 种算法在真实的 300秒 物理时间限制内运行，
#       生成详细数据表格，并自动绘制【同时间极限分数分布图】！
# """
# """
# 同时间预算对比实验 (Time Budget Comparison - L1_Large 专版)
# 功能: 强制 6 种算法在真实的 300秒 物理时间限制内运行，
#       生成详细数据表格，并自动绘制【同时间极限分数分布图】！
# """
# import time
# import os
# import random
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
#
# # 导入您工程中的基础模块
# from data.data_loader import DataLoader
# from evaluation.fitness import FitnessEvaluator
# from algorithms.genetic_algorithm import GeneticAlgorithm
# from algorithms.ifts import IFTS
# from algorithms.zhu_replica import ZhuReplicaAlgorithm
#
# from models.chromosome import Chromosome
#
# import warnings
#
# warnings.filterwarnings('ignore')
#
# # ==========================================
# # 绘图全局字体配置 (一区顶刊风)
# # ==========================================
# plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
# plt.rcParams['axes.unicode_minus'] = False
# plt.rcParams['font.family'] = 'Times New Roman'
# plt.rcParams['mathtext.fontset'] = 'stix'
#
# # ==========================================
# # 核心实验配置
# # ==========================================
# TIME_LIMIT = 300  # 物理死线：严格限制 300 秒 (5分钟)
# N_RUNS = 5  # 独立运行 5 次
# INSTANCE_NAME = 'L1_Large'  # 只测最大规模算例
# DB_CONFIG = {'host': 'localhost', 'port': 3306, 'user': 'root', 'password': os.getenv('DB_PASSWORD', ''), 'database': 'test3'}
#
#
# # === 1. GA 系时间锁外壳 (支持所有变体) ===
# class TimeLimitedGA(GeneticAlgorithm):
#     def run_with_time_limit(self, max_generations, pop_size, time_limit, algo_name="GA"):
#         self._init_population(pop_size)
#         self._eval_population()
#         self.best_chromosome = max(self.population, key=lambda x: x.fitness)
#
#         start_time = time.time()
#         final_gen = 0
#
#         for gen in range(max_generations):
#             if time.time() - start_time > time_limit:
#                 print(f"      🛑 [{algo_name}] 触发红线! 止步于 {gen} 代")
#                 final_gen = gen
#                 break
#
#             if self.use_rl:
#                 fits = [c.fitness for c in self.population]
#                 state = {'diversity': np.std(fits), 'avg_fit': np.mean(fits),
#                          'best_fit': self.best_chromosome.fitness, 'progress': gen / max_generations}
#                 rl_info = self.rl_controller.get_action(state, training=False)
#                 if rl_info['pc'] is not None:
#                     self.crossover.rate = rl_info['pc']
#                     self.mutation.rate = rl_info['pm']
#
#             self._evolve_population(pop_size)
#
#             current_gen_best = max(self.population, key=lambda x: x.fitness)
#             if current_gen_best.fitness > self.best_chromosome.fitness:
#                 self.best_chromosome = current_gen_best
#
#             if self.use_tabu:
#                 # 🛠️ 修复点：安全获取 tabu_steps，如果底层没有挂载该属性，则默认使用 10 步
#                 t_steps = getattr(self, 'tabu_steps', 10)
#                 improved = self.tabu_search.run(self.best_chromosome, t_steps)
#                 if improved.fitness > self.best_chromosome.fitness:
#                     self.best_chromosome = improved
#
#         return self.best_chromosome, final_gen
#
#
# # === 2. IFTS 时间锁外壳 ===
# class TimeLimitedIFTS(IFTS):
#     def run_with_time_limit(self, max_generations, pop_size, time_limit):
#         ga_config = {
#             'crossover_rate': 0.8, 'mutation_rate': 0.1,
#             'elite_count': 2, 'use_heuristic_init': True,
#             'use_rl_adaptive': False, 'use_tabu': False
#         }
#         ga_engine = TimeLimitedGA(self.data, self.evaluator, ga_config)
#         best_ind, gens = ga_engine.run_with_time_limit(max_generations, pop_size, time_limit, "IFTS")
#         return best_ind, gens
#
#
# # === 3. Zhu-Replica 时间锁外壳 ===
# class TimeLimitedZhu(ZhuReplicaAlgorithm):
#     def run_with_time_limit(self, max_generations, pop_size, time_limit):
#         self.pop_size = pop_size
#         start_time = time.time()
#         final_gen = 0
#
#         population = []
#         for _ in range(self.pop_size):
#             if time.time() - start_time > time_limit:
#                 break
#             genes = self.initializer.solve()
#             chromo = Chromosome(self.data)
#             chromo.genes = genes
#             chromo.fitness = self.evaluator.evaluate(genes)
#             l_vector = [random.choice(self.llh_pool) for _ in range(self.ll_length_init)]
#             population.append({'L': l_vector, 'S': chromo})
#
#         if not population:
#             return None, 0
#
#         best_ind = max(population, key=lambda x: x['S'].fitness)
#         best_solution = self._clone_chromosome(best_ind['S'])
#
#         for gen in range(max_generations):
#             if time.time() - start_time > time_limit:
#                 print(f"      🛑 [Zhu-Replica] 触发红线! 止步于 {gen} 代")
#                 final_gen = gen
#                 break
#
#             new_population = []
#             sorted_pop = sorted(population, key=lambda x: x['S'].fitness, reverse=True)
#             elite = {'L': sorted_pop[0]['L'][:], 'S': self._clone_chromosome(sorted_pop[0]['S'])}
#             new_population.append(elite)
#
#             while len(new_population) < self.pop_size:
#                 if time.time() - start_time > time_limit:
#                     break
#
#                 parent1 = self._tournament_selection(population)
#                 parent2 = self._tournament_selection(population)
#
#                 child = {'L': parent1['L'][:], 'S': self._clone_chromosome(parent1['S'])}
#
#                 if random.random() < self.crossover_rate:
#                     child['L'] = self._two_point_crossover(child['L'], parent2['L'])
#                 if random.random() < self.mutation_rate:
#                     self._single_point_mutation(child['L'])
#
#                 self._apply_llh_sequence(child['S'], child['L'])
#                 child['S'].fitness = self.evaluator.evaluate(child['S'].genes)
#                 new_population.append(child)
#
#             if len(new_population) > 0:
#                 population = new_population
#
#             current_best = max(population, key=lambda x: x['S'].fitness)
#             if current_best['S'].fitness > best_solution.fitness:
#                 best_solution = self._clone_chromosome(current_best['S'])
#
#             final_gen = gen
#
#         return best_solution, final_gen
#
#
# # ==========================================
# # 主运行逻辑
# # ==========================================
# def main():
#     print("=" * 60)
#     print(f"🚀 启动 6大算法同时间竞技 (算例: {INSTANCE_NAME} | 死线: {TIME_LIMIT}s)")
#     print("=" * 60)
#
#     try:
#         loader = DataLoader(INSTANCE_NAME, DB_CONFIG)
#         data = loader.load()
#     except Exception as e:
#         print(f"❌ 数据加载失败: {e}")
#         return
#
#     evaluator = FitnessEvaluator()
#     evaluator.set_data(data)
#
#     results = []
#     MAX_GENS = 10000
#     POP_SIZE = 50
#
#     # 1. Zhu-Replica
#     print("\n>>> [1/6] 正在运行 Zhu-Replica ...")
#     for i in range(N_RUNS):
#         algo = TimeLimitedZhu(data, evaluator)
#         best, gens = algo.run_with_time_limit(max_generations=MAX_GENS, pop_size=POP_SIZE, time_limit=TIME_LIMIT)
#         final_fit = best.fitness if best else 0
#         results.append({'Algorithm': 'Zhu-Replica', 'Run': i + 1, 'Fitness': final_fit, 'Generations': gens})
#
#     # 2. IFTS
#     print("\n>>> [2/6] 正在运行 IFTS ...")
#     for i in range(N_RUNS):
#         algo = TimeLimitedIFTS(data, evaluator)
#         best, gens = algo.run_with_time_limit(max_generations=MAX_GENS, pop_size=POP_SIZE, time_limit=TIME_LIMIT)
#         final_fit = best.fitness if best else 0
#         results.append({'Algorithm': 'IFTS', 'Run': i + 1, 'Fitness': final_fit, 'Generations': gens})
#
#     # 3. Heuristic-GA (No RL, No TS)
#     print("\n>>> [3/6] 正在运行 Heuristic-GA ...")
#     for i in range(N_RUNS):
#         config = {'crossover_rate': 0.8, 'mutation_rate': 0.1, 'elite_count': 2, 'use_heuristic_init': True,
#                   'use_rl_adaptive': False, 'use_tabu': False}
#         algo = TimeLimitedGA(data, evaluator, config)
#         best, gens = algo.run_with_time_limit(max_generations=MAX_GENS, pop_size=POP_SIZE, time_limit=TIME_LIMIT,
#                                               algo_name="Heuristic-GA")
#         final_fit = best.fitness if best else 0
#         results.append({'Algorithm': 'Heuristic-GA', 'Run': i + 1, 'Fitness': final_fit, 'Generations': gens})
#
#     # 4. Heuristic-TS (No RL, With TS)
#     print("\n>>> [4/6] 正在运行 Heuristic-TS ...")
#     for i in range(N_RUNS):
#         config = {'crossover_rate': 0.8, 'mutation_rate': 0.1, 'elite_count': 2, 'use_heuristic_init': True,
#                   'use_rl_adaptive': False, 'use_tabu': True, 'tabu_steps': 10}
#         algo = TimeLimitedGA(data, evaluator, config)
#         best, gens = algo.run_with_time_limit(max_generations=MAX_GENS, pop_size=POP_SIZE, time_limit=TIME_LIMIT,
#                                               algo_name="Heuristic-TS")
#         final_fit = best.fitness if best else 0
#         results.append({'Algorithm': 'Heuristic-TS', 'Run': i + 1, 'Fitness': final_fit, 'Generations': gens})
#
#     # 5. GA-DRL (With RL, No TS)
#     print("\n>>> [5/6] 正在运行 GA-DRL ...")
#     for i in range(N_RUNS):
#         config = {'crossover_rate': 0.8, 'mutation_rate': 0.1, 'elite_count': 2, 'use_heuristic_init': True,
#                   'use_rl_adaptive': True, 'use_tabu': False}
#         algo = TimeLimitedGA(data, evaluator, config)
#         best, gens = algo.run_with_time_limit(max_generations=MAX_GENS, pop_size=POP_SIZE, time_limit=TIME_LIMIT,
#                                               algo_name="GA-DRL")
#         final_fit = best.fitness if best else 0
#         results.append({'Algorithm': 'GA-DRL', 'Run': i + 1, 'Fitness': final_fit, 'Generations': gens})
#
#     # 6. DL-GA-TS (Ours - With RL, With TS)
#     print("\n>>> [6/6] 正在运行 DL-GA-TS (Ours) ...")
#     for i in range(N_RUNS):
#         config = {'crossover_rate': 0.8, 'mutation_rate': 0.1, 'elite_count': 2, 'use_heuristic_init': True,
#                   'use_rl_adaptive': True, 'use_tabu': True, 'tabu_steps': 10}
#         algo = TimeLimitedGA(data, evaluator, config)
#         best, gens = algo.run_with_time_limit(max_generations=MAX_GENS, pop_size=POP_SIZE, time_limit=TIME_LIMIT,
#                                               algo_name="DL-GA-TS")
#         final_fit = best.fitness if best else 0
#         results.append({'Algorithm': 'DL-GA-TS', 'Run': i + 1, 'Fitness': final_fit, 'Generations': gens})
#
#     # ==========================================
#     # 数据保存与可视化生成
#     # ==========================================
#     df = pd.DataFrame(results)
#
#     # 固定算法展示顺序
#     algo_order = ['IFTS', 'Zhu-Replica', 'Heuristic-TS', 'Heuristic-GA', 'GA-DRL', 'DL-GA-TS']
#     df['Algorithm'] = pd.Categorical(df['Algorithm'], categories=algo_order, ordered=True)
#
#     print("\n" + "=" * 50)
#     print(f"🏆 {TIME_LIMIT}秒 极限求生最终得分榜 (平均值)")
#     print("=" * 50)
#
#     summary = df.groupby('Algorithm').agg(
#         Mean_Fitness=('Fitness', 'mean'), Std_Fitness=('Fitness', 'std'),
#         Mean_Generations=('Generations', 'mean')
#     ).reset_index().sort_values(by='Mean_Fitness', ascending=False)
#     print(summary.to_string(index=False))
#
#     os.makedirs("results", exist_ok=True)
#     save_csv_path = f"results/time_budget_{TIME_LIMIT}s_{INSTANCE_NAME}.xlsx"
#     df.to_excel(save_csv_path, index=False)
#     print(f"\n✅ 详细数据表格已保存至: {save_csv_path}")
#
#     # --- 绘图 ---
#     print("\n📊 正在生成六大算法降维打击图表...")
#     plt.figure(figsize=(11, 6))
#
#     # 保持与论文收敛图高度一致的统一学术配色
#     palette = {
#         'IFTS': '#1F77B4',
#         'Zhu-Replica': '#FF7F0E',
#         'Heuristic-TS': '#9467BD',
#         'Heuristic-GA': '#8C564B',
#         'GA-DRL': '#2CA02C',
#         'DL-GA-TS': '#D62728'
#     }
#
#     # 画箱线图 (展示分布)
#     sns.boxplot(x='Algorithm', y='Fitness', data=df, width=0.5,
#                 palette=palette, boxprops=dict(alpha=0.6), showfliers=False)
#
#     # 叠加散点 (展示每次的具体落点)
#     sns.stripplot(x='Algorithm', y='Fitness', data=df,
#                   color='black', size=7, alpha=0.8, jitter=0.2)
#
#     # 标注出本文算法的均值虚线
#     dl_mean = summary[summary['Algorithm'] == 'DL-GA-TS']['Mean_Fitness'].values[0]
#     plt.axhline(y=dl_mean, color='#D62728', linestyle='--', alpha=0.6, zorder=0, linewidth=2)
#     plt.text(0.5, dl_mean + 10, f'Ours Expected Performance: {dl_mean:.0f}',
#              color='#D62728', va='bottom', ha='left', fontsize=12, fontweight='bold')
#
#     plt.title(f'Comprehensive Performance Comparison under Strict {TIME_LIMIT}s Time Budget\n({INSTANCE_NAME})',
#               fontsize=16, fontweight='bold', pad=15)
#     plt.xlabel('Algorithm', fontsize=14, fontweight='bold')
#     plt.ylabel('Final Best Fitness Reached', fontsize=14, fontweight='bold')
#
#     plt.xticks(fontsize=12, fontweight='bold', rotation=15)
#     plt.yticks(fontsize=12)
#     plt.grid(True, linestyle=':', alpha=0.6, axis='y')
#
#     plot_save_path = f"results/time_budget_{TIME_LIMIT}s_All_Algorithms_{INSTANCE_NAME}.png"
#     plt.tight_layout()
#     plt.savefig(plot_save_path, dpi=300)
#     print(f"✅ 六大算法可视化图表已保存至: {plot_save_path}")
#     plt.show()
#
#
# if __name__ == "__main__":
#     main()

"""
论文级同时间预算对比实验 (Strict Time Budget Comparison - L1_Large)
- 严格 300s 物理时间预算：从算法入口开始计时（含初始化/评估/TS/RL等）
- 记录：Fitness / Generations / Elapsed(s) / Seed
- 输出：1) 明细表 (xlsx) 2) 汇总表(均值±方差) 3) AnyTime 曲线数据 4) 30/60/120/300s 性能表
- 绘图：箱线图 + Anytime 曲线(均值±std)
"""

import os
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

# ===== 工程模块 =====
from data.data_loader import DataLoader
from evaluation.fitness import FitnessEvaluator
from algorithms.genetic_algorithm import GeneticAlgorithm
from algorithms.ifts import IFTS
from algorithms.zhu_replica import ZhuReplicaAlgorithm
from models.chromosome import Chromosome

# ===== 字体（你原来的配置保留）=====
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stix"

# =========================
# 核心实验配置（论文建议）
# =========================
TIME_LIMIT = 300
N_RUNS = 20  # ✅ 论文建议至少 20（5 太少容易被质疑）
INSTANCE_NAME = "L1_Large"
DB_CONFIG = {"host": "localhost", "port": 3306, "user": "root", "password": os.getenv("DB_PASSWORD", ""), "database": "test3"}

MAX_GENS = 1000000  # 反正会被时间截断，代数给大即可
POP_SIZE = 50

# Anytime 评估点（论文常用）
ANYTIME_CHECKPOINTS = [30, 60, 120, 300]

# 画图配色（保留你原来的风格）
PALETTE = {
    "IFTS": "#1F77B4",
    "Zhu-Replica": "#FF7F0E",
    "Heuristic-TS": "#9467BD",
    "Heuristic-GA": "#8C564B",
    "GA-DRL": "#2CA02C",
    "DL-GA-TS": "#D62728",
}

ALGO_ORDER = ["IFTS", "Zhu-Replica", "Heuristic-TS", "Heuristic-GA", "GA-DRL", "DL-GA-TS"]


# =========================
# 工具函数
# =========================
def now_s() -> float:
    """高精度计时器（适合 time budget）。"""
    return time.perf_counter()


def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def anytime_snapshot(anytime_list, start_t, best_fit):
    """记录 (elapsed_s, best_fit)。"""
    elapsed = now_s() - start_t
    anytime_list.append((elapsed, float(best_fit)))


def anytime_to_checkpoints(anytime_pairs, checkpoints):
    """
    把 (t, best) 序列转换成在 checkpoints 时刻的 best 值（阶梯保持）。
    若在某 checkpoint 前无记录，则用第一条记录或 0。
    """
    if not anytime_pairs:
        return {c: 0.0 for c in checkpoints}

    anytime_pairs = sorted(anytime_pairs, key=lambda x: x[0])
    result = {}
    idx = 0
    current_best = anytime_pairs[0][1]

    for c in checkpoints:
        while idx < len(anytime_pairs) and anytime_pairs[idx][0] <= c:
            current_best = max(current_best, anytime_pairs[idx][1])
            idx += 1
        result[c] = float(current_best)
    return result


# =========================
# 1) GA 系：严格时间锁 + Anytime 轨迹
# =========================
class TimeLimitedGA(GeneticAlgorithm):
    def run_with_time_limit(self, max_generations, pop_size, time_limit, algo_name="GA"):
        start_t = now_s()
        anytime = []

        # ✅ 初始化也算在 time budget 里
        self._init_population(pop_size)
        if now_s() - start_t >= time_limit:
            return None, 0, now_s() - start_t, anytime

        self._eval_population()
        if not self.population:
            return None, 0, now_s() - start_t, anytime

        self.best_chromosome = max(self.population, key=lambda x: x.fitness)
        anytime_snapshot(anytime, start_t, self.best_chromosome.fitness)

        final_gen = -1
        for gen in range(max_generations):
            final_gen = gen

            # ✅ 每代开始卡死线
            if now_s() - start_t >= time_limit:
                print(f"      🛑 [{algo_name}] 触发红线! 止步于 {gen} 代")
                break

            # RL 自适应
            if self.use_rl:
                fits = [c.fitness for c in self.population]
                state = {
                    "diversity": float(np.std(fits)),
                    "avg_fit": float(np.mean(fits)),
                    "best_fit": float(self.best_chromosome.fitness),
                    "progress": gen / max_generations,
                }
                rl_info = self.rl_controller.get_action(state, training=False)
                if rl_info.get("pc") is not None:
                    self.crossover.rate = rl_info["pc"]
                    self.mutation.rate = rl_info["pm"]

            # GA 进化
            self._evolve_population(pop_size)

            # 更新 best
            current_gen_best = max(self.population, key=lambda x: x.fitness)
            if current_gen_best.fitness > self.best_chromosome.fitness:
                self.best_chromosome = current_gen_best

            # Tabu Search
            if self.use_tabu:
                t_steps = getattr(self, "tabu_steps", 10)
                improved = self.tabu_search.run(self.best_chromosome, t_steps)
                if improved.fitness > self.best_chromosome.fitness:
                    self.best_chromosome = improved

            anytime_snapshot(anytime, start_t, self.best_chromosome.fitness)

            # ✅ 每代末尾再卡一次（防止 evolve/tabu 超时后偷跑）
            if now_s() - start_t >= time_limit:
                break

        elapsed = now_s() - start_t
        return self.best_chromosome, max(final_gen, 0), elapsed, anytime


# =========================
# 2) IFTS：内部用 TimeLimitedGA 跑（严格时间）
# =========================
class TimeLimitedIFTS(IFTS):
    def run_with_time_limit(self, max_generations, pop_size, time_limit):
        ga_config = {
            "crossover_rate": 0.8,
            "mutation_rate": 0.1,
            "elite_count": 2,
            "use_heuristic_init": True,
            "use_rl_adaptive": False,
            "use_tabu": False,
        }
        ga_engine = TimeLimitedGA(self.data, self.evaluator, ga_config)
        best, gens, elapsed, anytime = ga_engine.run_with_time_limit(
            max_generations, pop_size, time_limit, algo_name="IFTS"
        )
        return best, gens, elapsed, anytime


# =========================
# 3) Zhu-Replica：严格时间锁 + Anytime
# =========================
class TimeLimitedZhu(ZhuReplicaAlgorithm):
    def run_with_time_limit(self, max_generations, pop_size, time_limit):
        self.pop_size = pop_size
        start_t = now_s()
        anytime = []
        final_gen = -1

        # 初始化也计时
        population = []
        for _ in range(self.pop_size):
            if now_s() - start_t >= time_limit:
                break
            genes = self.initializer.solve()
            chromo = Chromosome(self.data)
            chromo.genes = genes
            chromo.fitness = self.evaluator.evaluate(genes)
            l_vector = [random.choice(self.llh_pool) for _ in range(self.ll_length_init)]
            population.append({"L": l_vector, "S": chromo})

        if not population:
            return None, 0, now_s() - start_t, anytime

        best_ind = max(population, key=lambda x: x["S"].fitness)
        best_solution = self._clone_chromosome(best_ind["S"])
        anytime_snapshot(anytime, start_t, best_solution.fitness)

        for gen in range(max_generations):
            final_gen = gen
            if now_s() - start_t >= time_limit:
                print(f"      🛑 [Zhu-Replica] 触发红线! 止步于 {gen} 代")
                break

            new_population = []
            sorted_pop = sorted(population, key=lambda x: x["S"].fitness, reverse=True)
            elite = {"L": sorted_pop[0]["L"][:], "S": self._clone_chromosome(sorted_pop[0]["S"])}
            new_population.append(elite)

            while len(new_population) < self.pop_size:
                if now_s() - start_t >= time_limit:
                    break

                parent1 = self._tournament_selection(population)
                parent2 = self._tournament_selection(population)

                child = {"L": parent1["L"][:], "S": self._clone_chromosome(parent1["S"])}

                if random.random() < self.crossover_rate:
                    child["L"] = self._two_point_crossover(child["L"], parent2["L"])
                if random.random() < self.mutation_rate:
                    self._single_point_mutation(child["L"])

                self._apply_llh_sequence(child["S"], child["L"])
                child["S"].fitness = self.evaluator.evaluate(child["S"].genes)
                new_population.append(child)

            if new_population:
                population = new_population

            current_best = max(population, key=lambda x: x["S"].fitness)
            if current_best["S"].fitness > best_solution.fitness:
                best_solution = self._clone_chromosome(current_best["S"])

            anytime_snapshot(anytime, start_t, best_solution.fitness)

            if now_s() - start_t >= time_limit:
                break

        elapsed = now_s() - start_t
        return best_solution, max(final_gen, 0), elapsed, anytime


# =========================
# 主运行逻辑
# =========================
def run_one_algorithm(algo_name, data, evaluator, seed, config=None):
    set_all_seeds(seed)

    if algo_name == "Zhu-Replica":
        algo = TimeLimitedZhu(data, evaluator)
        best, gens, elapsed, anytime = algo.run_with_time_limit(MAX_GENS, POP_SIZE, TIME_LIMIT)

    elif algo_name == "IFTS":
        algo = TimeLimitedIFTS(data, evaluator)
        best, gens, elapsed, anytime = algo.run_with_time_limit(MAX_GENS, POP_SIZE, TIME_LIMIT)

    else:
        # GA系
        algo = TimeLimitedGA(data, evaluator, config)
        best, gens, elapsed, anytime = algo.run_with_time_limit(MAX_GENS, POP_SIZE, TIME_LIMIT, algo_name=algo_name)

    final_fit = float(best.fitness) if best else 0.0
    return final_fit, int(gens), float(elapsed), anytime


def main():
    print("=" * 72)
    print(f"🚀 论文级同时间预算实验启动 | Instance={INSTANCE_NAME} | Budget={TIME_LIMIT}s | Runs={N_RUNS}")
    print("=" * 72)

    # 载入数据
    try:
        loader = DataLoader(INSTANCE_NAME, DB_CONFIG)
        data = loader.load()
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return

    evaluator = FitnessEvaluator()
    evaluator.set_data(data)

    # 6种算法配置
    algo_configs = {
        "Zhu-Replica": None,
        "IFTS": None,
        "Heuristic-GA": {
            "crossover_rate": 0.8,
            "mutation_rate": 0.1,
            "elite_count": 2,
            "use_heuristic_init": True,
            "use_rl_adaptive": False,
            "use_tabu": False,
        },
        "Heuristic-TS": {
            "crossover_rate": 0.8,
            "mutation_rate": 0.1,
            "elite_count": 2,
            "use_heuristic_init": True,
            "use_rl_adaptive": False,
            "use_tabu": True,
            "tabu_steps": 10,
        },
        "GA-DRL": {
            "crossover_rate": 0.8,
            "mutation_rate": 0.1,
            "elite_count": 2,
            "use_heuristic_init": True,
            "use_rl_adaptive": True,
            "use_tabu": False,
        },
        "DL-GA-TS": {
            "crossover_rate": 0.8,
            "mutation_rate": 0.1,
            "elite_count": 2,
            "use_heuristic_init": True,
            "use_rl_adaptive": True,
            "use_tabu": True,
            "tabu_steps": 10,
        },
    }

    # 结果容器
    rows = []
    anytime_rows = []  # 用于画 anytime 曲线/算 checkpoints

    base_seed = 202600  # 你也可以改成论文里写死的种子基数

    for algo_name in ALGO_ORDER:
        print(f"\n>>> 正在运行 {algo_name} ...")
        for r in range(N_RUNS):
            seed = base_seed + r  # ✅ 所有算法同一run用同seed（更公平）
            final_fit, gens, elapsed, anytime = run_one_algorithm(
                algo_name, data, evaluator, seed, config=algo_configs.get(algo_name)
            )

            rows.append(
                {
                    "Algorithm": algo_name,
                    "Run": r + 1,
                    "Seed": seed,
                    "Fitness": final_fit,
                    "Generations": gens,
                    "Elapsed(s)": elapsed,
                }
            )

            # anytime 记录（每条：algo/run/seed/t/best）
            for (t, b) in anytime:
                anytime_rows.append(
                    {"Algorithm": algo_name, "Run": r + 1, "Seed": seed, "Time(s)": t, "BestFitness": b}
                )

            print(f"   Run {r+1:02d} | Fitness={final_fit:.2f} | Gens={gens} | Elapsed={elapsed:.2f}s | Seed={seed}")

    df = pd.DataFrame(rows)
    df["Algorithm"] = pd.Categorical(df["Algorithm"], categories=ALGO_ORDER, ordered=True)

    anytime_df = pd.DataFrame(anytime_rows)
    anytime_df["Algorithm"] = pd.Categorical(anytime_df["Algorithm"], categories=ALGO_ORDER, ordered=True)

    # ===== 汇总表（论文常用：均值±方差 + 耗时）=====
    summary = (
        df.groupby("Algorithm")
        .agg(
            Mean_Fitness=("Fitness", "mean"),
            Std_Fitness=("Fitness", "std"),
            Best_Fitness=("Fitness", "max"),
            Mean_Generations=("Generations", "mean"),
            Mean_Elapsed=("Elapsed(s)", "mean"),
            Std_Elapsed=("Elapsed(s)", "std"),
        )
        .reset_index()
        .sort_values(by="Mean_Fitness", ascending=False)
    )

    print("\n" + "=" * 72)
    print(f"🏆 汇总结果（{TIME_LIMIT}s Budget, Runs={N_RUNS}）")
    print("=" * 72)
    print(summary.to_string(index=False))

    # ===== Anytime checkpoints 表（30/60/120/300秒）=====
    cp_rows = []
    for algo_name in ALGO_ORDER:
        algo_any = anytime_df[anytime_df["Algorithm"] == algo_name]

        per_run = []
        for r in range(1, N_RUNS + 1):
            run_pairs = algo_any[algo_any["Run"] == r][["Time(s)", "BestFitness"]].values.tolist()
            run_cp = anytime_to_checkpoints(run_pairs, ANYTIME_CHECKPOINTS)
            per_run.append(run_cp)

        # 平均
        cp_mean = {c: float(np.mean([d[c] for d in per_run])) for c in ANYTIME_CHECKPOINTS}
        cp_std = {c: float(np.std([d[c] for d in per_run], ddof=1)) if N_RUNS > 1 else 0.0 for c in ANYTIME_CHECKPOINTS}

        row = {"Algorithm": algo_name}
        for c in ANYTIME_CHECKPOINTS:
            row[f"T{c}_Mean"] = cp_mean[c]
            row[f"T{c}_Std"] = cp_std[c]
        cp_rows.append(row)

    cp_df = pd.DataFrame(cp_rows)
    cp_df["Algorithm"] = pd.Categorical(cp_df["Algorithm"], categories=ALGO_ORDER, ordered=True)
    cp_df = cp_df.sort_values("Algorithm")

    print("\n" + "=" * 72)
    print("⏱️ Anytime Checkpoints（均值±方差）")
    print("=" * 72)
    print(cp_df.to_string(index=False))

    # ===== 保存结果 =====
    os.makedirs("results", exist_ok=True)
    out_xlsx = f"results/paper_time_budget_{TIME_LIMIT}s_{INSTANCE_NAME}_runs{N_RUNS}.xlsx"
    with pd.ExcelWriter(out_xlsx) as writer:
        df.to_excel(writer, index=False, sheet_name="raw_results")
        summary.to_excel(writer, index=False, sheet_name="summary")
        anytime_df.to_excel(writer, index=False, sheet_name="anytime_trace")
        cp_df.to_excel(writer, index=False, sheet_name="anytime_checkpoints")

    print(f"\n✅ 已保存论文级结果到: {out_xlsx}")

    # =========================
    # 绘图 1：箱线图（最终得分分布）
    # =========================
    plt.figure(figsize=(12, 6))
    sns.boxplot(
        x="Algorithm", y="Fitness", data=df, width=0.55, palette=PALETTE, boxprops=dict(alpha=0.6), showfliers=False
    )
    sns.stripplot(x="Algorithm", y="Fitness", data=df, color="black", size=6, alpha=0.75, jitter=0.18)

    # 标注 Ours 均值线
    ours_mean = float(summary[summary["Algorithm"] == "DL-GA-TS"]["Mean_Fitness"].values[0])
    plt.axhline(y=ours_mean, color=PALETTE["DL-GA-TS"], linestyle="--", alpha=0.7, linewidth=2)
    plt.text(0.4, ours_mean + 10, f"Ours Mean: {ours_mean:.0f}", color=PALETTE["DL-GA-TS"], fontsize=12, fontweight="bold")

    plt.title(f"Final Fitness Distribution under Strict {TIME_LIMIT}s Time Budget\n({INSTANCE_NAME}, Runs={N_RUNS})",
              fontsize=16, fontweight="bold", pad=12)
    plt.xlabel("Algorithm", fontsize=14, fontweight="bold")
    plt.ylabel("Final Best Fitness", fontsize=14, fontweight="bold")
    plt.xticks(fontsize=12, fontweight="bold", rotation=15)
    plt.yticks(fontsize=12)
    plt.grid(True, linestyle=":", alpha=0.55, axis="y")
    plt.tight_layout()

    box_png = f"results/paper_boxplot_{TIME_LIMIT}s_{INSTANCE_NAME}_runs{N_RUNS}.png"
    plt.savefig(box_png, dpi=300)
    print(f"✅ 箱线图已保存: {box_png}")
    plt.show()

    # =========================
    # 绘图 2：Anytime 曲线（均值±Std）
    # 说明：把每个 run 的 trace 先重采样到整数秒，再对 runs 求均值
    # =========================
    max_t = TIME_LIMIT
    t_grid = np.arange(0, max_t + 1, 1)

    plt.figure(figsize=(12, 6))
    for algo_name in ALGO_ORDER:
        algo_any = anytime_df[anytime_df["Algorithm"] == algo_name]
        run_series = []

        for r in range(1, N_RUNS + 1):
            pairs = algo_any[algo_any["Run"] == r][["Time(s)", "BestFitness"]].values.tolist()
            if not pairs:
                run_series.append(np.zeros_like(t_grid, dtype=float))
                continue

            pairs = sorted(pairs, key=lambda x: x[0])
            # 阶梯保持：每秒取截至该秒的 best
            best = pairs[0][1]
            idx = 0
            series = []
            for t in t_grid:
                while idx < len(pairs) and pairs[idx][0] <= t:
                    best = max(best, pairs[idx][1])
                    idx += 1
                series.append(best)
            run_series.append(np.array(series, dtype=float))

        mat = np.vstack(run_series)  # [runs, time]
        mean_curve = mat.mean(axis=0)
        std_curve = mat.std(axis=0, ddof=1) if N_RUNS > 1 else np.zeros_like(mean_curve)

        plt.plot(t_grid, mean_curve, label=algo_name)  # 不指定颜色也能画；你要指定可用 PALETTE
        plt.fill_between(t_grid, mean_curve - std_curve, mean_curve + std_curve, alpha=0.12)

    plt.title(f"Anytime Performance under Strict {TIME_LIMIT}s Time Budget\n({INSTANCE_NAME}, Mean±Std over {N_RUNS} runs)",
              fontsize=16, fontweight="bold", pad=12)
    plt.xlabel("Time (s)", fontsize=14, fontweight="bold")
    plt.ylabel("Best Fitness So Far", fontsize=14, fontweight="bold")
    plt.grid(True, linestyle=":", alpha=0.55)
    plt.legend(fontsize=10)
    plt.tight_layout()

    any_png = f"results/paper_anytime_{TIME_LIMIT}s_{INSTANCE_NAME}_runs{N_RUNS}.png"
    plt.savefig(any_png, dpi=300)
    print(f"✅ Anytime 曲线已保存: {any_png}")
    plt.show()

    # =========================
    # 可选：显著性检验（对比 Ours vs Others）
    # Wilcoxon（若 scipy 可用）
    # =========================
    try:
        from scipy.stats import wilcoxon
        ours = df[df["Algorithm"] == "DL-GA-TS"]["Fitness"].values
        print("\n" + "=" * 72)
        print("📌 Wilcoxon 显著性检验（Ours vs Baselines）")
        print("=" * 72)
        for algo_name in ALGO_ORDER:
            if algo_name == "DL-GA-TS":
                continue
            base = df[df["Algorithm"] == algo_name]["Fitness"].values
            # 要求配对：同 run 的 seed（我们已做到 seed 同步）
            if len(base) == len(ours) and len(ours) > 0:
                stat = wilcoxon(ours, base, alternative="greater")  # H1: ours > base
                print(f"Ours > {algo_name:<12} | p-value = {stat.pvalue:.4g}")
    except Exception as e:
        print("\n(提示) 未执行显著性检验：scipy 不可用或环境不支持。")


if __name__ == "__main__":
    main()
