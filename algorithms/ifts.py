"""
Xiang et al. (2024) Algorithm Replication
Algorithm Name: IFTS (Improved Fast Tabu Search) - as cited in Zhu et al. (2026)

【参数严格对齐 (Parameter Alignment)】
基于 Xiang et al. (2024) Section 6.2 & Table 5:
1. GA 阶段:
   - N_Population (Pop Size) = 100
   - T (Generations) = 200
   - Pc (Crossover Rate) = 0.1
   - Pm (Mutation Rate) = 0.1

2. TS 阶段:
   - Iter_Total (Max Steps) = 1000
   - Iter_NoImprove (Stop Threshold) = 30
   - TT (Tabu Tenure) = 8

【算法逻辑 (Logic)】
1. Stage 1 (Global Search): GA 进化，初始种群由 Greedy Strategy 生成 (符合 Zhu 的公平性)。
2. Stage 2 (Local Search): TS 搜索，严格维持可行性 (Strict Feasibility)，不接受违约解。
"""
import random
import copy
import numpy as np
from algorithms.genetic_algorithm import GeneticAlgorithm
from algorithms.constraint_solver import ConstraintSolver
from models.chromosome import Chromosome


class IFTS:
    def __init__(self, data, evaluator):
        self.data = data
        self.evaluator = evaluator

        # --- 1. 统一初始化接口 (Fairness Baseline) ---
        self.initializer = ConstraintSolver(data)

        # --- 2. 初始化 GA 引擎 ---
        # [Fix] 创建一个符合 GA __init__ 要求的配置字典
        ga_config = {
            'crossover_rate': 0.1,  # 初始占位，稍后手动控制
            'mutation_rate': 0.1,  # 初始占位，稍后手动控制
            'use_heuristic_init': False,  # 关闭内置启发式，使用我们手动注入的全贪心种群
            'use_rl_adaptive': False,
            'use_tabu': False,
            'patience': 999999  # 防止 GA 内部早停，由 IFTS 控制代数
        }
        self.ga_engine = GeneticAlgorithm(data, evaluator, ga_config)

        # --- 3. 参数严格设置 (Hardcoded from Paper) ---
        # Stage 1: GA Parameters
        self.ga_pop_size = 100  # NPopulation
        self.ga_generations = 200  # T
        self.ga_pc = 0.1  # Pc
        self.ga_pm = 0.1  # Pm

        # Stage 2: TS Parameters
        self.ts_max_iter = 1000  # IterTotal
        self.ts_no_improve = 30  # IterNoImprove
        self.tabu_tenure = 8  # TT
        self.neighbor_sample = 40  # 邻域采样数 (经验值)

    def run(self, generations=None):
        """
        执行 IFTS 两阶段搜索
        generations: 仅做接口兼容，实际由内部 self.ga_generations 和 self.ts_max_iter 控制
        """
        print(f"IFTS: Start (Pop={self.ga_pop_size}, T={self.ga_generations}, IterTotal={self.ts_max_iter})")
        print(f"      Params: Pc={self.ga_pc}, Pm={self.ga_pm}, TT={self.tabu_tenure}")

        history = []

        # ==========================================
        # Stage 1: Global Search (GA)
        # ==========================================
        print("IFTS: [Stage 1] Running GA (Seeded by Greedy)...")

        # 1.1 贪心初始化种群 (Seeding with Greedy)
        # 严格遵循 Zhu 的要求：所有个体由贪心策略生成
        seeded_population = []
        for _ in range(self.ga_pop_size):
            genes = self.initializer.solve()
            chromo = Chromosome(self.data)
            chromo.genes = genes
            chromo.fitness = self.evaluator.evaluate(genes)
            seeded_population.append(chromo)

        # 1.2 注入种群与参数
        self.ga_engine.population = seeded_population
        # 强制更新 GA 引擎内部的算子参数
        self.ga_engine.crossover.rate = self.ga_pc
        self.ga_engine.mutation.rate = self.ga_pm

        # 1.3 手动执行 GA 循环 (避免调用 ga.run 导致种群被重置)
        # 初始化最优解
        best_ga_solution = max(self.ga_engine.population, key=lambda x: x.fitness)

        for gen in range(self.ga_generations):
            # 执行进化 (选择 -> 交叉 -> 变异)
            self.ga_engine._evolve_population(self.ga_pop_size)

            # 更新本代最优
            current_gen_best = max(self.ga_engine.population, key=lambda x: x.fitness)
            if current_gen_best.fitness > best_ga_solution.fitness:
                best_ga_solution = self._clone_chromosome(current_gen_best)

            # 记录历史
            avg_fit = np.mean([c.fitness for c in self.ga_engine.population])
            violations = self.evaluator.constraint_checker.check_all(best_ga_solution.genes)

            history.append({
                'gen': gen,
                'best_fit': best_ga_solution.fitness,
                'avg_fit': avg_fit,
                'violations': violations['total'],
                'rewards': 0, 'loss': 0
            })

            if gen % 50 == 0:
                print(f"IFTS [Stage 1] Gen {gen}: BestFit={best_ga_solution.fitness:.2f}")

        print(f"IFTS: [Stage 1 End] Best GA Fitness = {best_ga_solution.fitness:.2f}")

        # ==========================================
        # Stage 2: Local Search (Strict Tabu)
        # ==========================================
        print("IFTS: [Stage 2] Running Strict Feasible Tabu Search...")

        # 继承 GA 最优解作为 TS 起点
        current_solution = self._clone_chromosome(best_ga_solution)
        best_solution = self._clone_chromosome(current_solution)

        # 禁忌表: {(task, slot): expiry_step}
        tabu_list = {}
        no_improve_count = 0

        for step in range(1, self.ts_max_iter + 1):
            # 提前终止检查
            if no_improve_count >= self.ts_no_improve:
                print(f"IFTS: Early stopping at step {step} (No improvement for {self.ts_no_improve} steps)")
                break

            # 2.1 生成严格可行邻居 (Xiang's Logic: No Violations Allowed)
            candidates = self._generate_strict_feasible_neighbors(current_solution)

            if not candidates:
                # 无路可走
                no_improve_count += 1
                self._record_ts_history(history, step, best_solution)
                continue

            # 2.2 选择最佳移动
            best_move = None
            best_move_fit = float('-inf')

            for move in candidates:
                is_tabu = self._is_tabu(move, tabu_list, step)
                is_aspirated = move['fitness'] > best_solution.fitness

                if (not is_tabu) or is_aspirated:
                    if move['fitness'] > best_move_fit:
                        best_move = move
                        best_move_fit = move['fitness']

            # 2.3 执行移动
            if best_move:
                current_solution = best_move['chromo']

                # 更新全局最优
                if current_solution.fitness > best_solution.fitness:
                    best_solution = self._clone_chromosome(current_solution)
                    no_improve_count = 0  # 重置计数器
                else:
                    no_improve_count += 1

                # 更新禁忌表
                key = (best_move['task'], best_move['from_slot'])
                tabu_list[key] = step + self.tabu_tenure
            else:
                no_improve_count += 1

            # 2.4 记录数据
            self._record_ts_history(history, step, best_solution)

            if step % 100 == 0:
                print(f"IFTS [Stage 2] Step {step}: Fit={best_solution.fitness:.2f}")

        return best_solution, history, []

    # --- 核心逻辑方法 ---

    def _generate_strict_feasible_neighbors(self, current_sol):
        """
        [Xiang's Core Constraint Logic]
        只生成硬约束违规数为 0 的邻居。
        """
        neighbors = []
        attempts = 0
        max_attempts = self.neighbor_sample * 5

        while len(neighbors) < self.neighbor_sample and attempts < max_attempts:
            attempts += 1
            if random.random() < 0.5:
                res = self._try_move_strictly(current_sol)
            else:
                res = self._try_swap_strictly(current_sol)
            if res:
                neighbors.append(res)
        return neighbors

    def _try_move_strictly(self, sol):
        """尝试移动，若违约则返回 None"""
        task_id = random.choice(list(self.data.tasks.keys()))
        old_gene = sol.genes[task_id]
        old_slot = old_gene >> 16

        new_slot = random.randint(0, 24)
        if new_slot == old_slot: return None

        new_room = random.choice(list(self.data.classrooms.keys()))
        try:
            r_int = int(new_room)
        except:
            r_int = hash(new_room) & 0xFFFF

        new_sol = self._clone_chromosome(sol)
        new_sol.genes[task_id] = (new_slot << 16) | (r_int & 0xFFFF)

        # === 核心检查 ===
        violations = self.evaluator.constraint_checker.check_all(new_sol.genes)

        if violations['total'] == 0:
            new_sol.fitness = self.evaluator.evaluate(new_sol.genes)
            return {
                'chromo': new_sol, 'task': task_id, 'from_slot': old_slot, 'fitness': new_sol.fitness
            }
        return None

    def _try_swap_strictly(self, sol):
        """尝试交换，若违约则返回 None"""
        t1, t2 = random.sample(list(self.data.tasks.keys()), 2)
        new_sol = self._clone_chromosome(sol)

        g1, g2 = new_sol.genes[t1], new_sol.genes[t2]
        new_sol.genes[t1], new_sol.genes[t2] = g2, g1

        # === 核心检查 ===
        violations = self.evaluator.constraint_checker.check_all(new_sol.genes)

        if violations['total'] == 0:
            new_sol.fitness = self.evaluator.evaluate(new_sol.genes)
            return {
                'chromo': new_sol, 'task': t1, 'from_slot': g1 >> 16, 'fitness': new_sol.fitness
            }
        return None

    def _is_tabu(self, move, tabu_list, step):
        key = (move['task'], move['from_slot'])
        if key in tabu_list and tabu_list[key] > step:
            return True
        return False

    def _clone_chromosome(self, chromo):
        new_c = Chromosome(self.data)
        new_c.genes = chromo.genes.copy()
        new_c.fitness = chromo.fitness
        return new_c

    def _record_ts_history(self, history, step, best):
        # 映射到图表 X 轴: 接在 GA 代数之后
        # 为了图表不过长，每 5 步 TS 算作一个 Plot 点
        mapped_gen = self.ga_generations + (step // 5)
        if not history or history[-1]['gen'] != mapped_gen:
            history.append({
                'gen': mapped_gen,
                'best_fit': best.fitness,
                'avg_fit': best.fitness,
                'violations': 0,
                'rewards': 0, 'loss': 0
            })