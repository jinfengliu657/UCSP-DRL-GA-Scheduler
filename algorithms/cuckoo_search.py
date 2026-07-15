import time
import random
import math
import numpy as np
from models.chromosome import Chromosome
from algorithms.constraint_solver import generate_feasible_solution


class CuckooSearchAlgorithm:
    """
    混合自适应布谷鸟搜索算法 (Hybrid Self-adaptive Cuckoo Search, HSCST)
    严格复刻自顶刊: Thepphakorn & Pongcharoen (2020)
    """

    def __init__(self, data, evaluator, config=None):
        self.data = data
        self.evaluator = evaluator

        # 忠实还原论文的自适应参数边界
        self.pa_max = 0.50  # 初始最大外星蛋发现概率
        self.pa_min = 0.05  # 末期最小外星蛋发现概率
        self.alpha_max = 0.50  # 初始最大步长缩放因子 (探索)
        self.alpha_min = 0.01  # 末期最小步长缩放因子 (开发)

        self.population = []
        self.best_chromosome = None

        # 【修复 2】提前安全缓存有效的 Room IDs，防止 L1 算例崩溃
        self.valid_room_ids = []
        for r_key, room in self.data.classrooms.items():
            try:
                r_int = int(room.classroom_id)
            except (ValueError, AttributeError):
                r_int = hash(r_key) & 0xFFFF
            self.valid_room_ids.append(r_int)

        self.total_slots = 25

    def _init_population(self, pop_size):
        """步骤 1: 初始化鸟窝 (种群)"""
        self.population = []
        for _ in range(pop_size):
            genes = generate_feasible_solution(self.data)
            c = Chromosome(self.data)
            c.genes = genes
            c.fitness = self.evaluator.evaluate(genes)
            self.population.append(c)
        self.best_chromosome = max(self.population, key=lambda x: x.fitness)

    def _mantegna_levy_flight(self):
        """步骤 2: 计算莱维飞行步长 (重尾分布)"""
        beta = 1.5
        sigma = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) /
                 (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.normal(0, sigma)
        v = np.random.normal(0, 1)
        # 防止除零错误
        v = v if v != 0 else 1e-8
        step = u / (abs(v) ** (1 / beta))
        return step

    def _discrete_levy_walk(self, target_chromo, current_alpha):
        """步骤 3: 离散化莱维游走 (纯随机破坏与重排)"""
        new_genes = target_chromo.genes.copy()
        total_tasks = len(new_genes)

        levy_step = abs(self._mantegna_levy_flight())
        mutation_rate = levy_step * current_alpha
        mutation_rate = min(mutation_rate, 0.5)  # 限制单次最大破坏比例为 50%

        num_to_mutate = max(1, int(total_tasks * mutation_rate))
        tasks_to_change = random.sample(list(new_genes.keys()), num_to_mutate)

        for tid in tasks_to_change:
            new_slot = random.randint(0, self.total_slots - 1)
            new_room = random.choice(self.valid_room_ids)
            new_genes[tid] = (new_slot << 16) | new_room

        c_new = Chromosome(self.data)
        c_new.genes = new_genes
        c_new.fitness = self.evaluator.evaluate(new_genes)
        return c_new

    def _abandon_alien_eggs(self, pop_size, current_pa, start_time, time_limit=float('inf')):
        """步骤 4: 外星蛋发现机制 (按数学概率随机淘汰)"""
        for i in range(pop_size):
            if time.perf_counter() - start_time > time_limit:
                break

            # 保护全局最优解不被随意抛弃
            if self.population[i].fitness == self.best_chromosome.fitness:
                continue

            if random.random() < current_pa:
                genes = generate_feasible_solution(self.data)
                c_new = Chromosome(self.data)
                c_new.genes = genes
                c_new.fitness = self.evaluator.evaluate(genes)
                self.population[i] = c_new

    # 【修复 3】提供标准的 run 接口，兼容所有的调用脚本
    def run(self, generations, pop_size=50):
        """标准调用接口"""
        return self.run_with_time_limit(generations, pop_size, time_limit=float('inf'))

    def run_with_time_limit(self, max_generations, pop_size, time_limit, algo_name="Cuckoo-Search"):
        """主控流程"""
        start_time = time.perf_counter()
        anytime_trace = []

        self._init_population(pop_size)

        if time.perf_counter() - start_time > time_limit:
            return self.best_chromosome, 0, time.perf_counter() - start_time, anytime_trace

        elapsed = time.perf_counter() - start_time
        anytime_trace.append((elapsed, float(self.best_chromosome.fitness)))

        final_gen = 0

        for gen in range(max_generations):
            if time.perf_counter() - start_time > time_limit:
                print(f"      🛑 [{algo_name}] 触发时间红线! 止步于 {gen} 代")
                break
            final_gen = gen

            # 【修复 1】动态衰减计算修复！保证贯穿整个 max_generations
            progress = gen / float(max_generations)
            current_alpha = self.alpha_max - (self.alpha_max - self.alpha_min) * progress
            current_pa = self.pa_max - (self.pa_max - self.pa_min) * progress

            # 阶段 A：布谷鸟寻找新窝 (Lévy Flights)
            for i in range(pop_size):
                if time.perf_counter() - start_time > time_limit: break

                target_chromo = self.population[i]
                new_chromo = self._discrete_levy_walk(target_chromo, current_alpha)

                random_j = random.randint(0, pop_size - 1)
                if new_chromo.fitness > self.population[random_j].fitness:
                    self.population[random_j] = new_chromo

                if new_chromo.fitness > self.best_chromosome.fitness:
                    self.best_chromosome = new_chromo

            # 阶段 B：宿主鸟发现外星蛋并抛弃
            if time.perf_counter() - start_time <= time_limit:
                self._abandon_alien_eggs(pop_size, current_pa, start_time, time_limit)

            elapsed = time.perf_counter() - start_time
            anytime_trace.append((elapsed, float(self.best_chromosome.fitness)))

        elapsed = time.perf_counter() - start_time
        return self.best_chromosome, final_gen, elapsed, anytime_trace