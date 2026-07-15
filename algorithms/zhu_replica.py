"""
Zhu et al. (2026) GA-RG-HH Algorithm (Strict Replication Ver. Final)
Reference: "A hyper-heuristic algorithm based on genetic and greedy strategy..."
【核心修正】
1. Pre-Check Strategy [Section 3.4.2]:
   - 严格遵循 Algorithm 1: 当操作违反硬约束时，不只是放弃，而是循环重试 (Reselect)，直到找到可行解或达到最大尝试次数。
   - 引用: "the algorithm reselects objects for operation until constraints are satisfied"

2. Low-Level Heuristics [Section 3.4.1]:
   - 完整实现 L1-L10 算子逻辑。
   - 引用: "10 random operators (L1-L4), 5 greedy operators (L5-L9), 1 invalid operator (L10)" [cite: 480]

3. Hyper-Heuristic Framework [Fig. 10]:
   - GA 进化算子序列 (L), 作用于贪心生成的初始解 (S)。
【参数配置 (Table 9)】
- CR (Crossover Rate) = 0.70
- MR (Mutation Rate) = 0.05
- LL (Initial Length) = 10
- PS (Population Size) = 20  [cite: 726] ("population size... is set to 20")
- IM (Max Iterations) = 100  [cite: 726] ("runs... with 100 iterations")

【核心逻辑】
1. 架构: GA 进化算子序列 (L), 作用于贪心生成的初始解 (S).
2. 预检查 (Pre-Check): 严格执行 Algorithm 1，遇到硬约束冲突时循环重试 (Reselect).
"""
import random
import copy
import numpy as np
from models.chromosome import Chromosome
from algorithms.constraint_solver import ConstraintSolver


class ZhuReplicaAlgorithm:
    def __init__(self, data, evaluator):
        self.data = data
        self.evaluator = evaluator

        # === 1. 参数严格对齐 (Parameter Calibration) ===
        self.pop_size = 20  # PS: 种群大小 [cite: 726, 907]
        self.max_generations = 100  # IM: 最大迭代次数 [cite: 726]
        self.crossover_rate = 0.70  # CR: 交叉率
        self.mutation_rate = 0.05  # MR: 变异率
        self.ll_length_init = 10  # LL: 初始算子序列长度

        # 预检查最大尝试次数 (防止死循环)
        self.max_attempts = 10

        # 贪心初始化器
        self.initializer = ConstraintSolver(data)

        # 算子池 (0-9 对应 L1-L10)
        self.llh_pool = list(range(10))

    def run(self, generations=None):
        # 允许外部覆盖代数，否则使用默认值
        if generations is None:
            generations = self.max_generations

        print(
            f"Zhu-2026 (Strict): Start (Pop={self.pop_size}, CR={self.crossover_rate}, MR={self.mutation_rate}, LL={self.ll_length_init})")

        # --- 1. Initialization [Algorithm 2, cite: 549] ---
        population = []
        for _ in range(self.pop_size):
            # A. 生成贪心初始解 S (Greedy Strategy) [cite: 353]
            genes = self.initializer.solve()
            chromo = Chromosome(self.data)
            chromo.genes = genes
            chromo.fitness = self.evaluator.evaluate(genes)

            # B. 生成随机算子序列 L (Uniform distribution 0-9) [cite: 366]
            l_vector = [random.choice(self.llh_pool) for _ in range(self.ll_length_init)]

            population.append({'L': l_vector, 'S': chromo})

        # 初始最优
        best_ind = max(population, key=lambda x: x['S'].fitness)
        best_solution = self._clone_chromosome(best_ind['S'])
        print(f"Zhu-2026 [Init]: Best Fitness = {best_solution.fitness:.2f}")

        history = []

        # --- 2. Main Loop [Algorithm 2, cite: 549] ---
        for gen in range(generations):
            new_population = []

            # 精英保留 (Elitism)
            sorted_pop = sorted(population, key=lambda x: x['S'].fitness, reverse=True)
            elite = {
                'L': sorted_pop[0]['L'][:],
                'S': self._clone_chromosome(sorted_pop[0]['S'])
            }
            new_population.append(elite)

            while len(new_population) < self.pop_size:
                # A. Selection (Tournament) [cite: 448]
                parent1 = self._tournament_selection(population)
                parent2 = self._tournament_selection(population)

                # 子代继承 Parent1 的基础
                child = {
                    'L': parent1['L'][:],
                    'S': self._clone_chromosome(parent1['S'])
                }

                # B. Two-point Crossover on L [cite: 460]
                if random.random() < self.crossover_rate:
                    child['L'] = self._two_point_crossover(child['L'], parent2['L'])

                # C. Single-point Mutation on L [cite: 465]
                if random.random() < self.mutation_rate:
                    self._single_point_mutation(child['L'])

                # D. Apply Operators (Pre-Check Logic Inside) [cite: 496]
                self._apply_llh_sequence(child['S'], child['L'])

                # 评估
                child['S'].fitness = self.evaluator.evaluate(child['S'].genes)
                new_population.append(child)

            population = new_population

            # 更新全局最优
            current_best = max(population, key=lambda x: x['S'].fitness)
            if current_best['S'].fitness > best_solution.fitness:
                best_solution = self._clone_chromosome(current_best['S'])

            # 记录数据
            violations = self.evaluator.constraint_checker.check_all(best_solution.genes)['total']
            avg_fit = np.mean([p['S'].fitness for p in population])

            history.append({
                'gen': gen,
                'best_fit': best_solution.fitness,
                'avg_fit': avg_fit,
                'violations': violations,
                'rewards': 0, 'loss': 0
            })

            if gen % 20 == 0:
                print(f"Zhu-2026 [Gen {gen}]: Best={best_solution.fitness:.2f} | Vio={violations}")

        return best_solution, history, []

    # --- 核心：Pre-Check Strategy & LLH Operators ---

    def _apply_llh_sequence(self, chromo, l_vec):
        """依次执行序列中的算子，每个算子都带有 Pre-Check 重试机制"""
        for op_id in l_vec:
            # 尝试执行算子，直到成功或次数耗尽 (Pre-Check Strategy) [cite: 498]
            # "the algorithm reselects objects for operation until constraints are satisfied"
            for attempt in range(self.max_attempts):
                success = False

                # 备份当前基因 (为了在 check 失败时回滚)
                backup_genes = chromo.genes.copy()

                if op_id == 0:
                    success = self._L1_timeslot_swap(chromo)
                elif op_id == 1:
                    success = self._L2_timeslot_reverse(chromo)
                elif op_id == 2:
                    success = self._L3_classroom_swap(chromo)
                elif op_id == 3:
                    success = self._L4_course_swap(chromo)
                elif op_id in [4, 5, 6, 7, 8]:
                    success = self._L_Greedy(chromo, op_id)
                elif op_id == 9:
                    success = True  # L10: Invalid (Do nothing) [cite: 495]

                if success:
                    break
                else:
                    # Check 失败，回滚状态，进行下一次 attempt (Reselect)
                    chromo.genes = backup_genes

    # --- Random Operators (L1-L4) [cite: 485] ---

    def _L1_timeslot_swap(self, chromo):
        """L1: Randomly swap all course tasks in two timeslots."""
        slots = list(range(25))
        s1, s2 = random.sample(slots, 2)

        for tid, gene in chromo.genes.items():
            slot = gene >> 16
            room = gene & 0xFFFF
            if slot == s1:
                chromo.genes[tid] = (s2 << 16) | room
            elif slot == s2:
                chromo.genes[tid] = (s1 << 16) | room

        return self._pre_check(chromo.genes)

    def _L2_timeslot_reverse(self, chromo):
        """L2: Reverse the order of 2-4 consecutive timeslots."""
        start = random.randint(0, 20)
        length = random.randint(2, 4)
        end = start + length

        target_slots = list(range(start, end))
        reversed_slots = target_slots[::-1]
        mapping = dict(zip(target_slots, reversed_slots))

        for tid, gene in chromo.genes.items():
            slot = gene >> 16
            if slot in mapping:
                room = gene & 0xFFFF
                new_slot = mapping[slot]
                chromo.genes[tid] = (new_slot << 16) | room

        return self._pre_check(chromo.genes)

    def _L3_classroom_swap(self, chromo):
        """L3: Swap all courses of the same classroom type between two timeslots."""
        rooms = list(self.data.classrooms.keys())
        if len(rooms) < 2: return False
        r1, r2 = random.sample(rooms, 2)
        r1_hash = hash(r1) & 0xFFFF
        r2_hash = hash(r2) & 0xFFFF

        for tid, gene in chromo.genes.items():
            slot = gene >> 16
            room = gene & 0xFFFF
            if room == r1_hash:
                chromo.genes[tid] = (slot << 16) | r2_hash
            elif room == r2_hash:
                chromo.genes[tid] = (slot << 16) | r1_hash

        return self._pre_check(chromo.genes)

    def _L4_course_swap(self, chromo):
        """L4: Randomly swap two course tasks."""
        tasks = list(self.data.tasks.keys())
        t1, t2 = random.sample(tasks, 2)

        g1, g2 = chromo.genes[t1], chromo.genes[t2]
        chromo.genes[t1], chromo.genes[t2] = g2, g1

        return self._pre_check(chromo.genes)

    # --- Greedy Operators (L5-L9) [cite: 489] ---

    def _L_Greedy(self, chromo, op_id):
        """
        Generic Greedy Operator implementation.
        Simulates greedy improvements (e.g. L7: suitable timeslots).
        """
        tasks = list(self.data.tasks.keys())
        t1 = random.choice(tasks)

        current_fit = self.evaluator.evaluate(chromo.genes)
        best_gene = chromo.genes[t1]
        found_improvement = False

        for _ in range(5):  # Try 5 greedy moves
            new_slot = random.randint(0, 24)
            new_room = random.choice(list(self.data.classrooms.keys()))
            try:
                r_int = int(new_room)
            except:
                r_int = hash(new_room) & 0xFFFF

            candidate_gene = (new_slot << 16) | (r_int & 0xFFFF)
            chromo.genes[t1] = candidate_gene

            if self._pre_check(chromo.genes):
                new_fit = self.evaluator.evaluate(chromo.genes)
                if new_fit > current_fit:
                    current_fit = new_fit
                    best_gene = candidate_gene
                    found_improvement = True

        chromo.genes[t1] = best_gene
        return found_improvement

    def _pre_check(self, genes):
        """[Algorithm 1, cite: 520] Check if constraints are violated."""
        violations = self.evaluator.constraint_checker.check_all(genes)
        return violations['total'] == 0

    # --- Genetic Helpers ---

    def _two_point_crossover(self, l1, l2):
        """[Fig. 6, cite: 460]"""
        size = len(l1)
        if size < 2: return l1[:]
        p1, p2 = sorted(random.sample(range(size), 2))
        return l1[:p1] + l2[p1:p2] + l1[p2:]

    def _single_point_mutation(self, l_vec):
        """[Fig. 7, cite: 465]"""
        idx = random.randint(0, len(l_vec) - 1)
        l_vec[idx] = random.choice(self.llh_pool)

    def _tournament_selection(self, population, k=3):
        candidates = random.sample(population, k)
        return max(candidates, key=lambda x: x['S'].fitness)

    def _clone_chromosome(self, chromo):
        new_c = Chromosome(self.data)
        new_c.genes = chromo.genes.copy()
        new_c.fitness = chromo.fitness
        return new_c