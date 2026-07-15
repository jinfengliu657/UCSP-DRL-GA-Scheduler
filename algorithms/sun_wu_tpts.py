import random
import time
from collections import deque
from models.chromosome import Chromosome

# 【关键修复】将导入移到文件最顶部，作为全局导入，彻底消除报红！
from algorithms.constraint_solver import generate_feasible_solution


class SunWuTabuSearch:
    def __init__(self, data, evaluator, config=None):
        self.data = data
        self.evaluator = evaluator

        task_ids = list(self.data.tasks.keys())
        self.tabu_size = int(10 + 0.1 * len(task_ids))
        self.tabu_list = deque(maxlen=self.tabu_size)

        self.valid_room_ids = []
        for r_key, room in self.data.classrooms.items():
            try:
                r_int = int(room.classroom_id)
            except:
                r_int = hash(r_key) & 0xFFFF
            self.valid_room_ids.append(r_int)

        self.total_slots = 25
        self.best_ind = Chromosome(self.data)

    def _get_metrics(self, genes):
        v = self.evaluator.constraint_checker.check_all(genes).get('total', 999)
        f = self.evaluator.evaluate(genes)
        return v, f

    def _generate_neighbor(self, genes):
        new_genes = genes.copy()
        tids = list(new_genes.keys())
        if random.random() < 0.5:
            tid = random.choice(tids)
            new_val = (random.randint(0, self.total_slots - 1) << 16) | random.choice(self.valid_room_ids)
            new_genes[tid] = new_val
            mid = f"m_{tid}_{new_val}"
        else:
            t1, t2 = random.sample(tids, 2)
            new_genes[t1], new_genes[t2] = new_genes[t2], new_genes[t1]
            mid = f"s_{t1}_{t2}"
        return new_genes, mid

    def run_with_time_limit(self, max_generations, pop_size, time_limit, algo_name="Sun-Wu-2023"):
        start_time = time.perf_counter()

        # 1. 初始解：直接使用您强大的构造求解器！
        init_genes = generate_feasible_solution(self.data)
        v, f = self._get_metrics(init_genes)

        self.best_ind.genes = init_genes.copy()
        self.best_ind.fitness = f
        current_genes = init_genes.copy()

        anytime_trace = [(0, float(f))]
        current_gen = 0

        # --- Phase 1: 消冲突 ---
        while v > 0 and current_gen < max_generations and (time.perf_counter() - start_time < time_limit):
            current_gen += 1
            best_cand = None
            min_v = float('inf')

            for _ in range(50):
                ng, mid = self._generate_neighbor(current_genes)
                nv, nf = self._get_metrics(ng)
                if mid not in self.tabu_list or nf > self.best_ind.fitness:
                    if nv < min_v:
                        min_v, best_cand = nv, (ng, mid, nv, nf)

            if best_cand:
                current_genes, mid, v, f = best_cand
                self.tabu_list.append(mid)
                if f > self.best_ind.fitness:
                    self.best_ind.fitness, self.best_ind.genes = f, current_genes.copy()

            anytime_trace.append((time.perf_counter() - start_time, float(self.best_ind.fitness)))

        # --- Phase 2: 优质量 ---
        if v == 0:
            while current_gen < max_generations and (time.perf_counter() - start_time < time_limit):
                current_gen += 1
                best_cand = None
                max_f = -float('inf')

                for _ in range(50):
                    ng, mid = self._generate_neighbor(current_genes)
                    nv, nf = self._get_metrics(ng)
                    if nv == 0 and (mid not in self.tabu_list or nf > self.best_ind.fitness):
                        if nf > max_f:
                            max_f, best_cand = nf, (ng, mid, nv, nf)

                if best_cand:
                    current_genes, mid, v, f = best_cand
                    self.tabu_list.append(mid)
                    if f > self.best_ind.fitness:
                        self.best_ind.fitness, self.best_ind.genes = f, current_genes.copy()
                        anytime_trace.append((time.perf_counter() - start_time, float(f)))

        duration = time.perf_counter() - start_time
        return self.best_ind, current_gen, duration, anytime_trace