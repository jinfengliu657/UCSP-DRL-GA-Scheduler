"""
基于单元的禁忌搜索算子 (Unit-based Tabu Search)
功能：大邻域搜索 (Large Move Local Search)，解决 Aliasing Bug，严格深挖软约束。
"""

import random
from collections import deque
from algorithms.combined_repair import CombinedRepair

# 避免循环内高频反射创建类，极大降低开销
class DummyChromosome:
    def __init__(self, genes):
        self.genes = genes
        self.fitness = None

class CombinedTabuSearch:
    def __init__(self, evaluator, helper, tabu_size=20):
        self.evaluator = evaluator
        self.helper = helper
        self.tabu_list = deque(maxlen=tabu_size)
        self.repair = CombinedRepair(helper)

    def run(self, chromosome, max_steps=10):
        self.tabu_list.clear()

        if chromosome.fitness is None:
            chromosome.fitness = self.evaluator.evaluate(chromosome.genes)

        best_f = chromosome.fitness
        # 核心修复: 必须使用 copy() 断开引用，防止 Aliasing Bug 污染历史最优解
        best_genes = chromosome.genes.copy()

        current_u = self.helper.to_unit_genes(chromosome.genes)
        u_keys = list(current_u.keys())

        for _ in range(max_steps):
            step_best_fit = -float("inf")
            step_best_u = None
            step_move = None
            step_best_genes = None

            # 核心修复: 邻域候选从 5 扩大到 8，适配更广阔的 Unit 空间
            for _ in range(8):
                target_u = random.choice(u_keys)
                new_g = self.helper.sample_random_gene()

                move_key = (target_u, new_g >> 16, new_g & 0xFFFF)

                neighbor_u = dict(current_u)
                neighbor_u[target_u] = new_g

                task_genes = self.helper.to_task_genes(neighbor_u)

                # 使用轻量级 Dummy 实例，且严格 copy 防污染
                tmp = DummyChromosome(task_genes.copy())
                self.repair.repair(tmp)
                task_genes = tmp.genes.copy()

                fit = self.evaluator.evaluate(task_genes)

                # Aspiration Rule (藐视准则)
                if move_key in self.tabu_list and fit <= best_f:
                    continue

                if fit > step_best_fit:
                    step_best_fit = fit
                    step_best_u = neighbor_u
                    step_move = move_key
                    # 工业级防引用污染
                    step_best_genes = task_genes.copy()

            if step_best_u is None:
                break

            current_u = step_best_u
            self.tabu_list.append(step_move)

            if step_best_fit > best_f:
                best_f = step_best_fit
                # 更新时同样必须 copy()
                best_genes = step_best_genes.copy()

        chromosome.genes = best_genes
        chromosome.fitness = best_f

        return chromosome