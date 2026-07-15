"""
True 冲突驱动变异算子 (True Conflict-Directed Mutation, TrueCDM)
功能：利用冲突检测结果，定向对冲突严重的逻辑单元进行局部搜索变异，实现搜索空间的精准压缩。
"""

import random
from collections import Counter
from algorithms.combined_repair import CombinedRepair
from evaluation.constraints import ConstraintChecker

class TrueCombinedConflictDirectedMutation:
    def __init__(self, rate, helper, data, evaluator, tries=8):
        """
        :param tries: 局部搜索尝试次数。在变异时，会尝试生成 tries 个随机基因并保留冲突下降最明显的。
        """
        self.rate = rate
        self.helper = helper
        self.data = data
        self.evaluator = evaluator
        self.tries = tries

        self.repair = CombinedRepair(helper)
        self.checker = ConstraintChecker(data)

        # 1. 性能优化：缓存单元键名列表，避免在变异循环中频繁转换 list [引用建议四]
        self.unit_keys = list(helper.units.keys())

    def mutate(self, chromosome):
        """
        执行冲突定向变异逻辑
        """
        if random.random() > self.rate:
            return

        # 2. 预修复：确保变异前的染色体满足合班一致性护盾
        self.repair.repair(chromosome)

        # 3. 映射到单元基因空间并获取当前冲突报告
        u_genes = self.helper.to_unit_genes(chromosome.genes)
        report = self.checker.check_all(chromosome.genes)
        conflict_tasks = report.get("conflict_tasks", [])

        # 4. 冲突驱动的目标选择 (Conflict-Directed Target Selection)
        if not conflict_tasks:
            # 4.1 无冲突时退化为逻辑单元级的随机变异
            target_unit = random.choice(self.unit_keys)
        else:
            # 4.2 有冲突时，通过 Counter 统计冲突频率最高的单元
            # 增加安全读取逻辑，确保 None 不会进入 Counter 污染目标 [引用建议三]
            unit_counts = Counter(
                self.helper.task_to_unit[t]
                for t in conflict_tasks
                if t in self.helper.task_to_unit
            )

            if unit_counts:
                # 选取引发冲突次数最多的逻辑单元作为变异目标
                target_unit = unit_counts.most_common(1)[0][0]
            else:
                target_unit = random.choice(self.unit_keys)

        # 5. 安全性检查：确保目标单元在当前基因组中存在 [引用建议二]
        if target_unit not in u_genes:
            return

        best_gene = u_genes[target_unit]
        best_v = report["total"]

        # 6. 启发式局部搜索变异 (Guided Local Search Mutation)
        for _ in range(self.tries):
            cand_gene = self.helper.sample_random_gene()

            # 6.1 副本隔离：防止在局部搜索尝试中污染原始染色体 [引用建议二]
            cand_u = dict(u_genes)
            cand_u[target_unit] = cand_gene

            # 6.2 将变异后的单元基因转换回任务基因并检测冲突
            cand_task_genes = self.helper.to_task_genes(cand_u)
            v_curr = self.checker.check_all(cand_task_genes)["total"]

            # 6.3 变异规则：仅保留冲突数量更少的基因 [引用建议五]
            # 注：Soft Score 的优化交由后续的 Tabu Search 阶段处理
            if v_curr < best_v:
                best_v = v_curr
                best_gene = cand_gene
                # 若已实现局部冲突消除，则提前结束搜索
                if v_curr == 0:
                    break

        # 7. 写回变异结果并重新执行护盾修复
        u_genes[target_unit] = best_gene
        chromosome.genes = self.helper.to_task_genes(u_genes)
        self.repair.repair(chromosome)

        # 8. 状态失效：基因已改变，适应度必须重算 [引用建议七]
        chromosome.fitness = None