"""
合班一致性修复模块 (Combined-class Repair Module)
功能：充当 Feasibility Shield，确保同一逻辑单元（Logical Unit）内的所有基因在进化后保持绝对同步。
"""


class CombinedRepair:
    def __init__(self, helper):
        """
        :param helper: CombinedOperatorsHelper 实例，包含单元映射关系
        """
        self.helper = helper

    def repair(self, chromosome):
        """
        对染色体执行一致性修复
        """
        # 获取原始基因字典的引用
        genes = chromosome.genes

        # 遍历由 Helper 定义的所有逻辑单元 (Logical Units)
        for uid, tasks in self.helper.units.items():

            # 1. 单任务单元无需修复，直接跳过
            if len(tasks) <= 1:
                continue

            # 2. 寻找该单元的代表基因 (Representative Gene)
            # 采用更稳健的遍历寻找方式，替代 next()，增强调试友好度 [引用建议二]
            rep_gene = None
            for tid in tasks:
                if tid in genes:
                    rep_gene = genes[tid]
                    break

            # 3. 如果该单元在当前基因组中无对应数据，则跳过
            if rep_gene is None:
                continue

            # 4. 同步执行：将单元内所有任务强制对齐至代表基因
            for tid in tasks:
                if tid in genes:
                    genes[tid] = rep_gene

        # 5. 防污染处理：通过 dict() 创建副本，断开引用链 [引用建议四]
        chromosome.genes = dict(genes)

        # 6. 状态重置：基因已变，适应度必须失效，强制下一轮重新 evaluate
        chromosome.fitness = None

        return chromosome