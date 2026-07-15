"""
合班单元级算子 (Combined-Class Block Operators)
功能：实现任务到逻辑单元的映射，并提供基于单元的块交叉操作，实现搜索空间降维。
"""

import random
from algorithms.combined_repair import CombinedRepair

class CombinedOperatorsHelper:
    """
    辅助类：管理任务与逻辑单元 (Unit) 之间的双向映射。
    """
    def __init__(self, data):
        self.data = data
        self.units = {}         # {unit_id: [task_ids]}
        self.task_to_unit = {}  # {task_id: unit_id}
        self._build()

    def _build(self):
        """
        构建逻辑单元映射：将合班任务聚合成组，普通任务独立成组。
        """
        # 1. 构建合班单元 (Combined Classes)
        # 采用更简洁的 dict 迭代逻辑，移除冗余的 set 判断 [引用建议二]
        for gid, tids in getattr(self.data, "combined_classes", {}).items():
            uid = f"G_{gid}"
            self.units[uid] = list(tids)
            for t in tids:
                self.task_to_unit[t] = uid

        # 2. 构建普通任务单元 (Single Tasks)
        for t in self.data.tasks.keys():
            if t not in self.task_to_unit:
                uid = f"T_{t}"
                self.units[uid] = [t]
                self.task_to_unit[t] = uid

    def to_unit_genes(self, task_genes):
        """将任务级基因转换为单元级基因"""
        res = {}
        for uid, tids in self.units.items():
            for t in tids:
                if t in task_genes:
                    res[uid] = task_genes[t]
                    break
        return res

    def to_task_genes(self, unit_genes):
        """将单元级基因转换回任务级基因，增加 ghost task 保护 [引用建议三]"""
        res = {}
        for uid, g in unit_genes.items():
            for t in self.units.get(uid, []):
                # 仅写入数据集中存在的合法任务
                if t in self.data.tasks:
                    res[t] = g
        return res

    def sample_random_gene(self):
        """
        生成随机基因：增加异常处理兜底，防止非数字 ID 导致崩溃 [引用建议一]
        """
        slot = random.randint(0, getattr(self.data, "total_slots", 25) - 1)
        rooms = list(self.data.classrooms.keys())

        if not rooms:
            room_id = 0
        else:
            r_key = random.choice(rooms)
            room = self.data.classrooms[r_key]
            # 尝试获取整型 ID，若为字符串或其他格式则使用 hash 兜底
            try:
                room_id = int(room.classroom_id)
            except (ValueError, TypeError, AttributeError):
                room_id = hash(r_key) & 0xFFFF

        return (slot << 16) | room_id


class CombinedBlockCrossover:
    """
    块交叉算子：在逻辑单元层面执行均匀交叉，实现 O(|units|) 的低复杂度计算。
    """
    def __init__(self, rate, helper, enable_repair=True):
        self.rate = rate
        self.helper = helper
        self.enable_repair = enable_repair
        self.repair_module = CombinedRepair(helper)

    def cross(self, p1, p2):
        """
        执行交叉操作
        """
        if random.random() > self.rate:
            return p1.clone(), p2.clone()

        # 1. 转换到单元空间
        u1 = self.helper.to_unit_genes(p1.genes)
        u2 = self.helper.to_unit_genes(p2.genes)

        c1_u = {}
        c2_u = {}

        # 2. 单元级均匀交叉：确保每个 Unit 至少有一个基因来源 [引用建议四]
        for uid in set(u1.keys()) | set(u2.keys()):
            g1 = u1.get(uid, u2.get(uid))
            g2 = u2.get(uid, u1.get(uid))

            if random.random() < 0.5:
                c1_u[uid], c2_u[uid] = g1, g2
            else:
                c1_u[uid], c2_u[uid] = g2, g1

        # 3. 转换回染色体
        res1 = p1.clone()
        res2 = p2.clone()
        res1.genes = self.helper.to_task_genes(c1_u)
        res2.genes = self.helper.to_task_genes(c2_u)

        # 4. 执行一致性修复 (Feasibility Shield) [引用建议五]
        if self.enable_repair:
            self.repair_module.repair(res1)
            self.repair_module.repair(res2)

        # 5. 显式失效 Fitness，确保重新评估 [引用建议七]
        res1.fitness = None
        res2.fitness = None

        return res1, res2