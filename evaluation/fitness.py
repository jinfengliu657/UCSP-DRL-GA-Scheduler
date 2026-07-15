# """
# 适应度函数 (Metrics Implemented)
# 文件名: evaluation/fitness.py
# """
"""
功能:
1. 计算适应度分数 (含硬约束惩罚)。
2. 提供详细的物理指标 (Daily_Var, Util_Avg, Interval_Rate, Build_Conc_Rate)。
"""
"""
1. IndexError: 增加对 day (0-4) 和 period (0-4) 的边界检查，防止越界崩溃。
2. 保持所有指标计算逻辑。
"""
import numpy as np
from typing import Dict, List
from collections import defaultdict
from .constraints import ConstraintChecker


class FitnessEvaluator:
    def __init__(self, weights: Dict[str, float] = None):
        # 默认权重
        self.weights = weights or {
            'f_daily': 0.20, 'f_interval': 0.20, 'f_room': 0.10,
            'f_util': 0.30, 'f_build': 0.20
        }
        self.constraint_checker = None
        self.data = None
        self.scaling_factor = 100.0

    def set_data(self, data):
        self.data = data
        self.constraint_checker = ConstraintChecker(data)

    def evaluate(self, genes: dict) -> float:
        # 硬约束检查 (违约即死)
        violations = self.constraint_checker.check_all(genes)
        if violations['total'] > 0:
            return -1000.0 * violations['total']

        # 软约束计算
        scores = self._calculate_raw_scores(genes)
        fitness = sum(scores[k] * self.weights.get(k, 0) for k in scores)
        return fitness * self.scaling_factor

    def evaluate_detail(self, genes: dict) -> Dict:
        """详细评估"""
        violations = self.constraint_checker.check_all(genes)
        raw_scores = self._calculate_raw_scores(genes)
        weighted_sum = sum(raw_scores[k] * self.weights.get(k, 0) for k in raw_scores)

        final_fitness = weighted_sum * self.scaling_factor
        if violations['total'] > 0:
            final_fitness = -1000.0 * violations['total']

        return {
            'fitness': final_fitness,
            'violations': violations,
            'raw_scores': raw_scores
        }

    def calculate_quality_metrics(self, genes: dict) -> Dict[str, float]:
        """
        [排课质量核心指标 - 论文数据源]
        包含了您要求的四个物理指标，增加了边界保护
        """
        # 1. 课程分布均匀度 (Daily Variance)
        class_daily = defaultdict(lambda: [0] * 5)
        # 2. 教室资源利用率 (Utilization)
        slot_room_stu = defaultdict(int)
        # 3. 间隔满意度 (Interval)
        course_slots = defaultdict(list)
        # 4. 教学楼集中度 (Building Concentration)
        class_block_buildings = defaultdict(lambda: defaultdict(set))

        for task_id, gene in genes.items():
            if task_id not in self.data.tasks: continue
            task = self.data.tasks[task_id]
            slot = gene >> 16
            room_id = gene & 0xFFFF

            # [关键修复] 边界检查
            if slot < 0 or slot >= 25: continue  # 跳过非法时间槽

            day = slot // 5
            period = slot % 5

            # F1 Data
            if 0 <= day < 5:
                class_daily[task.class_id][day] += 1

            # F2 Data
            key = (task.class_id, task.course_id)
            course_slots[key].append(day)

            # F4 Data
            slot_room_stu[(slot, room_id)] += task.student_count

            # F5 Data (Building)
            r_key = str(room_id)
            if r_key not in self.data.classrooms and r_key.zfill(6) in self.data.classrooms:
                r_key = r_key.zfill(6)
            if r_key in self.data.classrooms:
                bid = self.data.classrooms[r_key].teach_building_id
                # 仅统计上午(0,1)和下午(2,3)
                if period < 4 and 0 <= day < 5:
                    blk = day * 2 + (1 if period >= 2 else 0)
                    class_block_buildings[task.class_id][blk].add(bid)

        # --- Metric 1: Daily Variance ---
        variances = [np.var(counts) for counts in class_daily.values()]
        metric_daily_var = np.mean(variances) if variances else 0.0

        # --- Metric 2: Utilization ---
        util_ratios = []
        for (slot, rid), stu in slot_room_stu.items():
            r_key = str(rid)
            if r_key not in self.data.classrooms and r_key.zfill(6) in self.data.classrooms:
                r_key = r_key.zfill(6)
            if r_key in self.data.classrooms:
                cap = self.data.classrooms[r_key].capacity
                if cap > 0: util_ratios.append(min(1.0, stu / cap))
        metric_util = np.mean(util_ratios) if util_ratios else 0.0

        # --- Metric 3: Interval Rate ---
        good_int = 0;
        total_int = 0
        for days in course_slots.values():
            days.sort()
            if len(days) < 2: continue
            for i in range(len(days) - 1):
                total_int += 1
                if (days[i + 1] - days[i]) >= 2: good_int += 1
        metric_interval = (good_int / total_int) if total_int > 0 else 1.0

        # --- Metric 4: Building Concentration Rate ---
        good_blks = 0;
        total_blks = 0
        for blks in class_block_buildings.values():
            for bset in blks.values():
                total_blks += 1
                if len(bset) <= 1: good_blks += 1
        metric_build = (good_blks / total_blks) if total_blks > 0 else 1.0

        return {
            'Daily_Var': metric_daily_var,
            'Util_Avg': metric_util,
            'Interval_Rate': metric_interval,
            'Build_Conc_Rate': metric_build
        }

    def _calculate_raw_scores(self, genes: dict) -> Dict[str, float]:
        """原始分计算 (用于GA进化)"""
        f_daily = 0.0;
        f_interval = 0.0;
        f_room = 0.0;
        f_util = 0.0;
        f_build = 0.0
        class_daily = defaultdict(lambda: defaultdict(int))
        course_sess = defaultdict(list)
        teaching_units = {};
        processed = set()
        class_blocks = defaultdict(lambda: defaultdict(set))

        for task_id, gene in genes.items():
            if task_id not in self.data.tasks: continue
            task = self.data.tasks[task_id]
            slot = gene >> 16;
            room = gene & 0xFFFF

            # [关键修复] 边界检查
            if slot < 0 or slot >= 25: continue

            day = slot // 5;
            period = slot % 5

            class_daily[task.class_id][day] += 1
            course_sess[(task.class_id, task.course_id)].append({'slot': slot, 'room': room})

            r_key = str(room)
            if r_key not in self.data.classrooms: r_key = r_key.zfill(6)
            if r_key in self.data.classrooms:
                bid = self.data.classrooms[r_key].teach_building_id
                if period < 4:
                    blk = day * 2 + (1 if period >= 2 else 0)
                    class_blocks[task.class_id][blk].add(bid)

            if task_id in processed: continue
            uid = f"T_{task_id}"
            t_stu = task.student_count
            if task.course_class_id and task.course_class_id in self.data.combined_classes:
                uid = f"G_{task.course_class_id}"
                for tid in self.data.combined_classes[task.course_class_id]:
                    if tid in self.data.tasks:
                        t_stu += self.data.tasks[tid].student_count
                        processed.add(tid)
            else:
                processed.add(task_id)
            teaching_units[uid] = {'stu': t_stu, 'room': r_key}

        for dmap in class_daily.values():
            loads = [dmap.get(d, 0) for d in range(5)]
            f_daily += max(0, 1.0 - np.std(loads))

        for sess in course_sess.values():
            n_r = len(set(s['room'] for s in sess))
            if n_r == 1:
                f_room += 1.0
            elif n_r == 2:
                f_room += 0.5
            slots = sorted([s['slot'] for s in sess])
            if len(slots) > 1:
                diffs = [slots[i + 1] - slots[i] for i in range(len(slots) - 1)]
                if min(diffs) >= 10:
                    f_interval += 1.0
                elif min(diffs) >= 5:
                    f_interval += 0.2
            else:
                f_interval += 1.0

        for unit in teaching_units.values():
            rk = unit['room']
            if rk in self.data.classrooms:
                cap = self.data.classrooms[rk].capacity
                if cap > 0:
                    ratio = unit['stu'] / cap
                    if 0.4 <= ratio <= 0.6:
                        f_util += 1.0
                    elif ratio < 0.4:
                        f_util += ratio / 0.4
                    elif ratio <= 0.8:
                        f_util += (1 - ratio) / 0.4
                    else:
                        f_util += 0.2

        for blks in class_blocks.values():
            for bset in blks.values():
                if len(bset) <= 1: f_build += 1.0

        return {'f_daily': f_daily, 'f_interval': f_interval, 'f_room': f_room, 'f_util': f_util, 'f_build': f_build}

# import numpy as np
# from typing import Dict, List
# from collections import defaultdict
# from .constraints import ConstraintChecker
#
#
# class FitnessEvaluator:
#     def __init__(self, weights: Dict[str, float] = None):
#         # 默认权重
#         self.weights = weights or {
#             'f_daily': 0.20, 'f_interval': 0.20, 'f_room': 0.10,
#             'f_util': 0.30, 'f_build': 0.20
#         }
#         self.constraint_checker = None
#         self.data = None
#         self.scaling_factor = 100.0
#
#     def set_data(self, data):
#         self.data = data
#         self.constraint_checker = ConstraintChecker(data)
#
#     def evaluate(self, genes: dict) -> float:
#         # 硬约束检查 (违约即死)
#         violations = self.constraint_checker.check_all(genes)
#         if violations['total'] > 0:
#             return -1000.0 * violations['total']
#
#         # 软约束计算
#         scores = self._calculate_raw_scores(genes)
#         fitness = sum(scores[k] * self.weights.get(k, 0) for k in scores)
#         return fitness * self.scaling_factor
#
#     def evaluate_detail(self, genes: dict) -> Dict:
#         """详细评估"""
#         violations = self.constraint_checker.check_all(genes)
#         raw_scores = self._calculate_raw_scores(genes)
#         weighted_sum = sum(raw_scores[k] * self.weights.get(k, 0) for k in raw_scores)
#
#         final_fitness = weighted_sum * self.scaling_factor
#         if violations['total'] > 0:
#             final_fitness = -1000.0 * violations['total']
#
#         return {
#             'fitness': final_fitness,
#             'violations': violations,
#             'raw_scores': raw_scores
#         }
#
#     def calculate_quality_metrics(self, genes: dict) -> Dict[str, float]:
#         """
#         [排课质量核心指标 - 论文数据源]
#         """
#         # 1. 课程分布均匀度 (Daily Variance)
#         # 计算每个班级周一至周五课时数的方差，取平均。
#         class_daily = defaultdict(lambda: [0] * 5)
#         for task_id, gene in genes.items():
#             if task_id not in self.data.tasks: continue
#             task = self.data.tasks[task_id]
#             day = (gene >> 16) // 5
#             if 0 <= day < 5:
#                 class_daily[task.class_id][day] += 1
#
#         variances = [np.var(counts) for counts in class_daily.values()]
#         metric_daily_var = np.mean(variances) if variances else 0.0
#
#         # 2. 教室资源利用率 (Utilization)
#         # 统计每个被占用的(Slot, Room)的座位利用率。
#         slot_room_stu = defaultdict(int)
#         for task_id, gene in genes.items():
#             if task_id not in self.data.tasks: continue
#             task = self.data.tasks[task_id]
#             slot = gene >> 16
#             room_id = gene & 0xFFFF
#             slot_room_stu[(slot, room_id)] += task.student_count
#
#         util_ratios = []
#         for (slot, rid), stu in slot_room_stu.items():
#             r_key = str(rid)
#             # 兼容 key
#             if r_key not in self.data.classrooms and r_key.zfill(6) in self.data.classrooms:
#                 r_key = r_key.zfill(6)
#             if r_key in self.data.classrooms:
#                 cap = self.data.classrooms[r_key].capacity
#                 if cap > 0: util_ratios.append(min(1.0, stu / cap))
#
#         metric_util = np.mean(util_ratios) if util_ratios else 0.0
#
#         # 3. 间隔满意度 (Interval > 1 day)
#         course_slots = defaultdict(list)
#         for task_id, gene in genes.items():
#             if task_id not in self.data.tasks: continue
#             task = self.data.tasks[task_id]
#             key = (task.class_id, task.course_id)  # 同一班级同一课
#             course_slots[key].append((gene >> 16) // 5)  # 存 Day Index
#
#         good = 0;
#         total = 0
#         for days in course_slots.values():
#             days.sort()
#             if len(days) < 2: continue
#             for i in range(len(days) - 1):
#                 total += 1
#                 if (days[i + 1] - days[i]) >= 2:  # 间隔>=2天 (如周一到周三)
#                     good += 1
#
#         metric_interval = (good / total) if total > 0 else 1.0
#
#         return {
#             'Daily_Var': metric_daily_var,
#             'Util_Avg': metric_util,
#             'Interval_Rate': metric_interval
#         }
#
#     def _calculate_raw_scores(self, genes: dict) -> Dict[str, float]:
#         """
#         [进化驱动] 计算软约束得分，用于 GA 寻优
#         """
#         f_daily = 0.0
#         f_interval = 0.0
#         f_room = 0.0
#         f_util = 0.0
#         f_build = 0.0
#
#         class_daily = defaultdict(lambda: defaultdict(int))
#         course_sess = defaultdict(list)
#         teaching_units = {}
#         processed = set()
#         class_blocks = defaultdict(lambda: defaultdict(set))  # F5
#
#         for task_id, gene in genes.items():
#             if task_id not in self.data.tasks: continue
#             task = self.data.tasks[task_id]
#             slot = gene >> 16
#             room = gene & 0xFFFF
#             day = slot // 5
#             period = slot % 5
#
#             # F1
#             class_daily[task.class_id][day] += 1
#             # F2/F3
#             course_sess[(task.class_id, task.course_id)].append({'slot': slot, 'room': room})
#
#             # F5
#             r_key = str(room)
#             if r_key not in self.data.classrooms: r_key = r_key.zfill(6)
#             if r_key in self.data.classrooms:
#                 bid = self.data.classrooms[r_key].teach_building_id
#                 if period < 4:
#                     blk = day * 2 + (1 if period >= 2 else 0)
#                     class_blocks[task.class_id][blk].add(bid)
#
#             # F4 Aggregation
#             if task_id in processed: continue
#
#             uid = f"T_{task_id}"
#             t_stu = task.student_count
#
#             if task.course_class_id and task.course_class_id in self.data.combined_classes:
#                 uid = f"G_{task.course_class_id}"
#                 for tid in self.data.combined_classes[task.course_class_id]:
#                     if tid in self.data.tasks:
#                         t_stu += self.data.tasks[tid].student_count
#                         processed.add(tid)
#             else:
#                 processed.add(task_id)
#
#             teaching_units[uid] = {'stu': t_stu, 'room': r_key}
#
#         # Calculate Scores
#         # F1
#         for dmap in class_daily.values():
#             loads = [dmap.get(d, 0) for d in range(5)]
#             f_daily += max(0, 1.0 - np.std(loads))
#
#         # F2/F3
#         for sess in course_sess.values():
#             # F3 Room
#             n_r = len(set(s['room'] for s in sess))
#             if n_r == 1:
#                 f_room += 1.0
#             elif n_r == 2:
#                 f_room += 0.5
#             # F2 Interval
#             slots = sorted([s['slot'] for s in sess])
#             if len(slots) > 1:
#                 diffs = [slots[i + 1] - slots[i] for i in range(len(slots) - 1)]
#                 if min(diffs) >= 10:
#                     f_interval += 1.0
#                 elif min(diffs) >= 5:
#                     f_interval += 0.2
#             else:
#                 f_interval += 1.0
#
#         # F4 Util
#         for unit in teaching_units.values():
#             rk = unit['room']
#             if rk in self.data.classrooms:
#                 cap = self.data.classrooms[rk].capacity
#                 if cap > 0:
#                     ratio = unit['stu'] / cap
#                     if 0.4 <= ratio <= 0.6:
#                         f_util += 1.0
#                     elif ratio < 0.4:
#                         f_util += ratio / 0.4
#                     elif ratio <= 0.8:
#                         f_util += (1 - ratio) / 0.4
#                     else:
#                         f_util += 0.2
#
#         # F5 Build
#         for blks in class_blocks.values():
#             for bset in blks.values():
#                 if len(bset) <= 1: f_build += 1.0
#
#         return {'f_daily': f_daily, 'f_interval': f_interval, 'f_room': f_room, 'f_util': f_util, 'f_build': f_build}
#
#     def _decode_gene(self, gene: int) -> tuple:
#         return (gene >> 16), (gene & 0xFFFF)
#
#
# # """
# # 适应度函数 - 多目标适应度评估 (论文终稿模型实现)
# #
# # 设计原则:
# # 1. 粒度细化: F_interval 使用时间槽(Slot)差值计算间隔。
# # 2. 聚合计算: F_util 按"授课单元"(Teaching Unit)聚合合班人数，避免重复计算。
# # 3. 扁平化权重: 支持5个独立子目标的权重配置。
# # 4. 无归一化: 采用累加制得分，保留大规模问题的梯度敏感性。
# # 5. 放大系数: K_scale = 100，增强选择压力。
# # """
# # import numpy as np
# # from typing import Dict, List, Set, Tuple
# # from collections import defaultdict
# #
# # from data.data_loader import ScheduleData
# # from .constraints import ConstraintChecker
# #
# #
# # class FitnessEvaluator:
# #     """
# #     多目标适应度评估器 (Final Paper Version)
# #     """
# #
# #     def __init__(self, weights: Dict[str, float] = None):
# #         """
# #         初始化适应度评估器
# #
# #         Args:
# #             weights: 扁平化的权重字典，默认配置:
# #             {
# #                 'f_daily': 0.2,    # F1 每日均衡
# #                 'f_interval': 0.2, # F2 课程间隔
# #                 'f_room': 0.1,     # F3 同室一致
# #                 'f_util': 0.3,     # F4 资源利用
# #                 'f_build': 0.2     # F5 楼宇集中
# #             }
# #         """
# #         # 论文基准权重配置
# #         default_weights = {
# #             'f_daily': 0.20,
# #             'f_interval': 0.20,
# #             'f_room': 0.10,
# #             'f_util': 0.30,
# #             'f_build': 0.20
# #         }
# #         self.weights = weights or default_weights
# #         self.constraint_checker = None
# #         self.data = None
# #         self.scaling_factor = 100.0  # 放大系数
# #
# #     def set_data(self, data: ScheduleData):
# #         """设置数据源并初始化约束检查器"""
# #         self.data = data
# #         self.constraint_checker = ConstraintChecker(data)
# #
# #     def evaluate(self, genes: dict) -> float:
# #         """
# #         计算全局适应度 F(x)
# #
# #         逻辑:
# #         1. 若存在硬约束违约 -> 返回 -1000 * 违约数
# #         2. 若无违约 -> 返回 K * sum(w_i * f_i)
# #         """
# #         # 1. 硬约束检查 (违约即死)
# #         violations = self.constraint_checker.check_all(genes)
# #         if violations['total'] > 0:
# #             # 硬约束惩罚 M = 1000
# #             return -1000.0 * violations['total']
# #
# #         # 2. 计算各子目标原始得分 (累加制)
# #         scores = self._calculate_raw_scores(genes)
# #
# #         # 3. 加权求和并放大
# #         fitness = (
# #                 scores['f_daily'] * self.weights.get('f_daily', 0) +
# #                 scores['f_interval'] * self.weights.get('f_interval', 0) +
# #                 scores['f_room'] * self.weights.get('f_room', 0) +
# #                 scores['f_util'] * self.weights.get('f_util', 0) +
# #                 scores['f_build'] * self.weights.get('f_build', 0)
# #         )
# #
# #         return fitness * self.scaling_factor
# #
# #     def evaluate_detail(self, genes: dict) -> Dict:
# #         """详细评估，用于日志记录和论文图表数据生成"""
# #         violations = self.constraint_checker.check_all(genes)
# #
# #         # 无论是否违约，都计算一下软约束得分以便分析
# #         raw_scores = self._calculate_raw_scores(genes)
# #         weighted_sum = sum(raw_scores[k] * self.weights.get(k, 0) for k in raw_scores)
# #
# #         final_fitness = weighted_sum * self.scaling_factor
# #         if violations['total'] > 0:
# #             final_fitness = -1000.0 * violations['total']
# #
# #         return {
# #             'fitness': final_fitness,
# #             'violations': violations,
# #             'raw_scores': raw_scores,  # 各项原始累加分
# #             'weighted_scores': {k: v * self.weights.get(k, 0) * self.scaling_factor for k, v in raw_scores.items()}
# #         }
# #
# #     def _calculate_raw_scores(self, genes: dict) -> Dict[str, float]:
# #         """
# #         计算所有软约束子目标函数值 (一次遍历，提高效率)
# #         """
# #         # --- 1. 数据预处理 ---
# #         # 班级每日课时数 {class_id: {day: count}}
# #         class_daily_counts = defaultdict(lambda: defaultdict(int))
# #
# #         # 课程维度数据 (用于间隔和教室一致性) {(class_id, course_id): [{'slot': s, 'room': r}]}
# #         course_sessions = defaultdict(list)
# #
# #         # 教学楼维度数据 {class_id: {block_idx: {buildings}}}
# #         # Block定义: 0-9 (5天 * 2半天)
# #         # 上午(Period 0,1) -> Block 2*d + 0
# #         # 下午(Period 2,3) -> Block 2*d + 1
# #         class_block_buildings = defaultdict(lambda: defaultdict(set))
# #
# #         # 授课单元(Teaching Unit)维度数据 {unit_id: {'students': sum_stu, 'room': r_id}}
# #         # 对于合班: unit_id = 'G_' + course_class_id
# #         # 对于单班: unit_id = 'T_' + task_id
# #         teaching_units = {}
# #
# #         processed_tasks = set()
# #
# #         # 遍历所有任务
# #         for task_id, task in self.data.tasks.items():
# #             if task_id not in genes:
# #                 continue
# #
# #             # 解码
# #             time_slot, room_id = self._decode_gene(genes[task_id])
# #             day = time_slot // 5
# #             period = time_slot % 5
# #
# #             # --- F1 数据收集: 班级每日课时 ---
# #             class_daily_counts[task.class_id][day] += 1
# #
# #             # --- F2 & F3 数据收集: 课程维度 ---
# #             course_sessions[(task.class_id, task.course_id)].append({
# #                 'slot': time_slot,
# #                 'room': room_id
# #             })
# #
# #             # --- F5 数据收集: 教学楼 ---
# #             room_key = self._get_classroom_key(room_id)
# #             if room_key in self.data.classrooms:
# #                 b_id = self.data.classrooms[room_key].teach_building_id
# #                 # 仅处理上午和下午，晚上暂不计入跨楼惩罚(通常晚上课少且独立)
# #                 if period < 4:
# #                     is_pm = 1 if period >= 2 else 0
# #                     block_idx = day * 2 + is_pm
# #                     class_block_buildings[task.class_id][block_idx].add(b_id)
# #
# #             # --- F4 数据收集: 授课单元聚合 ---
# #             # 如果该任务属于某个合班组，且该组尚未处理
# #             if task_id in processed_tasks:
# #                 continue
# #
# #             # 确定Unit ID和总人数
# #             unit_id = None
# #             total_students = 0
# #
# #             if task.course_class_id and task.course_class_id in self.data.combined_classes:
# #                 # 合班逻辑
# #                 unit_id = f"G_{task.course_class_id}"
# #                 # 聚合该组所有任务的学生数
# #                 group_task_ids = self.data.combined_classes[task.course_class_id]
# #                 for tid in group_task_ids:
# #                     if tid in self.data.tasks:
# #                         total_students += self.data.tasks[tid].student_count
# #                         processed_tasks.add(tid)  # 标记已处理
# #             else:
# #                 # 单班逻辑
# #                 unit_id = f"T_{task_id}"
# #                 total_students = task.student_count
# #                 processed_tasks.add(task_id)
# #
# #             teaching_units[unit_id] = {
# #                 'students': total_students,
# #                 'room_key': room_key
# #             }
# #
# #         # --- 2. 计算各指标得分 ---
# #
# #         # F1: 每日课程均衡度 (Sum of max(0, 1 - std/2))
# #         f_daily = 0.0
# #         for _, daily_map in class_daily_counts.items():
# #             # 补全一周5天的数据
# #             loads = [daily_map.get(d, 0) for d in range(5)]
# #             std_dev = np.std(loads)
# #             f_daily += max(0, 1.0 - std_dev / 2.0)
# #
# #         # F2: 课程时间间隔 (Sum of step function) & F3: 同室一致性 (Sum of step function)
# #         f_interval = 0.0
# #         f_room = 0.0
# #
# #         for _, sessions in course_sessions.items():
# #             # -- F3 --
# #             unique_rooms = set(s['room'] for s in sessions)
# #             n_rooms = len(unique_rooms)
# #             if n_rooms == 1:
# #                 f_room += 1.0
# #             elif n_rooms == 2:
# #                 f_room += 0.5
# #             else:
# #                 f_room += 0.0  # >2
# #
# #             # -- F2 --
# #             slots = sorted([s['slot'] for s in sessions])
# #             if len(slots) > 1:
# #                 # 计算最小间隔
# #                 min_diff = min([slots[i + 1] - slots[i] for i in range(len(slots) - 1)])
# #                 if min_diff >= 10:  # >= 1天 (最优)
# #                     f_interval += 1.0
# #                 elif min_diff >= 5:  # 相邻天 (次优)
# #                     f_interval += 0.2
# #                 else:  # 同一天 (极差)
# #                     f_interval += 0.0
# #             else:
# #                 # 每周只有一次课，视为完美间隔
# #                 f_interval += 1.0
# #
# #         # F4: 教室利用率 (基于聚合单元)
# #         f_util = 0.0
# #         for _, unit_data in teaching_units.items():
# #             r_key = unit_data['room_key']
# #             if r_key in self.data.classrooms:
# #                 cap = self.data.classrooms[r_key].capacity
# #                 if cap > 0:
# #                     ratio = unit_data['students'] / cap
# #                     # 分段线性函数
# #                     if 0.4 <= ratio <= 0.6:
# #                         f_util += 1.0
# #                     elif ratio < 0.4:
# #                         f_util += ratio / 0.4
# #                     elif ratio <= 0.8:
# #                         f_util += (1.0 - ratio) / 0.4
# #                     else:  # > 0.8
# #                         f_util += 0.2
# #
# #         # F5: 教学楼集中度 (半天粒度 Block)
# #         f_build = 0.0
# #         for _, blocks in class_block_buildings.items():
# #             for _, buildings in blocks.items():
# #                 if len(buildings) <= 1:
# #                     f_build += 1.0
# #                 else:
# #                     f_build += 0.0
# #
# #         return {
# #             'f_daily': f_daily,
# #             'f_interval': f_interval,
# #             'f_room': f_room,
# #             'f_util': f_util,
# #             'f_build': f_build
# #         }
# #
# #     def _decode_gene(self, gene: int) -> tuple:
# #         time_slot_id = gene >> 16
# #         classroom_id = gene & 0xFFFF
# #         return time_slot_id, classroom_id
# #
# #     def _get_classroom_key(self, classroom_id: int) -> str:
# #         str_id = str(classroom_id)
# #         if str_id in self.data.classrooms: return str_id
# #         padded_id = str_id.zfill(6)
# #         if padded_id in self.data.classrooms: return padded_id
# #         return str_id
# #
