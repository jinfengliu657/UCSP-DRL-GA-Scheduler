"""
硬约束检查器 - 检查排课方案是否满足所有硬约束 (Fixed Version)
包含详细冲突诊断功能与 TrueCDM 双向冲突追踪
"""
"""
硬约束检查器 (Ablation Study Ready Version)
文件名: evaluation/constraints.py
核心增强: 
1. 冲突双向追踪 (tid + prev_tid) -> 支持 TrueCDM 算子
2. 逻辑单元隔离 (curr_unit) -> 解决合班内部“自撞”伪冲突
3. 显式字段求和 -> 确保 L1 大规模实例下的稳定性
"""
from typing import Dict, List, Tuple
from data.data_loader import ScheduleData


class ConstraintChecker:
    def __init__(self, data: ScheduleData):
        self.data = data

    def check_all(self, genes: dict) -> Dict[str, int]:
        # 1. 初始化统计字典 (严格对齐论文 C1-C7)
        violations = {
            'fixed_time': 0,  # C7 (部分)
            'teacher': 0,  # C2
            'class': 0,  # C3
            'classroom': 0,  # C4
            'special_room': 0,  # C7 (部分)
            'capacity': 0,  # C5
            'combined_class': 0,  # C6
            'total': 0,
            'conflict_tasks': []  # ⭐ TrueCDM 核心导航数据
        }

        conflict_tasks_set = set()

        # 2. 建立合班映射 (Unit Mapping)
        task_group_map = {}
        for gid, tids in self.data.combined_classes.items():
            for t in tids:
                if t in self.data.tasks:
                    task_group_map[t] = gid

        # 3. 资源占用映射表 (记录 Unit_ID 以排除合班内部冲突)
        teacher_slot_map = {}  # {(teacher_id, slot): (unit_id, task_id)}
        class_slot_map = {}  # {(class_id, slot): task_id}
        room_slot_map = {}  # {(room_id, slot): (unit_id, task_id)}

        # 4. 逐任务扫描
        for tid, task in self.data.tasks.items():
            if tid not in genes: continue

            slot, rid = self._decode_gene(genes[tid])
            r_key = self._get_classroom_key(rid)
            curr_unit = task_group_map.get(tid, f"S_{tid}")

            # --- 教师冲突 (C2) ---
            t_key = (task.teacher_id, slot)
            if t_key in teacher_slot_map:
                u_id, prev_tid = teacher_slot_map[t_key]
                if u_id != curr_unit:
                    violations['teacher'] += 1
                    conflict_tasks_set.update([tid, prev_tid])
            else:
                teacher_slot_map[t_key] = (curr_unit, tid)

            # --- 班级冲突 (C3) ---
            c_key = (task.class_id, slot)
            if c_key in class_slot_map:
                violations['class'] += 1
                conflict_tasks_set.update([tid, class_slot_map[c_key]])
            else:
                class_slot_map[c_key] = tid

            # --- 教室冲突 (C4) ---
            rk_key = (rid, slot)
            if rk_key in room_slot_map:
                u_id, prev_tid = room_slot_map[rk_key]
                if u_id != curr_unit:
                    violations['classroom'] += 1
                    conflict_tasks_set.update([tid, prev_tid])
            else:
                room_slot_map[rk_key] = (curr_unit, tid)

            # --- 固定时间 (C7-1) ---
            if task.is_fixed and tid in self.data.fixed_time_tasks:
                if slot != self.data.fixed_time_tasks[tid]:
                    violations['fixed_time'] += 1
                    conflict_tasks_set.add(tid)

            # --- 特殊教室 (C7-2) ---
            if task.course_attr == '07':
                if r_key not in self.data.classrooms or self.data.classrooms[r_key].attr != '07':
                    violations['special_room'] += 1
                    conflict_tasks_set.add(tid)

            # --- 容量约束 (C5) ---
            if r_key in self.data.classrooms:
                if self.data.classrooms[r_key].capacity < task.student_count:
                    violations['capacity'] += 1
                    conflict_tasks_set.add(tid)

        # 5. 合班一致性显式检查 (C6)
        for gid, tids in self.data.combined_classes.items():
            valid = [t for t in tids if t in genes]
            if len(valid) < 2: continue
            base_g = genes[valid[0]]
            for t in valid[1:]:
                if genes[t] != base_g:
                    violations['combined_class'] += 1
                    conflict_tasks_set.update([valid[0], t])

        # 6. ⭐ 最终安全补丁: 显式求和 (按您的建议修改)
        violations['total'] = (
                violations['fixed_time']
                + violations['teacher']
                + violations['class']
                + violations['classroom']
                + violations['special_room']
                + violations['capacity']
                + violations['combined_class']
        )

        # 7. 导出冲突源
        violations['conflict_tasks'] = list(conflict_tasks_set)

        return violations

    def _decode_gene(self, gene: int) -> Tuple[int, int]:
        return gene >> 16, gene & 0xFFFF

    def _get_classroom_key(self, rid: int) -> str:
        s = str(rid)
        # 兼容性 Padding 逻辑
        return s if s in self.data.classrooms else s.zfill(6)

# """
# 硬约束检查器 - 检查排课方案是否满足所有硬约束 (Fixed Version)
# 包含详细冲突诊断功能
# """
# from typing import Dict, List, Tuple
# from collections import defaultdict
#
# from data.data_loader import ScheduleData
#
#
# class ConstraintChecker:
#     """硬约束检查器"""
#
#     # 对应论文中的7个硬约束
#     def __init__(self, data: ScheduleData):
#         self.data = data
#
#     def check_all(self, genes: dict) -> Dict[str, int]:
#         """
#         检查所有硬约束，返回违约次数
#         """
#         violations = {
#             'fixed_time': 0,  # C7 (Part 1)
#             'teacher': 0,  # C2
#             'class': 0,  # C3
#             'classroom': 0,  # C4
#             'special_room': 0,  # C7 (Part 2)
#             'capacity': 0,  # C5
#             'combined_class': 0,  # C6
#             'total': 0,
#             'conflict_tasks': []  # ⭐ 新增：专门记录发生冲突的任务
#         }
#
#         conflict_tasks_set = set()  # ⭐ 用于去重收集冲突任务
#
#         # 辅助结构: 合班组映射 {task_id -> group_id}
#         task_group_map = {}
#         for gid, tids in self.data.combined_classes.items():
#             valid_tids = [t for t in tids if t in self.data.tasks]
#             for t in valid_tids:
#                 task_group_map[t] = gid
#
#         # 冲突检测表 (修改为同时保存 task_id 以便追踪冲突双方)
#         teacher_schedule = {}  # {(teacher_id, time_slot): (unit_id, task_id)}
#         class_schedule = {}  # {(class_id, time_slot): task_id}
#         classroom_schedule = {}  # {(classroom_id, time_slot): (unit_id, task_id)}
#
#         for task_id, task in self.data.tasks.items():
#             if task_id not in genes:
#                 continue
#
#             time_slot, classroom_id = self._decode_gene(genes[task_id])
#             classroom_key = self._get_classroom_key(classroom_id)
#
#             # 当前任务的逻辑单元ID (如果是合班，则为合班ID，否则为任务ID)
#             current_unit_id = task_group_map.get(task_id, f"SINGLE_{task_id}")
#
#             # 1. 固定时间约束
#             if task.is_fixed and task_id in self.data.fixed_time_tasks:
#                 if time_slot != self.data.fixed_time_tasks[task_id]:
#                     violations['fixed_time'] += 1
#                     conflict_tasks_set.add(task_id)
#
#             # 2. 教师约束 - 排除合班内部冲突，并双向记录冲突任务
#             t_key = (task.teacher_id, time_slot)
#             if t_key in teacher_schedule:
#                 existing_unit, existing_task_id = teacher_schedule[t_key]
#                 if existing_unit != current_unit_id:
#                     violations['teacher'] += 1
#                     conflict_tasks_set.add(task_id)  # 记录当前任务
#                     conflict_tasks_set.add(existing_task_id)  # 记录原本占用的任务
#             else:
#                 teacher_schedule[t_key] = (current_unit_id, task_id)
#
#             # 3. 班级约束 - 双向记录冲突任务
#             c_key = (task.class_id, time_slot)
#             if c_key in class_schedule:
#                 violations['class'] += 1
#                 conflict_tasks_set.add(task_id)
#                 conflict_tasks_set.add(class_schedule[c_key])
#             class_schedule[c_key] = task_id
#
#             # 4. 教室约束 - 排除合班内部占用，并双向记录冲突任务
#             r_key = (classroom_id, time_slot)
#             if r_key in classroom_schedule:
#                 existing_unit, existing_task_id = classroom_schedule[r_key]
#                 if existing_unit != current_unit_id:
#                     violations['classroom'] += 1
#                     conflict_tasks_set.add(task_id)
#                     conflict_tasks_set.add(existing_task_id)
#             else:
#                 classroom_schedule[r_key] = (current_unit_id, task_id)
#
#             # 5. 特殊教室约束
#             if task.course_attr == '07':
#                 if classroom_key not in self.data.classrooms:
#                     violations['special_room'] += 1
#                     conflict_tasks_set.add(task_id)
#                 elif self.data.classrooms[classroom_key].attr != '07':
#                     violations['special_room'] += 1
#                     conflict_tasks_set.add(task_id)
#
#             # 6. 容量约束
#             if classroom_key in self.data.classrooms:
#                 room = self.data.classrooms[classroom_key]
#                 if room.capacity < task.student_count:
#                     violations['capacity'] += 1
#                     conflict_tasks_set.add(task_id)
#
#         # 7. 合班同步约束
#         for course_class_id, task_ids in self.data.combined_classes.items():
#             valid_ids = [t for t in task_ids if t in genes]
#             if len(valid_ids) <= 1:
#                 continue
#
#             base_slot, base_room = self._decode_gene(genes[valid_ids[0]])
#             for tid in valid_ids[1:]:
#                 slot, room = self._decode_gene(genes[tid])
#                 if slot != base_slot or room != base_room:
#                     violations['combined_class'] += 1
#                     conflict_tasks_set.add(tid)
#                     conflict_tasks_set.add(valid_ids[0])
#
#         # ⭐ 最终安全补丁：显式字段求和，防止字典新增字段导致 sum 崩溃
#         violations['total'] = (
#                 violations.get('fixed_time', 0)
#                 + violations.get('teacher', 0)
#                 + violations.get('class', 0)
#                 + violations.get('classroom', 0)
#                 + violations.get('special_room', 0)
#                 + violations.get('capacity', 0)
#                 + violations.get('combined_class', 0)
#         )
#
#         # 保存冲突任务列表供 TrueCDM 使用
#         violations['conflict_tasks'] = list(conflict_tasks_set)
#
#         return violations
#
#     def get_conflict_details(self, genes: dict) -> List[str]:
#         """
#         获取详细的冲突描述列表 (用于调试)
#         """
#         details = []
#
#         # 辅助结构: 合班组映射
#         task_group_map = {}
#         for gid, tids in self.data.combined_classes.items():
#             valid_tids = [t for t in tids if t in self.data.tasks]
#             for t in valid_tids:
#                 task_group_map[t] = gid
#
#         # 冲突检测表
#         teacher_schedule = {}
#         class_schedule = {}
#         classroom_schedule = {}
#
#         for task_id, task in self.data.tasks.items():
#             if task_id not in genes:
#                 continue
#
#             time_slot, classroom_id = self._decode_gene(genes[task_id])
#             classroom_key = self._get_classroom_key(classroom_id)
#             current_unit_id = task_group_map.get(task_id, f"SINGLE_{task_id}")
#
#             # 1. 固定时间
#             if task.is_fixed and task_id in self.data.fixed_time_tasks:
#                 target = self.data.fixed_time_tasks[task_id]
#                 if time_slot != target:
#                     details.append(f"[固定时间] 任务{task_id}({task.course_class_name}) 应在槽{target} 实际在{time_slot}")
#
#             # 2. 教师冲突
#             t_key = (task.teacher_id, time_slot)
#             if t_key in teacher_schedule:
#                 existing_unit = teacher_schedule[t_key]
#                 if existing_unit != current_unit_id:
#                     details.append(f"[教师冲突] 老师{task.teacher_id} 在槽{time_slot} 撞课: {existing_unit} vs {current_unit_id}")
#             else:
#                 teacher_schedule[t_key] = current_unit_id
#
#             # 3. 班级冲突
#             c_key = (task.class_id, time_slot)
#             if c_key in class_schedule:
#                 details.append(f"[班级冲突] 班级{task.class_id} 在槽{time_slot} 有多门课")
#             class_schedule[c_key] = task_id
#
#             # 4. 教室冲突
#             r_key = (classroom_id, time_slot)
#             if r_key in classroom_schedule:
#                 existing_unit = classroom_schedule[r_key]
#                 if existing_unit != current_unit_id:
#                     details.append(f"[教室冲突] 教室{classroom_id} 在槽{time_slot} 重叠: {existing_unit} vs {current_unit_id}")
#             else:
#                 classroom_schedule[r_key] = current_unit_id
#
#             # 5. 特殊教室
#             if task.course_attr == '07':
#                 if classroom_key not in self.data.classrooms or self.data.classrooms[classroom_key].attr != '07':
#                     details.append(f"[特殊教室] 任务{task_id} 需要实验室，但分配了 {classroom_key}")
#
#             # 6. 容量
#             if classroom_key in self.data.classrooms:
#                 cap = self.data.classrooms[classroom_key].capacity
#                 if cap < task.student_count:
#                     details.append(f"[容量不足] 任务{task_id} 人数{task.student_count} > 教室{classroom_key}容量{cap}")
#
#         return details
#
#     def _decode_gene(self, gene: int) -> Tuple[int, int]:
#         time_slot_id = gene >> 16
#         classroom_id = gene & 0xFFFF
#         return time_slot_id, classroom_id
#
#     def _get_classroom_key(self, classroom_id: int) -> str:
#         str_id = str(classroom_id)
#         if str_id in self.data.classrooms:
#             return str_id
#         padded_id = str_id.zfill(6)
#         if padded_id in self.data.classrooms:
#             return padded_id
#         return str_id
#
#     def is_valid(self, genes: dict) -> bool:
#         return self.check_all(genes)['total'] == 0
#
#
#
#
#
#
#
#
#
# # """
# # 硬约束检查器 - 检查排课方案是否满足所有硬约束 (Fixed Version)
# # 包含详细冲突诊断功能
# # """
# # from typing import Dict, List, Tuple
# # from collections import defaultdict
# #
# # from data.data_loader import ScheduleData
# #
# #
# # class ConstraintChecker:
# #     """硬约束检查器"""
# #
# #     # 对应论文中的7个硬约束
# #     def __init__(self, data: ScheduleData):
# #         self.data = data
# #
# #     def check_all(self, genes: dict) -> Dict[str, int]:
# #         """
# #         检查所有硬约束，返回违约次数
# #         """
# #         violations = {
# #             'fixed_time': 0,  # C7 (Part 1)
# #             'teacher': 0,  # C2
# #             'class': 0,  # C3
# #             'classroom': 0,  # C4
# #             'special_room': 0,  # C7 (Part 2)
# #             'capacity': 0,  # C5
# #             'combined_class': 0,  # C6
# #             'total': 0
# #         }
# #
# #         # 辅助结构: 合班组映射 {task_id -> group_id}
# #         task_group_map = {}
# #         for gid, tids in self.data.combined_classes.items():
# #             valid_tids = [t for t in tids if t in self.data.tasks]
# #             for t in valid_tids:
# #                 task_group_map[t] = gid
# #
# #         # 冲突检测表
# #         teacher_schedule = {}  # {(teacher_id, time_slot): unit_id}
# #         class_schedule = {}  # {(class_id, time_slot): task_id}
# #         classroom_schedule = {}  # {(classroom_id, time_slot): unit_id}
# #
# #         for task_id, task in self.data.tasks.items():
# #             if task_id not in genes:
# #                 continue
# #
# #             time_slot, classroom_id = self._decode_gene(genes[task_id])
# #             classroom_key = self._get_classroom_key(classroom_id)
# #
# #             # 当前任务的逻辑单元ID (如果是合班，则为合班ID，否则为任务ID)
# #             current_unit_id = task_group_map.get(task_id, f"SINGLE_{task_id}")
# #
# #             # 1. 固定时间约束
# #             if task.is_fixed and task_id in self.data.fixed_time_tasks:
# #                 if time_slot != self.data.fixed_time_tasks[task_id]:
# #                     violations['fixed_time'] += 1
# #
# #             # 2. 教师约束 - 排除合班内部冲突
# #             t_key = (task.teacher_id, time_slot)
# #             if t_key in teacher_schedule:
# #                 existing_unit = teacher_schedule[t_key]
# #                 if existing_unit != current_unit_id:
# #                     violations['teacher'] += 1
# #             else:
# #                 teacher_schedule[t_key] = current_unit_id
# #
# #             # 3. 班级约束
# #             c_key = (task.class_id, time_slot)
# #             if c_key in class_schedule:
# #                 violations['class'] += 1
# #             class_schedule[c_key] = task_id
# #
# #             # 4. 教室约束 - 排除合班内部占用
# #             r_key = (classroom_id, time_slot)
# #             if r_key in classroom_schedule:
# #                 existing_unit = classroom_schedule[r_key]
# #                 if existing_unit != current_unit_id:
# #                     violations['classroom'] += 1
# #             else:
# #                 classroom_schedule[r_key] = current_unit_id
# #
# #             # 5. 特殊教室约束
# #             if task.course_attr == '07':
# #                 if classroom_key not in self.data.classrooms:
# #                     violations['special_room'] += 1
# #                 elif self.data.classrooms[classroom_key].attr != '07':
# #                     violations['special_room'] += 1
# #
# #             # 6. 容量约束
# #             if classroom_key in self.data.classrooms:
# #                 room = self.data.classrooms[classroom_key]
# #                 if room.capacity < task.student_count:
# #                     violations['capacity'] += 1
# #
# #         # 7. 合班同步约束
# #         for course_class_id, task_ids in self.data.combined_classes.items():
# #             valid_ids = [t for t in task_ids if t in genes]
# #             if len(valid_ids) <= 1:
# #                 continue
# #
# #             base_slot, base_room = self._decode_gene(genes[valid_ids[0]])
# #             for tid in valid_ids[1:]:
# #                 slot, room = self._decode_gene(genes[tid])
# #                 if slot != base_slot or room != base_room:
# #                     violations['combined_class'] += 1
# #
# #         violations['total'] = sum(v for k, v in violations.items() if k != 'total')
# #         return violations
# #
# #     def get_conflict_details(self, genes: dict) -> List[str]:
# #         """
# #         获取详细的冲突描述列表 (用于调试)
# #         """
# #         details = []
# #
# #         # 辅助结构: 合班组映射
# #         task_group_map = {}
# #         for gid, tids in self.data.combined_classes.items():
# #             valid_tids = [t for t in tids if t in self.data.tasks]
# #             for t in valid_tids:
# #                 task_group_map[t] = gid
# #
# #         # 冲突检测表
# #         teacher_schedule = {}
# #         class_schedule = {}
# #         classroom_schedule = {}
# #
# #         for task_id, task in self.data.tasks.items():
# #             if task_id not in genes:
# #                 continue
# #
# #             time_slot, classroom_id = self._decode_gene(genes[task_id])
# #             classroom_key = self._get_classroom_key(classroom_id)
# #             current_unit_id = task_group_map.get(task_id, f"SINGLE_{task_id}")
# #
# #             # 1. 固定时间
# #             if task.is_fixed and task_id in self.data.fixed_time_tasks:
# #                 target = self.data.fixed_time_tasks[task_id]
# #                 if time_slot != target:
# #                     details.append(f"[固定时间] 任务{task_id}({task.course_class_name}) 应在槽{target} 实际在{time_slot}")
# #
# #             # 2. 教师冲突
# #             t_key = (task.teacher_id, time_slot)
# #             if t_key in teacher_schedule:
# #                 existing_unit = teacher_schedule[t_key]
# #                 if existing_unit != current_unit_id:
# #                     details.append(f"[教师冲突] 老师{task.teacher_id} 在槽{time_slot} 撞课: {existing_unit} vs {current_unit_id}")
# #             else:
# #                 teacher_schedule[t_key] = current_unit_id
# #
# #             # 3. 班级冲突
# #             c_key = (task.class_id, time_slot)
# #             if c_key in class_schedule:
# #                 details.append(f"[班级冲突] 班级{task.class_id} 在槽{time_slot} 有多门课")
# #             class_schedule[c_key] = task_id
# #
# #             # 4. 教室冲突
# #             r_key = (classroom_id, time_slot)
# #             if r_key in classroom_schedule:
# #                 existing_unit = classroom_schedule[r_key]
# #                 if existing_unit != current_unit_id:
# #                     details.append(f"[教室冲突] 教室{classroom_id} 在槽{time_slot} 重叠: {existing_unit} vs {current_unit_id}")
# #             else:
# #                 classroom_schedule[r_key] = current_unit_id
# #
# #             # 5. 特殊教室
# #             if task.course_attr == '07':
# #                 if classroom_key not in self.data.classrooms or self.data.classrooms[classroom_key].attr != '07':
# #                     details.append(f"[特殊教室] 任务{task_id} 需要实验室，但分配了 {classroom_key}")
# #
# #             # 6. 容量
# #             if classroom_key in self.data.classrooms:
# #                 cap = self.data.classrooms[classroom_key].capacity
# #                 if cap < task.student_count:
# #                     details.append(f"[容量不足] 任务{task_id} 人数{task.student_count} > 教室{classroom_key}容量{cap}")
# #
# #         return details
# #
# #     def _decode_gene(self, gene: int) -> Tuple[int, int]:
# #         time_slot_id = gene >> 16
# #         classroom_id = gene & 0xFFFF
# #         return time_slot_id, classroom_id
# #
# #     def _get_classroom_key(self, classroom_id: int) -> str:
# #         str_id = str(classroom_id)
# #         if str_id in self.data.classrooms:
# #             return str_id
# #         padded_id = str_id.zfill(6)
# #         if padded_id in self.data.classrooms:
# #             return padded_id
# #         return str_id
# #
# #     def is_valid(self, genes: dict) -> bool:
# #         return self.check_all(genes)['total'] == 0
# #
