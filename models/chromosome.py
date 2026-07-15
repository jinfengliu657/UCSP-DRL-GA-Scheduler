"""
染色体模型定义 (Fixed: Added initialize method)
文件名: models/chromosome.py
"""
import random
import copy


class Chromosome:
    def __init__(self, data):
        """
        初始化染色体
        :param data: ScheduleData 对象，包含所有排课资源
        """
        self.data = data
        self.genes = {}  # 基因存储 {task_id: encoded_value}
        self.fitness = None

    def initialize(self):
        """
        [修复] 随机初始化染色体
        为每个任务随机分配一个 [0-24] 的时间槽和一个随机教室。
        """
        self.genes = {}

        # 1. 准备可用教室列表 (转换为 int ID)
        valid_room_ids = []
        for r_key, room in self.data.classrooms.items():
            try:
                # 尝试直接转 int
                r_int = int(room.classroom_id)
            except ValueError:
                # 如果是字符串（如 "A101"），则 Hash 后取低16位
                r_int = hash(r_key) & 0xFFFF
            valid_room_ids.append(r_int)

        if not valid_room_ids:
            # 防御性编程：如果没有教室数据，造一个假的 0 号教室防止报错
            valid_room_ids = [0]

        # 2. 为每个任务生成基因
        for task_id in self.data.tasks:
            # 随机时间槽: 0-24 (5天 * 5节)
            time_slot = random.randint(0, 24)

            # 随机教室
            room_id = random.choice(valid_room_ids)

            # 编码: 高16位时间，低16位教室
            gene = (time_slot << 16) | room_id
            self.genes[task_id] = gene

    def clone(self):
        """
        克隆个体 (深拷贝基因，避免引用问题)
        """
        new_instance = Chromosome(self.data)
        new_instance.genes = self.genes.copy()
        new_instance.fitness = self.fitness
        return new_instance


# """
# 染色体编码 - 排课方案的基因表示
# """
# import numpy as np
# import random
# from typing import Dict, List, Tuple, Optional
# from copy import deepcopy
#
# from data.data_loader import ScheduleData
#
#
# class Chromosome:
#     """
#     染色体类 - 表示一个排课方案
#
#     编码方案:
#     - 每个课程任务对应一个基因
#     - 基因值 = (time_slot_id << 16) | classroom_id
#     - time_slot_id: 0-24 (5天 × 5时段)
#     - classroom_id: 教室ID
#     """
#
#     def __init__(self, num_tasks: int, data: ScheduleData = None):
#         """
#         初始化染色体
#
#         Args:
#             num_tasks: 任务数量
#             data: 排课数据
#         """
#         self.num_tasks = num_tasks
#         self.data = data
#         # 使用字典存储基因，以支持不连续的任务ID
#         self.genes = {}  # {task_id: gene_value}
#         self.fitness = float('-inf')
#         self.constraint_violations = 0
#         self.age = 0
#
#     def encode_gene(self, time_slot_id: int, classroom_id) -> int:
#         """
#         编码单个基因
#
#         Args:
#             time_slot_id: 时间槽ID (0-24)
#             classroom_id: 教室ID (可以是int或str)
#
#         Returns:
#             编码后的基因值
#         """
#         # 确保classroom_id是整数
#         if isinstance(classroom_id, str):
#             # 尝试将字符串ID转换为整数，如果是纯数字字符串
#             if classroom_id.isdigit():
#                 classroom_id = int(classroom_id)
#             else:
#                 # 如果不是纯数字，使用哈希值
#                 classroom_id = hash(classroom_id) & 0xFFFF
#         return (time_slot_id << 16) | int(classroom_id)
#
#     def decode_gene(self, gene: int) -> Tuple[int, int]:
#         """
#         解码基因
#
#         Args:
#             gene: 编码后的基因值
#
#         Returns:
#             (time_slot_id, classroom_id)
#         """
#         time_slot_id = gene >> 16
#         classroom_id = gene & 0xFFFF
#         return time_slot_id, classroom_id
#
#     def get_gene(self, task_id: int) -> Tuple[int, int]:
#         """获取指定任务的时间槽和教室"""
#         return self.decode_gene(self.genes[task_id])
#
#     def set_gene(self, task_id: int, time_slot_id: int, classroom_id: int):
#         """设置指定任务的时间槽和教室"""
#         self.genes[task_id] = self.encode_gene(time_slot_id, classroom_id)
#
#     def clone(self) -> 'Chromosome':
#         """创建染色体的深拷贝"""
#         new_chromo = Chromosome(self.num_tasks, self.data)
#         new_chromo.genes = self.genes.copy()
#         new_chromo.fitness = self.fitness
#         new_chromo.constraint_violations = self.constraint_violations
#         return new_chromo
#
#     def random_init(self, available_slots: List[int] = None,
#                     available_rooms: List[str] = None):
#         """
#         随机初始化染色体
#
#         Args:
#             available_slots: 可用时间槽列表
#             available_rooms: 可用教室列表
#         """
#         if available_slots is None:
#             available_slots = list(range(25))  # 5天 × 5时段
#
#         if available_rooms is None and self.data:
#             available_rooms = list(self.data.classrooms.keys())
#
#         # 使用实际的任务ID列表
#         task_ids = list(self.data.tasks.keys()) if self.data else list(range(self.num_tasks))
#
#         for task_id in task_ids:
#             # 跳过固定时间课程
#             if self.data and task_id in self.data.fixed_time_tasks:
#                 fixed_slot = self.data.fixed_time_tasks[task_id]
#                 room = random.choice(available_rooms) if available_rooms else 1
#                 self.set_gene(task_id, fixed_slot, room)
#             else:
#                 slot = random.choice(available_slots)
#                 room = random.choice(available_rooms) if available_rooms else 1
#                 self.set_gene(task_id, slot, room)
#
#     def heuristic_init(self, available_slots: List[int] = None,
#                        available_rooms: List[str] = None):
#         """
#         启发式初始化 - 考虑约束的贪心初始化
#
#         优先级:
#         1. 固定时间任务
#         2. 需要实验室的任务（资源最紧张）
#         3. 合班任务
#         4. 其他任务
#
#         Args:
#             available_slots: 可用时间槽列表
#             available_rooms: 可用教室列表
#         """
#         if not self.data:
#             self.random_init(available_slots, available_rooms)
#             return
#
#         if available_slots is None:
#             available_slots = list(range(25))
#
#         if available_rooms is None:
#             available_rooms = list(self.data.classrooms.keys())
#
#         # 跟踪已分配的资源
#         used_teacher_slots = set()
#         used_class_slots = set()
#         used_room_slots = set()  # {(room, slot)}
#
#         processed_tasks = set()
#
#         # 分类教室
#         lab_rooms = [r for r in available_rooms if r in self.data.classrooms and self.data.classrooms[r].attr == '07']
#         normal_rooms = [r for r in available_rooms if r in self.data.classrooms and self.data.classrooms[r].attr != '07']
#
#         # 第1步: 处理固定时间任务
#         for task_id in self.data.fixed_time_tasks:
#             if task_id not in self.data.tasks:
#                 continue
#             task = self.data.tasks[task_id]
#             slot = self.data.fixed_time_tasks[task_id]
#
#             # 选择合适教室
#             if task.course_attr == '07':
#                 room = self._select_room_for_task(task, slot, lab_rooms, used_room_slots)
#             else:
#                 room = self._select_room_for_task(task, slot, normal_rooms, used_room_slots)
#
#             self.set_gene(task_id, slot, room)
#             used_teacher_slots.add((task.teacher_id, slot))
#             used_class_slots.add((task.class_id, slot))
#             used_room_slots.add((room, slot))
#             processed_tasks.add(task_id)
#
#         # 第2步: 处理需要实验室的任务（优先级最高）
#         lab_tasks = [tid for tid, task in self.data.tasks.items()
#                     if task.course_attr == '07' and tid not in processed_tasks]
#
#         # 按人数降序排序（大班优先）
#         lab_tasks.sort(key=lambda tid: -self.data.tasks[tid].student_count)
#
#         for task_id in lab_tasks:
#             task = self.data.tasks[task_id]
#
#             # 找可用时间槽
#             slot = self._find_slot_for_task(task, available_slots, used_teacher_slots, used_class_slots)
#
#             # 找可用实验室
#             room = self._select_room_for_task(task, slot, lab_rooms, used_room_slots)
#
#             self.set_gene(task_id, slot, room)
#             used_teacher_slots.add((task.teacher_id, slot))
#             used_class_slots.add((task.class_id, slot))
#             used_room_slots.add((room, slot))
#             processed_tasks.add(task_id)
#
#         # 第3步: 处理合班课程
#         for course_class_id, task_ids in self.data.combined_classes.items():
#             # 跳过已处理的
#             remaining_ids = [tid for tid in task_ids if tid not in processed_tasks]
#             if len(remaining_ids) == 0:
#                 continue
#             if len(remaining_ids) == 1:
#                 # 单任务不需要合班处理，留给第4步
#                 continue
#
#             # 计算合班总人数
#             total_students = sum(self.data.tasks[tid].student_count for tid in remaining_ids)
#             first_task = self.data.tasks[remaining_ids[0]]
#
#             # 找所有合班任务的教师和班级都空闲的时间槽
#             slot = self._find_slot_for_combined(remaining_ids, available_slots, used_teacher_slots, used_class_slots)
#
#             # 找合适教室
#             room = self._select_room_for_combined(total_students, first_task.course_attr, slot, normal_rooms, used_room_slots)
#
#             # 设置所有合班任务
#             for tid in remaining_ids:
#                 self.set_gene(tid, slot, room)
#                 task = self.data.tasks[tid]
#                 used_teacher_slots.add((task.teacher_id, slot))
#                 used_class_slots.add((task.class_id, slot))
#                 processed_tasks.add(tid)
#
#             used_room_slots.add((room, slot))
#
#         # 第4步: 处理剩余普通任务
#         remaining_tasks = [tid for tid in self.data.tasks.keys() if tid not in processed_tasks]
#         remaining_tasks.sort(key=lambda tid: -self.data.tasks[tid].student_count)
#
#         for task_id in remaining_tasks:
#             task = self.data.tasks[task_id]
#
#             # 找可用时间槽
#             slot = self._find_slot_for_task(task, available_slots, used_teacher_slots, used_class_slots)
#
#             # 找可用教室
#             room = self._select_room_for_task(task, slot, normal_rooms, used_room_slots)
#
#             self.set_gene(task_id, slot, room)
#             used_teacher_slots.add((task.teacher_id, slot))
#             used_class_slots.add((task.class_id, slot))
#             used_room_slots.add((room, slot))
#
#     def _find_slot_for_task(self, task, available_slots: List[int],
#                            used_teacher_slots: set, used_class_slots: set) -> int:
#         """为单个任务找可用时间槽"""
#         slots_copy = available_slots.copy()
#         random.shuffle(slots_copy)
#
#         for slot in slots_copy:
#             if (task.teacher_id, slot) not in used_teacher_slots and \
#                (task.class_id, slot) not in used_class_slots:
#                 return slot
#
#         # 如果没有完美槽，返回任意槽
#         return random.choice(available_slots)
#
#     def _find_slot_for_combined(self, task_ids: List[int], available_slots: List[int],
#                                used_teacher_slots: set, used_class_slots: set) -> int:
#         """为合班任务找可用时间槽"""
#         slots_copy = available_slots.copy()
#         random.shuffle(slots_copy)
#
#         combined_teachers = set(self.data.tasks[tid].teacher_id for tid in task_ids)
#         combined_classes = set(self.data.tasks[tid].class_id for tid in task_ids)
#
#         for slot in slots_copy:
#             valid = True
#             for teacher_id in combined_teachers:
#                 if (teacher_id, slot) in used_teacher_slots:
#                     valid = False
#                     break
#             if valid:
#                 for class_id in combined_classes:
#                     if (class_id, slot) in used_class_slots:
#                         valid = False
#                         break
#             if valid:
#                 return slot
#
#         return random.choice(available_slots)
#
#     def _select_room_for_task(self, task, slot: int, room_pool: List[str],
#                              used_room_slots: set) -> str:
#         """为任务选择教室"""
#         suitable_rooms = []
#
#         for room_id in room_pool:
#             if (room_id, slot) in used_room_slots:
#                 continue
#             if room_id not in self.data.classrooms:
#                 continue
#
#             room = self.data.classrooms[room_id]
#             if room.capacity >= task.student_count:
#                 suitable_rooms.append((room_id, room.capacity))
#
#         if suitable_rooms:
#             suitable_rooms.sort(key=lambda x: x[1])
#             return suitable_rooms[0][0]
#
#         # 如果没有合适的，返回任意未占用的
#         for room_id in room_pool:
#             if (room_id, slot) not in used_room_slots:
#                 return room_id
#
#         return random.choice(room_pool)
#
#     def _select_room_for_combined(self, total_students: int, course_attr: str,
#                                  slot: int, room_pool: List[str],
#                                  used_room_slots: set) -> str:
#         """为合班任务选择教室"""
#         suitable_rooms = []
#
#         for room_id in room_pool:
#             if (room_id, slot) in used_room_slots:
#                 continue
#             if room_id not in self.data.classrooms:
#                 continue
#
#             room = self.data.classrooms[room_id]
#             if course_attr == '07' and room.attr != '07':
#                 continue
#             if room.capacity >= total_students:
#                 suitable_rooms.append((room_id, room.capacity))
#
#         if suitable_rooms:
#             suitable_rooms.sort(key=lambda x: x[1])
#             return suitable_rooms[0][0]
#
#         # 如果没有合适的，返回任意未占用的
#         for room_id in room_pool:
#             if (room_id, slot) not in used_room_slots:
#                 return room_id
#
#         return random.choice(room_pool)
#
#     def repair(self, max_iterations: int = 200):
#         """
#         修复硬约束违约 - 增强版
#
#         Args:
#             max_iterations: 最大修复迭代次数
#         """
#         if not self.data:
#             return
#
#         # 构建合班任务映射
#         task_to_combined = {}  # task_id -> course_class_id
#         for course_class_id, task_ids in self.data.combined_classes.items():
#             if len(task_ids) > 1:
#                 for tid in task_ids:
#                     task_to_combined[tid] = course_class_id
#
#         # 多轮修复，每轮处理不同类型的冲突
#         for iteration in range(max_iterations):
#             # 收集所有冲突
#             conflicts = self._collect_all_conflicts(task_to_combined)
#
#             if not conflicts:
#                 break  # 无冲突，修复完成
#
#             # 按冲突严重程度排序（涉及更多任务的冲突优先处理）
#             conflicts.sort(key=lambda x: -x[4] if len(x) > 4 else 0)
#
#             # 选择最严重的冲突修复
#             conflict = conflicts[0]
#             conflict_type, task_id = conflict[0], conflict[1]
#             task = self.data.tasks[task_id]
#
#             # 跳过固定时间任务
#             if task_id in self.data.fixed_time_tasks:
#                 continue
#
#             # 检查是否是合班任务
#             if task_id in task_to_combined:
#                 course_class_id = task_to_combined[task_id]
#                 combined_task_ids = [tid for tid in self.data.combined_classes[course_class_id]
#                                     if tid in self.data.tasks]
#                 success = self._move_combined_tasks_smart(combined_task_ids, task_to_combined)
#             else:
#                 success = self._move_single_task_smart(task_id)
#
#             if not success and iteration > max_iterations // 2:
#                 # 后半段如果智能移动失败，尝试随机移动
#                 if task_id in task_to_combined:
#                     self._move_combined_tasks(self.data.combined_classes[task_to_combined[task_id]], task_to_combined)
#                 else:
#                     self._move_single_task(task_id)
#
#     def _collect_all_conflicts(self, task_to_combined: dict) -> list:
#         """收集所有冲突"""
#         teacher_slot_tasks = {}
#         class_slot_tasks = {}
#         room_slot_tasks = {}
#
#         for task_id, task in self.data.tasks.items():
#             if task_id not in self.genes:
#                 continue
#             slot, room = self.get_gene(task_id)
#
#             key = (task.teacher_id, slot)
#             if key not in teacher_slot_tasks:
#                 teacher_slot_tasks[key] = []
#             teacher_slot_tasks[key].append(task_id)
#
#             key = (task.class_id, slot)
#             if key not in class_slot_tasks:
#                 class_slot_tasks[key] = []
#             class_slot_tasks[key].append(task_id)
#
#             key = (room, slot)
#             if key not in room_slot_tasks:
#                 room_slot_tasks[key] = []
#             room_slot_tasks[key].append(task_id)
#
#         conflicts = []
#
#         # 教师冲突
#         for (teacher_id, slot), task_ids in teacher_slot_tasks.items():
#             if len(task_ids) > 1:
#                 for tid in task_ids[1:]:
#                     if tid not in self.data.fixed_time_tasks:
#                         conflicts.append(('teacher', tid, teacher_id, slot, len(task_ids)))
#
#         # 班级冲突
#         for (class_id, slot), task_ids in class_slot_tasks.items():
#             if len(task_ids) > 1:
#                 for tid in task_ids[1:]:
#                     if tid not in self.data.fixed_time_tasks:
#                         conflicts.append(('class', tid, class_id, slot, len(task_ids)))
#
#         # 教室冲突
#         for (room, slot), task_ids in room_slot_tasks.items():
#             if len(task_ids) > 1:
#                 for tid in task_ids[1:]:
#                     if tid not in self.data.fixed_time_tasks:
#                         conflicts.append(('room', tid, room, slot, len(task_ids)))
#
#         return conflicts
#
#     def _move_single_task_smart(self, task_id: int) -> bool:
#         """智能移动单个任务 - 找冲突最少的位置"""
#         task = self.data.tasks[task_id]
#
#         # 收集当前占用情况
#         occupied = self._get_occupied_resources(task_id)
#
#         best_slot = None
#         best_room = None
#         min_conflicts = float('inf')
#
#         for slot in range(25):
#             # 计算该时间槽的冲突数
#             conflicts = 0
#             if slot in occupied['teacher_slots']:
#                 conflicts += 1
#             if slot in occupied['class_slots']:
#                 conflicts += 1
#
#             # 找该时间槽可用的教室
#             for room_id in self.data.classrooms.keys():
#                 room = self.data.classrooms[room_id]
#                 room_conflicts = conflicts
#
#                 if (slot, room_id) in occupied['room_slots']:
#                     room_conflicts += 1
#
#                 # 检查容量和类型
#                 if task.course_attr == '07' and room.attr != '07':
#                     continue
#                 if room.capacity < task.student_count:
#                     room_conflicts += 0.5  # 容量不足轻微惩罚
#
#                 if room_conflicts < min_conflicts:
#                     min_conflicts = room_conflicts
#                     best_slot = slot
#                     best_room = room_id
#
#         if best_slot is not None and min_conflicts < 3:
#             self.set_gene(task_id, best_slot, best_room)
#             return True
#         return False
#
#     def _move_combined_tasks_smart(self, task_ids: list, task_to_combined: dict) -> bool:
#         """智能移动合班任务组"""
#         if not task_ids:
#             return False
#
#         total_students = sum(self.data.tasks[tid].student_count for tid in task_ids if tid in self.data.tasks)
#         first_task = self.data.tasks[task_ids[0]]
#
#         combined_teachers = set(self.data.tasks[tid].teacher_id for tid in task_ids if tid in self.data.tasks)
#         combined_classes = set(self.data.tasks[tid].class_id for tid in task_ids if tid in self.data.tasks)
#
#         # 收集其他任务的占用
#         occupied = self._get_occupied_resources_for_combined(task_ids, combined_teachers, combined_classes)
#
#         best_slot = None
#         best_room = None
#         min_conflicts = float('inf')
#
#         for slot in range(25):
#             conflicts = 0
#             for t in combined_teachers:
#                 if slot in occupied['teacher_slots'].get(t, set()):
#                     conflicts += 1
#             for c in combined_classes:
#                 if slot in occupied['class_slots'].get(c, set()):
#                     conflicts += 1
#
#             # 找合适教室
#             for room_id in self.data.classrooms.keys():
#                 room = self.data.classrooms[room_id]
#                 room_conflicts = conflicts
#
#                 if (slot, room_id) in occupied['room_slots']:
#                     room_conflicts += 1
#                 if first_task.course_attr == '07' and room.attr != '07':
#                     continue
#                 if room.capacity < total_students:
#                     continue
#
#                 if room_conflicts < min_conflicts:
#                     min_conflicts = room_conflicts
#                     best_slot = slot
#                     best_room = room_id
#
#         if best_slot is not None and min_conflicts < 3:
#             for tid in task_ids:
#                 if tid not in self.data.fixed_time_tasks:
#                     self.set_gene(tid, best_slot, best_room)
#             return True
#         return False
#
#     def _get_occupied_resources(self, exclude_task_id: int) -> dict:
#         """获取除指定任务外的资源占用情况"""
#         task = self.data.tasks[exclude_task_id]
#         occupied = {'teacher_slots': set(), 'class_slots': set(), 'room_slots': set()}
#
#         for tid, t in self.data.tasks.items():
#             if tid == exclude_task_id or tid not in self.genes:
#                 continue
#             s, r = self.get_gene(tid)
#             if t.teacher_id == task.teacher_id:
#                 occupied['teacher_slots'].add(s)
#             if t.class_id == task.class_id:
#                 occupied['class_slots'].add(s)
#             occupied['room_slots'].add((s, r))
#
#         return occupied
#
#     def _get_occupied_resources_for_combined(self, exclude_ids: list, teachers: set, classes: set) -> dict:
#         """获取合班任务组外的资源占用"""
#         occupied = {
#             'teacher_slots': {t: set() for t in teachers},
#             'class_slots': {c: set() for c in classes},
#             'room_slots': set()
#         }
#
#         for tid, t in self.data.tasks.items():
#             if tid in exclude_ids or tid not in self.genes:
#                 continue
#             s, r = self.get_gene(tid)
#             if t.teacher_id in teachers:
#                 occupied['teacher_slots'][t.teacher_id].add(s)
#             if t.class_id in classes:
#                 occupied['class_slots'][t.class_id].add(s)
#             occupied['room_slots'].add((s, r))
#
#         return occupied
#
#     def _move_single_task(self, task_id: int):
#         """移动单个任务到无冲突位置"""
#         task = self.data.tasks[task_id]
#
#         # 收集当前占用情况
#         occupied_teacher_slots = set()
#         occupied_class_slots = set()
#         occupied_room_slots = {}  # {slot: set(rooms)}
#
#         for tid, t in self.data.tasks.items():
#             if tid == task_id:
#                 continue
#             s, r = self.get_gene(tid)
#             if t.teacher_id == task.teacher_id:
#                 occupied_teacher_slots.add(s)
#             if t.class_id == task.class_id:
#                 occupied_class_slots.add(s)
#             if s not in occupied_room_slots:
#                 occupied_room_slots[s] = set()
#             occupied_room_slots[s].add(r)
#
#         # 找新的时间槽
#         available_slots = [s for s in range(25)
#                           if s not in occupied_teacher_slots
#                           and s not in occupied_class_slots]
#
#         if available_slots:
#             new_slot = random.choice(available_slots)
#             # 找该时间槽可用的教室
#             occupied_rooms = occupied_room_slots.get(new_slot, set())
#             available_rooms = [r for r in self.data.classrooms.keys()
#                               if r not in occupied_rooms]
#
#             if available_rooms:
#                 # 找合适容量的教室
#                 suitable_rooms = []
#                 for room_id in available_rooms:
#                     room = self.data.classrooms[room_id]
#                     if task.course_attr == '07' and room.attr != '07':
#                         continue
#                     if room.capacity >= task.student_count:
#                         suitable_rooms.append((room_id, room.capacity))
#
#                 if suitable_rooms:
#                     suitable_rooms.sort(key=lambda x: x[1])
#                     new_room = suitable_rooms[0][0]
#                 else:
#                     new_room = random.choice(available_rooms)
#
#                 self.set_gene(task_id, new_slot, new_room)
#
#     def _move_combined_tasks(self, task_ids: List[int], task_to_combined: dict):
#         """移动整个合班组到无冲突位置"""
#         # 计算合班总人数
#         total_students = sum(self.data.tasks[tid].student_count for tid in task_ids)
#         first_task = self.data.tasks[task_ids[0]]
#
#         # 收集所有合班任务的教师和班级
#         combined_teachers = set(self.data.tasks[tid].teacher_id for tid in task_ids)
#         combined_classes = set(self.data.tasks[tid].class_id for tid in task_ids)
#
#         # 收集其他任务的占用情况
#         occupied_teacher_slots = {t: set() for t in combined_teachers}
#         occupied_class_slots = {c: set() for c in combined_classes}
#         occupied_room_slots = {}
#
#         for tid, t in self.data.tasks.items():
#             if tid in task_ids:
#                 continue
#             s, r = self.get_gene(tid)
#
#             if t.teacher_id in combined_teachers:
#                 occupied_teacher_slots[t.teacher_id].add(s)
#             if t.class_id in combined_classes:
#                 occupied_class_slots[t.class_id].add(s)
#
#             if s not in occupied_room_slots:
#                 occupied_room_slots[s] = set()
#             occupied_room_slots[s].add(r)
#
#         # 找所有教师和班级都空闲的时间槽
#         available_slots = []
#         for s in range(25):
#             valid = True
#             for teacher_id in combined_teachers:
#                 if s in occupied_teacher_slots.get(teacher_id, set()):
#                     valid = False
#                     break
#             if valid:
#                 for class_id in combined_classes:
#                     if s in occupied_class_slots.get(class_id, set()):
#                         valid = False
#                         break
#             if valid:
#                 available_slots.append(s)
#
#         if available_slots:
#             new_slot = random.choice(available_slots)
#
#             # 找该时间槽可用且容量足够的教室
#             occupied_rooms = occupied_room_slots.get(new_slot, set())
#             suitable_rooms = []
#
#             for room_id in self.data.classrooms.keys():
#                 if room_id in occupied_rooms:
#                     continue
#                 room = self.data.classrooms[room_id]
#                 if first_task.course_attr == '07' and room.attr != '07':
#                     continue
#                 if room.capacity >= total_students:
#                     suitable_rooms.append((room_id, room.capacity))
#
#             if suitable_rooms:
#                 suitable_rooms.sort(key=lambda x: x[1])
#                 new_room = suitable_rooms[0][0]
#
#                 # 移动所有合班任务
#                 for tid in task_ids:
#                     if tid not in self.data.fixed_time_tasks:
#                         self.set_gene(tid, new_slot, new_room)
#
#     def _count_violations(self) -> int:
#         """计算当前违约数"""
#         teacher_slot_tasks = {}
#         class_slot_tasks = {}
#         room_slot_tasks = {}
#
#         for task_id, task in self.data.tasks.items():
#             slot, room = self.get_gene(task_id)
#
#             key = (task.teacher_id, slot)
#             if key not in teacher_slot_tasks:
#                 teacher_slot_tasks[key] = []
#             teacher_slot_tasks[key].append(task_id)
#
#             key = (task.class_id, slot)
#             if key not in class_slot_tasks:
#                 class_slot_tasks[key] = []
#             class_slot_tasks[key].append(task_id)
#
#             key = (room, slot)
#             if key not in room_slot_tasks:
#                 room_slot_tasks[key] = []
#             room_slot_tasks[key].append(task_id)
#
#         count = 0
#         for tasks in teacher_slot_tasks.values():
#             if len(tasks) > 1:
#                 count += len(tasks) - 1
#         for tasks in class_slot_tasks.values():
#             if len(tasks) > 1:
#                 count += len(tasks) - 1
#         for tasks in room_slot_tasks.values():
#             if len(tasks) > 1:
#                 count += len(tasks) - 1
#
#         return count
#
#     def __hash__(self):
#         """计算染色体哈希值（用于禁忌搜索）"""
#         return hash(self.genes.tobytes())
#
#     def __eq__(self, other):
#         """染色体相等比较"""
#         if not isinstance(other, Chromosome):
#             return False
#         return np.array_equal(self.genes, other.genes)
#
#     def __repr__(self):
#         return f"Chromosome(fitness={self.fitness:.2f}, violations={self.constraint_violations})"
