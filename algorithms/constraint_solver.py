"""
约束满足求解器 (Constraint Satisfaction Solver)
用于生成初始可行解，确保满足所有硬约束。
"""
import random
import time
from collections import defaultdict
from typing import Dict, List, Set, Tuple

from data.data_loader import ScheduleData, Task


def generate_feasible_solution(data: ScheduleData, max_attempts: int = 100) -> Dict[int, int]:
    """
    生成一个满足硬约束的可行解 (Wrapper function)
    """
    solver = ConstraintSolver(data)
    return solver.solve(max_attempts)


class ConstraintSolver:
    def __init__(self, data: ScheduleData):
        self.data = data
        self.genes = {}  # {task_id: encoded_gene}

        # 资源占用表 (用于快速查找空闲)
        # {time_slot: {resource_id}}
        self.occupied_teachers = defaultdict(set)
        self.occupied_classes = defaultdict(set)
        self.occupied_rooms = defaultdict(set)  # {time_slot: {room_id}}

    def solve(self, max_attempts: int = 10) -> Dict[int, int]:
        """
        尝试生成可行解，带有回溯重试机制
        """
        for attempt in range(max_attempts):
            self._reset()
            if self._construct_solution():
                return self.genes
            # print(f"Attempt {attempt + 1} failed, retrying...")

        # 如果实在找不到完美解，返回当前最好的结果（虽然会有冲突）
        # 但在排课场景下，最好是抛出异常或返回部分解
        print("Warning: Could not generate a completely conflict-free solution.")
        return self.genes

    def _reset(self):
        """重置状态"""
        self.genes = {}
        self.occupied_teachers.clear()
        self.occupied_classes.clear()
        self.occupied_rooms.clear()

    def _construct_solution(self) -> bool:
        """
        构造式启发算法
        策略:
        1. 将任务按"逻辑单元"(Teaching Unit)分组。
        2. 按难度排序(固定时间 > 大合班 > 特殊教室 > 普通)。
        3. 贪心分配。
        """
        # 1. 预处理：构建排课单元 (Unit)
        # unit = {'id': str, 'tasks': [task_obj], 'priority': int}
        units = self._build_scheduling_units()

        # 2. 排序：难排的先排
        # 优先级: 固定时间(100) > 特殊教室(50) > 人数/合班规模(10-40)
        units.sort(key=lambda x: x['priority'], reverse=True)

        # 3. 逐个分配
        for unit in units:
            success = self._assign_unit(unit)
            if not success:
                return False  # 本次构造失败
        return True

    def _build_scheduling_units(self) -> List[dict]:
        """将任务打包成排课单元，并计算优先级"""
        units = []
        processed_tasks = set()

        # 处理合班
        for gid, tids in self.data.combined_classes.items():
            valid_tids = [t for t in tids if t in self.data.tasks]
            if not valid_tids:
                continue

            tasks = [self.data.tasks[t] for t in valid_tids]
            priority = 20 + len(tasks) * 2  # 合班基础分 + 规模分

            # 检查是否有特殊约束
            has_fixed = any(t.is_fixed for t in tasks)
            has_special = any(t.course_attr == '07' for t in tasks)

            if has_fixed: priority += 100
            if has_special: priority += 50

            # 计算总人数 (用于筛选教室)
            total_students = sum(t.student_count for t in tasks)

            units.append({
                'id': f"G_{gid}",
                'tasks': tasks,
                'priority': priority,
                'total_students': total_students,
                'is_fixed': has_fixed,
                'is_special': has_special
            })
            processed_tasks.update(valid_tids)

        # 处理单班
        for tid, task in self.data.tasks.items():
            if tid in processed_tasks:
                continue

            priority = 10
            if task.is_fixed: priority += 100
            if task.course_attr == '07': priority += 50

            units.append({
                'id': f"T_{tid}",
                'tasks': [task],
                'priority': priority,
                'total_students': task.student_count,
                'is_fixed': task.is_fixed,
                'is_special': task.course_attr == '07'
            })

        return units

    def _assign_unit(self, unit: dict) -> bool:
        """为单个单元分配时间槽和教室"""
        tasks = unit['tasks']
        first_task = tasks[0]  # 用第一个任务的属性代表整个组

        # 1. 确定候选时间槽
        candidate_slots = []

        # 如果有固定时间约束
        fixed_slot = None
        if unit['is_fixed']:
            # 检查组内是否有任务被固定了
            for t in tasks:
                if t.id in self.data.fixed_time_tasks:
                    fixed_slot = self.data.fixed_time_tasks[t.id]
                    break

        if fixed_slot is not None:
            # 必须排在这个时间
            if self._is_slot_available_for_resources(fixed_slot, tasks):
                candidate_slots = [fixed_slot]
            else:
                return False  # 固定时间冲突，无法排课
        else:
            # 查找所有可用时间槽 (0-24)
            # 随机打乱以增加多样性
            all_slots = list(range(25))
            random.shuffle(all_slots)

            for slot in all_slots:
                if self._is_slot_available_for_resources(slot, tasks):
                    candidate_slots.append(slot)
                    if len(candidate_slots) >= 5:  # 找到5个候选即可
                        break

        if not candidate_slots:
            return False

        # 2. 确定候选教室
        # 筛选条件: 容量 >= 总人数, 属性匹配
        req_attr = '07' if unit['is_special'] else None  # 假设07是实验室

        valid_rooms = []
        for r in self.data.classrooms.values():
            # 容量检查
            if r.capacity < unit['total_students']:
                continue

            # 属性检查
            # 如果课程需要实验室(07)，教室必须是07
            # 如果课程是普通课，优先用普通教室，也可以用大教室，但暂时不限制严格对应
            if req_attr == '07':
                if r.attr != '07': continue

            # (可选) 避免把普通课排进实验室浪费资源，除非真的没地儿了
            # if req_attr != '07' and r.attr == '07': continue

            valid_rooms.append(r)

        # 随机排序教室
        random.shuffle(valid_rooms)

        # 3. 尝试组合 (Slot, Room)
        for slot in candidate_slots:
            for room in valid_rooms:
                # 检查教室在该时间槽是否被占用
                if room.classroom_id in self.occupied_rooms[slot]:
                    continue

                # 找到可行解！执行分配
                self._finalize_assignment(unit, slot, room.classroom_id)
                return True

        return False

    def _is_slot_available_for_resources(self, slot: int, tasks: List[Task]) -> bool:
        """检查时间槽对于 Teacher 和 Class 是否空闲"""
        for t in tasks:
            # 检查老师
            if t.teacher_id in self.occupied_teachers[slot]:
                return False
            # 检查班级
            if t.class_id in self.occupied_classes[slot]:
                return False
        return True

    def _finalize_assignment(self, unit: dict, slot: int, room_id: int):
        """将分配结果写入状态"""
        # 编码基因
        # 这里的 room_id 是字符串ID，我们需要转换成整数ID用于编码吗？
        # 之前的代码 fitness 中 _decode_gene 是 (gene & 0xFFFF)。
        # 这意味着 gene 存的是 classroom_id (int type or converted to int).
        # 如果数据库里的 classroomId 是字符串 '101', 必须保证能转成 int 或者有一套映射机制。
        # 假设: classroom_id 是数字字符串 '1001' -> int(1001)

        try:
            r_int = int(room_id)
        except ValueError:
            # 如果教室ID包含字母，需要做一个映射表。这里假设是纯数字
            r_int = hash(room_id) % 10000  # 极其简易的处理，实际应使用 Map

        gene_value = (slot << 16) | r_int

        for task in unit['tasks']:
            self.genes[task.id] = gene_value

            # 锁定资源
            self.occupied_teachers[slot].add(task.teacher_id)
            self.occupied_classes[slot].add(task.class_id)

        # 锁定教室 (整个单元只占一个教室)
        self.occupied_rooms[slot].add(room_id)




