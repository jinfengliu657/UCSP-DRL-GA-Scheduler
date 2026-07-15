"""
数据加载器 (Data Loader) - Final Version
文件名: data/data_loader.py
修改内容:
1. 修复了班级统计 Bug: 在加载 ClassInfo 时，增加了基于 college_id 的过滤逻辑。
2. 解析规则: ClassId (e.g., 2021020101) 的第 5-6 位 (索引 4:6) 被解析为学院代码。
3. 实例控制: 根据 S1/M1/L1 等实例名称，自动锁定目标学院，仅加载相关数据。
"""
from typing import Dict, List, Any
from collections import defaultdict
from .database import Database


class Task:
    def __init__(self, row: Dict[str, Any]):
        self.id = row['id']
        self.class_id = str(row['classId'])
        self.teacher_id = row['teacherId']
        self.course_id = row['courseId']
        self.course_attr = row['courseAttr']
        self.course_class_id = row['courseClassId']
        self.course_class_name = row['courseClassName']
        self.student_count = row['studentNumber']
        self.is_fixed = row['isFix'] == '1'
        self.class_time = row['classTime']
        self.weeks = row.get('weeks', '1-16')

        # 解析学生所属学院 (classId: 2021020101 -> Year:2021, Col:02, Maj:01, Cls:01)
        # 取第 5,6 位作为学院代码 (索引 4:6)
        cid_str = self.class_id.strip()
        if len(cid_str) >= 6:
            self.student_college_id = cid_str[4:6]
        else:
            self.student_college_id = "00"


class ClassInfo:
    def __init__(self, row):
        self.id = row['id']
        self.class_id = str(row['classId'])
        self.class_name = row['className']

        # 同样解析学院ID，用于过滤班级列表
        cid_str = self.class_id.strip()
        if len(cid_str) >= 6:
            self.college_id = cid_str[4:6]
        else:
            self.college_id = "00"


class Teacher:
    def __init__(self, row):
        self.id = row['id']
        self.teacher_id = row['teacherId']
        self.teacher_name = row['teacherName']


class Classroom:
    def __init__(self, row):
        self.id = row['id']
        self.classroom_id = row['classroomId']
        self.capacity = row['capacity']
        self.attr = row['classroomAttr']
        self.classroom_name = row['classroomName']
        self.teach_building_id = row.get('teachBuildingId', 'Unknown')


class ScheduleData:
    def __init__(self):
        self.tasks = {}
        self.classes = {}
        self.teachers = {}
        self.classrooms = {}
        self.fixed_time_tasks = {}
        self.combined_classes = defaultdict(list)
        self.num_tasks = 0
        self.num_classes = 0
        self.num_teachers = 0
        self.num_classrooms = 0


class DataLoader:
    def __init__(self, instance_name: str, db_config: Dict[str, str]):
        """
        初始化加载器
        :param instance_name: 实例名称 (e.g., 'S1_Small', 'M2_Medium', 'L1_Large')
        :param db_config: 数据库配置
        """
        self.instance_name = instance_name
        self.db = Database(db_config)
        self.db.connect()

    def load(self) -> ScheduleData:
        """
        根据 instance_name 自动加载对应规模的数据
        """
        # 1. 获取所有学院列表 (按任务量降序排序)
        all_colleges = self.get_college_list()
        college_ids = [c['college_id'] for c in all_colleges]

        target_colleges = []

        # 2. 实例定义逻辑 (S系列: 1, 1, 2, 3)
        if self.instance_name == 'S1_Small':
            target_colleges = college_ids[:1] if college_ids else []
        elif self.instance_name == 'S2_Small':
            target_colleges = college_ids[1:2] if len(college_ids) > 1 else []
        elif self.instance_name == 'S3_Small':
            target_colleges = college_ids[:2] if len(college_ids) > 1 else []
        elif self.instance_name == 'S4_Small':
            target_colleges = college_ids[:3] if len(college_ids) > 2 else []

        # M系列: (5, 6, 8)
        elif self.instance_name == 'M1_Medium':
            target_colleges = college_ids[:5]
        elif self.instance_name == 'M2_Medium':
            target_colleges = college_ids[:6]
        elif self.instance_name == 'M3_Medium':
            target_colleges = college_ids[:8]

        # L系列: 全校
        elif self.instance_name == 'L1_Large':
            target_colleges = None  # None 表示不过滤，加载所有

        else:
            print(f"Warning: Unknown instance name '{self.instance_name}', loading first college only.")
            target_colleges = college_ids[:1]

        print(f"DataLoader: Loading Instance [{self.instance_name}]")
        print(f"Target Colleges: {target_colleges if target_colleges else 'ALL'}")

        return self._load_data_internal(target_colleges)

    def _load_data_internal(self, college_ids: List[str] = None) -> ScheduleData:
        """内部实际加载逻辑"""
        data = ScheduleData()

        # 1. 加载并过滤班级 (Classes) [核心修复点]
        # 先加载数据库中所有班级
        all_classes_raw = self.db.fetch_all("SELECT * FROM classinfo")
        all_classes_objs = {r['classId']: ClassInfo(r) for r in all_classes_raw}

        if college_ids is not None:
            # 仅保留 college_id 在目标列表中的班级
            # 例如: 如果 target_colleges=['02'], 那么只保留 classId 中间是 '02' 的班级
            data.classes = {
                cid: cinfo
                for cid, cinfo in all_classes_objs.items()
                if cinfo.college_id in college_ids
            }
        else:
            # 全校模式，保留所有
            data.classes = all_classes_objs

        # 2. 加载其他资源 (教师、教室)
        data.teachers = {r['teacherId']: Teacher(r) for r in self.db.fetch_all("SELECT * FROM teacherinfo")}
        data.classrooms = {r['classroomId']: Classroom(r) for r in self.db.fetch_all("SELECT * FROM classroominfo")}

        # 3. 加载任务并过滤
        raw_tasks = self.db.fetch_all("SELECT * FROM classtaskinfo")

        tasks = {}
        fixed_tasks = {}
        combined = defaultdict(list)

        for row in raw_tasks:
            task = Task(row)

            # 过滤逻辑: 仅保留 target_colleges 中的学院
            if college_ids is not None:
                if task.student_college_id not in college_ids:
                    continue

            tasks[task.id] = task

            if task.is_fixed and task.class_time:
                slot = self._parse_time_slot(task.class_time)
                if slot is not None:
                    fixed_tasks[task.id] = slot

            if task.course_class_id:
                combined[task.course_class_id].append(task.id)

        data.tasks = tasks
        data.fixed_time_tasks = fixed_tasks
        data.combined_classes = dict(combined)

        data.num_tasks = len(tasks)
        data.num_classes = len(data.classes)
        data.num_teachers = len(data.teachers)
        data.num_classrooms = len(data.classrooms)

        self.db.close()
        return data

    def get_college_list(self) -> List[Dict]:
        """统计每个学院的任务数量并排序"""
        raw_tasks = self.db.fetch_all("SELECT classId FROM classtaskinfo")
        college_counts = defaultdict(int)
        for row in raw_tasks:
            cid = str(row['classId']).strip()
            # 按照 2021020101 -> 02 的规则截取
            if len(cid) >= 6:
                col_id = cid[4:6]
                college_counts[col_id] += 1

        result = [{'college_id': k, 'task_count': v} for k, v in college_counts.items()]
        result.sort(key=lambda x: x['task_count'], reverse=True)
        return result

    def _parse_time_slot(self, class_time: str) -> int:
        try:
            val = int(class_time) - 1
            return val if 0 <= val < 25 else None
        except:
            return None