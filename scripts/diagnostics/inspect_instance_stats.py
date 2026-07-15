# -*- coding: utf-8 -*-
"""
更合理的教室压力估计

输出每个实例：
1. Teacher_Count                    实例实际涉及教师数
2. Total_Units                      授课单元总数
3. Combined_Task_Ratio              合班比例（任务口径）
4. Peak_Room_Demand                 峰值总教室需求（估计）
5. Peak_Normal_Room_Demand          峰值普通教室需求（估计）
6. Peak_Lab_Room_Demand             峰值实验/特殊教室需求（估计）
7. Avg_Room_Load                    平均每时间片教室占用量

逻辑：
- 合班组按一个授课单元统计
- weekHours 视为该单元一周需要占用的时间片数
- 采用“负载最小时间片优先”的贪心装箱，模拟排课时的并发压力
- 普通课与实验课分开统计峰值
"""

import csv
import os
import sys
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.database import Database

DB_CONFIG = {
    'host': 'localhost',
    'port': 3306,
    'user': 'root',
    'password': os.getenv('DB_PASSWORD', ''),
    'database': 'test3'
}

INSTANCES = [
    'S1_Small', 'S2_Small', 'S3_Small', 'S4_Small',
    'M1_Medium', 'M2_Medium', 'M3_Medium', 'L1_Large'
]

N_SLOTS = 25  # 一周25个时间片


def norm(x):
    return "" if x is None else str(x).strip()


def get_college_id_from_class_id(class_id: str) -> str:
    cid = norm(class_id)
    return cid[4:6] if len(cid) >= 6 else "00"


def get_all_colleges_sorted_by_task_count(db):
    rows = db.fetch_all("SELECT classId FROM classtaskinfo")
    cnt = defaultdict(int)
    for r in rows:
        col = get_college_id_from_class_id(r["classId"])
        cnt[col] += 1
    result = sorted(cnt.items(), key=lambda x: x[1], reverse=True)
    return [k for k, _ in result]


def get_target_colleges(instance_name, all_colleges):
    if instance_name == 'S1_Small':
        return all_colleges[:1]
    elif instance_name == 'S2_Small':
        return all_colleges[1:2]
    elif instance_name == 'S3_Small':
        return all_colleges[:2]
    elif instance_name == 'S4_Small':
        return all_colleges[:3]
    elif instance_name == 'M1_Medium':
        return all_colleges[:5]
    elif instance_name == 'M2_Medium':
        return all_colleges[:6]
    elif instance_name == 'M3_Medium':
        return all_colleges[:8]
    elif instance_name == 'L1_Large':
        return None
    else:
        return all_colleges[:1]


def load_instance_tasks(db, instance_name):
    all_cols = get_all_colleges_sorted_by_task_count(db)
    target_cols = get_target_colleges(instance_name, all_cols)

    rows = db.fetch_all("""
        SELECT id, classId, teacherId, courseId, courseAttr,
               courseClassId, courseClassName, studentNumber,
               courseHours, weekHours, weeks, isFix, classTime
        FROM classtaskinfo
    """)

    kept = []
    for r in rows:
        col = get_college_id_from_class_id(r["classId"])
        if target_cols is not None and col not in target_cols:
            continue
        kept.append(r)

    return target_cols, kept


def build_units(tasks):
    """
    构建授课单元：
    - courseClassId 非空：按合班组聚合
    - 否则：单任务单元
    返回：
      units: [{
          unit_id,
          week_hours,
          course_attr
      }, ...]
      combined_task_ids
      combined_group_count
    """
    grouped = defaultdict(list)
    singletons = []
    combined_task_ids = set()

    for t in tasks:
        gid = norm(t["courseClassId"])
        if gid:
            grouped[gid].append(t)
        else:
            singletons.append(t)

    units = []
    combined_group_count = 0

    for gid, arr in grouped.items():
        if len(arr) >= 2:
            combined_group_count += 1
            for t in arr:
                combined_task_ids.add(t["id"])

            # weekHours 取组内最大值，避免异常数据低估
            week_hours = max(int(t["weekHours"] or 0) for t in arr)
            # 课程属性取组内第一个；正常应一致
            course_attr = norm(arr[0]["courseAttr"])

            units.append({
                "unit_id": gid,
                "week_hours": week_hours,
                "course_attr": course_attr
            })
        else:
            t = arr[0]
            units.append({
                "unit_id": f"T_{t['id']}",
                "week_hours": int(t["weekHours"] or 0),
                "course_attr": norm(t["courseAttr"])
            })

    for t in singletons:
        units.append({
            "unit_id": f"T_{t['id']}",
            "week_hours": int(t["weekHours"] or 0),
            "course_attr": norm(t["courseAttr"])
        })

    return units, combined_task_ids, combined_group_count


def greedy_pack_peak(units, n_slots=25):
    """
    用贪心法把每个授课单元需要的 week_hours 个“课次”
    尽量均匀铺到 25 个时间片中，估计峰值并发需求。

    返回：
      peak_total
      peak_normal
      peak_lab
      avg_total
      slot_total_loads
      slot_normal_loads
      slot_lab_loads
    """
    slot_total_loads = [0] * n_slots
    slot_normal_loads = [0] * n_slots
    slot_lab_loads = [0] * n_slots

    # 先排周学时大的单元，避免低估峰值
    units_sorted = sorted(units, key=lambda x: x["week_hours"], reverse=True)

    for u in units_sorted:
        need = max(0, int(u["week_hours"]))
        is_lab = (u["course_attr"] == "07")

        for _ in range(need):
            # 找当前总负载最小的时间片
            min_load = min(slot_total_loads)
            candidate_slots = [i for i, v in enumerate(slot_total_loads) if v == min_load]

            # 在这些最小负载时间片里，优先选对应类型负载更小的
            if is_lab:
                best_slot = min(candidate_slots, key=lambda i: slot_lab_loads[i])
            else:
                best_slot = min(candidate_slots, key=lambda i: slot_normal_loads[i])

            slot_total_loads[best_slot] += 1
            if is_lab:
                slot_lab_loads[best_slot] += 1
            else:
                slot_normal_loads[best_slot] += 1

    peak_total = max(slot_total_loads) if slot_total_loads else 0
    peak_normal = max(slot_normal_loads) if slot_normal_loads else 0
    peak_lab = max(slot_lab_loads) if slot_lab_loads else 0
    avg_total = round(sum(slot_total_loads) / n_slots, 4) if n_slots > 0 else 0.0

    return (
        peak_total,
        peak_normal,
        peak_lab,
        avg_total,
        slot_total_loads,
        slot_normal_loads,
        slot_lab_loads
    )


def analyze_instance(db, instance_name):
    target_cols, tasks = load_instance_tasks(db, instance_name)

    teacher_ids = {norm(t["teacherId"]) for t in tasks if norm(t["teacherId"])}
    teacher_count = len(teacher_ids)

    units, combined_task_ids, combined_group_count = build_units(tasks)

    total_tasks = len(tasks)
    total_units = len(units)
    combined_task_count = len(combined_task_ids)

    combined_task_ratio = combined_task_count / total_tasks if total_tasks else 0.0
    combined_unit_ratio = combined_group_count / total_units if total_units else 0.0

    total_week_hours = sum(int(u["week_hours"]) for u in units)

    (
        peak_total,
        peak_normal,
        peak_lab,
        avg_total,
        slot_total_loads,
        slot_normal_loads,
        slot_lab_loads
    ) = greedy_pack_peak(units, N_SLOTS)

    return {
        "Instance": instance_name,
        "Target_Colleges": "ALL" if target_cols is None else "|".join(target_cols),
        "Task_Count": total_tasks,
        "Teacher_Count": teacher_count,
        "Total_Units": total_units,
        "Total_Weekly_Hours": total_week_hours,
        "Peak_Room_Demand": peak_total,
        "Peak_Normal_Room_Demand": peak_normal,
        "Peak_Lab_Room_Demand": peak_lab,
        "Avg_Room_Load": avg_total,
        "Combined_Group_Count": combined_group_count,
        "Combined_Task_Count": combined_task_count,
        "Combined_Task_Ratio": round(combined_task_ratio, 4),
        "Combined_Unit_Ratio": round(combined_unit_ratio, 4),
        "Slot_Total_Loads": "|".join(map(str, slot_total_loads)),
        "Slot_Normal_Loads": "|".join(map(str, slot_normal_loads)),
        "Slot_Lab_Loads": "|".join(map(str, slot_lab_loads)),
    }


def main():
    db = Database(DB_CONFIG)
    db.connect()

    try:
        results = []

        print("=" * 120)
        print("实例教室压力估计开始")
        print("=" * 120)

        for inst in INSTANCES:
            row = analyze_instance(db, inst)
            results.append(row)

            print(f"\n[{inst}]")
            print(f"Teacher_Count             : {row['Teacher_Count']}")
            print(f"Task_Count                : {row['Task_Count']}")
            print(f"Total_Units               : {row['Total_Units']}")
            print(f"Total_Weekly_Hours        : {row['Total_Weekly_Hours']}")
            print(f"Peak_Room_Demand          : {row['Peak_Room_Demand']}")
            print(f"Peak_Normal_Room_Demand   : {row['Peak_Normal_Room_Demand']}")
            print(f"Peak_Lab_Room_Demand      : {row['Peak_Lab_Room_Demand']}")
            print(f"Avg_Room_Load             : {row['Avg_Room_Load']}")
            print(f"Combined_Task_Ratio       : {row['Combined_Task_Ratio']}")
            print(f"Combined_Unit_Ratio       : {row['Combined_Unit_Ratio']}")

        with open("instance_room_pressure.csv", "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)

        print("\n" + "=" * 120)
        print("完成，已输出：instance_room_pressure.csv")
        print("=" * 120)

    finally:
        db.close()


if __name__ == "__main__":
    main()



# # -*- coding: utf-8 -*-
# """
# 实例过滤与数据一致性诊断（最终版）
#
# 用途：
# 1. 检查每个实例中：
#    - 学院数量
#    - 任务数
#    - 班级数
#    - DataLoader加载的教师数 / 教室数
#    - 任务实际涉及的去重教师数
#    - 课程属性约束下的候选教室数
#    - 合班任务比例 / 合班授课单元比例
#
# 2. 重点核查：
#    - 教师是否未按实例过滤
#    - 教室是否未按实例过滤
#    - teacherinfo 与 classtaskinfo 是否存在 teacherId 不一致问题
#
# 输出：
# - 控制台打印
# - instance_filter_diagnosis_v2.csv
# - teacher_mismatch_details.csv
# """
#
# import csv
# from collections import defaultdict
#
# from data.data_loader import DataLoader
# from data.database import Database
#
# DB_CONFIG = {
#     'host': 'localhost',
#     'port': 3306,
#     'user': 'root',
#     'password': os.getenv('DB_PASSWORD', ''),
#     'database': 'test3'
# }
#
# INSTANCES = [
#     'S1_Small', 'S2_Small', 'S3_Small', 'S4_Small',
#     'M1_Medium', 'M2_Medium', 'M3_Medium', 'L1_Large'
# ]
#
#
# def norm(x):
#     """统一转成去空格字符串，避免 teacherId 因格式差异误判"""
#     if x is None:
#         return ""
#     return str(x).strip()
#
#
# def get_target_colleges_by_rule(instance_name, all_college_ids):
#     """复现 DataLoader 中的学院选取规则"""
#     if instance_name == 'S1_Small':
#         return all_college_ids[:1] if all_college_ids else []
#     elif instance_name == 'S2_Small':
#         return all_college_ids[1:2] if len(all_college_ids) > 1 else []
#     elif instance_name == 'S3_Small':
#         return all_college_ids[:2] if len(all_college_ids) > 1 else []
#     elif instance_name == 'S4_Small':
#         return all_college_ids[:3] if len(all_college_ids) > 2 else []
#     elif instance_name == 'M1_Medium':
#         return all_college_ids[:5]
#     elif instance_name == 'M2_Medium':
#         return all_college_ids[:6]
#     elif instance_name == 'M3_Medium':
#         return all_college_ids[:8]
#     elif instance_name == 'L1_Large':
#         return None
#     else:
#         return all_college_ids[:1]
#
#
# def fetch_global_teacher_ids():
#     """从 teacherinfo 表读取全局教师ID集合"""
#     db = Database(DB_CONFIG)
#     db.connect()
#     try:
#         rows = db.fetch_all("SELECT teacherId FROM teacherinfo")
#         return {norm(r.get('teacherId')) for r in rows if norm(r.get('teacherId'))}
#     finally:
#         db.close()
#
#
# def fetch_global_classroom_ids():
#     """从 classroominfo 表读取全局教室ID集合"""
#     db = Database(DB_CONFIG)
#     db.connect()
#     try:
#         rows = db.fetch_all("SELECT classroomId FROM classroominfo")
#         return {norm(r.get('classroomId')) for r in rows if norm(r.get('classroomId'))}
#     finally:
#         db.close()
#
#
# def estimate_candidate_classrooms(data):
#     """
#     粗略估计当前实例在“容量+课程属性”条件下可用的候选教室集合
#     注意：这是实例任务可能涉及的候选教室，不是最终真正使用的教室
#     """
#     candidate_room_ids = set()
#
#     for task in data.tasks.values():
#         stu = task.student_count
#         req_attr = '07' if task.course_attr == '07' else None
#
#         for room in data.classrooms.values():
#             room_id = norm(room.classroom_id)
#             if not room_id:
#                 continue
#             if room.capacity < stu:
#                 continue
#             if req_attr == '07' and norm(room.attr) != '07':
#                 continue
#             candidate_room_ids.add(room_id)
#
#     return candidate_room_ids
#
#
# def analyze_instance(instance_name, global_teacher_ids, global_classroom_ids):
#     # 先用 DataLoader 的学院排序规则拿 target colleges
#     loader_tmp = DataLoader(instance_name, DB_CONFIG)
#     all_colleges = loader_tmp.get_college_list()
#     all_college_ids = [c['college_id'] for c in all_colleges]
#     target_colleges = get_target_colleges_by_rule(instance_name, all_college_ids)
#
#     # 正式加载实例
#     loader = DataLoader(instance_name, DB_CONFIG)
#     data = loader.load()
#
#     # 已加载学院
#     loaded_colleges = sorted({
#         norm(c.college_id) for c in data.classes.values()
#         if hasattr(c, 'college_id') and norm(c.college_id)
#     })
#
#     # 实例实际涉及教师
#     involved_teacher_ids = {
#         norm(t.teacher_id) for t in data.tasks.values()
#         if norm(t.teacher_id)
#     }
#
#     # teacherinfo 中存在，但当前实例没用到
#     teacher_unused_in_instance = global_teacher_ids - involved_teacher_ids
#
#     # 当前实例任务中出现，但 teacherinfo 中不存在
#     teacher_missing_in_teacherinfo = involved_teacher_ids - global_teacher_ids
#
#     # 合班统计
#     combined_groups = {
#         gid: tids for gid, tids in data.combined_classes.items()
#         if norm(gid) and len(tids) >= 2
#     }
#     combined_task_ids = set()
#     for tids in combined_groups.values():
#         combined_task_ids.update(tids)
#
#     total_tasks = len(data.tasks)
#     combined_task_count = len(combined_task_ids)
#     combined_group_count = len(combined_groups)
#
#     combined_task_ratio = combined_task_count / total_tasks if total_tasks > 0 else 0.0
#
#     single_task_unit_count = total_tasks - combined_task_count
#     total_units = combined_group_count + single_task_unit_count
#     combined_unit_ratio = combined_group_count / total_units if total_units > 0 else 0.0
#
#     # 候选教室统计
#     candidate_room_ids = estimate_candidate_classrooms(data)
#     classroom_unused_in_instance = global_classroom_ids - candidate_room_ids
#
#     # 判断逻辑
#     teachers_filtered = (len(global_teacher_ids) == len(involved_teacher_ids))
#     classrooms_filtered = (len(global_classroom_ids) == len(candidate_room_ids))
#
#     result = {
#         "Instance": instance_name,
#         "Target_Colleges_From_Rule": "ALL" if target_colleges is None else "|".join(target_colleges),
#         "Loaded_College_Count": len(loaded_colleges),
#         "Loaded_College_IDs": "ALL" if target_colleges is None else "|".join(loaded_colleges),
#
#         "Loaded_Tasks": len(data.tasks),
#         "Loaded_Classes": len(data.classes),
#
#         # DataLoader 实际装入数量
#         "Loaded_Teachers": len(data.teachers),
#         "Loaded_Classrooms": len(data.classrooms),
#
#         # 数据库全局唯一数量
#         "Global_Teacher_Count": len(global_teacher_ids),
#         "Global_Classroom_Count": len(global_classroom_ids),
#
#         # 当前实例真实涉及数量
#         "Involved_Teachers_By_Tasks": len(involved_teacher_ids),
#         "Candidate_Classrooms_By_Tasks": len(candidate_room_ids),
#
#         # 一致性检查
#         "Teacher_IDs_Missing_In_TeacherInfo": len(teacher_missing_in_teacherinfo),
#         "Teacher_IDs_Unused_In_Instance": len(teacher_unused_in_instance),
#         "Classrooms_Unused_In_Instance": len(classroom_unused_in_instance),
#
#         # 合班统计
#         "Combined_Group_Count": combined_group_count,
#         "Combined_Task_Count": combined_task_count,
#         "Combined_Task_Ratio": round(combined_task_ratio, 4),
#         "Total_Units": total_units,
#         "Combined_Unit_Ratio": round(combined_unit_ratio, 4),
#
#         # 最终判定
#         "Teachers_Filtered": "YES" if teachers_filtered else "NO",
#         "Classrooms_Filtered": "YES" if classrooms_filtered else "NO",
#         "TeacherInfo_Consistent": "YES" if len(teacher_missing_in_teacherinfo) == 0 else "NO"
#     }
#
#     detail = {
#         "instance": instance_name,
#         "teacher_missing_in_teacherinfo": sorted(list(teacher_missing_in_teacherinfo)),
#         "teacher_unused_in_instance": sorted(list(teacher_unused_in_instance))[:200],  # 防止过长
#         "candidate_room_ids_sample": sorted(list(candidate_room_ids))[:200]
#     }
#
#     return result, detail
#
#
# def main():
#     global_teacher_ids = fetch_global_teacher_ids()
#     global_classroom_ids = fetch_global_classroom_ids()
#
#     all_results = []
#     all_details = []
#
#     print("=" * 120)
#     print("实例过滤与数据一致性诊断开始")
#     print("=" * 120)
#     print(f"Global teacherinfo unique teacherIds : {len(global_teacher_ids)}")
#     print(f"Global classroominfo unique roomIds  : {len(global_classroom_ids)}")
#
#     for inst in INSTANCES:
#         try:
#             row, detail = analyze_instance(inst, global_teacher_ids, global_classroom_ids)
#             all_results.append(row)
#             all_details.append(detail)
#
#             print(f"\n[{inst}]")
#             for k, v in row.items():
#                 print(f"{k:35s}: {v}")
#
#             if row["Teachers_Filtered"] == "NO":
#                 print(">>> 结论：教师未按实例过滤")
#             if row["Classrooms_Filtered"] == "NO":
#                 print(">>> 结论：教室未按实例过滤")
#             if row["TeacherInfo_Consistent"] == "NO":
#                 print(">>> 警告：classtaskinfo 中存在 teacherinfo 表中不存在的 teacherId")
#
#         except Exception as e:
#             print(f"\n[{inst}] 运行失败: {e}")
#
#     # 保存主结果
#     if all_results:
#         with open("instance_filter_diagnosis_v2.csv", "w", newline="", encoding="utf-8-sig") as f:
#             writer = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
#             writer.writeheader()
#             writer.writerows(all_results)
#
#     # 保存细节
#     with open("teacher_mismatch_details.csv", "w", newline="", encoding="utf-8-sig") as f:
#         fieldnames = [
#             "instance",
#             "teacher_missing_in_teacherinfo",
#             "teacher_unused_in_instance",
#             "candidate_room_ids_sample"
#         ]
#         writer = csv.DictWriter(f, fieldnames=fieldnames)
#         writer.writeheader()
#         for d in all_details:
#             writer.writerow({
#                 "instance": d["instance"],
#                 "teacher_missing_in_teacherinfo": "|".join(d["teacher_missing_in_teacherinfo"]),
#                 "teacher_unused_in_instance": "|".join(d["teacher_unused_in_instance"]),
#                 "candidate_room_ids_sample": "|".join(d["candidate_room_ids_sample"]),
#             })
#
#     print("\n" + "=" * 120)
#     print("诊断完成：")
#     print("1. instance_filter_diagnosis_v2.csv")
#     print("2. teacher_mismatch_details.csv")
#     print("=" * 120)
#
#
# if __name__ == "__main__":
#     main()
#
# # # -*- coding: utf-8 -*-
# # """
# # 诊断每个实例的数据范围与“是否过滤”情况
# #
# # 用途：
# # 1. 统计每个实例的学院数量、任务数、班级数、教师数、教室数
# # 2. 区分：
# #    - DataLoader加载值（当前实验实际使用）
# #    - 实例实际涉及的去重教师数
# #    - 依据课程属性估计的候选教室数
# # 3. 统计合班比例
# # 4. 输出一个总表 CSV，便于论文整理
# #
# # 运行前请确认：
# # - 当前项目目录结构可正常 import data.data_loader
# # - 数据库配置正确
# # """
# #
# # import csv
# # from collections import defaultdict
# #
# # from data.data_loader import DataLoader
# #
# # DB_CONFIG = {
# #     'host': 'localhost',
# #     'port': 3306,
# #     'user': 'root',
# #     'password': os.getenv('DB_PASSWORD', ''),
# #     'database': 'test3'
# # }
# #
# # INSTANCES = [
# #     'S1_Small', 'S2_Small', 'S3_Small', 'S4_Small',
# #     'M1_Medium', 'M2_Medium', 'M3_Medium', 'L1_Large'
# # ]
# #
# #
# # def get_target_colleges(loader: DataLoader):
# #     """复现 DataLoader.load() 中的实例学院选择逻辑"""
# #     all_colleges = loader.get_college_list()
# #     college_ids = [c['college_id'] for c in all_colleges]
# #
# #     if loader.instance_name == 'S1_Small':
# #         return college_ids[:1] if college_ids else []
# #     elif loader.instance_name == 'S2_Small':
# #         return college_ids[1:2] if len(college_ids) > 1 else []
# #     elif loader.instance_name == 'S3_Small':
# #         return college_ids[:2] if len(college_ids) > 1 else []
# #     elif loader.instance_name == 'S4_Small':
# #         return college_ids[:3] if len(college_ids) > 2 else []
# #     elif loader.instance_name == 'M1_Medium':
# #         return college_ids[:5]
# #     elif loader.instance_name == 'M2_Medium':
# #         return college_ids[:6]
# #     elif loader.instance_name == 'M3_Medium':
# #         return college_ids[:8]
# #     elif loader.instance_name == 'L1_Large':
# #         return None
# #     else:
# #         return college_ids[:1]
# #
# #
# # def estimate_candidate_classrooms(data):
# #     """
# #     粗略估计当前实例任务可用的候选教室数量（按课程属性，不考虑时间冲突）
# #     说明：
# #     - 若 course_attr == '07'，只统计 attr == '07' 的教室
# #     - 其他课程统计容量满足即可（与 constraint_solver 逻辑一致）
# #     """
# #     candidate_room_ids = set()
# #
# #     for task in data.tasks.values():
# #         req_attr = '07' if task.course_attr == '07' else None
# #         stu = task.student_count
# #
# #         for room in data.classrooms.values():
# #             if room.capacity < stu:
# #                 continue
# #             if req_attr == '07' and room.attr != '07':
# #                 continue
# #             candidate_room_ids.add(room.classroom_id)
# #
# #     return len(candidate_room_ids), candidate_room_ids
# #
# #
# # def analyze_instance(instance_name):
# #     # 先建一个 loader，用来拿学院排序信息
# #     loader_for_colleges = DataLoader(instance_name, DB_CONFIG)
# #     target_colleges = get_target_colleges(loader_for_colleges)
# #
# #     # 再正式 load
# #     loader = DataLoader(instance_name, DB_CONFIG)
# #     data = loader.load()
# #
# #     # 学院集合（由已加载 classes 推出）
# #     loaded_colleges = sorted({
# #         c.college_id for c in data.classes.values()
# #         if hasattr(c, 'college_id') and c.college_id
# #     })
# #
# #     # 实例实际涉及的教师（从任务中反推）
# #     involved_teacher_ids = sorted({
# #         t.teacher_id for t in data.tasks.values()
# #         if t.teacher_id
# #     })
# #
# #     # 合班统计
# #     combined_groups = {
# #         gid: tids for gid, tids in data.combined_classes.items()
# #         if gid and len(tids) >= 2
# #     }
# #     combined_task_ids = set()
# #     for tids in combined_groups.values():
# #         combined_task_ids.update(tids)
# #
# #     total_tasks = len(data.tasks)
# #     combined_task_count = len(combined_task_ids)
# #     combined_group_count = len(combined_groups)
# #
# #     combined_task_ratio = (combined_task_count / total_tasks) if total_tasks > 0 else 0.0
# #
# #     # 授课单元口径：合班组 + 非合班单任务
# #     single_task_unit_count = total_tasks - combined_task_count
# #     total_units = combined_group_count + single_task_unit_count
# #     combined_unit_ratio = (combined_group_count / total_units) if total_units > 0 else 0.0
# #
# #     # 粗略候选教室统计
# #     candidate_room_count, candidate_room_ids = estimate_candidate_classrooms(data)
# #
# #     # 结果
# #     result = {
# #         "Instance": instance_name,
# #
# #         # 学院
# #         "Target_Colleges_From_Rule": "ALL" if target_colleges is None else "|".join(target_colleges),
# #         "Loaded_College_Count": len(loaded_colleges),
# #         "Loaded_College_IDs": "ALL" if target_colleges is None else "|".join(loaded_colleges),
# #
# #         # 已加载统计（当前 run_experiment 直接用的）
# #         "Loaded_Tasks": len(data.tasks),
# #         "Loaded_Classes": len(data.classes),
# #         "Loaded_Teachers": len(data.teachers),       # 当前代码里这是全表加载
# #         "Loaded_Classrooms": len(data.classrooms),   # 当前代码里这是全表加载
# #
# #         # 实际涉及统计（更适合论文解释）
# #         "Involved_Teachers_By_Tasks": len(involved_teacher_ids),
# #         "Candidate_Classrooms_By_Tasks": candidate_room_count,
# #
# #         # 合班统计
# #         "Combined_Group_Count": combined_group_count,
# #         "Combined_Task_Count": combined_task_count,
# #         "Combined_Task_Ratio": round(combined_task_ratio, 4),
# #         "Total_Units": total_units,
# #         "Combined_Unit_Ratio": round(combined_unit_ratio, 4),
# #
# #         # 过滤判定
# #         "Teachers_Filtered?": "NO" if len(data.teachers) >= len(involved_teacher_ids) else "CHECK",
# #         "Classrooms_Filtered?": "NO" if len(data.classrooms) >= candidate_room_count else "CHECK",
# #     }
# #
# #     return result
# #
# #
# # def main():
# #     results = []
# #
# #     print("=" * 120)
# #     print("实例过滤诊断开始")
# #     print("=" * 120)
# #
# #     for inst in INSTANCES:
# #         try:
# #             row = analyze_instance(inst)
# #             results.append(row)
# #
# #             print(f"\n[{inst}]")
# #             for k, v in row.items():
# #                 print(f"{k:30s}: {v}")
# #
# #             # 重点提示
# #             if row["Loaded_Teachers"] != row["Involved_Teachers_By_Tasks"]:
# #                 print(">>> 教师存在“未按实例裁剪”的现象：Loaded_Teachers != Involved_Teachers_By_Tasks")
# #             if row["Loaded_Classrooms"] != row["Candidate_Classrooms_By_Tasks"]:
# #                 print(">>> 教室存在“未按实例裁剪”的现象：Loaded_Classrooms != Candidate_Classrooms_By_Tasks")
# #
# #         except Exception as e:
# #             print(f"\n[{inst}] 运行失败: {e}")
# #
# #     # 输出 CSV
# #     if results:
# #         out_csv = "instance_filter_diagnosis.csv"
# #         fieldnames = list(results[0].keys())
# #         with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
# #             writer = csv.DictWriter(f, fieldnames=fieldnames)
# #             writer.writeheader()
# #             writer.writerows(results)
# #
# #         print("\n" + "=" * 120)
# #         print(f"诊断完成，结果已保存到: {out_csv}")
# #         print("=" * 120)
# #
# #
# # if __name__ == "__main__":
# #     main()
