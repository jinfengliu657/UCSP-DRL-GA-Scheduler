# import os
# import json
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
# from data.data_loader import DataLoader
#
# # ==========================================
# # 学术图表全局设置
# # ==========================================
# plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
# plt.rcParams['axes.unicode_minus'] = False
# plt.rcParams['savefig.dpi'] = 600
#
# DB_CONFIG = {
#     'host': 'localhost',
#     'port': 3306,
#     'user': 'root',
#     'password': os.getenv('DB_PASSWORD', ''),
#     'database': 'test3'
# }
#
# # --- 核心配色方案 ---
# COLOR_FIXED_BG = '#D5F5E3'  # 浅青绿：固定预排课程 (优先级1)
# COLOR_SHARED_BG = '#F5B7B1'  # 浅砖红：合班共享课程 (优先级2)
# COLOR_NORMAL_BG = '#EBF5FB'  # 浅灰蓝：普通独立课程 (优先级3)
# COLOR_EMPTY_BG = '#FFFFFF'  # 纯白：空白时段
# COLOR_HEADER_BG = '#F2F4F4'  # 浅灰：表头
# COLOR_GRID_LINE = '#D5D8DC'  # 浅灰：网格线
# COLOR_TEXT_MAIN = '#2C3E50'  # 深炭黑：主文字
#
#
# def format_course_name(name):
#     """智能处理长课程名"""
#     if not name: return ""
#     name = str(name).strip()
#     if len(name) > 8:
#         if '\n' not in name and len(name) >= 10:
#             mid = len(name) // 2
#             name = name[:mid] + '\n' + name[mid:]
#     return name
#
#
# def plot_6_classes_schedule(genes, data, output_dir):
#     print("📊 正在执行【6班级纵向矩阵 + 绿色固定优先 + 图例说明】排版...")
#
#     # 1. 严格锁定指定的 6 个班级 (机械、电气、特定ID)
#     target_names_hints = ["机械23-2", "机械23-4", "电气23-1", "电气23-2", "2022110201", "2022110202"]
#     target_class_ids = []
#     target_class_names = []
#
#     for hint in target_names_hints:
#         found = False
#         for cid, cinfo in data.classes.items():
#             if (hint in cinfo.class_name or hint == str(cid)) and cid not in target_class_ids:
#                 target_class_ids.append(cid)
#                 target_class_names.append(cinfo.class_name)
#                 found = True
#                 break
#
#         if not found:
#             print(f"⚠️ 提示：未查到 '{hint}' 的名字，已强制按ID提取数据")
#             force_id = int(hint) if hint.isdigit() else hint
#             target_class_ids.append(force_id)
#             target_class_names.append(f"班级号 {hint}")
#
#     scheds = {cid: [[None for _ in range(5)] for _ in range(5)] for cid in target_class_ids}
#
#     for task_id_str, gene in genes.items():
#         task_id = int(task_id_str)
#         if task_id not in data.tasks: continue
#         task = data.tasks[task_id]
#
#         if task.class_id in target_class_ids:
#             slot = gene >> 16
#             room_id = gene & 0xFFFF
#             day, period = slot // 5, slot % 5
#
#             if 0 <= day < 5 and 0 <= period < 5:
#                 t_id = str(task.teacher_id).strip()
#                 t_name = data.teachers[t_id].teacher_name if t_id in data.teachers else f"{t_id}"
#
#                 r_key = str(room_id).zfill(6) if str(room_id) not in data.classrooms else str(room_id)
#                 r_name = data.classrooms[r_key].classroom_name if r_key in data.classrooms else f"{room_id}"
#
#                 # 提取数据库中的 isFix 字段
#                 is_fixed = getattr(task, 'is_fixed', False)
#
#                 scheds[task.class_id][day][period] = {
#                     'course': task.course_class_name,
#                     'teacher_name': t_name,
#                     'room_name': r_name,
#                     'is_shared': False,
#                     'is_fixed': is_fixed
#                 }
#
#     # 结果导向合班判定
#     for d in range(5):
#         for p in range(5):
#             sig_counts = {}
#             for cid in target_class_ids:
#                 item = scheds[cid][d][p]
#                 if item:
#                     sig = f"{item['course']}_{item['teacher_name']}_{item['room_name']}"
#                     sig_counts[sig] = sig_counts.get(sig, 0) + 1
#
#             for cid in target_class_ids:
#                 item = scheds[cid][d][p]
#                 if item:
#                     sig = f"{item['course']}_{item['teacher_name']}_{item['room_name']}"
#                     if sig_counts[sig] >= 2:
#                         item['is_shared'] = True
#
#     # ==========================================
#     # 绘图渲染区 (3x2 纵向布局)
#     # ==========================================
#     fig, axes = plt.subplots(3, 2, figsize=(16, 18))
#     axes = axes.flatten()
#
#     days_str = ['周一', '周二', '周三', '周四', '周五']
#     periods_str = ['第1-2节', '第3-4节', '第5-6节', '第7-8节', '第9-10节']
#
#     for idx, cid in enumerate(target_class_ids):
#         ax = axes[idx]
#         name = target_class_names[idx]
#         sched = scheds[cid]
#
#         ax.axis('off')
#         cell_text = [['' for _ in range(5)] for _ in range(5)]
#         cell_colors = [[COLOR_EMPTY_BG for _ in range(5)] for _ in range(5)]
#
#         for d in range(5):
#             for p in range(5):
#                 item = sched[d][p]
#                 if item:
#                     c_name_formatted = format_course_name(item['course'])
#                     prefix = "[固]" if item['is_fixed'] else ""
#                     cell_text[p][d] = f"{prefix}《{c_name_formatted}》\n{item['teacher_name']}\n{item['room_name']}"
#
#                     # 优先级：固定绿 > 合班红 > 普通蓝
#                     if item['is_fixed']:
#                         cell_colors[p][d] = COLOR_FIXED_BG
#                     elif item['is_shared']:
#                         cell_colors[p][d] = COLOR_SHARED_BG
#                     else:
#                         cell_colors[p][d] = COLOR_NORMAL_BG
#
#         table = ax.table(cellText=cell_text, rowLabels=periods_str, colLabels=days_str,
#                          cellColours=cell_colors, loc='center', cellLoc='center')
#
#         table.auto_set_font_size(False)
#         table.set_fontsize(10)
#         table.scale(1, 4.2)
#
#         for (row, col), cell in table.get_celld().items():
#             cell.set_edgecolor(COLOR_GRID_LINE)
#             cell.set_linewidth(0.8)
#             cell.set_text_props(color=COLOR_TEXT_MAIN)
#
#             if row == 0 or col == -1:
#                 cell.set_facecolor(COLOR_HEADER_BG)
#                 cell.set_text_props(weight='bold', fontsize=11)
#             elif col >= 0 and row > 0 and sched[col][row - 1]:
#                 item = sched[col][row - 1]
#                 if item['is_shared'] or item['is_fixed']:
#                     cell.set_text_props(weight='bold', color='#17202A')
#             elif col >= 0 and row > 0 and not sched[col][row - 1]:
#                 cell.set_edgecolor('#E5E7E9')
#
#         ax.set_title(f"行政班：{name}", fontsize=14, weight='bold', color=COLOR_TEXT_MAIN, pad=18)
#
#     # 全局标题与图例
#     fig.suptitle('合班与固定课程时空调度全景图',
#                  fontsize=22, weight='bold', color='#1C2833', y=0.98)
#
#     legend_elements = [
#         mpatches.Patch(facecolor=COLOR_FIXED_BG, edgecolor='#AAB7B8', label='固定预排课程 (硬约束优先)'),
#         mpatches.Patch(facecolor=COLOR_SHARED_BG, edgecolor='#AAB7B8', label='合班共享课程 (物理时空同步)'),
#         mpatches.Patch(facecolor=COLOR_NORMAL_BG, edgecolor='#AAB7B8', label='普通独立课程 (自主寻优)')
#     ]
#     fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.94),
#                ncol=3, fontsize=12, frameon=False, columnspacing=3.0)
#
#     # 调整纵向间距 (hspace=0.35 让上下图拉开距离)
#     plt.subplots_adjust(wspace=0.15, hspace=0.35)
#     plt.tight_layout(rect=[0, 0, 1, 0.92])
#
#     # 明确路径保存
#     png_path = os.path.join(output_dir, 'Fig1_Result_Oriented_6Classes_Final.png')
#     svg_path = os.path.join(output_dir, 'Fig1_Result_Oriented_6Classes_Final.svg')
#     plt.savefig(png_path, dpi=600, format='png')
#     plt.savefig(svg_path, format='svg')
#     plt.close()
#
#     print(f"✨ 图1 (6班级终极版) 绘制完成！\n📁 已保存至: {output_dir}")
#
#
# def main():
#     path_ablation = "results/<experiment>/L1_Large/best_schedules.json"
#     output_dir = "paper_plots"
#     os.makedirs(output_dir, exist_ok=True)
#
#     with open(path_ablation, 'r', encoding='utf-8') as f:
#         schedules = json.load(f)
#
#     try:
#         loader = DataLoader('L1_Large', DB_CONFIG)
#         data = loader.load()
#
#         algo_key = next((k for k in schedules if "TrueCDM" in k), None)
#         if algo_key:
#             plot_6_classes_schedule(schedules[algo_key]['schedule'], data, output_dir)
#         else:
#             print("⚠️ 未找到包含 TrueCDM 的算法数据。")
#     except Exception as e:
#         print(f"⚠️ 绘图失败: {e}")
#
#
# if __name__ == "__main__":
#     main()

import argparse
import os
import sys
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ==========================================
# 【核心修复 1】：解决 ModuleNotFoundError: No module named 'data'
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# 确保环境变量加入后再导入您的自定义模块
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.data_loader import DataLoader

# ==========================================
# 学术图表全局设置 (SCI 顶刊级排版规范)
# ==========================================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['savefig.dpi'] = 600

DB_CONFIG = {
    'host': 'localhost',
    'port': 3306,
    'user': 'root',
    'password': os.getenv('DB_PASSWORD', ''),
    'database': 'test3'
}

# --- 核心配色方案 (绿色最高优先级) ---
COLOR_FIXED_BG = '#D5F5E3'  # 浅青绿：固定课程 (优先级1)
COLOR_SHARED_BG = '#F5B7B1'  # 浅砖红：普通合班共享 (优先级2)
COLOR_NORMAL_BG = '#EBF5FB'  # 浅灰蓝：普通独立课程 (优先级3)
COLOR_EMPTY_BG = '#FFFFFF'  # 纯白：空白时段
COLOR_HEADER_BG = '#F2F4F4'  # 浅灰：表头
COLOR_GRID_LINE = '#D5D8DC'  # 浅灰：网格线
COLOR_TEXT_MAIN = '#2C3E50'  # 深炭黑：主文字


def format_course_name(name):
    """智能处理超长课程名，缓解横向拥挤"""
    if not name: return ""
    name = str(name).strip()
    if len(name) > 8:
        if '\n' not in name and len(name) >= 10:
            mid = len(name) // 2
            name = name[:mid] + '\n' + name[mid:]
    return name


def plot_6_classes_schedule(genes, data, output_dir):
    print("\n📊 正在执行【指定6班级 + 绿色固定优先 + 3x2布局】排版...")

    # 1. 严格锁定指定的 6 个目标
    target_names_hints = ["机械23-2", "机械23-4", "电气21-1", "电气21-2", "2022110201", "2022110202"]
    target_class_ids = []
    target_class_names = []

    # 强制精准匹配逻辑
    for hint in target_names_hints:
        found = False
        for cid, cinfo in data.classes.items():
            # 将 cid 和 hint 都转为字符串对比，防止类型不匹配
            if (hint in cinfo.class_name or hint == str(cid)) and str(cid) not in [str(x) for x in target_class_ids]:
                target_class_ids.append(cid)
                target_class_names.append(cinfo.class_name)
                found = True
                break

        # 强制兜底：没查到名字，强制把 ID 塞进去
        if not found:
            print(f"⚠️ 提示：未在字典查到 '{hint}' 的名字，已强制按ID提取！")
            target_class_ids.append(hint)  # 直接存入原始 hint
            target_class_names.append(f"班级 {hint}")

    print(f"🎯 最终入列展示的 6 个班级（严格锁定）: {target_class_names}")

    scheds = {str(cid): [[None for _ in range(5)] for _ in range(5)] for cid in target_class_ids}

    # 统一将目标 ID 转为字符串列表，方便后续精确比对
    target_class_ids_str = [str(x) for x in target_class_ids]

    print("\n🔍 --- 开始提取这 6 个班的排课数据 ---")
    extracted_count = 0

    for task_id_str, gene in genes.items():
        task_id = int(task_id_str)
        if task_id not in data.tasks: continue
        task = data.tasks[task_id]

        # 【核心修复 2】：强制将数据库里的 task.class_id 转为字符串后再 in 判断！
        if str(task.class_id) in target_class_ids_str:

            # 【核心修复 3】：加入控制台打印，让您一眼看出到底读出了几节课！
            extracted_count += 1
            print(f"✅ 提取到第 {extracted_count} 节课 -> 班级:{task.class_id} | 课程:《{task.course_class_name}》")

            slot = gene >> 16
            room_id = gene & 0xFFFF
            day, period = slot // 5, slot % 5

            # 只有在正常的 5x5 时段内的才会被画出
            if 0 <= day < 5 and 0 <= period < 5:
                t_id = str(task.teacher_id).strip()
                t_name = data.teachers[t_id].teacher_name if t_id in data.teachers else f"{t_id}"

                r_key = str(room_id).zfill(6) if str(room_id) not in data.classrooms else str(room_id)
                r_name = data.classrooms[r_key].classroom_name if r_key in data.classrooms else f"{room_id}"

                is_fixed = getattr(task, 'is_fixed', False)

                scheds[str(task.class_id)][day][period] = {
                    'course': task.course_class_name,
                    'teacher_name': t_name,
                    'room_name': r_name,
                    'is_shared': False,
                    'is_fixed': is_fixed
                }

    print(f"🔍 --- 数据提取完毕。总计为您指定的 6 个班级提取到了 {extracted_count} 个授课任务 ---\n")

    # 结果导向合班判定
    for d in range(5):
        for p in range(5):
            sig_counts = {}
            for cid_str in target_class_ids_str:
                item = scheds[cid_str][d][p]
                if item:
                    sig = f"{item['course']}_{item['teacher_name']}_{item['room_name']}"
                    sig_counts[sig] = sig_counts.get(sig, 0) + 1

            for cid_str in target_class_ids_str:
                item = scheds[cid_str][d][p]
                if item:
                    sig = f"{item['course']}_{item['teacher_name']}_{item['room_name']}"
                    if sig_counts[sig] >= 2:
                        item['is_shared'] = True

    # ==========================================
    # 绘图渲染区 (3行 x 2列 布局)
    # ==========================================
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    axes = axes.flatten()

    days_str = ['周一', '周二', '周三', '周四', '周五']
    periods_str = ['第1-2节', '第3-4节', '第5-6节', '第7-8节', '第9-10节']

    for idx, cid_str in enumerate(target_class_ids_str):
        ax = axes[idx]
        name = target_class_names[idx]
        sched = scheds[cid_str]

        ax.axis('off')
        cell_text = [['' for _ in range(5)] for _ in range(5)]
        cell_colors = [[COLOR_EMPTY_BG for _ in range(5)] for _ in range(5)]

        for d in range(5):
            for p in range(5):
                item = sched[d][p]
                if item:
                    c_name_formatted = format_course_name(item['course'])
                    prefix = "[固]" if item['is_fixed'] else ""
                    cell_text[p][d] = f"{prefix}《{c_name_formatted}》\n{item['teacher_name']}\n{item['room_name']}"

                    if item['is_fixed']:
                        cell_colors[p][d] = COLOR_FIXED_BG
                    elif item['is_shared']:
                        cell_colors[p][d] = COLOR_SHARED_BG
                    else:
                        cell_colors[p][d] = COLOR_NORMAL_BG

        table = ax.table(cellText=cell_text, rowLabels=periods_str, colLabels=days_str,
                         cellColours=cell_colors, loc='center', cellLoc='center')

        table.auto_set_font_size(False)
        table.set_fontsize(10.5)
        table.scale(1, 4.2)

        for (row, col), cell in table.get_celld().items():
            cell.set_edgecolor(COLOR_GRID_LINE)
            cell.set_linewidth(0.8)
            cell.set_text_props(color=COLOR_TEXT_MAIN)

            if row == 0 or col == -1:
                cell.set_facecolor(COLOR_HEADER_BG)
                cell.set_text_props(weight='bold', fontsize=12)
            elif col >= 0 and row > 0 and sched[col][row - 1]:
                item = sched[col][row - 1]
                if item['is_shared'] or item['is_fixed']:
                    cell.set_text_props(weight='bold', color='#17202A')
            elif col >= 0 and row > 0 and not sched[col][row - 1]:
                cell.set_edgecolor('#E5E7E9')

        ax.set_title(f"行政班：{name}", fontsize=15, weight='bold', color=COLOR_TEXT_MAIN, pad=18)

    # 全局大标题
    fig.suptitle('图3-xx 本文算法跨专业合班与固定课程时空调度全景图',
                 fontsize=22, weight='bold', color='#1C2833', y=0.98)

    # 图例 (Legend)
    legend_elements = [
        mpatches.Patch(facecolor=COLOR_FIXED_BG, edgecolor='#AAB7B8', linewidth=1, label='固定预排课程 (硬约束优先)'),
        mpatches.Patch(facecolor=COLOR_SHARED_BG, edgecolor='#AAB7B8', linewidth=1, label='合班共享课程 (物理时空同步)'),
        mpatches.Patch(facecolor=COLOR_NORMAL_BG, edgecolor='#AAB7B8', linewidth=1, label='普通独立课程 (自主寻优)')
    ]
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.94),
               ncol=3, fontsize=13, frameon=False, columnspacing=3.0)

    plt.subplots_adjust(wspace=0.15, hspace=0.35)
    plt.tight_layout(rect=[0, 0, 1, 0.90])

    png_path = os.path.join(output_dir, 'Fig1_Result_Oriented_6Classes_Final.png')
    svg_path = os.path.join(output_dir, 'Fig1_Result_Oriented_6Classes_Final.svg')

    plt.savefig(png_path, dpi=600, format='png')
    plt.savefig(svg_path, format='svg')
    plt.close()

    print(f"✨ 图1 (含图例 3x2 布局版) 绘制完美收工！")


def main():
    parser = argparse.ArgumentParser(description="绘制六个代表性行政班课表")
    parser.add_argument("best_schedules", help="best_schedules.json 路径")
    parser.add_argument("--output-dir", default="paper_plots", help="图片输出目录")
    args = parser.parse_args()
    path_ablation = args.best_schedules
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(path_ablation):
        print(f"❌ 找不到 JSON 文件: {path_ablation}")
        return

    with open(path_ablation, 'r', encoding='utf-8') as f:
        schedules = json.load(f)

    try:
        print("🔗 正在连接数据库拉取最新实例数据...")
        loader = DataLoader('L1_Large', DB_CONFIG)
        data = loader.load()

        algo_key = next((k for k in schedules if "TrueCDM" in k), None)
        if algo_key is None and "Ours" in schedules:
            algo_key = "Ours"
        if algo_key:
            plot_6_classes_schedule(schedules[algo_key]['schedule'], data, output_dir)
            print(f"\n🎉 完美！双格式图表已成功导出至目录:\n📁 {output_dir}")
        else:
            print("⚠️ 未找到包含 TrueCDM 的算法数据。")
    except Exception as e:
        print(f"⚠️ 绘图失败: {e}")


if __name__ == "__main__":
    main()
