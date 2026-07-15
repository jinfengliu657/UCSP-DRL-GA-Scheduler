import argparse
import os
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.data_loader import DataLoader

# ==========================================
# 学术图表全局设置 (适配中文与 SCI 规范)
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


# ---------------------------------------------------------
# 图2：周内负载均衡对比 (严格按【授课单元】计数)
# ---------------------------------------------------------
def plot_unit_based_daily_distribution(schedules, data, output_dir):
    print("📊 正在生成 [图2] 基于授课单元的负载均衡对比图...")

    algo_proposed = next((a for a in schedules if "TrueCDM" in a), None)
    algo_baseline = next((a for a in schedules if a == "DRL-GA-TS"), None)

    if not algo_proposed or not algo_baseline:
        print("⚠️ 未同时找到本文算法与基线算法，跳过分布图。")
        return

    target_algos = [algo_baseline, algo_proposed]
    daily_units = {algo: [set() for _ in range(5)] for algo in target_algos}

    for algo in target_algos:
        genes = schedules[algo]['schedule']
        for task_id_str, gene in genes.items():
            task_id = int(task_id_str)
            if task_id not in data.tasks: continue

            task = data.tasks[task_id]
            slot = gene >> 16
            day = slot // 5

            if 0 <= day < 5:
                # 【核心】：按单元去重
                unit_id = f"G_{task.course_class_id}" if task.course_class_id and task.course_class_id in data.combined_classes else f"T_{task_id}"
                daily_units[algo][day].add(unit_id)

    counts_baseline = [len(daily_units[algo_baseline][d]) for d in range(5)]
    counts_proposed = [len(daily_units[algo_proposed][d]) for d in range(5)]

    x = np.arange(5)
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.bar(x - width / 2, counts_baseline, width, label='基线算法 (DRL-GA-TS)', color='#1f77b4', alpha=0.85)
    ax.bar(x + width / 2, counts_proposed, width, label='本文算法 (TrueCDM)', color='#d62728', edgecolor='black',
           linewidth=1.5, alpha=0.95)

    ax.set_xticks(x)
    ax.set_xticklabels(['周一', '周二', '周三', '周四', '周五'], fontsize=13)

    # 【修改落实】：去掉英文，纯中文纵轴
    ax.set_ylabel('单日安排授课单元总数', fontsize=13, weight='bold')
    # 【修改落实】：精简图题
    ax.set_title('图3-xx 本文算法与基线算法周内授课单元分布对比', fontsize=15, pad=15, weight='bold')

    # 将图例移入图内合适位置
    ax.legend(fontsize='12', loc='upper right')
    ax.grid(axis='y', linestyle='--', alpha=0.6)

    # 设置Y轴略微起跳的底部余量，防止柱子显得“飘”
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Fig2_Unit_Daily_Distribution.png'))
    plt.close()
    print("✨ 图2 绘制完成！")


# ---------------------------------------------------------
# 图3：热力图 (固定上界，强化稳定可解释性)
# ---------------------------------------------------------
def plot_true_utilization_heatmap(genes, data, output_dir):
    print("📊 正在生成 [图3] 时空资源利用热力图...")
    slot_room_stu = defaultdict(int)
    for task_id_str, gene in genes.items():
        try:
            task_id = int(task_id_str)
        except ValueError:
            continue
        if task_id not in data.tasks: continue

        task = data.tasks[task_id]
        slot = gene >> 16
        room_id = gene & 0xFFFF
        if 0 <= slot < 25:
            slot_room_stu[(slot, room_id)] += task.student_count

    util_matrix_sum = np.zeros((5, 5))
    util_matrix_count = np.zeros((5, 5))

    for (slot, rid), stu in slot_room_stu.items():
        day = slot // 5
        period = slot % 5
        r_key = str(rid).zfill(6) if str(rid) not in data.classrooms else str(rid)
        if r_key in data.classrooms:
            cap = data.classrooms[r_key].capacity
            if cap > 0:
                util_matrix_sum[day, period] += (stu / cap) * 100
                util_matrix_count[day, period] += 1

    with np.errstate(invalid='ignore'):
        avg_util_matrix = np.divide(util_matrix_sum, util_matrix_count)
        avg_util_matrix = np.nan_to_num(avg_util_matrix)

    fig, ax = plt.subplots(figsize=(8, 6))

    # 【修改落实】：增加 vmax=70，使得 64.4 显得醒目但不会吞噬 30-40 的梯度差异
    sns.heatmap(avg_util_matrix.T, annot=True, fmt=".1f", cmap="YlOrRd",
                vmin=0, vmax=70,  # 锚定色标范围，确保论文图表稳定性
                xticklabels=['周一', '周二', '周三', '周四', '周五'],
                yticklabels=['第1-2节', '第3-4节', '第5-6节', '第7-8节', '第9-10节'],
                cbar_kws={'label': '座位平均利用率 (%)'}, linewidths=1, linecolor='white',
                annot_kws={"size": 11})

    # 【修改落实】：精简图题，去掉“热力图”三字
    ax.set_title('图3-xx L1实例下本文算法教室座位利用率时空分布图', fontsize=15, pad=15, weight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Fig3_Utilization_Heatmap.png'))
    plt.close()
    print("✨ 图3 绘制完成！")


def main():
    parser = argparse.ArgumentParser(description="绘制授课单元日分布和教室利用率图")
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

        algo_proposed = next((a for a in schedules.keys() if "TrueCDM" in a), None)
        if algo_proposed is None and "Ours" in schedules:
            algo_proposed = "Ours"

        if algo_proposed:
            plot_unit_based_daily_distribution(schedules, data, output_dir)
            plot_true_utilization_heatmap(schedules[algo_proposed]['schedule'], data, output_dir)
            print(f"\n🎉 完美！图2 与 图3 已成功导出至:\n📁 {output_dir}")
        else:
            print("⚠️ 未找到包含 TrueCDM 的算法数据。")

    except Exception as e:
        print(f"⚠️ 绘图失败: {e}")


if __name__ == "__main__":
    main()
