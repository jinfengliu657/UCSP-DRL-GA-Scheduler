import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import os
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. 全局学术规范排版
# ==========================================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.alpha'] = 0.5
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'


def plot_final_true_convergence(instance_name, history_path, summary_path):
    print(f"\n📊 正在生成 [{instance_name}] 的终极版真实收敛图...")
    if not os.path.exists(history_path) or not os.path.exists(summary_path):
        print("❌ 找不到数据文件，请检查底部的路径！")
        return

    df_hist = pd.read_csv(history_path)
    df_summ = pd.read_csv(summary_path)

    df_hist = df_hist[df_hist['Instance'] == instance_name]
    df_summ = df_summ[df_summ['Instance'] == instance_name]

    # 定义目标算法（颜色、线型、标记点）
    target_algos = [
        ('IFTS', {'label': 'IFTS', 'color': '#1F77B4', 'ls': '-.', 'marker': 's'}),
        ('Zhu-Replica', {'label': 'Zhu-Replica', 'color': '#FF7F0E', 'ls': ':', 'marker': 'X'}),
        ('Heuristic-TS', {'label': 'Heuristic-TS', 'color': '#9467BD', 'ls': '-.', 'marker': '^'}),
        ('Heuristic-GA', {'label': 'Heuristic-GA', 'color': '#8C564B', 'ls': '--', 'marker': 'v'}),
        ('GA-DRL', {'label': 'GA-DRL', 'color': '#2CA02C', 'ls': '--', 'marker': 'o'}),
        ('DL-GA-TS', {'label': 'DL-GA-TS (Ours)', 'color': '#D62728', 'ls': '-', 'marker': '*'})
    ]

    # 大后期需要局部放大的三剑客
    tail_algos = ['Heuristic-GA', 'GA-DRL', 'DL-GA-TS']

    fig, ax = plt.subplots(figsize=(14, 8))

    # ==========================================
    # 2. 局部放大图设置 (置于中间偏上方空余处)
    # ==========================================
    axins = ax.inset_axes([0.35, 0.55, 0.35, 0.35])
    axins.set_facecolor('white')  # 白底防透视
    axins.set_zorder(10)

    global_max_gen = 0
    zoom_y_finals = []

    # ==========================================
    # 3. 核心计算与绘制
    # ==========================================
    for algo, style in target_algos:
        algo_hist = df_hist[df_hist['Algorithm'] == algo]
        algo_summ = df_summ[df_summ['Algorithm'] == algo]
        if algo_hist.empty: continue

        # 💥 灵魂逻辑 1：获取该算法在10次运行中，真实存在的最大代数！作为X轴尽头
        algo_max_gen = int(algo_hist['Gen'].max())
        global_max_gen = max(global_max_gen, algo_max_gen)

        # 获取 summary 中的平均耗时，用于文本标注
        algo_avg_time = algo_summ['Time_s'].mean()

        # 💥 灵魂逻辑 2：底层计算补齐 (ffill)，确保每代都有10个数来算平均分
        aligned_runs = []
        for run_id in algo_hist['Run'].unique():
            run_data = algo_hist[algo_hist['Run'] == run_id].sort_values('Gen')
            run_series = run_data.set_index('Gen')['Best_Fitness']
            aligned_runs.append(run_series.reindex(range(algo_max_gen + 1)).ffill().values)

        if not aligned_runs: continue
        aligned_runs = np.array(aligned_runs)

        # 垂直求取 10 次的平滑均值线
        mean_curve = np.mean(aligned_runs, axis=0)
        x_gens = np.arange(algo_max_gen + 1)

        lw = 2.5 if algo == 'DL-GA-TS' else 2

        # --- 主图绘制 ---
        ax.plot(x_gens, mean_curve, label=style['label'], color=style['color'], linestyle=style['ls'], linewidth=lw)

        # 非大后期的算法，直接在主图标注终点
        if algo not in tail_algos:
            ax.scatter(x_gens[-1], mean_curve[-1], color=style['color'], s=150, marker=style['marker'],
                       edgecolor='white', zorder=5)
            # 标注终点得分和时间
            ax.text(x_gens[-1] + 15, mean_curve[-1], f"{mean_curve[-1]:.0f}\n({algo_avg_time:.1f}s)",
                    color=style['color'], fontsize=11, fontweight='bold', va='center', ha='left',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=0.5))

        # --- 局部放大图绘制 ---
        if algo in tail_algos:
            # 主图仅画点，不在主图写字防止重叠
            ax.scatter(x_gens[-1], mean_curve[-1], color=style['color'], s=100, marker=style['marker'],
                       edgecolor='white', zorder=5)

            # 画进放大图
            axins.plot(x_gens, mean_curve, color=style['color'], linestyle=style['ls'], linewidth=lw + 1, zorder=11)
            axins.scatter(x_gens[-1], mean_curve[-1], color=style['color'], s=250, marker=style['marker'], zorder=12,
                          edgecolor='white')

            # 智能错开文本，确保放大图内依然清爽
            y_offset = 15 if algo == 'DL-GA-TS' else (-15 if algo == 'Heuristic-GA' else 0)
            axins.text(x_gens[-1] + 25, mean_curve[-1] + y_offset,
                       f"Score: {mean_curve[-1]:.0f}\nTime: {algo_avg_time:.1f}s",
                       color=style['color'], fontsize=10, fontweight='bold', va='center', ha='left',
                       bbox=dict(facecolor='white', alpha=0.9, edgecolor='none', pad=0.5), zorder=13)

            zoom_y_finals.append(mean_curve[-1])

    # ==========================================
    # 4. 图表美化与布局
    # ==========================================
    ax.set_title(f'Convergence Process Comparison ({instance_name})', fontsize=18, pad=20, fontweight='bold')
    ax.set_xlabel('Number of Generations', fontsize=14, fontweight='bold')
    ax.set_ylabel('Average Best Fitness', fontsize=14, fontweight='bold')

    # 图例放于右上角
    ax.legend(fontsize=12, loc='upper right', framealpha=0.9, edgecolor='gray')
    ax.set_xlim(0, global_max_gen * 1.15)

    # 💥 局部放大框视角锁定：截取 1000 代以后的大后期
    zoom_x_min = 1000
    zoom_x_max = global_max_gen * 1.35
    axins.set_xlim(zoom_x_min, zoom_x_max)

    # 紧凑贴合 Y 轴，彻底放大这 3 个算法的分数差距
    if zoom_y_finals:
        y_min_limit = min(zoom_y_finals) - 50
        y_max_limit = max(zoom_y_finals) + 50
        axins.set_ylim(y_min_limit, y_max_limit)

    axins.set_title('Zoom-in: End-stage Details', fontsize=11, color='dimgray', fontweight='bold')
    axins.grid(True, linestyle=':', alpha=0.6, zorder=10)

    # 虚线连接主图与放大图
    mark_inset(ax, axins, loc1=3, loc2=4, fc="none", ec="0.5", alpha=0.5, linestyle='--', zorder=9)

    plt.tight_layout()

    # 保存输出
    save_dir = 'paper_plots'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'Final_Convergence_{instance_name}.png')
    plt.savefig(save_path, bbox_inches='tight')
    print(f"✅ 图表生成完毕！完全忠实于历史寿命，放大细节清晰无误！已保存至: {os.path.abspath(save_path)}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="绘制算法收敛曲线及尾部放大图")
    parser.add_argument("history", help="实验 history.csv 路径")
    parser.add_argument("summary", help="实验 summary.csv 路径")
    parser.add_argument("--instance", default="L1_Large", help="实例名称")
    args = parser.parse_args()
    plot_final_true_convergence(args.instance, history_path=args.history, summary_path=args.summary)


# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.axes_grid1.inset_locator import mark_inset
# import os
# import warnings
#
# warnings.filterwarnings('ignore')
#
# # ==========================================
# # 1. 学术规范格式
# # ==========================================
# plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
# plt.rcParams['axes.unicode_minus'] = False
# plt.rcParams['figure.dpi'] = 300
# plt.rcParams['axes.grid'] = True
# plt.rcParams['grid.linestyle'] = '--'
# plt.rcParams['grid.alpha'] = 0.5
#
#
# def plot_true_convergence(instance_name, history_path, summary_path):
#     print(f"\n📊 正在拨乱反正，绘制 [{instance_name}] 的最真实收敛图...")
#
#     df_hist = pd.read_csv(history_path)
#     df_summ = pd.read_csv(summary_path)
#
#     df_hist = df_hist[df_hist['Instance'] == instance_name]
#     df_summ = df_summ[df_summ['Instance'] == instance_name]
#
#     target_algos = [
#         ('IFTS', {'label': 'IFTS', 'color': '#1F77B4', 'ls': '-.', 'marker': 's'}),
#         ('Zhu-Replica', {'label': 'Zhu-Replica', 'color': '#FF7F0E', 'ls': ':', 'marker': 'X'}),
#         ('Heuristic-TS', {'label': 'Heuristic-TS', 'color': '#9467BD', 'ls': '-.', 'marker': '^'}),
#         ('Heuristic-GA', {'label': 'Heuristic-GA', 'color': '#8C564B', 'ls': '--', 'marker': 'v'}),
#         ('GA-DRL', {'label': 'GA-DRL', 'color': '#2CA02C', 'ls': '--', 'marker': 'o'}),
#         ('DL-GA-TS', {'label': 'DL-GA-TS (Ours)', 'color': '#D62728', 'ls': '-', 'marker': '*'})
#     ]
#
#     tail_algos = ['Heuristic-GA', 'GA-DRL', 'DL-GA-TS']
#     fig, ax = plt.subplots(figsize=(14, 8))
#
#     # 局部放大图：中间偏上
#     axins = ax.inset_axes([0.35, 0.55, 0.35, 0.35])
#     axins.set_facecolor('white')
#     axins.set_zorder(10)
#
#     global_max_gen = 0
#     zoom_y_finals = []
#
#     for algo, style in target_algos:
#         algo_hist = df_hist[df_hist['Algorithm'] == algo]
#         algo_summ = df_summ[df_summ['Algorithm'] == algo]
#         if algo_hist.empty: continue
#
#         # 💥 必须以 history 中该算法真实的最长代数作为 X 轴终点！绝不截断！
#         algo_max_gen = int(algo_hist['Gen'].max())
#         global_max_gen = max(global_max_gen, algo_max_gen)
#
#         # 获取官方统计的平均运行时间
#         algo_avg_time = algo_summ['Time_s'].mean()
#
#         # 对短于最大代数的运行进行水平补齐（这是算平均值的唯一正确数学方法）
#         aligned_runs = []
#         for run_id in algo_hist['Run'].unique():
#             run_data = algo_hist[algo_hist['Run'] == run_id].sort_values('Gen')
#             run_series = run_data.set_index('Gen')['Best_Fitness']
#             aligned_runs.append(run_series.reindex(range(algo_max_gen + 1)).ffill().values)
#
#         if not aligned_runs: continue
#         aligned_runs = np.array(aligned_runs)
#
#         # 求 10 次运行的垂直平均值
#         mean_curve = np.mean(aligned_runs, axis=0)
#         x_gens = np.arange(algo_max_gen + 1)
#
#         lw = 2.5 if algo == 'DL-GA-TS' else 2
#
#         # --- 主图绘制 ---
#         ax.plot(x_gens, mean_curve, label=style['label'], color=style['color'], linestyle=style['ls'], linewidth=lw)
#
#         # 针对不需要放大的算法，直接在主图打点写字
#         if algo not in tail_algos:
#             ax.scatter(x_gens[-1], mean_curve[-1], color=style['color'], s=150, marker=style['marker'],
#                        edgecolor='white', zorder=5)
#             # 标注分数和平均耗时
#             ax.text(x_gens[-1] + 15, mean_curve[-1], f"{mean_curve[-1]:.0f}\n({algo_avg_time:.1f}s)",
#                     color=style['color'], fontsize=11, fontweight='bold', va='center', ha='left',
#                     bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=0.5))
#
#         # --- 局部放大图绘制 (专供大后期三剑客) ---
#         if algo in tail_algos:
#             # 主图仅打点防重叠
#             ax.scatter(x_gens[-1], mean_curve[-1], color=style['color'], s=100, marker=style['marker'],
#                        edgecolor='white', zorder=5)
#
#             # 画进放大图
#             axins.plot(x_gens, mean_curve, color=style['color'], linestyle=style['ls'], linewidth=lw + 1, zorder=11)
#             axins.scatter(x_gens[-1], mean_curve[-1], color=style['color'], s=250, marker=style['marker'], zorder=12,
#                           edgecolor='white')
#
#             # 放大图中文本错开
#             y_offset = 15 if algo == 'DL-GA-TS' else (-15 if algo == 'Heuristic-GA' else 0)
#             axins.text(x_gens[-1] + 25, mean_curve[-1] + y_offset,
#                        f"Score: {mean_curve[-1]:.0f}\nTime: {algo_avg_time:.1f}s",
#                        color=style['color'], fontsize=10, fontweight='bold', va='center', ha='left',
#                        bbox=dict(facecolor='white', alpha=0.9, edgecolor='none', pad=0.5), zorder=13)
#
#             zoom_y_finals.append(mean_curve[-1])
#
#     # ==========================================
#     # 图表布局收尾
#     # ==========================================
#     ax.set_title(f'Convergence Process Comparison ({instance_name})', fontsize=18, pad=20, fontweight='bold')
#     ax.set_xlabel('Number of Generations', fontsize=14, fontweight='bold')
#     ax.set_ylabel('Average Best Fitness', fontsize=14, fontweight='bold')
#
#     # 图例放右上角
#     ax.legend(fontsize=12, loc='upper right', framealpha=0.9, edgecolor='gray')
#     ax.set_xlim(0, global_max_gen * 1.15)
#
#     # 局部放大框锁定：1000代以后的尾部激战
#     zoom_x_min = 1000
#     zoom_x_max = global_max_gen * 1.35
#     axins.set_xlim(zoom_x_min, zoom_x_max)
#
#     # 严丝合缝截取 Y 轴放大差异
#     if zoom_y_finals:
#         y_min_limit = min(zoom_y_finals) - 50
#         y_max_limit = max(zoom_y_finals) + 50
#         axins.set_ylim(y_min_limit, y_max_limit)
#
#     axins.set_title('Zoom-in: End-stage Details', fontsize=11, color='dimgray')
#     axins.grid(True, linestyle=':', alpha=0.6, zorder=10)
#     mark_inset(ax, axins, loc1=3, loc2=4, fc="none", ec="0.5", alpha=0.5, linestyle='--', zorder=9)
#
#     plt.tight_layout()
#     os.makedirs('paper_plots', exist_ok=True)
#     save_path = os.path.join('paper_plots', f'Final_Convergence_TrueData_{instance_name}.png')
#     plt.savefig(save_path, bbox_inches='tight')
#     print(f"✅ 图表已拨乱反正！完全忠实于历史数据的最长代数！保存至: {os.path.abspath(save_path)}")
#     plt.show()
#
#
# if __name__ == "__main__":
#     MY_HISTORY_PATH = 'results/<experiment>/history.csv'
#     MY_SUMMARY_PATH = 'results/<experiment>/summary.csv'
#     plot_true_convergence('L1_Large', history_path=MY_HISTORY_PATH, summary_path=MY_SUMMARY_PATH)
