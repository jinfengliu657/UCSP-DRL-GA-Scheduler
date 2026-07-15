import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'


def plot_time_budget_convergence_5min(history_path, summary_path, target_instances):
    print("\n📊 正在生成降维打击：【5分钟工程落地生死线收敛图】...")
    if not os.path.exists(history_path) or not os.path.exists(summary_path):
        print("❌ 找不到数据文件，请检查路径！")
        return

    df_hist = pd.read_csv(history_path)
    df_summ = pd.read_csv(summary_path)

    target_algos = [
        ('IFTS', '#1F77B4', '-.'),
        ('Zhu-Replica', '#FF7F0E', ':'),
        ('Heuristic-TS', '#9467BD', '-.'),
        ('Heuristic-GA', '#8C564B', '--'),
        ('GA-DRL', '#2CA02C', '--'),
        ('DL-GA-TS', '#D62728', '-')
    ]

    for instance in target_instances:
        inst_hist = df_hist[df_hist['Instance'] == instance]
        inst_summ = df_summ[df_summ['Instance'] == instance]
        if inst_hist.empty: continue

        fig, ax = plt.subplots(figsize=(12, 7))

        max_plot_time = 0

        for algo, color, ls in target_algos:
            algo_hist = inst_hist[inst_hist['Algorithm'] == algo]
            algo_summ = inst_summ[inst_summ['Algorithm'] == algo]
            if algo_hist.empty or algo_summ.empty: continue

            all_time_curves = []
            all_fitness_curves = []

            for run_id in algo_hist['Run'].unique():
                run_hist = algo_hist[algo_hist['Run'] == run_id].sort_values('Gen')
                run_time = algo_summ[algo_summ['Run'] == run_id]['Time_s'].values[0]

                max_gen = run_hist['Gen'].max()
                if max_gen == 0: continue

                # 估算耗时
                run_hist['Cumulative_Time'] = (run_hist['Gen'] / max_gen) * run_time
                all_time_curves.append(run_hist['Cumulative_Time'].values)
                all_fitness_curves.append(run_hist['Best_Fitness'].values)
                max_plot_time = max(max_plot_time, run_time)

            # 插值对齐
            # 为了看清 300 秒前后的对比，我们将 X 轴插值到 600 秒或该算法的最大耗时
            common_time_grid = np.linspace(0, min(max_plot_time, 600), 1000)
            interp_fitness = []

            for t_curve, f_curve in zip(all_time_curves, all_fitness_curves):
                # 保证单调递增，如果某个算法早就停机了，后面的时间就平推它的最终分数
                interp_f = np.interp(common_time_grid, t_curve, f_curve)
                interp_fitness.append(interp_f)

            mean_fitness = np.mean(interp_fitness, axis=0)
            lw = 3 if algo == 'DL-GA-TS' else 2

            ax.plot(common_time_grid, mean_fitness, label=algo, color=color, linestyle=ls, linewidth=lw)

        # 💥 灵魂画法：在 X 轴 300 秒（5分钟）处画生死线！
        cutoff_time = 300.0
        ax.axvline(x=cutoff_time, color='black', linestyle='-.', linewidth=2, zorder=5)

        # 标注文字
        ax.text(cutoff_time + 10, ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.15,
                'Practical Time Limit\n(5 Minutes)',
                color='black', fontsize=13, fontweight='bold', ha='left',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

        # 图表美化
        ax.set_title(f'Fitness Evolution Over Actual Computation Time ({instance})', fontsize=18, pad=15,
                     fontweight='bold')
        ax.set_xlabel('Cumulative Computation Time (Seconds)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Average Best Fitness', fontsize=14, fontweight='bold')

        # 限制 X 轴看前 600 秒，让 300 秒的线刚好在中间
        ax.set_xlim(0, 600)

        ax.tick_params(axis='both', labelsize=12)
        ax.grid(True, linestyle=':', alpha=0.7)
        ax.legend(fontsize=12, loc='lower right', framealpha=0.9, edgecolor='gray')

        plt.tight_layout()

        save_dir = 'paper_plots'
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'Time_Evolution_5Min_{instance}.png')
        plt.savefig(save_path, bbox_inches='tight')
        print(f"✅ [{instance}] 的【5分钟生死线收敛图】生成完毕！已保存至: {os.path.abspath(save_path)}")
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="绘制基于实际计算时间的适应度演化曲线")
    parser.add_argument("history", help="实验 history.csv 路径")
    parser.add_argument("summary", help="实验 summary.csv 路径")
    parser.add_argument("--instances", nargs="+", default=["L1_Large"], help="实例名称列表")
    args = parser.parse_args()
    plot_time_budget_convergence_5min(args.history, args.summary, args.instances)
