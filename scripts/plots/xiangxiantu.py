import argparse
import pandas as pd
import matplotlib.pyplot as plt
import os
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. 全局学术规范排版 (Times New Roman)
# ==========================================
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300


def plot_zhu_style_boxplot(summary_path, target_instances):
    print("\n📊 正在生成像素级复刻 Zhu 等人的一区极简学术风箱线图...")
    if not os.path.exists(summary_path):
        print("❌ 找不到 summary.csv，请检查路径！")
        return

    df = pd.read_csv(summary_path)

    # 定义算法与顺序
    algo_order = ['IFTS', 'Zhu-Replica', 'Heuristic-TS', 'Heuristic-GA', 'GA-DRL', 'DL-GA-TS']
    df = df[df['Algorithm'].isin(algo_order)]

    for instance in target_instances:
        df_inst = df[df['Instance'] == instance]
        if df_inst.empty:
            continue

        # 提取各个算法的 Best_Fitness 数据为列表
        data_to_plot = []
        labels = []
        for algo in algo_order:
            algo_data = df_inst[df_inst['Algorithm'] == algo]['Best_Fitness'].values
            if len(algo_data) > 0:
                data_to_plot.append(algo_data)
                labels.append(algo)

        # 创建画布 (比例与 Zhu 原文保持一致)
        fig, ax = plt.subplots(figsize=(8, 5))

        # 💥 核心：Zhu 风格的参数设置 (极简黑白线条，橙色中位数，空心圆圈异常值)
        boxprops = dict(linestyle='-', linewidth=1.0, color='black')  # 箱体边框
        medianprops = dict(linestyle='-', linewidth=1.2, color='#FFA500')  # 橙色中位数线
        whiskerprops = dict(linestyle='-', linewidth=1.0, color='black')  # 须线
        capprops = dict(linestyle='-', linewidth=1.0, color='black')  # 顶部/底部横线
        flierprops = dict(marker='o', markerfacecolor='none', markeredgecolor='black', markersize=5,
                          alpha=0.8)  # 空心圆异常值

        # 使用 matplotlib 原生箱线图，关闭 patch_artist 以实现无填充效果
        ax.boxplot(data_to_plot, labels=labels,
                   boxprops=boxprops, medianprops=medianprops,
                   whiskerprops=whiskerprops, capprops=capprops,
                   flierprops=flierprops, patch_artist=False,
                   widths=0.4)  # 稍微调窄箱子显得更精致

        # 标题与坐标轴 (完全模仿 Zhu 的样式：顶部加个 Instance X 的小标题，左侧写 Fitness)
        ax.set_title(f'Instance {instance}', fontsize=14, pad=10)
        ax.set_xlabel('Algorithm', fontsize=12)
        ax.set_ylabel('Fitness', fontsize=12)

        # 刻度字体设置
        ax.tick_params(axis='x', labelsize=10)
        ax.tick_params(axis='y', labelsize=10)

        # 原文中没有网格线，为了极致的干净，我们也不加网格线
        ax.grid(False)

        # 确保四周的黑色边框线都是可见的 (标准学术图)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.8)
            spine.set_color('black')

        plt.tight_layout()

        # 保存图片
        save_dir = 'paper_plots'
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'Zhu_Style_Boxplot_{instance}.png')
        plt.savefig(save_path, bbox_inches='tight')
        print(f"✅ [{instance}] 的极简学术箱线图生成完毕！已保存至: {os.path.abspath(save_path)}")
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="绘制算法适应度箱线图")
    parser.add_argument("summary", help="实验 summary.csv 路径")
    parser.add_argument("--instances", nargs="+", default=["S1", "M1", "L1_Large"], help="实例名称列表")
    args = parser.parse_args()
    plot_zhu_style_boxplot(args.summary, args.instances)
