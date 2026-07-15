"""
通用可视化工具 - 用于单次运行时的快速展示 (Fixed for 25 time slots)
文件名: utils/visualization.py
"""
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from pathlib import Path
from typing import Dict, List

# 设置中文字体 (兼容不同操作系统)
font_list = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'PingFang SC', 'Heiti TC']
plt.rcParams['font.sans-serif'] = font_list
plt.rcParams['axes.unicode_minus'] = False


class EvolutionVisualizer:
    """
    进化过程可视化 (调试用)
    """

    def __init__(self, output_dir: str = "results/debug_viz"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_evolution_history(self, history: List[Dict], save_path: str = None):
        """
        绘制适应度进化曲线
        """
        if not history:
            return

        generations = [h['gen'] for h in history]
        best_fits = [h['best_fitness'] for h in history]
        avg_fits = [h['avg_fitness'] for h in history]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(generations, best_fits, 'r-', linewidth=2, label='最优适应度 (Best)')
        ax.plot(generations, avg_fits, 'b--', linewidth=1.5, label='平均适应度 (Avg)')

        ax.set_xlabel('迭代次数 (Generation)', fontsize=12)
        ax.set_ylabel('适应度值 (Fitness)', fontsize=12)
        ax.set_title('遗传算法进化收敛曲线', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(self.output_dir / "evolution_history.png", dpi=300, bbox_inches='tight')
        plt.close()


class ResultVisualizer:
    """
    结果统计可视化 (调试用)
    """

    def __init__(self, data, output_dir: str = "results/debug_viz"):
        self.data = data
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_school_daily_distribution(self, chromosome, save_path: str = None):
        """
        绘制全校每日课程数量分布 (单算法调试用 - 柱状图)
        """
        # 1. 统计每天(周一到周五)的课程总数
        daily_counts = np.zeros(5, dtype=int)

        for task_id in self.data.tasks:
            if task_id in chromosome.genes:
                gene = chromosome.genes[task_id]
                # 解码: 高16位是时间槽
                slot = gene >> 16

                # 严格校验范围 [0, 24]
                if 0 <= slot < 25:
                    day = slot // 5  # 0=周一, ..., 4=周五
                    daily_counts[day] += 1

        # 2. 绘图
        days = ['周一', '周二', '周三', '周四', '周五']

        fig, ax = plt.subplots(figsize=(10, 6))
        # 绘制柱状图
        bars = ax.bar(days, daily_counts, color='#3498db', alpha=0.8, width=0.6, edgecolor='black')

        # 3. 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                    f'{int(height)}', ha='center', va='bottom', fontsize=11)

        ax.set_ylabel('全校课程总数', fontsize=12)
        ax.set_title('全校每日课程分布 (调试视图)', fontsize=14, fontweight='bold')
        ax.grid(True, axis='y', linestyle='--', alpha=0.3)

        # 添加平均线
        avg_count = np.mean(daily_counts)
        ax.axhline(y=avg_count, color='red', linestyle='--', linewidth=1, label=f'平均值: {avg_count:.1f}')
        ax.legend()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(self.output_dir / "school_daily_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()

    def plot_resource_utilization(self, chromosome, save_path: str = None):
        """
        绘制各时间槽的课程负载 (25个时间槽)
        """
        slot_usage = np.zeros(25, dtype=int)
        for gene in chromosome.genes.values():
            slot = gene >> 16
            if 0 <= slot < 25:
                slot_usage[slot] += 1

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.bar(range(25), slot_usage, color='skyblue', edgecolor='black', alpha=0.8)

        ax.set_title('各时间槽课程负载分布 (资源压力)', fontsize=14)
        ax.set_xlabel('时间槽 (周-节次)', fontsize=12)
        ax.set_ylabel('并发课程数', fontsize=12)

        # 设置X轴标签: 1-1 (周一第1节) ... 5-5
        ax.set_xticks(range(25))
        labels = [f"{i // 5 + 1}-{i % 5 + 1}" for i in range(25)]
        ax.set_xticklabels(labels, rotation=90, fontsize=9)

        ax.grid(True, axis='y', linestyle='--', alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(self.output_dir / "resource_utilization.png", dpi=300, bbox_inches='tight')
        plt.close()