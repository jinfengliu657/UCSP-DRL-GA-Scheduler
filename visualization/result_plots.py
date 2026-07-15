"""
全能绘图库 (最终完整覆盖版 - 严禁删减)
文件名: visualization/result_plots.py
功能: 提供符合 SCI 论文标准的 DRL-GA 实验可视化图表
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D


class ResultVisualizer:
    def __init__(self, output_dir):
        # 确保输出目录存在
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._init_style()

    def _init_style(self):
        """初始化绘图风格，解决字体和负号显示问题"""
        try:
            sns.set_style("whitegrid")
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
        except Exception:
            pass

    # --- 1. 学习曲线 (Loss & Reward) ---
    def plot_learning_curve(self, df_trace):
        """绘制 RL 训练 Loss 和 Reward"""
        if df_trace.empty: return
        self._init_style()
        df = df_trace.copy()

        # Loss 曲线 (对接 dqn_agent.py 的返回)
        if 'Loss' in df.columns:
            valid_loss = df[df['Loss'] > 1e-6]
            if not valid_loss.empty:
                plt.figure(figsize=(10, 5))
                sns.lineplot(x='Gen', y='Loss', data=valid_loss, alpha=0.6, linewidth=1.5, color='#1f77b4')
                plt.title('DQN 训练损失收敛曲线 (Loss Curve)')
                plt.xlabel('Generation')
                plt.ylabel('Loss (Log Scale)')
                plt.yscale('log')
                plt.grid(True, which="both", ls="--", alpha=0.3)
                plt.tight_layout()
                plt.savefig(self.output_dir / "rl_loss_curve.png", dpi=300)
                plt.close()

        # Reward 曲线 (对接 rl_controller.py 奖励逻辑)
        plt.figure(figsize=(10, 5))
        df['Reward_MA'] = df['Reward'].rolling(window=20, min_periods=1).mean()
        sns.lineplot(x='Gen', y='Reward', data=df, alpha=0.3, label='Raw Reward', color='gray')
        sns.lineplot(x='Gen', y='Reward_MA', data=df, color='red', label='Moving Avg (20)', linewidth=2)
        plt.title('强化学习奖励趋势 (Reward Trend)')
        plt.xlabel('Generation');
        plt.ylabel('Reward');
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.output_dir / "rl_reward_curve.png", dpi=300)
        plt.close()

    # --- 2. 详细收敛分析 ---
    def plot_convergence_details(self, df_hist):
        """绘制收敛过程及最终稳定性"""
        self._init_style()

        # A. 最优适应度收敛曲线
        plt.figure(figsize=(10, 6))
        sns.lineplot(x='Gen', y='BestFit', hue='Algorithm', data=df_hist, errorbar='sd')
        plt.title('最优适应度收敛曲线 (Best Fitness ± Std)')
        plt.savefig(self.output_dir / "conv_best_fit.png", dpi=300);
        plt.close()

        # B. 平均适应度收敛曲线 (补回内容)
        plt.figure(figsize=(10, 6))
        sns.lineplot(x='Gen', y='AvgFit', hue='Algorithm', data=df_hist, errorbar='sd')
        plt.title('平均适应度收敛曲线 (Avg Fitness ± Std)')
        plt.savefig(self.output_dir / "conv_avg_fit.png", dpi=300);
        plt.close()

        # C. 约束冲突分析
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        sns.lineplot(x='Gen', y='Violations', hue='Algorithm', data=df_hist, ax=ax1, errorbar=None)
        ax1.set_title('硬约束冲突数 (Hard Violations)')
        if 'Soft_Score_Sum' in df_hist.columns:
            sns.lineplot(x='Gen', y='Soft_Score_Sum', hue='Algorithm', data=df_hist, ax=ax2, errorbar='sd')
            ax2.set_title('软约束得分 (Soft Score)')
        plt.tight_layout()
        plt.savefig(self.output_dir / "conv_constraints.png", dpi=300);
        plt.close()

        # D. [增强版] 稳定性箱线图
        final_gen = df_hist['Gen'].max()
        final_data = df_hist[df_hist['Gen'] == final_gen]
        plt.figure(figsize=(9, 6))
        sns.boxplot(x='Algorithm', y='BestFit', data=final_data,
                    palette='Set3', notch=True, showmeans=True,
                    meanprops={"marker": "D", "markerfacecolor": "white", "markeredgecolor": "black", "markersize": 6})
        sns.stripplot(x='Algorithm', y='BestFit', data=final_data, color="orange", size=4, jitter=0.2, alpha=0.4)
        plt.title(f'算法最终解稳定性分布 (迭代: {final_gen}代)')
        plt.ylabel('最优适应度 (Best Fitness)')
        plt.tight_layout()
        plt.savefig(self.output_dir / "stability_boxplot.png", dpi=300);
        plt.close()

    # --- 3. RL 深度分析 ---
    def plot_rl_analysis(self, df_trace):
        """分析 RL 决策过程与 Q 值演化"""
        if df_trace.empty: return
        self._init_style()

        # A. 动作分布
        plt.figure(figsize=(8, 5))
        sns.countplot(x='Action', data=df_trace, palette='viridis')
        plt.title('RL Agent 动作选择频率分布')
        plt.savefig(self.output_dir / "rl_action_dist.png", dpi=300);
        plt.close()

        # B. [增强版] Q-Value 进化热力图
        try:
            run1 = df_trace[df_trace['Run'] == 1]
            q_list = [eval(str(x)) for x in run1['Q_Values']]
            q_mat = np.array(q_list).T
            plt.figure(figsize=(12, 6))
            action_labels = ['Hold', 'Converge', 'Balanced', 'Explore', 'Emergency']
            sns.heatmap(q_mat, cmap='magma', cbar_kws={'label': 'Q-Value'})
            plt.yticks(np.arange(len(action_labels)) + 0.5, action_labels, rotation=0)
            plt.title('强化学习 Agent 决策偏好演化 (Q-Value Heatmap)')
            plt.xlabel('Generation');
            plt.ylabel('Actions')
            plt.tight_layout()
            plt.savefig(self.output_dir / "rl_q_heatmap.png", dpi=300);
            plt.close()

            # C. 参数轨迹
            fig, ax1 = plt.subplots(figsize=(10, 6))
            ax1.plot(run1['Gen'], run1['Mutation'], 'r-', label='Pm')
            ax1.plot(run1['Gen'], run1['Crossover'], 'g--', label='Pc')
            ax1.set_ylabel('GA Parameters');
            ax1.legend(loc='upper left')
            ax2 = ax1.twinx()
            ax2.scatter(run1['Gen'], run1['Action'], c='blue', alpha=0.2, s=8)
            ax2.set_ylabel('Action Index')
            plt.savefig(self.output_dir / "rl_params_traj.png", dpi=300);
            plt.close()
        except Exception as e:
            print(f"RL Analysis Error: {e}")

    # --- 4. 每日分布图 (核心补全) ---
    def plot_daily_dist(self, solutions, num_tasks):
        """全校课程每日分布对比"""
        self._init_style()
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']
        data = []
        for algo, genes in solutions.items():
            counts = np.zeros(5)
            for g in genes.values():
                day = (g >> 16) // 5
                if 0 <= day < 5: counts[day] += 1
            for i, c in enumerate(counts):
                data.append({'Algorithm': algo, 'Day': days[i], 'Count': c})

        df = pd.DataFrame(data)
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Day', y='Count', hue='Algorithm', data=df, palette='viridis')
        plt.axhline(num_tasks / 5, color='red', ls='--', label='Ideal Mean')
        plt.title('全校课程每日分布对比')
        plt.legend();
        plt.tight_layout()
        plt.savefig(self.output_dir / "daily_distribution.png", dpi=300);
        plt.close()

    # --- 5. 教室与课表 ---
    def plot_classroom_heatmap(self, genes, data):
        """教室利用率热力图"""
        self._init_style()
        room_ids = sorted(list(data.classrooms.keys()))[:30]
        if not room_ids: return
        matrix = np.zeros((len(room_ids), 25))
        room_map = {rid: i for i, rid in enumerate(room_ids)}
        for task_id, gene in genes.items():
            slot = gene >> 16;
            r_key = str(gene & 0xFFFF)
            if r_key in room_map and 0 <= slot < 25: matrix[room_map[r_key], slot] = 1
        plt.figure(figsize=(14, 10))
        sns.heatmap(matrix, cmap="Blues", cbar_kws={'label': 'Occupied'}, linewidths=0.5)
        plt.title('教室资源利用率热力图 (Top 30)')
        plt.savefig(self.output_dir / "classroom_utilization_heatmap.png", dpi=300);
        plt.close()

    def plot_complex_schedule(self, genes, data, title="合班详细课表"):
        """绘制详细课表"""
        self._init_style()
        target_gid = None
        for gid, tids in data.combined_classes.items():
            if len(tids) >= 2: target_gid = (gid, tids); break
        if not target_gid: return
        gid, tids = target_gid

        schedule = [["" for _ in range(5)] for _ in range(5)]
        for tid in tids:
            if str(tid) not in genes: continue
            task = data.tasks[str(tid)] if isinstance(tid, str) else data.tasks[tid]
            gene = genes[str(tid)]
            slot = gene >> 16;
            day, period = slot // 5, slot % 5
            if 0 <= day < 5 and 0 <= period < 5:
                tname = data.teachers[
                    task.teacher_id].teacher_name if task.teacher_id in data.teachers else task.teacher_id
                info = f"{task.course_class_name}\n{tname}\n[{gene & 0xFFFF}]"
                schedule[period][day] += info + "\n\n"

        fig, ax = plt.subplots(figsize=(12, 8));
        ax.axis('off')
        ax.table(cellText=schedule, colLabels=['Mon', 'Tue', 'Wed', 'Thu', 'Fri'],
                 rowLabels=['1-2节', '3-4节', '5-6节', '7-8节', '晚间'], loc='center').scale(1, 4)
        plt.title(f"{title} - Group {gid}");
        plt.tight_layout()
        plt.savefig(self.output_dir / "schedule_complex_combined.png", dpi=300);
        plt.close()

    # --- 6. 统计报表 ---
    def export_metrics_table(self, df_sum):
        """生成 SCI 三线表"""
        metrics = ['BestFit', 'Daily_Var', 'Util_Avg', 'Interval_Rate', 'Time']
        fmt = lambda x: f"{x.mean():.2f} ± {x.std():.2f}"
        try:
            table = df_sum.groupby('Algorithm')[metrics].apply(
                lambda x: pd.Series({
                    'BestFit': fmt(x['BestFit']),
                    'Uniformity (Daily_Var ↓)': fmt(x['Daily_Var']),
                    'Utilization (Util_Avg 0.5)': fmt(x['Util_Avg']),
                    'Interval (Interval_Rate ↑)': fmt(x['Interval_Rate']),
                    'Time (s)': fmt(x['Time'])
                })
            )
            table.to_csv(self.output_dir / "metrics_three_line_table.csv", encoding='utf-8-sig')
        except:
            pass

    def plot_metrics_comparison(self, df_sum):
        """指标柱状图"""
        metrics = [('Daily_Var', '分布均匀度'), ('Util_Avg', '教室利用率'), ('Interval_Rate', '间隔满意度')]
        for col, title in metrics:
            plt.figure(figsize=(8, 6))
            sns.barplot(x='Algorithm', y=col, data=df_sum, palette='Set2', errorbar='sd', capsize=0.1)
            plt.title(title);
            plt.tight_layout()
            plt.savefig(self.output_dir / f"metric_{col}.png", dpi=300);
            plt.close()

    def plot_significance(self, df_sum):
        """显著性检验"""
        target = 'DL-GA-TS'
        algos = [a for a in df_sum['Algorithm'].unique() if a != target]
        if not algos or target not in df_sum['Algorithm'].unique(): return
        p_vals, labels = [], []
        target_data = df_sum[df_sum['Algorithm'] == target]['BestFit']
        for algo in algos:
            other_data = df_sum[df_sum['Algorithm'] == algo]['BestFit']
            try:
                min_len = min(len(target_data), len(other_data))
                _, p = stats.wilcoxon(target_data[:min_len], other_data[:min_len])
                p_vals.append(p);
                labels.append(f"{target}\nvs\n{algo}")
            except:
                p_vals.append(1.0); labels.append(f"{target}\nvs\n{algo}")
        plt.figure(figsize=(8, 6))
        plt.bar(labels, p_vals, color=['#e74c3c' if p < 0.05 else 'gray' for p in p_vals])
        plt.axhline(y=0.05, color='red', ls='--');
        plt.yscale('log')
        plt.tight_layout();
        plt.savefig(self.output_dir / "significance_bar.png", dpi=300);
        plt.close()

    def plot_parameter_sensitivity(self, df_param):
        """3D 参数分析"""
        if df_param.empty: return
        try:
            fig = plt.figure(figsize=(10, 8));
            ax = fig.add_subplot(111, projection='3d')
            pivot = df_param.pivot(index="mutation_rate", columns="crossover_rate", values="fitness")
            X, Y = np.meshgrid(pivot.columns, pivot.index)
            ax.plot_surface(X, Y, pivot.values, cmap='viridis')
            plt.savefig(self.output_dir / "param_sensitivity_3d.png", dpi=300);
            plt.close()
        except:
            pass