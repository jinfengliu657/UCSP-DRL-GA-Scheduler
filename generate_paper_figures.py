"""
图表生成器 (最终修订版 - 确保路径隔离与全功能覆盖)
文件名: generate_paper_figures.py
"""
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from visualization.result_plots import ResultVisualizer
from data.data_loader import DataLoader

# 消除警告并设置通用字体
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']

# 数据库配置 (用于加载 Large 实例的详细数据以绘制课表)
DB_CONFIG = {'host': 'localhost', 'port': 3306, 'user': 'root', 'password': os.getenv('DB_PASSWORD', ''), 'database': 'test3'}


def main():
    res_root = "results"
    print(f"🚀 [Start] 开始执行绘图脚本...")
    print(f"📂 正在扫描目录: {os.path.abspath(res_root)}\n")

    if not os.path.exists(res_root):
        print(f"❌ 错误: 找不到 results 文件夹！请先运行实验。")
        return

    # ===========================================================
    # 1. 查找并绘制参数敏感性分析 (3D Surface)
    # ===========================================================
    param_exps = sorted([d for d in os.listdir(res_root) if d.startswith("param_analysis_")])
    if param_exps:
        param_path = os.path.join(res_root, param_exps[-1])
        print(f"📊 [1/2] 处理参数敏感性分析: {param_exps[-1]}")

        # 初始化可视化器，指向参数分析目录
        viz_p = ResultVisualizer(param_path)

        # 尝试加载常见的敏感性分析文件名
        for c_name in ["sensitivity_medium.csv", "sensitivity_small.csv", "sensitivity_result.csv"]:
            p_file = os.path.join(param_path, c_name)
            if os.path.exists(p_file):
                try:
                    df_p = pd.read_csv(p_file)
                    viz_p.plot_parameter_sensitivity(df_p)
                    print(f"      ✅ 已生成 3D 参数热力图 ({c_name})")
                except Exception as e:
                    print(f"      ❌ 参数绘图失败: {e}")
    else:
        print("   ⚠️ 未找到参数分析结果，跳过此步骤。")

    print("-" * 60)

    # ===========================================================
    # 2. 处理主实验数据 (Final Experiments)
    # ===========================================================
    exps = sorted([d for d in os.listdir(res_root) if d.startswith("final_exp_")])
    if not exps:
        print(f"❌ 未找到主实验结果 (final_exp_*)！脚本结束。")
        return

    base_path = os.path.join(res_root, exps[-1])
    print(f"📈 [2/2] 处理最新主实验数据: {exps[-1]}")

    # 初始化数据加载器 (仅在需要绘制详细课表时使用)
    loader = DataLoader(DB_CONFIG)

    # 获取所有实例文件夹 (排除 figures 目录)
    instances = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d)) and d != "figures"]

    if not instances:
        print("   ⚠️ 实验目录下没有实例文件夹。")

    for i, inst in enumerate(instances):
        inst_path = os.path.join(base_path, inst)
        print(f"\n   👉 ({i + 1}/{len(instances)}) 正在处理实例: {inst}")

        # [关键] 每次循环重新初始化 viz，确保 output_dir 指向当前实例的 figures 文件夹
        # 这样生成的图表绝对不会“串门”
        viz = ResultVisualizer(os.path.join(inst_path, "figures"))

        # ---------------------------
        # A. CSV 数据绘图 (收敛、RL、统计)
        # ---------------------------
        try:
            # 1. 收敛历史 (History.csv)
            #    负责: BestFit, AvgFit, Constraints, Stability Boxplot(增强版)
            if os.path.exists(f"{base_path}/history.csv"):
                df_hist = pd.read_csv(f"{base_path}/history.csv")
                # [关键] 筛选当前实例的数据
                inst_hist = df_hist[df_hist['Instance'] == inst]
                if not inst_hist.empty:
                    viz.plot_convergence_details(inst_hist)
                    print("      - 收敛分析图 (含 AvgFit, 箱线图)...OK")

            # 2. RL 轨迹 (rl_trace.csv)
            #    负责: Loss, Reward, Q-Heatmap(增强版), Action Dist
            if os.path.exists(f"{base_path}/rl_trace.csv"):
                df_trace = pd.read_csv(f"{base_path}/rl_trace.csv")
                inst_trace = df_trace[df_trace['Instance'] == inst]
                if not inst_trace.empty:
                    viz.plot_learning_curve(inst_trace)
                    viz.plot_rl_analysis(inst_trace)
                    print("      - RL 深度分析 (含 Q-Heatmap, Loss)...OK")

            # 3. 统计汇总 (summary.csv)
            #    负责: Metrics Table, Comparisons, Significance
            if os.path.exists(f"{base_path}/summary.csv"):
                df_sum = pd.read_csv(f"{base_path}/summary.csv")
                inst_sum = df_sum[df_sum['Instance'] == inst]
                if not inst_sum.empty:
                    viz.export_metrics_table(inst_sum)
                    viz.plot_metrics_comparison(inst_sum)
                    viz.plot_significance(inst_sum)
                    print("      - 指标对比与显著性分析...OK")

        except Exception as e:
            print(f"      ❌ CSV 数据处理出错: {e}")

        # ---------------------------
        # B. JSON 数据绘图 (课表、热力图、每日分布)
        # ---------------------------
        try:
            json_path = f"{inst_path}/best_schedules.json"
            if os.path.exists(json_path):
                with open(json_path, 'r', encoding='utf-8') as f:
                    sols = json.load(f)

                if sols:
                    # 1. 绘制全校课程每日分布图 (修复：之前版本可能遗漏)
                    first_algo = next(iter(sols))
                    n_tasks = len(sols[first_algo])
                    viz.plot_daily_dist(sols, n_tasks)
                    print("      - 课程每日分布图...OK")

                    # 2. Large 实例专属: 详细课表 & 教室热力图
                    if "Large" in inst:
                        print(f"      - [Large 实例] 正在加载完整数据以生成详细课表...")
                        try:
                            full_data = loader.load_all_data(None)

                            target_algo = 'DL-GA-TS' if 'DL-GA-TS' in sols else first_algo

                            # [修复] 确保 Key 为字符串，匹配 result_plots.py 的查找逻辑
                            # 之前版本使用了 int(k)，这里改为 str(k)
                            best_genes = {str(k): v for k, v in sols[target_algo].items()}

                            viz.plot_complex_schedule(best_genes, full_data)
                            viz.plot_classroom_heatmap(best_genes, full_data)
                            print("      - 复杂合班课表与教室热力图...OK")
                        except Exception as e:
                            print(f"      ⚠️ 无法连接数据库或数据加载失败，跳过详细课表绘制: {e}")
            else:
                print(f"      ⚠️ 未找到 best_schedules.json，跳过课表绘制。")

        except Exception as e:
            print(f"      ❌ JSON 数据处理出错: {e}")

    print("\n" + "=" * 60)
    print("✅ 所有图表生成完毕！请查看 results/*/figures 文件夹。")
    print("=" * 60)


if __name__ == "__main__":
    main()



# """
# 学术论文图表一键生成器 (Final Academic Version)
# 文件名: generate_paper_figures.py
# 功能：自动扫描实验结果，生成包含收敛分析、显著性检验、稳定性分析、
#       RL轨迹、3D敏感性、教室热力图、复杂课表在内的全套论文图片。
# """
# import os
# import json
# import pandas as pd
# import matplotlib.pyplot as plt
# from visualization.result_plots import ResultVisualizer
# from data.data_loader import DataLoader
#
# # ==========================================
# # 全局学术配置
# # ==========================================
# plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
# # 优先使用 Times New Roman，若环境没有则回退
# plt.rcParams['font.sans-serif'] = ['Times New Roman', 'SimHei', 'Arial']
#
# # 数据库配置（保持与项目一致）
# DB_CONFIG = {'host': 'localhost', 'port': 3306, 'user': 'root', 'password': os.getenv('DB_PASSWORD', ''), 'database': 'test3'}
#
#
# def main():
#     # 自动定位最新的实验结果目录
#     res_root = "results"
#     print(f"🚀 [Senior Reviewer Mode] 开始扫描实验数据...")
#
#     if not os.path.exists(res_root):
#         print(f"❌ 错误: 找不到 {res_root} 文件夹！请确认实验已完成。")
#         return
#
#     # -----------------------------------------------------------
#     # 1. 绘制：参数敏感性分析 (3D Surface Plot)
#     # -----------------------------------------------------------
#     param_exps = sorted([d for d in os.listdir(res_root) if d.startswith("param_analysis_")])
#     if param_exps:
#         param_path = os.path.join(res_root, param_exps[-1])
#         print(f"📊 [1/4] 检测到参数分析数据: {param_path}")
#         viz_param = ResultVisualizer(param_path)
#         for csv_name in ["sensitivity_small.csv", "sensitivity_medium.csv"]:
#             csv_p = os.path.join(param_path, csv_name)
#             if os.path.exists(csv_p):
#                 df_p = pd.read_csv(csv_p)
#                 viz_param.plot_parameter_sensitivity(df_p)
#         print("      - 3D参数敏感性分析图...完成")
#
#     # -----------------------------------------------------------
#     # 2. 绘制：正式对比实验 (全规模实例：Small/Medium/Large)
#     # -----------------------------------------------------------
#     final_exps = sorted([d for d in os.listdir(res_root) if d.startswith("final_exp_")])
#     if not final_exps:
#         print("⚠️ 未发现正式对比实验数据 (final_exp_*)。")
#         return
#
#     exp_path = os.path.join(res_root, final_exps[-1])
#     print(f"📂 [2/4] 正在处理最新实验结果: {exp_path}")
#
#     # A. 读取全局历史数据 (用于全实例收敛图)
#     hist_p = os.path.join(exp_path, "history.csv")
#     if os.path.exists(hist_p):
#         df_hist_all = pd.read_csv(hist_p)
#         viz_global = ResultVisualizer(os.path.join(exp_path, "global_plots"))
#         # 自动遍历所有规模实例
#         for inst in df_hist_all['Instance'].unique():
#             df_inst = df_hist_all[df_hist_all['Instance'] == inst]
#             viz_global.plot_convergence_details(df_inst)  # 调用带 Inset 的专业收敛图
#         print(f"      - 所有规模实例的收敛分析图 (含阴影与放大图)...完成")
#
#     # B. 读取 RL 训练轨迹 (针对 DL-GA-TS 的深度学习分析)
#     trace_p = os.path.join(exp_path, "rl_trace.csv")
#     if os.path.exists(trace_p):
#         df_trace = pd.read_csv(trace_p)
#         viz_rl = ResultVisualizer(os.path.join(exp_path, "rl_analysis"))
#         viz_rl.plot_learning_curve(df_trace)
#         viz_rl.plot_rl_analysis(df_trace)
#         print("      - DRL 代理训练奖励与 Loss 曲线...完成")
#
#     # -----------------------------------------------------------
#     # 3. 递归处理每个实例的详细指标 (显著性/热力图/课表)
#     # -----------------------------------------------------------
#     print(f"🔍 [3/4] 正在深度解析各子实例目录...")
#     loader = DataLoader(DB_CONFIG)
#
#     # 获取所有子实例文件夹 (S1_Small, M1_Medium, L1_Large 等)
#     instances = [d for d in os.listdir(exp_path) if os.path.isdir(os.path.join(exp_path, d))
#                  and d not in ['global_plots', 'rl_analysis']]
#
#     for inst in instances:
#         inst_path = os.path.join(exp_path, inst)
#         fig_dir = os.path.join(inst_path, "figures")
#         viz = ResultVisualizer(fig_dir)
#
#         print(f"   📍 实例: {inst}")
#
#         # 1. 显著性分析与三线表 (关键学术支撑)
#         sum_p = os.path.join(inst_path, "评价指标表.xlsx - Sheet1.csv")
#         if os.path.exists(sum_p):
#             df_sum = pd.read_csv(sum_p)
#             viz.plot_significance(df_sum)  # 专业 Wilcoxon 显著性图
#             viz.plot_metrics_comparison(df_sum)  # 指标柱状图
#             viz.export_metrics_table(df_sum)  # 导出 CSV 格式的三线表
#             print("      - 显著性分析与统计指标表...完成")
#
#         # 2. 课表分布、热力图与复杂课表 (Large 实例专属或按需生成)
#         json_path = os.path.join(inst_path, "best_schedules.json")
#         if os.path.exists(json_path):
#             with open(json_path) as f:
#                 sols = json.load(f)
#
#             # 绘制课程每日分布
#             viz.plot_daily_dist(sols, len(next(iter(sols.values()))))
#
#             # 如果是 Large 实例，生成高密度图表
#             if "Large" in inst or "Medium" in inst:
#                 try:
#                     full_data = loader.load_all_data(None)
#                     # 优先展示我们提出的算法结果
#                     target_algo = 'DL-GA-TS' if 'DL-GA-TS' in sols else list(sols.keys())[0]
#                     best_genes = {int(k): v for k, v in sols[target_algo].items()}
#
#                     viz.plot_complex_schedule(best_genes, full_data)  # 复杂课表可视化
#                     viz.plot_classroom_heatmap(best_genes, full_data)  # 教室利用率热力图
#                     print("      - 详细课表与教室热力图...完成")
#                 except Exception as e:
#                     print(f"      ⚠️ 课表渲染跳过 (数据依赖未满足): {e}")
#
#     print(f"\n✨ [4/4] 恭喜！所有论文所需图表已按 SCI 一区标准生成。")
#     print(f"📂 结果请查看: {os.path.abspath(exp_path)}")
#
#
# if __name__ == "__main__":
#     main()
