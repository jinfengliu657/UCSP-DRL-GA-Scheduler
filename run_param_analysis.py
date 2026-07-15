# """
# 参数敏感性分析 (Dual Scale: Small & Medium)
# 文件名: run_param_analysis.py
# 修改内容:
# 1. 适配新的 DataLoader(instance_name, db_config) 接口。
# 2. 明确指定 S1_Small 代表小规模，M1_Medium 代表中规模进行分析。
# 3. 结果分别保存为 best_params_small.json 和 best_params_medium.json。
# """
"""

"""
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from data.data_loader import DataLoader
from evaluation.fitness import FitnessEvaluator
from algorithms.genetic_algorithm import GeneticAlgorithm

# 数据库配置
DB_CONFIG = {'host': 'localhost', 'port': 3306, 'user': 'root', 'password': os.getenv('DB_PASSWORD', ''), 'database': 'test3'}


def run_grid_search(data, res_dir, scale_name):
    """
    执行网格搜索
    scale_name: "small" 或 "medium"
    """
    print(f"\n--- 开始分析: {scale_name.upper()} 规模 (Tasks={data.num_tasks}, Classes={data.num_classes}) ---")

    # 定义网格
    pc_grid = [0.6, 0.7, 0.8, 0.9]
    pm_grid = [0.01, 0.05, 0.1, 0.15]

    best_score = -float('inf')
    best_params = {'crossover_rate': 0.8, 'mutation_rate': 0.1}  # 默认值

    results = []

    total_steps = len(pc_grid) * len(pm_grid)
    step = 0

    evaluator = FitnessEvaluator()
    evaluator.set_data(data)

    for pc in pc_grid:
        for pm in pm_grid:
            step += 1
            print(f"[{step}/{total_steps}] Testing Pc={pc}, Pm={pm} ...", end="", flush=True)

            # 配置 GA
            config = {
                'crossover_rate': pc,
                'mutation_rate': pm,
                'elite_count': 2,
                'use_heuristic_init': True,  # 参数分析时开启启发式，接近真实场景
                'use_rl_adaptive': False,  # 关闭 RL，只测静态参数基准
                'use_tabu': False,  # 关闭 Tabu，只测 GA 核心参数
                'patience': 20,
                'tabu_steps': 10
            }

            # 运行 3 次取平均，减少随机性
            fits = []
            for run_id in range(3):
                ga = GeneticAlgorithm(data, evaluator, config)
                # [关键修正] population_size -> pop_size
                # 小规模跑 50 代，中规模跑 80 代，足够区分优劣
                gens = 50 if scale_name == 'small' else 80
                pop = 60 if scale_name == 'small' else 100

                try:
                    best, _, _ = ga.run(max_generations=gens, pop_size=pop, run_id=run_id)
                    fits.append(best.fitness)
                except Exception as e:
                    print(f" Error: {e}")
                    fits.append(-999999)

            avg_fit = np.mean(fits)
            print(f" Avg Fit: {avg_fit:.2f}")

            results.append({
                'Pc': pc, 'Pm': pm, 'Fitness': avg_fit
            })

            if avg_fit > best_score:
                best_score = avg_fit
                best_params = {'crossover_rate': pc, 'mutation_rate': pm}

    # 保存结果矩阵
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(res_dir, f"grid_search_{scale_name}.csv"), index=False)

    # 保存最优参数 JSON
    json_path = os.path.join("results", f"best_params_{scale_name}.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(best_params, f, indent=4)

    print(f"✅ {scale_name} 最优参数已保存: {best_params}")
    return best_params


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    res_dir = f"results/param_analysis_{timestamp}"
    os.makedirs(res_dir, exist_ok=True)

    print(f"=== 启动参数敏感性分析 ===")
    print(f"结果目录: {res_dir}")

    # 1. 分析小规模 (S1_Small)
    try:
        print("\n[Step 1] Loading Small Instance (S1_Small)...")
        loader_s = DataLoader('S1_Small', DB_CONFIG)
        data_small = loader_s.load()
        run_grid_search(data_small, res_dir, "small")
    except Exception as e:
        print(f"❌ Small Instance Analysis Failed: {e}")

    # 2. 分析中规模 (M1_Medium)
    try:
        print("\n[Step 2] Loading Medium Instance (M1_Medium)...")
        loader_m = DataLoader('M1_Medium', DB_CONFIG)
        data_medium = loader_m.load()
        run_grid_search(data_medium, res_dir, "medium")
    except Exception as e:
        print(f"❌ Medium Instance Analysis Failed: {e}")

    print("\n=== 参数分析完成 ===")
    print("现在您可以运行 run_experiment.py，它会自动读取生成的 json 文件。")


if __name__ == "__main__":
    main()

# import os
# import json
# import numpy as np
# import pandas as pd
# from datetime import datetime
# from data.data_loader import DataLoader
# from evaluation.fitness import FitnessEvaluator
# from algorithms.genetic_algorithm import GeneticAlgorithm
#
# # 数据库配置
# DB_CONFIG = {'host': 'localhost', 'port': 3306, 'user': 'root', 'password': os.getenv('DB_PASSWORD', ''), 'database': 'test3'}
#
# # 参数网格 (Grid Search Space)
# # 您可以根据需要调整搜索范围
# CROSSOVER_RATES = [0.6, 0.7, 0.8, 0.9]
# MUTATION_RATES = [0.01, 0.05, 0.1, 0.2]
#
#
# def run_grid_search(data, res_dir, suffix):
#     """
#     通用的网格搜索函数
#     :param data: 加载好的排课数据
#     :param res_dir: 结果保存目录
#     :param suffix: 文件后缀 ('small' 或 'medium')
#     """
#     print(f"\n--- 开始分析: {suffix.upper()} 规模 (Tasks={data.num_tasks}, Classes={data.num_classes}) ---")
#     evaluator = FitnessEvaluator()
#     evaluator.set_data(data)
#
#     results = []
#     global_best_fit = -float('inf')
#     # 默认最优参数
#     best_params = {'crossover_rate': 0.8, 'mutation_rate': 0.1}
#
#     total_combinations = len(CROSSOVER_RATES) * len(MUTATION_RATES)
#     count = 0
#
#     for pc in CROSSOVER_RATES:
#         for pm in MUTATION_RATES:
#             count += 1
#             print(f"[{count}/{total_combinations}] Testing Pc={pc}, Pm={pm} ...", end="", flush=True)
#
#             fits = []
#             # 每个参数组合跑 2 次取平均，MaxGen=50 (快速验证趋势)
#             # 注意: 参数分析通常不需要跑太长代数，50-100代足以看出收敛优劣
#             for _ in range(2):
#                 config = {
#                     'crossover_rate': pc,
#                     'mutation_rate': pm,
#                     'population_size': 60,  # 分析时种群可以稍大
#                     'elite_count': 2,
#                     'use_heuristic_init': True,  # 开启启发式初始化，加速收敛
#                     'use_rl_adaptive': False,  # 关闭 RL，只分析纯 GA 参数
#                     'use_tabu': False,  # 关闭 Tabu
#                     'patience': 15
#                 }
#                 ga = GeneticAlgorithm(data, evaluator, config)
#                 # run 返回: best_chromosome, history, rl_logs
#                 best, _, _ = ga.run(max_generations=50, population_size=60, run_id=0)
#                 fits.append(best.fitness)
#
#             avg_fit = np.mean(fits)
#             print(f" Avg Fit: {avg_fit:.2f}")
#             results.append({'crossover_rate': pc, 'mutation_rate': pm, 'fitness': avg_fit})
#
#             if avg_fit > global_best_fit:
#                 global_best_fit = avg_fit
#                 best_params = {'crossover_rate': pc, 'mutation_rate': pm}
#
#     # 1. 保存敏感性分析数据 (CSV) - 用于画热力图
#     csv_path = os.path.join(res_dir, f"sensitivity_{suffix}.csv")
#     pd.DataFrame(results).to_csv(csv_path, index=False)
#     print(f"  -> 敏感性数据已保存: {csv_path}")
#
#     # 2. 保存最佳参数 (JSON)
#     json_name = f"best_params_{suffix}.json"
#
#     # 保存到本次结果目录
#     with open(os.path.join(res_dir, json_name), 'w') as f:
#         json.dump(best_params, f, indent=4)
#
#     # 同时保存到项目根目录的 results/ 下，供 run_experiment.py 读取
#     root_results_dir = "results"
#     os.makedirs(root_results_dir, exist_ok=True)
#     root_json_path = os.path.join(root_results_dir, json_name)
#     with open(root_json_path, 'w') as f:
#         json.dump(best_params, f, indent=4)
#
#     print(f"🏆 {suffix.upper()} 最佳参数: {best_params}")
#     print(f"  -> 配置文件已更新: {root_json_path}")
#     return best_params
#
#
# def main():
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     res_dir = f"results/param_analysis_{timestamp}"
#     os.makedirs(res_dir, exist_ok=True)
#
#     print(f"=== 启动参数敏感性分析 ===")
#     print(f"结果目录: {res_dir}")
#
#     # ---------------------------------------------------------
#     # 1. 小规模分析 (Small Scale Analysis)
#     # 使用 'S1_Small' 作为代表实例
#     # ---------------------------------------------------------
#     print("\n[Step 1] Loading Small Instance (S1_Small)...")
#     loader_s = DataLoader('S1_Small', DB_CONFIG)
#     data_small = loader_s.load()
#
#     # 运行网格搜索，结果保存为 best_params_small.json
#     run_grid_search(data_small, res_dir, "small")
#
#     # ---------------------------------------------------------
#     # 2. 中规模分析 (Medium Scale Analysis)
#     # 使用 'M1_Medium' 作为代表实例 (代表中规模和大规模)
#     # ---------------------------------------------------------
#     print("\n[Step 2] Loading Medium Instance (M1_Medium)...")
#     loader_m = DataLoader('M1_Medium', DB_CONFIG)
#     data_medium = loader_m.load()
#
#     # 运行网格搜索，结果保存为 best_params_medium.json
#     run_grid_search(data_medium, res_dir, "medium")
#
#     print("\n=== 全部参数分析完成 ===")
#     print(f"请检查 results/ 目录下的 best_params_*.json 文件")
#
#
# if __name__ == "__main__":
#     main()
#
#
#
#
# # """
# # 参数敏感性分析 (Dual Scale: Small & Medium)
# # 文件名: run_param_analysis.py
# # """
# # import os
# # import json
# # import numpy as np
# # import pandas as pd
# # from datetime import datetime
# # from data.data_loader import DataLoader
# # from evaluation.fitness import FitnessEvaluator
# # from algorithms.genetic_algorithm import GeneticAlgorithm
# #
# # DB_CONFIG = {'host': 'localhost', 'port': 3306, 'user': 'root', 'password': os.getenv('DB_PASSWORD', ''), 'database': 'test3'}
# #
# # # 参数网格
# # CROSSOVER_RATES = [0.6, 0.7, 0.8, 0.9]
# # MUTATION_RATES = [0.01, 0.05, 0.1, 0.2]
# #
# #
# # def run_grid_search(data, res_dir, suffix):
# #     """通用的网格搜索函数"""
# #     print(f"\n--- 开始分析: {suffix.upper()} 规模 (Tasks={data.num_tasks}) ---")
# #     evaluator = FitnessEvaluator()
# #     evaluator.set_data(data)
# #
# #     results = []
# #     global_best_fit = -float('inf')
# #     best_params = {'crossover_rate': 0.8, 'mutation_rate': 0.1}  # 默认值
# #
# #     total = len(CROSSOVER_RATES) * len(MUTATION_RATES)
# #     count = 0
# #
# #     for pc in CROSSOVER_RATES:
# #         for pm in MUTATION_RATES:
# #             count += 1
# #             print(f"[{count}/{total}] Testing Pc={pc}, Pm={pm} ...", end="", flush=True)
# #
# #             fits = []
# #             # 跑 2 次取平均，MaxGen=50 (足够看出优劣趋势)
# #             for _ in range(2):
# #                 config = {
# #                     'crossover_rate': pc, 'mutation_rate': pm,
# #                     'population_size': 60, 'elite_count': 2,
# #                     'use_heuristic_init': True,  # 开启启发式加速收敛
# #                     'use_rl_adaptive': False, 'use_tabu': False, 'patience': 15
# #                 }
# #                 ga = GeneticAlgorithm(data, evaluator, config)
# #                 best, _, _ = ga.run(max_generations=50, pop_size=60, run_id=0)
# #                 fits.append(best.fitness)
# #
# #             avg_fit = np.mean(fits)
# #             print(f" Fit: {avg_fit:.1f}")
# #             results.append({'crossover_rate': pc, 'mutation_rate': pm, 'fitness': avg_fit})
# #
# #             if avg_fit > global_best_fit:
# #                 global_best_fit = avg_fit
# #                 best_params = {'crossover_rate': pc, 'mutation_rate': pm}
# #
# #     # 保存结果 CSV (用于画3D图)
# #     df = pd.DataFrame(results)
# #     df.to_csv(os.path.join(res_dir, f"sensitivity_{suffix}.csv"), index=False)
# #
# #     # 保存最佳参数 JSON
# #     json_name = f"best_params_{suffix}.json"
# #
# #     # 1. 存到本次结果目录
# #     with open(os.path.join(res_dir, json_name), 'w') as f:
# #         json.dump(best_params, f, indent=4)
# #
# #     # 2. 存到根目录 results/ (供主实验读取)
# #     root_json = os.path.join("results", json_name)
# #     if not os.path.exists("results"): os.makedirs("results")
# #     with open(root_json, 'w') as f:
# #         json.dump(best_params, f, indent=4)
# #
# #     print(f"🏆 {suffix.upper()} 最佳参数: {best_params}")
# #     return best_params
# #
# #
# # def main():
# #     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# #     res_dir = f"results/param_analysis_{timestamp}"
# #     os.makedirs(res_dir, exist_ok=True)
# #
# #     loader = DataLoader(DB_CONFIG)
# #     colleges = loader.get_college_list()
# #     while len(colleges) < 6: colleges.extend(colleges)
# #
# #     # 1. 小规模分析 (S1)
# #     data_small = loader.load_all_data([colleges[0]['college_id']])
# #     run_grid_search(data_small, res_dir, "small")
# #
# #     # 2. 中规模分析 (M1)
# #     m_ids = [c['college_id'] for c in colleges[:3]]
# #     data_medium = loader.load_all_data(m_ids)
# #     run_grid_search(data_medium, res_dir, "medium")
# #
# #     print("\n=== 全部参数分析完成 ===")
# #     print("已生成配置文件:")
# #     print("  - results/best_params_small.json")
# #     print("  - results/best_params_medium.json")
# #
# #
# # if __name__ == "__main__":
# #     main()
