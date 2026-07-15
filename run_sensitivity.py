"""
权重灵敏度分析脚本 (Sensitivity Analysis)
文件名: run_sensitivity.py
功能: 仅运行 DL-GA-TS 算法，测试不同权重配置下的表现，用于生成雷达图或对比表。
"""
import os
import csv
import time
import pandas as pd
from datetime import datetime
from data.data_loader import DataLoader
from evaluation.fitness import FitnessEvaluator
from algorithms.genetic_algorithm import GeneticAlgorithm

# === 配置 ===
INSTANCE_NAME = 'S1_Small'  # 选一个代表性算例即可
N_RUNS = 5  # 每个场景跑5次取平均
GENS = 200  # 足够收敛的代数
POP_SIZE = 100
DB_CONFIG = {'host': 'localhost', 'port': 3306, 'user': 'root', 'password': os.getenv('DB_PASSWORD', ''), 'database': 'test3'}


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    res_dir = f"results/sensitivity_{timestamp}"
    os.makedirs(res_dir, exist_ok=True)

    # 1. 定义权重场景 (Scenarios)
    scenarios = [
        {
            'name': 'Balanced (Default)',
            'weights': {'f_daily': 0.20, 'f_interval': 0.20, 'f_room': 0.10, 'f_util': 0.30, 'f_build': 0.20}
        },
        {
            'name': 'Resource_Focus (High Util)',
            'weights': {'f_daily': 0.10, 'f_interval': 0.10, 'f_room': 0.10, 'f_util': 0.60, 'f_build': 0.10}
        },
        {
            'name': 'Comfort_Focus (Student)',
            'weights': {'f_daily': 0.35, 'f_interval': 0.35, 'f_room': 0.05, 'f_util': 0.10, 'f_build': 0.15}
        }
    ]

    # 初始化结果文件
    cols = ['Scenario', 'Run', 'Best_Fitness', 'f_daily', 'f_interval', 'f_room', 'f_util', 'f_build',
            'Daily_Var', 'Util_Avg', 'Interval_Rate', 'Build_Conc_Rate']
    f_csv = open(os.path.join(res_dir, "sensitivity_results.csv"), 'w', newline='', encoding='utf-8-sig')
    writer = csv.writer(f_csv)
    writer.writerow(cols)

    # 加载数据
    print(f"📦 Loading Instance: {INSTANCE_NAME}")
    loader = DataLoader(INSTANCE_NAME, DB_CONFIG)
    data = loader.load()

    # 2. 循环跑场景
    for sc in scenarios:
        s_name = sc['name']
        w_config = sc['weights']
        print(f"\n=== Running Scenario: {s_name} ===")
        print(f"    Weights: {w_config}")

        # 关键：用特定权重初始化评估器
        evaluator = FitnessEvaluator(weights=w_config)
        evaluator.set_data(data)

        # 算法配置 (只跑 Ours)
        algo_config = {
            'use_heuristic_init': True, 'use_rl_adaptive': True, 'use_tabu': True,
            'crossover_rate': 0.8, 'mutation_rate': 0.1,
            'metric_interval': 50, 'elite_count': 2, 'patience': 50
        }

        for run in range(1, N_RUNS + 1):
            ga = GeneticAlgorithm(data, evaluator, algo_config)
            best_ind, _, _, _ = ga.run(GENS, POP_SIZE, run)

            # 计算各项指标 (Raw Scores 和 Physical Metrics)
            raw_scores = evaluator._calculate_raw_scores(best_ind.genes)
            metrics = evaluator.calculate_quality_metrics(best_ind.genes)

            # 记录数据
            row = [
                s_name, run, best_ind.fitness,
                raw_scores['f_daily'], raw_scores['f_interval'], raw_scores['f_room'], raw_scores['f_util'],
                raw_scores['f_build'],
                metrics['Daily_Var'], metrics['Util_Avg'], metrics['Interval_Rate'], metrics['Build_Conc_Rate']
            ]
            writer.writerow(row)
            f_csv.flush()
            print(f"    Run {run}/{N_RUNS}: Util_Avg={metrics['Util_Avg']:.4f}, Daily_Var={metrics['Daily_Var']:.4f}")

    f_csv.close()
    print(f"\n✅ 灵敏度分析完成！结果已保存至 {res_dir}")


if __name__ == "__main__":
    main()
