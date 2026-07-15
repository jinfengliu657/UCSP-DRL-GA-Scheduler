"""
参数敏感性分析 (只运行 Large 规模)
文件名: run_param_analysis.py
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
    scale_name: "small", "medium" 或 "large"
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

                # [新增] 针对大规模增加迭代代数和种群规模
                if scale_name == 'small':
                    gens, pop = 50, 60
                elif scale_name == 'medium':
                    gens, pop = 80, 100
                else:  # large
                    gens, pop = 100, 120  # 大规模问题建议给更多的搜索空间

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

    # 注释掉已经跑完的小规模和中规模，节省时间
    """
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
    """

    # 3. 分析大规模 (L1_Large) - 假设实例名称为 L1_Large
    try:
        print("\n[Step 3] Loading Large Instance (L1_Large)...")
        # 如果你的数据库中大实例的名字不是 L1_Large，请在这里修改
        loader_l = DataLoader('L1_Large', DB_CONFIG)
        data_large = loader_l.load()
        run_grid_search(data_large, res_dir, "large")
    except Exception as e:
        print(f"❌ Large Instance Analysis Failed: {e}")

    print("\n=== 大规模参数分析完成 ===")
    print("现在所有的 best_params_*.json 文件都已准备完毕。")


if __name__ == "__main__":
    main()
