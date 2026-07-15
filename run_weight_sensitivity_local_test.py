# -*- coding: utf-8 -*-
"""
本地增强测试版：1个实例 + 2个权重方案 + 各2次运行
文件名: run_weight_sensitivity_local_2x2_test.py

目的：
1. 验证权重灵敏性分析主流程是否打通
2. 验证不同权重方案是否能独立保存各自的 DRL memory
3. 验证 summary 输出是否正常
4. 验证同一方案内连续两次 run 是否能正常复用 memory

说明：
- 只跑 Ours
- 只跑 1 个实例
- 跑 2 个权重方案
- 每个方案跑 2 次
"""

import os
import csv
import json
import time
import pickle
import shutil
from copy import deepcopy

from evaluation.fitness import FitnessEvaluator
from algorithms.genetic_algorithm import GeneticAlgorithm


# =========================
# 基本配置
# =========================
GLOBAL_WEIGHTS_FILE = "drl_model_memory.pth"
OURS_NAME = "DL-GA-TS-Block-Repair-TrueCDM"

# 只测 1 个实例
TEST_INSTANCE = "S1_Small"

# 本地增强测试参数：比最小测试稍强，但仍较轻
TEST_CONFIG = {
    "type": "Small",
    "gen": 80,
    "pop": 50,
    "patience": 20
}

# 2 个权重方案
WEIGHT_SCHEMES = {
    "B0_Base": {
        "f_daily": 0.20,
        "f_interval": 0.20,
        "f_room": 0.10,
        "f_util": 0.30,
        "f_build": 0.20
    },
    "B4_Util": {
        "f_daily": 0.15,
        "f_interval": 0.15,
        "f_room": 0.10,
        "f_util": 0.40,
        "f_build": 0.20
    }
}

N_RUNS = 2

# Ours 固定配置
OURS_FLAGS = {
    "use_heuristic_init": True,
    "use_rl_adaptive": True,
    "use_block_operators": True,
    "use_repair": True,
    "use_cdm": True,
    "use_tabu": True,
    "elite_count": 2,
    "metric_interval": 10,
    "tabu_steps": 10
}


# =========================
# 工具函数
# =========================
def validate_weights(weights: dict):
    required = {"f_daily", "f_interval", "f_room", "f_util", "f_build"}
    if set(weights.keys()) != required:
        raise ValueError(f"权重键不匹配: {weights.keys()}")
    s = sum(float(v) for v in weights.values())
    if abs(s - 1.0) > 1e-8:
        raise ValueError(f"权重和必须为1.0，当前为 {s}")


def load_best_params_by_type(inst_type: str):
    suffix = "small" if inst_type == "Small" else "medium"
    if inst_type == "Large":
        suffix = "medium"

    json_path = os.path.join("results", f"best_params_{suffix}.json")
    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass

    return {"crossover_rate": 0.8, "mutation_rate": 0.1}


def load_cached_instance(inst_name: str):
    cache_file = os.path.join("data", "cache", f"{inst_name}_cache.pkl")
    if not os.path.exists(cache_file):
        raise FileNotFoundError(f"缓存文件不存在: {cache_file}")

    with open(cache_file, "rb") as f:
        data = pickle.load(f)
    return data


def sanitize_name(name: str) -> str:
    out = []
    for ch in name:
        if ch.isalnum() or ch in ("-", "_"):
            out.append(ch)
        else:
            out.append("_")
    return "".join(out)


def compute_final_metrics(evaluator: FitnessEvaluator, best_ind, history: list):
    last_rec = history[-1] if history else {}
    vio = evaluator.constraint_checker.check_all(best_ind.genes)["total"]

    if vio > 0:
        return {
            "Best_Fitness": float(best_ind.fitness),
            "Avg_Fitness": float(last_rec.get("avg_fitness", last_rec.get("avg_fit", 0))),
            "Violations": int(vio),
            "Soft_Score_Sum": 0.0,
            "f_daily": 0.0,
            "f_interval": 0.0,
            "f_room": 0.0,
            "f_util": 0.0,
            "f_build": 0.0,
            "Daily_Var": 0.0,
            "Util_Avg": 0.0,
            "Interval_Rate": 0.0,
            "Build_Conc_Rate": 0.0
        }

    s_sum_raw = last_rec.get("soft_score_sum", 0)
    s_sum = sum(s_sum_raw.values()) if isinstance(s_sum_raw, dict) else float(s_sum_raw)

    if "f_daily" in last_rec:
        raw_scores = {
            "f_daily": float(last_rec.get("f_daily", 0)),
            "f_interval": float(last_rec.get("f_interval", 0)),
            "f_room": float(last_rec.get("f_room", 0)),
            "f_util": float(last_rec.get("f_util", 0)),
            "f_build": float(last_rec.get("f_build", 0))
        }
    else:
        raw_scores = evaluator._calculate_raw_scores(best_ind.genes)

    d_var = last_rec.get("Daily_Var", 0)
    if d_var == 0:
        mets = evaluator.calculate_quality_metrics(best_ind.genes)
        d_var = mets["Daily_Var"]
        u_avg = mets["Util_Avg"]
        i_rate = mets["Interval_Rate"]
        b_rate = mets["Build_Conc_Rate"]
    else:
        u_avg = last_rec.get("Util_Avg", 0)
        i_rate = last_rec.get("Interval_Rate", 0)
        b_rate = last_rec.get("Build_Conc_Rate", 0)

    return {
        "Best_Fitness": float(best_ind.fitness),
        "Avg_Fitness": float(last_rec.get("avg_fitness", last_rec.get("avg_fit", 0))),
        "Violations": int(vio),
        "Soft_Score_Sum": float(s_sum),
        "f_daily": float(raw_scores["f_daily"]),
        "f_interval": float(raw_scores["f_interval"]),
        "f_room": float(raw_scores["f_room"]),
        "f_util": float(raw_scores["f_util"]),
        "f_build": float(raw_scores["f_build"]),
        "Daily_Var": float(d_var),
        "Util_Avg": float(u_avg),
        "Interval_Rate": float(i_rate),
        "Build_Conc_Rate": float(b_rate)
    }


# =========================
# 主程序
# =========================
def main():
    print("=== 本地增强测试开始 ===")
    print(f"实例: {TEST_INSTANCE}")
    print(f"权重方案数: {len(WEIGHT_SCHEMES)}")
    print(f"每个方案运行次数: {N_RUNS}")

    for scheme_name, weights in WEIGHT_SCHEMES.items():
        validate_weights(weights)
        print(f"  - {scheme_name}: {weights}")

    # 1. 实例读取
    data = load_cached_instance(TEST_INSTANCE)
    print("✅ 实例缓存读取成功")

    # 2. 参数读取
    best_params = load_best_params_by_type(TEST_CONFIG["type"])
    print(f"✅ 参数读取成功: Pc={best_params['crossover_rate']}, Pm={best_params['mutation_rate']}")

    # 3. 输出目录
    result_dir = os.path.join("results", "weight_test_local_2x2")
    os.makedirs(result_dir, exist_ok=True)

    summary_csv = os.path.join(result_dir, "summary.csv")
    detail_json = os.path.join(result_dir, "detail_results.json")

    # 4. 结果容器
    summary_rows = []
    detail_results = {}

    # 5. 固定算法配置
    base_config = {
        **OURS_FLAGS,
        "crossover_rate": best_params["crossover_rate"],
        "mutation_rate": best_params["mutation_rate"],
        "patience": TEST_CONFIG["patience"],
        "population_size": TEST_CONFIG["pop"]
    }

    # 6. 逐权重方案运行
    for scheme_name, weights in WEIGHT_SCHEMES.items():
        print(f"\n=== 开始方案: {scheme_name} ===")

        scheme_memory_file = os.path.join(
            result_dir,
            f"{sanitize_name(TEST_INSTANCE)}_{sanitize_name(scheme_name)}_memory.pth"
        )

        scheme_runs = []

        # 每个新方案开始前，清掉全局 memory，避免串味
        if os.path.exists(GLOBAL_WEIGHTS_FILE):
            os.remove(GLOBAL_WEIGHTS_FILE)

        # 若已有该方案 memory，则加载；否则冷启动
        if os.path.exists(scheme_memory_file):
            shutil.copy2(scheme_memory_file, GLOBAL_WEIGHTS_FILE)
            print(f"✅ 已加载该方案历史 memory: {scheme_memory_file}")
        else:
            print("🆕 该方案按冷启动开始")

        for run_id in range(1, N_RUNS + 1):
            print(f"\n  -> Run {run_id}/{N_RUNS}")
            start_t = time.time()

            evaluator = FitnessEvaluator(weights=deepcopy(weights))
            evaluator.set_data(data)

            ga = GeneticAlgorithm(data, evaluator, deepcopy(base_config))

            # 显式沿用您项目里的 brain 逻辑
            if hasattr(ga, "rl_controller") and ga.rl_controller:
                try:
                    ga.rl_controller.load_brain(GLOBAL_WEIGHTS_FILE)
                    print("     ✅ RL brain 已加载")
                except Exception as e:
                    print(f"     ⚠️ RL brain 加载失败，但继续运行: {e}")

            ret = ga.run(TEST_CONFIG["gen"], TEST_CONFIG["pop"], run_id=run_id)

            if len(ret) == 4:
                best_ind, history, rl_logs, best_stats = ret
            else:
                best_ind, history, rl_logs = ret
                best_stats = {}

            if hasattr(ga, "rl_controller") and ga.rl_controller:
                try:
                    ga.rl_controller.save_brain(GLOBAL_WEIGHTS_FILE)
                    shutil.copy2(GLOBAL_WEIGHTS_FILE, scheme_memory_file)
                    print("     ✅ RL brain 已保存并回写方案 memory")
                except Exception as e:
                    print(f"     ⚠️ RL brain 保存失败: {e}")

            dur = time.time() - start_t
            final_metrics = compute_final_metrics(evaluator, best_ind, history)

            print(f"     Best Fitness: {final_metrics['Best_Fitness']:.2f}")
            print(f"     Violations: {final_metrics['Violations']}")
            print(f"     Soft Score: {final_metrics['Soft_Score_Sum']:.2f}")
            print(f"     Time: {dur:.2f}s")
            print(f"     RL Logs: {len(rl_logs)}")

            summary_rows.append({
                "Instance": TEST_INSTANCE,
                "Algorithm": OURS_NAME,
                "Weight_Scheme": scheme_name,
                "Run": run_id,
                "Generations": len(history),
                "Best_Fitness": final_metrics["Best_Fitness"],
                "Avg_Fitness": final_metrics["Avg_Fitness"],
                "Violations": final_metrics["Violations"],
                "Soft_Score_Sum": final_metrics["Soft_Score_Sum"],
                "f_daily": final_metrics["f_daily"],
                "f_interval": final_metrics["f_interval"],
                "f_room": final_metrics["f_room"],
                "f_util": final_metrics["f_util"],
                "f_build": final_metrics["f_build"],
                "Time_s": dur,
                "Daily_Var": final_metrics["Daily_Var"],
                "Util_Avg": final_metrics["Util_Avg"],
                "Interval_Rate": final_metrics["Interval_Rate"],
                "Build_Conc_Rate": final_metrics["Build_Conc_Rate"],
                "RL_Log_Count": len(rl_logs)
            })

            scheme_runs.append({
                "run_id": run_id,
                "duration_s": dur,
                "history_len": len(history),
                "rl_log_len": len(rl_logs),
                "best_fitness": final_metrics["Best_Fitness"],
                "violations": final_metrics["Violations"],
                "best_stats": best_stats
            })

        detail_results[scheme_name] = {
            "weights": weights,
            "runs": scheme_runs,
            "memory_file": scheme_memory_file
        }

    # 7. 写 summary.csv
    fieldnames = [
        "Instance", "Algorithm", "Weight_Scheme", "Run", "Generations",
        "Best_Fitness", "Avg_Fitness", "Violations", "Soft_Score_Sum",
        "f_daily", "f_interval", "f_room", "f_util", "f_build",
        "Time_s", "Daily_Var", "Util_Avg", "Interval_Rate", "Build_Conc_Rate",
        "RL_Log_Count"
    ]

    with open(summary_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    # 8. 写 detail_results.json
    with open(detail_json, "w", encoding="utf-8") as f:
        json.dump(detail_results, f, indent=4, ensure_ascii=False)

    print("\n=== 测试完成 ===")
    print(f"✅ summary.csv 已保存: {summary_csv}")
    print(f"✅ detail_results.json 已保存: {detail_json}")

    # 9. 打印一个简单对比摘要
    print("\n=== 简要结果对比 ===")
    for scheme_name in WEIGHT_SCHEMES.keys():
        rows = [r for r in summary_rows if r["Weight_Scheme"] == scheme_name]
        if rows:
            mean_fit = sum(r["Best_Fitness"] for r in rows) / len(rows)
            mean_vio = sum(r["Violations"] for r in rows) / len(rows)
            mean_time = sum(r["Time_s"] for r in rows) / len(rows)
            print(
                f"{scheme_name}: "
                f"Mean Best Fitness={mean_fit:.2f}, "
                f"Mean Violations={mean_vio:.2f}, "
                f"Mean Time={mean_time:.2f}s"
            )

    print("=== 本地增强测试结束 ===")


if __name__ == "__main__":
    main()