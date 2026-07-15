# -*- coding: utf-8 -*-
"""
权重灵敏性分析专用脚本
文件名: run_weight_sensitivity.py

用途:
1. 只运行 Ours: DL-GA-TS-Block-Repair-TrueCDM
2. 以“权重方案”为唯一自变量进行灵敏性分析
3. 为每个权重方案分配独立的 DRL memory 文件，避免跨方案污染
4. 保留 summary/history/rl_trace/report_metrics/best_schedules 输出格式

说明:
- 本脚本默认使用缓存实例: data/cache/{inst_name}_cache.pkl
- 本脚本默认只跑 3 个代表性实例: S2_Small, M2_Medium, L1_Large
- 若需扩展，可修改 INSTANCES_CONFIG
- 若需严格冷启动(每次run都不继承同方案 memory)，将 PERSIST_MEMORY_WITHIN_SCHEME 改为 False
"""

# -*- coding: utf-8 -*-
"""
权重灵敏性分析专用脚本（并行安全版）
文件名: run_weight_sensitivity.py

功能:
1. 只运行 Ours: DL-GA-TS-Block-Repair-TrueCDM
2. 以“权重方案”为唯一自变量进行灵敏性分析
3. 支持 local / server 两种运行模式
4. server 模式下按 (Instance, Weight_Scheme) 并行
5. 每个并行任务使用独立 workspace，避免 drl_model_memory.pth 并发覆盖
6. 每个权重方案保留自己的 memory 文件，避免跨方案污染

使用建议:
- 本地简单测试:
    python run_weight_sensitivity.py
- 服务器并行:
    python run_weight_sensitivity.py --mode server --max-workers 8 --n-runs 5
"""

import os
import csv
import json
import time
import shutil
import pickle
import logging
import traceback
import argparse
from copy import deepcopy
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

from evaluation.fitness import FitnessEvaluator
from algorithms.genetic_algorithm import GeneticAlgorithm


# =========================
# 全局常量
# =========================
OURS_NAME = "DL-GA-TS-Block-Repair-TrueCDM"
GLOBAL_WEIGHTS_FILE = "drl_model_memory.pth"

# 您给的新正式参数
INSTANCES_CONFIG = {
    'S1_Small': {'type': 'Small', 'gen': 500, 'pop': 100, 'patience': 70},
    'S2_Small': {'type': 'Small', 'gen': 600, 'pop': 100, 'patience': 70},
    'S3_Small': {'type': 'Small', 'gen': 700, 'pop': 100, 'patience': 70},
    'S4_Small': {'type': 'Small', 'gen': 800, 'pop': 100, 'patience': 80},
    'M1_Medium': {'type': 'Medium', 'gen': 1200, 'pop': 150, 'patience': 100},
    'M2_Medium': {'type': 'Medium', 'gen': 1500, 'pop': 150, 'patience': 120},
    'M3_Medium': {'type': 'Medium', 'gen': 2000, 'pop': 150, 'patience': 150},
    'L1_Large': {'type': 'Large', 'gen': 3000, 'pop': 200, 'patience': 200}
}

# 基准 + 单项强化
WEIGHT_SCHEMES = {
    "B0_Base": {
        "f_daily": 0.20,
        "f_interval": 0.20,
        "f_room": 0.10,
        "f_util": 0.30,
        "f_build": 0.20,
    },
    "B1_Daily": {
        "f_daily": 0.30,
        "f_interval": 0.15,
        "f_room": 0.10,
        "f_util": 0.25,
        "f_build": 0.20,
    },
    "B2_Interval": {
        "f_daily": 0.15,
        "f_interval": 0.30,
        "f_room": 0.10,
        "f_util": 0.25,
        "f_build": 0.20,
    },
    "B3_Room": {
        "f_daily": 0.15,
        "f_interval": 0.15,
        "f_room": 0.25,
        "f_util": 0.25,
        "f_build": 0.20,
    },
    "B4_Util": {
        "f_daily": 0.15,
        "f_interval": 0.15,
        "f_room": 0.10,
        "f_util": 0.40,
        "f_build": 0.20,
    },
    "B5_Build": {
        "f_daily": 0.15,
        "f_interval": 0.15,
        "f_room": 0.10,
        "f_util": 0.25,
        "f_build": 0.35,
    },
}


# =========================
# 工具函数
# =========================
def sanitize_name(name: str) -> str:
    out = []
    for ch in name:
        if ch.isalnum() or ch in ("-", "_"):
            out.append(ch)
        else:
            out.append("_")
    return "".join(out)


def validate_weight_scheme(weight_dict: dict):
    required_keys = {"f_daily", "f_interval", "f_room", "f_util", "f_build"}
    if set(weight_dict.keys()) != required_keys:
        raise ValueError(f"Weight keys mismatch: {weight_dict.keys()}")
    s = sum(float(v) for v in weight_dict.values())
    if abs(s - 1.0) > 1e-8:
        raise ValueError(f"Weight sum must be 1.0, got {s}")


def setup_logger(log_path: str, logger_name: str):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S"
    )

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


def save_instance_metadata(inst_path: str, inst_name: str, data):
    college_ids = set()
    try:
        for c in data.classes.values():
            if hasattr(c, "college_id") and c.college_id:
                college_ids.add(str(c.college_id))
    except Exception:
        pass

    meta = {
        "Instance": inst_name,
        "College_Count": len(college_ids),
        "College_IDs": "|".join(sorted(list(college_ids))),
        "Tasks": data.num_tasks,
        "Teachers": data.num_teachers,
        "Classrooms": data.num_classrooms,
        "Classes": data.num_classes,
    }
    pd.DataFrame([meta]).to_csv(
        os.path.join(inst_path, "metadata.csv"),
        index=False,
        encoding="utf-8-sig"
    )


def load_best_params_by_type(project_root: str, inst_type: str):
    suffix = "small" if inst_type == "Small" else "medium"
    if inst_type == "Large":
        suffix = "medium"

    json_path = os.path.join(project_root, "results", f"best_params_{suffix}.json")
    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass

    return {"crossover_rate": 0.8, "mutation_rate": 0.1}


def load_cached_instance(project_root: str, inst_name: str):
    cache_file = os.path.join(project_root, "data", "cache", f"{inst_name}_cache.pkl")
    if not os.path.exists(cache_file):
        raise FileNotFoundError(f"Cache file not found: {cache_file}")
    with open(cache_file, "rb") as f:
        return pickle.load(f)


def update_best_schedules_json(json_path: str, scheme_name: str, fitness: float, schedule_genes: dict):
    scheme_order = list(WEIGHT_SCHEMES.keys())
    data = {}

    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    data = json.loads(content)
        except Exception:
            data = {}

    data[scheme_name] = {
        "fitness": float(fitness),
        "schedule": {str(k): int(v) for k, v in schedule_genes.items()}
    }

    ordered = {}
    for key in scheme_order:
        if key in data:
            ordered[key] = data[key]
    for key in data:
        if key not in ordered:
            ordered[key] = data[key]

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(ordered, f, indent=4, ensure_ascii=False)


def compute_final_metrics(evaluator: FitnessEvaluator, best_ind, history: list, score_cols: list):
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
            "Build_Conc_Rate": 0.0,
        }

    s_sum_raw = last_rec.get("soft_score_sum", 0)
    s_sum = sum(s_sum_raw.values()) if isinstance(s_sum_raw, dict) else float(s_sum_raw)

    if "f_daily" in last_rec:
        raw_scores = {k: float(last_rec.get(k, 0)) for k in score_cols}
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
        "Build_Conc_Rate": float(b_rate),
    }


def build_fixed_config(best_params: dict, cfg: dict):
    return {
        "use_heuristic_init": True,
        "use_rl_adaptive": True,
        "use_block_operators": True,
        "use_repair": True,
        "use_cdm": True,
        "use_tabu": True,
        "elite_count": 2,
        "metric_interval": 50,
        "patience": cfg["patience"],
        "population_size": cfg["pop"],
        "tabu_steps": 10,
        "crossover_rate": best_params["crossover_rate"],
        "mutation_rate": best_params["mutation_rate"],
    }


# =========================
# 单个 (实例, 权重方案) 任务
# =========================
def run_one_scheme_task(
    project_root: str,
    res_dir: str,
    inst_name: str,
    cfg: dict,
    scheme_name: str,
    weights: dict,
    n_runs: int,
    persist_memory_within_scheme: bool,
):
    """
    单个并行任务：
    固定一个实例 + 一个权重方案，内部串行跑 N_RUNS 次
    """
    score_cols = ["f_daily", "f_interval", "f_room", "f_util", "f_build"]

    task_tag = f"{inst_name}__{scheme_name}"
    task_safe = sanitize_name(task_tag)

    # 独立 workspace：关键！
    workspace_dir = os.path.join(res_dir, "_workspace", task_safe)
    os.makedirs(workspace_dir, exist_ok=True)

    # 切到独立目录，使 RLController 默认使用的 drl_model_memory.pth 互不冲突
    old_cwd = os.getcwd()
    os.chdir(workspace_dir)

    try:
        data = load_cached_instance(project_root, inst_name)
        best_params = load_best_params_by_type(project_root, cfg["type"])
        fixed_config = build_fixed_config(best_params, cfg)

        inst_path = os.path.join(res_dir, inst_name)
        os.makedirs(inst_path, exist_ok=True)

        # 方案专属 memory 文件，放到正式结果目录里，便于保留
        scheme_memory_dir = os.path.join(inst_path, "drl_memory")
        os.makedirs(scheme_memory_dir, exist_ok=True)
        scheme_memory_file = os.path.join(
            scheme_memory_dir,
            f"{sanitize_name(scheme_name)}_memory.pth"
        )

        # 本任务自己的日志
        task_log_path = os.path.join(inst_path, f"log_{sanitize_name(scheme_name)}.txt")
        logger = setup_logger(task_log_path, logger_name=f"logger_{task_safe}")

        logger.info(f"▶ Start Task: {inst_name} | {scheme_name}")
        logger.info(f"  Weights = {weights}")
        logger.info(f"  Config  = {cfg}")
        logger.info(f"  BestParams = {best_params}")

        # history / rl / raw 汇总在任务内先写临时 csv，主进程再统一汇总
        task_summary_rows = []
        task_history_rows = []
        task_rl_rows = []
        task_excel_rows = []

        scheme_best_fit = -float("inf")
        scheme_best_genes = None

        # 如果不希望同方案共享 memory，则整个任务开始前就删掉旧 memory
        if not persist_memory_within_scheme and os.path.exists(scheme_memory_file):
            os.remove(scheme_memory_file)

        for run_id in range(1, n_runs + 1):
            start_t = time.time()

            try:
                # 处理 memory
                # 每个任务都在自己的 workspace 下，GLOBAL_WEIGHTS_FILE 不会撞别的进程
                if os.path.exists(GLOBAL_WEIGHTS_FILE):
                    os.remove(GLOBAL_WEIGHTS_FILE)

                if persist_memory_within_scheme and os.path.exists(scheme_memory_file):
                    shutil.copy2(scheme_memory_file, GLOBAL_WEIGHTS_FILE)

                evaluator = FitnessEvaluator(weights=deepcopy(weights))
                evaluator.set_data(data)

                ga = GeneticAlgorithm(data, evaluator, deepcopy(fixed_config))

                # 若 rl_controller 存在，则显式 load/save，延续您新版 run_ablation 的逻辑
                if hasattr(ga, "rl_controller") and ga.rl_controller:
                    ga.rl_controller.load_brain(GLOBAL_WEIGHTS_FILE)

                ret = ga.run(cfg["gen"], cfg["pop"], run_id)
                if len(ret) == 4:
                    best_ind, history, rl_logs, _ = ret
                else:
                    best_ind, history, rl_logs = ret

                if hasattr(ga, "rl_controller") and ga.rl_controller:
                    ga.rl_controller.save_brain(GLOBAL_WEIGHTS_FILE)

                # 回写该方案自己的 memory
                if os.path.exists(GLOBAL_WEIGHTS_FILE):
                    shutil.copy2(GLOBAL_WEIGHTS_FILE, scheme_memory_file)

                final_gen = len(history)
                dur = time.time() - start_t

                if best_ind.fitness > scheme_best_fit:
                    scheme_best_fit = best_ind.fitness
                    scheme_best_genes = best_ind.genes.copy()

                final_metrics = compute_final_metrics(evaluator, best_ind, history, score_cols)

                # summary row
                task_summary_rows.append({
                    "Instance": inst_name,
                    "Algorithm": OURS_NAME,
                    "Weight_Scheme": scheme_name,
                    "Run": run_id,
                    "Generations": final_gen,
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
                })

                # history rows
                for h in history:
                    h_s_raw = h.get("soft_score_sum", 0)
                    h_s = sum(h_s_raw.values()) if isinstance(h_s_raw, dict) else float(h_s_raw)
                    task_history_rows.append({
                        "Instance": inst_name,
                        "Algorithm": OURS_NAME,
                        "Weight_Scheme": scheme_name,
                        "Run": run_id,
                        "Gen": h.get("gen", 0),
                        "Best_Fitness": h.get("best_fitness", h.get("best_fit", 0)),
                        "Avg_Fitness": h.get("avg_fitness", h.get("avg_fit", 0)),
                        "Violations": h.get("hard_violations", h.get("violations", 0)),
                        "Soft_Score_Sum": h_s,
                        "f_daily": h.get("f_daily", 0),
                        "f_interval": h.get("f_interval", 0),
                        "f_room": h.get("f_room", 0),
                        "f_util": h.get("f_util", 0),
                        "f_build": h.get("f_build", 0),
                        "Time_s": h.get("time_s", 0),
                        "Daily_Var": h.get("Daily_Var", 0),
                        "Util_Avg": h.get("Util_Avg", 0),
                        "Interval_Rate": h.get("Interval_Rate", 0),
                        "Build_Conc_Rate": h.get("Build_Conc_Rate", 0),
                    })

                # rl rows
                for log in rl_logs:
                    task_rl_rows.append({
                        "Instance": inst_name,
                        "Weight_Scheme": scheme_name,
                        "Run": run_id,
                        "Gen": log.get("gen", 0),
                        "Action": log.get("action", log.get("action_id", "")),
                        "Reward": float(log.get("reward", 0)),
                        "Loss": "" if log.get("loss") is None else log.get("loss"),
                        "Mutation": float(log.get("mutation", 0)),
                        "Crossover": float(log.get("crossover", 0)),
                        "Q_Values": str(log.get("q_values", "")),
                    })

                task_excel_rows.append(dict(task_summary_rows[-1]))

                logger.info(
                    f"  ✅ Run {run_id}/{n_runs} | "
                    f"Best={final_metrics['Best_Fitness']:.2f} | "
                    f"Soft={final_metrics['Soft_Score_Sum']:.2f} | "
                    f"Vio={final_metrics['Violations']} | "
                    f"Time={dur:.2f}s"
                )

            except Exception as e:
                logger.error(f"  ❌ Run Failed ({inst_name} | {scheme_name} | Run {run_id}): {e}")
                logger.error(traceback.format_exc())

        # 最优课表
        if scheme_best_genes is not None:
            update_best_schedules_json(
                os.path.join(inst_path, "best_schedules.json"),
                scheme_name,
                scheme_best_fit,
                scheme_best_genes
            )

        # 任务结果落盘为中间文件，主进程再汇总
        partial_dir = os.path.join(res_dir, "_partials")
        os.makedirs(partial_dir, exist_ok=True)

        pd.DataFrame(task_summary_rows).to_csv(
            os.path.join(partial_dir, f"{task_safe}_summary.csv"),
            index=False,
            encoding="utf-8-sig"
        )
        pd.DataFrame(task_history_rows).to_csv(
            os.path.join(partial_dir, f"{task_safe}_history.csv"),
            index=False,
            encoding="utf-8-sig"
        )
        pd.DataFrame(task_rl_rows).to_csv(
            os.path.join(partial_dir, f"{task_safe}_rl.csv"),
            index=False,
            encoding="utf-8-sig"
        )
        pd.DataFrame(task_excel_rows).to_csv(
            os.path.join(partial_dir, f"{task_safe}_excel.csv"),
            index=False,
            encoding="utf-8-sig"
        )

        return {
            "task": task_tag,
            "status": "ok",
            "summary_rows": len(task_summary_rows),
            "history_rows": len(task_history_rows),
            "rl_rows": len(task_rl_rows),
        }

    finally:
        os.chdir(old_cwd)


# =========================
# 汇总输出
# =========================
def merge_partial_results(res_dir: str):
    partial_dir = os.path.join(res_dir, "_partials")
    os.makedirs(partial_dir, exist_ok=True)

    summary_files = []
    history_files = []
    rl_files = []
    excel_files = []

    for fn in os.listdir(partial_dir):
        p = os.path.join(partial_dir, fn)
        if fn.endswith("_summary.csv"):
            summary_files.append(p)
        elif fn.endswith("_history.csv"):
            history_files.append(p)
        elif fn.endswith("_rl.csv"):
            rl_files.append(p)
        elif fn.endswith("_excel.csv"):
            excel_files.append(p)

    def load_concat(files):
        dfs = []
        for f in sorted(files):
            try:
                df = pd.read_csv(f)
                if not df.empty:
                    dfs.append(df)
            except Exception:
                pass
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    df_sum = load_concat(summary_files)
    df_hist = load_concat(history_files)
    df_rl = load_concat(rl_files)
    df_excel = load_concat(excel_files)

    # 总表
    if not df_sum.empty:
        df_sum.to_csv(os.path.join(res_dir, "summary.csv"), index=False, encoding="utf-8-sig")

    if not df_hist.empty:
        df_hist.to_csv(os.path.join(res_dir, "history1.csv"), index=False, encoding="utf-8-sig")

    if not df_rl.empty:
        df_rl.to_csv(os.path.join(res_dir, "rl_trace.csv"), index=False, encoding="utf-8-sig")

    # 每个实例单独 Excel
    if not df_excel.empty:
        for inst_name, df_inst in df_excel.groupby("Instance"):
            inst_path = os.path.join(res_dir, inst_name)
            os.makedirs(inst_path, exist_ok=True)

            stats_cols = [
                "Generations", "Best_Fitness", "Avg_Fitness", "Violations", "Soft_Score_Sum",
                "f_daily", "f_interval", "f_room", "f_util", "f_build",
                "Time_s", "Daily_Var", "Util_Avg", "Interval_Rate", "Build_Conc_Rate"
            ]

            summary_stats = df_inst.groupby("Weight_Scheme")[stats_cols].agg(["mean", "std"]).reset_index()

            new_cols = []
            for c in summary_stats.columns:
                if isinstance(c, tuple):
                    new_cols.append(f"{c[0]}_{c[1]}" if c[1] else c[0])
                else:
                    new_cols.append(c)
            summary_stats.columns = new_cols

            with pd.ExcelWriter(os.path.join(inst_path, "report_metrics.xlsx")) as writer:
                df_inst.to_excel(writer, sheet_name="Raw_Data", index=False)
                summary_stats.to_excel(writer, sheet_name="Summary", index=False)


# =========================
# 主程序
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["local", "server"], default="local")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--n-runs", type=int, default=3)  # 本地默认3次，服务器可传5
    parser.add_argument("--persist-memory-within-scheme", action="store_true")
    args = parser.parse_args()

    for _, weights in WEIGHT_SCHEMES.items():
        validate_weight_scheme(weights)

    project_root = os.path.abspath(os.getcwd())
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    res_dir = os.path.join(project_root, "results", f"weight_sensitivity_{timestamp}")
    os.makedirs(res_dir, exist_ok=True)

    main_logger = setup_logger(
        os.path.join(res_dir, "experiment.log"),
        logger_name="main_weight_sensitivity_logger"
    )

    main_logger.info("=== 🚀 权重灵敏性分析开始 ===")
    main_logger.info(f"Mode = {args.mode}")
    main_logger.info(f"N_RUNS = {args.n_runs}")
    main_logger.info(f"MaxWorkers = {args.max_workers}")
    main_logger.info(f"PERSIST_MEMORY_WITHIN_SCHEME = {args.persist_memory_within_scheme}")
    main_logger.info(f"Ours = {OURS_NAME}")

    # 先把各实例元数据写好
    for inst_name in INSTANCES_CONFIG.keys():
        try:
            data = load_cached_instance(project_root, inst_name)
            inst_path = os.path.join(res_dir, inst_name)
            os.makedirs(inst_path, exist_ok=True)
            save_instance_metadata(inst_path, inst_name, data)
        except Exception as e:
            main_logger.error(f"Metadata failed for {inst_name}: {e}")

    tasks = []
    for inst_name, cfg in INSTANCES_CONFIG.items():
        for scheme_name, weights in WEIGHT_SCHEMES.items():
            tasks.append((inst_name, cfg, scheme_name, weights))

    # local: 串行，便于调试
    if args.mode == "local":
        for inst_name, cfg, scheme_name, weights in tasks:
            try:
                ret = run_one_scheme_task(
                    project_root=project_root,
                    res_dir=res_dir,
                    inst_name=inst_name,
                    cfg=cfg,
                    scheme_name=scheme_name,
                    weights=weights,
                    n_runs=args.n_runs,
                    persist_memory_within_scheme=args.persist_memory_within_scheme,
                )
                main_logger.info(f"Done: {ret}")
            except Exception as e:
                main_logger.error(f"Task failed ({inst_name}, {scheme_name}): {e}")
                main_logger.error(traceback.format_exc())

    # server: 并行
    else:
        with ProcessPoolExecutor(max_workers=args.max_workers) as ex:
            futures = []
            for inst_name, cfg, scheme_name, weights in tasks:
                futures.append(
                    ex.submit(
                        run_one_scheme_task,
                        project_root,
                        res_dir,
                        inst_name,
                        cfg,
                        scheme_name,
                        weights,
                        args.n_runs,
                        args.persist_memory_within_scheme,
                    )
                )

            for fut in as_completed(futures):
                try:
                    ret = fut.result()
                    main_logger.info(f"Done: {ret}")
                except Exception as e:
                    main_logger.error(f"Parallel task failed: {e}")
                    main_logger.error(traceback.format_exc())

    merge_partial_results(res_dir)
    main_logger.info("=== ✅ 权重灵敏性分析全部完成 ===")


if __name__ == "__main__":
    main()