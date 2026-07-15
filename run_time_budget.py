"""
论文级同时间预算对比实验 (Strict Time Budget - 15核多进程狂暴版)
功能:
1. 强制所有算法在 300s 物理时间内运行。
2. 注入了完整的 DL-GA-TS-Block-Repair-TrueCDM (包含 RL自适应 与 Tabu增强)。
3. 利用 ProcessPoolExecutor 压榨 15 核 CPU，将 5 小时实验压缩至 20 分钟。
"""
import os
import sys

# 自动路径锚定
current_script_path = os.path.abspath(__file__)
project_root = os.path.dirname(current_script_path)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import time
import random
import pickle
import logging
import json
import numpy as np
import pandas as pd
import warnings
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

warnings.filterwarnings("ignore")

# ===== 导入工程模块 =====
try:
    from evaluation.fitness import FitnessEvaluator
    from algorithms.genetic_algorithm import GeneticAlgorithm
    from algorithms.cuckoo_search import CuckooSearchAlgorithm
    from algorithms.sun_wu_tpts import SunWuTabuSearch
except ImportError as e:
    print(f"❌ 导入失败: {e}\n请确保在项目根目录下运行，或检查模块路径。")
    sys.exit(1)

# =========================
# 核心实验配置
# =========================
TIME_LIMIT = 300  # 严格物理死线 (秒)
N_RUNS = 20  # 独立运行次数
MAX_WORKERS = 15  # 🚀 你的服务器核心数 (拉满 15 核)
INSTANCE_NAME = "L1_Large"
MAX_GENS = 1000000
ANYTIME_CHECKPOINTS = [30, 60, 120, 300]

ALGO_ORDER = ["Cuckoo-Search", "Sun-Wu-2023", "DL-GA-TS-Block-Repair-TrueCDM"]


# =========================
# 工具函数
# =========================
def setup_logger(res_dir):
    logger = logging.getLogger("TimeBudget")
    logger.setLevel(logging.INFO)
    if logger.hasHandlers(): logger.handlers.clear()
    formatter = logging.Formatter('%(message)s')

    fh = logging.FileHandler(os.path.join(res_dir, "experiment.log"), encoding='utf-8')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger


def now_s() -> float:
    return time.perf_counter()


def anytime_to_checkpoints(anytime_pairs, checkpoints):
    if not anytime_pairs: return {c: 0.0 for c in checkpoints}
    anytime_pairs = sorted(anytime_pairs, key=lambda x: x[0])
    result = {}
    idx, current_best = 0, anytime_pairs[0][1]
    for c in checkpoints:
        while idx < len(anytime_pairs) and anytime_pairs[idx][0] <= c:
            current_best = max(current_best, anytime_pairs[idx][1])
            idx += 1
        result[c] = float(current_best)
    return result


# ==========================================
# 🌟 修复版 GA 核心：纯净静默版 (适配多进程)
# ==========================================
class TimeLimitedGA(GeneticAlgorithm):
    # 去掉了 logger 参数，子进程专心算力全开，不处理 IO
    def run_with_time_limit(self, max_generations, pop_size, time_limit, algo_name="GA"):
        start_t = now_s()
        anytime = []

        self._init_population(pop_size)
        self._eval_population()
        self.best_chromosome = max(self.population, key=lambda x: x.fitness)

        if self.use_rl and self.rl_controller:
            self.rl_controller.reset_episode()

        for gen in range(max_generations):
            if now_s() - start_t >= time_limit: break

            current_avg = np.mean([c.fitness for c in self.population])

            if self.use_rl and self.rl_controller:
                state = {
                    'diversity': self._calculate_diversity(self.population),
                    'avg_fit': current_avg,
                    'best_fit': self.best_chromosome.fitness,
                    'progress': gen / max_generations
                }
                rl_info = self.rl_controller.get_action(state, training=False)

                if rl_info and rl_info.get('pc') is not None:
                    self.pc = max(0.0, min(1.0, float(rl_info['pc'])))
                    self.crossover.rate = self.pc
                if rl_info and rl_info.get('pm') is not None:
                    self.pm = max(0.0, min(1.0, float(rl_info['pm'])))
                    if hasattr(self, 'mutation') and self.mutation is not None:
                        self.mutation.rate = self.pm

            self._evolve_population(pop_size)

            current_gen_best = max(self.population, key=lambda x: x.fitness)
            if current_gen_best.fitness > self.best_chromosome.fitness:
                self.best_chromosome = current_gen_best.clone()

            if self.use_tabu and self.best_chromosome.fitness > 0:
                if self.use_block:
                    improved_best = self.best_chromosome.clone()
                    improved_best = self.tabu_search.run(improved_best, max_steps=5)
                    if improved_best.fitness > self.best_chromosome.fitness:
                        self.best_chromosome = improved_best
                else:
                    need_repair = (self.best_chromosome.fitness < 0)
                    improved_best = self.tabu_search.run(self.best_chromosome, self.config.get('tabu_steps', 10),
                                                         need_repair)
                    if improved_best.fitness > self.best_chromosome.fitness:
                        self.best_chromosome = improved_best

            elapsed = now_s() - start_t
            anytime.append((elapsed, float(self.best_chromosome.fitness)))

            if elapsed >= time_limit: break

        final_elapsed = now_s() - start_t
        return self.best_chromosome, gen, final_elapsed, anytime


# =========================
# 🌟 多核子进程 Worker 任务包装器
# =========================
def worker_task(args):
    """
    独立的工作进程函数：接收参数，并在进程内部独立实例化 Evaluator 以避免 Pickle 锁死。
    """
    algo_name, data, seed, config, run_idx = args

    # 🌟 关键：在子进程中独立实例化评价器，彻底避免多进程内存交织
    evaluator = FitnessEvaluator()
    evaluator.set_data(data)

    random.seed(seed)
    np.random.seed(seed)

    if algo_name == "Cuckoo-Search":
        algo = CuckooSearchAlgorithm(data, evaluator)
        best, gens, elapsed, anytime = algo.run_with_time_limit(MAX_GENS, 50, TIME_LIMIT, algo_name)
    elif algo_name == "Sun-Wu-2023":
        algo = SunWuTabuSearch(data, evaluator)
        best, gens, elapsed, anytime = algo.run_with_time_limit(MAX_GENS, 1, TIME_LIMIT, algo_name)
    else:
        algo = TimeLimitedGA(data, evaluator, config)
        best, gens, elapsed, anytime = algo.run_with_time_limit(MAX_GENS, 50, TIME_LIMIT, algo_name)

    return float(best.fitness), int(gens), float(elapsed), anytime


# =========================
# 主函数
# =========================
def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    res_dir = os.path.join(project_root, "results", f"TimeBudget_{timestamp}")
    os.makedirs(res_dir, exist_ok=True)
    logger = setup_logger(res_dir)

    logger.info("=" * 72)
    logger.info(f"🚀 启动 15核狂暴版 | 算例: {INSTANCE_NAME} | 预算: {TIME_LIMIT}s | 总任务: {len(ALGO_ORDER) * N_RUNS}")
    logger.info("=" * 72)

    # 1. 加载数据
    cache_file = os.path.join(project_root, "data", "cache", f"{INSTANCE_NAME}_cache.pkl")
    try:
        with open(cache_file, "rb") as f:
            data = pickle.load(f)
    except Exception as e:
        logger.error(f"❌ 数据加载失败: {e}")
        return

    # 2. 准备参数
    best_params = {'crossover_rate': 0.8, 'mutation_rate': 0.1}
    param_path = os.path.join(project_root, "results", "best_params_medium.json")
    if os.path.exists(param_path):
        with open(param_path, 'r') as f: best_params.update(json.load(f))

    ours_config = {
        **best_params,
        "use_heuristic_init": True, "use_block_operators": True, "use_repair": True,
        "use_rl_adaptive": True, "use_tabu": True, "use_cdm": True
    }

    # 3. 构建多进程任务队列
    tasks = []
    base_seed = 2026
    for algo_name in ALGO_ORDER:
        config = ours_config if algo_name == "DL-GA-TS-Block-Repair-TrueCDM" else None
        for r in range(N_RUNS):
            tasks.append((algo_name, data, base_seed + r, config, r + 1))

    rows, anytime_rows = [], []
    total_tasks = len(tasks)
    completed_tasks = 0

    logger.info(f"⚡ 引擎点火完毕... 正在向 {MAX_WORKERS} 个核心分发计算任务，请稍候...")

    # 4. 🚀 开启进程池，压榨算力
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 提交所有任务
        future_to_task = {executor.submit(worker_task, t): t for t in tasks}

        # as_completed 会在任意一个任务完成时立刻返回，完美适合做进度条
        for future in as_completed(future_to_task):
            task_args = future_to_task[future]
            algo_name, _, current_seed, _, run_idx = task_args

            try:
                fit, gens, sec, trace = future.result()
                completed_tasks += 1

                rows.append(
                    {"Algorithm": algo_name, "Run": run_idx, "Seed": current_seed, "Fitness": fit, "Generations": gens,
                     "Elapsed(s)": sec})
                for t, f in trace:
                    anytime_rows.append(
                        {"Algorithm": algo_name, "Run": run_idx, "Seed": current_seed, "Time(s)": t, "BestFitness": f})

                # 主进程统一优雅输出，绝不串行乱码
                logger.info(
                    f"   [{completed_tasks:02d}/{total_tasks}] ✅ {algo_name:<30} (Run {run_idx:02d}) | Fit: {fit:8.2f} | Gens: {gens:<6} | Time: {sec:5.1f}s")

            except Exception as exc:
                logger.error(f"   [{completed_tasks:02d}/{total_tasks}] ❌ {algo_name} (Run {run_idx}) 任务崩溃: {exc}")

    # 5. 数据统计与生成 Excel
    logger.info("\n📊 正在汇总所有核心数据并生成 Excel...")
    df = pd.DataFrame(rows)
    df_any = pd.DataFrame(anytime_rows)

    summary = df.groupby("Algorithm").agg(
        Mean_Fitness=("Fitness", "mean"), Std_Fitness=("Fitness", "std"), Best_Fitness=("Fitness", "max"),
        Mean_Generations=("Generations", "mean"), Mean_Elapsed=("Elapsed(s)", "mean")
    ).reset_index().sort_values(by="Mean_Fitness", ascending=False)

    cp_list = []
    for algo in ALGO_ORDER:
        algo_data = df_any[df_any["Algorithm"] == algo]
        per_run_cp = []
        for r in range(1, N_RUNS + 1):
            run_trace = algo_data[algo_data["Run"] == r][["Time(s)", "BestFitness"]].values.tolist()
            per_run_cp.append(anytime_to_checkpoints(run_trace, ANYTIME_CHECKPOINTS))

        row = {"Algorithm": algo}
        for c in ANYTIME_CHECKPOINTS:
            row[f"T{c}_Mean"] = float(np.mean([d[c] for d in per_run_cp])) if per_run_cp else 0.0
            row[f"T{c}_Std"] = float(np.std([d[c] for d in per_run_cp], ddof=1)) if len(per_run_cp) > 1 else 0.0
        cp_list.append(row)

    df_cp = pd.DataFrame(cp_list)
    df_cp = df_cp.set_index('Algorithm').loc[summary['Algorithm']].reset_index()

    out_xlsx = os.path.join(res_dir, f"paper_time_budget_{TIME_LIMIT}s_{INSTANCE_NAME}_runs{N_RUNS}.xlsx")
    with pd.ExcelWriter(out_xlsx, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name="raw_results", index=False)
        summary.to_excel(writer, sheet_name="summary", index=False)
        df_any.to_excel(writer, sheet_name="anytime_trace", index=False)
        df_cp.to_excel(writer, sheet_name="anytime_checkpoints", index=False)

    logger.info("\n" + "=" * 72)
    logger.info(f"🏆 最终汇总结果 (Runs={N_RUNS}, Budget={TIME_LIMIT}s)")
    logger.info("=" * 72)
    logger.info("\n" + summary.to_string(index=False))

    logger.info("\n" + "=" * 72)
    logger.info("⏱️ Anytime 检查点进展 (均值 ± 标准差)")
    logger.info("=" * 72)
    logger.info("\n" + df_cp.to_string(index=False))

    logger.info(f"\n✅ 详细数据已成功保存至: {out_xlsx}")


if __name__ == "__main__":
    # 多核必须在这个 if 语句的保护伞下运行，已经为你写好了
    main()