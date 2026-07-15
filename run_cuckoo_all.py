import os
import csv
import json
import time
import logging
import pandas as pd
import pickle
from datetime import datetime
from evaluation.fitness import FitnessEvaluator
from algorithms.cuckoo_search import CuckooSearchAlgorithm

# === 核心参数 ===
ALGO_NAME = "Cuckoo-Search"


INSTANCES = ["S1_Small", "S2_Small", "S3_Small", "S4_Small", "M1_Medium", "M2_Medium", "M3_Medium", "L1_Large"]

GENS = 10000  # 本地测试可先改小
POP_SIZE = 50
N_RUNS = 10  # 本地测试可先改小


# 恢复您的日志系统
def setup_logger(res_dir):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if logger.hasHandlers(): logger.handlers.clear()
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
    fh = logging.FileHandler(os.path.join(res_dir, "experiment.log"), encoding='utf-8')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger


# 恢复您的 JSON 写入系统
def update_best_schedules_json(json_path, algo_name, fitness, schedule_genes):
    data = {}
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                content = f.read()
                if content.strip(): data = json.loads(content)
        except:
            data = {}
    serializable_genes = {str(k): int(schedule_genes[k]) for k in sorted(schedule_genes.keys(), key=lambda x: int(x))}
    data[algo_name] = {"fitness": float(fitness), "schedule": serializable_genes}
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    res_dir = f"results/cuckoo_only_{timestamp}"
    os.makedirs(res_dir, exist_ok=True)
    logger = setup_logger(res_dir)

    score_cols = ['f_daily', 'f_interval', 'f_room', 'f_util', 'f_build']
    common_cols = ['Instance', 'Algorithm', 'Run', 'Generations', 'Best_Fitness', 'Avg_Fitness',
                   'Violations', 'Soft_Score_Sum'] + score_cols + \
                  ['Time_s', 'Daily_Var', 'Util_Avg', 'Interval_Rate', 'Build_Conc_Rate']
    hist_cols = ['Instance', 'Algorithm', 'Run', 'Gen'] + common_cols[4:]

    f_sum = open(os.path.join(res_dir, "summary.csv"), 'w', newline='', encoding='utf-8-sig')
    csv_sum = csv.writer(f_sum);
    csv_sum.writerow(common_cols)

    f_hist = open(os.path.join(res_dir, "history.csv"), 'w', newline='', encoding='utf-8-sig')
    csv_hist = csv.writer(f_hist);
    csv_hist.writerow(hist_cols)

    evaluator = FitnessEvaluator()

    for inst in INSTANCES:
        logger.info(f"🚀 Processing: {inst} ...")

        cache_file = os.path.join("data", "cache", f"{inst}_cache.pkl")
        if not os.path.exists(cache_file):
            logger.error(f"❌ 找不到缓存: {cache_file}");
            continue

        with open(cache_file, "rb") as f:
            data = pickle.load(f)

        evaluator.set_data(data)
        inst_path = os.path.join(res_dir, inst)
        os.makedirs(inst_path, exist_ok=True)

        instance_excel_data = []
        algo_best_fit = -float('inf')
        algo_best_genes = None

        for r in range(1, N_RUNS + 1):
            start = time.time()
            try:
                algo = CuckooSearchAlgorithm(data, evaluator)
                best_ind, final_gen, dur, trace = algo.run_with_time_limit(max_generations=GENS, pop_size=POP_SIZE,
                                                                           time_limit=float('inf'))

                # 记录 JSON 用的最好基因
                if best_ind.fitness > algo_best_fit:
                    algo_best_fit = best_ind.fitness
                    algo_best_genes = best_ind.genes.copy()

                vio = evaluator.constraint_checker.check_all(best_ind.genes)['total']
                if vio > 0:
                    raw = {k: 0.0 for k in score_cols};
                    s_sum = 0.0
                    mets = {k: 0.0 for k in ['Daily_Var', 'Util_Avg', 'Interval_Rate', 'Build_Conc_Rate']}
                else:
                    raw = evaluator._calculate_raw_scores(best_ind.genes)
                    s_sum = sum(raw.values())
                    mets = evaluator.calculate_quality_metrics(best_ind.genes)

                row = [inst, ALGO_NAME, r, final_gen, f"{best_ind.fitness:.2f}", f"{best_ind.fitness:.2f}", vio,
                       f"{s_sum:.2f}"]
                row.extend([f"{raw.get(k, 0):.2f}" for k in score_cols])
                row.extend([f"{dur:.2f}", f"{mets['Daily_Var']:.4f}", f"{mets['Util_Avg']:.4f}",
                            f"{mets['Interval_Rate']:.4f}", f"{mets['Build_Conc_Rate']:.4f}"])
                csv_sum.writerow(row)
                f_sum.flush()

                step_size = max(1, len(trace) // 200)
                for idx, (t_s, fit) in enumerate(trace):
                    if idx % step_size == 0 or idx == len(trace) - 1:
                        h_row = [inst, ALGO_NAME, r, idx, f"{fit:.2f}", f"{fit:.2f}", vio, f"{s_sum:.2f}"]
                        h_row.extend([f"{raw.get(k, 0):.2f}" for k in score_cols])
                        h_row.extend([f"{t_s:.2f}", f"{mets['Daily_Var']:.4f}", f"{mets['Util_Avg']:.4f}",
                                      f"{mets['Interval_Rate']:.4f}", f"{mets['Build_Conc_Rate']:.4f}"])
                        csv_hist.writerow(h_row)
                f_hist.flush()

                excel_dict = {"Instance": inst, "Algorithm": ALGO_NAME, "Run": r, "Generations": final_gen,
                              "Best_Fitness": best_ind.fitness, "Avg_Fitness": best_ind.fitness, "Violations": vio,
                              "Soft_Score_Sum": s_sum, "Time_s": dur, "Daily_Var": mets['Daily_Var'],
                              "Util_Avg": mets['Util_Avg'], "Interval_Rate": mets['Interval_Rate'],
                              "Build_Conc_Rate": mets['Build_Conc_Rate']}
                excel_dict.update(raw)
                instance_excel_data.append(excel_dict)
                logger.info(f"      Run {r} DONE: Fit={best_ind.fitness:.2f} | Time={dur:.1f}s")

            except Exception as e:
                logger.error(f"❌ Error in Run {r}: {e}")

        # 恢复生成 best_schedules.json
        if algo_best_genes:
            update_best_schedules_json(os.path.join(inst_path, "best_schedules.json"), ALGO_NAME, algo_best_fit,
                                       algo_best_genes)

        # 恢复 Excel 双表单 (Raw_Data + Summary)
        if instance_excel_data:
            df_raw = pd.DataFrame(instance_excel_data).reindex(columns=common_cols)
            stats_cols = ['Generations', 'Best_Fitness', 'Avg_Fitness', 'Violations', 'Soft_Score_Sum'] + score_cols + [
                'Time_s', 'Daily_Var', 'Util_Avg', 'Interval_Rate', 'Build_Conc_Rate']
            summary_stats = df_raw.groupby('Algorithm')[stats_cols].agg(['mean', 'std']).reset_index()
            summary_stats.columns = [f"{c[0]}_{c[1]}" if (isinstance(c, tuple) and c[1]) else c[0] for c in
                                     summary_stats.columns]

            with pd.ExcelWriter(os.path.join(inst_path, "report_metrics.xlsx")) as writer:
                df_raw.to_excel(writer, sheet_name='Raw_Data', index=False)
                summary_stats.to_excel(writer, sheet_name='Summary', index=False)

    f_sum.close();
    f_hist.close()
    logger.info("=== ✅ DONE ===")


if __name__ == "__main__": main()
