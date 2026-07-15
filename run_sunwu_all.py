import os
import csv
import json
import time
import logging
import pandas as pd
import pickle
from datetime import datetime
from evaluation.fitness import FitnessEvaluator
from algorithms.sun_wu_tpts import SunWuTabuSearch

# ================================================================
# 核心参数：严格尊重 Sun & Wu (2023) 原文设定
# ================================================================
ALGO_NAME = "Sun-Wu-2023"
TOTAL_ITERS = 105000  # 10.5万次邻域搜索 (本地测试可先改小，如 100)
N_RUNS = 10  # 每个算例跑 10 轮 (本地测试可先改小，如 1)

INSTANCES = ["S1_Small", "S2_Small", "S3_Small", "S4_Small", "M1_Medium", "M2_Medium", "M3_Medium", "L1_Large"]


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
    res_dir = f"results/sunwu_only_{timestamp}"
    os.makedirs(res_dir, exist_ok=True)
    logger = setup_logger(res_dir)

    # 严格对齐您的列名顺序
    score_cols = ['f_daily', 'f_interval', 'f_room', 'f_util', 'f_build']
    common_cols = ['Instance', 'Algorithm', 'Run', 'Generations', 'Best_Fitness', 'Avg_Fitness',
                   'Violations', 'Soft_Score_Sum'] + score_cols + \
                  ['Time_s', 'Daily_Var', 'Util_Avg', 'Interval_Rate', 'Build_Conc_Rate']
    hist_cols = ['Instance', 'Algorithm', 'Run', 'Gen'] + common_cols[4:]

    # 初始化全局 Summary 和 History 表
    f_sum = open(os.path.join(res_dir, "summary.csv"), 'w', newline='', encoding='utf-8-sig')
    csv_sum = csv.writer(f_sum);
    csv_sum.writerow(common_cols)

    f_hist = open(os.path.join(res_dir, "history.csv"), 'w', newline='', encoding='utf-8-sig')
    csv_hist = csv.writer(f_hist);
    csv_hist.writerow(hist_cols)

    evaluator = FitnessEvaluator()

    for inst in INSTANCES:
        logger.info(f"🚀 Processing: {inst} ...")

        # 1. 安全高效的本地缓存读取方式
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
                algo = SunWuTabuSearch(data, evaluator)

                # 2. 执行孙哲算法 (禁忌搜索单点 Pop=1)
                best_ind, final_gen, dur, trace = algo.run_with_time_limit(
                    max_generations=TOTAL_ITERS, pop_size=1, time_limit=float('inf')
                )

                # 记录 JSON 用的全局最好基因
                if best_ind.fitness > algo_best_fit:
                    algo_best_fit = best_ind.fitness
                    algo_best_genes = best_ind.genes.copy()

                # 3. 物理指标与得分计算
                vio = evaluator.constraint_checker.check_all(best_ind.genes)['total']
                if vio > 0:
                    raw = {k: 0.0 for k in score_cols};
                    s_sum = 0.0
                    mets = {k: 0.0 for k in ['Daily_Var', 'Util_Avg', 'Interval_Rate', 'Build_Conc_Rate']}
                else:
                    raw = evaluator._calculate_raw_scores(best_ind.genes)
                    s_sum = sum(raw.values())
                    mets = evaluator.calculate_quality_metrics(best_ind.genes)

                # 4. 写入 Summary CSV
                row = [inst, ALGO_NAME, r, final_gen, f"{best_ind.fitness:.2f}", f"{best_ind.fitness:.2f}", vio,
                       f"{s_sum:.2f}"]
                row.extend([f"{raw.get(k, 0):.2f}" for k in score_cols])
                row.extend([f"{dur:.2f}", f"{mets['Daily_Var']:.4f}", f"{mets['Util_Avg']:.4f}",
                            f"{mets['Interval_Rate']:.4f}", f"{mets['Build_Conc_Rate']:.4f}"])
                csv_sum.writerow(row)
                f_sum.flush()

                # 5. 写入 History CSV (抽样记录防止文件爆炸)
                step_size = max(1, len(trace) // 200)
                for idx, (t_s, fit) in enumerate(trace):
                    if idx % step_size == 0 or idx == len(trace) - 1:
                        h_row = [inst, ALGO_NAME, r, idx, f"{fit:.2f}", f"{fit:.2f}", vio, f"{s_sum:.2f}"]
                        h_row.extend([f"{raw.get(k, 0):.2f}" for k in score_cols])
                        h_row.extend([f"{t_s:.2f}", f"{mets['Daily_Var']:.4f}", f"{mets['Util_Avg']:.4f}",
                                      f"{mets['Interval_Rate']:.4f}", f"{mets['Build_Conc_Rate']:.4f}"])
                        csv_hist.writerow(h_row)
                f_hist.flush()

                # 6. 收集 Excel 行数据
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

        # 7. 导出本算例的最佳课表 (JSON)
        if algo_best_genes:
            update_best_schedules_json(os.path.join(inst_path, "best_schedules.json"), ALGO_NAME, algo_best_fit,
                                       algo_best_genes)

        # 8. 导出双表单 Excel (Raw_Data + Summary 统计页)
        if instance_excel_data:
            df_raw = pd.DataFrame(instance_excel_data).reindex(columns=common_cols)
            stats_cols = ['Generations', 'Best_Fitness', 'Avg_Fitness', 'Violations', 'Soft_Score_Sum'] + score_cols + [
                'Time_s', 'Daily_Var', 'Util_Avg', 'Interval_Rate', 'Build_Conc_Rate']
            # 计算均值和标准差
            summary_stats = df_raw.groupby('Algorithm')[stats_cols].agg(['mean', 'std']).reset_index()
            # 展平列名 (例如 Best_Fitness_mean)
            summary_stats.columns = [f"{c[0]}_{c[1]}" if (isinstance(c, tuple) and c[1]) else c[0] for c in
                                     summary_stats.columns]

            with pd.ExcelWriter(os.path.join(inst_path, "report_metrics.xlsx")) as writer:
                df_raw.to_excel(writer, sheet_name='Raw_Data', index=False)
                summary_stats.to_excel(writer, sheet_name='Summary', index=False)

    f_sum.close();
    f_hist.close()
    logger.info("=== ✅ 孙哲禁忌搜索全量实验执行完毕 ===")


if __name__ == "__main__":
    main()
