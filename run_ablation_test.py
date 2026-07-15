# """
# run_ablation_test.py
#
# 完整结构测试版：
# - 跑全部 8 个实例
# - 跑全部 4 个算法
# - 只跑 1 run
# - 缩小 GA 参数
# - 保持与 run_ablation.py 相同的输出结构
# """
#
# import os
# import csv
# import json
# import logging
# import time
# import pickle
# import numpy as np
# import pandas as pd
# from datetime import datetime
#
# from evaluation.fitness import FitnessEvaluator
# from algorithms.genetic_algorithm import GeneticAlgorithm
#
#
# N_RUNS = 1
#
#
# def setup_logger(res_dir):
#     logger = logging.getLogger()
#     logger.setLevel(logging.INFO)
#
#     if logger.hasHandlers():
#         logger.handlers.clear()
#
#     formatter = logging.Formatter(
#         '%(asctime)s [%(levelname)s] %(message)s',
#         datefmt='%H:%M:%S'
#     )
#
#     fh = logging.FileHandler(os.path.join(res_dir, "experiment.log"), encoding='utf-8')
#     fh.setFormatter(formatter)
#     logger.addHandler(fh)
#
#     sh = logging.StreamHandler()
#     sh.setFormatter(formatter)
#     logger.addHandler(sh)
#
#     return logger
#
#
# def save_instance_metadata(inst_path, inst_name, data):
#     college_ids = set()
#     try:
#         for c in data.classes.values():
#             if hasattr(c, 'college_id') and c.college_id:
#                 college_ids.add(str(c.college_id))
#     except Exception:
#         pass
#
#     college_list_str = "|".join(sorted(list(college_ids)))
#     meta = {
#         'Instance': inst_name,
#         'College_Count': len(college_ids),
#         'College_IDs': college_list_str,
#         'Tasks': data.num_tasks,
#         'Teachers': data.num_teachers,
#         'Classrooms': data.num_classrooms,
#         'Classes': data.num_classes
#     }
#
#     pd.DataFrame([meta]).to_csv(
#         os.path.join(inst_path, "metadata.csv"),
#         index=False,
#         encoding='utf-8-sig'
#     )
#
#
# def update_best_schedules_json(json_path, algo_name, fitness, schedule_genes):
#     data = {}
#
#     if os.path.exists(json_path):
#         try:
#             with open(json_path, 'r', encoding='utf-8') as f:
#                 content = f.read()
#                 if content.strip():
#                     data = json.loads(content)
#         except Exception:
#             data = {}
#
#     serializable_genes = {str(k): int(v) for k, v in schedule_genes.items()}
#     data[algo_name] = {"fitness": fitness, "schedule": serializable_genes}
#
#     try:
#         with open(json_path, 'w', encoding='utf-8') as f:
#             json.dump(data, f, indent=4)
#         return True
#     except Exception:
#         return False
#
#
# def main():
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     res_dir = f"results/TEST_ablation_{timestamp}"
#     os.makedirs(res_dir, exist_ok=True)
#
#     logger = setup_logger(res_dir)
#     logger.info("=== 🚀 TEST RUN_ABLATION START ===")
#
#     score_cols = ['f_daily', 'f_interval', 'f_room', 'f_util', 'f_build']
#     common_cols = [
#         'Instance', 'Algorithm', 'Run', 'Generations', 'Best_Fitness', 'Avg_Fitness',
#         'Violations', 'Soft_Score_Sum'
#     ] + score_cols + [
#         'Time_s', 'Daily_Var', 'Util_Avg', 'Interval_Rate', 'Build_Conc_Rate'
#     ]
#
#     f_hist = open(os.path.join(res_dir, "history.csv"), 'w', newline='', encoding='utf-8-sig')
#     csv_hist = csv.writer(f_hist)
#     csv_hist.writerow(['Instance', 'Algorithm', 'Run', 'Gen'] + common_cols[4:])
#
#     f_sum = open(os.path.join(res_dir, "summary.csv"), 'w', newline='', encoding='utf-8-sig')
#     csv_sum = csv.writer(f_sum)
#     csv_sum.writerow(common_cols)
#
#     f_rl = open(os.path.join(res_dir, "rl_trace.csv"), 'w', newline='', encoding='utf-8-sig')
#     csv_rl = csv.writer(f_rl)
#     csv_rl.writerow(
#         ['Instance', 'Algorithm', 'Run', 'Gen', 'Action', 'Reward', 'Loss', 'Mutation', 'Crossover', 'Q_Values']
#     )
#
#     # 全部实例，但缩小测试规模
#     instances_config = {
#         'S1_Small': {'type': 'Small', 'gen': 50, 'pop': 30, 'patience': 10},
#         'S2_Small': {'type': 'Small', 'gen': 50, 'pop': 30, 'patience': 10},
#         'S3_Small': {'type': 'Small', 'gen': 50, 'pop': 30, 'patience': 10},
#         'S4_Small': {'type': 'Small', 'gen': 50, 'pop': 30, 'patience': 10},
#         'M1_Medium': {'type': 'Medium', 'gen': 50, 'pop': 30, 'patience': 10},
#         'M2_Medium': {'type': 'Medium', 'gen': 50, 'pop': 30, 'patience': 10},
#         'M3_Medium': {'type': 'Medium', 'gen': 50, 'pop': 30, 'patience': 10},
#         'L1_Large': {'type': 'Large', 'gen': 50, 'pop': 30, 'patience': 10}
#     }
#
#     variants = {
#         'DRL-GA-TS': {
#             'use_rl_adaptive': True,
#             'use_block_operators': False,
#             'use_repair': False,
#             'use_cdm': False,
#             'use_tabu': True
#         },
#         'DL-GA-TS-Block': {
#             'use_rl_adaptive': True,
#             'use_block_operators': True,
#             'use_repair': False,
#             'use_cdm': False,
#             'use_tabu': True
#         },
#         'DL-GA-TS-Block-Repair': {
#             'use_rl_adaptive': True,
#             'use_block_operators': True,
#             'use_repair': True,
#             'use_cdm': False,
#             'use_tabu': True
#         },
#         'DL-GA-TS-Block-Repair-TrueCDM': {
#             'use_rl_adaptive': True,
#             'use_block_operators': True,
#             'use_repair': True,
#             'use_cdm': True,
#             'use_tabu': True
#         }
#     }
#
#     try:
#         for inst_name, cfg in instances_config.items():
#             logger.info(f"\n📦 Loading Instance: {inst_name}")
#
#             cache_file = os.path.join("data", "cache", f"{inst_name}_cache.pkl")
#             if not os.path.exists(cache_file):
#                 raise RuntimeError(f"Cache file not found: {cache_file}")
#
#             with open(cache_file, "rb") as f:
#                 data = pickle.load(f)
#
#             evaluator = FitnessEvaluator()
#             evaluator.set_data(data)
#
#             inst_path = os.path.join(res_dir, inst_name)
#             os.makedirs(inst_path, exist_ok=True)
#
#             save_instance_metadata(inst_path, inst_name, data)
#
#             instance_excel_data = []
#
#             for algo_name, flags in variants.items():
#                 logger.info(f"   ▶ Algorithm: {algo_name}")
#
#                 algo_best_fit = -float('inf')
#                 algo_best_genes = None
#
#                 for run_id in range(1, N_RUNS + 1):
#                     start_t = time.time()
#
#                     try:
#                         config = {
#                             'use_heuristic_init': True,
#                             'elite_count': 2,
#                             'crossover_rate': 0.8,
#                             'mutation_rate': 0.1,
#                             'metric_interval': 20,
#                             'patience': cfg['patience'],
#                             'population_size': cfg['pop'],
#                             'tabu_steps': 5,
#                             **flags
#                         }
#
#                         ga = GeneticAlgorithm(data, evaluator, config)
#                         ret = ga.run(cfg['gen'], cfg['pop'], run_id)
#                         best_ind, history, rl_logs = ret[:3]
#
#                         final_gen = len(history)
#                         dur = time.time() - start_t
#
#                         if best_ind.fitness > algo_best_fit:
#                             algo_best_fit = best_ind.fitness
#                             algo_best_genes = best_ind.genes.copy()
#
#                         last_rec = history[-1] if history else {}
#                         vio = evaluator.constraint_checker.check_all(best_ind.genes)['total']
#
#                         if vio > 0:
#                             s_sum, d_var, u_avg, i_rate, b_rate = 0.0, 0.0, 0.0, 0.0, 0.0
#                             raw_scores = {k: 0.0 for k in score_cols}
#                         else:
#                             s_sum_raw = last_rec.get('soft_score_sum', 0)
#                             s_sum = sum(s_sum_raw.values()) if isinstance(s_sum_raw, dict) else float(s_sum_raw)
#
#                             d_var = last_rec.get('Daily_Var', 0)
#                             if d_var == 0:
#                                 mets = evaluator.calculate_quality_metrics(best_ind.genes)
#                                 d_var = mets['Daily_Var']
#                                 u_avg = mets['Util_Avg']
#                                 i_rate = mets['Interval_Rate']
#                                 b_rate = mets['Build_Conc_Rate']
#                             else:
#                                 u_avg = last_rec.get('Util_Avg', 0)
#                                 i_rate = last_rec.get('Interval_Rate', 0)
#                                 b_rate = last_rec.get('Build_Conc_Rate', 0)
#
#                             if 'f_daily' in last_rec:
#                                 raw_scores = {k: last_rec.get(k, 0) for k in score_cols}
#                             else:
#                                 raw_scores = evaluator._calculate_raw_scores(best_ind.genes)
#
#                         row = [
#                             inst_name,
#                             algo_name,
#                             run_id,
#                             final_gen,
#                             f"{best_ind.fitness:.2f}",
#                             f"{last_rec.get('avg_fitness', last_rec.get('avg_fit', 0)):.2f}",
#                             vio,
#                             f"{s_sum:.2f}"
#                         ]
#                         row.extend([f"{raw_scores.get(k, 0):.2f}" for k in score_cols])
#                         row.extend([f"{dur:.2f}", f"{d_var:.4f}", f"{u_avg:.4f}", f"{i_rate:.4f}", f"{b_rate:.4f}"])
#                         csv_sum.writerow(row)
#                         f_sum.flush()
#
#                         for h in history:
#                             b_fit_val = h.get('best_fitness', h.get('best_fit', 0))
#                             a_fit_val = h.get('avg_fitness', h.get('avg_fit', 0))
#                             vio_val = h.get('hard_violations', h.get('violations', 0))
#
#                             s_val_raw = h.get('soft_score_sum', 0)
#                             s_val = sum(s_val_raw.values()) if isinstance(s_val_raw, dict) else float(s_val_raw)
#
#                             h_row = [
#                                 inst_name,
#                                 algo_name,
#                                 run_id,
#                                 h.get('gen', 0),
#                                 f"{b_fit_val:.2f}",
#                                 f"{a_fit_val:.2f}",
#                                 vio_val,
#                                 f"{s_val:.2f}"
#                             ]
#                             h_row.extend([f"{h.get(k, 0):.2f}" for k in score_cols])
#                             h_row.extend([
#                                 f"{h.get('time_s', 0):.2f}",
#                                 f"{h.get('Daily_Var', 0):.4f}",
#                                 f"{h.get('Util_Avg', 0):.4f}",
#                                 f"{h.get('Interval_Rate', 0):.4f}",
#                                 f"{h.get('Build_Conc_Rate', 0):.4f}"
#                             ])
#                             csv_hist.writerow(h_row)
#                         f_hist.flush()
#
#                         if rl_logs:
#                             for log in rl_logs:
#                                 loss_val = "" if log.get('loss') is None else f"{log['loss']}"
#                                 csv_rl.writerow([
#                                     inst_name,
#                                     algo_name,
#                                     run_id,
#                                     log['gen'],
#                                     log['action'],
#                                     f"{log['reward']:.4f}",
#                                     loss_val,
#                                     f"{log['mutation']:.4f}",
#                                     f"{log['crossover']:.4f}",
#                                     str(log['q_values'])
#                                 ])
#                             f_rl.flush()
#
#                         excel_dict = {
#                             "Instance": inst_name,
#                             "Algorithm": algo_name,
#                             "Run": run_id,
#                             "Generations": final_gen,
#                             "Best_Fitness": best_ind.fitness,
#                             "Avg_Fitness": last_rec.get('avg_fitness', last_rec.get('avg_fit', 0)),
#                             "Violations": vio,
#                             "Soft_Score_Sum": s_sum,
#                             "Time_s": dur,
#                             "Daily_Var": d_var,
#                             "Util_Avg": u_avg,
#                             "Interval_Rate": i_rate,
#                             "Build_Conc_Rate": b_rate
#                         }
#                         excel_dict.update(raw_scores)
#                         instance_excel_data.append(excel_dict)
#
#                     except Exception as e:
#                         logger.error(f"❌ Run Failed ({inst_name} | {algo_name} | Run {run_id}): {e}")
#                         import traceback
#                         logger.error(traceback.format_exc())
#
#                 if algo_best_genes:
#                     update_best_schedules_json(
#                         os.path.join(inst_path, "best_schedules.json"),
#                         algo_name,
#                         algo_best_fit,
#                         algo_best_genes
#                     )
#
#             if instance_excel_data:
#                 df_raw = pd.DataFrame(instance_excel_data)
#                 df_raw = df_raw.reindex(columns=common_cols)
#
#                 stats_cols = [
#                     'Generations', 'Best_Fitness', 'Avg_Fitness', 'Violations',
#                     'Soft_Score_Sum'
#                 ] + score_cols + [
#                     'Time_s', 'Daily_Var', 'Util_Avg', 'Interval_Rate', 'Build_Conc_Rate'
#                 ]
#
#                 summary_stats = df_raw.groupby('Algorithm')[stats_cols].agg(['mean', 'std']).reset_index()
#
#                 new_cols = []
#                 for col in summary_stats.columns:
#                     if isinstance(col, tuple):
#                         new_cols.append(f"{col[0]}_{col[1]}" if col[1] else col[0])
#                     else:
#                         new_cols.append(col)
#                 summary_stats.columns = new_cols
#
#                 with pd.ExcelWriter(os.path.join(inst_path, "report_metrics.xlsx")) as writer:
#                     df_raw.to_excel(writer, sheet_name='Raw_Data', index=False)
#                     summary_stats.to_excel(writer, sheet_name='Summary', index=False)
#
#         logger.info("=== ✅ TEST FINISHED ===")
#
#     finally:
#         f_sum.close()
#         f_hist.close()
#         f_rl.close()
#
#
# if __name__ == "__main__":
#     main()

import os
import csv
import json
import logging
import time
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from evaluation.fitness import FitnessEvaluator
from algorithms.genetic_algorithm import GeneticAlgorithm

N_RUNS = 1
GLOBAL_WEIGHTS_FILE = "drl_model_memory.pth"


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


def save_instance_metadata(inst_path, inst_name, data):
    college_ids = set()
    try:
        for c in data.classes.values():
            if hasattr(c, 'college_id') and c.college_id:
                college_ids.add(str(c.college_id))
    except:
        pass
    meta = {'Instance': inst_name, 'College_Count': len(college_ids), 'Tasks': data.num_tasks,
            'Teachers': data.num_teachers, 'Classrooms': data.num_classrooms, 'Classes': data.num_classes}
    pd.DataFrame([meta]).to_csv(os.path.join(inst_path, "metadata.csv"), index=False, encoding='utf-8-sig')


def update_best_schedules_json(json_path, algo_name, fitness, schedule_genes):
    data = {}
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                content = f.read()
                if content.strip(): data = json.loads(content)
        except:
            data = {}
    serializable_genes = {str(k): int(v) for k, v in schedule_genes.items()}
    data[algo_name] = {"fitness": float(fitness), "schedule": serializable_genes}
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    res_dir = f"results/test_ablation_{timestamp}"
    os.makedirs(res_dir, exist_ok=True)
    logger = setup_logger(res_dir)

    score_cols = ['f_daily', 'f_interval', 'f_room', 'f_util', 'f_build']
    common_cols = ['Instance', 'Algorithm', 'Run', 'Generations', 'Best_Fitness', 'Avg_Fitness', 'Violations',
                   'Soft_Score_Sum'] + score_cols + ['Time_s', 'Daily_Var', 'Util_Avg', 'Interval_Rate',
                                                     'Build_Conc_Rate']

    f_hist = open(os.path.join(res_dir, "history.csv"), 'w', newline='', encoding='utf-8-sig')
    f_sum = open(os.path.join(res_dir, "summary.csv"), 'w', newline='', encoding='utf-8-sig')
    f_rl = open(os.path.join(res_dir, "rl_trace.csv"), 'w', newline='', encoding='utf-8-sig')
    csv_hist, csv_sum, csv_rl = csv.writer(f_hist), csv.writer(f_sum), csv.writer(f_rl)

    csv_hist.writerow(['Instance', 'Algorithm', 'Run', 'Gen'] + common_cols[4:])
    csv_sum.writerow(common_cols)
    csv_rl.writerow(
        ['Instance', 'Algorithm', 'Run', 'Gen', 'Action', 'Reward', 'Loss', 'Mutation', 'Crossover', 'Q_Values'])

    instances_config = {
        'S1_Small': {'gen': 20, 'pop': 20, 'patience': 10},
        'S2_Small': {'gen': 20, 'pop': 20, 'patience': 10},
        'S3_Small': {'gen': 20, 'pop': 20, 'patience': 10},
        'S4_Small': {'gen': 20, 'pop': 20, 'patience': 10},
        'M1_Medium': {'gen': 30, 'pop': 30, 'patience': 15},
        'M2_Medium': {'gen': 30, 'pop': 30, 'patience': 15},
        'M3_Medium': {'gen': 30, 'pop': 30, 'patience': 15},
        'L1_Large': {'gen': 50, 'pop': 40, 'patience': 20}
    }

    try:
        for inst_name, cfg in instances_config.items():
            cache_file = os.path.join("data", "cache", f"{inst_name}_cache.pkl")
            if not os.path.exists(cache_file): continue
            with open(cache_file, "rb") as f:
                data = pickle.load(f)

            evaluator = FitnessEvaluator()
            evaluator.set_data(data)
            inst_path = os.path.join(res_dir, inst_name)
            os.makedirs(inst_path, exist_ok=True)
            save_instance_metadata(inst_path, inst_name, data)

            # ✨ 逻辑对齐：instance_excel_data 定义在这里，汇总该实例下所有算法
            instance_excel_data = []

            variants = {
                'DRL-GA-TS': {'use_rl_adaptive': True, 'use_block_operators': False, 'use_repair': False,
                              'use_cdm': False},
                'DL-GA-TS-Block': {'use_rl_adaptive': True, 'use_block_operators': True, 'use_repair': False,
                                   'use_cdm': False},
                'DL-GA-TS-Block-Repair': {'use_rl_adaptive': True, 'use_block_operators': True, 'use_repair': True,
                                          'use_cdm': False},
                'DL-GA-TS-Block-Repair-TrueCDM': {'use_rl_adaptive': True, 'use_block_operators': True,
                                                  'use_repair': True, 'use_cdm': True}
            }

            for algo_name, flags in variants.items():
                algo_best_fit = -float('inf')
                algo_best_genes = None

                for run_id in range(1, N_RUNS + 1):
                    start_t = time.time()
                    config = {'use_heuristic_init': True, 'patience': cfg['patience'], 'population_size': cfg['pop'],
                              'tabu_steps': 5, **flags}
                    ga = GeneticAlgorithm(data, evaluator, config)

                    if hasattr(ga, 'rl_controller'): ga.rl_controller.load_brain(GLOBAL_WEIGHTS_FILE)
                    best_ind, history, rl_logs, _ = ga.run(cfg['gen'], cfg['pop'], run_id)
                    if hasattr(ga, 'rl_controller'): ga.rl_controller.save_brain(GLOBAL_WEIGHTS_FILE)

                    dur = time.time() - start_t
                    last_rec = history[-1]
                    vio = evaluator.constraint_checker.check_all(best_ind.genes)['total']

                    b_val = last_rec.get('best_fit', last_rec.get('best_fitness', best_ind.fitness))
                    a_val = last_rec.get('avg_fit', last_rec.get('avg_fitness', 0))
                    s_sum_raw = last_rec.get('soft_score_sum', 0)
                    s_sum = sum(s_sum_raw.values()) if isinstance(s_sum_raw, dict) else float(s_sum_raw)
                    raw_scores = evaluator._calculate_raw_scores(best_ind.genes)

                    if b_val > algo_best_fit:
                        algo_best_fit, algo_best_genes = b_val, best_ind.genes.copy()

                    row = [inst_name, algo_name, run_id, len(history), f"{b_val:.2f}", f"{a_val:.2f}", vio,
                           f"{s_sum:.2f}"]
                    row.extend([f"{raw_scores.get(k, 0):.2f}" for k in score_cols])
                    row.extend(
                        [f"{dur:.2f}", f"{last_rec.get('Daily_Var', 0):.4f}", f"{last_rec.get('Util_Avg', 0):.4f}",
                         f"{last_rec.get('Interval_Rate', 0):.4f}", f"{last_rec.get('Build_Conc_Rate', 0):.4f}"])
                    csv_sum.writerow(row);
                    f_sum.flush()

                    for h in history:
                        h_b = h.get('best_fit', h.get('best_fitness', 0))
                        h_a = h.get('avg_fit', h.get('avg_fitness', 0))
                        h_row = [inst_name, algo_name, run_id, h.get('gen', 0), f"{h_b:.2f}", f"{h_a:.2f}",
                                 h.get('violations', 0), "0.00"]
                        h_row.extend(["0.00"] * 10)
                        csv_hist.writerow(h_row)
                    f_hist.flush()

                    if rl_logs:
                        for log in rl_logs:
                            l_val = log.get('loss');
                            l_str = f"{l_val:.4f}" if l_val is not None else ""
                            csv_rl.writerow([inst_name, algo_name, run_id, log.get('gen', 0), log.get('action', 0),
                                             f"{log.get('reward', 0):.4f}", l_str, f"{log.get('mutation', 0):.4f}",
                                             f"{log.get('crossover', 0):.4f}", str(log.get('q_values', []))])
                        f_rl.flush()

                    excel_row = {"Instance": inst_name, "Algorithm": algo_name, "Run": run_id,
                                 "Generations": len(history), "Best_Fitness": b_val, "Avg_Fitness": a_val,
                                 "Violations": vio, "Soft_Score_Sum": s_sum, "Time_s": dur}
                    excel_row.update(raw_scores)
                    excel_row.update(
                        {"Daily_Var": last_rec.get('Daily_Var', 0), "Util_Avg": last_rec.get('Util_Avg', 0),
                         "Interval_Rate": last_rec.get('Interval_Rate', 0),
                         "Build_Conc_Rate": last_rec.get('Build_Conc_Rate', 0)})
                    instance_excel_data.append(excel_row)

                # 每个算法跑完 10 次后更新该算法在该实例下的最好解
                if algo_best_genes:
                    update_best_schedules_json(os.path.join(inst_path, "best_schedules.json"), algo_name, algo_best_fit,
                                               algo_best_genes)

            # ✨ 逻辑对齐：所有算法（Variants）跑完后，汇总生成该实例唯一的 Excel
            if instance_excel_data:
                df_raw = pd.DataFrame(instance_excel_data)
                df_raw = df_raw.reindex(columns=common_cols)
                stats_cols = ['Generations', 'Best_Fitness', 'Avg_Fitness', 'Violations',
                              'Soft_Score_Sum'] + score_cols + ['Time_s', 'Daily_Var', 'Util_Avg', 'Interval_Rate',
                                                                'Build_Conc_Rate']

                summary_stats = df_raw.groupby('Algorithm')[stats_cols].agg(['mean', 'std']).reset_index()
                summary_stats.columns = [f"{c[0]}_{c[1]}" if (isinstance(c, tuple) and c[1]) else c[0] for c in
                                         summary_stats.columns]

                with pd.ExcelWriter(os.path.join(inst_path, "report_metrics.xlsx")) as writer:
                    df_raw.to_excel(writer, sheet_name='Raw_Data', index=False)
                    summary_stats.to_excel(writer, sheet_name='Summary', index=False)

    finally:
        f_sum.close();
        f_hist.close();
        f_rl.close()
    logger.info("=== ✅ TEST FINISHED ===")


if __name__ == "__main__":
    main()

