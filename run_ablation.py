# """
# 正式消融实验主程序 (Ablation Study: Block / Repair / CDM)
# 文件名: run_ablation.py
# 功能: 以 DRL-GA-TS 为基准，验证合班编码降维、修复护盾与冲突定向变异的有效性。
# (包含最新的数据写入安全防护、完善的异常溯源日志与文件句柄容灾)
# """
# import os
# import csv
# import json
# import logging
# import time
# import numpy as np
# import pandas as pd
# from datetime import datetime
# from data.data_loader import DataLoader
# from evaluation.fitness import FitnessEvaluator
# from algorithms.genetic_algorithm import GeneticAlgorithm
#
# # === 全局配置 ===
# N_RUNS = 10
# DB_CONFIG = {'host': 'localhost', 'port': 3306, 'user': 'root', 'password': os.getenv('DB_PASSWORD', ''), 'database': 'test3'}
#
#
# def setup_logger(res_dir):
#     logger = logging.getLogger()
#     logger.setLevel(logging.INFO)
#     if logger.hasHandlers(): logger.handlers.clear()
#     formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
#     fh = logging.FileHandler(os.path.join(res_dir, "experiment.log"), encoding='utf-8')
#     fh.setFormatter(formatter)
#     logger.addHandler(fh)
#     sh = logging.StreamHandler()
#     sh.setFormatter(formatter)
#     logger.addHandler(sh)
#     return logger
#
#
# def save_instance_metadata(inst_path, inst_name, data):
#     college_ids = set()
#     try:
#         for c in data.classes.values():
#             if hasattr(c, 'college_id') and c.college_id:
#                 college_ids.add(str(c.college_id))
#     except Exception as e:
#         pass
#     college_list_str = "|".join(sorted(list(college_ids)))
#     meta = {
#         'Instance': inst_name, 'College_Count': len(college_ids), 'College_IDs': college_list_str,
#         'Tasks': data.num_tasks, 'Teachers': data.num_teachers, 'Classrooms': data.num_classrooms,
#         'Classes': data.num_classes
#     }
#     pd.DataFrame([meta]).to_csv(os.path.join(inst_path, "metadata.csv"), index=False, encoding='utf-8-sig')
#
#
# def load_best_params_by_type(inst_type):
#     suffix = "small" if inst_type == "Small" else "medium"
#     if inst_type == 'Large': suffix = 'medium'
#     json_path = os.path.join("results", f"best_params_{suffix}.json")
#     if os.path.exists(json_path):
#         try:
#             with open(json_path, 'r', encoding='utf-8') as f:
#                 return json.load(f)
#         except:
#             pass
#     return {'crossover_rate': 0.8, 'mutation_rate': 0.1}
#
#
# def update_best_schedules_json(json_path, algo_name, fitness, schedule_genes):
#     data = {}
#     if os.path.exists(json_path):
#         try:
#             with open(json_path, 'r', encoding='utf-8') as f:
#                 content = f.read()
#                 if content.strip(): data = json.loads(content)
#         except:
#             data = {}
#     serializable_genes = {str(k): int(v) for k, v in schedule_genes.items()}
#     data[algo_name] = {"fitness": fitness, "schedule": serializable_genes}
#     try:
#         with open(json_path, 'w', encoding='utf-8') as f:
#             json.dump(data, f, indent=4)
#         return True
#     except:
#         return False
#
#
# def main():
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     res_dir = f"results/ablation_exp_{timestamp}"
#     os.makedirs(res_dir, exist_ok=True)
#     logger = setup_logger(res_dir)
#     logger.info(f"=== 🚀 正式核心算子消融实验发车 (Block/Repair/CDM) ===")
#
#     score_cols = ['f_daily', 'f_interval', 'f_room', 'f_util', 'f_build']
#     common_cols = ['Instance', 'Algorithm', 'Run', 'Generations', 'Best_Fitness', 'Avg_Fitness',
#                    'Violations', 'Soft_Score_Sum'] + score_cols + \
#                   ['Time_s', 'Daily_Var', 'Util_Avg', 'Interval_Rate', 'Build_Conc_Rate']
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
#         ['Instance', 'Algorithm', 'Run', 'Gen', 'Action', 'Reward', 'Loss', 'Mutation', 'Crossover', 'Q_Values'])
#
#     instances_config = {
#         'S1_Small': {'type': 'Small', 'gen': 300, 'pop': 100, 'patience': 70},
#         'S2_Small': {'type': 'Small', 'gen': 500, 'pop': 100, 'patience': 70},
#         'S3_Small': {'type': 'Small', 'gen': 500, 'pop': 100, 'patience': 70},
#         'S4_Small': {'type': 'Small', 'gen': 600, 'pop': 100, 'patience': 80},
#         'M1_Medium': {'type': 'Medium', 'gen': 1000, 'pop': 150, 'patience': 100},
#         'M2_Medium': {'type': 'Medium', 'gen': 1000, 'pop': 150, 'patience': 100},
#         'M3_Medium': {'type': 'Medium', 'gen': 1200, 'pop': 150, 'patience': 110},
#         'L1_Large': {'type': 'Large', 'gen': 2000, 'pop': 200, 'patience': 150}
#     }
#
#     # ⭐ 优化: try...finally 包裹主逻辑，防止异常时文件句柄未关闭
#     try:
#         for inst_name, cfg in instances_config.items():
#             # logger.info(f"\n📦 Loading Instance: {inst_name} ...")
#             # try:
#             #     loader = DataLoader(inst_name, DB_CONFIG)
#             #     data = loader.load()
#             # except Exception as e:
#             #     logger.error(f"❌ Failed to load instance {inst_name}: {e}")
#             #     continue
#             logger.info(f"\n📦 Loading Instance: {inst_name} ...")
#
#             try:
#                 import pickle
#
#                 cache_file = os.path.join("data", "cache", f"{inst_name}_cache.pkl")
#
#                 if not os.path.exists(cache_file):
#                     raise RuntimeError(f"Cache file not found: {cache_file}")
#
#                 logger.info(f"💊 Loading cached instance: {cache_file}")
#
#                 with open(cache_file, "rb") as f:
#                     data = pickle.load(f)
#
#             except Exception as e:
#                 logger.error(f"❌ Failed to load instance {inst_name}: {e}")
#                 continue
#
#             evaluator = FitnessEvaluator()
#             evaluator.set_data(data)
#             inst_path = os.path.join(res_dir, inst_name)
#             os.makedirs(inst_path, exist_ok=True)
#             save_instance_metadata(inst_path, inst_name, data)
#
#             best_params = load_best_params_by_type(cfg['type'])
#
#             variants = {
#                 'DRL-GA-TS': {
#                     'use_rl_adaptive': True, 'use_block_operators': False,
#                     'use_repair': False, 'use_cdm': False, 'use_tabu': True
#                 },
#                 'DL-GA-TS-Block': {
#                     'use_rl_adaptive': True, 'use_block_operators': True,
#                     'use_repair': False, 'use_cdm': False, 'use_tabu': True
#                 },
#                 'DL-GA-TS-Block-Repair': {
#                     'use_rl_adaptive': True, 'use_block_operators': True,
#                     'use_repair': True, 'use_cdm': False, 'use_tabu': True
#                 },
#                 'DL-GA-TS-Block-Repair-TrueCDM': {
#                     'use_rl_adaptive': True, 'use_block_operators': True,
#                     'use_repair': True, 'use_cdm': True, 'use_tabu': True
#                 }
#             }
#
#             instance_excel_data = []
#
#             for algo_name, flags in variants.items():
#                 logger.info(f"   ▶ Algorithm: {algo_name}")
#                 algo_best_fit = -float('inf')
#                 algo_best_genes = None
#
#                 for run_id in range(1, N_RUNS + 1):
#                     start_t = time.time()
#                     try:
#                         config = {
#                             'use_heuristic_init': True, 'elite_count': 2,
#                             'crossover_rate': best_params['crossover_rate'],
#                             'mutation_rate': best_params['mutation_rate'],
#                             'metric_interval': 50, 'patience': cfg['patience'], 'population_size': cfg['pop'],
#                             'tabu_steps': 10, **flags
#                         }
#                         ga = GeneticAlgorithm(data, evaluator, config)
#                         ret = ga.run(cfg['gen'], cfg['pop'], run_id)
#                         best_ind, history, rl_logs = ret[:3]
#
#                         final_gen = len(history)
#                         dur = time.time() - start_t
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
#                                 d_var, u_avg, i_rate, b_rate = mets['Daily_Var'], mets['Util_Avg'], mets[
#                                     'Interval_Rate'], mets['Build_Conc_Rate']
#                             else:
#                                 u_avg, i_rate, b_rate = last_rec.get('Util_Avg', 0), last_rec.get('Interval_Rate',
#                                                                                                   0), last_rec.get(
#                                     'Build_Conc_Rate', 0)
#
#                             if 'f_daily' in last_rec:
#                                 raw_scores = {k: last_rec.get(k, 0) for k in score_cols}
#                             else:
#                                 raw_scores = evaluator._calculate_raw_scores(best_ind.genes)
#
#                         row = [inst_name, algo_name, run_id, final_gen, f"{best_ind.fitness:.2f}",
#                                f"{last_rec.get('avg_fitness', last_rec.get('avg_fit', 0)):.2f}", vio, f"{s_sum:.2f}"]
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
#                             h_row = [inst_name, algo_name, run_id, h.get('gen', 0), f"{b_fit_val:.2f}",
#                                      f"{a_fit_val:.2f}", vio_val, f"{s_val:.2f}"]
#                             h_row.extend([f"{h.get(k, 0):.2f}" for k in score_cols])
#                             h_row.extend([f"{h.get('time_s', 0):.2f}", f"{h.get('Daily_Var', 0):.4f}",
#                                           f"{h.get('Util_Avg', 0):.4f}", f"{h.get('Interval_Rate', 0):.4f}",
#                                           f"{h.get('Build_Conc_Rate', 0):.4f}"])
#                             csv_hist.writerow(h_row)
#                         f_hist.flush()
#
#                         if rl_logs:
#                             for log in rl_logs:
#                                 # ⭐ 优化: loss 若为 None，转为空字符串防 Pandas 解析异常
#                                 loss_val = "" if log.get('loss') is None else f"{log['loss']}"
#                                 csv_rl.writerow(
#                                     [inst_name, algo_name, run_id, log['gen'], log['action'], f"{log['reward']:.4f}",
#                                      loss_val, f"{log['mutation']:.4f}", f"{log['crossover']:.4f}",
#                                      str(log['q_values'])])
#                             f_rl.flush()
#
#                         excel_dict = {"Instance": inst_name, "Algorithm": algo_name, "Run": run_id,
#                                       "Generations": final_gen, "Best_Fitness": best_ind.fitness,
#                                       "Avg_Fitness": last_rec.get('avg_fitness', last_rec.get('avg_fit', 0)),
#                                       "Violations": vio, "Soft_Score_Sum": s_sum, "Time_s": dur, "Daily_Var": d_var,
#                                       "Util_Avg": u_avg, "Interval_Rate": i_rate, "Build_Conc_Rate": b_rate}
#                         excel_dict.update(raw_scores)
#                         instance_excel_data.append(excel_dict)
#
#                     except Exception as e:
#                         logger.error(f"❌ Run Failed ({inst_name} | {algo_name} | Run {run_id}): {e}")
#                         import traceback
#                         logger.error(traceback.format_exc())
#
#                 # 放在 for run_id 外部，仅保存当前算法最好的基因
#                 if algo_best_genes:
#                     update_best_schedules_json(os.path.join(inst_path, "best_schedules.json"), algo_name, algo_best_fit,
#                                                algo_best_genes)
#
#             if instance_excel_data:
#                 df_raw = pd.DataFrame(instance_excel_data)
#                 df_raw = df_raw.reindex(columns=common_cols)
#                 stats_cols = ['Generations', 'Best_Fitness', 'Avg_Fitness', 'Violations',
#                               'Soft_Score_Sum'] + score_cols + ['Time_s', 'Daily_Var', 'Util_Avg', 'Interval_Rate',
#                                                                 'Build_Conc_Rate']
#                 summary_stats = df_raw.groupby('Algorithm')[stats_cols].agg(['mean', 'std']).reset_index()
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
#         logger.info("=== ✅ 所有消融实验执行完毕 ===")
#
#     finally:
#         f_sum.close()
#         f_hist.close()
#         f_rl.close()
#
#
# if __name__ == "__main__":
#     main()

"""
正式消融实验主程序 (Ablation Study: Block / Repair / CDM)
文件名: run_ablation.py
功能: 严格对齐原始汇总逻辑、参数配置，并修复 JSON 乱序与 DRL 记忆热启动
"""
import os
import csv
import json
import logging
import time
import numpy as np
import pandas as pd
import pickle
from datetime import datetime
from evaluation.fitness import FitnessEvaluator
from algorithms.genetic_algorithm import GeneticAlgorithm

# === 全局配置 ===
N_RUNS = 10
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
    college_list_str = "|".join(sorted(list(college_ids)))
    meta = {
        'Instance': inst_name, 'College_Count': len(college_ids), 'College_IDs': college_list_str,
        'Tasks': data.num_tasks, 'Teachers': data.num_teachers, 'Classrooms': data.num_classrooms,
        'Classes': data.num_classes
    }
    pd.DataFrame([meta]).to_csv(os.path.join(inst_path, "metadata.csv"), index=False, encoding='utf-8-sig')


def load_best_params_by_type(inst_type):
    suffix = "small" if inst_type == "Small" else "medium"
    if inst_type == 'Large': suffix = 'medium'
    json_path = os.path.join("results", f"best_params_{suffix}.json")
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            pass
    return {'crossover_rate': 0.8, 'mutation_rate': 0.1}


def update_best_schedules_json(json_path, algo_name, fitness, schedule_genes):
    # ⭐ 修复乱序：定义固定的算法顺序
    algo_order = ['DRL-GA-TS', 'DL-GA-TS-Block', 'DL-GA-TS-Block-Repair', 'DL-GA-TS-Block-Repair-TrueCDM']
    data = {}
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                content = f.read()
                if content.strip(): data = json.loads(content)
        except:
            data = {}

    # ⭐ 修复乱序：对基因 Key 排序
    serializable_genes = {str(k): int(schedule_genes[k]) for k in sorted(schedule_genes.keys(), key=lambda x: int(x))}
    data[algo_name] = {"fitness": float(fitness), "schedule": serializable_genes}

    # ⭐ 修复乱序：按固定顺序重构字典
    ordered_data = {name: data[name] for name in algo_order if name in data}

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(ordered_data, f, indent=4)


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    res_dir = f"results/ablation_exp_{timestamp}"
    os.makedirs(res_dir, exist_ok=True)
    logger = setup_logger(res_dir)

    score_cols = ['f_daily', 'f_interval', 'f_room', 'f_util', 'f_build']
    common_cols = ['Instance', 'Algorithm', 'Run', 'Generations', 'Best_Fitness', 'Avg_Fitness',
                   'Violations', 'Soft_Score_Sum'] + score_cols + \
                  ['Time_s', 'Daily_Var', 'Util_Avg', 'Interval_Rate', 'Build_Conc_Rate']

    f_hist = open(os.path.join(res_dir, "history.csv"), 'w', newline='', encoding='utf-8-sig')
    csv_hist = csv.writer(f_hist)
    csv_hist.writerow(['Instance', 'Algorithm', 'Run', 'Gen'] + common_cols[4:])

    f_sum = open(os.path.join(res_dir, "summary.csv"), 'w', newline='', encoding='utf-8-sig')
    csv_sum = csv.writer(f_sum)
    csv_sum.writerow(common_cols)

    f_rl = open(os.path.join(res_dir, "rl_trace.csv"), 'w', newline='', encoding='utf-8-sig')
    csv_rl = csv.writer(f_rl)
    csv_rl.writerow(
        ['Instance', 'Algorithm', 'Run', 'Gen', 'Action', 'Reward', 'Loss', 'Mutation', 'Crossover', 'Q_Values'])

 
    instances_config = {
        'S1_Small': {'type': 'Small', 'gen': 500, 'pop': 100, 'patience': 70},
        'S2_Small': {'type': 'Small', 'gen': 600, 'pop': 100, 'patience': 70},
        'S3_Small': {'type': 'Small', 'gen': 700, 'pop': 100, 'patience': 70},
        'S4_Small': {'type': 'Small', 'gen': 800, 'pop': 100, 'patience': 80},
        'M1_Medium': {'type': 'Medium', 'gen': 1200, 'pop': 150, 'patience': 100},
        'M2_Medium': {'type': 'Medium', 'gen': 1500, 'pop': 150, 'patience': 120},
        'M3_Medium': {'type': 'Medium', 'gen': 2000, 'pop': 150, 'patience': 150},
        'L1_Large': {'type': 'Large', 'gen': 3000, 'pop': 200, 'patience': 200}
    }

    try:
        for inst_name, cfg in instances_config.items():
            logger.info(f"\n📦 Loading Instance: {inst_name} ...")
            cache_file = os.path.join("data", "cache", f"{inst_name}_cache.pkl")
            if not os.path.exists(cache_file): continue
            with open(cache_file, "rb") as f:
                data = pickle.load(f)

            evaluator = FitnessEvaluator();
            evaluator.set_data(data)
            inst_path = os.path.join(res_dir, inst_name);
            os.makedirs(inst_path, exist_ok=True)
            save_instance_metadata(inst_path, inst_name, data)
            best_params = load_best_params_by_type(cfg['type'])

            # ⭐ 实例级汇总列表，确保一个实例一个 Excel
            instance_excel_data = []

            variants = {
                'DRL-GA-TS': {'use_rl_adaptive': True, 'use_block_operators': False, 'use_repair': False,
                              'use_cdm': False, 'use_tabu': True},
                'DL-GA-TS-Block': {'use_rl_adaptive': True, 'use_block_operators': True, 'use_repair': False,
                                   'use_cdm': False, 'use_tabu': True},
                'DL-GA-TS-Block-Repair': {'use_rl_adaptive': True, 'use_block_operators': True, 'use_repair': True,
                                          'use_cdm': False, 'use_tabu': True},
                'DL-GA-TS-Block-Repair-TrueCDM': {'use_rl_adaptive': True, 'use_block_operators': True,
                                                  'use_repair': True, 'use_cdm': True, 'use_tabu': True}
            }

            for algo_name, flags in variants.items():
                algo_best_fit = -float('inf');
                algo_best_genes = None

                for run_id in range(1, N_RUNS + 1):
                    start_t = time.time()
                    config = {
                        'use_heuristic_init': True, 'elite_count': 2,
                        'crossover_rate': best_params['crossover_rate'],
                        'mutation_rate': best_params['mutation_rate'],
                        'metric_interval': 50, 'patience': cfg['patience'], 'population_size': cfg['pop'],
                        'tabu_steps': 10, **flags
                    }
                    ga = GeneticAlgorithm(data, evaluator, config)

                    # 🚀 DRL 记忆热启动
                    if hasattr(ga, 'rl_controller'): ga.rl_controller.load_brain(GLOBAL_WEIGHTS_FILE)

                    best_ind, history, rl_logs = ga.run(cfg['gen'], cfg['pop'], run_id)[:3]

                    # 🚀 DRL 记忆保存
                    if hasattr(ga, 'rl_controller'): ga.rl_controller.save_brain(GLOBAL_WEIGHTS_FILE)

                    final_gen = len(history);
                    dur = time.time() - start_t
                    last_rec = history[-1] if history else {}
                    vio = evaluator.constraint_checker.check_all(best_ind.genes)['total']

                    # 核心指标提取
                    b_val = best_ind.fitness
                    a_val = last_rec.get('avg_fitness', last_rec.get('avg_fit', 0))

                    if vio > 0:
                        s_sum, d_var, u_avg, i_rate, b_rate = 0.0, 0.0, 0.0, 0.0, 0.0
                        raw_scores = {k: 0.0 for k in score_cols}
                    else:
                        s_sum_raw = last_rec.get('soft_score_sum', 0)
                        s_sum = sum(s_sum_raw.values()) if isinstance(s_sum_raw, dict) else float(s_sum_raw)
                        d_var = last_rec.get('Daily_Var', 0)
                        if d_var == 0:
                            mets = evaluator.calculate_quality_metrics(best_ind.genes)
                            d_var, u_avg, i_rate, b_rate = mets['Daily_Var'], mets['Util_Avg'], mets['Interval_Rate'], \
                                                           mets['Build_Conc_Rate']
                        else:
                            u_avg, i_rate, b_rate = last_rec.get('Util_Avg', 0), last_rec.get('Interval_Rate',
                                                                                              0), last_rec.get(
                                'Build_Conc_Rate', 0)
                        raw_scores = last_rec.get('f_daily') and {k: last_rec.get(k, 0) for k in
                                                                  score_cols} or evaluator._calculate_raw_scores(
                            best_ind.genes)

                    if b_val > algo_best_fit: algo_best_fit, algo_best_genes = b_val, best_ind.genes.copy()

                    row = [inst_name, algo_name, run_id, final_gen, f"{b_val:.2f}", f"{a_val:.2f}", vio, f"{s_sum:.2f}"]
                    row.extend([f"{raw_scores.get(k, 0):.2f}" for k in score_cols])
                    row.extend([f"{dur:.2f}", f"{d_var:.4f}", f"{u_avg:.4f}", f"{i_rate:.4f}", f"{b_rate:.4f}"])
                    csv_sum.writerow(row);
                    f_sum.flush()

                    for h in history:
                        h_b = h.get('best_fitness', h.get('best_fit', 0))
                        h_a = h.get('avg_fitness', h.get('avg_fit', 0))
                        h_v = h.get('hard_violations', h.get('violations', 0))
                        h_s_raw = h.get('soft_score_sum', 0)
                        h_s = sum(h_s_raw.values()) if isinstance(h_s_raw, dict) else float(h_s_raw)
                        h_row = [inst_name, algo_name, run_id, h.get('gen', 0), f"{h_b:.2f}", f"{h_a:.2f}", h_v,
                                 f"{h_s:.2f}"]
                        h_row.extend([f"{h.get(k, 0):.2f}" for k in score_cols])
                        h_row.extend(
                            [f"{h.get('time_s', 0):.2f}", f"{h.get('Daily_Var', 0):.4f}", f"{h.get('Util_Avg', 0):.4f}",
                             f"{h.get('Interval_Rate', 0):.4f}", f"{h.get('Build_Conc_Rate', 0):.4f}"])
                        csv_hist.writerow(h_row)
                    f_hist.flush()

                    if rl_logs:
                        for log in rl_logs:
                            loss_v = "" if log.get('loss') is None else f"{log['loss']}"
                            csv_rl.writerow(
                                [inst_name, algo_name, run_id, log['gen'], log['action'], f"{log['reward']:.4f}",
                                 loss_v, f"{log['mutation']:.4f}", f"{log['crossover']:.4f}", str(log['q_values'])])
                        f_rl.flush()

                    excel_dict = {"Instance": inst_name, "Algorithm": algo_name, "Run": run_id,
                                  "Generations": final_gen, "Best_Fitness": b_val, "Avg_Fitness": a_val,
                                  "Violations": vio, "Soft_Score_Sum": s_sum, "Time_s": dur, "Daily_Var": d_var,
                                  "Util_Avg": u_avg, "Interval_Rate": i_rate, "Build_Conc_Rate": b_rate}
                    excel_dict.update(raw_scores);
                    instance_excel_data.append(excel_dict)

                if algo_best_genes:
                    update_best_schedules_json(os.path.join(inst_path, "best_schedules.json"), algo_name, algo_best_fit,
                                               algo_best_genes)

            if instance_excel_data:
                df_raw = pd.DataFrame(instance_excel_data).reindex(columns=common_cols)
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
    logger.info("=== ✅ 所有消融实验执行完毕 ===")


if __name__ == "__main__":
    main()
