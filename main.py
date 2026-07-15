# # # # # """
# # # # # 系统主入口 (Final Battle - Adjusted)
# # # # # 文件名: main.py
# # #
# # # """
# # # 调试专用脚本 (Debug Experiment)
# # # 文件名: debug_experiment.py (基于 main.py 改造)
# # # 功能:
# # # 1. 极速运行所有算法 (只跑5代，种群10个)。
# # # 2. 验证所有输出格式 (CSV列对齐, Soft不为0, Time真实, JSON增量保存)。
# # # """
# # # """
# # # 调试专用脚本 (Debug Experiment - Final Fix)
# # # 文件名: main.py
# # # 修复内容:
# # # 1. KeyError: 增加 .get() 安全读取，兼容所有算法的 history 格式。
# # # 2. Excel Error: 增加列名拍平逻辑 (Flatten MultiIndex)，解决无法保存的问题。
# # # 3. Hard Logic: 再次确认 Violations > 0 时 Soft Score 为 0。
# # # """
# # # """
# # # 修改点:
# # # 1. [新增] 算法列表加入 'GA-DRL' (消融实验项)。
# # # 2. 保留所有修复: KeyError保护, Excel列名拍平, 硬约束优先逻辑。
# # # """
# # # """
# # # 1. History 写入时的 KeyError: 对 best_fitness, hard_violations 等所有字段增加多重键名兼容。
# # # 2. 保持 Excel 列名拍平修复。
# # # 3. 保持 Hard First 逻辑。
# # # """
# # # """
# # # 1. 验证所有指标 (含 Build_Conc_Rate) 是否正确输出。
# # # 2. 验证 RL Trace 是否生成。
# # # 3. 验证 Excel 格式是否正确 (拍平多级表头)。
# # # """
# # # """
# # # 调试专用脚本 (Debug Experiment - Full Algorithms Check)
# # # 文件名: main.py
# # # 功能:
# # # 1. 包含所有 7 种算法 (含基准算法)。
# # # 2. 验证 Excel 汇总是否包含所有算法。
# # # 3. 保持 N_RUNS=1, Gen=5 极速测试模式。
# # # """
# # # """
# # # 调试专用脚本 (Debug Experiment - Excel Stats Check)
# # # 文件名: main.py
# # # 功能:
# # # 1. N_RUNS = 2: 强制跑两遍，验证 Excel 中的 mean 和 std 计算是否正常。
# # # 2. 全算法覆盖: 包含 7 种算法。
# # # 3. 全指标覆盖: 包含 f_daily 等详细分和 Build_Conc_Rate。
# # # """
# # # """
# # # 调试专用脚本 (Debug Experiment - Final Ultimate Version)
# # # 文件名: main.py
# # # 修复内容:
# # # 1. [统计] N_RUNS=2, 确保 Excel 输出 Mean 和 Std。
# # # 2. [完整性] 包含所有 7 种算法 (含 Standard-GA 等基准)。
# # # 3. [指标] 包含所有 5 个软约束分 (f_daily...) 和 4 个物理指标 (Build_Conc_Rate...)。
# # # """
# #
# # """
# # 调试专用脚本 (Final Ultimate + Metadata Statistics)
# # 文件名: main.py
# # 功能:
# # 1. [元数据] 详细统计实例的学院分布、班级数、教师数、任务数、教室数。
# # 2. [保持] N_RUNS=2, 全算法, 全指标。
# # """
# # import os
# # import csv
# # import json
# # import logging
# # import time
# # import numpy as np
# # import pandas as pd
# # from datetime import datetime
# # from data.data_loader import DataLoader
# # from evaluation.fitness import FitnessEvaluator
# # from algorithms.genetic_algorithm import GeneticAlgorithm
# # from algorithms.zhu_replica import ZhuReplicaAlgorithm
# # from algorithms.ifts import IFTS
# #
# """
# 系统主入口 (Ablation Debug Version)
# 文件名: main.py
# 功能说明:
# 1. [调试模式] 默认只跑 S1_Small 实例，Pop=10, Gen=5，用于极速验证消融算子逻辑。
# 2. [消融集成] 包含 DL-GA-TS-Block, DL-GA-TS-Block-Repair, DL-GA-TS-Block-Repair-TrueCDM。
# 3. [参数尊重] 正式运行时将严格读取您的 instances_config 字典。
# 4. [逻辑保护] 保留了您原有的 KeyError 保护、Excel 列名拍平及 Violations > 0 时 Soft Score 置零的硬逻辑。
# """
"""
调试专用脚本 (Debug Experiment: Ablation Variants)
文件名: main.py
"""
import os, csv, json, logging, time, pandas as pd
from datetime import datetime
from data.data_loader import DataLoader
from evaluation.fitness import FitnessEvaluator
from algorithms.genetic_algorithm import GeneticAlgorithm

N_RUNS = 2
POP_SIZE = 10
GENS = 5
PATIENCE = 5
INTERVAL = 2
DB_CONFIG = {'host': 'localhost', 'port': 3306, 'user': 'root', 'password': os.getenv('DB_PASSWORD', ''), 'database': 'test3'}


def setup_logger(res_dir):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if logger.hasHandlers(): logger.handlers.clear()
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
    fh = logging.FileHandler(os.path.join(res_dir, "debug.log"), encoding='utf-8')
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
    meta = {'Instance': inst_name, 'College_Count': len(college_ids), 'College_IDs': college_list_str,
            'Classes_Count': len(data.classes), 'Teachers_Count': len(data.teachers), 'Tasks_Count': len(data.tasks),
            'Classrooms_Count': len(data.classrooms)}
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
    data[algo_name] = {"fitness": fitness, "schedule": {str(k): int(v) for k, v in schedule_genes.items()}}
    try:
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
        return True
    except:
        return False


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    res_dir = f"results/DEBUG_ABLATION_{timestamp}"
    os.makedirs(res_dir, exist_ok=True)
    logger = setup_logger(res_dir)

    score_cols = ['f_daily', 'f_interval', 'f_room', 'f_util', 'f_build']
    common_cols = ['Instance', 'Algorithm', 'Run', 'Generations', 'Best_Fitness', 'Avg_Fitness', 'Violations',
                   'Soft_Score_Sum'] + score_cols + ['Time_s', 'Daily_Var', 'Util_Avg', 'Interval_Rate',
                                                     'Build_Conc_Rate']

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

    inst_name = 'S1_Small'

    # ⭐ 优化: 加入 DataLoader 的异常保护，防崩溃查不到原因
    try:
        data = DataLoader(inst_name, DB_CONFIG).load()
    except Exception as e:
        logger.error(f"❌ Fatal: Failed to load debug instance {inst_name}: {e}")
        f_sum.close();
        f_hist.close();
        f_rl.close()
        return

    evaluator = FitnessEvaluator();
    evaluator.set_data(data)
    inst_path = os.path.join(res_dir, inst_name)
    os.makedirs(inst_path, exist_ok=True)
    save_instance_metadata(inst_path, inst_name, data)

    base_config = {'crossover_rate': 0.8, 'mutation_rate': 0.1, 'metric_interval': INTERVAL, 'elite_count': 2,
                   'patience': PATIENCE, 'population_size': POP_SIZE, 'use_heuristic_init': True}

    algorithms = {
        'DRL-GA-TS': {**base_config, 'use_rl_adaptive': True, 'use_block_operators': False, 'use_repair': False,
                      'use_cdm': False, 'use_tabu': True},
        'DL-GA-TS-Block': {**base_config, 'use_rl_adaptive': True, 'use_block_operators': True, 'use_repair': False,
                           'use_cdm': False, 'use_tabu': True},
        'DL-GA-TS-Block-Repair': {**base_config, 'use_rl_adaptive': True, 'use_block_operators': True,
                                  'use_repair': True, 'use_cdm': False, 'use_tabu': True},
        'DL-GA-TS-Block-Repair-TrueCDM': {**base_config, 'use_rl_adaptive': True, 'use_block_operators': True,
                                          'use_repair': True, 'use_cdm': True, 'use_tabu': True}
    }

    instance_excel_data = []

    # ⭐ 优化: try...finally 包裹，防止中途 Ctrl+C 文件写残
    try:
        for algo_name, flags in algorithms.items():
            logger.info(f"   ▶ Testing Algorithm: {algo_name}")
            algo_best_fit = -float('inf');
            algo_best_genes = None

            for run_id in range(1, N_RUNS + 1):
                start_t = time.time()
                try:
                    ga = GeneticAlgorithm(data, evaluator, flags)
                    ret = ga.run(GENS, POP_SIZE, run_id)
                    best_ind, history, rl_logs = ret[:3]
                    final_gen = len(history)
                    dur = time.time() - start_t

                    if best_ind.fitness > algo_best_fit:
                        algo_best_fit = best_ind.fitness;
                        algo_best_genes = best_ind.genes.copy()

                    last_rec = history[-1] if history else {}
                    vio = evaluator.constraint_checker.check_all(best_ind.genes)['total']

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

                        if 'f_daily' in last_rec:
                            raw_scores = {k: last_rec.get(k, 0) for k in score_cols}
                        else:
                            raw_scores = evaluator._calculate_raw_scores(best_ind.genes)

                    row = [inst_name, algo_name, run_id, final_gen, f"{best_ind.fitness:.2f}",
                           f"{last_rec.get('avg_fitness', last_rec.get('avg_fit', 0)):.2f}", vio, f"{s_sum:.2f}"]
                    row.extend([f"{raw_scores.get(k, 0):.2f}" for k in score_cols])
                    row.extend([f"{dur:.2f}", f"{d_var:.4f}", f"{u_avg:.4f}", f"{i_rate:.4f}", f"{b_rate:.4f}"])
                    csv_sum.writerow(row)
                    f_sum.flush()  # ⭐ 优化: 及时的 IO flush

                    for h in history:
                        b_fit_val = h.get('best_fitness', h.get('best_fit', 0))
                        a_fit_val = h.get('avg_fitness', h.get('avg_fit', 0))
                        vio_val = h.get('hard_violations', h.get('violations', 0))
                        s_val_raw = h.get('soft_score_sum', 0)
                        s_val = sum(s_val_raw.values()) if isinstance(s_val_raw, dict) else float(s_val_raw)

                        h_row = [inst_name, algo_name, run_id, h.get('gen', 0), f"{b_fit_val:.2f}", f"{a_fit_val:.2f}",
                                 vio_val, f"{s_val:.2f}"]
                        h_row.extend([f"{h.get(k, 0):.2f}" for k in score_cols])
                        h_row.extend(
                            [f"{h.get('time_s', 0):.2f}", f"{h.get('Daily_Var', 0):.4f}", f"{h.get('Util_Avg', 0):.4f}",
                             f"{h.get('Interval_Rate', 0):.4f}", f"{h.get('Build_Conc_Rate', 0):.4f}"])
                        csv_hist.writerow(h_row)
                    f_hist.flush()  # ⭐ 优化: 及时的 IO flush

                    if rl_logs:
                        for log in rl_logs:
                            loss_val = "" if log.get('loss') is None else f"{log['loss']}"  # ⭐ 优化: loss none 防御
                            csv_rl.writerow(
                                [inst_name, algo_name, run_id, log['gen'], log['action'], f"{log['reward']:.4f}",
                                 loss_val, f"{log['mutation']:.4f}", f"{log['crossover']:.4f}", str(log['q_values'])])
                        f_rl.flush()  # ⭐ 优化: 及时的 IO flush

                    excel_dict = {"Instance": inst_name, "Algorithm": algo_name, "Run": run_id,
                                  "Generations": final_gen, "Best_Fitness": best_ind.fitness,
                                  "Avg_Fitness": last_rec.get('avg_fitness', last_rec.get('avg_fit', 0)),
                                  "Violations": vio, "Soft_Score_Sum": s_sum, "Time_s": dur, "Daily_Var": d_var,
                                  "Util_Avg": u_avg, "Interval_Rate": i_rate, "Build_Conc_Rate": b_rate}
                    excel_dict.update(raw_scores)
                    instance_excel_data.append(excel_dict)

                except Exception as e:
                    logger.error(f"❌ Run Failed ({inst_name} | {algo_name} | Run {run_id}): {e}")
                    import traceback
                    logger.error(traceback.format_exc())

            # ⭐ 必改修复: 放回正确的作用域层级
            if algo_best_genes:
                update_best_schedules_json(os.path.join(inst_path, "best_schedules.json"), algo_name, algo_best_fit,
                                           algo_best_genes)

        if instance_excel_data:
            df_raw = pd.DataFrame(instance_excel_data)
            df_raw = df_raw.reindex(columns=common_cols)
            stats_cols = ['Generations', 'Best_Fitness', 'Avg_Fitness', 'Violations', 'Soft_Score_Sum'] + score_cols + [
                'Time_s', 'Daily_Var', 'Util_Avg', 'Interval_Rate', 'Build_Conc_Rate']
            summary_stats = df_raw.groupby('Algorithm')[stats_cols].agg(['mean', 'std']).reset_index()
            new_cols = []
            for col in summary_stats.columns:
                if isinstance(col, tuple):
                    new_cols.append(f"{col[0]}_{col[1]}" if col[1] else col[0])
                else:
                    new_cols.append(col)
            summary_stats.columns = new_cols
            with pd.ExcelWriter(os.path.join(inst_path, "report_metrics.xlsx")) as writer:
                df_raw.to_excel(writer, sheet_name='Raw_Data', index=False)
                summary_stats.to_excel(writer, sheet_name='Summary', index=False)

    finally:
        f_sum.close()
        f_hist.close()
        f_rl.close()


if __name__ == "__main__":
    main()




#
# # # === 调试配置 (验证统计版) ===
# # N_RUNS = 2  # [关键] 跑 2 次，为了出标准差 (Std)
# # POP_SIZE = 10  # 种群 10 (极速验证)
# # GENS = 5  # 代数 5 (极速验证)
# # PATIENCE = 5
# # INTERVAL = 2
# #
# # DB_CONFIG = {'host': 'localhost', 'port': 3306, 'user': 'root', 'password': os.getenv('DB_PASSWORD', ''), 'database': 'test3'}
# #
# #
# # def setup_logger(res_dir):
# #     logger = logging.getLogger()
# #     logger.setLevel(logging.INFO)
# #     if logger.hasHandlers(): logger.handlers.clear()
# #     formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
# #     fh = logging.FileHandler(os.path.join(res_dir, "debug.log"), encoding='utf-8')
# #     fh.setFormatter(formatter)
# #     logger.addHandler(fh)
# #     sh = logging.StreamHandler()
# #     sh.setFormatter(formatter)
# #     logger.addHandler(sh)
# #     return logger
# #
# #
# # def save_instance_metadata(inst_path, inst_name, data):
# #     """
# #     [新增] 详细统计实例信息并保存 (与 run_experiment.py 同步)
# #     """
# #     # 1. 统计学院信息
# #     college_ids = set()
# #     try:
# #         for c in data.classes.values():
# #             if hasattr(c, 'college_id') and c.college_id:
# #                 college_ids.add(str(c.college_id))
# #     except Exception as e:
# #         logging.error(f"Metadata warning: {e}")
# #
# #     college_list_str = "|".join(sorted(list(college_ids)))
# #
# #     meta = {
# #         'Instance': inst_name,
# #         'College_Count': len(college_ids),
# #         'College_IDs': college_list_str,
# #         'Classes_Count': len(data.classes),
# #         'Teachers_Count': len(data.teachers),
# #         'Tasks_Count': len(data.tasks),
# #         'Classrooms_Count': len(data.classrooms)
# #     }
# #
# #     meta_path = os.path.join(inst_path, "metadata.csv")
# #     pd.DataFrame([meta]).to_csv(meta_path, index=False, encoding='utf-8-sig')
# #
# #     # 控制台打印确认
# #     logging.info(f"📊 [Metadata] {inst_name}: Colleges={len(college_ids)} ({college_list_str}), "
# #                  f"Classes={len(data.classes)}, Tasks={len(data.tasks)}")
# #
# #
# # def update_best_schedules_json(json_path, algo_name, fitness, schedule_genes):
# #     data = {}
# #     if os.path.exists(json_path):
# #         try:
# #             with open(json_path, 'r', encoding='utf-8') as f:
# #                 content = f.read()
# #                 if content.strip(): data = json.loads(content)
# #         except:
# #             data = {}
# #
# #     serializable_genes = {str(k): int(v) for k, v in schedule_genes.items()}
# #     data[algo_name] = {"fitness": fitness, "schedule": serializable_genes}
# #
# #     try:
# #         with open(json_path, 'w', encoding='utf-8') as f:
# #             json.dump(data, f, indent=4)
# #         return True
# #     except:
# #         return False
# #
# #
# # def main():
# #     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# #     res_dir = f"results/DEBUG_METADATA_{timestamp}"
# #     os.makedirs(res_dir, exist_ok=True)
# #     logger = setup_logger(res_dir)
# #
# #     logger.info(f"=== 🧪 开始包含详细元数据的验证测试 (Metadata Ready) ===")
# #
# #     # 1. 定义列头
# #     score_cols = ['f_daily', 'f_interval', 'f_room', 'f_util', 'f_build']
# #     common_cols = ['Instance', 'Algorithm', 'Run', 'Generations', 'Best_Fitness', 'Avg_Fitness',
# #                    'Violations', 'Soft_Score_Sum'] + score_cols + \
# #                   ['Time_s', 'Daily_Var', 'Util_Avg', 'Interval_Rate', 'Build_Conc_Rate']
# #
# #     hist_cols = ['Instance', 'Algorithm', 'Run', 'Gen'] + common_cols[4:]
# #     sum_cols = common_cols
# #     rl_cols = ['Instance', 'Run', 'Gen', 'Action', 'Reward', 'Loss', 'Mutation', 'Crossover', 'Q_Values']
# #
# #     f_hist = open(os.path.join(res_dir, "history.csv"), 'w', newline='', encoding='utf-8-sig')
# #     csv_hist = csv.writer(f_hist)
# #     csv_hist.writerow(hist_cols)
# #
# #     f_sum = open(os.path.join(res_dir, "summary.csv"), 'w', newline='', encoding='utf-8-sig')
# #     csv_sum = csv.writer(f_sum)
# #     csv_sum.writerow(sum_cols)
# #
# #     f_rl = open(os.path.join(res_dir, "rl_trace.csv"), 'w', newline='', encoding='utf-8-sig')
# #     csv_rl = csv.writer(f_rl)
# #     csv_rl.writerow(rl_cols)
# #
# #     inst_name = 'S1_Small'
# #     logger.info(f"\n📦 Loading Instance: {inst_name} ...")
# #     try:
# #         loader = DataLoader(inst_name, DB_CONFIG)
# #         data = loader.load()
# #     except Exception as e:
# #         logger.error(f"Load Failed: {e}")
# #         return
# #
# #     evaluator = FitnessEvaluator()
# #     evaluator.set_data(data)
# #
# #     inst_path = os.path.join(res_dir, inst_name)
# #     os.makedirs(inst_path, exist_ok=True)
# #
# #     # [关键] 保存详细元数据
# #     save_instance_metadata(inst_path, inst_name, data)
# #
# #     # === 全算法列表 ===
# #     base_config = {
# #         'crossover_rate': 0.8, 'mutation_rate': 0.1,
# #         'metric_interval': INTERVAL, 'elite_count': 2,
# #         'patience': PATIENCE, 'population_size': POP_SIZE
# #     }
# #
# #     algorithms = {
# #         'DL-GA-TS': {**base_config, 'use_heuristic_init': True, 'use_rl_adaptive': True, 'use_tabu': True,
# #                      'tabu_steps': 2},
# #         'GA-DRL': {**base_config, 'use_heuristic_init': True, 'use_rl_adaptive': True, 'use_tabu': False},
# #         'Standard-GA': {**base_config, 'use_heuristic_init': False, 'use_rl_adaptive': False, 'use_tabu': False},
# #         'Heuristic-GA': {**base_config, 'use_heuristic_init': True, 'use_rl_adaptive': False, 'use_tabu': False},
# #         'Heuristic-TS': {**base_config, 'use_heuristic_init': True, 'use_rl_adaptive': False, 'use_tabu': True,
# #                          'crossover_rate': 0.0, 'mutation_rate': 0.0},
# #         'Zhu-Replica': {},
# #         'IFTS': {}
# #     }
# #
# #     instance_excel_data = []
# #
# #     for algo_name, flags in algorithms.items():
# #         logger.info(f"   ▶ Testing Algorithm: {algo_name}")
# #
# #         algo_best_fit = -float('inf')
# #         algo_best_genes = None
# #
# #         for run_id in range(1, N_RUNS + 1):
# #             start_t = time.time()
# #             best_ind = None;
# #             history = [];
# #             rl_logs = [];
# #             final_gen = 0
# #
# #             try:
# #                 if algo_name == 'Zhu-Replica':
# #                     algo = ZhuReplicaAlgorithm(data, evaluator)
# #                     best_ind, history, rl_logs = algo.run(generations=GENS)
# #                 elif algo_name == 'IFTS':
# #                     algo = IFTS(data, evaluator)
# #                     best_ind, history, rl_logs = algo.run(generations=GENS)
# #                 else:
# #                     ga = GeneticAlgorithm(data, evaluator, flags)
# #                     ret = ga.run(GENS, POP_SIZE, run_id)
# #                     if len(ret) == 4:
# #                         best_ind, history, rl_logs, _ = ret
# #                     else:
# #                         best_ind, history, rl_logs = ret
# #
# #                 final_gen = len(history)
# #                 dur = time.time() - start_t
# #                 if best_ind.fitness > algo_best_fit:
# #                     algo_best_fit = best_ind.fitness;
# #                     algo_best_genes = best_ind.genes.copy()
# #
# #                 # === Data Prep ===
# #                 last_rec = history[-1] if history else {}
# #                 final_fit = best_ind.fitness
# #                 avg_fit = last_rec.get('avg_fitness', last_rec.get('avg_fit', 0))
# #                 vio = evaluator.constraint_checker.check_all(best_ind.genes)['total']
# #
# #                 if vio > 0:
# #                     s_sum = 0.0;
# #                     d_var = 0;
# #                     u_avg = 0;
# #                     i_rate = 0;
# #                     b_rate = 0
# #                     raw_scores = {k: 0.0 for k in score_cols}
# #                 else:
# #                     s_sum_raw = last_rec.get('soft_score_sum', last_rec.get('soft_score', 0))
# #                     s_sum = sum(s_sum_raw.values()) if isinstance(s_sum_raw, dict) else s_sum_raw
# #
# #                     if 'f_daily' in last_rec:
# #                         raw_scores = {k: last_rec.get(k, 0) for k in score_cols}
# #                     else:
# #                         raw_scores = evaluator._calculate_raw_scores(best_ind.genes)
# #
# #                     d_var = last_rec.get('Daily_Var', 0)
# #                     if d_var == 0:
# #                         mets = evaluator.calculate_quality_metrics(best_ind.genes)
# #                         d_var = mets['Daily_Var'];
# #                         u_avg = mets['Util_Avg']
# #                         i_rate = mets['Interval_Rate'];
# #                         b_rate = mets['Build_Conc_Rate']
# #                     else:
# #                         u_avg = last_rec.get('Util_Avg', 0);
# #                         i_rate = last_rec.get('Interval_Rate', 0)
# #                         b_rate = last_rec.get('Build_Conc_Rate', 0)
# #
# #                 # 1. Summary Write
# #                 row = [inst_name, algo_name, run_id, final_gen, f"{final_fit:.2f}", f"{avg_fit:.2f}", vio,
# #                        f"{s_sum:.2f}"]
# #                 row.extend([f"{raw_scores.get(k, 0):.2f}" for k in score_cols])
# #                 row.extend([f"{dur:.2f}", f"{d_var:.4f}", f"{u_avg:.4f}", f"{i_rate:.4f}", f"{b_rate:.4f}"])
# #                 csv_sum.writerow(row)
# #                 f_sum.flush()
# #
# #                 # 2. History Write
# #                 for h in history:
# #                     b_fit = h.get('best_fitness', h.get('best_fit', 0))
# #                     a_fit = h.get('avg_fitness', h.get('avg_fit', 0))
# #                     h_vio = h.get('hard_violations', h.get('violations', 0))
# #                     s_val_raw = h.get('soft_score_sum', h.get('soft_score', 0))
# #                     s_val = sum(s_val_raw.values()) if isinstance(s_val_raw, dict) else s_val_raw
# #                     h_row = [inst_name, algo_name, run_id, h.get('gen', 0), f"{b_fit:.2f}", f"{a_fit:.2f}", h_vio,
# #                              f"{s_val:.2f}"]
# #                     h_row.extend([f"{h.get(k, 0):.2f}" for k in score_cols])
# #                     h_row.extend(
# #                         [f"{h.get('time_s', 0):.2f}", f"{h.get('Daily_Var', 0):.4f}", f"{h.get('Util_Avg', 0):.4f}",
# #                          f"{h.get('Interval_Rate', 0):.4f}", f"{h.get('Build_Conc_Rate', 0):.4f}"])
# #                     csv_hist.writerow(h_row)
# #                 f_hist.flush()
# #
# #                 # 3. RL Log
# #                 if rl_logs:
# #                     for log in rl_logs:
# #                         csv_rl.writerow([inst_name, run_id, log['gen'], log['action'], f"{log['reward']:.4f}",
# #                                          f"{log['loss']}", f"{log['mutation']:.4f}", f"{log['crossover']:.4f}",
# #                                          str(log['q_values'])])
# #                     f_rl.flush()
# #
# #                 # Excel Data
# #                 excel_dict = {
# #                     "Instance": inst_name, "Algorithm": algo_name, "Run": run_id,
# #                     "Generations": final_gen, "Best_Fitness": final_fit, "Avg_Fitness": avg_fit,
# #                     "Violations": vio, "Soft_Score_Sum": s_sum,
# #                     "Time_s": dur, "Daily_Var": d_var, "Util_Avg": u_avg,
# #                     "Interval_Rate": i_rate, "Build_Conc_Rate": b_rate
# #                 }
# #                 excel_dict.update(raw_scores)
# #                 instance_excel_data.append(excel_dict)
# #
# #                 logger.info(f"      Run {run_id}: Fit={final_fit:.2f} | Vio={vio} | Soft={s_sum:.1f}")
# #                 if best_ind.fitness >= algo_best_fit:
# #                     update_best_schedules_json(os.path.join(inst_path, "best_schedules.json"), algo_name, algo_best_fit,
# #                                                algo_best_genes)
# #
# #             except Exception as e:
# #                 logger.error(f"❌ Failed Run {run_id}: {e}")
# #                 import traceback
# #                 traceback.print_exc()
# #
# #     # 4. Export Excel
# #     if instance_excel_data:
# #         df_raw = pd.DataFrame(instance_excel_data)
# #         df_raw = df_raw.reindex(columns=sum_cols)
# #
# #         stats_cols = ['Generations', 'Best_Fitness', 'Avg_Fitness', 'Violations', 'Soft_Score_Sum'] + score_cols + \
# #                      ['Time_s', 'Daily_Var', 'Util_Avg', 'Interval_Rate', 'Build_Conc_Rate']
# #
# #         summary_stats = df_raw.groupby('Algorithm')[stats_cols].agg(['mean', 'std']).reset_index()
# #
# #         new_cols = []
# #         for col in summary_stats.columns:
# #             if isinstance(col, tuple):
# #                 if col[1]:
# #                     new_cols.append(f"{col[0]}_{col[1]}")
# #                 else:
# #                     new_cols.append(col[0])
# #             else:
# #                 new_cols.append(col)
# #         summary_stats.columns = new_cols
# #
# #         with pd.ExcelWriter(os.path.join(inst_path, "report_metrics.xlsx")) as writer:
# #             df_raw.to_excel(writer, sheet_name='Raw_Data', index=False)
# #             summary_stats.to_excel(writer, sheet_name='Summary', index=False)
# #         logger.info(f"✅ Excel Saved with Full Stats (Runs=2)")
# #
# #     f_sum.close();
# #     f_hist.close();
# #     f_rl.close()
# #     logger.info("=== ✅ 调试完成，请检查 Std 列是否有值 ===")
# #
# #
# # if __name__ == "__main__":
# #     main()
# #
# #
