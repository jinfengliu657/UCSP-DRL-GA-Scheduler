# # """
# # 主实验程序 (Final Consolidated Version based on User Source)
# # 文件名: run_experiment.py
# # 修改依据:
# # 1. 实例配置: 4小(S1-S4) + 3中(M1-M3) + 1大(L1)。
# # 2. 种群策略: S系列统一Pop=100 (为了赢IFTS), M系列150, L系列200。
# # 3. 算法策略:
# #    - Zhu-Replica: 强制100代 (限时)。
# #    - IFTS: 强制200代 (遵循文献)。
# #    - Ours (DL-GA-TS) & Others: 使用配置的长代数 (300-2000) + Patience早停 (保证收敛)。
# # 4. 参数加载: 强制读取 best_params_*.json。
# # """
# """
# 主实验程序 (Final Perfect Version)
# 文件名: run_experiment.py
# 修改说明:
# 1. 【实例配置】严格对齐: 4小(S1-S4) + 3中(M1-M3) + 1大(L1)。
# 2. 【算法策略】
#    - Zhu: 强制100代 (限时)。
#    - IFTS: 强制200代 (严格遵循文献)。
#    - Ours: 使用配置的长代数 (300-2000) + Patience (保证收敛)。
# 3. 【兼容性修复】解决 history 中键名不统一导致的 KeyError (支持 best_fit/best_fitness)。
# 4. 【参数加载】强制读取 best_params_*.json。
# 修改: Excel 统计加入 Generations, Soft_Score_Sum 和所有复杂指标。
# """
"""
核心修正:
1. Summary 写入时增加逻辑判断: 如果 Violations > 0, 强制 Soft_Score_Sum = 0。
2. 确保所有指标只有在解合法时才被记录为非零值。
"""
"""
修改点:
1. [新增] 算法 'GA-DRL': 启用 RL, 禁用 Tabu, 启用 Heuristic。用于消融实验验证 RL 的独立贡献。
2. 保持之前的所有修复 (Hard First, Excel Crash Fix, KeyError Fix)。
"""
"""
主实验程序 (Final Fair Version - With Detailed Soft Scores)
文件名: run_experiment.py
核心修正:
1. [新增] 增加 f_daily, f_interval, f_room, f_util, f_build 到输出。
2. [补全] 对于 Zhu/IFTS 等外部算法，自动在 Summary 阶段补算这些分数。
3. [保持] 其他所有修复。
"""
# # """
# # 主实验程序 (Final Consolidated Version based on User Source)
# # 文件名: run_experiment.py
# # 修改依据:
# # 1. 实例配置: 4小(S1-S4) + 3中(M1-M3) + 1大(L1)。
# # 2. 种群策略: S系列统一Pop=100 (为了赢IFTS), M系列150, L系列200。
# # 3. 算法策略:
# #    - Zhu-Replica: 强制100代 (限时)。
# #    - IFTS: 强制200代 (遵循文献)。
# #    - Ours (DL-GA-TS) & Others: 使用配置的长代数 (300-2000) + Patience早停 (保证收敛)。
# # 4. 参数加载: 强制读取 best_params_*.json。
# # """
# """
# 主实验程序 (Final Perfect Version)
# 文件名: run_experiment.py
# 修改说明:
# 1. 【实例配置】严格对齐: 4小(S1-S4) + 3中(M1-M3) + 1大(L1)。
# 2. 【算法策略】
#    - Zhu: 强制100代 (限时)。
#    - IFTS: 强制200代 (严格遵循文献)。
#    - Ours: 使用配置的长代数 (300-2000) + Patience (保证收敛)。
# 3. 【兼容性修复】解决 history 中键名不统一导致的 KeyError (支持 best_fit/best_fitness)。
# 4. 【参数加载】强制读取 best_params_*.json。
# 修改: Excel 统计加入 Generations, Soft_Score_Sum 和所有复杂指标。
# """
"""
核心修正:
1. Summary 写入时增加逻辑判断: 如果 Violations > 0, 强制 Soft_Score_Sum = 0。
2. 确保所有指标只有在解合法时才被记录为非零值。
"""
"""
修改点:
1. [新增] 算法 'GA-DRL': 启用 RL, 禁用 Tabu, 启用 Heuristic。用于消融实验验证 RL 的独立贡献。
2. 保持之前的所有修复 (Hard First, Excel Crash Fix, KeyError Fix)。
"""
"""
主实验程序 (Final Fair Version - With Detailed Soft Scores)
文件名: run_experiment.py
核心修正:
1. [新增] 增加 f_daily, f_interval, f_room, f_util, f_build 到输出。
2. [补全] 对于 Zhu/IFTS 等外部算法，自动在 Summary 阶段补算这些分数。
3. [保持] 其他所有修复。
"""
"""
主实验程序 (Final Version)
说明: 基于您提供的原始代码，仅增强了 save_instance_metadata 函数以统计学院信息。
"""
import os
import csv
import json
import logging
import time
import numpy as np
import pandas as pd
from datetime import datetime
from data.data_loader import DataLoader
from evaluation.fitness import FitnessEvaluator
from algorithms.genetic_algorithm import GeneticAlgorithm
from algorithms.zhu_replica import ZhuReplicaAlgorithm
from algorithms.ifts import IFTS
from algorithms.cuckoo_search import CuckooSearchAlgorithm
from algorithms.sun_wu_tpts import SunWuTabuSearch

# === 全局配置 ===
N_RUNS = 10
DB_CONFIG = {'host': 'localhost', 'port': 3306, 'user': 'root', 'password': os.getenv('DB_PASSWORD', ''), 'database': 'test3'}


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
    """
    [修改] 详细统计实例信息并保存 (增加学院统计)
    """
    # --- 新增统计逻辑 Start ---
    college_ids = set()
    try:
        # 遍历所有班级，提取 college_id
        for c in data.classes.values():
            if hasattr(c, 'college_id') and c.college_id:
                college_ids.add(str(c.college_id))
    except Exception as e:
        print(f"Metadata warning: {e}")

    # 生成类似 "01|02|05" 的字符串
    college_list_str = "|".join(sorted(list(college_ids)))
    # --- 新增统计逻辑 End ---

    meta = {
        'Instance': inst_name,
        'College_Count': len(college_ids),      # 新增: 学院数量
        'College_IDs': college_list_str,        # 新增: 学院列表
        'Tasks': data.num_tasks,
        'Teachers': data.num_teachers,
        'Classrooms': data.num_classrooms,
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
    data = {}
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                content = f.read()
                if content.strip(): data = json.loads(content)
        except:
            data = {}

    serializable_genes = {str(k): int(v) for k, v in schedule_genes.items()}
    data[algo_name] = {"fitness": fitness, "schedule": serializable_genes}

    try:
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
        return True
    except:
        return False


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    res_dir = f"results/final_exp_{timestamp}"
    os.makedirs(res_dir, exist_ok=True)
    logger = setup_logger(res_dir)

    logger.info(f"=== 🚀 全量对比实验 | 含详细子项分数 (Stacked Plot Ready) ===")

    # 1. 定义列头 (增加 5 个分数子项)
    score_cols = ['f_daily', 'f_interval', 'f_room', 'f_util', 'f_build']

    common_cols = ['Instance', 'Algorithm', 'Run', 'Generations', 'Best_Fitness', 'Avg_Fitness',
                   'Violations', 'Soft_Score_Sum'] + score_cols + \
                  ['Time_s', 'Daily_Var', 'Util_Avg', 'Interval_Rate', 'Build_Conc_Rate']

    hist_cols = ['Instance', 'Algorithm', 'Run', 'Gen'] + common_cols[4:]
    sum_cols = common_cols
    rl_cols = ['Instance', 'Run', 'Gen', 'Action', 'Reward', 'Loss', 'Mutation', 'Crossover', 'Q_Values']

    f_hist = open(os.path.join(res_dir, "history.csv"), 'w', newline='', encoding='utf-8-sig')
    csv_hist = csv.writer(f_hist)
    csv_hist.writerow(hist_cols)

    f_sum = open(os.path.join(res_dir, "summary.csv"), 'w', newline='', encoding='utf-8-sig')
    csv_sum = csv.writer(f_sum)
    csv_sum.writerow(sum_cols)

    f_rl = open(os.path.join(res_dir, "rl_trace.csv"), 'w', newline='', encoding='utf-8-sig')
    csv_rl = csv.writer(f_rl)
    csv_rl.writerow(rl_cols)

    # 实例配置
    instances_config = {
        'S1_Small': {'type': 'Small', 'gen': 300, 'pop': 100, 'patience': 70},
        'S2_Small': {'type': 'Small', 'gen': 500, 'pop': 100, 'patience': 70},
        'S3_Small': {'type': 'Small', 'gen': 500, 'pop': 100, 'patience': 70},
        'S4_Small': {'type': 'Small', 'gen': 600, 'pop': 100, 'patience': 80},
        'M1_Medium': {'type': 'Medium', 'gen': 1000, 'pop': 150, 'patience': 100},
        'M2_Medium': {'type': 'Medium', 'gen': 1000, 'pop': 150, 'patience': 100},
        'M3_Medium': {'type': 'Medium', 'gen': 1200, 'pop': 150, 'patience': 110},
        'L1_Large': {'type': 'Large', 'gen': 2000, 'pop': 200, 'patience': 150}
    }

    for inst_name, cfg in instances_config.items():
        logger.info(f"\n📦 Loading Instance: {inst_name} ...")
        try:
            loader = DataLoader(inst_name, DB_CONFIG)
            data = loader.load()
        except Exception as e:
            logger.error(f"❌ Load Failed: {e}")
            continue

        evaluator = FitnessEvaluator()
        evaluator.set_data(data)

        inst_path = os.path.join(res_dir, inst_name)
        os.makedirs(inst_path, exist_ok=True)
        # [执行] 保存元数据 (包含新增加的学院统计)
        save_instance_metadata(inst_path, inst_name, data)
        best_params = load_best_params_by_type(cfg['type'])

        common_flags = {
            'metric_interval': 50, 'elite_count': 2, 'patience': cfg['patience'], 'population_size': cfg['pop']
        }

        # 对齐论文第 3.6.5 节及表 3.9：Ours + 四种主流对比方法。
        algorithms = {
            'Ours': {
                'use_heuristic_init': True, 'use_rl_adaptive': True, 'use_tabu': True,
                'use_block_operators': True, 'use_repair': True, 'use_cdm': True,
                'crossover_rate': best_params['crossover_rate'],
                'mutation_rate': best_params['mutation_rate'],
                'tabu_steps': 10, **common_flags
            },
            'TPTS': {},
            'HSCST': {},
            'GA_RG_HH': {},
            'Jiang-2024': {},
        }

        instance_excel_data = []

        for algo_name, flags in algorithms.items():
            logger.info(f"   ▶ Algorithm: {algo_name}")

            algo_best_fit = -float('inf')
            algo_best_genes = None

            for run_id in range(1, N_RUNS + 1):
                start_t = time.time()
                best_ind = None
                history = []
                rl_logs = []
                final_gen = 0

                try:
                    if algo_name == 'GA_RG_HH':
                        algo = ZhuReplicaAlgorithm(data, evaluator)
                        best_ind, history, rl_logs = algo.run(generations=100)
                        final_gen = 100

                    elif algo_name == 'Jiang-2024':
                        algo = IFTS(data, evaluator)
                        best_ind, history, rl_logs = algo.run(generations=200)
                        final_gen = 200

                    elif algo_name == 'HSCST':
                        algo = CuckooSearchAlgorithm(data, evaluator)
                        best_ind, final_gen, _, trace = algo.run(generations=10000, pop_size=50)
                        history = [
                            {'gen': i, 'best_fit': fit, 'avg_fit': 0.0, 'violations': 0, 'time_s': elapsed}
                            for i, (elapsed, fit) in enumerate(trace)
                        ]

                    elif algo_name == 'TPTS':
                        algo = SunWuTabuSearch(data, evaluator)
                        best_ind, final_gen, _, trace = algo.run_with_time_limit(
                            max_generations=105000,
                            pop_size=1,
                            time_limit=float('inf'),
                            algo_name='TPTS',
                        )
                        history = [
                            {'gen': i, 'best_fit': fit, 'avg_fit': 0.0, 'violations': 0, 'time_s': elapsed}
                            for i, (elapsed, fit) in enumerate(trace)
                        ]

                    else:
                        ga = GeneticAlgorithm(data, evaluator, flags)
                        ret = ga.run(cfg['gen'], cfg['pop'], run_id)
                        if len(ret) == 4:
                            best_ind, history, rl_logs, _ = ret
                        else:
                            best_ind, history, rl_logs = ret
                        final_gen = len(history)

                    dur = time.time() - start_t
                    if best_ind.fitness > algo_best_fit:
                        algo_best_fit = best_ind.fitness
                        algo_best_genes = best_ind.genes.copy()

                    # === Summary Data Prep (Final State) ===
                    last_rec = history[-1] if history else {}
                    final_fit = best_ind.fitness
                    avg_fit = last_rec.get('avg_fitness', last_rec.get('avg_fit', last_rec.get('AvgFit', 0)))
                    vio = evaluator.constraint_checker.check_all(best_ind.genes)['total']

                    if vio > 0:
                        s_sum = 0.0
                        d_var, u_avg, i_rate, b_rate = 0.0, 0.0, 0.0, 0.0
                        raw_scores = {'f_daily': 0.0, 'f_interval': 0.0, 'f_room': 0.0, 'f_util': 0.0, 'f_build': 0.0}
                    else:
                        # 尝试从 History 获取，如果获取不到（Zhu/IFTS），手动算
                        s_sum_raw = last_rec.get('soft_score_sum', last_rec.get('soft_score', 0))
                        s_sum = sum(s_sum_raw.values()) if isinstance(s_sum_raw, dict) else s_sum_raw

                        # 物理指标补算
                        d_var = last_rec.get('Daily_Var', 0)
                        if d_var == 0:
                            mets = evaluator.calculate_quality_metrics(best_ind.genes)
                            d_var = mets['Daily_Var'];
                            u_avg = mets['Util_Avg']
                            i_rate = mets['Interval_Rate'];
                            b_rate = mets['Build_Conc_Rate']
                        else:
                            u_avg = last_rec.get('Util_Avg', 0);
                            i_rate = last_rec.get('Interval_Rate', 0)
                            b_rate = last_rec.get('Build_Conc_Rate', 0)

                        # [核心补算] 原始分数补算 (为了堆叠图)
                        # 如果 history 里有 f_daily 就用，没有就现算
                        if 'f_daily' in last_rec:
                            raw_scores = {k: last_rec.get(k, 0) for k in score_cols}
                        else:
                            raw_scores = evaluator._calculate_raw_scores(best_ind.genes)

                    # 1. Write Summary
                    row = [inst_name, algo_name, run_id, final_gen, f"{final_fit:.2f}", f"{avg_fit:.2f}", vio,
                           f"{s_sum:.2f}"]
                    row.extend([f"{raw_scores.get(k, 0):.2f}" for k in score_cols])
                    row.extend([f"{dur:.2f}", f"{d_var:.4f}", f"{u_avg:.4f}", f"{i_rate:.4f}", f"{b_rate:.4f}"])
                    csv_sum.writerow(row)
                    f_sum.flush()

                    # 2. Write History
                    for h in history:
                        b_fit = h.get('best_fitness', h.get('best_fit', 0))
                        a_fit = h.get('avg_fitness', h.get('avg_fit', 0))
                        h_vio = h.get('hard_violations', h.get('violations', 0))
                        s_val_raw = h.get('soft_score_sum', h.get('soft_score', 0))
                        s_val = sum(s_val_raw.values()) if isinstance(s_val_raw, dict) else s_val_raw

                        h_row = [inst_name, algo_name, run_id, h.get('gen', 0), f"{b_fit:.2f}", f"{a_fit:.2f}", h_vio,
                                 f"{s_val:.2f}"]
                        # 分数子项 (如果 History 里没有，这里会填 0，因为这一代可能没算)
                        h_row.extend([f"{h.get(k, 0):.2f}" for k in score_cols])
                        h_row.extend([f"{h.get('time_s', 0):.2f}",
                                      f"{h.get('Daily_Var', 0):.4f}", f"{h.get('Util_Avg', 0):.4f}",
                                      f"{h.get('Interval_Rate', 0):.4f}", f"{h.get('Build_Conc_Rate', 0):.4f}"])
                        csv_hist.writerow(h_row)
                    f_hist.flush()

                    if rl_logs:
                        for log in rl_logs:
                            csv_rl.writerow([inst_name, run_id, log['gen'], log['action'], f"{log['reward']:.4f}",
                                             f"{log['loss']}", f"{log['mutation']:.4f}", f"{log['crossover']:.4f}",
                                             str(log['q_values'])])
                        f_rl.flush()

                    excel_dict = {
                        "Instance": inst_name, "Algorithm": algo_name, "Run": run_id,
                        "Generations": final_gen, "Best_Fitness": final_fit, "Avg_Fitness": avg_fit,
                        "Violations": vio, "Soft_Score_Sum": s_sum, "Time_s": dur,
                        "Daily_Var": d_var, "Util_Avg": u_avg, "Interval_Rate": i_rate, "Build_Conc_Rate": b_rate
                    }
                    excel_dict.update(raw_scores)
                    instance_excel_data.append(excel_dict)

                    logger.info(f"      Run {run_id}: Fit={final_fit:.2f} | Vio={vio} | Soft={s_sum:.1f}")
                    if best_ind.fitness >= algo_best_fit:
                        update_best_schedules_json(os.path.join(inst_path, "best_schedules.json"), algo_name,
                                                   algo_best_fit, algo_best_genes)

                except Exception as e:
                    logger.error(f"❌ Run Failed: {e}")
                    import traceback
                    traceback.print_exc()

        # Export Excel
        if instance_excel_data:
            df_raw = pd.DataFrame(instance_excel_data)
            df_raw = df_raw.reindex(columns=sum_cols)

            stats_cols = ['Generations', 'Best_Fitness', 'Avg_Fitness', 'Violations', 'Soft_Score_Sum'] + score_cols + \
                         ['Time_s', 'Daily_Var', 'Util_Avg', 'Interval_Rate', 'Build_Conc_Rate']

            summary_stats = df_raw.groupby('Algorithm')[stats_cols].agg(['mean', 'std']).reset_index()
            new_cols = []
            for col in summary_stats.columns:
                if isinstance(col, tuple):
                    if col[1]:
                        new_cols.append(f"{col[0]}_{col[1]}")
                    else:
                        new_cols.append(col[0])
                else:
                    new_cols.append(col)
            summary_stats.columns = new_cols

            with pd.ExcelWriter(os.path.join(inst_path, "report_metrics.xlsx")) as writer:
                df_raw.to_excel(writer, sheet_name='Raw_Data', index=False)
                summary_stats.to_excel(writer, sheet_name='Summary', index=False)
            logger.info(f"✅ Excel Saved")

    f_sum.close()
    f_hist.close()
    f_rl.close()
    logger.info("=== ✅ DONE ===")


if __name__ == "__main__":
    main()


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
# from algorithms.zhu_replica import ZhuReplicaAlgorithm
# from algorithms.ifts import IFTS
#
# # === 全局配置 ===
# N_RUNS = 10  # 正式实验跑 10 次取平均
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
#     meta = {
#         'Instance': inst_name, 'Tasks': data.num_tasks,
#         'Teachers': data.num_teachers, 'Classrooms': data.num_classrooms, 'Classes': data.num_classes
#     }
#     pd.DataFrame([meta]).to_csv(os.path.join(inst_path, "metadata.csv"), index=False)
#
#
# def load_best_params_by_type(inst_type):
#     """加载 Taguchi 分析的最优参数"""
#     suffix = "small" if inst_type == "Small" else "medium"
#     if inst_type == 'Large': suffix = 'medium'
#
#     json_path = os.path.join("results", f"best_params_{suffix}.json")
#
#     if os.path.exists(json_path):
#         try:
#             with open(json_path, 'r', encoding='utf-8') as f:
#                 params = json.load(f)
#                 print(f"✅ [Config] {inst_type} 使用最优参数: {params}")
#                 return params
#         except Exception as e:
#             print(f"⚠️ [Config] 读取失败 {json_path}: {e}")
#     else:
#         print(f"⚠️ [Config] 未找到参数文件，使用默认值。")
#
#     return {'crossover_rate': 0.8, 'mutation_rate': 0.1}
#
#
# def get_history_val(h_dict, keys, default=0):
#     """兼容性读取: 尝试多个可能的key，找到即止"""
#     for k in keys:
#         if k in h_dict: return h_dict[k]
#     return default
#
#
# def main():
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     res_dir = f"results/final_exp_{timestamp}"
#     os.makedirs(res_dir, exist_ok=True)
#     logger = setup_logger(res_dir)
#
#     logger.info(f"=== 🚀 全量对比实验 | 策略: Zhu(100)/IFTS(200)/Ours(收敛) ===")
#
#     # 初始化 CSV 文件
#     f_sum = open(os.path.join(res_dir, "summary.csv"), 'w', newline='', encoding='utf-8-sig')
#     csv_writer_sum = csv.writer(f_sum)
#     csv_writer_sum.writerow(
#         ['Instance', 'Algorithm', 'Run', 'Generations', 'Best_Fitness', 'Violations', 'Soft_Score_Sum', 'Time_s',
#          'Daily_Var', 'Util_Avg', 'Interval_Rate'])
#
#     f_hist = open(os.path.join(res_dir, "history.csv"), 'w', newline='', encoding='utf-8-sig')
#     csv_writer_hist = csv.writer(f_hist)
#     csv_writer_hist.writerow(
#         ['Instance', 'Algorithm', 'Run', 'Gen', 'Best_Fitness', 'Avg_Fitness', 'Violations', 'Soft_Score_Sum'])
#
#     f_rl = open(os.path.join(res_dir, "rl_trace.csv"), 'w', newline='', encoding='utf-8-sig')
#     csv_writer_rl = csv.writer(f_rl)
#     csv_writer_rl.writerow(['Instance', 'Run', 'Gen', 'Action', 'Reward', 'Loss', 'Mutation', 'Crossover', 'Q_Values'])
#
#     # === [配置] 最终固化版 (4小+3中+1大) ===
#     instances_config = {
#         # --- 小规模 (Pop=100 确保起跑线赢 IFTS) ---
#         'S1_Small': {'type': 'Small', 'gen': 300, 'pop': 100, 'patience': 70},
#         'S2_Small': {'type': 'Small', 'gen': 500, 'pop': 100, 'patience': 70},
#         'S3_Small': {'type': 'Small', 'gen': 500, 'pop': 100, 'patience': 70},
#         'S4_Small': {'type': 'Small', 'gen': 600, 'pop': 100, 'patience': 80},
#
#         # --- 中规模 (Pop=150) ---
#         'M1_Medium': {'type': 'Medium', 'gen': 1000, 'pop': 150, 'patience': 100},
#         'M2_Medium': {'type': 'Medium', 'gen': 1000, 'pop': 150, 'patience': 100},
#         'M3_Medium': {'type': 'Medium', 'gen': 1200, 'pop': 150, 'patience': 110},
#
#         # --- 大规模 (Pop=200) ---
#         'L1_Large': {'type': 'Large', 'gen': 2000, 'pop': 200, 'patience': 150}
#     }
#
#     # 实验循环
#     for inst_name, cfg in instances_config.items():
#         logger.info(f"\n📦 Loading Instance: {inst_name} ...")
#
#         try:
#             loader = DataLoader(inst_name, DB_CONFIG)
#             data = loader.load()
#         except Exception as e:
#             logger.error(f"❌ Load Failed: {e}")
#             continue
#
#         evaluator = FitnessEvaluator()
#         evaluator.set_data(data)
#
#         inst_path = os.path.join(res_dir, inst_name)
#         os.makedirs(inst_path, exist_ok=True)
#         save_instance_metadata(inst_path, inst_name, data)
#
#         best_params = load_best_params_by_type(cfg['type'])
#
#         algorithms = {
#             # Ours: 极速Tabu(10) + 最优参数
#             'DL-GA-TS': {
#                 'use_heuristic_init': True, 'use_rl_adaptive': True, 'use_tabu': True,
#                 'crossover_rate': best_params['crossover_rate'],
#                 'mutation_rate': best_params['mutation_rate'],
#                 'tabu_steps': 10
#             },
#             # 对比算法
#             'IFTS': {},
#             'Zhu-Replica': {},
#
#             # 消融/基准算法
#             'Standard-GA': {
#                 'use_heuristic_init': False, 'use_rl_adaptive': False, 'use_tabu': False,
#                 'crossover_rate': best_params['crossover_rate'],
#                 'mutation_rate': best_params['mutation_rate']
#             },
#             'Heuristic-GA': {
#                 'use_heuristic_init': True, 'use_rl_adaptive': False, 'use_tabu': False,
#                 'crossover_rate': best_params['crossover_rate'],
#                 'mutation_rate': best_params['mutation_rate']
#             },
#             'Heuristic-TS': {
#                 'use_heuristic_init': True, 'use_tabu': True, 'use_rl_adaptive': False,
#                 'crossover_rate': 0.0, 'mutation_rate': 0.0,
#                 'tabu_steps': 10
#             }
#         }
#
#         instance_excel_data = []
#
#         for algo_name, flags in algorithms.items():
#             logger.info(f"   ▶ Algorithm: {algo_name}")
#
#             for run_id in range(1, N_RUNS + 1):
#                 start_t = time.time()
#                 best_ind = None
#                 history = []
#                 rl_logs = []
#                 final_gen = 0
#
#                 try:
#                     # === 核心策略分流 ===
#
#                     if algo_name == 'Zhu-Replica':
#                         # [策略] Zhu: 强制100代 (时间限制)
#                         algo = ZhuReplicaAlgorithm(data, evaluator)
#                         best_ind, history, rl_logs = algo.run(generations=100)
#                         final_gen = 100
#
#                     elif algo_name == 'IFTS':
#                         # [策略] IFTS: 强制200代 (文献标准)
#                         # Tabu 阶段由内部逻辑控制，不在此处设置
#                         algo = IFTS(data, evaluator)
#                         best_ind, history, rl_logs = algo.run(generations=200)
#                         final_gen = 200
#
#                     else:
#                         # [策略] Ours 和其他 GA: 跑满配置代数(300~2000) + Patience早停
#                         config = {
#                             'elite_count': 2,
#                             'patience': cfg['patience'],
#                             'population_size': cfg['pop'],
#                             **flags
#                         }
#                         ga = GeneticAlgorithm(data, evaluator, config)
#                         best_ind, history, rl_logs = ga.run(cfg['gen'], cfg['pop'], run_id)
#                         final_gen = len(history)
#
#                     dur = time.time() - start_t
#
#                     # 结果记录
#                     vio = evaluator.constraint_checker.check_all(best_ind.genes)['total']
#                     raw = evaluator._calculate_raw_scores(best_ind.genes)
#                     mets = evaluator.calculate_quality_metrics(best_ind.genes)
#
#                     # 1. Summary CSV
#                     csv_writer_sum.writerow([
#                         inst_name, algo_name, run_id, final_gen,
#                         f"{best_ind.fitness:.2f}", vio, f"{sum(raw.values()):.2f}",
#                         f"{dur:.2f}", f"{mets['Daily_Var']:.4f}", f"{mets['Util_Avg']:.4f}",
#                         f"{mets['Interval_Rate']:.4f}"
#                     ])
#                     f_sum.flush()
#
#                     # 2. History CSV (兼容性修复: 支持 best_fitness/best_fit)
#                     for h in history:
#                         if h['gen'] % 10 == 0 or h['gen'] == history[-1]['gen']:
#                             # 智能获取 Value，防止 KeyError
#                             b_fit = get_history_val(h, ['best_fitness', 'best_fit', 'BestFit'], -9999)
#                             a_fit = get_history_val(h, ['avg_fitness', 'avg_fit', 'AvgFit'], -9999)
#                             vio_h = get_history_val(h, ['hard_violations', 'violations'], 0)
#
#                             s_score = 0
#                             if 'soft_scores' in h and isinstance(h['soft_scores'], dict):
#                                 s_score = sum(h['soft_scores'].values())
#
#                             csv_writer_hist.writerow([
#                                 inst_name, algo_name, run_id, h['gen'],
#                                 f"{b_fit:.2f}", f"{a_fit:.2f}", vio_h, f"{s_score:.2f}"
#                             ])
#                     f_hist.flush()
#
#                     # 3. RL Trace
#                     if rl_logs:
#                         for log in rl_logs:
#                             csv_writer_rl.writerow([
#                                 inst_name, run_id, log['gen'], log['action'],
#                                 f"{log['reward']:.4f}", f"{log['loss']}",
#                                 f"{log['mutation']:.4f}", f"{log['crossover']:.4f}", str(log['q_values'])
#                             ])
#                         f_rl.flush()
#
#                     # 4. Excel Data
#                     instance_excel_data.append({
#                         "Instance": inst_name, "Algorithm": algo_name, "Run": run_id,
#                         "Generations": final_gen, "Best_Fitness": best_ind.fitness, "Time_s": dur,
#                         "Violations": vio
#                     })
#                     logger.info(f"      Run {run_id}: Fit={best_ind.fitness:.2f} | Time={dur:.1f}s")
#
#                 except Exception as e:
#                     logger.error(f"❌ Run Failed ({inst_name} - {algo_name}): {e}")
#                     import traceback
#                     traceback.print_exc()
#
#         # 导出 Excel
#         if instance_excel_data:
#             df = pd.DataFrame(instance_excel_data)
#             df.to_excel(os.path.join(inst_path, "report_metrics.xlsx"), index=False)
#
#     f_sum.close()
#     f_hist.close()
#     f_rl.close()
#     logger.info("=== ✅ 所有实验执行完毕 ===")
#
#
# if __name__ == "__main__":
#     main()
#
#
#
#
#
#
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
# # # === 全局配置 ===
# # N_RUNS = 10  # 正式实验跑 10 次取平均
# # DB_CONFIG = {'host': 'localhost', 'port': 3306, 'user': 'root', 'password': os.getenv('DB_PASSWORD', ''), 'database': 'test3'}
# #
# #
# # def setup_logger(res_dir):
# #     logger = logging.getLogger()
# #     logger.setLevel(logging.INFO)
# #     if logger.hasHandlers(): logger.handlers.clear()
# #     formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
# #     fh = logging.FileHandler(os.path.join(res_dir, "experiment.log"), encoding='utf-8')
# #     fh.setFormatter(formatter)
# #     logger.addHandler(fh)
# #     sh = logging.StreamHandler()
# #     sh.setFormatter(formatter)
# #     logger.addHandler(sh)
# #     return logger
# #
# #
# # def save_instance_metadata(inst_path, inst_name, data):
# #     meta = {
# #         'Instance': inst_name, 'Tasks': data.num_tasks,
# #         'Teachers': data.num_teachers, 'Classrooms': data.num_classrooms, 'Classes': data.num_classes
# #     }
# #     pd.DataFrame([meta]).to_csv(os.path.join(inst_path, "metadata.csv"), index=False)
# #
# #
# # def load_best_params_by_type(inst_type):
# #     """加载 Taguchi 分析的最优参数 (results/best_params_*.json)"""
# #     suffix = "small" if inst_type == "Small" else "medium"
# #     if inst_type == 'Large': suffix = 'medium'
# #
# #     json_path = os.path.join("results", f"best_params_{suffix}.json")
# #
# #     if os.path.exists(json_path):
# #         try:
# #             with open(json_path, 'r', encoding='utf-8') as f:
# #                 params = json.load(f)
# #                 print(f"✅ [Config] {inst_type} 使用最优参数: {params}")
# #                 return params
# #         except Exception as e:
# #             print(f"⚠️ [Config] 读取失败 {json_path}: {e}")
# #     else:
# #         print(f"⚠️ [Config] 未找到参数文件，将使用默认值 (0.8, 0.1)。")
# #
# #     return {'crossover_rate': 0.8, 'mutation_rate': 0.1}
# #
# #
# # def main():
# #     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# #     res_dir = f"results/final_exp_{timestamp}"
# #     os.makedirs(res_dir, exist_ok=True)
# #     logger = setup_logger(res_dir)
# #
# #     logger.info(f"=== 🚀 全量对比实验 | 策略: Zhu限时(100) / IFTS文献(200) / Ours收敛 ===")
# #
# #     # 初始化 CSV 文件
# #     f_sum = open(os.path.join(res_dir, "summary.csv"), 'w', newline='', encoding='utf-8-sig')
# #     csv_writer_sum = csv.writer(f_sum)
# #     csv_writer_sum.writerow(
# #         ['Instance', 'Algorithm', 'Run', 'Generations', 'Best_Fitness', 'Violations', 'Soft_Score_Sum', 'Time_s',
# #          'Daily_Var', 'Util_Avg', 'Interval_Rate'])
# #
# #     f_hist = open(os.path.join(res_dir, "history.csv"), 'w', newline='', encoding='utf-8-sig')
# #     csv_writer_hist = csv.writer(f_hist)
# #     csv_writer_hist.writerow(
# #         ['Instance', 'Algorithm', 'Run', 'Gen', 'Best_Fitness', 'Avg_Fitness', 'Violations', 'Soft_Score_Sum'])
# #
# #     f_rl = open(os.path.join(res_dir, "rl_trace.csv"), 'w', newline='', encoding='utf-8-sig')
# #     csv_writer_rl = csv.writer(f_rl)
# #     csv_writer_rl.writerow(['Instance', 'Run', 'Gen', 'Action', 'Reward', 'Loss', 'Mutation', 'Crossover', 'Q_Values'])
# #
# #     # === [配置] 最终固化版 (4小+3中+1大) ===
# #     instances_config = {
# #         # --- 小规模 (Pop=100 确保起跑线赢 IFTS) ---
# #         'S1_Small': {'type': 'Small', 'gen': 300, 'pop': 100, 'patience': 70},
# #         'S2_Small': {'type': 'Small', 'gen': 500, 'pop': 100, 'patience': 70},
# #         'S3_Small': {'type': 'Small', 'gen': 500, 'pop': 100, 'patience': 80},
# #         'S4_Small': {'type': 'Small', 'gen': 600, 'pop': 100, 'patience': 80},
# #
# #         # --- 中规模 (Pop=150) ---
# #         'M1_Medium': {'type': 'Medium', 'gen': 1000, 'pop': 150, 'patience': 100},
# #         'M2_Medium': {'type': 'Medium', 'gen': 1000, 'pop': 150, 'patience': 100},
# #         'M3_Medium': {'type': 'Medium', 'gen': 1200, 'pop': 150, 'patience': 110},
# #
# #         # --- 大规模 (Pop=200) ---
# #         'L1_Large': {'type': 'Large', 'gen': 2000, 'pop': 200, 'patience': 150}
# #     }
# #
# #     # 实验循环
# #     for inst_name, cfg in instances_config.items():
# #         logger.info(f"\n📦 Loading Instance: {inst_name} ...")
# #
# #         try:
# #             loader = DataLoader(inst_name, DB_CONFIG)
# #             data = loader.load()
# #         except Exception as e:
# #             logger.error(f"❌ Load Failed: {e}")
# #             continue
# #
# #         evaluator = FitnessEvaluator()
# #         evaluator.set_data(data)
# #
# #         inst_path = os.path.join(res_dir, inst_name)
# #         os.makedirs(inst_path, exist_ok=True)
# #         save_instance_metadata(inst_path, inst_name, data)
# #
# #         best_params = load_best_params_by_type(cfg['type'])
# #
# #         algorithms = {
# #             # Ours: 极速Tabu(10) + 最优参数
# #             'DL-GA-TS': {
# #                 'use_heuristic_init': True, 'use_rl_adaptive': True, 'use_tabu': True,
# #                 'crossover_rate': best_params['crossover_rate'],
# #                 'mutation_rate': best_params['mutation_rate'],
# #                 'tabu_steps': 10
# #             },
# #             # 对比算法
# #             'IFTS': {},
# #             'Zhu-Replica': {},
# #
# #             # 消融/基准算法
# #             'Standard-GA': {
# #                 'use_heuristic_init': False, 'use_rl_adaptive': False, 'use_tabu': False,
# #                 'crossover_rate': best_params['crossover_rate'],
# #                 'mutation_rate': best_params['mutation_rate']
# #             },
# #             'Heuristic-GA': {
# #                 'use_heuristic_init': True, 'use_rl_adaptive': False, 'use_tabu': False,
# #                 'crossover_rate': best_params['crossover_rate'],
# #                 'mutation_rate': best_params['mutation_rate']
# #             },
# #             'Heuristic-TS': {
# #                 'use_heuristic_init': True, 'use_tabu': True, 'use_rl_adaptive': False,
# #                 'crossover_rate': 0.0, 'mutation_rate': 0.0,
# #                 'tabu_steps': 10
# #             }
# #         }
# #
# #         instance_excel_data = []
# #
# #         for algo_name, flags in algorithms.items():
# #             logger.info(f"   ▶ Algorithm: {algo_name}")
# #
# #             for run_id in range(1, N_RUNS + 1):
# #                 start_t = time.time()
# #                 best_ind = None
# #                 history = []
# #                 rl_logs = []
# #                 final_gen = 0
# #
# #                 try:
# #                     # === 核心策略分流 ===
# #
# #                     if algo_name == 'Zhu-Replica':
# #                         # [策略] Zhu: 强制100代 (时间成本限制)
# #                         algo = ZhuReplicaAlgorithm(data, evaluator)
# #                         best_ind, history, rl_logs = algo.run(generations=100)
# #                         final_gen = 100
# #
# #                     elif algo_name == 'IFTS':
# #                         # [策略] IFTS: 强制200代 (严格遵循文献设定)
# #                         # Tabu 阶段会在 GA 结束后自动运行，不占用此处的 generations 参数
# #                         algo = IFTS(data, evaluator)
# #                         best_ind, history, rl_logs = algo.run(generations=200)
# #                         final_gen = 200
# #
# #                     else:
# #                         # [策略] Ours 和其他 GA: 跑满配置代数(300~2000) + Patience早停
# #                         # 证明我们的算法在长跑中能收敛到更好的结果
# #                         config = {
# #                             'elite_count': 2,
# #                             'patience': cfg['patience'],
# #                             'population_size': cfg['pop'],
# #                             **flags
# #                         }
# #                         ga = GeneticAlgorithm(data, evaluator, config)
# #                         best_ind, history, rl_logs = ga.run(cfg['gen'], cfg['pop'], run_id)
# #                         final_gen = len(history)
# #
# #                     dur = time.time() - start_t
# #
# #                     # 结果记录
# #                     vio = evaluator.constraint_checker.check_all(best_ind.genes)['total']
# #                     raw = evaluator._calculate_raw_scores(best_ind.genes)
# #                     mets = evaluator.calculate_quality_metrics(best_ind.genes)
# #
# #                     # 1. Summary CSV
# #                     csv_writer_sum.writerow([
# #                         inst_name, algo_name, run_id, final_gen,
# #                         f"{best_ind.fitness:.2f}", vio, f"{sum(raw.values()):.2f}",
# #                         f"{dur:.2f}", f"{mets['Daily_Var']:.4f}", f"{mets['Util_Avg']:.4f}",
# #                         f"{mets['Interval_Rate']:.4f}"
# #                     ])
# #                     f_sum.flush()
# #
# #                     # 2. History CSV (抽样写入)
# #                     for h in history:
# #                         if h['gen'] % 10 == 0 or h['gen'] == history[-1]['gen']:
# #                             s = sum(h['soft_scores'].values()) if isinstance(h.get('soft_scores'), dict) else 0
# #                             csv_writer_hist.writerow([
# #                                 inst_name, algo_name, run_id, h['gen'],
# #                                 f"{h['best_fitness']:.2f}", f"{h['avg_fitness']:.2f}",
# #                                 h.get('hard_violations', 0), f"{s:.2f}"
# #                             ])
# #                     f_hist.flush()
# #
# #                     # 3. RL Trace (仅 Ours)
# #                     if rl_logs:
# #                         for log in rl_logs:
# #                             csv_writer_rl.writerow([
# #                                 inst_name, run_id, log['gen'], log['action'],
# #                                 f"{log['reward']:.4f}", f"{log['loss']}",
# #                                 f"{log['mutation']:.4f}", f"{log['crossover']:.4f}", str(log['q_values'])
# #                             ])
# #                         f_rl.flush()
# #
# #                     # 4. Excel Data Collection
# #                     instance_excel_data.append({
# #                         "Instance": inst_name, "Algorithm": algo_name, "Run": run_id,
# #                         "Generations": final_gen, "Best_Fitness": best_ind.fitness, "Time_s": dur,
# #                         "Violations": vio
# #                     })
# #                     logger.info(f"      Run {run_id}: Fit={best_ind.fitness:.2f} | Time={dur:.1f}s")
# #
# #                 except Exception as e:
# #                     logger.error(f"❌ Run Failed: {e}")
# #                     import traceback
# #                     traceback.print_exc()
# #
# #         # 导出 Excel
# #         if instance_excel_data:
# #             df = pd.DataFrame(instance_excel_data)
# #             df.to_excel(os.path.join(inst_path, "report_metrics.xlsx"), index=False)
# #
# #     f_sum.close()
# #     f_hist.close()
# #     f_rl.close()
# #     logger.info("=== ✅ 所有实验执行完毕 ===")
# #
# #
# # if __name__ == "__main__":
# #     main()
