"""
SCI级服务器断点续传脚本 - 终极防御与全局聚合版 (resume_ablation.py)
重构亮点:
1. 嵌套续跑防御: 动态扫描所有历史目录，按 mtime 锁定最新断点。
2. 全局聚合统计: 自动缝合旧断点数据与新跑数据，输出完整的 10 Runs 统计报表。
3. API 安全解包: 防范 GA 返回值突变导致的 ValueError。
4. 算力公平控制: 锁定 PyTorch 单线程，确保 CPU 耗时 (Time_s) 统计严格公平。
"""
import os
import csv
import json
import logging
import time
import pickle
import random
import numpy as np
import pandas as pd
import torch
from datetime import datetime

from evaluation.fitness import FitnessEvaluator
from algorithms.genetic_algorithm import GeneticAlgorithm

# === 全局配置 ===
N_RUNS = 10


# ==========================================
# 工具函数层 (Utility Layer)
# ==========================================
def setup_logger(res_dir):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if logger.hasHandlers(): logger.handlers.clear()
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
    fh = logging.FileHandler(os.path.join(res_dir, "resume_experiment.log"), encoding='utf-8')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger


def set_global_seed(seed_val):
    """SCI级随机种子固定，确保服务器与本地多轮Run结果绝对一致"""
    os.environ['PYTHONHASHSEED'] = str(seed_val)
    random.seed(seed_val)
    np.random.seed(seed_val)

    torch.manual_seed(seed_val)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_val)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 限制单线程，确保服务器多核环境下的 Time_s (CPU耗时) 统计绝对公平
    torch.set_num_threads(1)


def safe_float(x, default=0.0):
    """安全的浮点数转换，防御 NaN 和 None"""
    try:
        if pd.isna(x) or x is None:
            return default
        return float(x)
    except:
        return default


def safe_fmt(x, digits=2, default=0.0):
    """安全的字符串格式化，防范脏数据导致的系统崩溃"""
    val = safe_float(x, default)
    return f"{val:.{digits}f}"


def safe_close(file_obj):
    """安全回收文件句柄"""
    if file_obj is not None:
        try:
            file_obj.close()
        except:
            pass


def find_latest_summary(logger):
    """✨ 嵌套断点寻路：支持多次中断后，按修改时间精准定位最后一次的断点"""
    res_root = "results"
    if not os.path.exists(res_root):
        return None

    candidates = []
    for d in os.listdir(res_root):
        # 同时兼容原生目录和续传产生的目录
        if d.startswith("ablation_exp_") or d.startswith("ablation_RESUME_"):
            summary_path = os.path.join(res_root, d, "summary.csv")
            if os.path.exists(summary_path):
                candidates.append((os.path.getmtime(summary_path), summary_path))

    if not candidates:
        return None

    # 按时间戳取最新 (O(n) 优雅写法)
    target_path = max(candidates, key=lambda x: x[0])[1]
    logger.info(f"🔍 自动寻路: 锁定最新实验记录 -> {target_path}")
    return target_path


def load_done_runs(old_summary_path, logger):
    """加载历史断点，包含 NaN 防御机制"""
    done_runs = set()
    if old_summary_path and os.path.exists(old_summary_path):
        try:
            df_old = pd.read_csv(old_summary_path)
            for _, row in df_old.iterrows():
                if pd.notna(row['Run']):
                    done_runs.add((row['Instance'], row['Algorithm'], int(row['Run'])))
            logger.info(f"✅ 成功读取断点，已跳过 {len(done_runs)} 个完成的 Run。")
        except Exception as e:
            logger.error(f"⚠️ 读取 summary.csv 失败: {e}")
    else:
        logger.error(f"❌ 未找到历史文件！请确认 results 文件夹已完整上传。")
    return done_runs


def load_best_params_by_type(inst_type):
    """动态读取对应规模的超参数"""
    json_path = os.path.join("results", f"best_params_{inst_type.lower()}.json")
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
    data[algo_name] = {"fitness": fitness, "schedule": {str(k): int(v) for k, v in schedule_genes.items()}}
    try:
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
    except:
        pass


# ==========================================
# 实验主控制流 (Experiment Control Layer)
# ==========================================
def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    res_dir = f"results/ablation_RESUME_{timestamp}"
    os.makedirs(res_dir, exist_ok=True)
    logger = setup_logger(res_dir)
    logger.info("=== 🚀 断点续传发车: 脱机运行 + 嵌套寻路 + 全局聚合 ===")

    old_summary_path = find_latest_summary(logger)
    done_runs = load_done_runs(old_summary_path, logger)
    if not done_runs and (not old_summary_path or not os.path.exists(old_summary_path)):
        return

    score_cols = ['f_daily', 'f_interval', 'f_room', 'f_util', 'f_build']
    common_cols = ['Instance', 'Algorithm', 'Run', 'Generations', 'Best_Fitness', 'Avg_Fitness',
                   'Violations', 'Soft_Score_Sum'] + score_cols + ['Time_s', 'Daily_Var', 'Util_Avg', 'Interval_Rate',
                                                                   'Build_Conc_Rate']

    f_hist = f_sum = f_rl = None

    try:
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

        inst_name = 'L1_Large'
        cfg = {'type': 'Large', 'gen': 2000, 'pop': 200, 'patience': 150}

        cache_file = f"{inst_name}_cache.pkl"
        if os.path.exists(cache_file):
            logger.info(f"💊 发现脱水胶囊！从 {cache_file} 加载数据...")
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
        else:
            logger.error(f"❌ 致命错误: 找不到脱水胶囊 {cache_file}！")
            return

        evaluator = FitnessEvaluator()
        evaluator.set_data(data)
        inst_path = os.path.join(res_dir, inst_name)
        os.makedirs(inst_path, exist_ok=True)
        best_params = load_best_params_by_type(cfg['type'])

        variants = {
            'DRL-GA-TS': {'use_rl_adaptive': True, 'use_block_operators': False, 'use_repair': False, 'use_cdm': False,
                          'use_tabu': True},
            'DL-GA-TS-Block': {'use_rl_adaptive': True, 'use_block_operators': True, 'use_repair': False,
                               'use_cdm': False, 'use_tabu': True},
            'DL-GA-TS-Block-Repair': {'use_rl_adaptive': True, 'use_block_operators': True, 'use_repair': True,
                                      'use_cdm': False, 'use_tabu': True},
            'DL-GA-TS-Block-Repair-TrueCDM': {'use_rl_adaptive': True, 'use_block_operators': True, 'use_repair': True,
                                              'use_cdm': True, 'use_tabu': True}
        }

        instance_excel_data = []

        for algo_name, flags in variants.items():
            logger.info(f"   ▶ Algorithm: {algo_name}")
            algo_best_fit = -float('inf')
            algo_best_genes = None

            for run_id in range(1, N_RUNS + 1):
                if (inst_name, algo_name, run_id) in done_runs:
                    logger.info(f"      ⏭ 智能跳过已完成的 Run {run_id}")
                    continue

                set_global_seed(42 + run_id)

                start_t = time.time()
                try:
                    config = {'use_heuristic_init': True, 'elite_count': 2,
                              'crossover_rate': best_params['crossover_rate'],
                              'mutation_rate': best_params['mutation_rate'], 'metric_interval': 50,
                              'patience': cfg['patience'], 'population_size': cfg['pop'], 'tabu_steps': 10, **flags}
                    ga = GeneticAlgorithm(data, evaluator, config)

                    # ✨ 修复: 安全类型解包，防御 None 返回值导致的解包异常
                    ret = ga.run(cfg['gen'], cfg['pop'], run_id)
                    best_ind = history = rl_logs = None
                    if isinstance(ret, (list, tuple)):
                        best_ind = ret[0] if len(ret) > 0 else None
                        history = ret[1] if len(ret) > 1 else None
                        rl_logs = ret[2] if len(ret) > 2 else None

                    if best_ind is None:
                        logger.error(f"❌ 严重异常: Run {run_id} 返回 best_ind=None，已跳过数据写入防污染。")
                        continue

                    final_gen = len(history) if history else 0
                    dur = time.time() - start_t

                    if best_ind.fitness > algo_best_fit:
                        algo_best_fit = best_ind.fitness
                        algo_best_genes = best_ind.genes.copy()

                    last_rec = history[-1] if history else {}
                    vio = int(evaluator.constraint_checker.check_all(best_ind.genes)['total'])

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
                        raw_scores = {k: last_rec.get(k, 0) for k in
                                      score_cols} if 'f_daily' in last_rec else evaluator._calculate_raw_scores(
                            best_ind.genes)

                    row = [
                        inst_name, algo_name, run_id, final_gen,
                        safe_fmt(best_ind.fitness),
                        safe_fmt(last_rec.get('avg_fitness', last_rec.get('avg_fit', 0))),
                        vio, safe_fmt(s_sum)
                    ]
                    row.extend([safe_fmt(raw_scores.get(k, 0)) for k in score_cols])
                    row.extend([
                        safe_fmt(dur), safe_fmt(d_var, 4), safe_fmt(u_avg, 4),
                        safe_fmt(i_rate, 4), safe_fmt(b_rate, 4)
                    ])
                    csv_sum.writerow(row)
                    f_sum.flush()

                    if history:
                        for h in history:
                            s_val_raw = h.get('soft_score_sum', 0)
                            s_val = sum(s_val_raw.values()) if isinstance(s_val_raw, dict) else float(s_val_raw)
                            h_row = [
                                inst_name, algo_name, run_id, h.get('gen', 0),
                                safe_fmt(h.get('best_fitness', h.get('best_fit', 0))),
                                safe_fmt(h.get('avg_fitness', h.get('avg_fit', 0))),
                                int(h.get('hard_violations', h.get('violations', 0))),
                                safe_fmt(s_val)
                            ]
                            h_row.extend([safe_fmt(h.get(k, 0)) for k in score_cols])
                            h_row.extend([
                                safe_fmt(h.get('time_s', 0)), safe_fmt(h.get('Daily_Var', 0), 4),
                                safe_fmt(h.get('Util_Avg', 0), 4), safe_fmt(h.get('Interval_Rate', 0), 4),
                                safe_fmt(h.get('Build_Conc_Rate', 0), 4)
                            ])
                            csv_hist.writerow(h_row)
                        f_hist.flush()

                    if rl_logs:
                        for log in rl_logs:
                            loss_val = "" if log.get('loss') is None else safe_fmt(log['loss'], 4)
                            csv_rl.writerow([
                                inst_name, algo_name, run_id, log['gen'], log['action'],
                                safe_fmt(log['reward'], 4), loss_val,
                                safe_fmt(log['mutation'], 4), safe_fmt(log['crossover'], 4),
                                str(log['q_values'])
                            ])
                        f_rl.flush()

                    excel_dict = {
                        "Instance": inst_name, "Algorithm": algo_name, "Run": run_id, "Generations": final_gen,
                        "Best_Fitness": safe_float(best_ind.fitness),
                        "Avg_Fitness": safe_float(last_rec.get('avg_fitness', last_rec.get('avg_fit', 0))),
                        "Violations": vio, "Soft_Score_Sum": safe_float(s_sum), "Time_s": safe_float(dur),
                        "Daily_Var": safe_float(d_var), "Util_Avg": safe_float(u_avg),
                        "Interval_Rate": safe_float(i_rate), "Build_Conc_Rate": safe_float(b_rate)
                    }
                    excel_dict.update({k: safe_float(raw_scores.get(k, 0)) for k in score_cols})
                    instance_excel_data.append(excel_dict)

                except Exception as e:
                    logger.error(f"❌ Run Failed: {e}")

            if algo_best_genes:
                update_best_schedules_json(os.path.join(inst_path, "best_schedules_resume.json"), algo_name,
                                           algo_best_fit, algo_best_genes)

        # ==========================================
        # 结果导出层：全局数据缝合与终极报表生成
        # ==========================================
        df_new = pd.DataFrame(instance_excel_data) if instance_excel_data else pd.DataFrame()

        # ✨ 清晰逻辑: 判断是否需要进行数据合并与分析
        if not df_new.empty or old_summary_path:
            df_all = df_new
            if old_summary_path and os.path.exists(old_summary_path):
                try:
                    df_old = pd.read_csv(old_summary_path)
                    df_old_filtered = df_old[df_old['Instance'] == inst_name]
                    if not df_new.empty:
                        df_all = pd.concat([df_old_filtered, df_new], ignore_index=True)
                    else:
                        df_all = df_old_filtered
                    logger.info(f"🔗 全局数据缝合成功！共纳入 {len(df_all)} 条统计记录。")
                except Exception as e:
                    logger.error(f"⚠️ 合并旧数据失败: {e}")
                    df_all = df_new

            if not df_all.empty:
                df_all = df_all.reindex(columns=common_cols)
                stats_cols = ['Generations', 'Best_Fitness', 'Avg_Fitness', 'Violations',
                              'Soft_Score_Sum'] + score_cols + ['Time_s', 'Daily_Var', 'Util_Avg', 'Interval_Rate',
                                                                'Build_Conc_Rate']

                # 强转数值类型，防御旧 CSV 中可能存在的字符串格式数字
                for c in stats_cols:
                    df_all[c] = pd.to_numeric(df_all[c], errors='coerce')

                summary_stats = df_all.groupby('Algorithm')[stats_cols].agg(['mean', 'std']).reset_index()

                new_cols = []
                for col in summary_stats.columns:
                    if isinstance(col, tuple):
                        new_cols.append(f"{col[0]}_{col[1]}" if col[1] else col[0])
                    else:
                        new_cols.append(col)
                summary_stats.columns = new_cols

                with pd.ExcelWriter(os.path.join(inst_path, "report_metrics_resume_COMPLETE.xlsx")) as writer:
                    df_all.to_excel(writer, sheet_name='Raw_Data', index=False)
                    summary_stats.to_excel(writer, sheet_name='Summary', index=False)

        logger.info("=== ✅ 实验全部执行完毕，报表已安全落盘 ===")

    finally:
        safe_close(f_sum)
        safe_close(f_hist)
        safe_close(f_rl)


if __name__ == "__main__":
    main()