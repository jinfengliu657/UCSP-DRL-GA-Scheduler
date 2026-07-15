# UCSP-DRL-GA-Scheduler

面向合班制大学排课的、基于深度强化学习的两阶段元启发式静态排课算法实现。

本项目对应论文《面向合班制排课的两阶段元启发式算法研究与应用》第三章“基于深度强化学习的两阶段元启发式算法静态排课求解”。算法采用“全局探索—局部强化”的协同框架：遗传进化负责种群层面的全局搜索，DQN 根据种群状态动态选择遗传搜索策略，禁忌搜索对当前最优可行解进行局部强化。

## 主要特点

- 以授课单元为基本操作对象，统一表达普通课程和合班课程。
- 使用时间槽—教室组合的紧凑型整数基因编码。
- 通过约束驱动的启发式构造提高初始种群的可行解密度。
- 使用保持授课单元完整性的块级交叉，避免破坏合班同步结构。
- 使用冲突定向变异（Conflict-Directed Mutation, CDM）定位并调整冲突单元。
- 使用合班一致性修复保证同一授课单元内部的时间和教室安排同步。
- 使用 DQN 在五种离散策略之间选择，动态调整交叉率与变异率。
- 仅在当前最优个体进入可行域后触发单元级禁忌搜索，集中优化软约束。
- 支持参数分析、消融实验、对比实验、时间预算实验和论文绘图。

## 算法框架

### 第一阶段：遗传进化全局探索

1. **紧凑型染色体编码**

   一个排课解表示为任务到“时间槽—教室”组合的映射。单个基因使用整数编码：

   ```text
   gene = (time_slot << 16) | classroom_id
   ```

2. **授课单元建模**

   合班课程中的多个任务被聚合为一个授课单元，普通课程各自构成单任务单元。交叉、变异和局部搜索均可在授课单元层面执行。

3. **约束驱动的启发式初始化**

   授课单元按照固定时间、特殊教室、合班规模等约束紧度排序，再通过随机化贪心方式依次分配时间槽和教室。启用该机制时，当前实现使用启发式方法构造约 90% 的初始个体，其余个体随机生成以保持种群多样性。这里的 90% 是初始种群构成比例，不代表硬约束覆盖率或可行率。

4. **块级交叉、冲突定向变异与修复**

   块级交叉以完整授课单元为粒度重组父代。CDM 根据硬约束检查结果定位冲突任务，将其映射到冲突频率较高的授课单元，并从多个候选时间—教室组合中保留冲突数更低的方案。合班一致性修复随后同步单元内部所有任务的基因。

5. **早停机制**

   当全局最优适应度连续若干代未达到最小改进阈值时，算法提前终止。

### DQN 自适应搜索策略

DQN 使用四维状态向量描述遗传搜索过程：

```text
[diversity, stagnation, progress, convergence_ratio]
```

- `diversity`：种群多样性；
- `stagnation`：连续未改进程度；
- `progress`：当前进化进度；
- `convergence_ratio`：平均适应度与最优适应度之比。

动作空间不是连续参数回归，而是五种离散搜索策略：

| 动作 | 交叉率 | 变异率 | 搜索倾向 |
| --- | ---: | ---: | --- |
| Hold | 保持当前值 | 保持当前值 | 维持当前搜索节奏 |
| Converge | 0.90 | 0.01 | 强化收敛 |
| Balanced | 0.80 | 0.10 | 平衡探索与开发 |
| Explore | 0.60 | 0.20 | 增强探索 |
| Emergency | 0.40 | 0.50 | 强扰动以摆脱停滞 |

DQN 采用经验回放、目标网络和 epsilon-greedy 策略进行在线学习。

### 第二阶段：禁忌搜索局部强化

禁忌搜索并非对整个种群无条件执行。当前实现仅在当前最优个体的适应度大于 0 时触发，并以该个体作为局部优化对象。

邻域移动表示为：

```text
(授课单元, 新时间槽, 新教室)
```

禁忌表使用固定长度的先进先出队列记录近期移动。若某个禁忌移动能够突破历史最优适应度，则通过藐视准则接受该移动。

## 约束与评价函数

### 硬约束

当前硬约束检查器实现以下约束类别：

- 教师时间冲突；
- 班级时间冲突；
- 教室时间冲突；
- 教室容量不足；
- 合班授课单元的时间与教室一致性；
- 固定时间要求；
- 特殊课程的教室类型要求。

### 软约束

只有硬约束违例数为 0 时才计算软约束得分。各子项均为“得分越高越好”。

| 指标 | 含义 | 默认权重 |
| --- | --- | ---: |
| `f_daily` | 班级每日课程分布均衡度 | 0.20 |
| `f_interval` | 同一授课单元多次授课的时间间隔合理性 | 0.20 |
| `f_room` | 同一授课单元多次授课的教室一致性 | 0.10 |
| `f_util` | 教室容量匹配度 | 0.30 |
| `f_build` | 班级半天内的教学楼集中度 | 0.20 |

适应度采用硬约束优先的分段评价：

```text
if violations > 0:
    fitness = -1000 * violations
else:
    fitness = 1000 * Σ(weight_i * soft_score_i)
```

## 项目结构

```text
DRL-GA-Scheduler/
├── algorithms/
│   ├── genetic_algorithm.py   # 两阶段协同主循环
│   ├── constraint_solver.py   # 约束驱动的启发式初始解构造
│   ├── combined_operators.py  # 授课单元映射与块级交叉
│   ├── combined_cdm.py        # 冲突定向变异
│   ├── combined_repair.py     # 合班一致性修复
│   ├── combined_tabu.py       # 授课单元级禁忌搜索
│   ├── rl_controller.py       # DQN 与遗传算法的控制接口
│   ├── sun_wu_tpts.py         # TPTS 轨迹搜索基线
│   ├── cuckoo_search.py       # HSCST 群体智能基线
│   ├── zhu_replica.py         # GA_RG_HH 超启发式基线
│   └── ifts.py                # Jiang-2024 两阶段基线
├── dl_module/
│   ├── dqn_agent.py           # DQN 智能体
│   └── model.py               # Q 网络
├── evaluation/
│   ├── constraints.py         # 硬约束检查与冲突任务追踪
│   └── fitness.py             # 软约束指标与适应度评价
├── data/
│   ├── data_loader.py         # 排课实例加载与领域对象
│   └── database.py            # MySQL 访问封装
├── models/chromosome.py       # 染色体表示
├── config/                    # 系统与约束配置
├── scripts/
│   ├── plots/                 # 收敛图、箱线图和实际课表可视化
│   └── diagnostics/           # 实例规模与资源压力诊断
├── main.py                    # 小规模连通性与消融变体测试入口
├── run_experiment.py          # 论文 3.6.5 五算法正式对比实验
├── run_ablation.py            # 消融实验
├── resume_ablation.py         # 消融实验恢复入口
├── run_cuckoo_all.py          # HSCST 独立批量实验
├── run_sunwu_all.py           # TPTS 独立批量实验
├── run_param_analysis.py      # 参数分析
├── run_weight_sensitivity.py  # 论文 3.6.7 正式权重灵敏性实验
├── run_time_budget.py         # 300 秒时间预算对比实验
└── test3.sql                  # 示例数据库结构与数据
```

`results/`、模型权重、日志、缓存以及绘图生成图片属于运行产物，默认不提交到 Git。

## 环境要求

- Python 3.9+
- MySQL 8.0+
- PyTorch
- NumPy、Pandas、PyMySQL、PyYAML、Matplotlib、OpenPyXL 等

安装依赖：

```bash
pip install -r requirements.txt
```

## 数据库配置

1. 在 MySQL 中导入 `test3.sql`。
2. 根据本机环境修改 `config/config.yaml` 中的主机、端口、用户名和数据库名。
3. 通过环境变量提供数据库密码。不要把真实密码写入配置文件或提交到 Git。

PowerShell：

```powershell
$env:DB_PASSWORD = "your_mysql_password"
```

Linux/macOS：

```bash
export DB_PASSWORD="your_mysql_password"
```

未设置 `DB_PASSWORD` 时，程序使用 `config/config.yaml` 中的空密码。

## 运行方式

快速检查算法模块连通性：

```bash
python main.py
```

参数分析：

```bash
python run_param_analysis.py
```

正式主流算法对比实验：

```bash
python run_experiment.py
```

该入口对齐论文第 3.6.5 节和表 3.9，统一运行以下五种方法：

| 实验名称 | 代码实现 | 论文中的方法类型/终止条件 |
| --- | --- | --- |
| `Ours` | 完整 Block + Repair + CDM + DQN + TS | 本文两阶段元启发式算法 |
| `TPTS` | `SunWuTabuSearch` | 轨迹搜索，最大迭代 105000 |
| `HSCST` | `CuckooSearchAlgorithm` | 群体智能混合，按对等计算强度设置 |
| `GA_RG_HH` | `ZhuReplicaAlgorithm` | 超启发式，100 代、种群 20 |
| `Jiang-2024` | `IFTS` | 图着色两阶段混合，200 代 |

其中 TPTS 和 HSCST 也可以独立运行：

```bash
python export_cache.py
python run_sunwu_all.py
python run_cuckoo_all.py
```

消融实验：

```bash
python run_ablation.py
```

论文第 3.6.7 节权重灵敏性实验：

```bash
python run_weight_sensitivity.py --mode local --n-runs 5
```

服务器并行运行：

```bash
python run_weight_sensitivity.py --mode server --max-workers 8 --n-runs 5
```

脚本固定使用论文中的六组权重 B0–B5，在 8 个实例上分别运行。正式运行前需要通过 `export_cache.py` 生成 `data/cache/*_cache.pkl`。

| 方案 | `(daily, interval, room, util, build)` | 侧重点 |
| --- | --- | --- |
| B0_Base | `(0.20, 0.20, 0.10, 0.30, 0.20)` | 基准均衡配置 |
| B1_Daily | `(0.30, 0.15, 0.10, 0.25, 0.20)` | 日分布均衡 |
| B2_Interval | `(0.15, 0.30, 0.10, 0.25, 0.20)` | 同课时间间隔 |
| B3_Room | `(0.15, 0.15, 0.25, 0.25, 0.20)` | 同室一致性 |
| B4_Util | `(0.15, 0.15, 0.10, 0.40, 0.20)` | 教室容量匹配 |
| B5_Build | `(0.15, 0.15, 0.10, 0.25, 0.35)` | 教学楼集中性 |

300 秒时间预算实验：

```bash
python run_time_budget.py
```

当前时间预算入口比较 Ours、TPTS 与 HSCST，每种算法在 L1 上独立运行 20 次，并输出 30、60、120、300 秒的 anytime 检查点。标准终止条件下的五算法完整比较使用 `run_experiment.py`。

绘图脚本通过命令行接收实验 CSV 路径，不包含本机绝对路径。例如：

```bash
python scripts/plots/plot_combined_tail_zoom.py results/<experiment>/history.csv results/<experiment>/summary.csv --instance L1_Large
python scripts/plots/time_fitness.py results/<experiment>/history.csv results/<experiment>/summary.csv --instances L1_Large
python scripts/plots/xiangxiantu.py results/<experiment>/summary.csv --instances S1 M1 L1_Large
```

实验规模、种群大小、最大代数、早停耐心值和算子开关主要由各实验入口脚本在运行时设置；脚本中的最终配置优先于通用 YAML 示例。

## 消融变体

当前消融入口包含以下四个逐步增强的实现变体：

1. `DRL-GA-TS`：任务级遗传算子、DQN 和禁忌搜索；
2. `DL-GA-TS-Block`：增加授课单元级块操作；
3. `DL-GA-TS-Block-Repair`：增加合班一致性修复；
4. `DL-GA-TS-Block-Repair-TrueCDM`：增加冲突定向变异，为当前完整变体。

上述名称沿用实验脚本中的标识。论文正文中统一将完整方法表述为“基于深度强化学习的两阶段元启发式算法”。

## 论文对应关系

本仓库重点对应论文第三章的静态排课求解部分。论文第四章的动态扰动重调度策略和第五章的原型系统不属于当前仓库 README 所描述的核心算法范围。
