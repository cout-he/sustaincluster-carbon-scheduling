# E0 源码溯源与调度模型审计

审计日期：2026-10-17（Asia/Shanghai）。本文件是源码静态审计；真实执行、退出码、耗时和数值结果见同目录 `environment.md`、`baseline_results.csv`、`E0-report.md` 及日志。以下风险不是本轮新增算法，也不代表已经修复。

## 1. 基线与官方对应关系

- 本轮起点分支：`research/rl-scheduling-v1`。
- 审计开始时 `HEAD` 与 `origin/main` 均为 `749dc6c1868065c12a71bac71b3ae08a25c45f56`；`git rev-list --count HEAD` 为 `1`。因此本分支确实从用户指定的 main 快照出发，起点没有其他 research 分支提交。
- 此仓库是新历史的 source-only 导出，并非官方完整 Git 历史的后代。`SOURCE_SNAPSHOT.json` 明确 `new_history=true`、`original_history_imported=false`、`data_and_weights_shipped=false`。
- 官方来源为 [HewlettPackard/sustain-cluster](https://github.com/HewlettPackard/sustain-cluster)，对应 [官方提交 3f6ea95cb835b89ba50b0ef76d66d14b8037643e](https://github.com/HewlettPackard/sustain-cluster/tree/3f6ea95cb835b89ba50b0ef76d66d14b8037643e)，提交时间 `2025-06-04T21:16:58Z`、说明 `updated readme`。审计读取相邻 `../upstream` 对应 Git 对象，不从其他 research 分支取源码。
- 清单共 `122` 条：逐条用 `git -C ../upstream show <upstream_sha>:<upstream_path>` 读取原始字节，全部通过清单的 SHA-256 和字节数校验。
- 其中 `121` 个导出文件：`120` 个在 main 基线 Git blob 中与官方逐字节相同；唯一字节差异为 `data/carbon_intensity/README.md` 的 CRLF→LF，官方 `4328` 字节，基线 `4230` 字节，正文完全一致。对这 `121` 个文件统一换行后，基线与官方 `0` 个内容差异。
- 清单的官方 `.gitignore` 对应 `release_path=null`，属于未直接导出的发布替换。另有 `README.md`、`.gitignore`、`.gitattributes` 发布覆盖；官方 README 保留为 `UPSTREAM_README.md`。这些不是调度器、物理模型或奖励修改。
- 溯源校验后发现官方数据获取示例 `data/electricity_prices/scripts/test_prices.py:6` 含硬编码凭据。本轮唯一模拟器目录之外的原有源码例外是将其替换为环境变量读取和缺失检查；不调用该联网示例、不记录凭据值、不修改 main 或重写历史。它不被实验执行路径导入，不影响任务、资源、能耗模型或奖励定义。最终变更差异须以此例外为准，不能再声称所有工作区源码逐字节保持不变。

实际执行的溯源命令包括：

```powershell
git branch --show-current
git rev-parse HEAD
git rev-parse origin/main
git rev-list --count HEAD
git -C ../upstream show -s --format='%H%n%aI%n%s' 3f6ea95cb835b89ba50b0ef76d66d14b8037643e
```

校验脚本通过 Python 标准库 `json`、`hashlib`、`pathlib`、`subprocess` 读取清单，对每条记录执行上述 `git show`，比较 `sha256(official)`、`len(official)`，再与 `git show 749dc6c...:<release_path>` 和工作区字节比较。校验第一次运行遇到清单 `.gitignore` 的空 `release_path` 后退出；第二次显式跳过该发布路径，完整校验 `122` 条成功。此错误未导致跳过其他条目或修改文件。

凭据筛查另以 AST 检查全部基线 Python blob 的凭据名称赋值（含属性/下标）和字典字段，只输出路径和行号，命中仅上述数据获取示例。这是限定规则扫描，不构成对任意文件、二进制或全部历史不存在敏感信息的证明。

## 2. 原始执行链与入口

| 层次 | 文件及一基行号 | 原始行为 |
|---|---|---|
| 仿真配置 | `configs/env/sim_config.yaml:1` | 2023 年 8 月 1 日 05:00 起，7 天、15 分钟；Alibaba 2020 年 pickle，AWS，默认 `manual_rl`，单动作、禁 defer |
| 数据中心配置 | `configs/env/datacenters.yaml:1` | US-CAL-CISO、DE-LU、CL-SIC、SG、AU-NSW 五站；CPU 核数、GPU 数、GB 内存，固定时区偏移和来源权重 |
| 常用创建入口 | `train_rl_agent.py:63` | 加载 YAML、创建 cluster 和 CompositeReward，再包成 TaskSchedulingEnv；不要调用 `train()` 作为 E0 仿真入口 |
| 官方评估创建入口 | `evaluate_sustaincluster_agent.py:118`、`scripts/generate_table1_metrics.py:115` | 支持规则/历史 RL 策略评估；有独立日期计算与汇总约定，不能自动视为科研真值 |
| 全局环境 | `envs/task_scheduling_env.py:28`、`:226`、`:255` | Gym reset/step、动作映射、defer 与在途任务、奖励和回合终止 |
| 集群内核 | `simulation/cluster_manager.py:28`、`:93`、`:134`、`:186` | 初始化 SustainDC，重置时钟和策略、按时间提取任务、规则分派或跳过分派、逐站 step 和网络计量 |
| 规则策略 | `utils/task_assignment_strategies.py:32` | most_available、random、priority_order、least_pending、lowest_carbon、round_robin、lowest_price、lowest_utilization、local_only |
| 单站任务与物理环境 | `envs/sustaindc/sustaindc_env.py:161`、`:188`、`:362` | 可行性检查、资源分配、FIFO 队列、完成释放、调用 DC 与电池模型并输出计量 |
| DC 物理模型 | `utils/make_envs.py:74`、`envs/sustaindc/dc_gym.py:159`、`envs/sustaindc/datacenter_model.py:373` | CPU、GPU、内存、IT 风扇与冷却功耗；绝对 CRAC 温度设置 |
| 电池/能源 | `envs/sustaindc/battery_env.py:83`、`:223` | 固定 15 分钟能量换算，gCO2e 碳足迹；SustainDC 固定 idle 电池动作 |

## 3. 工作负载、资源与时间单位

`DatacenterClusterManager._load_or_extract_tasks_file()`（`simulation/cluster_manager.py:358`）优先读取原始 `.pkl`，缺失则解压同名 `.zip`，然后把 `interval_15m` 转为 UTC。`get_tasks_for_timestep()`（`:134`）把当前仿真年份替换为 `2020`，精确匹配对应 15 分钟行，仅使用匹配的第一行。

`utils/workload_utils.py:48` 原始提取参数为 `scale=1`、`task_scale=5`、`group_size=1`（调用在 `simulation/cluster_manager.py:157`）。矩阵索引 `[4]` 为分钟时长；CPU 为 `5 × [5] / 100` 核，GPU 为 `5 × [6] / 100` 单位，内存为 `5 × [7]` GB；`[8]` 为传输 GB，**不乘 5**。这是官方基线已经存在的任务放大，不是本研究新造负载；未来比较必须保持相同提取代码、参数与原始文件哈希。

`assign_task_origins()`（`utils/workload_utils.py:7`）按人口权重与当地小时的活动系数随机产生来源；8–20 点系数 1，其余 0.3。`Task`（`rl_components/task.py:34`）添加 Python 随机后缀、默认 SLA 为到达后 `1.5 × duration` 分钟。因此相同原始任务行不足以保证相同来源与任务标识，必须控制 Python 和 NumPy 随机状态及数据中心遍历顺序，并比较完整任务签名。

单站按 CPU/GPU/内存三资源同时可行才开工（`sustaindc_env.py:161`）。分配时扣资源，完成时间为实际开工时间加原始 duration（`:214`）；在后续 tick 释放资源并用实际完成时间判断 SLA（`:170`）。FIFO 每 tick 每个 pending 最多尝试一次。源码注释提到超时丢弃，但实现仅增加等待计数后重新入队（`:237`），没有对应丢弃分支；不能把注释当作已实现行为。

五站合计原始容量为 `385000` 核、`2900` GPU、`330000` GB 内存。集群额定功率估计为 `6 W/核 + 500 W/GPU + 2.5 W/GB`（`cluster_manager.py:46`）；这是传入物理环境的容量配置，**不是实测能耗公式**。实际模型另有温度/利用率曲线与背景内存功率（`datacenter_model.py:507`、`:534`）。

时钟有原始不一致：wrapper 从配置日历时间开始，但 `cluster.reset()`（`:107`）把数据管理器起始日随机为配置日 ±7 天、小时随机为 0–23。创建入口用 `int(month*30.5)` 或 `int(month*30.5+init_day)` 近似年内天数，而不是日历 day-of-year（`train_rl_agent.py:77` 与官方评估入口）。wrapper 的任务日期与天气/CI/电价日期因此可能不一致。相同 seed 可重复此行为，不能证明日历对齐。本轮保留并报告该原始行为，下一阶段应先单列修复实验。

## 4. reset/step 与规则路径差异

1. wrapper reset（`task_scheduling_env.py:226`）固定任务起始时间、播种 Python RNG、调用 cluster.reset（播种 NumPy/单站）、清空 defer/在途队列，加载第一批任务，返回 `(obs, {})`。
2. wrapper step 首先交付此前在途且到达的任务；然后执行当前动作。禁 defer 时动作 `0..N-1` 映射到当前字典顺序的 DC；有 defer 时 0 为 defer，1..N 为 DC。单动作模式把整批任务路由到同一 DC；多动作模式逐任务处理。没有资源可行性动作掩码，由单站队列等候。
3. 当前路由只添加 `in_transit_tasks`（`:306`、`:354`），本 tick 不再交付。因此即使本地传输，至少下一个 15 分钟 tick 才可能进入 pending；网络延迟也按 tick 到达检查离散化。超过 SLA 的任务强制回来源队列。
4. cluster.step（`cluster_manager.py:186`）更新任务时钟。`manual_rl` 不提取/分派；规则策略则在此时提取任务并**立即**加入目标 pending，没有 wrapper 的在途队列或延迟。
5. 单站 step 先 `_perform_actions()` 计算物理功率/电池，然后推进天气/CI/价格管理器，再释放完成任务、调度 pending、更新新利用率供下一 tick 使用（`sustaindc_env.py:380`–`:425`）。因此当步资源快照与当步能耗天然存在一个 tick 的负载滞后；能源计量用的电池 CI 与已前进的 `__common__.ci` 也不能不加区别地混用。
6. wrapper 计算奖励、前进 15 分钟、加载下一批，时间达到 end 时同时令 terminated/truncated 为 true（`task_scheduling_env.py:358`–`:386`）。结束时可能还有 pending、运行和在途任务，短期结果不是所有到达任务的最终 SLA 结果。

**比较边界：**共同使用 wrapper、相同 seed、同一原始负载、同一资源和碳排内核，可以验证入口和计量一致性；原始规则模式与 manual_rl 模式的服务时序仍然不同。E0 应把“原始规则结果”“manual_rl 接口的固定规则动作结果”分别标识，不得据其差值声称算法性能改善。

## 5. 能源、碳排、电价与奖励的真实口径

| 指标 | 正确读取位置与单位 | 审计说明 |
|---|---|---|
| 站点功率 | `agent_dc.dc_total_power_kW`，kW | `dc_gym.py:231` 将 W 除 1000；总量为 IT + cooling tower + compressor，其他 HVAC 项的覆盖范围应按原始公式解释 |
| 站点能耗 | `agent_bat.bat_total_energy_with_battery_KWh` 或 `__common__.energy_consumption_kwh`，kWh/tick | `battery_env.py:223`：idle 时 `MW × 1000 × 0.25`；应与 `dc_total_power_kW × 0.25` 一致 |
| 站点碳排 | `__common__.carbon_emissions_kg`，kgCO2e/tick | `sustaindc_env.py:466`：实际电池 kWh × `bat_avg_CI`（g/kWh）/1000；亦可交叉核验 `bat_CO2_footprint/1000` |
| 电价/成本 | 原始 `Price (USD/MWh)`；`__common__.energy_cost_USD` 为 USD/tick | `managers.py:526` 读 USD/MWh，先按 15/85 分位 IQR 裁剪，再插值为 15 分钟；成本 `kWh × price/1000` 数值单位正确，但源码 `price_USD_kwh` 键名及注释具有误导性 |
| 网络成本 | `transmission_cost_total_usd`，USD/tick | `cluster_manager.py:294`：实际开始执行任务的 GB × 区域矩阵 USD/GB，不是在路由动作瞬间记账 |
| 网络能耗/排放 | `transmission_energy_total_kwh`、`transmission_emissions_total_kg` | 跨站采用 `0.06 kWh/GB` 与来源 CI，原地为 0（`:299`）；属于固定强度估计，不能当作网络实测 |
| wrapper 名为 total 的输出 | `info.total_energy_kwh`、`info.total_emissions_kg` | **实际只有网络能耗/排放**（`task_scheduling_env.py:380`）。不能用于全系统总量；应显式求五站总和再加网络 |

CI 用 `Carbon Intensity gCO₂eq/kWh (direct)` 列（`managers.py:251`），天气与 CI 插值/时区处理由原管理器完成。电价与 CI 使用的插值网格不同（CI `:262` 对比电价 `:544`）；气象、价格和 CI 时间语义需要另做时标审计。恢复文件不能替换这些官方处理逻辑或输入列。

默认奖励为未归一化 CompositeReward：energy_price 权重 `0.9`、carbon_emissions 权重 `0.3`（`configs/env/reward_config.yaml:1`）。两个组件实际遍历当前决策任务，以 `task.cores_req * task.duration / 10000` 作为任务能量代理，再分别乘目标站当前价格或 CI，最后默认除 `100`（`rewards/predefined/energy_price_reward.py:37`、`carbon_emissions_reward.py:16`）。这**没有使用实际物理模型能耗**，没有直接纳入 GPU、内存、冷却、网络或已经运行的任务，也没有给 CI 的 g→kg 换算。不能把该 reward 命名为实际 kgCO2/USD 改善。defer 任务可能无目标而跳过；在途任务有目标，即使尚未执行也计入代理奖励。规则模式的 `current_tasks` 通常为空，wrapper 奖励为零不代表没有能耗。下一阶段必须明确奖励代理与报告真值分离。

官方 `scripts/generate_table1_metrics.py:306`–`:308` 读取 wrapper 未提供的 `total_energy_cost_usd_this_step`、`total_energy_kwh_this_step`、`total_emissions_kg_this_step`，默认回退 0。这是静默零值风险；E0 的汇总应直接由逐站原始 info 字段计算并保留公式校验，不能照抄该汇总结果。

## 6. 原有测试的覆盖缺口

- `tests/test_env_setup.py:6` 仅 reset 并要求 obs 是 list。实际调用的 `train_rl_agent.make_env` 没有传递 sim_config，故回退到多动作 list 输出并通过；默认配置写出的单动作和禁 defer 选项被静默忽略。该测试没有验证预期的单动作路径。
- `tests/test_model.py:4` 仅 ActorNet/CriticNet 前向与形状检查，不覆盖调度、时钟、能耗、碳排或奖励计量。
- `tests/test_training_loop.py:7` 创建临时 YAML 后在目录上下文之外用 `os.system` 调用缺失的 `train_SAC_single_action.py`，并且不检查退出状态、没有结果断言。该测试即使显示通过，也不能证明训练运行成功。本轮不开发/训练新算法，应记录此原有测试真实行为和限制。
- 规则策略可能返回 None，但 `cluster_manager.py:177` 在判断 None 前访问 `assigned_dc.dc_id`；资源耗尽或任务过大时可能崩溃。短窗口成功不能证明长时负载安全。
- 原有源码没有为本轮要求的同任务输入、三资源守恒、DC 电量换算、kg/g 换算和网络分项提供完整测试。E0 增加的验证应作为独立实验脚本，只读原始内核并对原始输出做断言，避免为了“跑通”悄悄改物理模型。

## 7. 下一阶段的源码风险优先级

1. **高：统计与奖励错位。** wrapper 的 total 只含网络，官方汇总读不存在字段；奖励为任务 CPU 代理。先冻结计量合同，再谈算法目标。
2. **高：基准服务时序不同。** 规则即时投队与 manual_rl 延迟交付、单站负载滞后、短回合未清空队列，可能主导表观收益。
3. **高：任务日期与能源数据日期错位。** 近似 day-of-year、随机 reset 时钟、固定时区及不同插值网格需显式记录与单独验证。
4. **中：原始规则调度健壮性。** None 解引用、资源耗尽、未实现的超时丢弃，需要用真实长窗口验证后单独修复。
5. **中：可重复输入。** 来源/名称随机化、数据中心洗牌、年份映射、任务放大参数和第三方 pickle 兼容性必须写进复现清单。
6. **中：测试可信度。** reset 测试只覆盖配置被忽略的回退路径、训练测试可假通过，不能把 pytest 总退出码替代实验守恒验证。
7. **中：物理口径的外部有效性。** GPU/内存背景模型、HVAC 覆盖范围、固定网络强度和裁剪电价属于官方假设；E0 忠实复现，不为其物理精度背书。

本审计未修改调度器、环境、奖励、任务提取或物理模型；不引入 ACP-PPO 或任何新训练算法。

## 8. 可重跑的最终源码验证

执行 `.venv/Scripts/python.exe research/e0/verify_source.py --source-repo ..\upstream`，结果记录在 `source_integrity.json`。验证清单全部官方字节摘要、原始 main 的换行等价性、当前工作区源码内容、唯一凭据外置例外、main 引用以及无合并提交的基线祖先关系。验证代码不打印原始源码或凭据值。
