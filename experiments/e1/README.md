# E1：单数据中心实验底座与三种基线

> **独立发布副本：** 真实数据、完整私有归档和本机路径不随本仓库发布。请先遵循[独立发布版运行说明](../../README.md)，设置外部数据绝对路径、仓库外 YAML 和输出目录。本页使用 `artifacts/smoke-007`、默认相对数据路径及 `--cd .../upstream` 的命令均为原始私有环境的历史示例；尤其指向原始 `upstream` 的 WSL 命令不得在独立发布副本的复现流程中执行。历史归档的来源版本与哈希保持不变。

E1-1A 建立 Immediate/FIFO 底座；E1-1B 在同一 `LocalExecutor` 上增加 EDF with Backfilling 和 Carbon-Threshold/FIFO。复用 `rl_components.task.Task`、`SustainDC` 的资源分配/释放方法、`make_dc_env` 和原有 IT/HVAC 公式。绕过上游 global/deferred queue、跨区域传输、环境随机时间偏移、奖励与 RL 调度；没有训练或参数扫描入口。E1-1A 历史归档保持原样。

## E1-1B 策略与公平对照

`policies.py` 的策略只接收不可变任务规格和当前 UTC/CI。`order()` 每拍处理**全部尚未开始任务**，包括旧 backlog；执行器验证任务集合与对象身份完整。FIFO 保留队列顺序；EDF 按 `(deadline, arrival, stable_id)` 排序。三个策略共享一次遍历和资源准入，放不下时继续尝试后续任务；这不预留未来资源，不保证已启动任务或 deadline。

Carbon-Threshold 使用实验前明确填写的 `policy_params: {threshold_g_per_kwh: 200}`，单位 gCO2e/kWh。仅当 `当前 CI > 阈值` 且 `当前时间 + 15min <= deadline - duration` 时主动 defer；下一拍恰好到 latest start 可以等，剩余不足一拍、已经到达/超过 latest start 或任务过期则停止主动 defer。没有未来 CI、分位数拟合、拥塞下 deadline 保证或终点强制启动。

等待归因为 **capacity-first per task turn**：该任务轮到时装不下，记录 capacity waiting；装得下且满足策略条件，记录 policy defer。每任务每拍互斥；这是执行时诊断，不是反事实因果分解。旧 `wait_intervals` 继续只计容量失败。新增 `policy_ledger.jsonl`、`policy_events.jsonl`、`policy_intervals.jsonl`、`policy_summary.json` 保存分类次数、原因、排队 ID、启动 ID、决策顺序和利用率摘要，不改变原八份科学输出格式。

同一入口通过 YAML 的 `policy: immediate_fifo / edf_backfill / carbon_threshold_fifo` 选策略。EDF/FIFO 不接受碳阈值参数。所有输入、容量、SLA、时序保持不变。严格 `--replay` 要求源码/依赖相同；显式 `--frozen-from` 允许新源码/策略复用经校验的旧归档，仅能改 policy/policy_params，并记录父版本溯源：

```bash
python -m experiments.e1.run --config PATH_TO_POLICY_CONFIG.yaml --frozen-from experiments/e1/artifacts/smoke-007 --output experiments/e1/artifacts/new-policy-run
python -m experiments.e1.compare --frozen-from experiments/e1/artifacts/smoke-007 --threshold-g-per-kwh 200 --output experiments/e1/artifacts/new-three-policy-comparison
```

对照入口恰好执行三次，校验冻结数据/物理配置哈希、共享配置、当前源码一致，并逐字节校验 FIFO 的八份科学文件与历史父归档。`comparison.csv` 同时报告服务量、截止时间状态、两类等待、队列、资源利用率和能碳；逐拍数据留在子目录。窗口总碳排放未按完成量归一，尤其完成数不同时不能单凭总碳排放宣称优劣。

## 运行（原始私有环境的历史示例）

在仓库根目录及已有 `sustaincluster` Python 环境中执行：

```bash
python -m experiments.e1.run --config experiments/e1/configs/single_dc.yaml --output experiments/e1/artifacts/baseline-review-001
python -m experiments.e1.run --replay experiments/e1/artifacts/baseline-review-001 --output experiments/e1/artifacts/replay-review-001
python -m pytest tests/e1 -q
```

本机 Windows/WSL 入口：

```powershell
wsl -d Ubuntu-22.04 --cd /mnt/d/A-myprofile/SustainCluster/upstream -- /home/hwj/miniconda3/envs/sustaincluster/bin/python -m experiments.e1.run --config experiments/e1/configs/single_dc.yaml --output experiments/e1/artifacts/baseline-review-001
```

输出目录必须尚不存在，包括空目录。一次失败会留下 `run_status.json: failed` 并抛出异常；不要将部分文件视为有效实验。成功状态为 `complete`。`artifacts/` 在本目录的 `.gitignore` 内，保留在磁盘上。原始数据 pickle 仅加载已信任的本地复现数据。

建议输出到上述 `artifacts/` 或仓库之外。位于 E1 源码树内的其他输出目录，通过启动时写入的 `run_status.json` 标记排除于源码枚举和 E1 的 pytest 发现流程；归档中的源码/测试副本不能被当成当前实现或重新执行。

默认配置为单 DC1、50,000 CPU core、1,000 GPU、80,000 GB MEM，固定 HVAC 22°C，无电池充放电。默认 UTC 窗口为 2023-08-01 05:00 至次日 05:00，15 分钟一拍。工作负载资源倍率为 5，合成 SLA multiplier 为 1.5，均由 YAML 显式参数化。E1-1A 不扫描这些参数。

## 工作负载与冻结

pickle 必须包含 `interval_15m` 和 `tasks_matrix`。矩阵每行必须恰好 12 列：

| 索引 | 字段 | 本实验用途 |
|---|---|---|
| 0 | job_name | 原始任务名，仅用于溯源；可能重复 |
| 1 | start_time | 原始时序元数据，保存在账本原始字段中 |
| 2 | end_time | 原始时序元数据 |
| 3 | start_dt | 原始时序元数据；不当作 submit time |
| 4 | duration_min | 运行时长（分钟），必须正数 |
| 5 | cpu_usage | CPU demand = 倍率 × 此列 / 100 |
| 6 | gpu_wrk_util | GPU demand = 倍率 × 此列 / 100 |
| 7 | avg_mem | MEM demand = 倍率 × 此列，沿用上游 GB 解释 |
| 8 | avg_gpu_wrk_mem | 原样保留；不作为网络流量或单独显存约束 |
| 9 | bandwidth_gb | 正确的网络元数据列；本地延迟与传输能耗均为零 |
| 10 | weekday_name | 原样保留 |
| 11 | weekday_num | 原样保留 |

CPU/GPU 是轨迹中的观测使用量转成的连续资源份额，允许分数 GPU；不等价于真实设备的整数独占申请。MEM 的物理单位与观测需求代理保持上游解释，未重新校准。倍率只缩放 CPU/GPU/MEM，不改变任务数量、到达、时长、网络量或队列规则。

原始元数据缺失值 NaN 在冻结 JSON 的对应列保存为 null（真实 smoke 有 3 个 `end_time` 缺失）。不据此删除任务或填补历史时间；duration/CPU/GPU/MEM/bandwidth 等必需执行字段仍必须有效，缺失会失败，绝不补零。源文件 SHA256 保留原始数据身份。

到达时间取 15 分钟 bucket。将实验时钟的年份替换为配置 `trace_year: 2020` 查询对应 bucket，保持月、日、时、分。只选 `[start,end)`；终点到达不会被加载。源时间范围内缺失 bucket 明确定义为无到达，超出范围失败；不循环补数据。数据集的年度重复结构沿用复现数据，不能据此声称年度真实 submit/deadline。

deadline = arrival + SLA multiplier × duration，是合成约束。稳定 ID 由源文件 SHA256 前缀、映射后的 arrival、源行位置和任务位置组成；不使用随机后缀识别任务。构造原始 `Task` 时恢复其随机数状态。配置 `seed` 留作实验溯源；本底座无任务或天气随机采样，输入在任意随机状态下相同。

`workload.json` 保存原始 12 列、全部已选任务和 SHA256。`environment.json` 保存每个区间的 UTC、碳强度、干球/湿球温度和 SHA256。碳强度与天气按真实 UTC 时间戳线性插值，禁用 roll/noise，禁止外推。湿球仍用上游 `0.9 × dry bulb` 近似，不是独立实测湿球。

天气文件含无时区的本地标签，必须有整数 `utc_offset_seconds`。配置 `weather_time_basis: metadata_fixed_offset` 按这个偏移转换到 UTC，再插值；冻结文件保存源 timezone、偏移和解释规则。本地 CAL 文件声明 UTC−7，故 `2023-07-31 22:00` 对应实验 UTC `2023-08-01 05:00`。[Open-Meteo 官方文档](https://open-meteo.com/en/docs/historical-weather-api)说明指定 timezone 后返回本地时间标签，`utc_offset_seconds` 是应用的偏移。本文件全年每天都有 24 个标签、只提供一个 −7h 偏移；本底座保留这一固定偏移语义，**未推断 IANA 夏令时切换或修复全年历史时区**。本次八月窗口不存在 DST 切换；全年/跨 DST 的天气真实性仍需重新核验或使用明确 UTC 数据。

## 执行和积分语义：相对上游的明确改变

每个左边界 `t` 按以下顺序处理：

1. 登记此刻到达任务。
2. 释放 `finish_time <= t` 的运行任务资源，登记观察到的完成。
3. 将新任务追加到本地 FIFO pending 队尾；旧 pending 顺序保留。
4. 策略对全部 pending 排序，随后公共执行器做一次遍历。Immediate/FIFO 保留原顺序，资源适配者立即启动；EDF 重排，Carbon 额外检查主动 defer 条件。不适配者留在队列，后续适配任务可回填。每拍容量失败令 `wait_intervals += 1`，策略主动 defer 不增加该旧计数。
5. 用启动后的 running 资源总量除以容量，先 `update_workloads`、再设置本拍天气、最后调用原始 `dc_env.step(22)`。
6. 在 `[t,t+15min)` 将此负载和天气/CI 保持常数并积分。

本地无传输阶段，首拍可启动并耗能。上游顶层传输能耗不参与总电能。新的 orchestration 消除了上游先计算功耗后更新负载、以及某些路径多一拍网络启动的时序差异。这会改变数值结果，不能将本底座结果直接当作上游策略结果复现。

启动/完成赋值、分配扣减、失败重入队和释放资源复用上游方法。资源准入比较改为 `math.fsum(当前 running 需求 + 候选需求) <= 容量`，与账本聚合一致，避免 `.2 + .6 + .2` 恰好装满时剩余量减法舍入产生额外一拍等待；不用超配 epsilon。available 在每次转换后由总容量减 running 聚合重算，不累计释放误差。CPU/GPU/MEM 三种恰好装满情况均有回归测试。账本对每个任务显式保存零传输延迟、电能和碳排放。

精确 finish = start + duration 用于 SLA。若任务在拍内完成，仍保留资源和整拍负载直到下一边界；`completion_observed_at` 记录边界观察时间。这是共同的离散近似，会使短任务的资源占用与能耗偏高，未实现拍内细分或部分区间积分。所有未来策略必须使用同一执行器/近似，不能改变积分口径获得优势。

终点 `T` 只释放 `finish <= T` 的任务，不新增到达、不尝试启动 pending、不多算一拍，也不 drain 队列。每个已到达任务唯一处于 pending/running/completed。超过 deadline 不取消；超过单 DC 总容量的任务仍留在账本并标记 `structurally_infeasible`。

| 终点状态 | deadline_outcome |
|---|---|
| completed 且精确 finish ≤ deadline | met |
| completed 且精确 finish > deadline | missed |
| pending/running 且 deadline ≤ T | missed_unfinished |
| pending/running 且 deadline > T | right_censored |

未完成任务的 `sla_met` 为 null，running 的 `finish_time` 是计划完成时刻，不表示已经完成。不要把 right_censored 自动算成成功，或把所有未完成任务都算为违约。

## 能源与碳排放

每拍 `IT kWh = dc_ITE_total_power_kW × 0.25h`，`HVAC kWh = dc_HVAC_total_power_kW × 0.25h`，total = IT + HVAC。碳排放 `kgCO2e = total kWh × 本拍 CI(gCO2e/kWh) / 1000`；总量逐拍相加，不能用简单平均 CI 乘电能。CI 默认是源文件 direct 列，不包括设备制造生命周期排放。空载 IT/HVAC 也计入。

沿用上游 `dc_HVAC_total_power_kW` 定义（冷却塔风机 + 压缩机），**不包括**该模型另算的 CRAC fan、CW/CT pump 等项。这里的 total 是“该上游模型边界内 IT + HVAC”，不能声称完整实测设施总电能。本阶段保留此模型边界，未重写物理公式或补算其他冷却部件。模型字段缺失、非有限数、负值或总功率不一致都会报错，不以零/旧值替代。

## 归档与复现

每次保存 `resolved_config.json`、冻结 workload/environment、`dc_config.json`、`ledger.jsonl`、`events.jsonl`、`intervals.csv`、`summary.json`、`provenance.json`、`source/` 和运行状态。provenance 包含 Git HEAD/分支/dirty 状态、配置/原始数据/冻结输入/输出 SHA256、所有上游 tracked Python 和 E1 Python 内容哈希与副本、Python 与已安装依赖版本。dirty HEAD 不能单独代表完整源码，所以保存内容快照。

回放不读取原始 workload/天气/CI 文件，而是验证归档完整性及当前源码/依赖版本，加载冻结输入和冻结物理配置，在新目录运行并比较全部科学输出字节哈希。源码或依赖改变会拒绝回放；应使用归档源码/环境另建可审核 checkout。`run_status` 的墙钟时间以及 provenance 的 Git dirty/replay 路径可不同；科学输出必须一致。

测试将 Task/资源机制与已知功率 fixture 结合，独立验证能碳单位；另用真实上游功耗适配器验证当前负载和完整回放。测试通过不证明上游 IT/HVAC 模型在真实硬件上准确，也不证明合成 SLA 或到达代理是真实用户约束。

## E1-2A：有限任务集合与真实数据排空

独立入口 `python -m experiments.e1.temporal --output <新目录>` 使用 `configs/e1_2a_selection.json`：先用不重叠的六月小时 CI 中位数确定固定阈值，再按预登记规则选择最早合格的 48 小时窗口。只在到达窗口加载任务，冻结真实环境至到达结束后 30 天的安全上限；运行通常提前排空。

E1-2A 的 CI 使用 `TemporalEnvironment`，每拍只提供时间戳不晚于当前时刻的最新小时观测，禁止旧线性插值环境被重新标记为因果输入。天气保留上述真实数据线性插值和固定偏移语义。小时 CI 在时间戳时即可获得是仿真假设，未模拟实际发布延迟。此输入口径独立于 E1-1B，旧环境加载器和归档不变。

`LocalExecutor(..., drain=True).run(stop_at_drain=True)` 在到达结束且队列清空的首个资源释放边界停止。随后把较早排空的执行器延长至三策略最晚排空边界，沿用同一个物理模型，以零 CPU/GPU/MEM 资源负载和对应真实天气/CI 计入空闲尾段。安全上限仍未完成时保留未完成状态，禁止等工作量最终减排结论。默认执行器和 `run.py` 仍执行 E1-1B 截断语义。

每策略 `phases.json` 分别统计固定到达窗口、各自排空窗口和公共窗口；`fixed_ledger.json` 重建固定截止时刻状态与等待次数。总表为 `comparison.csv`，逐拍为 `timeseries.csv`，配对开始时间诊断为 `paired_task_shifts.csv`。完成资源小时必须一致；开始时 CI 更低不等于独立的逐任务碳排放归因。有限任务 drain 不代表持续到达的稳态数据中心。

严格回放：`python -m experiments.e1.temporal --replay <归档> --output <新目录>`，核对全部 43 个科学输出、冻结输入、模型、源码和依赖。绘图：`python -m experiments.e1.plot_temporal --experiment <归档> --output <新目录>`，保存四张 PNG/PDF 和全部绘图 CSV。所有入口拒绝已存在输出目录，历史成果不覆盖。
