# E1-5B 数据与核算协议冻结

## Material Passport

- 任务：普通 PPO-v0 模式选择基线；本轮仅 Train 训练与 Validation 选型。
- 证据：E1-5A `data_split.json`、`data-audit-002/coverage_audit.json`、`integrity_checks.json`，本轮只读原始文件 SHA256 核验，以及现行 workload/stress/selection/cars/power/rl_env 源码。
- 对应机器协议：`data-protocol.json`。协议在本轮策略结果前冻结；逐次运行的 frozen snapshot/checkpoint/output 哈希由各运行清单补充，不把计划哈希写成已产生数据。
- Test 状态：继续封存。下表 Test 数量和覆盖来自前轮覆盖审计；本轮不切出、扩增、运行或评测 Test。

## 文件身份与独立性边界

| 输入（相对原科研根目录） | 大小 B | SHA256 |
|---|---:|---|
| `data/workload/alibaba_2020_dataset/result_df_full_year_2020.pkl` | 142336499 | `3ba8a8e0067288f8d2542752a292f7da04cb79305689ba145cc6060e96cb7f2e` |
| `data/carbon_intensity/US-CAL-CISO/2023/US-CAL-CISO_2023_hourly.csv` | 749471 | `44bafab775c6b0d4762b1dc971eb293cfea154f80799a5230f9268d32e1ad55b` |
| `data/weather/US-CAL-CISO/2023.json` | 912513 | `090bddc8a5f4e9edb5f3f193c6c147ed03841337b4bb17c7740f3b63a03cdaf5` |
| `configs/dcs/dc_config.json` | 5624 | `bbe439176605512523a3a0ecd2bd22f18fdb9abf6ae6e782217157beedbf705b` |

本轮只读哈希与 E1-5A 审计一致。读取整个既存容器以计算身份哈希，或现有 loader 解析容器后只选 Train/Validation，不等于向策略提供 Test 数据；正式边界是禁止提取 Test episode、拟合 Test 统计量或执行 Test 策略。训练和评测入口应先检查 split 白名单与准确时间边界，再加载；训练只允许 Train，选型只允许 Validation。

原 workload 有 37552 个 15 分钟桶，跨度 2020-01-01 至 2021-01-26。原文件由约七周 trace 重复平移构成，不能描述成独立采集的全年 2023 AI 到达轨迹。simulation UTC 时间只替换年份为 2020 后匹配桶，保留月、日、时、分。本轮窗口无缺桶，不触发 loader 的“覆盖内缺桶视作无到达”规则。原容器全局有 80 个缺桶，不能据此假定所有其他窗口均完整。

## 冻结窗口

| split | 到达起点 UTC（含） | 到达终点 UTC（不含） | 环境安全终点 UTC | 基础任务数 | 到达桶 / 必需 | CI/天气小时点 / 必需 |
|---|---|---|---|---:|---:|---:|
| Train | 2023-05-01 05:00 | 2023-05-03 05:00 | 2023-06-02 05:00 | 5542 | 192/192 | 各 769/769 |
| Validation | 2023-07-08 05:00 | 2023-07-10 05:00 | 2023-08-09 05:00 | 8708 | 192/192 | 各 769/769 |
| Test（前轮审计，未运行） | 2023-09-09 05:00 | 2023-09-11 05:00 | 2023-10-11 05:00 | 6723 | 192/192 | 各 769/769 |

每个到达窗口为 48 小时，环境延伸至到达停止后 720 小时。小时源点覆盖包含安全终点，积分区间仍是左闭右开；这保证最后一个 15 分钟段有真实插值端点，不能在终点引入新任务。

前轮审计：三个“到达加最大 Drain”包络互不重叠；三个 raw task identity 集合交集均为零，也与已见 7 月 1–3 日队列交集为零。原始 identity 字段为 job_name/start_time/end_time/start_dt；窗口内重复 identity 数 9/263/146 原样保留，stable task ID 仍唯一，不做去重。时间隔离与零 identity 交集不能证明统计独立，重复 trace 的结构、重复训练 episode 和 UTC 时间特征都可能支持记忆。

9 月 8 日 Test 候选曾因一个缺桶被前轮覆盖审计改为 9 月 9 日；本轮不按策略胜负重选。全源 CI 有两个 estimated 标注点，但不在上述包络或 4 月校准内。天气声明固定 UTC−7，三个夏季包络的 IANA 实际偏移也为 −7；不能把这一结论自动扩展到冬季窗口。

## 阈值、任务、容量与物理模型

固定阈值 **161.075 gCO₂e/kWh**，精确 float hex `0x1.4226666666666p+7`。来源为 `[2023-04-01 00:00, 2023-05-01 00:00)` UTC 的 720 个完整、有限、非 estimated 小时 direct CI 中位数。校准早于 Train；禁止使用旧六月阈值 137.65 在五月训练，也不在 Validation/Test 调 CI 分位数。

十二列顺序为 job_name、start_time、end_time、start_dt、duration_min、cpu_usage、gpu_wrk_util、avg_mem、avg_gpu_wrk_mem、bandwidth_gb、weekday_name、weekday_num。原资源倍率 5：CPU=`5*cpu_usage/100` 核、GPU=`5*gpu_wrk_util/100` 卡、MEM=`5*avg_mem`（保留上游内存单位 convention）。duration 直接使用原 duration_min；合成 deadline=`arrival+1.5*duration` 分钟，原 trace 没有本研究的真实 deadline 字段。declared duration 在到达时可知，这是既有模拟器假设。

容量固定 CPU 50000、GPU 1000、MEM 80000；单 DC1/US-CAL-CISO；HVAC 22°C。`power.py` 资源规模公式给出 1.0 MW，`make_dc_env(month=1)` 与 reset(seed=123) 保留。每次 episode reset 创建新物理实例、拍间保留实例；每拍实际资源利用率、对应天气在物理 step 前注入。month/seed 是既有构造与 reset 设定，不作为独立实验复本或调参手段；不改 IT/HVAC 公式。

15 分钟非抢占执行：当拍已到达任务登记，完成释放，全部 pending EDF 排序，逐候选容量检查后按模式判定；已启动任务立即改变后续候选可用资源，放不下可回填其他候选。finish_time 与资源释放边界分开：精确声明完成时间 `<=deadline` 算按时，资源在其后首个网格边界释放。过期任务继续执行，不能删除以减少违约。post-decision 负载保持在 `[t,t+15min)`；IT/HVAC kW×0.25 h 得 kWh，总碳=`(IT+HVAC)kWh*当前CI/1000` kg。顶层跨区传输能耗不参与本地总电能。

CI 为不晚于当前时刻的最近真实小时观测，不插值未来 CI 给策略。天气按真实对应 UTC 线性插值，只进入物理积分；wet-bulb 保留 `0.9*temperature` 代理。小时观测在其时间戳即在线可用是模拟假设，尚未研究真实上报延迟。没有环境循环、外推或复制最后一项。

## 压力与随机性

预检查只用 Train，档位 1.0×/1.5×/2.0×，stress seed 510001。任务量分别 5542/8313/11084。固定选择规则：在已有档位中选最高的、全部策略均排空、至少一策略出现 capacity waiting，且三种启发式的 `(deadline_violation_rate_observed, mean_wait_minutes_started)` 组合存在差异的档位。逐拍队列另行报告，不代替该选择判据。若全部没有训练信号，则停止并报告，不访问 Test 寻找有利压力。

扩增严格使用 E1-2B `stress_workload`：按 canonical(seed,task_id) SHA256 排序，选前 `floor(N*(factor−1))` 个父任务复制一次；同 seed 高倍率包含低倍率。新 ID 为唯一 suffix，duration/demand/arrival 不动。原任务原样保留。额外任务同刻到达是合成突发压力，不能称为真实增量 trace，也不能解释为独立随机 workload 采样。

在任何策略或 PPO 结果产生前，最终控制变量方案固定 Train stress seed **510001**：所有 episode 与三位 RL seed 使用同一任务集合；510002/510003 只预留，不按 reset 混换。Validation 正式候选对照固定 **520001** 与 Train 选择的同档位，520002/520003 只预留，不冒充已运行结果；Test **530001/530002/530003** 只预留。RL smoke seed **55100**；正式 candidate seeds **55101/55102/55103**，与压力 seed 分离。2×复制全体，因此 stress seed 只改变复制 ID 与其稳定排序；不能宣称不同 stress seed 提供独立 task arrival profile。本阶段三 candidate 的变化来源主要是 RL 初始化与采样，重复单一 Train 窗口的泛化限制须保留。

## Observation、Action、Reward

直接复用 E1-5A 40 维 causal Observation 与 Discrete(3)：0=EDF immediate，1=EDF+Carbon，2=EDF+RiskOnly。每拍只选择一个全局模式；非最终任务级风险预算。RiskOnly 历史窗口 120 分钟，rho 最大多资源近期工作量压力阈值 **0.5/1.0**，buffer **0/1/2** 拍，cold buffer=2；不运行 Shadow Planner。

归一化沿用工程尺度 CI1000 g/kWh、count1000、slack/work120分钟、elapsed24小时，`x/(scale+abs(x))`；利用率 demand/capacity，周期时间 sin/cos。空队列零值、首拍缺 history 明确 cold 标志。无 Validation/Test 统计拟合，也不自动套 VecNormalize 改奖励或观察尺度。

PPO-v0 默认 Reward 完全保留：alpha=1、beta=1、gamma=10；carbon_scale=1000 kg，wait_scale=15000 task·min，deadline_scale=1000 tasks；terminal_scale=1000 tasks、terminal_weight=10。拍碳来自真实 IT+HVAC；等待为 post-decision pending×15 分钟；首次当前确定违约单次计罚。截断对已到达 unfinished 数及剩余多资源工作量计负债，不给 unseen future task 罚分。同一任务晚完成不得重复 deadline 计罚。每项 raw/normalized/weighted 数值独立存储；不因首次训练弱于 RiskOnly 改权重。

## Drain、公平比较与模型选择

训练 episode 停在自己的首次完整 Drain；安全截断只在明确真实环境终点，或另行记录的训练长度。PPO learn 的 rollout batch 截止不是任务删失，不可把未完成一个 episode 当排空结果。若 learn 总步数结束时留下部分 episode，应明确完整 episode 和 partial rollout 的分母。

正式 Validation 同一 frozen cohort 的 EDF/EDF+Carbon/RiskOnly/所有保留 PPO 候选使用共同终点：最慢排空时刻。较早排空的执行器保持原物理实例，按真实后续 CI/天气零任务利用率积分。若某方法到安全终点未完成，则公共终点为 safety_end，报告删失和未完成；不能宣称不等工作量碳优势。任务账本、deadline、资源小时、IT/HVAC/总电能与碳同时审计，不能只比较 own Drain reward。

候选准入在看到结果前固定为：所有到达任务完成、无删失、deadline miss rate≤**5%**。准入后按公共窗口 carbon 最小、再 miss rate、再 seed、再 checkpoint SHA256 排序，与预注册 PPO config 完全一致；没有准入候选就报告 none，保留全部结果。5% 是本阶段预先声明的操作性 SLA 门槛，不是从数据推导的天然可行边界。训练回报不替代论文指标。

## 实施 guard 与审计要点

1. Train/evaluate 入口采用 split role 白名单及固定 UTC 边界；test 字样或 Test 边界在任何数据加载、策略运行前拒绝。不能仅信调用者任意传入的 split 标签。
2. 训练 checkpoint 附 protocol/config/snapshot hash 与 action/schema/Reward 定义；加载核对 40 维、三动作及参数，禁止混入旧阈值。
3. 每个 task arrival 检查所属窗口；FrozenWorkload/environment bounds 与源 SHA/容量/物理 config 核对；train 与 validation snapshot hash 必须不同。
4. 只向 actor 输入现有观察，未来到达/未来 CI/weather、完整 frozen任务集、其他策略 outcome 保留在离线模拟器/审计层。条件动作分析可以事后读当前拍原始 observed fields，不能倒灌 actor。
5. 原始源文件、旧 archive/provenance 与执行核保持不变；新增代码导致当前源码全清单变化，应建立 E1-5B 新 freeze，不弱化旧回放哈希检查。

以上是协议与覆盖审计，不是训练或选型结果。重复单一 Train 窗口、共同七周 trace、非抢占与已知 duration、小时 CI 即时可得、wet-bulb 代理和未含迁移/传输/故障成本仍是模型边界。最终 Test 未运行，本轮不得称独立 Test 泛化验证。
