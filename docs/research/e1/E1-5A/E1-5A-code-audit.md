# E1-5A 执行器代码审计与公共逐拍内核提取边界

审计时间：2026-10-12。审计对象是原科研工作区 `D:/A-myprofile/SustainCluster/upstream` 中的现有执行器、策略、状态、冻结输入与 E1-3/E1-4A provenance。此文先于公共逐拍内核修改产出；本次审计只读源码、历史档案和 Git 元数据，只新增本 E1-5A 文档。没有运行实验、RL、历史 replay 或测试，没有修改源码、历史 provenance、`.git` 或发布副本。

结论：现有 `LocalExecutor.run()` 已包含唯一的真实执行、三资源分配、非抢占运行、逐拍功耗和能碳核算路径。E1-5A 应在该类中最小提取“准备当前边界 / 推进一拍 / 完成与结果”，使完整 `run()` 和 RL 适配器调用同一执行路径。动作只选择本拍的既有策略，不能另写一个仿真器。RiskOnly 的到达历史必须在每个当前边界持续观测，不能只在动作 2 被选中时更新。

## 1. 来源身份与本次读取的字节基线

原科研历史 HEAD 是 `3f6ea95cb835b89ba50b0ef76d66d14b8037643e`，历史 branch 是 `research/e1-carbon-deadline`；未提交科研修改是来源的一部分，不能清理、stash、reset 或把发布提交充当原模型 HEAD。最近已记录的发布提交是 `63173df23fbe7a01e09291914973552ee200cb1a`，父提交为 `ff09c06ebac789716416abc4771f238b629cf39d`。这些身份来自历史 provenance 和 `E1-4A publication-result.json` (private local archival evidence)，本审计没有以网络查询重写这些记录。

| 本次读取的关键源 | SHA-256（原始文件字节） |
| --- | --- |
| [engine.py](../../../../experiments/e1/engine.py) | `966d456b504d7e7f5b2e0a743bdab216d9c9ab35e1160fd70318eb9beb3d5f9f` |
| [cars.py](../../../../experiments/e1/cars.py) | `8c38fa046c5ee0d28afbb978b535e49fe3f1ee17b1a01afbb917024c08fcc96a` |
| [cars_state.py](../../../../experiments/e1/cars_state.py) | `032ada5d6dcb79eb3fcaf90a0c91d98a80e03686ad1cc64688e60adea9fa26d5` |
| [workload.py](../../../../experiments/e1/workload.py) | `14a3576c1f569632463516b8589a58e1955bdba94150088471affd6b6aa4f961` |
| [power.py](../../../../experiments/e1/power.py) | `11d200fe2f6e5b6f5ac4443c5b2ea6cd50af9297e27fbbb1986f8f250fddcf0d` |
| [cars_experiment.py](../../../../experiments/e1/cars_experiment.py) | `6df44f6ce2363be449af00a571463caebbba3a6d7c7598a3ca8a519eecd09667` |

上述关键字节与 E1-3/E1-4A 归档 source 清单中的对应项一致；这不等于全部当前 source 清单与每个历史清单一致。

| 历史 provenance | schema / 清单 | 解释 |
| --- | --- | --- |
| `E1-3 主档` (private local archival evidence)及 `load_1p50` 子档 | `e1-3-cars-v1`；142 个 Python 源；清单 SHA `209e89b4aca8edeec764366108e91e5de978833ab741812ee84b981f83ff882b` | 主档科学文件 6 项，子档 136 项；子档记录原科研 HEAD。 |
| `E1-4A 主档` (private local archival evidence)及 `load_1p50` 子档 | `e1-4a-feasibility-ablation-v1`；150 个 Python 源；清单 SHA `495b572c6f5fdbb3cdbd7987a0626d4309e3926fd6f2018410ac76cc272402d6` | 主档科学文件 7 项，子档 112 项；原 142 项保留，新增 8 项。 |

两阶段均记录完整 Python `3.10.22 (main, Oct  1 2026, 20:14:04) [GCC 14.3.0]` 与 115 项依赖版本。E1-4A 的 `model_source_version` 另记 E1-3 原科研 HEAD、E1-3 发布提交、旧 source manifest 与旧 provenance 身份；这些字段有不同角色。

## 2. 执行状态与不可改变的所有权

[LocalExecutor.__init__](../../../../experiments/e1/engine.py#L57) 验证 workload/environment 起点相同；截断模式要求终点相同，drain 模式要求环境覆盖 workload 到达终点；所有任务必须来自 DC1。`self.specs` 按 `(arrival_time, source_row, source_task_index)` 排列。`ledger`、`events`、`steps`、`policy_events`、`policy_steps`、`pool`、`power_model` 和策略实例均属于同一个 executor 生命周期。当前 `has_run` 禁止第二次 `run()`。

[TaskSpec](../../../../experiments/e1/workload.py#L19) 是不可变任务规格，保留 12 个原始字段、稳定任务 ID、精确分钟 duration、固定需求与 synthetic deadline。`FrozenWorkload` 只允许 `start <= arrival_time < end`，并拒绝重复 ID。`TaskSpec.make_task()` 保存/恢复全局 `random` 状态后构造真实 `Task`，把随机后缀替换为稳定 ID；逐拍运行不能每拍重建同一个 Task。

[LocalResourcePool](../../../../experiments/e1/engine.py#L13) 重用 `SustainDC.try_to_schedule_task` 与 `release_resources`。它的 `can_schedule()` 逐维计算 `math.fsum(现有 running demand + candidate demand) <= capacity`，不引入超配 epsilon。`inventory()` 校验资源守恒，并从实际 running 集合重新计算 available，避免多次减加造成累计误差。可供策略读取的是不可变 `TaskSpec` 和 `ResourceSnapshot`；真实 `Task`、可写 pool、未来 workload、未来 CI/weather 不应进入策略或 RL observation。

[check_ownership()](../../../../experiments/e1/engine.py#L84) 强制每个已到达 ID 恰在 pending、running、completed 三类之一，且 queue 状态与 ledger 一致。完成任务保留在 ledger 中；超容量、已过 deadline 或当前无法启动的任务仍保留，执行器不会自动丢弃或取消。

## 3. 每个 15 分钟边界的完整顺序

`STEP` 是 [contracts.py:9](../../../../experiments/e1/contracts.py#L9) 的 15 分钟。冻结环境逐项覆盖 ``environment.start, environment.end)` 的全部左边界；时刻与窗口均显式 UTC、按全局 15 分钟网格对齐。下面的“当前拍”指 `[t, t+STEP)`。

| 顺序 | 现有位置 / 关键变量 | 实际语义与边界 |
| --- | --- | --- |
| 0 初始化 | `LocalExecutor.run():95–104`，`arrivals`、`has_run` | 将已有 `self.specs` 按到达边界索引；不再读原 trace。`stop_at_drain=True` 只接受明确 drain 模式。 |
| 1 arrival | `run():105–115`，`sample`、`time`、`incoming`、`ledger` | 仅为 `arrivals.get(time)` 构造 Task，ledger 初态 pending，计数初态 0；先写 `arrived` 事件。当前 incoming 尚未放入 pool pending。 |
| 2 release | `run():116` → `release():76–82` | 释放所有 `finish_time <= time` 的真实 running；ledger 改 completed，写 `completion_observed_at=time` 与 `completed_resources_released`。先到达登记、后完成释放是事件顺序；所有新旧 pending 启动之前已经释放容量。 |
| 3 drain 判定 | `run():117–123`，`drained_at`、`observed_end` | 只有 `time >= workload.end` 且 pending/running/incoming 全空才记录排空。`stop_at_drain` 时在该边界退出：此处没有本拍排序、启动或功耗区间。到达窗内短暂空队列不会提前结束。 |
| 4 enqueue | `run():124–126`，`pool.pending_tasks` | incoming 依原 specs 顺序追加至旧 backlog，写 `enqueued`。 |
| 5 sort | `run():127–134`，`waiting`、`ordered`、`originals` | 每个策略取得全部尚未启动任务，而非只看新到达。`policy.order()` 必须保留每个 immutable spec 恰一次且对象身份不变；随后重建 pending deque。 |
| 6 当前观察 / 历史 | `run():135–145`，`DecisionContext`、`ResourceSnapshot` | context 只含当前 time/CI。可选 `on_step_start()` 得到容量、释放后的 available、ordered pending、当前 running 精确 finish/deadline、当前已到达 incoming；没有未来输入。静态 CARS 每拍在这里更新历史。 |
| 7 capacity gate | `run():151–160`，`rank`、`can_fit`、`available` | 全部 ordered 候选只遍历一遍；popleft 当前候选并检查当时三资源。不能 fit 时直接调用真实 failed-schedule 回填，`Task.wait_intervals` 与 ledger `capacity_wait_intervals` 各加 1；不调用 `policy.decide()`。后面的任务仍可启动，即 one-pass backfill。 |
| 8 decide | `run():162–174`，`PolicyDecision`、`reason` | 只有可 fit 候选调用 `on_candidate_resources()` 与 `decide()`。返回值必须是 `PolicyDecision` 且 defer 为 bool。defer 回 pending，ledger `policy_defer_intervals` 加 1，通知 blocked-until `time+STEP`；不增加 Task 的 capacity wait 计数。 |
| 9 start | `run():175–186` → upstream `try_to_schedule_task()` | 不 defer 时由同一真实分配器扣容量，赋 `start_time=time`、`finish_time=time+duration` 并入 running。若 gate 与分配不一致立即失败。ledger 改 running，目的 DC=1；通知 `on_task_started()`，下一候选随即看到本拍已有启动后的状态。 |
| 10 候选记录 / 校验 | `run():187–195`，`policy_events`、`detail` | 每个候选写 capacity_waited / policy_deferred / started 及真实 reason、rank、CI、threshold、latest_start、gate 与 available-before。每候选调用 inventory；一轮结束调用 ownership。事件记录穿插在 arrival/release/enqueue/start 中，不能重排成统一拍末日志。 |
| 11 power | `run():195–197` → `UpstreamPowerModel.measure():36–47`，`used`、`fractions` | 用本拍真实 allocation 后的 CPU/GPU/MEM fractions。先 `update_workloads(cpu,mem,gpu)`，再 supplied 当前 weather，最后 step HVAC setpoint。模型初始化只 reset 一次（seed 123）；不能每拍重建、reset，或用前一拍 info。模型报错必须传播。 |
| 12 energy / carbon | `run():198–199` → `interval_record():8–23` | 左边界 allocation 在整拍保持。hours=0.25；IT/HVAC/total kWh = 对应 kW×0.25；kgCO2e = total kWh×当前 CI(g/kWh)/1000。资源负载、pending/running 数均是本轮分配后的状态。传输能量为 0。 |
| 13 interval / policy log | `run():198–202`，`steps`、`policy_steps` | 追加一个且仅一个 interval record；记录 pending_end_ids、running_ids 与这一拍的候选/排序/启动/等待集合。推进到下一个环境 sample，不在拍中重新排序或抢占。 |
| 14 finalize | `run():203–210`，`observed_end` | 循环退出后只在观察终点进行 completion flush、ownership 和 ledger 完整性检查；若在 safety end 排空，记录 drained_at。T 没有 arrival、start、power 或额外 drain 拍。随后调用 result。 |
| 15 result | `result():230–263` | 由真实 ledger 和所有已核算拍组装 ExecutionResult；输出完成、pending/running、deadline outcome、等待归因、能碳汇总。result 本身不再推进或执行物理模型。 |

`SustainDC.try_to_schedule_task()` 的旧 docstring 提到 MAX_WAIT_INTERVALS/drop，但 [实际分支` (private local archival evidence) 总是增加等待后回 pending，并不丢任务；不能根据该 docstring 在新适配器加入取消行为。

## 4. 精确完成、资源可观察完成与非抢占

`upstream release_resources()` (private local archival evidence) 用 `task.finish_time <= current_time` 判断释放，并以精确 `finish_time <= sla_deadline` 设置 sla_met。任务于 05:00 启动、duration=5min 时，物理任务 finish 是 05:05，SLA 用 05:05 判断；资源与 completed 状态在 05:15 边界才可观察释放。`RunningTask.release_time` 也明确是 [finish_time.ceil('15min')](../../../../experiments/e1/cars_state.py#L27)。不能把 finish 改成 ceil 时刻，也不能在 05:05 插入新的调度边界。

已启动任务只在真正完成时释放；策略动作只能作用于本拍尚未启动任务。任何动作切换都不能清空 running、调整其 demand/duration、移动 start/finish、暂停或恢复任务。整拍 allocation 的 left-boundary hold 意味着 duration=5min 的任务仍贡献整个 15min 的资源占用和该拍功耗；这正是既有核算约定。

`result()` 对完成任务输出 met/missed；未完成任务在 `deadline <= observed_end` 时为 missed_unfinished，其余 right_censored，不能冒充 completed 或 met。`completion_observed_at` 与 `finish_time` 必须分开保留。`Task.wait_intervals` 只计 capacity shortage；完整 pending 时长另由 `(start_time 或 observed_end)-arrival_time` 计算。policy defer 与 capacity wait 要分别统计，capacity-first 归因不因选择了碳策略而改变。

## 5. 固定三动作的策略对应和 Risk 历史要求

| E1-5A 动作 | 直接复用的既有策略 | 本拍 eligibility |
| --- | --- | --- |
| 0 EDF | `EDFBackfill` / `edf_backfill` | 按 `(sla_deadline, arrival_time, task_id)` 稳定 EDF；可 fit 则立即启动。 |
| 1 EDFCarbon | `CarsPolicy('edf_carbon_threshold', ...)` | 同一 EDF；CI <= threshold 执行；高 CI 且 `now+STEP > latest_start` 执行；否则 defer。 |
| 2 RiskOnly | `CarsPolicy('cars_risk_only', ...)` | 同一 EDF 与碳边界；高 CI 时另以 `now+STEP*(1+buffer_steps) > latest_start` 强制执行。没有 Planner。 |

动作是在每个调度拍开始选择一次；一拍内不能对不同候选切换 meta-action。所有三动作都必须经过同一 capacity gate 与 one-pass backfill。动作 1 不应替换成 `CarbonThresholdFIFO`，否则顺序改变为 FIFO；动作 2 不应使用 Full/KnownOnly、增加 shadow、调 threshold 或学习 risk bins。

既有 EDF 排序见 [policies.py:52](../../../../experiments/e1/policies.py#L52)，既有 EDFCarbon/RiskOnly 判断见 [cars.py:53](../../../../experiments/e1/cars.py#L53)。精确 `latest_start = sla_deadline-duration`；等号仍允许 defer，只有严格 `>` 才触发无网格 slack/risk guard。CI 恰等于 threshold 属于低碳执行。正式冻结输入继承 threshold 浮点字节值 `137.64999999999998`；名义书写 `137.65` 不能替换封存值。

[CarsPolicy.on_step_start()](../../../../experiments/e1/cars.py#L32) 目前同时重建本拍 readonly snapshot/pending/running/blocked/available 和执行 `history.observe(snapshot.now, snapshot.arrivals)`、`history.pressure(snapshot.now)`。这对整段固定策略是正确的。若 meta-policy 仅把 hook 转发给当拍被选中的策略，RiskOnly 将漏掉动作 0/1 拍的 arrivals；首次再选动作 2 时低估 rho，或反复构造实例使其回到 cold start。

最小适配应保持一个持久 RiskOnly 实例，并在每个当前边界（包括空 pending、所有候选 capacity-wait、动作 0/1、drain 内无新 arrivals）持续观测。最直接复用现有代码的方式是：准备边界时构造当前已观察、完整 EDF 排序的 `ResourceSnapshot`，无条件调用该持久实例的 `on_step_start()`；然后构建包含本拍压力的 RL observation；再锁定动作并通过统一分配器推进。三个动作的 EDF 排序相同，因此不会用所选动作影响 RiskOnly 已观察的当前到达集。真正候选 resource/start/defer 通知只服务该拍被选中的 eligibility 路径；下一边界再以真实 pool 状态重建 snapshot。旧固定 policy 的完整 run 仍须保持原 hook 时序与调用次数。

若为复用边界实现新增一个 observation hook，其职责必须局限于当前不可变 observed state；不能在 RL 外重新实现 risk 公式或构造第二套到达历史。准备边界到选动作之间不得先调用 `decide()`、分配资源或测功耗。重复读取同一准备好的 observation 不应再次到达登记、释放、滑动 history 或执行 hooks。

[ArrivalHistory](../../../../experiments/e1/cars_state.py#L64) 必須原样保留：窗口 `(now-120min, now]`；每个稳定 ID 计一次；每任务贡献 `demand * ceil(duration/15)*15` resource-minutes；分母 `capacity * min(120min, now-start)`。起点分母 0，rho 为 null、cold buffer=2；以后 rho=max 三资源，rho<0.5→0、0.5<=rho<1→1、rho>=1→2。窗口左端等号过期，drain 继续滑动，没有伪造 prehistory、future arrivals 或把 buffer 当容量预留。`seen` 集合、`records` deque、`last`、`start`、capacity 与固定映射都要跨拍保持。

## 6. 最小公共内核提取契约

以下为审计建议的职责边界，尚未实现。方法名可采用 Python 风格 `prepare_current_boundary()`、`advance_one()`、`finalize()`，但 run 与 RL 必须调用同一组方法；不能复制 `run()` 循环到 RL 文件。

| 方法 / 状态 | 最小职责 | 不能发生的副作用 |
| --- | --- | --- |
| 一次性启动状态 | 将 run 中 `arrivals` 索引、环境 sample cursor、stop_at_drain 模式、prepared-boundary cache、finished 标记纳入同一 executor；保留现有 pool/ledger/power/策略。 | 每拍重建 executor、Task、power model 或 history；一边 run 一边重新开始 step 生命周期。 |
| `prepare_current_boundary()` | 取当前 sample；按现有事件顺序 arrival→release→drain 判定→enqueue；缓存当前只读 sample/arrivals/pending/running/capacity。到排空或环境 exhausted 时返回 terminal 边界。给持续观察 hook 机会更新本拍历史后才给 RL observation。 | 政策 eligibility 决策、start、power、carbon；暴露 sample cursor 之后的输入；重复调用导致重复事件或 history 更新。 |
| `advance_one()` | 消费一个已准备非 terminal 边界和本拍已锁定策略；复用原排序验证、optional policy hooks、capacity-first 候选循环、inventory/ownership、同一 power measure、interval_record、政策日志；恰好提交一拍并推进 cursor。 | 用 action 修改已经 running 的任务；重复提交同一 prepared 边界；提前调用下一拍 decide；生成额外末端功耗。 |
| `finalize()` | 到环境终点或 drain stop 后完成现有 T-only flush、ownership、ledger 完整性与 drained_at 判定；只执行一次，重复 finalize 返回已有完成状态。 | 添加 T 的 arrivals/starts/interval；触发新的 policy eligibility；覆盖历史源身份。 |
| `result()` | 保留现有完成后序列化与汇总路径；finalize 后可重复读取；`extend_idle()` 后可重算扩展结果。 | 作为 step 的中间状态 getter：未到达 future specs 尚未入 ledger，当前实现会 KeyError，零拍 policy 汇总也没有通用 partial-result 契约。 |
| `run()` | 作为兼容批处理 wrapper：一次启动→循环 prepare/advance→finalize→result。旧调用方继续用 `run(stop_at_drain=...)`。 | 保留另一份独立循环，导致 batch/RL 同名函数却有两套物理实现。 |
| RL adapter | reset 创建一个 fresh executor/持久 meta-policy；observation 使用准备后的当前 observed state；step 锁定动作、advance 一拍，再准备下一边界作为下一 observation；终态 finalize。 | 重新实现 pool、SLA、release、功耗、碳积分或 replay；用 `result()` 读取尚未到达任务的终态。 |

拆分的关键是 prepared 边界只消费一次。sample cursor、cached incoming/specs 和边界 terminal 状态必须属于 executor。`advance_one()` 的返回值可包含刚提交 interval、事件增量与已完成边界信息；未来 observation 由下一次 prepare 取得。若 wrapper 在 step 返回前准备下一边界，就会在那里登记下一拍 arrivals、释放上一拍 finish；日志与奖励归属必须明确：动作 t 的 carbon 是 `[t,t+15)`，completed/SLA 的可观察增量由下一边界 release 提供，不能用 ceil completion 改写精确 SLA 判断。

旧 `run()` 的一次性 guard 与新 step 生命周期必须统一。`has_run` 当前表示“已启动”，不能在每拍置 False 以绕过 guard。不可重入、不可向后推进、不可重复最后一拍；错误应当传播，不能恢复旧 power reading 或静默继续。

根任务实施草案拟使用一个生成器：保留现有执行顺序，在 arrival/release/enqueue 后、order 前 yield 当前 sample；`start()` / `advance()` / `run()` 共用该生成器，advance 后准备下一边界。该结构可以满足上述职责，前提是 yield 当前边界仅一次、由独立当前观察 hook 在动作选择前更新持久 history，并保留旧固定 policy 的 hook 时序。草案的训练 `until` 应是 exclusive cutoff：末端只 release/flush，不登记该截止时刻的新 arrival、不产生最后额外功耗拍。现有 result 尚无 partial 契约；若增加它，应明确只序列化已经进入 ledger 的 observed prefix，记录当前 observation_end，并把 observed-prefix 完整性断言与原 full-run 的全 specs 完整性断言分开。不能把尚未到达任务算 arrived、把 partial finish 称 cohort drain，或让新 until 默认改变旧 run 行为。

`extend_idle()` 保持 [engine.py:212](../../../../experiments/e1/engine.py#L212) 的原契约：仅完成 drain 后、同一 physical model、覆盖内且不早于当前 observed_end 的对齐终点；循环 `[observed_end, endpoint)` 的真实 sample，以 zero utilization 测功耗并写空 policy_steps，随后 observed_end 更新、result 重算。它不是 RL scheduling step，不应要求新的 action 或凭空核算 completion。

## 7. 旧严格回放与新增 source 身份的边界

[cars_experiment._verify_level()](../../../../experiments/e1/cars_experiment.py#L377) 在 384–386 行要求归档 `prov['sources'] == experiment_sources()`、完整 source-manifest SHA、`sys.version` 与完整依赖映射一致；还验证 archived source 字节、输入、全部科学输出与 runtime。`replay_levels()` 在 542–546 行对主档做同类严格检查，再核查子档。任何最小 engine 提取或新增 Python 文件都会改变当前 source 身份，即使行为逐字节等价，也不能在当前新代码环境宣称通过旧归档的 strict-source replay。

[feasibility_experiment._verify_sources()](../../../../experiments/e1/feasibility_experiment.py#L110) 更明确地区分：E1-4A strict replay 要求完整 source 清单相同；E1-3→E1-4A 的 cross-version import 只豁免固定 [8 个 NEW_SOURCE_FILES](../../../../experiments/e1/feasibility_experiment.py#L31)，仍要求所有旧项在当前 checkout 中的字节一致。改 engine 会在 117–118 行被拒绝；另增不在豁免集合的 E1-5A Python 文件也会在 115–116 行被拒绝。该拒绝是来源身份边界，不是应当放宽的测试问题。

因此旧 E1-3/E1-4A strict replay 必须继续使用各自历史 `source/` 字节、记录的解释器/完整依赖与相应数据依赖，在单独复现环境中执行。source 快照并非包含原 CSV 的完整 checkout；历史 approved import 还需要原数据哈希，E1-4A replay 则使用其封存输入。旧 `source/`、provenance、验证器规则、NEW_SOURCE_FILES 和完成日志不应为新内核而改写。

E1-5A 的新源码必须使用自己的 source manifest/provenance。新内核与旧内核的语义回归可以对同一冻结输入独立运行后比较科学输出；这不能冒称原归档在新 source 下通过 strict replay，也不能重标旧 source SHA。发布副本此前存在换行规范化与一个 source 项缺失，详见 `E1-4A-reproducibility.md:114` (private local archival evidence)；公开 synthetic tests 与私有历史输入端到端 replay 的证据必须分开。

## 8. 现有 tests/e1 的只读兼容检查

本审计仅检查测试代码与 fixture 来源，没有执行或声称测试通过。当前核心行为测试已明确约束 release-before-start、三资源守恒、精确 finish 与 ceil observable release、one-pass backfill、末端只 flush、drain/safety cap、idle extension、单次使用、同拍 CARS state 通知、未来 trace/CI 不影响当前决策。来源见 [test_execution.py](../../../../tests/e1/test_execution.py)、[test_drain.py](../../../../tests/e1/test_drain.py)、[test_cars_policy.py](../../../../tests/e1/test_cars_policy.py) 和 [test_cars_state.py](../../../../tests/e1/test_cars_state.py)。

已检查的 CARS/E1-4A strict replay 测试主要在 `tmp_path` 中新建归档，再在同一当前 source 下 replay；它们能验证新源码的 replay 完整性，却不能自动证明当前代码可 consume 历史 E1-3/E1-4A 原始 source 清单。旧全套 green 也不能取代“新内核 batch 与逐拍动作固定回归 + 历史环境 strict replay”这两种不同证据。

| 测试 / fixture 位置 | 只读检查所见 | engine 提取后的含义 |
| --- | --- | --- |
| [test_cars_experiment.py:21](../../../../tests/e1/test_cars_experiment.py#L21) `setup_reference()` / `run_case():46` | 当前 temporal/contention 先生成 reference，再于 tmp_path 新建当前 CARS archive。`:85 test_sealed_level_replay...`、`:105 test_level_replay_mismatch...`、`:138 test_campaign_imports...` 均消费该 fixture。 | 当前 source 同源 replay 仍应工作；这些不是历史 E1-3 strict replay。 |
| [test_cars_experiment.py:224](../../../../tests/e1/test_cars_experiment.py#L224) `test_approved_archive_head...` | 真正读取历史 E1-2B/results-001/load_1p50；经 `_load_reference()`→`contention.verify_archive(current_sources=False)` 验证历史自身完整性。 | 不要求历史 source 等于当前；真实历史输入本身的 source 完整性仍要保持。 |
| [test_feasibility_experiment.py:34](../../../../tests/e1/test_feasibility_experiment.py#L34) `reference()` / `run_case():50` | 名为 old / e1-3 的 archive 也由当前 CARS 在 tmp_path 中新建；`:79 test_level_strict_replay...`、`:100 test_cross_version_import...`、`:125 test_campaign_offline_replay...` 用这一链条。 | 包含提取后的 current source，无法揭示真实历史 E1-3→当前 E1-4A import 的 source 拒绝。 |
| [test_policy_runs.py:43](../../../../tests/e1/test_policy_runs.py#L43) 与 `:67` | 均先新建当前 archive。cross-version 测试人为改归档 engine 字节与 manifest，要求 strict replay 拒绝、显式 frozen import 接受。 | 旧显式 frozen-import 行为要保持，不能由此推出 CARS/E1-4A import 可忽略旧源。 |
| [test_policy_runs.py:117](../../../../tests/e1/test_policy_runs.py#L117) `test_historical_e1_1a_fifo_legacy_fields_and_numbers_unchanged()` | 读取私有历史 artifacts/smoke-007，以当前 LocalExecutor 运行并比较历史数值，不做 source 校验。 | 这是实际数值回归，若失败需要检查执行行为；不能与 source 身份拒绝混为一谈。 |
| [test_diagnose_feasibility.py:100](../../../../tests/e1/test_diagnose_feasibility.py#L100)、`:147`；[test_stress.py:8](../../../../tests/e1/test_stress.py#L8) | 前者手写 JSON/CSV fixture，后者内存任务。 | 没有真实历史 E1-3/E1-4A 当前-source strict replay 依赖。 |

额外源码收集风险：[run.py:33](../../../../experiments/e1/run.py#L33) `source_hashes()` 收集全部 tracked Python 与当前 E1 Python；[temporal.py:26](../../../../experiments/e1/temporal.py#L26) `experiment_sources()` 再加全部 `tests/e1/*.py`。所以新增/修改测试文件同样改变 strict source 身份。E1-4A level 在 `feasibility_experiment.py:444`、campaign 在 `:617` / `:648` 调用严格源校验；历史 import level 在 `:152`、campaign 在 `:481`、缓存 identity 离线执行在 `:329` 仍执行 cross-version 限制。

给根任务的兼容建议：把 source mismatch 明确记录为预期身份拒绝，不改验证器使它变绿；当前 tmp_path tests 验证新源码可自封存自回放；历史数值 fixture 验证执行行为；真实旧 archive strict replay 放在相应历史 source 环境。不要把前两种证据写成真实历史 import/replay 兼容。测试建议另文档，当前没有测试执行证据。

## 9. 审计决定

允许的后续最小实现是同一 `LocalExecutor` 中提取一次准备、一次推进与一次 finalize，run 作为 wrapper，RL 只切换固定三动作并读取已观察当前状态。真实 Task/资源所有权、当前负载后 supplied weather 的持久功耗模型、左边界整拍核算、精确 finish/SLA 与可观察完成、drain/common idle、原三动作数学边界必须保持。RiskOnly history 无条件逐拍持续观测是必要条件。旧 archive strict source 校验应继续拒绝不匹配的新 source；通过历史环境保留旧 replay 能力，而非修改历史或放宽验证器。
