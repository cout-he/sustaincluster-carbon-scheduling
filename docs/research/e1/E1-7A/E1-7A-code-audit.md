# E1-7A 执行语义审计（实施前）

## Material Passport

诊断范围为用户同SHA的两份完整E1-7A要求（SHA256 `3da8d64f7f4b7f8184da7e679ab0d0ff124d738e8ac3eaffb85786907504a805`）。原科研HEAD/main为`3f6ea95cb835b89ba50b0ef76d66d14b8037643e`、研究分支及历史dirty状态保留；独立发布研究起点`80f5c84c2ae45bf86de397a4421d8cdedba7e03f`、main `749dc6c1868065c12a71bac71b3ae08a25c45f56`、唯一personal remote。本轮只读Train归档，Test不读、不生成、不运行。

## 当前公共调用链

| 阶段 | 代码 | 审计结论 |
|---|---|---|
| 输入 | workload.TaskSpec/FrozenWorkload、rl_training_utils.load_bundle | immutable原任务；loader先拒绝Test/错误角色再读payload，核对manifest/payload/环境/物理SHA。12列、需求倍率5、SLA1.5不变。 |
| 开始 | engine.LocalExecutor.start | 单次实例，创建持久pool/ledger/physics，驱动唯一_execute生成器。 |
| 拍准备 | _execute | 记录本拍arrival→释放已完成任务→入队，prepare后yield；边界T只flush，不新增arrival/启动/积分。 |
| 排序 | policy.order | 全部未启动任务必须保持同一immutable对象、唯一ID；EDF deadline/arrival/ID排序。 |
| 在线通知 | on_step_start | 只提供本拍已到达pending、current running/available、current arrivals；历史观察一次。 |
| 资源判断 | LocalResourcePool.can_schedule | 三维math.fsum全running+候选≤capacity，无放宽epsilon；资源不足先计capacity waiting，允许后续回填。 |
| 候选决定 | on_candidate_resources→decide | 只有当拍资源可容纳才调用策略；同拍先前启动已影响available。 |
| 实际动作 | try_to_schedule_task/on_task_started/on_task_deferred | 每候选一次，启动设置start与真实finish；不可抢占。hook反映实际动作。 |
| 功率/积分 | power.UpstreamPowerModel.measure→metrics.interval_record | 当拍启动后资源利用率先送原DC模型，真实weather，15min左边界负载保持；kW×.25=kWh，kWh×gCI/1000=kgCO2e；IT+HVAC，无传输代用。 |
| 完成 | release→result | SLA用真实finish，不用观察释放时刻。释放只在finish后的对应15min网格。 |
| 公共尾段 | extend_idle | 完整Drain后，同一持久物理实例，真实CI/weather、零CPU/GPU/MEM积分至公共终点。 |

## 既有机制边界

policies.CarbonThresholdFIFO、cars.CarsPolicy保留原阈值/网格slack/历史120min规则；cars_feasibility.ShadowPlanner为有界已知任务条件预测，并非严格SLA保证。rl_env仍40维/3动作，adaptive_rl_env仍83维/4动作，所有reward/参数/checkpoint不改。task_risk_state.calculate_budget只接收已知running和EDF前驱及past pressure，冻结z/B用于本轮描述，不参与新的在线决定。

temporal/contention/cars_experiment均使用同LocalExecutor及phase_summary，完整Drain与idle公平约束沿用。E1-5C必要条件仅给违约下界，不以EDF违约充当最优。任务资源小时采用ceil-grid占用，真实duration/JCT与占用时间分开。

## 实施选择

新增独立CP-SAT诊断、碳分解、EDF单次动作覆盖；不复制执行器、不改旧生产路径。A1是封闭子集；A2固定非候选完整EDF背景，包括原running及后续真实reference starts，条件EDF必须保护固定背景预约，因此不同于原无预约EDF。A2不是全系统随新决策重新调度。

CP-SAT首先尝试binary float比例/gcd精确整数映射，仅在安全int64可表示时使用；否则固定1e6保守上取整需求/下取整容量。即便精确binary数学值映射也与math.fsum舍入边界不完全同义（.1+.2+.3+.4边界），因此最优性只用于所声明的整数子问题。每个解独立原需求/真实finish/release/grid重新检查，所有任务包括迟到任务必须完成。

C采用EDF后续控制器的完整双重重放；只在指定(time,task_id,can-fit occurrence)改一次动作，当前已知状态及完整功率对象状态只读指纹比较。允许干预后自然分叉，不将未来结果反馈策略。
