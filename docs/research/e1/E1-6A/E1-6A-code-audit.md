# E1-6A 实施前代码适配审计

审计日期：2026-10-13（Asia/Shanghai）。审计角色：只读代码与方法核查。本文在新策略实现和实验之前形成，结论是接口可支持候选方法，不是新方法的测试通过或性能结论。实验与测试证据另见本阶段最终交付文档。

原科研工作区 HEAD 为 `3f6ea95cb835b89ba50b0ef76d66d14b8037643e`，E1 研究代码此前存在于 dirty 研究分支；发布副本 HEAD 为 `8e499e2882355e5ee5706b944ef9b70a68d67cc9`。未修改这些 Git 元数据、旧源码或历史实验。只读阅读 E1-5C 的方案、报告、评价协议，以及下列执行源码；没有加载 Test workload、Test 环境切片、Test 结果或训练模型。

## 1. 核心结论

现有四个事件接口足以支持任务需求匹配预算。新增独立 `EDFBackfill` 子类及注册函数即可；不需要改写 `LocalExecutor`、旧 `CarsPolicy`、RL 的 40 维 observation、`Discrete(3)` 映射或物理模型。

需要纠正一个容易产生错误实现的接口印象：`on_candidate_resources(now, available)` **只对当前能够容纳的候选调用**，没有 task ID 或 `can_fit` 参数，也没有容量失败通知。新策略应在 `on_step_start` 保存本拍全部 EDF pending，用 `decide(spec, context)` 确定当前候选；只在真实 `on_task_started` 时删除该 pending。于是容量失败和主动延迟任务均保持竞争集合身份，同拍已经启动者则立即移出。容量失败不通知策略，不意味着任务从策略集合中消失。

## 2. 一拍执行顺序与所有权

所有 batch `run()` 和交互 `start()/advance()` 都进入 `engine.py:_execute()`，不存在另一个 RL 仿真执行内核。

| 顺序 | 文件/入口 | 实际执行与新策略可见边界 |
|---|---|---|
| 1 | `engine.py:139–167` `_execute` | 当前样本时刻先登记 `arrived` 与 ledger；原 workload 有限集合已在仿真内部索引，但策略不获得此索引。 |
| 2 | `engine.py:77–83` `release` | 对 `finish_time <= now` 的 running 任务释放资源，更新 completed 与 SLA；在任何新启动前完成释放。 |
| 3 | `engine.py:168–175` | 当前新到达任务入 pending，交互内核在策略和功率计算前挂起。 |
| 4 | `engine.py:177–184` / `policies.py:55` | 策略对全部尚未启动任务排序；验证同一不可变 TaskSpec 对象、唯一 ID、数量与集合完全保留。EDF key 为 `(deadline, arrival, stable_id)`。 |
| 5 | `engine.py:186–194` | `on_step_start(ResourceSnapshot)`：全部 EDF pending、释放后的 running、当前 arrivals、capacity/available。没有未来 CI 或未来任务。 |
| 6 | `engine.py:199–209` | 单次 EDF 遍历，先 `pool.can_schedule`。失败调用上游 requeue，累计 capacity waiting；不调用 `decide`，继续检查后续候选，保留回填。 |
| 7 | `engine.py:210–225` | 能容纳则通知当前 available，调用纯 eligibility 决策。defer 重新入队且不分配资源；通知 `on_task_deferred(id, now+15min)`。 |
| 8 | `engine.py:226–234` | execute 再次进行实际分配，立即追加 running，设置原始 `finish_time=now+duration`；ledger 标记 running，通知 `on_task_started(RunningTask)`。后续候选已经受到该占用影响。 |
| 9 | `engine.py:235–254` | 记录事件、每次 inventory 资源守恒、拍末任务所有权检查；按本拍启动后的实际 CPU/GPU/MEM 占用计算 IT/HVAC，再积分 `[now,now+15min)`。 |
| 10 | `engine.py:255–261` | 观察终点只进行 completion flush，不加入任务、启动、功率拍。达到 arrival end 后可以完整 Drain。 |
| 11 | `engine.py:263–279` `extend_idle` | 排空较早策略沿原持久物理对象积分真实同时间环境的空闲尾段，到达组内公共终点。 |

`LocalResourcePool.can_schedule` 使用 running 需求的 `math.fsum` 加候选需求与容量严格比较，不使用超容量 epsilon；`inventory` 检查并重新校准 available。新策略不得直接分配、抢占、预留或删除资源/任务。

上游 `SustainDC.try_to_schedule_task` 的 docstring 提到最大等待后 drop，但实际分支始终 `pending_tasks.append(task)`。不能根据该过时注释推断 E1 会删除超时任务。`engine.py:result` 对完成任务使用实际 SLA，对未完成任务分别报告 `missed_unfinished`/`right_censored`；budget guard 不改变这些规则。

## 3. 四个 hook 的精确签名和建议实现

| 实际接口 | 参数意义 | 新策略状态更新 |
|---|---|---|
| `on_step_start(snapshot)` | 只读 `ResourceSnapshot(now, capacity, available, pending, running, arrivals)`；pending 已是本拍 EDF 顺序 | 重建 `rank[id]`、全部 pending 字典与 running；只在这里 observe 当拍 arrivals；计算本拍固定历史 pressure。 |
| `on_candidate_resources(now, available)` | `available` 是 `(cores, gpus, mem)`；仅 can-fit 候选；当前 task ID 不在此参数中 | 验证 now 不陈旧并保存当前 available；候选由后续 `decide(spec, context)` 标识。 |
| `on_task_started(running)` | 新启动的 `RunningTask(id, demand, actual_finish, deadline)` | 删除对应 pending，追加 running，即时更新已占用需求/紧迫前序工作索引。 |
| `on_task_deferred(task_id, until)` | 已 can-fit 且主动延一拍；until 为 now+STEP | 保留 pending 和 urgent contribution；不占 running 资源。可记录延迟身份，不能把它作为本拍可重试请求。 |

建议独立类接口（类名可在实现冻结时确定）：

```python
class TaskRiskBudgetPolicy(EDFBackfill):
    def __init__(self, name, parameters, experiment_start, capacity): ...
    def on_step_start(self, snapshot): ...
    def on_candidate_resources(self, now, available): ...
    def on_task_started(self, running): ...
    def on_task_deferred(self, task_id, until): ...
    def decide(self, spec, context) -> PolicyDecision: ...

def make_task_risk_policy(name, parameters, experiment_start, capacity): ...
```

预算函数应接收当前已到达 `spec`、本拍 rank、仍 pending 的前序工作、当前 running 利用率与原 `ArrivalHistory.pressure`，返回纯数据预算及可审计分项。独立注册只新增新名称；原 Carbon/RiskOnly 委托旧 `make_cars_policy`，不改旧类、参数或版本。

## 4. 动态 pending 与同拍资源更新

“前序”由拍初 EDF rank 确定，当前竞争集合由真实启动事件删除；本拍不重新排序。对候选 i，统计严格 `rank_j < rank_i`、仍 pending 且 `floor_grid(D_j-d_j)-now <= 120min` 的任务。负余量/过期任务仍属于紧迫集合；候选自身不计。先前主动 defer 或 capacity fail 的任务仍在集合，先前启动者即使本拍尚未完成也仅计 running，不重复计 pending。

当前利用率优先由维护的 running 三维需求求和/除 capacity，或用容量与最新 available 交叉核查；不能一直复用 `snapshot.available`。每个 can-fit hook 都提供上一次启动后的实际 available，启动通知进一步保证候选之间的状态一致。浮点求和/缓存优化不得改变公共执行器容量判断。

正确性参考版可保存 EDF ordered tuple 与 pending 字典，在候选查询时只扫描初始顺序中严格前序且仍 pending 的任务；无需每次排序，单拍最坏 `O(n²)`。性能测量若证实必要，可优化为三维 EDF 前缀工作，启动删除用 Fenwick/后缀更新；只有启动删除，defer 和 capacity fail 不删。优化必须逐候选对照参考版，不能用不同集合改善速度或效果。

### 4.1 实施后的缓存适配复核

随后实现选择 `query_backend='reference'|'cached_work'`，默认cached_work。它只缓存本拍静态 TaskSpec 的网格最迟启动与单任务三资源占用乘积 `q_r*occupied_minutes`，不采用浮点prefix subtraction或Fenwick累计。每次候选仍从同一严格EDF前序中过滤仍pending成员、按同顺序用 `math.fsum` 聚合，running也按真实启动后的列表重算；defer/capfail没有删除缓存或pending，已启动成员因为pending过滤不再贡献紧迫队列。

因此两后端每候选成本为 `O(prefix+running)`，完整一拍可达 `O(n²+n*m)`，另含一次 `O(n log n)` EDF排序；m为拍初running，n为pending。缓存降低timestamp转换和重复乘积构造的常数，**没有宣称线性或log级查询**。512候选合成profiling日志中前序扫描均130816，参考3.151792s、缓存0.065773s；这是实现profiling，不是论文算法加速或真实全窗结果。参考版持续保留，逐候选5策略diagnostics与微型完整ExecutionResult相等见新增测试。

## 5. 历史、cold、两种时间量

`cars_state.py:ArrivalHistory` 保存最近 `(now-120min, now]` 到达工作，ID 去重，`arrival <= cutoff` 会过期。工作使用旧语义 `15*ceil(duration/15)`；有效分母 `min(120min, now-experiment_start)`。每拍仅观察当前 arrivals，不在每次 candidate 重复 observe；旧 ID 去重仍保留。

旧 cold 的精确定义是 **effective_minutes==0**，此时三维 rho 为 `None`、cold buffer=2。之后即便历史窗口没有任务，已观察的空窗口给出 rho=0；不能任意重新加 cold。新 no_history 仅关闭历史压力项，并保留所有策略共同的 cold 定义。

本轮预算诊断分母固定为：公共容量判断可容纳、当前 CI 严格高于阈值且 `S_i>=15min` 的 budget-eligible 候选。其他 can-fit early-exit 候选记录 `B=null`，不得当0纳入缓冲分布；capacity-failed 候选只有公共执行器容量事件，不产生本policy预算诊断。比较不同策略分布时同时列明eligible数量、全部can-fit数量与容量失败数。

原 duration 和合成 deadline 来自 `workload.py:TaskSpec`，不能改写。实际 finish 为 `s+d`、SLA 为 `finish<=D`；资源释放为首个网格 `ceil_grid(finish)`。最迟可按时网格启动为 `floor_grid(D-d)`，不是 `D-ceil15(d)`。`d=16min,D=35min` 时 start=15、finish=31、release=45、原 SLA 按时；now=0 的网格 slack=15min。

## 6. RL/物理模型与在线边界

`rl_env.py` 固定动作 0/1/2 仍调用 EDF/EDF+Carbon/旧 RiskOnly；reset 创建一次物理模型、每拍复用。其历史观测也去重；E1-6A 不给这些旧模式注入新 budget，不修改 checkpoint 的 40 维 schema。旧固定动作账本/逐拍能耗等价测试应继续通过。

`power.py:UpstreamPowerModel.measure` 在 step 前注入本拍实际 CPU/GPU/MEM 利用率与真实温度，积分的是分配后整拍负载。新策略改变合法启动时机，不改变物理方程、容量规模、输入时间轴或能碳单位。

`sla_feasibility.py` 是离线有限集合必要条件分析，含未来 arrival/deadline，**禁止 import/call 为本在线方法的输入**。新 policy 应不持有 FrozenWorkload/environment/full-ledger 的引用。在线输入限已到达 spec、当前和历史 CI、snapshot、旧 arrival history；未来环境或公共排空终点只能由实验核算层使用。

## 7. 实施验收要求与尚未验证边界

需要重点测试：capacity-failed 前序仍贡献 urgent；deferred 前序保留；started 前序删除且 running 增加；同拍 current u 更新；raw finish/grid release；safe==15 等号；120min 过期与 cold；CPU/GPU/MEM-only 无关资源隔离；零需求；weighted 小需求瓶颈与 required-max；未来轨迹分叉；全队列 EDF/backfill；资源与任务守恒；旧 RL 固定动作/训练环境回归；冻结 replay。

预算名称中的 safe 只表示扣除启发式风险保留量。它没有建模未知未来任务、同时到达装箱最优性、非抢占多资源耦合的充分可行条件，不能给出严格 SLA 或概率保证。真实 trace 的 duration/usage 在到达已知、合成 SLA1.5、分数资源池、15min积分与有限集合 Drain 仍是继承的模型边界。

## 8. 本次实读旧源码 SHA256

以下哈希是实施前读到的原工作区文件字节身份，后续保护检查应独立复核；不是改写历史 provenance。

| 文件 | SHA256 |
|---|---|
| `engine.py` | `01ed38ba8a513fb4e0a60bf707f4160a0f7f45674eaf29ff32103421006f9ef8` |
| `cars.py` | `8c38fa046c5ee0d28afbb978b535e49fe3f1ee17b1a01afbb917024c08fcc96a` |
| `cars_state.py` | `032ada5d6dcb79eb3fcaf90a0c91d98a80e03686ad1cc64688e60adea9fa26d5` |
| `policies.py` | `f4a58cc5cb4c07be0c0c34e1ce95d8020b948d14d3eaf2453d048a0df9461618` |
| `rl_env.py` | `088060427ffdc492f47dfb7bbd20384fe1d35c96d3e6480dd0c8a48a52f910c4` |
| `workload.py` | `14a3576c1f569632463516b8589a58e1955bdba94150088471affd6b6aa4f961` |
| `sla_feasibility.py` | `432599bc5d79c8db00ecaa4fb27efc2f4c4aa032afe114cb0d97e68bee92c1d4` |
