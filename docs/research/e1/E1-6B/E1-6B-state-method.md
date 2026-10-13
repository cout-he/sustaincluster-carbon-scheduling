# E1-6B 连续任务风险状态与候选过滤接口

本文件说明独立四动作环境的83维状态：原 E1-5A 的40维公共状态，加43维连续风险块。E1-6A v1公式与参数未改变。这个风险块描述当前已知队列，过滤模块是待验证的启发式限制；测试通过不等于能保证未知未来下的SLA或获得稳定收益。

## 1. 对外接口和消融含义

`experiments/e1/adaptive_observation.py`导出：

```python
RISK_SCHEMA = 'e1-6b-continuous-risk-observation-v1'
RISK_FIELDS: tuple[str, ...]  # 固定43项顺序
risk_observation(snapshot, pressure, normalization=None)
# -> (np.ndarray(shape=(43,), dtype=np.float32), rawdict)
```

参数只有当前`ResourceSnapshot`、已观测`Pressure`和预注册工程归一化配置，没有完整workload、环境轨迹、其他策略结果或离线可行性下界入参。raw含`schema`、`population`、43项原始`fields`、逐任务`tasks`（task_id、frozen Budget、can_fit）、`units`和`normalization`。Actor只收到43维vector，逐任务raw用于info审计。

独立`AdaptiveCarbonSchedulingEnv`组合原`CarbonSchedulingEnv`，拼接其原样40维输出与新块；旧类仍为40维/Discrete(3)，所有旧checkpoint接口不变。新类仍推进内部同一个LocalExecutor并沿用同一reward和持久物理对象，不建立另一个分配执行路径。

四组都输入83维；`risk_state=False`时新增43维全部为0，网络结构/参数量不变。**公共40维本来已含资源利用率、历史到达压力、原RiskOnly全局缓冲和solo slack统计**。因此PPO-Plain是关闭新增任务风险信息，不是完全没有风险信息的策略。新增状态的效果应解释为在公共基础上的边际贡献。

## 2. 时间位置和预算公式

拍前快照位于当前到达已接收、此前完成已释放、尚未做本拍调度的位置。全部当前pending按公共EDF `(deadline, arrival, stable_id)`排序，包含capacity-blocked和deadline已过任务。拍前预算不假设该拍任何候选将被启动。

对于任务i：

- `L_i=floor_15min(deadline_i−raw_duration_i)`；`S_i=(L_i−now)`分钟；允许负值。
- `m_ir=q_ir/C_r`，`w_ir=m_ir/sum_r(m_ir)`，各维为CPU/GPU/MEM；全零需求另规定z=B=0。
- `u_r=math.fsum(current_running_demands_r)/C_r`。
- `Q_ir=sum_{j<i, j仍pending, L_j−now≤120min}(q_jr*occupied_minutes_j)/(C_r*120min)`。过期前驱也纳入。`occupied_minutes=15*ceil(raw_duration/15)`，与资源释放网格一致。
- `rho_r=observed_arrival_resource_minutes_r/(C_r*effective_history_minutes)`；窗口`(now−120min,now]`，分母`min(elapsed,120min)`。只有elapsed=0时未知，不将较短已知窗口误认为未知。
- `z_i=sum_r w_ir*(u_r+Q_ir+max(0,rho_r−0.5))`。rho未知时历史数值项0；`cold=2`仅在elapsed=0，否则0。
- `B_i=15*ceil(cold+min(8,z_i))`分钟；`safe_i=S_i−B_i`。cold最大加到150分钟。未修改E1-6A系数或ceil规则。

观察块的solo/safe采用上述网格L。旧40维的既有solo统计仍保持原来的非floor理论latest_start；不为新模块重写旧维度。

候选过滤在executor容量检查成功之后，使用本拍**实时**running和仍pending严格前驱重新计算B：本拍已启动任务立即加入running并从pending移除；defer和capacity-failed任务保留。拍前状态预算不可作为整拍不可变过滤预算。Observation原始Budget与候选diagnostics的分母不同：前者全pending，后者只含真正调用策略的can-fit候选，且只有需要过滤的候选有B。

## 3. 43项固定顺序、单位与归一化

记`f(x;s)=x/(s+abs(x))`，它单调、有界、不硬裁切，保留正负；默认slack尺度s=120分钟，z/rho尺度1，缓冲尺度150分钟。所有尺度预先声明，不从Validation/Test或全窗口数据拟合。分位数采用排序样本上`(n−1)*p`位置的线性插值；均值用`math.fsum/n`。

下面“块位置”为1起算；在83维总体状态中的位置为块位置+40。

| 块位置 | 字段 | 原始量/分母 | 单位 | Actor变换 |
|---|---|---|---|---|
|1|risk_u_cpu|当前running CPU/C_cpu|无量纲|原比例[0,1]|
|2|risk_u_gpu|当前running GPU/C_gpu|无量纲|原比例[0,1]|
|3|risk_u_mem|当前running MEM/C_mem|无量纲|原比例[0,1]|
|4|risk_rho_cpu|历史CPU资源分钟/(容量×有效分钟)|无量纲|f(x;1)|
|5|risk_rho_gpu|历史GPU资源分钟/(容量×有效分钟)|无量纲|f(x;1)|
|6|risk_rho_mem|历史MEM资源分钟/(容量×有效分钟)|无量纲|f(x;1)|
|7|risk_z_min|全部pending z最小值|无量纲|f(x;1)|
|8|risk_z_mean|全部pending z均值|无量纲|f(x;1)|
|9|risk_z_p50|全部pending z中位数|无量纲|f(x;1)|
|10|risk_z_p90|全部pending z的p90|无量纲|f(x;1)|
|11|risk_z_max|全部pending z最大值|无量纲|f(x;1)|
|12|risk_z_cpu_mean|CPU主导任务z均值|无量纲|f(x;1)|
|13|risk_z_cpu_present|CPU主导任务是否存在|flag|0/1|
|14|risk_z_gpu_mean|GPU主导任务z均值|无量纲|f(x;1)|
|15|risk_z_gpu_present|GPU主导任务是否存在|flag|0/1|
|16|risk_z_mem_mean|MEM主导任务z均值|无量纲|f(x;1)|
|17|risk_z_mem_present|MEM主导任务是否存在|flag|0/1|
|18|risk_z_tie_mean|归一化需求最大值并列任务z均值|无量纲|f(x;1)|
|19|risk_z_tie_present|tie类别是否存在|flag|0/1|
|20|risk_z_zero_mean|全零资源需求任务z均值（规定为0）|无量纲|f(x;1)|
|21|risk_z_zero_present|全零资源需求任务是否存在|flag|0/1|
|22|risk_solo_slack_min|全部pending S最小值|分钟|f(x;120)|
|23|risk_solo_slack_mean|全部pending S均值|分钟|f(x;120)|
|24|risk_solo_slack_p10|全部pending S的p10|分钟|f(x;120)|
|25|risk_solo_slack_max|全部pending S最大值|分钟|f(x;120)|
|26|risk_safe_slack_min|全部pending safe最小值|分钟|f(x;120)|
|27|risk_safe_slack_mean|全部pending safe均值|分钟|f(x;120)|
|28|risk_safe_slack_p10|全部pending safe的p10|分钟|f(x;120)|
|29|risk_safe_slack_max|全部pending safe最大值|分钟|f(x;120)|
|30|risk_solo_ge60|S≥60任务数/pending任务数|比例|[0,1]|
|31|risk_solo_ge30|S≥30任务数/pending任务数|比例|[0,1]|
|32|risk_solo_ge15|S≥15任务数/pending任务数|比例|[0,1]|
|33|risk_safe_ge60|safe≥60任务数/pending任务数|比例|[0,1]|
|34|risk_safe_ge30|safe≥30任务数/pending任务数|比例|[0,1]|
|35|risk_safe_ge15|safe≥15任务数/pending任务数|比例|[0,1]|
|36|risk_pending_present|pending非空|flag|0/1|
|37|risk_history_known|有效历史分钟>0|flag|0/1|
|38|risk_buffer_mean|全部pending B均值|分钟|x/150|
|39|risk_buffer_max|全部pending B最大值|分钟|x/150|
|40|risk_z_bin_0_1|0<z≤1任务数/pending任务数|比例|[0,1]|
|41|risk_z_bin_1_2|1<z≤2任务数/pending任务数|比例|[0,1]|
|42|risk_z_bin_gt2|z>2任务数/pending任务数|比例|[0,1]|
|43|risk_can_fit_fraction|逐任务当前三资源可容纳的任务数/pending任务数|比例|[0,1]|

类别根据`q/C`最大的维度判定，不以CPU/GPU/MEM原始单位直接比较；并列为tie，全零为zero。类均值和presence成对，缺类别时均值0/presence0。z=0不包含在三种正区间比例中，因此它们之和可能小于1；pending_present可区别空队列与非空零分数。

can-fit只逐任务与**当前**running求`math.fsum`后比较各容量，不模拟整个pending同时放入、不预留未来资源。它采用与LocalResourcePool一致的加法判定，避免`available=capacity−used`相减误差将`.2+.6+.2`误判为超容量。

空队列时所有task聚合为0，pending_present=0；u和已知rho仍反映真实当前running/历史，不能因为没有pending而强制置零。初始无历史rho数值置0，history_known=0；已知零压力则数值也0但history_known=1。风险块关闭时才整块置0，包括presence。

## 4. 四动作与过滤

|动作|无过滤允许本拍主动defer的条件|过滤开启的条件|
|---|---|---|
|0|从不主动defer|同左|
|1|当前CI>161.075且S≥60min|当前CI>161.075且safe≥60min|
|2|当前CI>161.075且S≥30min|当前CI>161.075且safe≥30min|
|3|当前CI>161.075且S≥15min|当前CI>161.075且safe≥15min|

门槛相等允许defer，CI相等执行；每次defer只有15分钟。60/30/15是本拍再次等待前必须具有的剩余slack，不是单次等待时长。候选先经过真实CPU/GPU/MEM容量检查，过滤不能绕过容量、修改EDF、抢占或保证成功启动。

无过滤Action0的任务执行与EDF一致，无过滤Action3与EDF+Carbon一致，过滤Action3与旧Task Full一致；diagnostic reason标签属于新schema，不能把标签变化当科学账本变化。固定动作等价性优先比较真实任务、事件、启动/defer集合、资源、IT/HVAC和碳积分。

在当前整数网格上，连续B与ceil后的B对三门槛判定严格等价；保留连续z的意义是给Actor更多当前风险信息，并不取消原B或假设它一定有效。PPO-State和PPO-Combined是否利用这些信息，需由正式消融结果判断。

## 5. 在线性与复杂度边界

一次观测将全部pending进行一次EDF排序，在同一快照中准备静态grid/work事实。逐候选的严格前驱资源工作量保留成员与顺序，采用与frozen参考一致的`math.fsum`；没有利用减法更新浮点前缀来改变ceil边界。对n个pending、m个running：准备和聚合排序O(n log n)，最坏前驱查询O(n²)，can-fit O(n*m)，额外内存O(n+m)。环境构造快照时还调用公共EDF排序；这一额外相同排序只增加成本，不改变策略语义。运行成本须在smoke与正式推理计时披露。

Observation是决策前的真实已知状态；step返回的下一状态可合法包括下一边界新到达和新CI，但它们不能进入此前action的特征或过滤。未来轨迹分叉测试分别检查当前Observation、相同当前action的真实interval，以及下一边界合法新信息。

风险不是实际违约概率；特别是weighted聚合可能压低小需求紧缺资源的影响，max又可能过度保守。B0反例与E1-6A负面结果保留，未据开发Validation调整权重。

## 6. 实际验证

`tests/e1/test_adaptive_observation.py`先运行RED：11项失败均因模块未存在（0.87秒）；新增实现后GREEN：11 passed（2.66秒）。随后新状态与原task_risk_state/task_risk/旧rl_env关键回归合并：**84 passed（2.28秒）**。测试覆盖手算43维统计与归一化、全部pending含容量阻塞、逐Budget与冻结参考精确相等、unknown/empty/tie/zero、fsum容量边界、真实start后的状态变化、future fork、工程尺度和快照不变性。

只读源码复核`adaptive_observation.py`、`adaptive_risk.py`、`adaptive_rl_env.py`确认历史到达在所有动作下保持观察、同拍重复通知去重、policy候选查询实时状态、CI/网格边界以及旧reward/物理内核复用。此处不替代完整E1回归、训练参数更新或公平Validation验收。
