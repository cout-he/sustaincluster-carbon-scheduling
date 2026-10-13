# E1-6A：任务级多资源竞争感知延迟预算 v1

方法版本：`e1-6a-task-risk-budget-v1`。日期：2026-10-13。本文在实验前按用户批准的 E1-5C 首选公式固定定义与参数，不依据 Validation 胜负调系数。它是可实现的在线启发式风险保留量，不是严格可行调度证明、违约概率或未知未来 SLA 保证。正确性测试与实际性能须分别交付。

## 1. 时钟、任务与容量

时间统一 UTC，决策拍 `Δ=15min`。任务 i 的到达 `a_i`、原 duration `d_i>0`、三维需求 `q_ir>=0`、deadline `D_i=a_i+1.5*d_i` 均保持冻结输入；合成 deadline 不是 trace 原有 SLA。资源 r 为 CPU/GPU/MEM，容量单位依次 core/GPU/memory-unit，`C_r>0`。

任务真实开始 s_i 为网格时刻，实际完成 `f_i=s_i+d_i`，按时定义为 `f_i<=D_i`。占用/释放采用公共执行器规则：`p_i=15*ceil(d_i/15)`，资源释放 `R_i=ceil_grid(f_i)`，整拍物理占用 `[s_i,R_i)`。最迟按时网格开始与当前 solo slack 为：

```text
L_i = floor_grid(D_i - d_i)
S_i(t) = (L_i - t) / 1min
```

floor_grid 在 UTC 固定 15min 网格，不改变 duration 或 deadline。`d=16min,D=35min,a=0` 得 L=15min；在15min开始，31min完成，45min释放。不能用 rounded p_i 替换 d_i 计算 L_i。

## 2. 在线状态与分项

| 符号 | 定义 | 单位/范围 | 在线来源 |
|---|---|---|---|
| `m_ir` | `q_ir/C_r` | 无量纲、非负；可超过1但此任务不能通过容量检查 | 当前候选原声明需求 |
| `w_ir` | `m_ir/sum_r(m_ir)` | `[0,1]`，正总需求时和为1；无需求维为0 | 仅当前候选 |
| `u_r(t)` | `running_demand_r(t)/C_r` | `[0,1]`；以公共执行器合法分配为准 | 当前候选时真实 running，含同拍此前启动者 |
| `P_i(t)` | 严格前序、仍 pending、`L_j-t<=120min` 的任务 j | 已知集合；含负slack、capacity waiting、policy defer | 当前拍 EDF rank 与动态启动通知 |
| `W_urgent,ir` | `sum_(j in P_i) q_jr*p_j` | core-/GPU-/memory-unit-minute | 原声明 duration/需求，不是未来完成信息 |
| `Q_ir` | `W_urgent,ir/(C_r*120min)` | 无量纲、非负、可>1 | 当前已知紧迫前序队列 |
| `W_hist,r(t)` | 旧 ArrivalHistory 最近 `(t-120min,t]` 已观测到达的 `sum(q_jr*p_j)` | 各资源-minute | 当前与历史已到达，不读取未来任务 |
| `H_eff(t)` | `min(120min,t-experiment_start)` | 分钟，0至120 | 当前已观察时长 |
| `rho_r` | `W_hist,r/(C_r*H_eff)`，仅 H_eff>0 可估计 | 无量纲、非负、可>1；冷时 `None` | 旧 ArrivalHistory |
| `b_cold` | H_eff=0 时2，否则0 | 拍 | 完全沿用旧 unknown 标志 |

同拍 EDF 排序 key 为 `(D_i,a_i,task_id)`。`P_i` 不包含 i，也不包含任何 rank 高于或等于 i 的任务。不存在紧迫性下界：已经错过 L_j 的已知 pending 仍形成竞争工作。已真实启动者从 P 删除并加入 running；已 defer/容量失败者留在 P，不虚构预留资源。

ArrivalHistory 的 no-arrival 空窗口在 H_eff>0 时 rho=0，表示这段已观测时间没有到达，**不是重新 cold**。cold 只在起始有效时长为0时出现；不使用完整工作负载填充前史。历史 ID 去重，即使重放边界通知也不能重复累计；新策略正常每拍只 observe 一次 arrivals。

## 3. 固定公式与参数

初版配置字面值：

```text
a_u=1, a_q=1, a_h=1
rho0=0.5, k=1, K=8
cold_buffer_steps=2, H=120min, Δ=15min
CI_threshold=161.075 gCO2e/kWh
```

历史已知时定义每维压力：

```text
h_r = max(0, rho_r - 0.5)
x_ir = a_u*u_r + a_q*Q_ir + a_h*h_r
z_i = sum_r w_ir*x_ir
B_steps,i = ceil(b_cold + min(8, z_i))
B_i = 15min * B_steps,i
safe_slack_i = S_i - B_i
```

历史 unknown 时省略 h 项，但仍应用 b_cold=2 和当前 u/Q。所有需求均为0时，单独令 `w=(0,0,0), z=0, B_steps=0, B=0`，不因cold获得容量风险缓冲。非冷 B 可为0至120min，cold 总 B 可为30至150min。ceil 按公式原值执行；不得使用未注册 epsilon 将略高于整数的 z 压下以制造更长延迟预算。

`safe_slack` 可为负，不截断成0掩盖风险；B 是风险保留分钟，不是睡眠时长，也不是消耗/预留的运行资源。下一次候选评估重新计算，未启动任务原 deadline、duration 与 demand 均不改变。

## 4. 确定性执行条件与伪代码

容量检查由公共执行器先进行，失败不调用预算决策，记录 capacity waiting 并继续回填。对能容纳候选：

```text
每拍开始：
  按全部当前 pending 的 EDF rank 建立观测状态
  ArrivalHistory.observe(now, arrivals)；pressure(now)
  保存当前 running 和全部 pending

公共执行器按 EDF 一次遍历每个候选 i：
  若容量不够：公共执行器 requeue，仍保留策略 pending；继续回填
  否则：on_candidate_resources 更新实际 available
    若 CI <= 161.075：execute（low_ci）
    否则若 S_i < 15min：execute（no_one_step_slack）
    否则：
      计算 w,u,P,Q,rho,z,B,safe_slack
      若 safe_slack_i < 15min：execute（task_budget_guard）
      否则：defer exactly 15min（task_budget_allows_one_step）
  真正启动通知：pending删除i；running加入i；后续候选更新u/P
  真正延迟通知：pending保留i；running/资源不变
```

阈值相等执行；`safe_slack==15min` 允许 defer；`S==15min,B==0` 也允许。理论最迟启动已过、deadline已过的任务仍正常启动或容量排队，不删除、不重写 SLA。只能禁止主动 defer，不能保证在拥塞时立即获得资源。

本轮冻结的诊断分母是 **can-fit、CI>161.075、S_i>=15min** 的 budget-eligible 候选，只在该分支计算/统计B。低CI和无一拍slack的early-exit记录 `B=null`，不能编码成已知0或加入B分布；容量失败不调用policy，因此没有预算诊断，仅在公共capacity事件中计数。报告同时列eligible、全部can-fit和capacity失败数量。不同策略产生不同后续状态与eligible分母，B分布与guard计数不能简单相减作因果解释。

## 5. 预注册消融

所有新策略使用同一EDF排序、容量/回填、CI阈值、SLA、历史窗口、cold、clock和物理模型，仅改变表中指定项。

| 名称 | 唯一改变 | 聚合 |
|---|---|---|
| `edf_carbon_threshold` | 原旧类，不注入新预算 | 原 solo一拍规则 |
| `cars_risk_only` | 原旧类和全资源历史max映射，不修改 | 旧公共0/1/2拍 |
| `task_risk_full` | 完整v1 | `sum(w*x)` |
| `task_risk_no_history` | 用户明确冻结为仅 `a_h=0`，其他保持；仍共同cold2，unknown仅episode elapsed=0 | `sum(w*x)` |
| `task_risk_no_urgent` | `a_q=0`，其他保持 | `sum(w*x)` |
| `task_risk_no_utilization` | `a_u=0`，其他保持 | `sum(w*x)` |
| `task_risk_max_required` | 聚合替代为 `max_(r:q_ir>0) x_ir`，不对无需求资源取max | required-resource max |

全零需求的 required-max 为空集，仍按共同零需求分支 B=0。`max_required` 是预先声明的瓶颈保护对照，不根据Validation最优结果开放其他聚合自由搜索。旧RiskOnly即便CPU-only任务也取GPU高rho，属于旧机制本身；不能为了公平而偷偷“修复”旧算法。

## 6. 表达能力微型验证与瓶颈反例

非冷历史，容量 `(50000 core,1000 GPU,80000 memory-unit)`。当前 GPU u=.5、GPU rho=1.6；CPU/MEM u与rho=0。两个 d=60min、S=30min、相同deadline/arrival的任务按稳定ID排序：A GPU-only q=250，B CPU-only q=12500；两者当前都能启动，高CI。

A 无紧迫前序，`z_A=.5+1.1=1.6`，B_A=30min，safe=0，执行；其启动把GPU u增至.75。B对GPU权重为0，前序A已从pending删除，CPU当前/历史/紧迫压力均0，因此 B_B=0、safe=30，可defer。反方向交换 CPU/GPU 构造同等测试。旧RiskOnly两任务公共2拍buffer、要求S>=45，因此都会执行；旧EDF执行全部，旧Carbon延迟全部。该任务子集是新的表达能力，不是实测论文优势。

加权平均存在明确局限：任务容量占比 m_CPU=.5、m_GPU=.001，权重GPU约.001996；若当前 GPU u=.999，未来其极小GPU需求仍可能遇到瓶颈，而大量CPU权重稀释GPU高压。已知可容纳条件和当前聚合压力不能推断未来充分可行性。后续 microtest 应让候选当前恰好能容纳，同时令GPU历史/紧迫压力高，检查 weighted 与 required-max 是否出现明显B差异；将该差异作为机制边界保留，不事后换公式抹去负例。

## 7. 实现复杂度与性能边界

简单参考版每拍排序 `O(n log n)`，历史观察沿用旧去重/窗口累计；每个候选只对既有 rank 前序仍pending扫描，不再次排序，不调用 Shadow 或前向模拟，同时对当时running三维fsum。每候选 `O(prefix+running)`，最坏每拍 `O(n²+n*m)`，额外状态 `O(n+m)` 加旧 history seen。n为pending、m为拍初running；同拍新启动的至多n个任务已计入n²。seen随累计已到达增长，这是沿用旧实现，不意味着政策可读取未到达任务。

实际初版在profiling后采用静态缓存 `cached_work`：缓存每任务的L与三资源q*p，候选仍使用同成员、同EDF顺序、同乘积与math.fsum，并从实时pending过滤启动者；没有对prefix作浮点相减。其最坏渐近复杂度与参考相同，减去重复datetime与单项计算常数。纯calculate_budget与query_backend='reference'保留。测试逐候选覆盖五变体的启动/defer/容量失败动态通知，并比较完整微型ExecutionResult；不允许凭最终总指标相等代替这种诊断一致性。

若未来profiling仍证明高压队列查询成为主要开销，可另评估每资源 EDF urgent prefix或Fenwick；它们在更新相减时可能引入浮点误差并跨越ceil整数边界，必须逐候选匹配原成员、数值与decision后另注册优化。当前v1没有采用该路径，不宣称 `O(log n)`。不得提前删除capacity-failed任务或改变ceil边界以通过速度测试。

## 8. 数据、验证与科研结论范围

先Train 1/1.5/2完成功能与参数冻结，再开发Validation1/1.5/2。数据身份与旧 E1-5C冻结集合一致；不得重写旧报告或把开发Validation称未见泛化。Test保持封存；不训练PPO、不改变既有动作接口，不将离线sla_feasibility、未来CI/arrival/common-end输入policy。

同窗口同倍率用完整同任务量、完整Drain与共同观察终点；先排空者按相同持久物理模型积分真实空闲尾段。B分布、资源主导类别、reason counts、主动/容量等待、任务资源小时与运行开销同DMR/Carbon/JCT一起报告。所有方法保持5%门槛，高压力另报相对配对EDF差异与Pareto，不能改目标。若B过大几乎总execute、退化EDF、weighted不如旧RiskOnly或只有SLA改善没有Carbon优势，均保留v1失败结果。

这套实现与测试最多证明按声明公式、正确事件和在线边界运行；真实效能以冻结消融结果决定；学术新颖性、实机duration预测误差与未知未来deadline保证均未在本阶段证实。
