# E1-5C：任务级风险预算与安全 RL 的下一阶段接口方案

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent，设计与只读验证。
- Version: e1-5c-next-stage-interface-proposal-v1。
- Source: E1-5A/E1-5B 的现有源码与冻结报告；发布父提交 `958c14137bfeb30aaf6245a93a3cb693bcb60d20`。
- Verification Status: ANALYZED。公式与接口为候选设计，本轮没有实现、训练或验证其性能。
- 范围：Train 与开发用 Validation；没有读取 Test 冻结输入、Test CI 切片或 Test 策略结果。历史数据划分说明只作为既有契约阅读。

本方案的可审查内容是两个模块的输入、单位、决策接口和消融方法。资源压力档位及 5% 的适用范围须以本轮实际诊断表为准；本文不把未运行的新方法称为有效创新或 SLA 保证。E1-5B 的 5% 准入失败与 `selected_policy=null` 保留。

## 1. 从现有实现得到的约束

`LocalExecutor` 是唯一执行内核。`start()/advance()/run()` 共享 `engine.py:_execute()`：当前拍到达登记 → 释放已完成任务 → 新任务入队 → 挂起并返回当前可见状态 → 对所有尚未启动任务按 EDF 排序 → 对每个候选先检查资源 → 只对能够容纳的任务调用策略 → 启动立即占用资源或主动延迟 → 继续回填 → 守恒检查 → IT/HVAC 物理计算 → 本拍积分。终点只 flush 完成，不增加新任务、启动或功耗拍。策略变化不得抢占已运行任务。

当前 `cars.py:CarsPolicy.decide()` 的 RiskOnly 使用 `cars_state.py:ArrivalHistory.pressure()`：最近 `(t-120min,t]` 已观察到达的占用工作量除以容量及有效历史分钟数，取 **三资源 rho 的最大值**，统一映射为 0/1/2 拍缓冲；冷启动缓冲为 2。不同任务只通过各自 solo slack 与这一个公共缓冲比较。它没有当前队列中任务需求与 deadline 的匹配信息，也没有因某任务主要依赖 GPU 或 CPU 而改变缓冲。

`rl_env.py` 的 40 维状态包含 CPU/GPU/MEM 的利用率、历史到达工作与 rho，以及 pending 的聚合资源需求/工作和 slack 统计。它没有保留“某任务的 GPU 需求与该任务紧迫程度”的配对；不同任务组合可能拥有相同聚合统计。`Discrete(3)` 则只选择既有 EDF、EDF+Carbon、RiskOnly；相同 CI/slack 下模式选择不能任意决定任务子集。

必须同时保存两个时间量，不能混用：

- 实际完成：开始于网格时刻 `s_i`，`f_i=s_i+d_i`；`f_i <= D_i` 为按时，`d_i` 不被重写。
- 资源释放与能源积分：`R_i=ceil_grid(f_i)`，资源占用 `[s_i,R_i)`，占用分钟 `p_i=15*ceil(d_i/15)`；最后不足 15 分钟仍按既有左边界整拍占用计算。

因为 deadline 按实际完成判定，理论网格最迟开始应为 `L_i=floor_grid(D_i-d_i)`，而不是 `D_i-p_i`。例如 `d=16min,D=35min`，15 分钟网格的最迟开始为 15min，完成 31min 按时，释放 45min；使用 `D-p=5min` 会错误排除这次可行开始。到达已在网格且集合左闭右开；已过 `L_i` 的任务仍执行，不能删除或隐藏违约。

## 2. 模块 A：轻量、需求匹配的任务级风险预算

### 2.1 共同变量与在线边界

令 `r∈{CPU,GPU,MEM}`，固定容量 `C_r` 单位分别为 core、GPU、memory-unit。只使用已到达任务的声明 duration 与需求 `q_ir`；这仍继承上游“duration/observed-usage 在到达时已知”的仿真假设，不能视为实机预测准确率保证。

| 变量 | 定义与单位 | 可观测性 |
|---|---|---|
| `S_i=L_i-t` | 当前网格 solo slack，min，可为负 | 本任务已知 deadline/duration 与当前时钟 |
| `m_ir=q_ir/C_r` | 本任务对资源 r 的容量占比，无量纲 | 本任务声明需求 |
| `w_ir=m_ir/sum_k(m_ik)` | 需求匹配权重，无量纲；`q_ir=0` 时权重为 0 | 当前任务；全部需求为 0 则三权重为 0 |
| `u_r=used_r/C_r` | 当前已运行利用率，无量纲 | 当拍释放后；每次真实启动后更新 |
| `rho_r=W_hist,r/(C_r*H_eff)` | 过去 120min 已观察到达占用工作压力，无量纲 | 沿用旧历史端点、去重和冷启动语义；不读取未来到达 |
| `P_i(t)` | 当前 EDF 顺序中优先于 i 且 `L_j-t<=H` 的尚未启动任务集合，`H=120min` | 已知队列，含已过最迟开始任务；排除 i；同拍已启动者移至 running |
| `W_urg,ir=sum_(j in P_i) q_jr*p_j` | 优先紧迫队列资源工作，core-/GPU-/memory-unit-min | 只用当前已知 pending；不是其未来实际完成情况 |
| `Q_ir=W_urg,ir/(C_r*H)` | 归一化紧迫队列压力，无量纲 | 同上 |
| `A_r(t+h)` | 仅按当前 running 的 `ceil_grid(f_j)` 释放估计未来 h 的可用容量 | 已知非抢占运行任务；未来新任务与未来策略启动不包括在内 |

历史压力冷启动不能用完整轨迹预先填充。候选实现应沿用“unknown 标志 + 保守 2 拍基值”的明确处理，在有效历史长度为 0 时不除零、不把未知 rho 当已证实零压力。实际新模块的冷启动及系数必须在下一阶段实现前注册。

### 2.2 首选候选：任务匹配的多源缓冲

推荐先评估下列轻量候选，不使用 Shadow Planner：

```text
z_i(t) = sum_r w_ir * [a_u*u_r + a_q*Q_ir + a_h*max(0,rho_r-rho0)]
B_i(t) = 15 * ceil( b_cold(t) + min(K, k*z_i(t)) )       [min]
safe_slack_i(t) = S_i(t) - B_i(t)                        [min]
```

`a_u,a_q,a_h,k,rho0,K` 均无量纲、非负；`b_cold` 单位为拍。`K` 仅限制风险保留时间，不删除任务；`S_i` 和 `safe_slack_i` 的负值保留。第一轮候选可以先固定 `a_u=a_q=a_h=1,rho0=0.5,k=1,K=8`、冷启动 `b_cold=2`（历史已知时 0），用于解释性微型验证。**这些是待三方审核的提案，不是本轮运行参数，不代表经过数据拟合的最佳值**；下一轮在任何训练/模型比较前将所选字面值与版本号冻结，后续调整须另注册，不用 Validation 胜负或 Test 调节。

公式的空值规则也须固定：冷历史时省略尚不可估计的历史项，同时应用 `b_cold=2`，仍使用当时已知的利用率与队列项；这不是把未知 rho 宣称为真实零压力。全部三资源需求为 0 的任务单独令 `B_i=0`，不因 cold或无关资源得到容量风险缓冲。`K` 是非冷风险项的封顶拍数，总缓冲最大为 `15*(b_cold+K)`；不得误称总缓冲始终最多120min。

需求匹配的含义是：GPU 主导任务的 GPU 压力具有较大权重，纯 GPU 任务的 CPU 压力权重为 0。它不同于“所有任务因任意资源最大 rho 而得到同一缓冲”。但加权平均可能低估小需求资源的瓶颈，极小需求也可能因几乎没有可用容量而阻塞，故它仍是风险估计，不是可行性证明。

同拍候选必须看到已启动任务后的 `u_r/available`，并从紧迫 pending 统计删除这些任务，防止对同一资源工作重复计算。已经主动 defer 的任务仍是 pending，也仍计入下一候选的已知竞争；不能悄悄预留额外资源。资源不足的任务由公共执行器计为 capacity waiting，不把容量失败变成 Shield 成功干预。

### 2.3 可选候选与必要消融

可选方案为资源匹配的近端缺口时间，而非加权压力：

```text
T_ir = W_urg,ir/C_r                         [min]
B_i = ceil_grid( max_(r:q_ir>0) T_ir + H*k_h*rho_matched_i )
```

它更强调某一必要资源的积压，但对极小需求也取最大值，可能过度保守；队列工作除容量忽略并行装箱、非抢占和其他维耦合，不是准确排队时间。首选使用前一方案，后一方案仅作为预注册替代，避免首轮同时引入复杂优化器。

在保持所有其他参数相同的未来消融中，对比 `sum_r w_ir*p_r`、`max_(r:q_ir>0) p_r` 与旧 `max_all_r p_r`。前者检查资源匹配增益，后两者检查“瓶颈保护”与“不相关资源过度触发”的取舍。再分别去除当前利用率、紧迫工作和历史到达项；可选 known-release 项应最后加入，不能把它与 A 的基本效果混合。

已知释放扩展可用长度 `h=15,30,...,120min` 的固定 8 桶缓存，计算候选需求是否在仅当前 running 释放后可容纳，或把未来这 8 桶的平均占用代替瞬时 `u_r`。这些只是已知运行释放情景，**不得假设已释放资源必然空闲**，不得作为未知未来 deadline 的安全保证。

复杂度：EDF 一次排序 `O(n log n)`；紧迫标记、每资源 EDF 前缀和 `O(3n)`；候选缓冲 `O(3)`，同拍启动/移除用后缀更新的 Fenwick 树可保持 `O(log n)`，或小队列采用明确的 `O(n²)` 参考实现并报告成本。历史滑动累计用 deque 与三资源增量和。可选 8 桶 known-release 构造 `O(8m)`，`m` 为 running 数；不执行前向重排仿真。先做参考与优化版本的逐候选数值等价测试，再报告开销。

## 3. 模块 B：有限预算动作与任务级风险过滤

### 3.1 新动作接口提案

保持 EDF `(deadline,arrival,stable_id)` 顺序、非抢占、资源优先检查与回填。候选下一版全局动作 `Discrete(4)` 不再选择旧策略名字，而是选择**本拍主动延迟所要求保留的额外网格余量**：

| 动作 | 无风险过滤时允许主动 defer 的条件 | 风险过滤打开时条件 |
|---|---|---|
| 0 | 不主动 defer | 不主动 defer |
| 1 | 高 CI 且 `S_i>=60min` | 高 CI 且 `safe_slack_i>=60min` |
| 2 | 高 CI 且 `S_i>=30min` | 高 CI 且 `safe_slack_i>=30min` |
| 3 | 高 CI 且 `S_i>=15min` | 高 CI 且 `safe_slack_i>=15min` |

高 CI 一律为 `current_CI>161.075`，相等及低于阈值执行。每次 defer 仍只延后一拍 15min；表中的 60/30/15 是**当前动作要求的延迟前剩余余量**，不是一次让任务睡眠 60/30/15min，也不是修改 duration/deadline。所有 overdue/没有一拍 solo slack 的候选立即停止主动 defer，资源不足仍只能等待。若后续允许低 CI 延迟，则另注册实验，不能混入首版接口。

动作 0 保留无延迟参考，使 episode 决策空间包含 EDF 行为。普通新接口 RL 与带 Shield RL 使用同一动作字面值；Shield 只将不满足风险余量条件的主动 defer 改为 execute，并记录原 action、候选、S/B/safe、约束理由和是否能容纳，不改变 EDF 或伪造资源。

该接口的“safe”命名仅指在声明模型下扣除估计风险的余量，未知未来到达可能继续拥塞；不能称为形式化安全性、SLA 概率或硬保证。若某场景连 EDF 都超过 5%，Shield 不能通过禁止主动延迟自动消除全部容量性违约。

### 3.2 与旧三个模式不同的决策例子

以下为设计用微型状态，不是本轮真实任务证据。容量仍为 CPU50000/GPU1000/MEM80000；高 CI、非冷历史，GPU rho=1.6而CPU/MEM rho=0；当前GPU运行占用500、CPU/MEM为0。两个同deadline、duration=60min、`S=30min` 的候选按稳定ID排列：A为GPU-only需求250，B为CPU-only需求12500，两者都能容纳。A没有前序pending，按首选示例系数 `z_A=0.5+max(0,1.6-0.5)=1.6`，`B_A=15*ceil(1.6)=30min`；A实际启动后，B的前序GPU工作在CPU维为0，其当前CPU利用率和历史CPU压力也为0，因此 `B_B=0`。safe分别为0和30min，新动作2的过滤执行GPU任务、延迟CPU任务。

在同一状态下，旧 EDF 执行两者；旧 EDF+Carbon 延迟两者；旧 RiskOnly 的公共 2 拍缓冲要求 `S>=45min`，故执行两者。新决策子集不能用旧三个全局模式在本拍复现。两任务的资源不相交且都能容纳，差异不是回填资源失败造成的。即使不加 Shield，动作 1/2 的固定余量还可在不同 solo slack 的队列中产生旧 EDF+Carbon/RiskOnly 以外的延迟子集。

这说明动作接口具备新的表达能力，**不证明 RL 会学到该动作，不证明此子集改善 SLA/碳，也不证明方法具有学术创新性**。下一阶段必须用可执行微型测试固定本例，并在未知未来分叉下验证当前动作一致。

### 3.3 风险状态接口

保留原 40 维作为公共块，新增固定长度预算块，例如：按资源主导类别划分的 pending 数、`safe_slack` 的 min/p10/median、`safe<15/30/60` 的计数或比例、预算 B 的分位数。必须保留需求与风险/紧迫度的关联，不能只再增加三个总工作量。

空类输出 0 加 presence flag；signed slack 保留负值；分钟用现有 120min 工程尺度与有界变换，不在 Validation/Test 拟合。可定义 4 个类别 CPU/GPU/MEM/tied，资源主导按 `q_ir/C_r`，并固定 tie-break，不能把原始稳定 ID、trace 行号、raw start/end 或全窗任务数输入网络。版本化 observation schema；没有将已有 checkpoint强行放入不同维空间的兼容声称。

只有已到达声明任务、当前资源、已知运行释放和过去到达/CI是在线输入。离线 SLA 区间下界、全窗峰值、未来到达/CI、未来 drain 终点、其他策略结果均禁止进入 budget、actor、Reward 当前状态。Shield 不得有冻结未来轨迹的 Python 引用。

## 4. 分离两个模块贡献的实验矩阵

旧 RiskOnly 和三个 E1-5B PPO-v0 保留为历史强基线，新版本不覆盖其 checkpoint/schema/模型选择结果。主消融使用同一新 4 动作接口、同一网络容量、相同训练步预算/RL seeds、公共基础状态、Reward/折扣、训练负载分布和数据分割：

| 组 | 新预算状态输入 | 任务级风险过滤 | 能回答的问题 |
|---|---|---|---|
| RL-new-plain | 否 | 否，仅 solo grid 边界 | 更细动作空间本身的普通 RL 对照 |
| RL-new-state | 是 | 否，仅 solo grid 边界 | 在相同动作/执行规则下，A 的风险信息是否改善学习 |
| RL-new-shield | 否 | 是，A 的固定公式 | 在相同基础观测下，A 的执行限制是否改善结果 |
| RL-new-combined | 是 | 是，同一 A 公式 | 两者组合及交互，不能用仅一对胜负认定两模块贡献 |

可在所有组固定统一输入维数，关闭风险块时全零并保留关闭标志，使第一层参数量相同；归一化、优化器、seed和更新预算同样固定。记录屏蔽率/动作后执行分布，防止将“几乎退化为 EDF”当作学习提升。PPO-v0 使用历史 3 动作空间，属于系统基线，不是单模块严密消融。

模块 A 另有不训练 RL 的确定性对照：旧 RiskOnly、固定动作 3 的 solo Carbon、新 A +固定动作 3；再进行上述资源匹配/历史/紧迫/释放消融。其公式和阈值在全部训练与后续正式 Test 之前冻结。E1-4A 的 Shadow/Full/CapOnly 仅保留负面证据，不重新优化。

公平评价仍使用同一有限任务集合、真实 CI/weather、48h 到达停止、完整 Drain、每组最慢终点和较早排空的同模型空闲积分。低/中/高压力分开报告，不能将 DMR 混合平均以掩盖高压退化。完成量不同或达 safety cap 时先报告删失/未完成，不宣称等工作量减排。

中压力组：采用本轮参考策略实际支持的候选负载，预注册绝对 5% 门槛，记录每个 seed 的达标与失败，不事后改成 35%。高压力组：单独报告相对同输入 EDF 的额外违约、匹配任务违约集合、碳变化及 Pareto 权衡；负额外违约也不证明 EDF 全局最优或新方法有严格保证。Validation 的新增负载属于探索性诊断，不重新开放 E1-5B 的选择结果。最终 Test 保持封存，待下一阶段参数、模型选择与评价协议全部冻结后另行授权评测。

## 5. 下一阶段实施前冻结清单

本轮冻结的是研究问题、接口候选和分离贡献的检验结构，不借没有实现的公式宣称可行性。三方审核后先进入 E1-6A 的微型预算验证，注册具体系数字面值、cold/tie/超界规则、数据角色、压力档位与计算预算，再进入 E1-6B 的安全 RL。

必要行为测试包括：`d=16,D=35` 的实际完成/网格释放区别；风险余量等号边界；GPU-only 不受无关 CPU 压力触发；零需求/冷启动/过期任务；同拍启动后更新风险与容量；队列全部候选与 backfill；当前相同历史/不同未来轨迹的分叉；Shield 仅改 defer 且仍由执行器资源检查；新 schema 与旧 checkpoint不混用；关闭模块的 factorial 执行语义；独立公共终点、任务/资源守恒及 Reward首次罚分。未来性能检查必须结合原始任务 ledger，不能仅看 Shield 触发次数或 Return。

当前 Reward 与折扣的长程限制见 `reward-horizon-audit.md`；本轮不改权重、gamma、episode、网络或原算法，不重新训练 PPO。

## 6. 已实读源码与位置

- `experiments/e1/engine.py`：`start/advance/_execute/result/extend_idle`，行 96–279；任务所有权与资源池行 13–94。
- `experiments/e1/policies.py`：`latest_start/EDFBackfill/CarbonThresholdFIFO.decide`。
- `experiments/e1/cars.py`：RiskOnly 规则与候选通知，行 32–71。
- `experiments/e1/cars_state.py`：网格释放行 27–29；历史窗口与需求占用工作、统一 rho，行 64–108。
- `experiments/e1/workload.py`：`TaskSpec.sla_deadline/make_task/load_workload`，原始 12 字段、需求倍率与合成 deadline。
- `envs/sustaindc/sustaindc_env.py`：`release_resources/try_to_schedule_task`，实际 finish 与 SLA 比较、原分配路径。
- `experiments/e1/power.py`：持久物理适配对象、每拍先注入工作负载再 `step`；物理公式未改。
- `experiments/e1/metrics.py`：`interval_record/summarize/summarize_policy`。
- `experiments/e1/rl_env.py`：40 维状态、动作注册、Reward首次罚分与终止/截断。
- `experiments/e1/rl_train.py`：完整/尾段训练账本、真实 PPO 调用与真实截断拒绝。
- `experiments/e1/rl_evaluate.py`：Train/Validation角色约束、完整 Drain 后公共空闲核算。
- `experiments/e1/rl_training_utils.py`：双角色冻结、不可覆盖、原始身份与物理配置验证。
- `experiments/e1/stress.py`：嵌套 deterministic 同拍复制；2×复制所有任务，seed不产生独立到达形态。
- `experiments/e1/configs/e1_5b_ppo.json`，E1-5A 的 RL接口/数据划分报告与 E1-5B 的训练/验证报告。

以上读操作没有修改这些源码、已训练模型、E1-5B报告或 provenance；本文只新增于 E1-5C。
