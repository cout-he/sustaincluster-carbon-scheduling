# E1-6B B0：冻结风险预算诊断与接口验收

本报告只解释 E1-6A v1 的已保存候选状态，未修改公式、参数、策略或历史归档。Train 是主要机制诊断来源；开发 Validation 仅描述已有结果，不用于权重、阈值或模型超参数选择。未访问 Test，未进行重放、训练或参数搜索。

## 1. 冻结公式与代码审计

审计文件：`experiments/e1/task_risk_state.py`、`task_risk.py`、`cars_state.py`。原版本 `e1-6a-task-risk-budget-v1` 保持冻结。

`m_r=q_r/C_r`，`w_r=m_r/sum(m)`；`P_r=u_r+Q_r+max(0,rho_r−0.5)`（历史未知时最后一项为零）。`z=sum(w_r*P_r)`；max_required 仅在正需求维度取 `max(P_r)`。`B=15*ceil(cold+min(8,z))`，cold 仅在有效历史分钟=0时为2，否则为0；全零需求规定 z=B=0。`solo_slack=floor_grid(deadline−raw_duration)−now`，`safe_slack=solo_slack−B`。历史窗口120分钟，阈值161.075 gCO₂e/kWh，SLA multiplier=1.5。

- 当前利用率来自实际运行集合，包含本拍较早已启动任务；真实启动 hook 立即更新运行集合，并移除相应 pending。
- 紧迫工作量只扫描初始 EDF 排名中当前候选之前、仍 pending、最迟网格启动距当前≤120分钟的任务，包含 overdue；已启动任务剔除，主动 defer 与容量未能启动任务保留。
- 固定任务工作量是 `q*15*ceil(raw_duration/15)`；latest start 则用原始 duration。两个语义分别对应实际资源释放网格和真实完成 deadline。
- ArrivalHistory 仅观察已到达任务，窗口 `(now−120,now]`，task ID 去重，分母为 min(120, elapsed)。没有未来 CI、未来任务、全窗口 SLA 下界或其他策略输入。
- cached_work 只缓存任务静态 grid/work 事实，逐候选仍按相同成员与顺序进行 `math.fsum`；与参考查询语义一致。该诊断不改缓存或性能实现。
- no_history 仍仅 a_h=0，保留共同 cold=2。它移除数值历史压力项，未完全移除历史冷启动保护。

## 2. 分母和输入保护

读取六组既有冻结数据中的五种任务风险策略，共30个策略轨迹；首轮66个输入文件（每轨迹 diagnostics 和 ledger、每组 parameters）前后 SHA256 全部一致。独立算术复核另读取每组 fairness 中的实际容量，共72个文件前后 SHA256 一致；**257,719 个 eligible 预算**的需求匹配权重、资源压力项、连续 z、cold、B、safe slack 和实际 defer 判定与冻结公式逐项精确一致，无浮点容差放宽。脚本显式枚举 Train/Validation，未枚举 Test。

**eligible 只指 can-fit、CI高于阈值、solo slack≥15 的已调用候选**。低 CI、无一拍 slack 的诊断 budget=null；容量失败不会调用策略预算。它们不应补成 z=0。事件分母会重复计数仍等待的同一任务；首次 eligible task 分母每任务只计一次。全量覆盖和缺失预算分母见 `coverage.csv`。所有五种策略的完整统计均保留，以下表主要展示 Full。

## 3. 连续风险分布

首次 eligible task 的分位数：

| 场景 | 首次 eligible 任务 | z中位数 | z p95 | z最大值 | (0,1] | (1,2] | (2,3] |
| --- | --- | --- | --- | --- | --- | --- | --- |
| train-1 | 1414 | 0.1824 | 0.6722 | 0.8472 | 100.000% | 0.000% | 0.000% |
| train-1.5 | 1994 | 0.2713 | 1.0879 | 1.3372 | 92.427% | 7.573% | 0.000% |
| train-2 | 2527 | 0.3861 | 1.3229 | 1.9203 | 84.052% | 15.948% | 0.000% |
| validation-1 | 3187 | 0.2328 | 0.4549 | 1.0079 | 99.969% | 0.031% | 0.000% |
| validation-1.5 | 4493 | 0.4400 | 0.8550 | 1.8025 | 97.708% | 2.292% | 0.000% |
| validation-2 | 4704 | 0.8020 | 1.4840 | 2.3366 | 75.553% | 24.384% | 0.064% |

Train 三档中没有 z>2 的首次 eligible task。轻载不是“所有任务风险相同”：Train1×的1414个任务有1414个不同连续 z，但缓冲均为15分钟。

量化值种类与 ceil 相对连续预算增加的分钟数：

| 场景 | 不同z值 | 不同B值 | ceil平均新增分钟 | cold>0首次事件 |
| --- | --- | --- | --- | --- |
| train-1 | 1414 | 1 | 10.924 | 0 |
| train-1.5 | 1940 | 2 | 10.052 | 0 |
| train-2 | 2445 | 2 | 9.475 | 0 |
| validation-1 | 3186 | 3 | 11.322 | 33 |
| validation-1.5 | 4394 | 3 | 8.543 | 47 |
| validation-2 | 4571 | 3 | 6.625 | 66 |

`distinct_value_compression=1−不同B数/不同z数` 只是**数值类别合并比例**，不是信息论熵、预测误差比例或 SLA 损失比例。完整事件与首次任务两套分母分别保存。Validation1×/1.5×的第三种B可能来自 cold保护，不能误读成 z>2。

## 4. 量化和风险过滤对动作的作用不同

令 solo slack=15*n，档位门槛=15*g，连续步缓冲 `x=cold+min(8,z)`。因为 n−g 为整数，`n−ceil(x)≥g` 与 `n−x≥g` 严格等价。因此在当前15分钟网格和15/30/60分钟门槛上，**ceil前后当前Execute/Defer判断相同**。量化会丢失 Observation 的连续区分信息，但不能仅据此认为 ceil 导致了不同动作或 Full 负收益。

扫描所有五种策略、六场景、三个门槛，连续缓冲与 ceil 缓冲动作差为0。4项独立微型验收（边界、网格等价、小需求瓶颈、全零需求）均通过。

另一方面，相对无风险过滤的 solo slack 规则，B 会真实改变允许延迟集合。Full 保存状态在15分钟门槛上的对照：

| 场景 | eligible事件 | 风险过滤额外Execute | 同状态weighted/max动作差 | 连续/ceil动作差 |
| --- | --- | --- | --- | --- |
| train-1 | 4829 | 1360 | 8 | 0 |
| train-1.5 | 6583 | 1894 | 145 | 0 |
| train-2 | 7911 | 2361 | 94 | 0 |
| validation-1 | 9082 | 2996 | 2 | 0 |
| validation-1.5 | 12302 | 4170 | 216 | 0 |
| validation-2 | 10632 | 4248 | 1023 | 0 |

这是**同一保存状态**上代数判定的差异，不是两条真实轨迹最终 DMR 的因果差异。30/60分钟档位的对应数值保存于 `same-state-action-checks.csv`。新环境必须逐候选重新检查当前 running 与剩余 pending；不得把拍初预算当成全拍不可变预算。

## 5. weighted 与 max_required 的小需求瓶颈反例

合成接口反例（只测机制，不作为科研胜利样本）：候选需求占容量 CPU0.8、GPU0.001、MEM0；当前CPU运行占比0.2、GPU0.9，候选仍 can-fit；CPU总压力0.2、GPU总压力5（含紧迫GPU工作量）。GPU匹配权重约0.125%，weighted z≈0.206、B15；max_required z5、B75。solo60时前者允许延迟、后者禁止。这说明需求大小加权可能掩盖候选仍依赖的小量紧缺资源；但 max 也可能过度保守，二者都不能保证改善真实 SLA。

真实保存状态反例采用预定义选择规则：每场景 Full 轨迹中，紧缺维度权重≤5%、两聚合当前动作不同，按 time、stable_id 排序取最早三项。以下展示各场景第一项：

| 场景/已知时刻 | 紧缺维度 | 需求匹配权重 | weighted z/B | max z/B | solo slack | 同状态动作 |
| --- | --- | --- | --- | --- | --- | --- |
| train-1.5 / 2023-05-03T03:45:00+00:00 | cpu | 2.915% | 0.3209 / 15 | 1.2423 / 30 | 30.0 | weighted defer；max execute |
| train-2 / 2023-05-02T10:30:00+00:00 | cpu | 4.779% | 0.3810 / 15 | 1.0530 / 30 | 30.0 | weighted defer；max execute |
| validation-2 / 2023-07-08T06:15:00+00:00 | gpu | 3.319% | 0.2193 / 15 | 1.1517 / 30 | 30.0 | weighted defer；max execute |

Train 反例紧缺维度是 CPU，不能写成真实 Train GPU反例；开发 Validation2×描述中存在 GPU小需求反例。此表只改变同状态聚合公式，没有让max策略沿Full状态执行完整未来轨迹；记录的 Full 最终结果仅作关联。完整ID、真实Full启动/完成与最终deadline结果保留于 `small-demand-bottleneck-examples.csv`。

## 6. Buffer、slack、资源类别与最终结果

按**首次 eligible task**的预算分层，再关联该策略自身最终任务账本：

| 场景 | 首次B分钟 | 任务数 | 最终miss | 该层DMR | 首次决定Defer数 |
| --- | --- | --- | --- | --- | --- |
| train-1 | 15 | 1414 | 0 | 0.000% | 1048 |
| train-1.5 | 15 | 1843 | 7 | 0.380% | 1294 |
| train-1.5 | 30 | 151 | 0 | 0.000% | 64 |
| train-2 | 15 | 2124 | 46 | 2.166% | 1556 |
| train-2 | 30 | 403 | 0 | 0.000% | 106 |
| validation-1 | 15 | 3153 | 7 | 0.222% | 2049 |
| validation-1 | 30 | 1 | 0 | 0.000% | 0 |
| validation-1 | 45 | 33 | 0 | 0.000% | 2 |
| validation-1.5 | 15 | 4343 | 69 | 1.589% | 2804 |
| validation-1.5 | 30 | 103 | 8 | 7.767% | 16 |
| validation-1.5 | 45 | 47 | 0 | 0.000% | 3 |
| validation-2 | 15 | 3488 | 99 | 2.838% | 1995 |
| validation-2 | 30 | 1147 | 87 | 7.585% | 500 |
| validation-2 | 45 | 69 | 3 | 4.348% | 7 |

资源类别按容量归一化需求最大的维度判定（tie/zero独立类别），不按原始CPU/GPU/MEM数值大小比较。各类别 z均值/范围、solo slack中位/范围、首次动作与最终DMR见 `first-budget-outcomes.csv`。这属于条件描述：预算不同的任务原本也可能具有不同duration/slack/需求；不能把B与违约关联直接解释为B的因果效果。相同任务在不同策略下的预算也基于不同后续状态，不能简单相减。

## 7. B0验收结论与B1边界

1. 冻结预算确实控制可延迟集合，且大多数Train连续差异被压到一至两种B。新增RL风险状态应保留连续 z、各资源压力和presence，不能只有B。
2. weighted与max在相同状态上确有动作差异，但现有证据不支持更换公式或选择Validation有利系数。E1-6A公式及全部负面结果保留。
3. 原v1可作为**待验证的启发式风险过滤**。它没有资源可行性充分证明、没有未知未来保证、没有稳定超过RiskOnly的证据；不得称为已证明安全的Shield。
4. 四动作语义是“还要主动等一拍之前当前剩余solo/safe slack必须≥60/30/15分钟”，不是一次延迟相应时长。CI==161.075应执行，Action0不主动defer。容量仍由公共执行器独立检查，EDF/非抢占/物理模型不变。
5. Observation可以在动作之前汇总当前所有已知 pending 的连续风险特征，但必须说明其与E1-6A saved eligible分母不同；候选过滤仍在真正 can-fit 时按当前运行状态重算，不能读取其他策略或未来标签。关闭风险状态块按注册规则置零；四组维数和参数量相同。
6. B0通过的是公式、接口和可复现诊断验收，并非方法收益验收。允许按预注册四组×三seed开展B1/B2；结果优劣必须保留，不据开发Validation临时更换公式或追加预算。

## 8. 复现和文件

分析命令（Windows，标准库）：

```powershell
& '<LOCAL_PYTHON>' -X utf8 -B '<LOCAL_RESEARCH_ROOT>/research_notes/E1/E1-6B/b0_risk_diagnostics.py' --campaign '<LOCAL_RESEARCH_ROOT>/research_notes/E1/E1-6A/campaign-001' --output '<全新输出目录>'
& '<LOCAL_PYTHON>' -X utf8 -B '<LOCAL_RESEARCH_ROOT>/research_notes/E1/E1-6B/b0_risk_diagnostics_selftest.py'
```

`b0-diagnostics-001/manifest.json`记录66个输入哈希、7个CSV哈希、脚本哈希及inputs_unchanged=true；`B0-budget-consistency-001.json`记录独立72文件算术复核和257,719个预算的精确通过证据，日志为 `B0-budget-consistency-001.log`。输出目录存在时拒绝覆写。报告由新文件 `b0_make_report.py`从CSV生成并追加精确复核收据；未改写任何 E1-6A provenance/hash/report。审计源代码仍归属 E1-6A 原版本。本阶段未给策略输入离线诊断统计。
