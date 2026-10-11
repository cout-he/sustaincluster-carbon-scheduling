# E1-4A：只读诊断方法与旧档案结果

新模块 `experiments/e1/diagnose_feasibility.py` 只读旧结果，不重新执行真实策略。它将 EDF shadow 的增量 miss 集合配对到实际 Full 终态，并检查同拍后续真实动作是否遵循 shadow NOW 的安排。旧 E1-3 源码、参数与归档均未由该模块写入。

## 预声明与边界

代表选择规则保存在 `diagnostic-selection.json`，在新真实消融之前固定：每档按 `(UTC time, task_id)` 选择首个 known risk、首个 pending-limit fallback、首个历史 risk-buffer guard；另外保留已知历史 `2023-07-01 10:15 UTC /17475/189-copy1`，及关联任务 `/17470/35`。无对应决策时保留空槽。大于 32 项的代表数组显式保存前 32 项和总数、截断标记；历史 23/24 集合保留完整。所有明确风险决策均进入统计，不只统计代表。

旧 Full 的三个互斥的实际 reason 计数分别是 A：`risk_buffer_guard`；B：`known_deadline_risk`；C：`known_feasibility_uncertain` 且 `planner.reason=pending_limit`。A 是触发历史 buffer guard 的决策数，不等于历史压力计算次数。所有 candidate checks、其他 reason、其他 uncertain 原因同时保留。

新消融不能仅按 `planner.reason=pending_limit` 计保守 Execute：CapOnly 的 `cap_execute_guard` 没有 planner 记录，但确实执行；ShadowOnly 的 `shadow_uncertain_risk_fallback` 可以报告 pending-limit uncertain，却保留 RiskOnly 的 Defer。`cap_execute_guards` 和兼容字段 `pending_cap_fallbacks` 只计前者或旧 Full 的 cap uncertain Execute；`other_uncertain_fallbacks` 只计旧式非 cap uncertain Execute；`uncertain_risk_fallbacks` 独立计 ShadowOnly 的 uncertain Defer。全部 uncertain 状态量另记 `planner_uncertain_status_count`，不会从 Execute 数中相减。

`planner_reported_status_decisions` 与状态占比包含 synthetic queue-limit bypass 的 uncertain 报告；`planner_actual_execution_decisions` 与 `planner_actual_*_proportion` 排除 bypass，优先使用新诊断的 `shadow_planner_executed`，旧 Full 使用 `planner_calls>0`。实际 scheduling passes 另列。全候选超限次数以 `pending_limit_exceeded` 或 `pending_count>256` 计，eligible 超限次数必须同时通过原 RiskOnly Defer rule。比例分别使用全 `policy.decide()` 检查数、RiskOnly 可 Defer 检查数、全部状态报告数、真实 planner 执行数；executor 因容量不足而未进入 decide 的 turn 不属于 policy candidate checks。各分母说明随 campaign JSON 输出。

对每个 `planner.status=risk`，以集合差 `M_defer - M_now` 生成一行 `(decision time, candidate_id, affected_id)`。重复被警告的同一任务按出现次数保留，另外报告唯一任务数；终态 `missed`/`missed_unfinished` 才计实际 miss。缺失 ledger ID 报错，不静默视作 met。此处是反事实集合与实际结果的描述性关联，没有概率输出、没有概率校准或误报率的含义。

## 状态与同拍检查

每个明确风险决策从 `policy_intervals.ordered_waiting_ids`、`policy_events.decision_rank` 和 Full ledger 重建候选时刻的 pending/running 状态。之前真实 started 从 pending 删除并加入 running；之前真实 deferred 在两个 shadow 分支都 blocked 到下一拍；容量等待仍保留 pending。运行中资源释放使用 `ceil(finish_time/15min)`，miss 仍由精确 finish 与 deadline 决定。重建 pending/running 数与已记录 diagnostic 数必须一致。

对所有 137 个明确风险决策重新调用旧 `ShadowPlanner.compare`，比较 status、reason 和四个完整 miss/unknown 集合。全部一致，证明这里的分析使用了原候选已知状态，而非事后把未来任务放入 shadow。

同拍检查单独重现 shadow NOW 的第一拍 starts：先强制候选 NOW，然后按已知集 EDF、资源限制和先前 blocks 启动其余任务。实际比较只使用候选及其后续 rank 的真实事件。输出 shadow-start/real-defer、shadow-start/real-capacity-wait、real-start/shadow-wait 的集合或计数，及启动集合是否不同。所有当前边界到达任务属于当前已知输入，单独报告当前边界全部到达数和候选快照中的到达数。

later arrivals 仅用于终态解释，定义为 `decision time < arrival_time <= affected task actual Full start_time`。报告数量与基于 `ceil(duration/15)*15` 的 CPU/GPU/MEM resource-work minutes；这些后来任务没有被送入重建的 shadow 计划。后来到达存在不是它单独造成 miss 的证据。

## 旧 E1-3 results-001 结果

| load | A history guard | B explicit shadow risk | C pending-cap fallback | incremental warnings | unique warned | warnings paired to actual miss | unique actual miss | same-step start-set mismatch |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.50 | 3,532 | 119 | 501 | 264 | 15 | 142 | 6 | 111/119 |
| 2.00 | 5,915 | 18 | 579 | 77 | 42 | 0 | 0 | 17/18 |

两档其他 uncertain 原因均为 0。1.50 的 142 个实际 miss 配对出现都存在 later arrivals，而且 6 个实际 miss 任务在 RiskOnly 中都 met。因此不能把 B 全称为误报，也不能把这些警告全视作已成功保护的任务。2.00 的 77 个增量警告对象实际全部 met，但这同样不是概率模型的误报校准。

历史 1.50、10:15 UTC 的 `/17475/189-copy1` 重现 NOW 23 / DEFER 24。候选自身两策略都 met。shadow NOW 和真实候选及后续动作都启动 14 个任务，但集合不同：其中两个 shadow 预定当拍启动的任务被真实后续碳 guard deferred，未出现 shadow-start/real-capacity-wait。这个差异发生于同一个边界，不能归因于尚未出现的 later arrivals。

关联 `/17470/35` 在全部明确风险决策中属于增量风险对象 51 次，最终 Full 于 11:15 UTC 启动、14:55:03 完成，超过 14:30:04.5 deadline；RiskOnly 于 10:30 启动、14:10:03 完成并 met。历史单个 23/24 对比不能替代整段真实策略，也不能单独给出该 45 分钟差异的因果效应；新 C/B 消融和固定候选一次性干预用于进一步区分机制。

## 输出与重现

旧分析 API：`analyze_reference(reference_root, output)`。CLI：

```bash
GIT_OPTIONAL_LOCKS=0 PYTHONDONTWRITEBYTECODE=1 /home/hwj/miniconda3/envs/sustaincluster/bin/python \
  -m experiments.e1.diagnose_feasibility \
  --reference-root research_notes/E1/E1-3/results-001 \
  --output research_notes/E1/E1-4A/reference-audit
```

新消融分析 API：`analyze_campaign(campaign, output)`；CLI 改用 `--campaign PATH`。读取两档 `comparison.csv` 的 5 个 `phase=common` 行、`fairness.json`、5 策略最终账本和诊断。输出 common 全部指标、guard counts/planner status proportions、对 RiskOnly 的 paired miss 集合计数、每任务 start 差异汇总和每类最多三项按 `(arrival_time, task_id)` 固定排序代表，不输出完整任务账本。所有写入要求路径包含 `E1-4A`，并禁止位于只读参考/消融根目录内部。

13 个新微型测试覆盖集合差、重复警告与 unique 计数、缺失终态检查、A/B/C 分类、预声明排序/空槽、先前 blocks 与同拍 carbon defer、只读输入保护和 campaign paired outcome/start time 汇总，新增三策略 cap Execute/uncertain Defer 区分、实际 planner 分母、全部/eligible 超限次数回归。最后 `green-guards-final.log` 为 13 passed、0.83s。RED/GREEN 及执行日志全部保存在 `diagnostic-tests`。按本轮明确范围该子任务未运行全 pytest 或新的真实策略实验。
