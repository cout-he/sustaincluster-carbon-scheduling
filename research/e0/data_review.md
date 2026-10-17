# E0 数据与产物独立复核

复核日期：2026-10-17（Asia/Shanghai）。复核读取已完成的真实 `artifacts/baseline/` 产物和当前 E0 runner，没有重新运行完整仿真、修改输入、改写任务、修正物理模型或训练算法。

执行命令：

```powershell
.venv/Scripts/python.exe research/e0/review_artifacts.py
```

首次独立复核退出码 0、工具 wall time 0.598799 秒。随后在 E0 的 Python 3.11 解释器重算，退出码 0、工具 wall time 0.841594 秒；`artifact_review.json` 保存最后一次真实结果。这一独立脚本不导入调度器，仅根据 CSV/JSON 重新计算和比较。

## 已核实的完整性

- 共 8 次成功运行，每次 96 步，合计 768 个 step 记录、3,840 个站点记录；各运行步号连续且唯一。
- 每步五站的能耗、排放、成本、已占用 CPU/GPU/MEM、开工/完成/SLA/运行/等待计数重新求和后，都与 `steps.csv` 一致；首次 Python 3.14 复算的最大绝对误差为 `4.547473508864641e-13`，E0 Python 3.11 最终复算为 `0.0`，均在检查容差内。逐步值再次求和与 `summary.json` 一致。
- 每个站点每步均满足 `site_emissions_kg = site_energy_kwh × battery_ci_g_per_kwh / 1000`，成本满足 `site_energy_cost_usd = site_energy_kwh × price_usd_per_mwh / 1000`。这里使用电池本步实际采用的 CI，没有把推进后的管理器 CI 混入同一步排放验证。
- 每一步的累计任务计数均满足 `incoming = started + pending + in_transit + deferred` 与 `started = finished + running`。这比仅检查期末更严格；其边界仍是数量守恒，未声称做了逐任务生命周期身份的形式证明。
- 8 个运行在每个步的 cohort SHA-256 完全相同，且从逐步哈希重新计算的序列 SHA-256 均等于 summary 中记录的值。runner 的 cohort 签名包含原始 job 名、到达时刻、duration、三资源、bandwidth、SLA 和随机来源 DC；仅移除 `Task` 创建时追加的随机 ID 后缀，未改变任何仿真对象。
- `ObservedCluster` 对原始 `get_tasks_for_timestep` 和 `step` 调用原实现后记录结果；没有重放替换工作负载。规则模式的 wrapper 不额外提取任务，规则任务由原始 `cluster.step` 读取；manual 模式由原始 wrapper 读取。
- `local_only`、`local_only_repeat`、`local_only_direct` 的逐步记录完全相同；这证明本轮固定 seed 和顺序下，经 wrapper 调用与直接调用原始规则内核有相同的测量结果。它不证明 manual 路由与规则路由具有相同服务时序。
- 全部 379 个已恢复输入在先前 `inspect_data.py` 中重新校验 SHA-256 相符；它们来自同一官方 Commit，PKL 仅由对应 ZIP 解压。实际 `execution.json` 的 PKL 哈希与数据 manifest 相符。
- `git diff --name-only` 显示原有文件的唯一改动是数据获取示例移除硬编码凭据；调度器、环境、任务提取、物理模型、奖励与原始配置没有修改。新增文件位于 E0 目录。
- 当前 Git 候选文件中没有超过 5 MiB 的文件，没有 `data/` 下 CSV/JSON/ZIP/PKL 原始输入候选。常见 token 格式筛查命中 0；扫描仅输出路径和行号，不输出候选秘密内容。此检查不覆盖任意秘密类型或全部历史。

## 必须保留在结论中的科研限制

1. **任务时钟和物理时钟没有对齐。** 任务起点为 `2023-08-01 05:00 UTC`；五站初始本地时间减去固定时区偏移后，一致指向 `2023-08-27 18:00 UTC`，相差 **26 天 13 小时**。同 seed 下的共同比较可复现，但不能把结果称为“2023-08-01 当日实况”。E0 忠实暴露这个官方初始化问题；下一阶段先建立并验证统一时标。
2. **RBC 与 manual 接口存在原始服务差异。** manual 本地动作也要先进入 in-transit、下一 tick 才交付；RBC 当步直接投队。本轮 manual_local_replay 的 2,506 到达中，2,496 已开工、2,342 完成、154 仍运行、10 在途；规则 local_only 为 2,506 开工、2,368 完成、138 仍运行。manual 总能耗较小不能解释为算法节能。
3. **原始 reward 不是物理排放和成本。** RBC wrapper 的 current_tasks 为空，原始 task-proxy reward 因而为 0；manual 有非零代理 reward。不要据此排名或将 reward 视为 USD/kgCO2。真实比较应明确使用分列的站点与网络统计。
4. **负载与 CI 有原始一步滞后。** `SustainDC.step` 先计算当步物理输出，再推进管理器并更新当前任务负载。因此 CSV 的当步资源快照与能耗不是同一负载时刻，推进后的 manager CI 也不同于 `bat_avg_CI`。本轮单位检查证明算术和口径一致，不证明物理时序已正确。
5. **窗口不包含排空阶段，且只测试一个 seed、一个短窗口。** 末尾未完成任务的后续能耗和 SLA 没有计入。本轮通过不能证明长期高负载、容量耗尽、季节变化或跨 seed 稳定性。
6. **原始工作负载并非名义上的完整 2020 年。** 文件包含 2021 年 1 月尾部、80 个缺失的 15 分钟槽；E0 没有补齐。应对 `current_time.replace(year=2020)` 的精确匹配语义及缺槽时行为单独建立后续实验。
7. **凭据历史问题需仓库所有者另行处理。** 原始上游数据获取示例中的硬编码凭据已经从当前文件移除，本轮未调用该联网示例。既有 Git 历史中的已公开值不能通过一次新提交撤销；本轮不改写 main 或历史，凭据有效性和外部撤销不属于已验证事项。

没有发现应阻止保存 E0 真实产物的新完整性错误。上述限制足以阻止直接将本轮数字作为新 RL 算法收益证据；它们应进入下一阶段的优先风险清单，而不是在 E0 中悄悄修正。
