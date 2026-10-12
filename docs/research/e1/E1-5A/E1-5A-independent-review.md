# E1-5A 第一版执行器与 RL 环境独立只读审查

日期：2026-10-12。审查范围：第一版 `experiments/e1/engine.py`、`rl_env.py`，以及 `tests/e1/test_stepwise.py`、`test_rl_env.py`。本审查只读源码与测试，未执行测试、实验或训练，未修改 production/test 源文件。下列结论是代码路径审查，不是性能、正式实验或统计优势证据。位置引用指第一版读取时的行号；修复后应另记验证状态，不覆盖本次发现。

## 确定发现

1. **[P2] 训练 cutoff 可被误标为整个 cohort 已排空。** [engine.py:258](../../../../experiments/e1/engine.py#L258) 的末端 `drained_at` 判定只检查当前 pending/running 全空，没有检查 `observed_end >= workload.end`。例如 a 在 t0 到达并于 t+15 完成，b 在 t+30 到达，完整到达窗止于 t+60，训练 `until=t+15`：exclusive cutoff 正确排除 b，但 finalize 将 `drained_at=t+15`。RL 的 terminated 条件仍正确为 false，然而 final_summary 错误标排空；`extend_idle()` 也因此接受这个未收完 cohort 的 executor，并可只积分 idle 而跳过 b。末端排空判定必须保留“到达窗已结束”条件。当前 `test_early_until_is_exclusive...` 使用未完成 running 任务，不能触发这个空 prefix 情形。

2. **[P2] 可接受的 history 配置与固定观察边界不一致。** [rl_env.py:80](../../../../experiments/e1/rl_env.py#L80) 把 parameters 直接交给 CARS；`ArrivalHistory` 接受任意正 history_minutes。观察向量在 [rl_env.py:172](../../../../experiments/e1/rl_env.py#L172) 固定使用 `effective_minutes/120`。`history_minutes=240` 在 elapsed=135min 时产生 1.125，必然超出 Box[-1,1] 并报 Invalid online observation；`history_minutes=60` 则静默改变本轮固定 120min RiskOnly 语义。对本轮固定契约，建议明确拒绝非 120 参数；如果另行支持通用窗口，必须同时声明新算法身份、统一使用真实 history.window 归一化，不能只改观察 divisor。

3. **[P2] 配置验证接受 numeric string，却把原 string 留给算术。** [ObservationNormalization.__post_init__](../../../../experiments/e1/rl_env.py#L31) 和 [RewardConfig.__post_init__](../../../../experiments/e1/rl_env.py#L46) 调用 `finite()`，未保存其 float 返回值。`ObservationNormalization(ci_g_per_kwh='1000')` 被接受，随后 `_bounded` 的 scale+abs(value) 报 TypeError；`RewardConfig(carbon_scale_kg='1')` 被接受，随后 reward 除法报 TypeError。应该明确拒绝不支持的原类型，或在 frozen dataclass 中用 `object.__setattr__` 保存 validated float；不能接受配置后在 rollout 中才失败。

4. **公开 result 的零拍状态缺少契约。** `start()` 新公开返回当前准备边界，`result()` 又支持只含已到达 IDs 的 partial rows；但立即 `start(); result()` 时 `policy_steps`/`steps` 尚空，`summarize_policy()` 对 len=0 求 mean 产生 ZeroDivisionError。若 result 只允许已执行至少一拍或 finalize 后读取，应有明确 RuntimeError/文档；若它是一般 observed-prefix getter，应支持零拍。现有 adapter reset 不调用 result，故这不是当前 reset/step 流程的阻断，但应明确公共 API 边界。

5. **[P2] close/reset 生命周期可重复关闭同一模型。** 第一版 `close()` 每次都调用当前 executor.power_model 的 closer，`reset()` 又先调用 close。`reset→close→close` 或 `reset→close→reset` 可重复关闭旧模型，非幂等 closer 会阻止 reset 重启。需环境跟踪模型关闭状态并只关闭一次；保持 executor 供结果读取是可行的。该问题也与 [Gymnasium close 文档](https://gymnasium.farama.org/api/env/#gymnasium.Env.close) 的重复关闭约定不一致。现有 StatefulPower 桩没有 closer，无法覆盖此行为。

## 已核对的主要正确路径

`run()` 与 `start()` / `advance()` 共用 `_execute()` 生成器。yield 位于当前 arrival→release→enqueue 后、order/decide/start/power 前；advance 恰好提交一拍，再准备下一边界。exclusive until 检查在该边界 arrivals 之前，末端 flush 不添加 scheduling/power 拍。真正资源分配、capacity-first 等待归因、one-pass backfill、SLA 精确完成判断和功耗核算路径均仍是原执行器代码。

模式切换只替换尚未执行的本拍 eligibility policy，pool、Task 与 power 实例跨拍持久；已 running 任务没有抢占路径。RiskOnly history 在 reset 和每次 step 的 `_observe_boundary()` 无条件观察当前已到达 arrivals，包括 EDF/EDFCarbon 拍；Action2 自己的 `on_step_start()` 重复观察相同 ID 会被 history.seen 去重，未见 arrivals 重复计数或遗漏。CI history 只在非 terminal 当前 sample 上追加；terminal observation 使用最后已观察 CI，没有索引截止之后的 CI、weather 或 arrival。RL info 的 arrived/pending/running/completed 数来自当前 ledger。

奖励 carbon 使用已执行 interval 的实际 kgCO2e，pending wait 使用该拍 held queue 的 task-minutes。SLA miss 在 completion/release 或当前 deadline 已到期的未完成状态中判定；episode 的 `_penalized` 集合阻止同 ID 再次扣罚。最终 flush 在 reward 之前，精确 deadline 已完成但可观察 release 较晚的任务按真实 finish 判 met。truncation liability 同时计入已到达 unfinished 数和其剩余资源工作量，right-censored 任务也计入；尚未到达 future 任务不进入这个义务。

Gymnasium reset/step 返回形状与 dtype 对应声明的 Discrete(3)/Box，step 拒绝 bool 与非法 action，终态后要求 reset。reset 为每个 episode 创建 fresh policy/executor/power，step 使用同一个 physical model。上述是代码路径检查；本审查没有自行执行 env_checker，也未复述根任务报告的测试数为独立证据。

## 建议补充的有效边界检查

- until 早于 arrival_end、已到达 prefix 全部完成且存在 future arrival：确认 truncated、drained_at=None、未来任务未登记，并拒绝 extend_idle。
- 固定 history 参数的非 120 输入；normalization/reward numeric string 或非数字输入，pin 明确支持/拒绝契约。
- History 跨 120min 左端等号过期时，期间含 EDF/EDFCarbon、空 pending、全部 capacity-wait 拍，再选择 RiskOnly，确认 records/rho/buffer 与持续 observed arrivals 一致。
- truncated 后已罚 miss 的任务继续到真实 late completion（另一次无 cutoff rollout）时不重复；终态 exact finish=deadline 与未完成 deadline=cutoff 两种边界分别保留。
- zero-interval `result()` 的明确 API 行为；异常 policy/power 后禁止继续复用已失败 iterator 或至少给出稳定失败状态。

测试建议针对语义边界，不要求扩大正式 RL 训练或性能实验。现有 six-policy golden / constant-action checks 是合理回归设计，但只有实际运行日志才能声明通过。

## 根任务修复后的只读复核

根任务已经对前三项边界与零拍 result 增加 RED/GREEN 验证，保留日志。本审查随后读取最新源码，确认：末端 drained_at 加回 `observed_end >= workload.end`；零拍 result 明确抛 RuntimeError；RewardConfig/ObservationNormalization 保存 validated finite float；RL 环境明确拒绝非 120min history。读取 `boundary-green-001.txt` (private local archival evidence) 的实际末行是 **57 passed in 2.85s**。本审查没有自行运行这些测试。

最新代码另加入 `_model_closed` 状态，`close()` 仅在尚未关闭时调用 closer，reset 新模型后把状态恢复为 false。此路径消除了成功 closer 的重复调用；还应读取随后非幂等 closer 的回归证据。late-completion 不重复扣 miss 的新测试也已加入源码。

额外 mixed-mode diagnostics 检查：RiskOnly history 持续正确，但 EDFCarbon 的独立 history 只在 Action1 被选择时观察，因此它的 diagnostics 可显示不完整的 past-arrivals pressure，即使该压力未参与 EDFCarbon eligibility。建议 reset 时将 EDFCarbon.history 指向同一持续 observed RiskOnly.history；两个策略的 start/capacity/固定窗口相同，history 自身按 ID 去重，constant-mode 行为应保持，而 mixed diagnostics 才对应同一已观察历史。这个建议不要求改变 CARS 原策略源码或 eligibility 公式。

根任务随后采用等价的持续观测方案：`_observe_boundary()` 遍历所有具有 history 的持久 policy 实例，给每个实例送入同一当前已到达 arrivals；每个 active `on_step_start()` 再观察时仍由 seen 去重。两个实例的状态独立，记录的 arrival 历史一致；没有改变 eligibility 公式。本审查再次读取最新实现，确认这条路径解决了 EDFCarbon mixed diagnostics 的遗漏。

根任务补充 mixed-mode history 与非幂等 closer 回归，并保留 `lifecycle-red-001.txt` (private local archival evidence) 的两项实际失败；随后 `rl-green-003.txt` (private local archival evidence) 实际记录 **34 passed in 1.70s**。读取当前测试源码确认包含重复 close、close 后 reset、新旧模型各关闭一次，以及已有 constant-action 全 ExecutionResult 相等断言。本审查未自行重运行。

截至本次最终只读复核，所有已报告的确定问题均有对应源码修复；零拍 result 已选择显式拒绝契约；mixed diagnostics history 也已修复。当前范围内没有未解决的确定缺陷。跨 120min 左端过期/空队列/全部 capacity-wait 的组合检查仍可作为补充覆盖，不能用该建议推断当前实现错误。正式历史 14 次冻结核验与完整回归须以后续运行日志为准，此处不预先声明完成。
