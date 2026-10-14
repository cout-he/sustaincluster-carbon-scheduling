# E1-7A A 模块独立只读审查（001）

日期：2026-10-14。范围为新增 `offline_schedule.py`、`offline_schedule_cases.py`、`offline_schedule_verify.py` 及三个对应测试，交叉阅读旧 `LocalExecutor`、`RunningTask`、时间/数值契约和本轮预注册。未改源码，未运行真实求解，未读取 Validation/Test；只读取已有微型测试记录，不能将本审查称为新一轮执行测试。

总体判断：有限 mandatory 排程的核心模型、真实 finish / 网格 release、固定背景、原资源校验及求解器状态边界正确；可进入预注册实验。真实能碳交付必须处理下面第1项隔离风险，并遵守第2项的输入身份责任和第3项命名范围。

## 1. 需要在正式能碳调用前显式处理的接口风险

`offline_schedule.py:305–317`、`:320–332` 中，即使调用者传了真实冻结 workload/environment，遗漏 `power_model` 也会静默选择 `_NoEnergyDiagnostic`，返回一个正常 `ExecutionResult`，其 summary 电能/碳都是0，**没有单独的 energy-mode 标记**。该默认类本身不是物理模型错误：A-only 微型资源/时序验证使用0功率完全合理；危险是实际环境输入也走相同默认而被后续表格当成真实总碳。

必须采取至少一种可验证措施：

1. 本轮真实 A2/B 物理重放的 driver 显式传独立 fresh `UpstreamPowerModel`，记录模型来源/配置哈希、非诊断标记，并自动断言该调用没有使用零功率诊断；或
2. 仅在新增 offline wrapper 中收紧契约，例如真实输入而未给 power 时拒绝，或者返回显式 `energy_kind=resource_only_diagnostic` 并让能碳汇总拒绝此模式；若改新增 wrapper，新增微型测试。

文档已有“默认零能量只测A资源时序”的正确声明，但 docstring 不能代替运行时检查。本审查没有更改默认，也没有宣称现有真实运行已经误用（正式结果实验尚未开始）。

## 2. 冻结输入身份的剩余责任

`_check_replay_inputs`（`:266–281`）检查完整候选+背景 ID 集、候选 `to_dict` 全字段相等、背景 demand、start≥arrival及release与原duration一致，并检查环境覆盖。它并未把 `case.source_workload_sha256` 与全 workload 内容SHA相等作为硬契约；对于背景，仅保留在 `FixedOccupancy` 中的三需求和资源时序，所以单独修改背景 deadline、source metadata，或向前移动 arrival且仍早于fixed start，可能不被此函数发现。

本轮 driver 必须沿用冻结全 workload/environment 原内容哈希和模型哈希，不能只因 wrapper 接受输入就称所有任务特征/CI/weather相同。A2应外部核对 source workload SHA及完整原 TaskSpec；A1 subset 则核对所选原任务字段并保留 full-source与subset两个身份。预注册已要求冻结输入；这是一项应在真实运行收据中落实的边界，而不是发现当前输入已经被修改。

同理，A2候选必须来自快照真实 **unstarted pending** 集合；`OfflineCase` 不接收/验证原候选已经运行的状态。已运行任务只能放固定背景而不能重新优化启动。根选择器冻结的 pending IDs / 候选来源是这项断言所需证据。

## 3. EDF 对照命名

`OfflineSchedulePolicy.order`（`:204–206`）先将背景置于候选前，再按deadline/arrival/ID；`_future_background_fits`（`:216–227`）检查候选整个 `[start,ceil(finish))` 占用区间，同时纳入固定未来背景和本拍已经启动的候选；`on_task_started` 即时追加候选，避免同拍后续候选忽略已占容量。

A2因此是 **conditional EDF under fixed background commitments**，有离线已知背景预约约束，绝不是旧无预约EDF。预注册和接口文档已正确区分。不能将A2相对CP-SAT差距直接写成“原EDF全集群最优差距”。

A1无背景、默认anchor=case.start时，未来容量只会因当前运行任务完成释放而改善，因此whole-hold容量检查不会额外禁用普通可启动候选。但是它仍有 `release<=case.end` 的有限终点约束，任意非默认anchor也会禁用提前启动。因此“A1等价普通EDF”要限定在**同一完整可排空的有限horizon且默认anchor**；更稳妥命名为“finite-horizon EDF reference”。若EDF无法在注册horizon全部完成，wrapper拒绝而不能默默延长窗口/删任务。

## 4. 核心机制逐项核查

| 项目 | 源码证据与判断 |
|---|---|
| 每个候选mandatory | 每TaskSpec有一个非optional interval；即使miss仍参加三维cumulative；未返回incumbent时不伪造starts；verify要求候选ID精确一次。没有按时任务子集选择或drop语义 |
| arrival/anchor | `OfflineCase`保持原TaskSpec；A1 arrival不能早于case.start；A2允许原arrival早于anchor，但solver下界/verify不得早于arrival或candidate_not_before。网格来自旧 `utc_time(...,True)` |
| 真实finish与release | solver以原 `pd.Timedelta(...).value` 算占用ceil；miss threshold以deadline−原duration纳秒落到启动网格，等号不miss。verify以实际finish判SLA，资源至finish.ceil15释放。近15min浮点转Timedelta、16min两拍、网格间deadline都有微测 |
| 非抢占/同一多资源区间 | 一个连续mandatory hold在CPU/GPU/MEM同一interval上使用；不是独立安排各资源，不更改duration/demand |
| 固定背景 | `FixedOccupancy`固定start/release/demand，要求三维/grid/正占用；候选与背景IDs不重叠。模型裁剪背景至有限case区间；完整A2 replay另从原起点执行全部候选与背景，并逐拍断言所有fixed due starts实际出现 |
| 当前运行任务 | 原正在运行的非候选以固定背景保持既有start和ceilrelease；候选不可已开始的事实由冻结pending selection保证，不能仅靠case类 |
| 原资源校验 | verify逐拍以每项原binary-float需求的 `math.fsum` 与capacity精确比较，无oversubscription epsilon；与旧LocalResourcePool规则一致。背景各计一次。真实执行仍由唯一旧LocalExecutor分配 |
| completion/终点 | verify要求候选release≤case.end；replay `start(until=case.end)` 不提前drain退出，保留尾部idle积分，并要求全部原任务账本completed。这是同固定horizon比较；环境必须覆盖终点 |
| integer资源 | Fraction使用原float精确二进制比例；binary denominators为2幂，因此取最大分母构成公共lattice，再GCD约分正确。超安全范围后固定1e6、Fraction ceil需求/floor容量，保守误差与capacity_loss明示 |
| int64 | 每维单系数及全部潜在累计需求和都≤INT64_MAX/4；CP-SAT `model.Validate()` 在Solve前运行，避免只防单系数却忽略cumulative和。给定预算worker1、seed710001、fallback1e6、时间≤120s，不追加搜索 |
| 最优性/下界 | OPTIMAL只证明已记录整数资源有限问题，FEASIBLE不是最优。保守舍入可使原可行问题INFEASIBLE；exact binary也可能与原fsum舍入边界不同。因此solver best_bound/miss_count不能无条件称原E1理论最优违约下界或全集群下界。源码OPTIMALITY_SCOPE和微测已说明 |
| 能碳与在线隔离 | wrappers未注册为在线policy，不提供给PPO。未来固定背景明确离线。碳证据必须显式真正IT/HVAC，不能用本模块0功率默认或求解器miss目标当碳模型 |

## 5. 测试证据及尚需正式验证事项

已读取 `offline-tests-final-001.txt` 的实际结尾：**97 passed in9.60s**，receipt说明43项新增和54项旧关键测试。三组测试包含资源各维、grid/非整duration、deadline等号/非整边界、所有迟到仍运行、固定背景避免重定义、同一原执行器回放、prefix原事件/功率相同、精确/保守整数反例、UNKNOWN/INFEASIBLE和小集合穷举对照。12任务合成例子的EDF6、整数CP-SAT OPTIMAL3只属既有微型机制例，不是本审查新运行的真实任务结果。

正式22个有限子问题仍需保留各solver状态、实际耗时、precision及original verification；不因TIME LIMIT继续搜索或只挑OPTIMAL案例。A2的全背景检查开销约为候选决策数×候选占用拍数×背景任务数；solver120秒不是整个Python replay的总耗时预算，driver应另记录reference/replay成本并监控而不是省略whole-hold检查。

真实A2物理结果必须满足：固定背景启动/释放全部不变、候选在合法网格开始且全部完成、full cohort一致、common horizon一致、资源小时守恒、真实环境与模型哈希一致、完整IT/HVAC积分。背景运行前缀的真实物理状态与历史基线应复核；已有KnownPower微测不足以单独证明原HVAC持久状态的真实全窗口一致。

本报告是独立代码审查结论，不是完整项目测试通过或科研机会已被证明的结论。
