# E1-6A 独立代码审查

审查日期：2026-10-13。范围：新 `task_risk_state.py`、`task_risk.py`、`task_risk_experiment.py`、`task_risk_analysis.py` 与四组新增测试。角色：独立只读审查；没有修改生产代码、测试、旧归档或 Git，没有读取 Test。首次审查时实验驱动仍在实施中，本节是实现前期参考版的静态核查，最终版本/优化与实际运行证据应在后续附录和测试报告中确认。

## 1. 首轮结论

参考版没有发现阻塞本轮声明实验的实际机制 bug。方法公式、动态 observed state 与公共执行器关系符合冻结需求。此结论不替代测试、真实全窗口守恒/回归或性能实验，也不宣称任务预算保证 SLA。

| 检查项 | 代码观察 | 结论 |
|---|---|---|
| 唯一内核 | run_group 构造原 `LocalExecutor` 与 bundle.power_factory；新policy只有eligible规则 | 不存在新仿真执行器或物理公式 |
| 全队列 EDF | 新类继承 `EDFBackfill.order`；保存拍初 ordered/rank | 含旧等待者、稳定deadline/arrival/id排序、保留公共单次回填 |
| 动态紧迫前序 | `ordered[:rank]` 严格前序，过滤仍在pending；budget排除self、L−now<=H | 排除自身/后序；含过期、先前defer和capacity fail |
| 同拍启动更新 | `on_task_started` 删除pending并追加running；budget对当前running `math.fsum` | 后续候选看到新利用率，工作不重复记pending |
| 容量失败 | 新policy从不擅自删除；engine失败分支不调用policy，公共requeue | 不丢竞争集合、不伪造guard或启动 |
| defer | hook验证until=now+STEP但不删除、不预留资源 | 仍占等待身份、不占running资源 |
| 历史与cold | 沿旧ArrivalHistory，每拍observe；effective=0时cold，nohistory仅a_h0 | 去重/开左120min/空已观测窗口与共同cold保持 |
| ceil与网格 | 原latest_start减原duration后floor15；B用math.ceil，无optimistic epsilon | finish/release没有混用，整数临界保守按浮点值执行 |
| 零需求/required-max | matched总和0在max前分支，B=0；max仅q>0维 | 不因空集合报错，不因无关资源触发 |
| 在线边界 | 新policy/state不持有workload/env/FrozenWorkload，不import sla_feasibility/Shadow | 当前spec、snapshot、past arrivals之外无预测输入 |
| 诊断分母 | 仅highCI、gridS>=15且can-fit算budget；early exit budget=None；capacityfail无policy诊断 | null没有当0统计，eligible分布语义可审计 |
| 旧策略/RL | 新独立registry委托旧factory；rl_env/action/schema未改 | 需要旧回归数值证据，静态无侵入 |

## 2. 驱动与分析核查

`run_group` 包含 EDF、旧Carbon、旧RiskOnly与5个新变体，共8个策略。每个策略持久模型先完整Drain，组内最慢实际终点作为共同观察端点，先排空者沿相同模型 `extend_idle`；未排空者记录safety-cap与比较有效性false。固定窗、own排空和common三phase同时保存。守恒核查包括不可变原task字段、唯一集合、pool ownership、capacity inventory，以及全完成组的三维资源小时一致。

基线回归对旧ledger/events/policy-ledger/policy-events逐字节比较；对可能不同公共尾段只要求共有时间前缀的资源、能碳和queue记录一致，尾段在新组内统一重算，没有改写历史归档。

固定campaign参数字典、6个Train/Validation输入身份和源代码哈希在运行前冻结。Validation要求Train完整三倍率正确性gate；gate不按DMR或carbon选择参数。入口拒绝Test与未注册倍率；config宣告不训练。单group结果不允许覆盖。replay要求同source/Python/dependencies/input身份、同科学输出哈希，运行耗时/provenance另列而不混入数值一致性。

分析的Pareto只在同split/factor内比较，等点不视为严格支配；paired任务必须完整一一匹配。新增/救回/启动差异标记为轨迹关联，未假称单次事件因果。

## 3. 新测试的有效覆盖与边界

读取测试看到：真实旧 Task/资源分配/LocalExecutor 在微型运行中执行，KnownPower仅隔离复杂物理来验证积分时序；duration16/deadline35，exact-safe15/CI等号，cold/open-left过期，三维only与小需求GPU瓶颈，CPU↔GPU不同任务子集，same-step start/defer/capacityfail，旧等待者重排、stable tie/backfill、三资源/任务守恒、future分叉、nonpreemption、删失/末拍、不变旧RL动作和固定Action完整result相等、重复replay均有断言。

driver测试覆盖Test拒绝在output/physics之前、不覆盖旧输出、8模型持续空闲积分、不可完成任务不能宣称等量减碳、旧archive不变、科学回放及tamper拒绝、Train gate不同source/params/incomplete拒绝。纯方法测试校验公式分项，分析测试校验分母与配对集合。

这仍不等于真实物理模型数值回归已经通过；该证据需要实际campaign对历史账本、每拍功率/碳以及资源小时验证。复杂缓存优化需要逐候选与参考版成员/数值/动作相等的独立检查，不能只看累计DMR或reward。

## 4. 非阻塞的防御性与解释边界

- 首轮 `_budget_from_pressures` 对cold只要求“至少一维None”；合法旧ArrivalHistory保证三维None，因此当时为非阻塞防御建议。后续新模块已经补上effective=0三维全None、effective>0三维数值的严格校验，没有修改旧ArrivalHistory或正常观测行为。
- 纯 `BudgetParameters` 可容纳其他非负系数用于微型公式验证；本轮campaign通过exact参数字典锁定用户值。不能把宽API误读为允许Validation系数搜索。
- 零需求分支输出B=0，但cold_steps字段仍描述当时历史cold状态；分析应以实际buffer为准，不能把元信息cold2再额外加到零任务B。
- 加权需求可能低估极小而稀缺资源；required-max对照与对应反例已明确保留。两者都不是未来容量可行充分条件。
- 浮点ceil临界可能使数值上稍高于整数的score多保留一拍；当前实现故意无optimistic epsilon，测试保留该规则。优化只能在参考同值范围保持动作，不能通过改ceil容差消除它。
- 不同policy改变后续状态与eligible分母，guard/预算分布计数不能直接加减解释某单因素的唯一因果贡献。真正性能判断以同输入common核算和完整配对结果为准。

## 5. 静态工作缓存的后续独立复核

后续代码保留纯参考并增加 `query_backend='reference'|'cached_work'`（默认缓存）。复核版本身份为：

| 文件 | SHA256 |
|---|---|
| `task_risk.py` | `6a77eb5af687c7183a527d7adcd549d4e5ec70f6e2190cc4db184bd83cbe00c2` |
| `task_risk_state.py` | `e2bf5d86b30170a9766428e287b529d36e0a0421447a3318f03eb220cc4547e1` |
| `test_task_risk_state.py` | `d50a6de5ef15bcc64234806c18d5b940339f7dc095edbc9567f67c5435759913` |

缓存只发生在 `on_step_start` 已知ordered成员内：`grid_starts`缓存原latest_start floor15，`urgent_work`按同now/horizon决定静态紧迫性、存每个 `q*ceil15(duration)`。该拍now不变、spec不可变，所以与reference在每候选重算相同。候选严格前序仍按原ordered slice过滤实时pending；启动通知删除pending/追加running，defer/capacityfail没有删除。于是缓存里虽然保留已启动任务的静态事实，其工作因实时pending过滤不会重复计算。

资源利用率继续对真实running逐维 `math.fsum`；urgent乘积与reference相同，成员/顺序不变，逐维fsum和容量×H分母相同。没有浮点prefix subtraction、新epsilon或新的ceil规则。共享 `_budget_from_pressures` 进一步使weighted/max、cold/zero及buffer边界相同。未发现缓存造成的实际在线、成员或数值机制偏差。

读取新增逐候选测试：5种新policy，45个不同duration/slack任务，预设不通知的capacity-failed前序及启动/defer混合，逐候选比较整个diagnostics与decision（非近似）；另在真实LocalExecutor中比较完整ExecutionResult。读取 `cache-green-001.txt` 的实际结果为 **53 passed in 2.05s**。读取profiling日志，512候选、同130816前序扫描，参考3.151792240s、缓存0.065772965s。该合成timing只证明常数开销下降，不代表真实trace速度、性能优势或统计显著性。

两后端最坏复杂度仍是排序 `O(n log n)` 加候选合计 `O(n²+n*m)`，额外状态 `O(n+m)`；没有Fenwick/log级实现。已经检查ceil边界通过同乘积/同fsum/同共享ceil避免人为差异；现有纯方法测试另验证score恰整数及刚高于整数的保守ceil。截止此次后续review，完整真实Train/Validation消融尚未形成结束证据，本文不声称真实实验已经通过或表现更好。

## 6. 冻结分析程序静态终审

正式campaign运行中，源码已冻结。此次仅只读审查分析程序和配置，没有修改生产代码、测试或任何已产出的scientific manifest。审查身份：

| 文件 | SHA256 |
|---|---|
| `task_risk_analysis.py` | `2c1f0f2a65acfce4191f485d459b5fc81107760540fa985989d7a6530da51280` |
| `task_risk_experiment.py` | `cd3a19bc66be307724bded78629f8660b644c42dc488bb9c9dffe1479ff031af` |
| `configs/e1_6a.json` | `e1e51b65eefc1529b2fecdff73243d3687fcaf7f0ce8f81755f269057797881a` |

未发现阻塞声明分析的实际计算bug。完整实验尚未结束，以下是代码路径结论，不是实际48组结果通过的预先声明。

### 6.1 完整分母与公平核算路径

分析先 `_checked_groups`，在创建输出目录前要求冻结6个Train/Validation cohort及每组8个公共窗口policy row；核查全部scientific hashes、complete运行状态、源哈希/config、冻结workload/environment、共同观察终点、旧baseline regression、无Test/无future CI。config保留用户字面参数、5%目标与固定倍率，没有根据结果挑负载或方法。

每组summary从complete ledger重建唯一task identity，逐policy核验policy ledger身份、所有方法完整同task集合与arrival/duration/三资源/deadline字段；summary DMR重算 `misses/arrived` 并与ledger/原summary核对，资源/任务数量与IT+HVAC总电能加法一致。原group已经验证完成量、capacity/task ownership与三资源小时相等，分析读取被哈希封存的fairness状态。fixed/own/common三phase均保存，公共窗口与各自drain没有混用。

若将来存在未排空组，分析的 `equal_workload_carbon_comparison_valid=False` 使相对EDF碳减排为null；未完成/删失仍留在统计。`meets_5_percent` 是观察到达任务DMR且无未来deadline删失的布尔项，正式论文判断仍必须同时读取完整Drain/等工作量有效性，不能脱离终态单独宣称准入。当前方法不把5%改为高压35%，不跨档平均DMR掩盖失效。

### 6.2 配对碳排放与因果边界

实际计算为：

```text
carbon_reduction_percent_vs_edf = 100 * (C_EDF - C_policy) / C_EDF
```

其中两种C均是同cohort、同公共观察窗口、空闲尾段已纳入的总kgCO2e。只有equal-work核算有效且C_EDF>0时产生值，其他为null。没有以不同倍率或Train与Validation的绝对carbon差异推导减排。

paired新增/救回/共同miss及start shift来自全部完整任务一一匹配，并与driver保存的paired summary复核。Pareto只在同split/factor内比较carbon与DMR、严格改善至少一维；等点保留共同nondominated。这是有限方法集合的描述性支配关系，不是全局最优、严格因果或显著性证明。

代表任务固定选三类：Full相对RiskOnly的added miss、avoided miss、changedstart/sameoutcome，分别按第一个lexical stable_id选取；无该类则保留found=false。每个选中任务输出全部8策略，不只展示成功策略/阶段。选择使用完整结果是离线解释性分析，未输入任何在线policy，也未据此重调v1参数。

### 6.3 图表和底层CSV

全部图的数据均来自已核验tables，图源CSV与summary/buffer/resource统计使用同对象写出；未截取有利任务行。图1/2覆盖6个cohort与全部8方法；高压Pareto明确固定Train2、Validation1.5/2三组；budget图仅5个实际拥有任务预算的新policy，first-eligible人口和n逐行声明，未把early-exit null当0。资源主导类别按normalized demand判定，缺失类别保留缺失，不填假均值。

DMR纵轴从0，5%线不变；配对carbon百分比纵轴关于0对称且含负值；Pareto横轴从0。没有凭单个确定性trace构造error bar或显著性，图注明确Validation是开发数据。PNG/PDF/SVG均来自同一fig，保存300dpi/可编辑字体，并提供完整图源CSV。静态审查只能确认数据路径；真实图像可读性和实际文件哈希仍需最终渲染QA与结果审核。

### 6.4 文字歧义的明确澄清

冻结程序的 `figure-captions.md` 自动文字含“100*(EDF-carbon)/EDF-carbon”，字面容易将EDF与carbon读成错误变量/运算。实际代码与本节公式明确为 `100*(C_EDF-C_policy)/C_EDF`，数值计算没有该歧义。源码已冻结，本审查不更改其字节或历史输出；交付主报告应使用上式并链接本澄清，不能把含混句照抄为论文数学公式。

### 6.5 provenance与仍待结果核验

analysis provenance保存campaign/config/experiment-source身份、单独analysis-source身份、每组scientific manifest与runtime哈希、实际命令/Python、无Test/无新training标志、开发Validation说明、单次确定性样本与代表案例规则。新的analysis输出通过_fresh_run拒绝覆盖、最后封存hash。它不改写原历史provenance，且没有从分析读数返回strategy参数更新。

本节没有声称六档已全部完成、等工作量减排成立、任务预算优于RiskOnly或科研创新成立。下一次独立结果审核须核对实际六组公平状态、48条完整指标、ledger/逐拍积分、配对case与图源，特别保留退化EDF、缓冲过大、不优于RiskOnly和小需求瓶颈失败现象。

## 7. 实际六档结果的独立核验

六个正式group与analysis-001全部完成后，另运行仅标准库的 `verify_completed_results.py`；未import仿真、policy、物理模型、analysis或冻结输入loader，没有读取Test。执行退出码0，实际结果为 `all_passed=true, groups=6, policies=8, rows=48`。详细收据 `independent-result-verification-001.json` SHA256=`cabf477f96e4cf751a05c6c89a572081cedbd2bbdb08952a3cae6221f52f56dd`；精简发布收据另存 `results-independent-verification.json`，没有覆盖campaign或旧归档。

独立检查逐个验证所有scientific manifest文件哈希、源/配置身份、完整八策略唯一task集合与全部冻结字段；用原ledger finish/deadline纳秒时间自行重算SLA；用开始/ceil15释放事件重建每个interval真实running与pending，检查三资源占用/可用守恒及容量；核查IT/HVAC各自kW×0.25h、总电能加法、逐拍kWh×gCO2e/kWh÷1000碳与累积值、同完成任务资源小时、等待账本和平均JCT。48运行全部完成，无未完成/删失，common endpoint分别Train57h、Validation55.75h。

分析程序的原输出哈希另独立校验后才生成 `E1-6A-ablation-report.md`。真实结论与公式期望没有混淆：Full六档没有一档同时优于RiskOnly的Carbon/DMR；Validation2 Full6265miss/14430.364kg，对比Risk5924miss/14396.650kg，双目标更差；Train2减少51miss但多17.588kg碳；Validation1.5多36miss但少7.006kg碳。Validation1所有新方法仍略超5%，没有舍入达标或改门槛。

真实first-eligible Full一拍预算比例依次100%、92.427%、84.052%、98.933%、96.661%、74.150%；有数千次主动延迟，非纯EDF退化，但轻/中载的预算多样性较小。No History的其余五档miss均少于Full，No Urgent在轻载几乎无影响、高压方向混合；已保留全部反向消融，不据微型表达能力宣称性能或新颖性成立。

代表性任务依据冻结的added/avoided/changed三类stable-ID规则列出；例如Train1.5的added与avoided案例自身均无主动defer，差异发生于容量等待/运行资源分配轨迹，进一步说明不能把全局净差直接归为该任务一次Carbon主动延迟的因果。本轮没有做一次动作干预。

图像最终可读性由独立render QA处理；发现展示问题应在新notes图目录重新渲染并保留原analysis字节/hash，不能为图修复改已冻结策略/分析源码或重写科学数值。审查支持按当前负面结果完整交付，不支持声称该v1已稳健优于RiskOnly，也不支持自动调参、训练或访问Test。
