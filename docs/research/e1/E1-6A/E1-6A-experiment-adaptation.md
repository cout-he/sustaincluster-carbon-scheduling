# E1-6A 实验驱动适配审计与运行方案

本文件是实施前的只读审计及驱动设计，不代表已完成实验。审计于2026-10-13（Asia/Shanghai）进行。仅读取已有Train/Validation冻结包和历史结果；未读取Test、checkpoint或任何未来轨迹作为在线输入，未修改旧科研源码、旧输出或Git。

## 一、可复用的接口与不能直接复用的注册路径

`rl_training_utils.load_bundle(path, allowed_split=...)` 已能核验冻结包的每个payload文件、任务/环境快照身份、原数据身份、容量、阈值161.075、SLA1.5、资源倍率5、stress seed及UTC到达/安全窗口。优先直接加载现有六包，不重新生成相同负载或修改任务ID。

`LocalExecutor(workload, environment, capacity, model, policy, drain=True)` 接受独立策略对象。`run(stop_at_drain=True)` 完成有限任务集合后，在实际释放资源的网格边界停止；每个策略仍持有运行至排空时刻的同一个IT/HVAC模型。`extend_idle(common_end)` 使用该模型内部状态及真实对应CI/天气继续积分至公共终点，不重新初始化模型。

`rl_evaluate.evaluate_group` 的直接基线注册被固定为旧`ACTION_POLICIES`与`make_cars_policy`。其`candidates`接收预测器，经旧Discrete(3) RL环境执行，不能直接表达新任务级策略。因此本轮不改该函数、不monkeypatch旧registry，也不伪装为第四个RL动作。新增独立`task_risk_experiment.py`，复用其公平核算模式和`LocalExecutor`，但策略解析仅在新驱动中完成。

可直接复用且无需改动的辅助接口：`cars_experiment.TimingPolicy`（真实order/decide及存在的四个hook计时）、`run._write_result`（完整任务/事件/逐拍输出）、`run._fresh_run`（拒绝覆盖并保留失败运行）、`drain_metrics.phase_summary`（固定到达、各自排空、公共窗口统计）、`temporal.write_csv`与`experiment_sources`（源码/测试哈希）。旧`cars_experiment._write_science`与`paired_shifts`绑定`immediate_fifo`，不作为本轮新八策略组的通用输出函数。

## 二、六个输入组与冻结身份

Train时间轴：2023-05-01 05:00至05-03 05:00 UTC；环境覆盖至06-02 05:00。Validation：2023-07-08 05:00至07-10 05:00 UTC；环境覆盖至08-09 05:00。每组容量CPU/GPU/MEM=(50000,1000,80000)，HVAC22°C、15分钟积分；原配置规范化SHA为`65e4f2fce03fbef88c16dac980cdf1104a2ca547a787802604e0e9242d5d7228`。以下路径均相对科研根目录。

| 组 | 任务数 | 冻结包路径 | workload SHA256 | bundle manifest SHA256 |
|---|---:|---|---|---|
| Train1× | 5542 | research_notes/E1/E1-5B/precheck-001/load-1/input | 4a2249a1f42ebf7bda6a1388038e03596084b648e2c99e99196220a8be927862 | bf80a31419983606431775f101141cdf7bb4de69f75180c77553a4b3c31a0d45 |
| Train1.5× | 8313 | research_notes/E1/E1-5B/precheck-001/load-1.5/input | 5f408a4639113f739b8b3096111af1067edb1793d856db20c040c9380dbbd879 | 4e02f1a1525c2de262ce25970674badd2e620177820951cb7c4cf1c1d8f686b6 |
| Train2× | 11084 | research_notes/E1/E1-5B/precheck-001/load-2/input | e110a35f4be9d40327139c7e587aa4c19898b89f290ce43c7a1261eac6e80399 | e3d76b2daff040e6d783cd0ce8a81766b98ea996cf02c1111f4fe3529576c77a |
| Validation1× | 8708 | research_notes/E1/E1-5C/diagnostics-001/inputs/validation-1 | 48c4cbfa5ffe865ceb8826bec9e05c89019ce40bd5434694b780cac25c74f55b | db65bf311e533991ef4cb2f0698c96570d2ebba0185b963c3bb3c16e656a0a18 |
| Validation1.5× | 13062 | research_notes/E1/E1-5C/diagnostics-001/inputs/validation-1.5 | a9615365e558c8f551ebac84ea43b8cdfdaef013c200e8a2284f3995ecd71959 | 55238b2cedba9326cc46921b2c786bff83350c2faf1e7dd69ab39083fda185ac |
| Validation2× | 17416 | research_notes/E1/E1-5B/validation-input-001 | 197b577646b89bc5c0c0727d0534b297851fd53a0e61eb7e73ce26f32c6b49de | 62658e53a8edcf1b4fe74f5b738f7f4628a8defdf271a1b8d848f23c7fc18f02 |

Train三个组环境SHA：`5aa2e4016f39c2230fa462349c7d1316e364d9364c9d4791feb3a580732bdb3c`；Validation三个组环境SHA：`1ae071256e280a9b2f56736aa80b88c07385f2e7c0e59865bc9911c2f563c319`。Train stress seed510001；Validation520001。2×复制完整原任务集合一次，seed不能提供独立到达形态。

新驱动应使用显式相对路径清单，不依赖E1-5C manifest内混合的WSL绝对路径和相对路径。E1-5C输入manifest仅作为只读核对来源，其历史provenance保持原字节。

## 三、八策略公平组

每组运行：EDF参考`edf_backfill`，用户要求的旧A/B `edf_carbon_threshold`、`cars_risk_only`，以及C–G `task_risk_full`、`task_risk_no_history`、`task_risk_no_urgent`、`task_risk_no_utilization`、`task_risk_max_required`。旧三策略由原`make_cars_policy`解析；五个新策略由独立registry解析。用户建议名`task_risk_budget_v1`可作为Full构造器的别名，但实验表不重复计算为第九策略。

每个策略构造全新执行器和物理模型，连续运行至各自真实排空或原安全终点，保留live执行器。八策略都完成后取`common_end=max(ex.observed_end)`；较早排空者调用`extend_idle`，其他仍未完成者保留安全终点结果。所有结果按同一终点统计资源小时、能碳及任务状态。安全数据覆盖不足时不能通过复制天气/CI补齐，也不能对不同完成量宣布减排优势。

同组必须检查：任务唯一、所有到达任务有终态、CPU/GPU/MEM不超容量、原task特征完全一致、工作负载/环境/硬件哈希一致；全部完成时三维资源小时相等。固定到达窗口、各自排空和公共观察窗口的结果分别保存；不跨Train与Validation比较绝对碳排放。

历史Train公共终点为05-03 14:00，历史Validation2×公共终点为07-10 12:45 UTC。新策略若更晚排空，会改变公共核算终点。必须在新的八策略公平组内重新积分旧EDF/Carbon/Risk轨迹，不能直接用E1-5C历史总碳相减。如果新公共终点与旧终点相同，可核对完整数值；若不同，旧策略账本、任务事件、策略事件及旧公共终点以前逐拍结果必须完全一致，并报告新增空闲尾段的能碳。回归不包含挂钟时间或新provenance字段。

## 四、先Train后Validation的功能gate

1. 先完成微型行为测试及旧固定动作/环境回归；冻结v1参数和源码身份。固定参数不得因Validation结果改变。
2. 读取并核验六个现有冻结包，写新的input-manifest和method-parameter-freeze；仅记录Validation输入身份，不提前运行其策略。
3. 运行Train1×、1.5×、2×。每组结束保存fairness、回归和诊断。
4. Train gate检查实现与边界：守恒、完整排空、历史策略轨迹回归、无未来泄漏、参数不变、GPU/CPU任务差异微型测试、零资源与cold语义、风险输出有限且B≤150分钟、回填行为。记录Train缓冲/理由/主动延迟分布及退化风险。
5. gate不以“胜过RiskOnly”或“DMR≤5%”为运行Validation的条件。预注册公式性能不好也是必须保留的结果。只有真实实现错误、数据问题或关键守恒失败才阻止后续，并保存失败证据。
6. gate通过后写不可变train-functional-gate.json，再按1×、1.5×、2×运行开发Validation。Validation不是独立泛化测试；Test保持封存。

## 五、诊断、运行开销与机制分析

完整每任务每次可容纳候选的风险诊断保存本地JSONL，字段含time/task_id、raw/grid solo slack、S、B、safe_slack、w、u、Q、rho/unknown/cold、各项贡献、需求主导类别、决策理由及容量可容纳条件。容量失败发生在policy.decide之前，必须从执行器policy_events单独统计，不能将缺少风险诊断的capacity事件记成新策略的主动执行或主动延迟。

建议缓冲分布分两口径：所有候选检查（同一任务可能多次）；每任务首次符合高碳且有一拍slack的风险评估（每任务一次）。低碳/无slack直接执行不能混进B分布并默认为0。按m_ir=q_ir/C_r最大维度确定需求主导类别，tie及零资源单独标注，不用原始CPU/GPU/MEM量的不可比数值判主导。

主要机制输出：每策略理由计数、policy/capacity waiting、B每拍分布及分组分位数、高碳有slack候选实际defer比例、相对EDF/旧RiskOnly配对违约集合、启动时刻变化及代表任务事件时间线。代表任务采用事先声明的首个稳定task_id规则，包括成功、失败、与EDF相同和加权平均可能低估关键资源的微型反例，避免仅选成功案例。配对差异表示关联，不能等同单次事件因果。

`TimingPolicy`可记录order/decide与hook耗时，完整run_seconds包含调度/模型/记录。运行开销必须区分总耗时、策略计算时间和候选次数。先正确参考实现，再按profiling决定前缀优化；禁止每候选重新排序或ShadowPlanner。流式diagnostic sink是可选内存优化，不能更改决策；旧RiskOnly已有内存diagnostics，必要时逐策略保存后释放结果/模型需谨慎，提前关闭模型会破坏公共空闲积分。

## 六、严格新目录与回放

推荐`research_notes/E1/E1-6A/campaign-001/`，每组子目录`train-1`等；Train gate、input/source/parameter manifest均写于执行前。任何路径已存在则拒绝覆盖，失败日志与run_status也保留。新驱动不将旧归档复制或重写为新版本。

回放须校验源字节、Python/依赖、参数、冻结输入与全部科学输出哈希，重新运行至同一公共终点，科学结果要求字节一致；运行时间/provenance时间戳分离不参与科学哈希。至少微型八策略组全回放，以及一个完整Train组回放或全部六组回放（按真实成本决定并如实报告范围）。旧历史回放函数可能因新增文件改变全源码哈希而拒绝当前源码；应使用历史归档代码或本轮旧策略数值前缀回归，不能降低旧strict-source要求。

GitHub仅同步必要源码、测试、配置、精简报告、小体量汇总CSV及选择的图。完整冻结包、风险诊断JSONL、逐任务账本/事件、日志、源码快照、原数据和checkpoint留在科研目录。发布副本仍依赖外部已冻结包，不能宣称包含完整真实数据。

## 七、建议驱动测试

- 仅Train/Validation允许、拒绝Test与unsafe相对路径；参数或bundle哈希改变拒绝执行。
- 八策略同一task/environment身份，完整Drain后公共终点一致且闲置模型不重置。
- 空闲后延长终点的真实能碳积分，与提前关闭/重建物理模型不同的fixture验证。
- 未完成安全截断保留账本和删失，不能通过结果筛选丢弃任务。
- train gate未通过时不运行Validation；gate仅检查正确性，不根据优胜结果挑选策略。
- oldbaseline短窗/完整账本和逐拍前缀精确回归。
- 新目录拒绝覆盖；科学回放字节一致，耗时独立。
- diagnostics不把capacity waiting当主动defer，首次候选分布不重复统计同一任务。

本轮该审计文件尚未触发实验，没有产生结果或算法性能结论。
