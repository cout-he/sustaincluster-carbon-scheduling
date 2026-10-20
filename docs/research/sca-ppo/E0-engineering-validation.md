# SCA-PPO V0.2-final：E0工程验证报告

**结论：E0通过，可以由用户审核后进入E1的数据谱系审计和规则+DDF验证。本次没有进入E1或E2。** 通过的范围是人工任务下的实现一致性、确定性保护和失败记账；尚未证明真实负载收益、训练可行性或泛化能力。

## 1. 仓库、隔离与设计身份

| 项目 | 实际值 |
|---|---|
| 原工作区 | `D:/A-myprofile/SustainCluster/object-model-a1` |
| 原分支 | `research/platform-v2-final`，开始时干净 |
| 冻结Tag / 解引用提交 | `platform-v2-frozen` / `e372f502fb58aaa5a742e8efb4ced254507ab298` |
| E0独立worktree | `D:/A-myprofile/SustainCluster/sca-ppo-v02` |
| E0分支 | `research/sca-ppo-v0.2`，从上述提交创建 |
| 本轮提交 | 本报告、代码和证据同属本地E0提交；用`git log -1 --format=%H`取得，不在文件中自引用其SHA |
| GitHub | 未推送，未修改main、其他研究分支或冻结Tag |

已完整阅读原工作区的`research_notes/sca-ppo-review/SCA-PPO-v0.2-final-design.md`和`external-discussion-record.md`，不再进行第三轮外部讨论。前者SHA256为`27d8c948ef71652abb1295049e368351ff6340510c8244c53324dfc8058f8bd9`；原文件处于Git忽略目录，本轮没有修改它们。执行契约和复现方法在本报告及代码中独立给出。

桌面worktree工具因聊天目录是仓库的父目录而返回`Not a git repository`，随后在已经核实的实际仓库使用`git worktree add -b`创建隔离检出，没有切换或清理原工作区。实施计划见[E0-implementation-plan.md](E0-implementation-plan.md)。

## 2. 工程实现

| 新增文件 | 作用 |
|---|---|
| `experiments/sca_ppo/edf_calendar.py` | 不可变已知任务快照、EDF尾日历、raw finish/grid release、当前槽完整预演 |
| `experiments/sca_ppo/deadline_filter.py` | 固定候选、7个确定性动作映射、逐候选累积DDF、总超期约束、联合重演及逐候选计时 |
| `experiments/sca_ppo/completion_guard.py` | 3960分钟完成检查、参考失败计数、首次失败时刻、回合锁存 |
| `experiments/sca_ppo/adapter.py` | 仅观察构造快照；输出V2原生Action；逐ID核对实际执行；失败、尾段及不可逆miss诊断 |
| `experiments/sca_ppo/e0_evidence.py` | 固定人工测试及8/32/128任务计时探针，写轻量CSV/manifest |
| `tests/sca_ppo/test_guards.py` | 30项新工程测试，含固定种子40个人工组合子案例 |
| `docs/research/sca-ppo/e0-results/` | 测试逐项结果、105次决策和360次候选检查的计时、代码哈希与运行环境 |

没有修改`experiments/platform_v1/`、`experiments/platform_v2/`、任何数据、容量、物理参数、Deadline/Duration定义或既有验收结果。没有网络、神经网络、PPO优化器或训练入口。未读取真实Train及封存Development/Validation/Test任务或环境输入。既有工程测试仅构造人工输入或临时无效文件名，不执行P1/P2完整验收。

### 2.1 EDF日历与V2一致的语义

输入`Snapshot(minute, capacity, sla, pending, running)`只包含已经观察到的任务。运行任务具有不可抢占的固定真实Finish；其Task从此前到达的观察缓存获得，不能借运行ID查未来任务注册表。适配器须与环境同时reset，不能中途接入一个从未观察过其running身份的新环境。

日历只做计算，没有真实环境引用，不访问未来到达、CI或天气，也不推进环境时钟。按`(arrival + sla*duration, arrival, task_id)`排序；对每个任务先逐项检查CPU/GPU/MEM，再检查defer，容量不足保留排队并允许后续任务回填。资源求和、比较次序和V2相同，不在容量适配判断中加入额外容差。

`finish = start + duration`；`release = 15*ceil(finish/15 - 1e-12)`。Deadline判断使用`finish <= deadline + 1e-9`，release只影响容量。反事实defer只作用于当前槽，以后均按无推迟EDF执行。日历保留全部已知任务的完成与释放，允许预测超过72小时，不截断或删除难任务。无pending后，所有剩余running的固定释放已知，不再逐槽空转计算排空。

非法任务或超单体容量会报错，不能静默遗漏后使CG通过。人工测试可使用小型抽象容量；与真实V2执行器比较时使用原OM清单，按其真实容量缩放人工需求，没有修改V2配置。

### 2.2 DDF与动作记录

共同候选集为参考EDF当前槽真正能启动的身份集合。7个动作仍是No-op、动态IT功率排序前1/4/16个、个人Slack排序前1/4/16个；平局使用EDF键。动态功率代理仍为`(0.005*CPU + 0.225*GPU)/0.94` kW，不称作净碳收益。E0用动作编号或显式人工请求，不采样PPO。

候选及其排名在槽开始时固定。第k个尝试被拒绝，不用新候选补位。每次均从同一快照模拟`已批准集合 ∪ 本候选`，不在上一次反事实末状态上继续。

记参考按时集合为`S_ref`、参考迟到集合为`L_ref`，接受须同时满足：

```text
S_ref ⊆ S_trial
Σ(i∈L_ref) max(F_trial[i]−d[i], 0)
    ≤ Σ(i∈L_ref) max(F_ref[i]−d[i], 0) + 1e−9分钟
max(release_trial) ≤ 3960分钟
```

没有新增风险评分、预测器、Slack保护模块或风险预算；为避免错误拒绝原迟到候选，本实现不做可选Slack快筛，全部进入原公式的日历检查。主版本保留总超期约束，不在E0运行其消融。

最后从原快照重新模拟全部批准集合，并完整预演本槽。`adapter.execute`用V2真实执行后的逐ID账本增量核对：

- `proposed`：固定规则/调用者请求，CG锁存后也可保留显式请求作为诊断。
- `approved`：DDF和CG实际允许的集合。
- `effective_policy_wait`：资源本来足够，实际因defer等待。
- `capacity_wait`：资源不足，包括某些已批准但实际被回填挤占的任务。
- `actual_started`：真正启动的任务。

同时核对资源使用、各轴瓶颈数、启动时间、真实Finish和Release。不一致则保存`PREVIEW_EXECUTION_MISMATCH`及预期/实际证据并抛异常；不静默修正，不重复执行已发生的真实槽。

### 2.3 CG、完整责任及额外账本诊断

CG是共同研究协议，不是任务Deadline，不修改原V2。如果无defer参考的已知任务不能在3960分钟内释放，记录`EDF_REFERENCE_COMPLETION_GUARD_FAILED`，锁存至reset，当前及后续均无主动推迟EDF。即使参考恢复通过，也不解除锁存；该失败不等于数学不可行性证明。

仿真仍完整288槽，不删任务或失败回合。所有槽请求20°C，原V2热安全器仍可覆盖；66–72小时不能通过改请求掩盖正在运行的任务。摘要分别保留参考失败次数/首次时刻、真实最后释放时刻、未完成数和真实公共空闲尾段有效性。若仍未完成，`actual_last_release_minute=None`，不将未来预计Release冒充实际最后释放。

适配器附带设计要求的逐ID不可逆miss诊断：步后时刻τ，已启动任务使用真实固定Finish，pending使用`τ+duration`这个最早合法启动下界，与原Deadline比较，每个ID最多一次；不拿EDF预测作为不可逆证据。全部完成时断言ID集合与原V2正式miss一致；失败/未完成回合保留censored区别。不实现奖励函数，尤其不擅自给未完成任务补罚。

## 3. 正确性测试：62/62 PASS

新E0 **30项**、既有V1人工工程回归**19项**、既有V2人工工程回归**13项**。逐测试名称与结果见[e0-results/tests.csv](e0-results/tests.csv)。固定种子测试内部还有40个组合子案例，不把它们重复加进62项计数。

| 用户要求 | 状态 | 对应关键测试/证据 |
|---|---|---|
| 1 无推迟EDF与V2一致 | PASS | `test_calendar_and_v2_full_known_suffix_agree`、`test_seeded_synthetic_calendar_v2_parity` |
| 2 CPU/GPU/MEM不足与回填 | PASS | `test_each_resource_and_backfill`、`test_calendar_parity_all_resources_and_numeric_boundaries` |
| 3 真实Finish与网格Release | PASS | `test_raw_finish_and_grid_release`、`test_running_grid_padding_blocks_capacity` |
| 4 自身可按时、伤害其他已知任务时拒绝 | PASS | `test_self_safe_but_other_known_job_late_is_rejected` |
| 5 单独安全、联合冲突 | PASS | `test_individually_safe_but_joint_defer_conflicts`，第二个候选拒绝 |
| 6 原迟到集合总超期不得增加 | PASS | `test_original_late_total_cannot_increase`；另验证负Slack原迟到候选总超期不变可通过 |
| 7 全拒绝正常回退EDF | PASS | `test_all_rejected_is_normal_edf_not_guard_failure`，不伪造CG失败 |
| 8 66小时边界 | PASS | `test_guard_exact_boundary_and_candidate_crossing`：3960通过、3975拒绝 |
| 9 失败后回合锁存 | PASS | 三类实际结果测试；后续空队列参考通过仍锁存，reset才解除 |
| 10 未知未来反例 | PASS | `test_unknown_future_counterexample`：无推迟0个miss；局部批准推迟后1个miss |
| 本槽预演与实际执行 | PASS | `test_preview_vs_real_capacity_before_defer`及所有适配器逐步断言 |
| 无未来、无副作用 | PASS | 限制观察键测试、真实环境深拷贝前后比较、反事实重复调用 |
| 无新到达的后缀性质 | PASS | `test_no_new_arrival_preserves_accepted_plan_suffix`，逐IDFinish/Release保持 |
| 不补位、16次预算、7动作 | PASS | 固定前缀测试、16候选实际检查、EDF平局与动作前缀测试 |
| 浮点Start反推边界 | PASS | `test_fractional_duration_running_start_cancellation`，D=60.01、a=15 |
| 工程不一致保留证据 | PASS | 人工破坏预演触发异常，保存诊断，真实时间已前进但不重试 |
| 完整回合miss/末态 | PASS | 已完成案例逐ID守恒；人工配对72小时末温差≤0.05K，检查储热差为有限值 |

最初先运行测试确认因尚无实现而导入失败，再实现；审查发现的浮点问题另外经历“新增回归失败→修复→通过”。没有通过改变测试期望、缩短时窗、丢任务或修改冻结源码取得通过。

### 3.1 两个关键DDF反例

以下均为人工单资源当量，CPU容量4、SLA2、所有到达0；可按OM CPU容量/4精确缩放，其他资源0。

**个体Slack不足以保护队列。** C=(D20,q2,d40)、A=(D30,q4,d60)、B=(D30,q2,d60)。参考C/B在0启动，A30–60；推迟B一槽后B15–45仍按时，但A45–75违约。DDF拒绝B，原因是丢失参考按时身份A。C的真实Finish20、Release30，是此反例的重要细节。

**分别通过不代表联合通过。** A/B=(D15,q1,d30)、C=(D30,q4,d60)、D=(D45,q2,d90)。参考A/B/D在0启动，C45–75已迟到15分钟。单独推迟A或B都不恶化参考按时集合及C的总超期；联合推迟会使C先0–30运行，A/B到30–45而违约。按请求(A,B)检查时只批准A，B被联合反事实拒绝；即使C改善也不能牺牲原按时身份。

### 3.2 CG的三种实际结果均保留

| 人工案例 | 初始已知参考最大Release | 真实最后Release | 未完成 | 66–72h真实空闲 | 失败锁存 |
|---|---:|---:|---:|---|---|
| EDF回填结构被未知新到达改变 | 4290 | 3960 | 0 | 是 | 保留 |
| D=4000分钟长任务 | 4005 | 4005 | 0 | 否 | 保留 |
| D=5000分钟长任务、SLA2 | 5010 | 未知/未全部释放 | 1 | 否 | 保留，原V2 censored=1 |

第一例同SLA2、容量4：A(a0,D990,q4)、B(a0,D1650,q1)、C(a0,D1320,q3)、D(a0,D1980,q1)、Z(a660,D330,q2)。0时不能看到Z；其到达改变后续回填，实际无推迟EDF恰在3960释放。它说明“参考EDF失败”不能称“任务集不可行”，也验证预测后来恢复并不会解除失败锁存。此例实际C迟到仍如实记录，不包装为服务质量成功。

## 4. 计算开销与复杂度

预先固定8/32/128个人工任务、每规模5次重复、7个动作，合计105次决策；没有调参或真实调度实验。任务D=`30+15*(i%7)`分钟，需求=`(1+i%4,(i%3)/2,1+i%5)`，抽象容量(16,8,64)，SLA3；只用于计算路径计时，不赋予真实设备或负载代表性。所有候选检查的原始计时见[timing-candidates.csv](e0-results/timing-candidates.csv)，决策见[timing-decisions.csv](e0-results/timing-decisions.csv)。

运行环境与准确数字见[manifest.json](e0-results/manifest.json)。使用`perf_counter_ns`转毫秒；P50/P95按nearest-rank `ceil(q*n)`，每规模池化7动作。参考各35次，候选分别110/130/120次，共360次。表中“完整决策”计时覆盖`adapter.decide`，含观察快照、参考、排名、CG、逐候选DDF和最终重演，不含V2物理step、日志落盘、网络或训练。

| 已知任务数 | 参考日历P50 / P95 / Max ms | 单候选P50 / P95 / Max ms | 完整决策P50 / P95 / Max ms |
|---:|---:|---:|---:|
| 8 | 0.0340 / 0.0521 / 0.0671 | 0.0399 / 0.0477 / 0.0583 | 0.2651 / 0.3630 / 0.4242 |
| 32 | 0.2728 / 0.3360 / 0.3443 | 0.2845 / 0.3450 / 0.4711 | 1.7665 / 3.0385 / 3.1458 |
| 128 | 3.3525 / 3.6228 / 3.7623 | 3.4406 / 3.7640 / 3.9414 | 20.7603 / 31.9178 / 31.9215 |

记N为当前known任务数、S为尾日历检查pending的15分钟槽数、k为本动作实际候选数（≤16）。一次日历先排序，然后逐槽扫描pending/running，保守时间上界`O(N log N + S*N)`、额外存储`O(N)`。一次正常决策包含1次参考、k次候选日历、1次最终联合日历及本槽预演，保守上界`O((k+2)*(N log N + S*N))`，另有身份集合、总超期检查。全部候选从原状态重算是主要开销；本实现没有日历缓存或不可靠增量近似。

小样本毫秒数不保证大队列成本可接受；长Duration、资源竞争造成长排队尾日历以及大known集合都可能明显放大开销。这里部分动作因本槽候选数小于预算而有相同请求；另有人工20候选测试确实执行16次检查，但不宣称真实动作覆盖充分。E1才应在授权Train上量化实际队列规模、动作别名率和决策成本。本轮不为追求更低时延改变算法定义。

## 5. 发现的问题、修复及保护边界

1. **已修复：反推Start的浮点消减。** V2只有running的Finish；例如`15+60.01`后再减`60.01`可能得到14.999999999999993。直接与arrival比较会误拒绝合法状态。现先检验反推Start距15分钟网格在固定1e-9分钟内，再还原该已知网格Start；Finish与Release原值保持。未修改V2或Deadline容差。
2. **已修复：预演失败的证据保存。** 实际step后若发现不一致，先保留预期/实际记录再抛异常；不能只抛异常而丢失诊断。人工破坏预演的测试确认此路径。
3. **不存在已发现的设计/V2核心不兼容。** 两轮约定的排序、资源先于defer、回填和网格释放均可在外部精确复现。以上修复属于实现数值/记账问题，不是变更研究方案。
4. **保护只针对当前已知集合的参考预测。** 未知未来到达反例已在原V2实跑人工任务证明；不保证未来DMR不高于立即EDF，不保证≤5%，EDF也不是多资源调度最优性证明。
5. **总超期只保总量。** 原迟到集合内部某个任务可以更晚、另一个更早；负Slack原迟到候选仍可能合法，不能额外禁止。
6. **CG失败不隐藏。** 失败回合仍完整执行且可有未完成/censored。现有设计的奖励在这些路径上可能少算未完成成本，因此不能凭E0通过立即训练；必须遵守后续E2终局责任门禁。
7. **E0只覆盖工程机制。** 未证明真实Train独立性、碳机会、DDF净收益、PPO增益或真实开销；没有把人工反例当算法性能实验。

独立只读代码审查已参与核验并指出第1项，修复后重新通过测试；全分支复核未发现阻塞问题。另根据其非阻塞建议，补充功率与Slack不等分数时预期排序不同的回归测试，再统一生成最终62项测试证据。所有正确性结论以本地可复现测试和证据为依据。

## 6. 复现与最小使用方式

在E0worktree根目录，使用Python标准库及仓库原有代码，无新增依赖：

```powershell
python -m unittest tests.sca_ppo.test_guards -v
python -m experiments.sca_ppo.e0_evidence
git diff e372f502fb58aaa5a742e8efb4ced254507ab298 -- experiments/platform_v1 experiments/platform_v2 data
```

最后一条应为空。证据入口只运行明确列举的人工测试模块，不做仓库全量发现，不调用真实数据加载器或正式验收入口。重复运行会更新计时/manifest，不改变算法或数据；绝对毫秒值自然会有波动。

适配器调用协议：

```python
from experiments.sca_ppo.adapter import ResearchAdapter

# env必须是已按原配置构造的V2环境；E0测试中全部是人工Scenario。
adapter = ResearchAdapter()
obs = env.reset()
done = False
while not done:
    decision = adapter.decide(obs, action_id=0)  # 或0..6的固定人工规则
    obs, done, row = adapter.execute(env, decision)
v2_result = env.result()          # 原V2全72小时账本，不改官方结果
research_summary = adapter.summary(env)
```

每个新回合同时reset环境与适配器。不能对旧状态重复执行Decision；陈旧Decision会报错。组件CG可独立复用于后续所有新对照，不能只给SCA-PPO加完成约束；E1尚未实现或运行这些对照。

## 7. 下一步决策

**E0完成。** 三组件、原生执行对照、十项指定边界及设计补充失败账本均已实现和验证。没有待修复的已知核心语义问题。

**具备进入E1的工程前提，但本次到此停止。** 用户审核后，E1先冻结合法Train谱系分组，再用共同CG、同候选/排序/预算比较EDF、规则和规则+DDF，并测实际开销和动作覆盖。仍不允许自动训练PPO。

**E2尚无启动依据。** 独立样本、真实动作空间、完整任务责任、成本与终局失败处理仍需后续阶段证据和授权；不得因为本报告通过就宣称SCA-PPO具有节碳优势。
