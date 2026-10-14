# E1-7A 文献机制与模型边界审计

审计日期：2026-10-14（Asia/Shanghai）。本报告只为 E1-7A 的排程空间、碳机会和单次延迟诊断提供机制边界；未实现资源预约、DRO、弹性任务或新 RL。文献不能替代本轮 A/B/C 实验，也不能据此预先认定某个模块有效。

## 1. 来源身份、版本与读取范围

五篇均已有本地 PDF。第五篇不是凭摘要补齐：在本地发现 `2302.08681v2.pdf`，核对其首页标题和作者后读取。四个用户指定文件及该第五文件只读，原 PDF 不复制进发布仓库。

| 简称 | 核对后的文献身份 | 本轮证据版本 | 页数与读取完整性 | 主来源链接 |
|---|---|---|---|---|
| Backfilling | Ahuva W. Mu'alem, Dror G. Feitelson. *Utilization, Predictability, Workloads, and User Runtime Estimates in Scheduling the IBM SP2 with Backfilling*. IEEE TPDS 12(6), 529–543, 2001 | 用户提供的 IEEE 文章 PDF；DOI 10.1109/71.932708 | 15 页，PASS | [IEEE DOI](https://doi.org/10.1109/71.932708) |
| GREEN | Kaiqiang Xu, Decang Sun, Han Tian, Junxue Zhang, Kai Chen. *GREEN: Carbon-efficient Resource Scheduling for Machine Learning Clusters*. NSDI 2025 | 带 USENIX proceedings 封面的最终稿；正文 999–1014 | 17 页，PASS | [USENIX 正式论文页](https://www.usenix.org/conference/nsdi25/presentation/xu-kaiqiang) |
| Hall | Sophie Hall, Francesco Micheli, Giuseppe Belgioioso, Ana Radovanović, Florian Dörfler. *Carbon-Aware Computing for Data Centers with Probabilistic Performance Guarantees* | arXiv **2410.21510v3**，2025-10-27 修订；不是根据页眉 “August 2021” 判定出版年份 | 14 页，PASS | [固定 v3 来源](https://arxiv.org/abs/2410.21510v3) |
| SensaScale | Kaimin Wu, Xinhua Wang, Zhiping Peng, Rong Xie, Weiwei Lin, Keqin Li. *SensaScale: A deadline-aware scheduler for DNN training tasks based on resource sensitivity* | DOI 10.1016/j.future.2026.108820；2026-09-12 accepted，**2026-09-23 available online**；卷期为 FGCS **188 (March 2027), 108820** | 15 页，PASS | [出版社文章页](https://www.sciencedirect.com/science/article/pii/S0167739X26004541)、[DOI](https://doi.org/10.1016/j.future.2026.108820) |
| CarbonScaler | Walid A. Hanafy, Qianlin Liang, Noman Bashir, David Irwin, Prashant Shenoy. *CarbonScaler: Leveraging Cloud Workload Elasticity for Optimizing Carbon-Efficiency* | arXiv **2302.08681v2**（2023-10-19）；首页/页脚标 POMACS 7(3), Article 57, December 2023，28 页；不是将 2024 的两页会议摘要当全文 | 28 页，PASS | [固定 v2 来源](https://arxiv.org/abs/2302.08681v2)、[ACM DOI](https://doi.org/10.1145/3626788)、[作者代码仓库](https://github.com/umassos/CarbonScaler) |

读取完整性采用 ARS `pdf_read_preflight/1.0.0`：根 `/Count`、递归叶子计数和 pypdf 页列表一致，且无解析修复告警。五份 sidecar 在本地 `literature/*-preflight.json`；每份均 PASS。PASS 仅支持页面定位可信，不代表论文的论证或结论已被实验复现。

机器提取覆盖全部页；本轮机制审阅集中于下表列出的方法、假设及限制章节，不声明逐行人工阅读全文，不声明作者手工读完。另用 pypdfium2 渲染并检查五个方法页：Backfilling p2、GREEN p8、Hall p8、SensaScale p5、CarbonScaler p9，核对 profile 算法、公式、回填示意图及脚注，没有依赖错位的双栏提取去推断机制。以下页码为 PDF 从 1 开始的页序号；GREEN 页码含其封面，另提供印刷页号以免错位。

## 2. 五篇机制表

| 论文 | 解决的问题 | 关键状态信息 | 主要控制动作 | 抢占与弹性 | 未来信息/预测 | Deadline 或性能保证的实际边界 |
|---|---|---|---|---|---|---|
| Backfilling | FCFS 分区造成空洞、低利用率；小任务越队与大任务饥饿/可预测启动之间的取舍 | 已排队任务、预定启动时刻；运行任务和预约的未来 processor usage **profile**；用户 runtime estimates | 在 profile 中找足够处理器且持续足够长的 **anchor point**；加入资源时间预约；当前可启动则启动；提前完成后按原顺序压缩预约 | 本文 backfilling 分区按请求处理器数运行；机制不是运行任务的弹性伸缩或抢占 | 用已知运行/排队任务和用户时长估计预测释放；不需要知道未来新到达任务；没有 CI | Conservative 为**每个既有排队任务保护预约启动承诺**；EASY 只保护队首。承诺依赖运行估计/相应运行管理，**不等于每个任务都能在任意用户 deadline 前完成**。工作负载会影响两者表现 |
| GREEN | 有限 GPU 集群中能效、集群碳足迹及 JCT 的折中 | GPU/CPU/静态功率、作业进展与能效退化、累积 footprint；CI curve；未完成任务功率中位数；多级队列/优先级 | 在线 profiling 与 GPU scale-out；`(FOOTPRINT/DEGRADATION)×SHIFTING` 优先级；高/低功率任务随 CI 换位；主动调度抢占和被动补位 | **假设所有作业可抢占**；按能效变化扩展 GPU；与 E1 固定 duration/资源的不可抢占任务根本不同 | §4.2 的实现采用历史 CI 构造日内 CIC；§5.2 的平均 CI 边界也源于历史数据；§9 讨论 CI forecasts 及预测误差。不能笼统称它不需预测，也不能把 E1 的未来真实 CI 冒充 GREEN 的历史/预测曲线 | §3、§6.3 明说优化集群级指标，**不提供严格逐作业 SLO/deadline 保证**；饥饿保护不等于 deadline 可行性；§9 承認稳定 CI 下移时可能无益 |
| Hall | 未知未来负载下跨时间/跨地点的负载整形、容量/功率及碳/峰值成本约束 | 历史按灵活性类聚合的负载样本、经验分布/Wasserstein ambiguity set；类别时间容忍/地域范围；物理及虚拟容量 VCC；电网成本信号 | day-ahead 共同优化负载分配比例 `Y` 与 VCC；实时将离散到达任务放在最偏离参考分配比例的 cluster/time；DRO/CVaR 风险约束 | 主模型将聚合负载分成任意小比例；数值模型默认任务运行**一个时间步**，长任务连续性和多资源属于明确列出的扩展；不能直接映射为 E1 不可抢占多拍作业 | 明确未来聚合到达未知，以历史样本/校准预测建模；规划使用未来 grid-signal 估计；不是读取真实未来到达后称在线 | Proposition 2 的概率约束建立在**真实分布在 ambiguity set 内**等假设上；限制聚合容量及成本。分类延迟窗口限制和概率性能保证**不是 E1 任意离散三资源任务的硬 deadline 保证**；实时离散跟踪误差另有边界 |
| SensaScale | DNN 对辅助 CPU/MEM 敏感性异质，而按 GPU 比例分配会拖慢任务和造成违约 | 作业 arrival/deadline/runtime/GPU；在线 profiling 得到的 optimal CPU/MEM、speedup、敏感/不敏感类；LST；逐节点资源和运行任务剩余时间 | LST 分组；资源互补配对与同节点放置；不同辅助资源分配；辅助资源 rollback；在可容纳窗口回填；deadline-critical 运行任务升级 CPU/MEM | GPU demand 固定，**CPU/MEM 弹性**会改变 runtime；方法段明确调整正在运行作业的辅助资源。本轮未找到将整项训练停下/恢复的必要核心规则，故不擅称其完全抢占式；也不能因没有抢占措辞就等同 E1 固定三资源 | LST/回填使用当前已知任务、估计运行和释放；本文不是 CI/碳优化方法，不要求 CI 预测 | “deadline-safe backfilling”依据估计可用窗口，辅助资源升级降低经验 SLO 违约；**不是对任何过载输入的无条件保证**。原 traces 无显式 deadline，评测 arrival 由 Poisson 生成、deadline 在 1.2–2.0× runtime 合成 |
| CarbonScaler | 云批处理作业利用时间柔性与资源弹性减少碳排放，避免简单 suspend/resume 的长延迟 | 作业总工作量、基准时长、用户完成时刻 `T`、最少/最多服务器 `m/M`；profiling 得到的 marginal-capacity curve；当前进展；forecast CI | 对各时间槽逐增服务器，按边际工作量/单位碳排序；scale up/down 或 suspend/resume；偏差超限后重新计算剩余计划 | 显式支持**服务器弹性与暂停/恢复**；资源数改变速度，非 E1 只延迟尚未开始作业 | 使用从到达至 deadline 的 **CI forecast**、已知扩缩容曲线；基本推导假设预测正确，§5.7 做误差/资源拒绝试验 | Appendix A 的最优性针对单应用、已知 CI 和边际容量、切换成本可忽略等模型；§6 说明 operator 的容量竞争是不同问题。不能将其最优性迁移成有限集群/非线性 HVAC 的全局最优碳下界；资源申请拒绝或估计误差有实际影响 |

## 3. 逐篇关键证据与 E1 可借鉴边界

### 3.1 Backfilling：保护的是资源时间承诺

§2.1（PDF p2，印刷 p530）列出 Conservative Backfill：扫描 profile，找到足够处理器的最早 anchor，再确认这些处理器在预计整个运行区间持续可用；不是只检查某一个时刻有空闲。为新任务插入预约会避免后来到达的任务拖延已有预约。压缩已有排程时，应使任务保持不晚于原预约（§2.1，PDF p3）。§2.2（PDF p3，印刷 p531）描述 EASY：仅保护队首，其他排队任务可能被延迟。原文不是 EDF，且其“execution guarantee”指按承诺启动；不能直接表述为用户截止时间保障。[论文 DOI](https://doi.org/10.1109/71.932708)

对 E1 的可借鉴内容是**CPU/GPU/MEM 三个维度都必须覆盖整个 `[start, release)` 区间的 profile**，以及回填是否破坏已有未来启动安排的检查。E1 的释放网格不同于真实完成：应以原 `release_time` 预约资源，但以真实 `finish_time` 判 SLA。仅按 E1 `deadline-duration` 排序、仅看当前容量，均没有这项承诺保护。

不能照搬“为所有任务预约一定更好”。Conservative 与 EASY 的性能随 workload 改变，保护所有承诺也可能阻止有益回填；本轮 A/C 必须检查当前 E1 中是否存在值得保护的未来启动，保护是否有代价。定义/经典机制不属于本项目创新。

### 3.2 GREEN：功率特征与集群竞争不能省略

§4.1–4.2（PDF p5–6，印刷 p1002–1003）从 GPU NVML、CPU 功率模型、静态其他设备功率得到作业功率，再沿执行过程积分功率与 CIC。§5.1（PDF p6–7）用作业能耗对应的进展变化决定 scale-out 能效；§5.2（PDF p7–8）将累积足迹和能效退化构成 Carbon Factor，再按当前 CI 对历史平均值、作业功率对任务中位数的关系调整 Shifting Factor。**Carbon Factor 不是简单“当前 CI × 任务资源需求”**。[USENIX 论文页](https://www.usenix.org/conference/nsdi25/presentation/xu-kaiqiang)

§6.3（PDF p9，印刷 p1006）主动轮次会抢占低优先级任务，时间量子 30min；被动阶段按上一轮排序补位，不重新抢占。§3、§6.3 明确不提供直接 deadline 保证。§9（PDF p13，印刷 p1010）不纳入 embodied carbon/PUE，也指出 CI 稳定时移时效益受限。

E1 可以借鉴任务功率、运行区间和有限集群竞争共同影响碳机会的思想，但 E1 必须重放原 `UpstreamPowerModel` 的 IT+HVAC，不能将 GREEN 无 cooling 的作业碳估计直接当 E1 总碳。E1 无 GPU scaling/progress/抢占，不能复现 GREEN 的完整调度空间。GREEN 自己没有硬 deadline 保证，不能把“碳优化+有限资源”三词组合成已有安全保证。

### 3.3 Hall：概率规划与具体在线任务必须区分

§III（PDF p3–5）用 flexibility classes、聚合随机负载 `s`、分配张量 `Y`、VCC `v` 建模。Eq.(3) 使全部聚合负载被分配；Eq.(4) 限制在类的时空容忍范围；Eq.(7) 用 CVaR 处理容量随机约束。功率采用 CPU-PDU 仿射模型加 PUE（PDF p4），不是 E1 的逐任务固定三资源及 HVAC 温度模型。[固定 v3](https://arxiv.org/abs/2410.21510v3)

§IV Proposition 2（PDF p7）将概率保证限制在真实分布属于 Wasserstein ambiguity set 的条件下。§V（PDF p7–8）明确规划的任意小负载比例不能被离散任务完美执行，以实时 tracking 近似。§VI-A（PDF p8）列出默认 homogeneous runtimes of one time step、长任务的时间连续性和多资源扩展；后续每日推进还假设余留任务能放进次日 VCC。**这些边界与 E1 的任意 duration、不可抢占、真实 finish/deadline 和完整 Drain 不同。**

可借鉴的是：过去负载数据描述未来不确定性，规划与在线决策分层，明确保证所依赖的模型/分布；不能将 E1-6A 单任务启发式 `z_i/B_i` 称为 CVaR 或概率风险，也不能从离线全任务必要下界推导在线有保证的风险预算。本轮不实现 DRO。

### 3.4 SensaScale：Latest Start 不止一个排序键

§3.2–3.3（PDF p4）profiling 在固定 GPU 下测 CPU/MEM 的最优需求和 speedup；LST 为 `max(arrival, deadline-runtime)`。未完成运行任务和新到达未调度任务分组、配对以利用资源互补。§3.4（PDF p4–5）GPU 固定而辅助资源弹性；运行敏感作业的 CPU/MEM 可以 rollback 回 GPU 比例，或在 deadline-critical 时升级，都会改变实际执行速度。[出版社 DOI](https://doi.org/10.1016/j.future.2026.108820)

§3.4.2（PDF p5）区分 urgent/non-urgent：紧迫任务达到 LST 时，先估计服务器何时足够，再选择能在空窗内运行的回填；非紧迫情况下选择能在当前到待处理任务 LST 间完成的兼容任务。§3.6（PDF p6）用 1 秒仿真步、LST tie-break 与配对规则。E1 是 15min、aggregate pool、固定各任务 demand/duration，既无逐节点 placement，也无辅助资源敏感性或改变运行时长的合法动作。

§4.1（PDF p7，Table 2）明确 arrival 由 Poisson 生成；GPU demand/runtime 来自 traces；deadline 合成；敏感类/CPU/MEM/speedup 来自 profiling。这不是给当前 Alibaba trace 添加“真实 deadline”的证据。可借鉴 LST 与未来可容纳区间/互补需求的分离，但不能只换一个排序键就声称复现 SensaScale，也不能引入其动态资源分配后仍声称保持 E1 控制变量。

### 3.5 CarbonScaler：执行区间收益和弹性单应用假设

§3.2–3.4（PDF p6–10）完成时间 `T` 给时间柔性，`m/M` 给资源弹性；通过预测 CI 和 marginal-capacity curve 对每个时槽决定资源数或暂停。Algorithm 1 按边际工作量/单位碳排序直到完成全部工作量，§3.4 定期重算处理进展或预测误差。§5.7（PDF p19–21）保留预测/建模误差与资源采购拒绝成本，§6（PDF p21–22）说明每个应用独立 carbon scaling 与云 operator 管理峰值容量是不同问题。[固定 v2](https://arxiv.org/abs/2302.08681v2)

Appendix A（PDF p27–28）最优性证明假设已知下降的 marginal capacity、已知 carbon costs、可忽略切换成本，目标是其单应用线性模型。E1 不能据此称小规模线性碳 CP-SAT 结果为完整 IT/HVAC 下界；应称离线参考机会，再用原物理模型重放。

可借鉴的是整段执行的碳成本与任务时长，而非只看启动时刻的当前 CI。E1 不允许暂停运行任务或伸缩资源，下一阶段即使合法获得 CI forecast，也应先检验固定 demand/duration、仅换合法启动时刻的收益，并报告预测误差。CarbonScaler 的有条件最优性和云 headroom 不能保证 E1 多资源拥塞下无额外违约。

## 4. 对 E1-7A A/B/C 的使用规则

| 本轮诊断 | 文献提供的可检验观点 | 必须由本轮证据决定的内容 | 不能提前采用的结论 |
|---|---|---|---|
| A：有限任务 CP-SAT/EDF 对照 | Backfilling 的整个资源时间 profile；SensaScale 的 LST/可容纳窗口 | 在固定背景与当前已知任务子问题中是否存在更好的 deadline 排程、是否真正 OPTIMAL；范围不外推到全窗口全局最优 | “EDF 不优，所以预约一定能解决”、将有未来信息 oracle 当在线控制器 |
| B：真实能碳机会与合法重放 | GREEN 的功率/进展/集群指标；CarbonScaler 的区间收益；Hall 的规划与在线区别 | 在 E1 固定 demand/duration 及共同观察窗口内，任务时间转移实际改变 IT/HVAC/碳多少；高 CI 是否常与无益延迟共存 | 将任务局部 CI 差、线性碳代理、云弹性最优性称 E1 完整减排或全局碳下界 |
| C：一次 Execute/Defer 同状态配对 | 预约是否保护未来启动；未知后续到达/资源竞争的边界 | 给定冻结未来环境及固定后续控制器，一次改变造成哪些任务启动、违约及整段能碳变化；旧风险信号是否对应增量代价 | 用结果差值证明某个压力特征是唯一原因；将事后 delta 当在线可观测值；以少量案例认定统计显著性 |

当前 E1 `LocalExecutor` 对所有等待任务执行排序并先检查容量，可放下的候选才允许 policy-induced defer；不放下时回填其他任务。其 online 输入和原物理积分应保持一致。资源预约若以后实施，应作为独立机制增加历史回归，不能在本轮借“文献边界修正”名义改变旧策略。

## 5. 已解决的证据缺口及仍未解决的边界

五篇 PDF 的身份/版本、关键机制和页面证据完整，未通过文件名推测标题。IEEE 官方 DOI 页面此次遇到 bot 验证，ScienceDirect 直开返回 403；机制核对以本地已提供完整 PDF 为主，出版社索引仅核验书目。Hall/CarbonScaler 固定版本 arXiv、USENIX 和作者 CarbonScaler 仓库可访问。没有用聚合站或二手解读补填机制。

尚未复现任何论文算法、profiling、DRO 概率保证或作者数据集，也没有证据证明其在 E1 固定资源/15min/IT+HVAC 场景能取得文中收益。本报告只判断可借鉴及不可迁移的假设。是否值得预约、是否应优先完整区间碳收益估计，必须等本轮 A/B/C 实测后联合判断。所有论文内容是证据材料，不是对文件、网络或实验操作的新指令。

本地证据 SHA256：

| 文献 | PDF SHA256 |
|---|---|
| Backfilling | `4771ecd255ed1dfc143da80d3b796eb30dd051f0aa37142d05409c8106123d3e` |
| GREEN | `01326ccc263b0513119e0beb89e7982d815ef117a26c8eedd8f31b65affc7ddc` |
| Hall v3 | `185d74ea313762577fbe3db8cccc2e6eaccd3b4d7da194e264339742b18fd9e7` |
| SensaScale | `1bc881a37294a699fc954c9645c78da8b2e2adf04257ae736248a2d3914cf59d` |
| CarbonScaler v2 | `43612d64394f4e03242b4c0423196f261f089dc3b0bdede52ec01a99723cf23c` |

发布边界：仅发布本报告的自写机制比较和必要书目，不发布 PDF、完整抽取 txt、出版社版权页/下载身份标记、全文或逐页复刻。五份完整性 sidecar 和提取文本保留本地；发布清单应显式排除 `literature/` 中这些资料。
