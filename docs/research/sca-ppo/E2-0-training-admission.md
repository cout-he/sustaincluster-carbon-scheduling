# E2-0：Train 数据与首次 PPO pilot 准入

日期：2026-10-21（Asia/Shanghai）。**结论：允许按本报告与已固定配置进入一次受限技术 pilot。E2 真实训练尚未运行；本文不报告训练回报、策略收益或泛化结果。**

本阶段承接 [E1 规则实验报告](/D:/A-myprofile/SustainCluster/sca-ppo-v02/docs/research/sca-ppo/E1-Train-rule-experiments.md) 的有条件继续判断。数据准入及人工工程测试通过，只说明该 pilot 有可审计的输入与执行边界，不证明 PPO 优于规则，也不构成多 seed 正式实验的完成证据。

## 1. 授权范围与输入身份

本次只审计明确授权的 16 个 Train 基础窗口，每个日期仅选择 `r1-s1.5`。场景目录固定为 `D:/A-myprofile/SustainCluster/publication/sustaincluster-carbon-scheduling-001/research_notes/joint_v2/scenarios-001/scenarios`。其他窗口只可按 `train-` 目录名称识别可用性，不读取其他角色的 manifest 或 payload；不读取封存 Development/Validation/Test、年度 PKL 或原始 CSV。

[准入代码](/D:/A-myprofile/SustainCluster/sca-ppo-v02/experiments/sca_ppo/e2_data.py) 的 `load_authorized` 在 payload 访问前检查固定父目录、16 个 ID 白名单、manifest 的 Train 角色、`test_access=False` 和目录/场景 ID 一致性。随后调用原角色受控 `load_train(..., hours=72)`，验证文件及内容哈希、24h 到达队列、legacy ×5 映射与 Observed ×1 需求；另核验 ID 所指 UTC 日期、稳定 TaskID、逐任务身份摘要和预期 provenance。`inputs` 中保存每个窗口的 manifest/workload/environment 哈希、源哈希、时间基准、任务数和任务身份摘要。原任务、原始字段、Duration、Deadline、资源需求和源身份均保留，不做去重或任务变换。

[机器审计结果](/D:/A-myprofile/SustainCluster/sca-ppo-v02/docs/research/sca-ppo/e2-results/data-admission.json) 显示：16/16 窗口可用，共 **52,339 个任务**；每窗保留 24h 到达、连续 288 个 15min 环境样本及完整 72h 责任窗口。环境观察采用原因果上一小时保持规则。全部任务单独适配 OM/OH/OL 的静态容量；这不等于队列调度可行性证明。本次 pilot 仅使用 OM。

## 2. 预声明分组及重复审计

在计算任何策略性能前固定：1–3 月作为训练，4–5 月作为 Train 内部留出。分组先连接 72h 环境输入重叠或共享原始 `job/start/end` 身份的窗口，再对连接分量取时间连续闭包；任何跨越边界的整块归入留出，禁止拆散共享输入块。实际得到下列 5 块，未触发跨边界合并。

| 连续块 | 窗口 UTC 起始日期（均为 05:00） | 环境覆盖区间，右端不含 | 分组 |
|---|---|---|---|
| 1 | 01-02、01-03、01-04、01-05 | 01-02 05:00 至 01-08 05:00 | 训练，4 窗 |
| 2 | 02-01、02-02、02-03、02-04 | 02-01 05:00 至 02-07 05:00 | 训练，4 窗 |
| 3 | 03-01、03-02、03-03、03-04 | 03-01 05:00 至 03-07 05:00 | 训练，4 窗 |
| 4 | 04-03、04-04 | 04-03 05:00 至 04-07 05:00 | 留出，2 窗 |
| 5 | 05-01、05-03 | 05-01 05:00 至 05-06 05:00 | 留出，2 窗 |

日期均属 2023 年。训练组为 **12 窗、37,862 个任务**；留出组为 **4 窗、14,477 个任务**。跨训练/留出的环境重叠分钟、共享完整原始内容、共享原始 `job/start/end` 身份及共享稳定 TaskID 均为 **0**。配置中的 `train_ids`、`holdout_ids` 和 `inputs` 必须与已哈希锁定的 admission 文件逐项一致；仅检查两组互斥不足以通过准入。

全量稳定 TaskID 和完整 12 字段原始内容的重复超额均为 0；原始 `job/start/end` 身份有 **1,309 组二重出现**，涉及 2,618 行、重复超额 1,309。单独按 job 名计有 2,013 个重复超额，但 job 名本身不是任务唯一身份。以上观察全部保留；不能将源观测中相同 job/start/end 的行擅自合并。

16 窗不等于 16 个独立环境样本，5 个连续块也不是从独立年度采集的 5 份数据。任务来自同一已处理 2020 trace，并按既有规则对齐 2023 环境时间轴；数据隔离不证明真实现场共时性或跨年度泛化。4 个留出窗口仍属授权 Train，其中部分已用于既有平台/E1 开发；它们不是未接触的封存 Validation/Test。重复训练曝光、容量变体和随机 seed 不能增加独立源样本数。本阶段只可据此判断工程可训练性及有限开发条件下的策略行为。

## 3. 冻结设计与 E2 数值预注册

算法依据是 [SCA-PPO v0.2-final 设计](/D:/A-myprofile/SustainCluster/object-model-a1/research_notes/sca-ppo-review/SCA-PPO-v0.2-final-design.md)。冻结 V2 的 annotated tag 解引用提交为 `e372f502fb58aaa5a742e8efb4ced254507ab298`；E1 基线提交为 `8e50b86fa2ea8ca5797605dc29318e5315a144be`。平台、E0 日历/DDF/CG 与执行适配器的受保护源码按既有协议哈希核对，不因结果调整任务、容量、Deadline 或物理参数。所有方法请求 20°C，原 V2 热安全器仍可覆盖。

原设计固定了小型 PPO、两层 64 单元 MLP、约 27 个聚合标量、7 动作、完整 DDF/CG、实际增量碳与逐 ID 不可逆 miss、`gamma=1`；具体优化器数值、训练预算及奖励尺度明确留待 E2 预注册。下列值是本次在结果产生前选定的一套工程起点，不是已优化的参数或经过留出标定的物理常数。完整数值以 [E2-pilot-config.json](/D:/A-myprofile/SustainCluster/sca-ppo-v02/docs/research/sca-ppo/E2-pilot-config.json) 为准。

| 项目 | 本次固定值 |
|---|---|
| 网络与状态 | 两层 `64×64` Tanh，共享骨干、categorical actor 与标量 critic；27 维，schema `sca-ppo-e2-state-v1` |
| 折扣与优势估计 | `gamma=1.0`；独立的 `lambda_gae=0.95` |
| Adam | learning rate `3e-4`，epsilon `1e-5` |
| PPO 更新 | clip `0.2`，每回合 4 epochs，minibatch 64；entropy coefficient `0.01`，value coefficient `0.5`，gradient norm 上限 `0.5` |
| 奖励 | `C_scale=100 kg`，`lambda_miss=0.01` |
| 执行与种子 | CPU，PyTorch 1 线程，确定性算法；唯一 seed `20261020`，seeded `torch.nn.Linear` 默认初始化 |
| 时间与预算 | 15min/step；24h 到达，72h/episode，66h 清空约束；32 个训练 episode，最多 `32×288=9,216` 训练 step |
| 抽样与选择 | 预先固定的 32 项训练序列，仅含 12 个训练窗口；不搜索 seed/超参数，不根据留出结果挑 checkpoint |

27 维状态由时间/阶段 5 维、当前与历史 CI 5 维、资源占用与下一槽释放 5 维、队列/候选/工作量/动态功率 5 维、参考 slack/迟到比例 4 维、当前室内外温度 2 维和失败锁存 1 维组成。编码与物理尺度见 [rl_environment.py](/D:/A-myprofile/SustainCluster/sca-ppo-v02/experiments/sca_ppo/rl_environment.py)。仅使用当前/历史公开观察及已到达任务的参考尾日历；不读取未来到达、未来 CI/天气或未来 idle 轨迹作为策略状态。状态是否 Markov 充分尚未证实。

7 动作为：0 不主动推迟；1/2/3 按高动态 IT 功率排序尝试前 1/4/16 个；4/5/6 按高个人 slack 排序尝试前 1/4/16 个。候选是同槽参考 EDF 实际启动集，排序在槽开始固定，拒绝不补位。原提议动作及原 logprob 与 DDF 批准集、真实启动/等待分类分开记录；不加动作 mask、预测器或新奖励项。

DDF 对全部已知任务的联合反事实要求：参考按时 ID 集合不减少，原迟到 ID 集合总超期不增加，且最大预测网格 release 不超过 3,960min。CG 失败锁存至回合结束。该局部参考保护不保证未知未来到达下的全程 DMR 非恶化或任务集全局可行。

每槽奖励固定为 `−(C_actual,t−C_idle,t)/100−0.01×N_new_irreversible_miss,t`；每个 ID 最多处罚一次，不用 EDF 预测迟到冒充真实不可逆事件。实际槽执行后才读取同槽公共空闲碳。该偏好下额外一次 miss 相当于 1kg 增量碳的奖励代价，是服务权衡而非校准常数或 DMR 硬约束。有效全完成回合核验奖励总和、碳积分及 V2 正式 miss ID 守恒。所有 288 个物理槽进入 critic/GAE；只有真实采样的提议动作进入 actor loss/entropy，强制 No-op 不伪造采样。

## 4. 评价、checkpoint 与失败停止

[pilot 驱动器](/D:/A-myprofile/SustainCluster/sca-ppo-v02/experiments/sca_ppo/e2_pilot.py) 在任何 optimizer 更新前保存初始 checkpoint，并对固定 4 窗留出执行 EDF-CG、CarbonRule-DDF-CG 和 PPO-initial 的确定性评价。初始评价后恢复初始 checkpoint 的全部 RNG/optimizer 状态再训练。训练结束固定使用最终第 32 回合 checkpoint 进行 PPO-final 确定性评价，不按留出分数选择中间模型。初始/最终评价和公共背景不会进入训练 batch 或尺度估计。

完整成功执行的预定账本为：训练最多 9,216 step；4 窗×4 个评价标签×288，即 4,608 个评价 step；16 窗公共空闲背景×288，即 4,608 个背景 step，后两类另计，不冒充训练样本或独立数据。共同背景只为空间/设施与外界序列一致的物理参照。配对方法核验 IT 能量守恒及末温差不超过 0.05K，不能因设置共同 6h 尾段就假定末态必然一致。这些是预定预算和门禁，尚无实际 E2 执行计数。

[PPO 实现](/D:/A-myprofile/SustainCluster/sca-ppo-v02/experiments/sca_ppo/ppo.py) 每回合保存模型、Adam 状态、更新/optimizer 计数、配置与元数据、Python/NumPy/PyTorch RNG 和确定性设置，采用临时文件写入后原子替换。保存后重新加载，核对模型 fingerprint 和计数；恢复接口保留完整训练状态。`--evaluate-checkpoint` 提供独立冻结策略评价入口，只评价不更新；本轮不会为此额外调用重评。支持状态恢复不代表失败 pilot 可以自动续训。

若 CG 参考失败，仍以无 defer EDF 完整执行至 72h，保留全部任务、真实 step 和失败标记。72h 未完成、censored、尾段无效或保存结果独立核验失败的回合，均保留结果及核验错误，并停止后续 PPO 更新/训练；不删除失败回合、不换窗口/seed、不补奖励或强改 miss。NaN/Inf、动作/预演/奖励失配、网络未更新或 checkpoint 不完整也停止；异常发生在回合中途时保存已实际执行的部分账本，不伪称已完成 288 槽。失败后须重新审查责任/终局成本处理，不能自动重试。

## 5. 人工验证与读取边界的证据限度

主执行者已再次运行受控人工测试：`tests/sca_ppo` **108 项**、显式 V1/V2 模块 **32 项**，合计 **140 项全部通过**。这些测试使用人工内存场景或工作区合成快照；不运行真实调度性能实验或原 V2 验收矩阵。覆盖输入角色/父目录/日期/哈希与分组绑定、未来信息隔离、逐 ID 奖励与实际执行守恒、CG 锁存/失败保留、全槽 GAE 与真正终止/bootstrap、真实反向更新，以及 checkpoint/RNG/optimizer 恢复。

独立只读复核发现的两处问题已先加 RED 回归再修复：训练/留出及 inputs 必须绑定 admission 内容；保存结果 verifier 失败时统一可追加的 failure reasons 并保留原 `verification.json` 错误，避免 tuple `.append` 异常掩盖核验原因。4 项 pilot 人工集成测试通过，已计入上述 108 项。

另有一次不应扩大的历史命令 `python -m unittest discover -s tests -p 'test_*.py'`：系统 Python 3.14 的输出显示 109 项、13 个失败和 21 个旧模块导入错误，相关缺失依赖包括 `gymnasium`/`tqdm`；其中 13 个失败发生于并行实现尚未提供 PPO core 时，不能当作当前受控测试结果。完整输出被工具截断且未保存，不能逐项重建全部 21 个 traceback，也不能声称所有错误只有同一个根因。此命令不再执行，后续测试严格限定上述目录/显式模块。

事后只读源码/AST 检查表明：旧 `tests/e1` 是 pytest 函数测试，没有 `unittest.TestCase`，该命令只导入旧模块而未执行其 pytest 函数；被检查的测试模块没有顶层年度/封存 workload/environment/原始 CSV 读取，实际 unittest 类位于 SCA 与 V1/V2，所查 V1/V2 使用合成场景。因此源码证据支持未读取封存数据，但**当时没有文件访问跟踪，不能把事后静态检查说成完整系统调用审计或绝对的间接读取证明**。已知输出与上述限制保存在 [本地 scope note](/D:/A-myprofile/SustainCluster/sca-ppo-v02/research_notes/sca-ppo/e2-discovery-scope-note.json)；完整旧终端输出不存在，不作补造。

## 6. 复现入口与执行状态

以下在 `D:/A-myprofile/SustainCluster/sca-ppo-v02`、已安装本项目 PyTorch/NumPy 依赖的核验解释器环境执行；`python` 应解析到该解释器，不能将旧系统 Python 缺包错误视为当前科学验证。

```powershell
Set-Location D:/A-myprofile/SustainCluster/sca-ppo-v02

# 仅受控人工测试
python -m unittest discover -s tests/sca_ppo -t . -q
python -m unittest tests.platform_v1.test_platform tests.platform_v1.test_saved_verification tests.platform_v2.test_platform tests.platform_v2.test_verification tests.platform_v2.test_acceptance -q

# 仅16个白名单Train的数据审计；不调度、不训练
python -m experiments.sca_ppo.e2_data --pool D:/A-myprofile/SustainCluster/publication/sustaincluster-carbon-scheduling-001/research_notes/joint_v2/scenarios-001/scenarios --output docs/research/sca-ppo/e2-results/data-admission.json

# 预注册文件/源码提交并通过门禁后，真实pilot的一次性入口；本报告未执行此行
python -m experiments.sca_ppo.e2_pilot --output research_notes/sca-ppo/e2-pilot-001
```

驱动器要求干净 Git 状态，核对受保护源码与 admission 身份，并创建排他的一次性 marker。不得删除 marker 冒充首次运行。checkpoint、逐槽/逐任务大日志和执行文件只保存在被忽略的 `research_notes/sca-ppo`；提交范围包括准入结果、固定配置、源码、人工测试及本报告。实施顺序见 [E2 计划](/D:/A-myprofile/SustainCluster/sca-ppo-v02/docs/research/sca-ppo/E2-implementation-plan.md)。后续 E2 执行报告须独立呈现实际步骤、失败、训练开销及固定留出结果，允许 PPO 无增益；本文不预写这些结果。
