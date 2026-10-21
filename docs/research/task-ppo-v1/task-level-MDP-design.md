# Task-PPO V1：任务级联合动作设计与实施计划

2026-10-21。基线提交 `865f1ef05cd3f663fe4e0392790854cbff4e9466`，分支 `research/task-ppo-v1`。本轮遵照用户授权连续完成实现与一次pilot；设计和全部数值配置在真实性能实验前提交，不根据结果改动。

## 1. 决策单位与边界

保留冻结V2、E0 EDF日历、DDF、CG、Observed×1任务、SLA1.5、20°C请求与完整72h责任。不改旧7动作实现，其冻结模型仅作历史对照。

物理时间步始终15min。槽开始冻结真实当前状态及无主动推迟的EDF参考计划，候选仅为该计划本槽启动任务，按 `(deadline,arrival,task_id)` 固定顺序全部处理，不补位。每任务采样二元原提议：0不主动推迟，1申请当前槽推迟；0不保证一定启动，真实容量与回填仍由V2决定。

同槽内部只更新已批准推迟prefix，不释放真实资源、不推进时间、不新增任务或天气观察。申请1时复用旧EDF日历对 `approved+[task_id]` 仅模拟一次，以固定参考按时集合、原迟到总超期、原EPS和CG执行原DDF三条件的等价增量判定；原始请求顺序另外完整保存。所有候选处理完后，调用原 `ResearchAdapter.decide(obs,proposed=全部原始1请求)`，断言其最终批准集与prefix完全一致，再用原 `execute`执行一次真实15min。内部重复核验只算计算开销，DDF尝试/拒绝按最终原始请求逐个计一次。

保留严格参考按时集合、原迟到总超期与3960min完成护栏。参考CG失败则锁存回退EDF，不伪造采样；继续完整72h保存失败账本并停止后续训练。未知未来到达的Deadline不在DDF保证内。

## 2. 48维共享网络输入

使用旧因果27维全局状态和固定物理尺度，不从留出估计归一化。附加当前任务12维：CPU/GPU/MEM需求÷各自容量，Duration÷1440，等待年龄÷1440，剩余Deadline÷1440，个人Slack÷1440，动态IT功率÷满逻辑容量动态功率，参考剩余完成时间÷1440，网格运行时间÷1440，原arrival÷4320，Deadline÷4320。`Pdyn=(0.005CPU+0.225GPU)/0.94 kW`，只是冻结模型代理；需求仍是观测使用量代理。

8维prefix状态：候选数÷1000、当前位置÷候选数、累计批准数÷候选数、批准CPU/GPU/MEM需求÷容量、批准GPU网格资源小时÷(GPU容量×24h)、累计拒绝数÷候选数。最后1维query标志：actor=1。canonical critic query仅含槽开始27维全局状态，附加21维全0；因此value不依赖本槽采样动作。

共享64×64 Tanh backbone、二分类actor head及标量critic head；没有任务ID嵌入、注意力、循环网络、预测器或新增风险模块。观察仅来自当前/历史CI、当前天气、已到达任务和已知任务日历预测；它仍是聚合部分观测，未证明严格Markov充分。

## 3. 联合概率与PPO

设槽t有n个候选，原提议向量为b，已批准prefix是原提议及固定DDF的确定性函数。共享actor的条件概率组成自回归联合分布：

`log pi(b|s) = sum_j log pi(b_j | task_j, global_t, approved_prefix_j)`。

包括0以及被DDF拒绝的1；绝不改成批准动作的概率。更新时对已保存原路径teacher forcing，不重新采样或重新安排prefix。联合log求和与ratio使用float64；不取平均log，不按候选数开根号。

每物理槽仅一个 `rho_t=exp(log_pi_new-log_pi_old)`、一个clip目标、一个全局value与实际reward。沿完整288槽计算旧 `compute_gae(gamma=1,lambda_gae=.95)`，真正终止bootstrap0。候选空集或CG强制槽不进入actor，仍保留critic/GAE/奖励；有候选全选0或全被拒仍是正常采样。

优化器沿用lr3e-4、Adam eps1e-5、clip .2、4epochs、minibatch64物理槽、value_coef .5、grad clip .5。为避免把旧prefix下的条件熵和误称精确联合熵，本次先验固定 `entropy_coef=0`；条件熵仅作诊断，不增加熵估计模块。该超参数及动作/状态/容量训练分布与旧E2不同，因此旧模型不是严格单变量消融。梯度有限性、joint ratio/clip fraction和队列大小均记录，数值异常即停止。

实际奖励严格沿用 `-(C_actual_t-C_common_idle_t)/100 - .01*N_new_irreversible_miss_t`，每ID一次；不发中间任务奖励、不重复惩罚、不用DDF预测替代真实miss。内部任务选择不是物理轨迹步。

PPO与GAE依据：[PPO原论文](https://arxiv.org/abs/1707.06347)、[GAE原论文](https://arxiv.org/abs/1506.02438)。联合自回归分解及共享基线是本实现设计，不将原文结果迁移为本平台收益保证。

## 4. 预定pilot与公平对照

复用E2已审计16个Train基础窗及其身份哈希：Jan–Mar12窗训练，Apr–May4窗内部留出，按5个原始时间连接块隔离。只读白名单；不读封存数据、不去重。OM和OH为冻结容量，每种16个训练episode，共32episode/9216物理step；唯一seed20261021。每容量以该seed生成独立预声明12窗打乱序列并轮转，两个容量逐episode交替。具体完整序列写入机器配置后提交，训练中不更改。

4留出窗×OM/OH均比较EDF-CG、CarbonRule-DDF-CG、Task-PPO-initial、Task-PPO-final、旧7动作PPO历史模型。保存initial后先评价，再恢复初始RNG/optimizer开始训练；final固定用第32回合模型，无留出模型选择或调参。旧PPO仅在OM训练过，应用OH需明确迁移和历史性质，不冒充同预算消融。每组使用同一72h任务/CI/天气、设备、固定HVAC请求和空闲背景。

只因工程或责任正确性阻塞才停，不因得分差改场景。所有CG失败、未完成、censored、尾失效、末温不公平、NaN/Inf、无权重更新或checkpoint不完整，保留当时真实账本并停止后续训练。不重选seed或重跑寻找好结果。

## 5. 实施计划与审阅重点

技术栈为既有Python3.14/PyTorch2.11 CPU单线程，无新增依赖。复用干净独立worktree `sca-ppo-v02`，只新建分支；旧分支和tag保持。新增文件限于 `experiments/task_ppo/`、`tests/task_ppo/`、`docs/research/task-ppo-v1/`。

- [x] `joint_ppo.py`：复用旧ActorCritic/GAE/checkpoint接口；新joint batch、物理槽clip、真实反传。人工枚举2候选联合概率、一次GAE、无候选、非法batch、完整checkpoint恢复先RED后GREEN。
- [x] `task_environment.py`：48维因果编码、同槽prefix选择、原DDF/CG与一次执行、旧RewardLedger和finish账本。人工联合冲突、回填、CG锁存、全0EDF一致、时间不前进与无未来读取测试。
- [x] `pilot.py`与机器配置：加载既有授权Train、OM/OH固定序列、initial/final和历史对照、奖励/资源/责任/末态验证、一次性marker、完整checkpoint和大日志仅本地忽略目录；小型合成全回合反传与保存测试。
- [x] 提交设计、配置、测试及全部运行源码后，完成预注册的32回合训练pilot；另行完整保留001训练前工程中断及002等价继续证据，未重选配置或训练种子。
- [x] `evidence.py`、报告与图：全部回合、任务加权DMR/JCT、真实碳与增量碳、任务动作/拒绝率；同槽概率差和任务属性关系区分“具备表达能力”和“已学会有效区分”。独立只读复核已完成，本交付提交推送后停止。

重点审阅5类边界：①CG/空候选不伪造actor样本；②全0/全拒绝仍保留原logprob；③联合批准不等于实际主动等待；④变长候选仅影响joint概率而不复制物理时间/GAE；⑤初始实现曾重复核验已批准前缀；修订为O(n)个日历调用并保留原DDF完整复核，不删候选或换条件。


## 工程修订：保留中断尝试，等价消除冗余

原提交7ad4ab1的首次训练前评价，在首OM场景第20个完整槽出现297候选和55.39秒单槽耗时。原DDF重复核验已批准prefix造成二次日历调用；没有进行任何训练更新。保留pilot-001全部32个背景、3个完整旧方法评价和20个持久化真实槽，中断时尚在纯候选计算，未将不完整episode伪称完成；未捕获的内存prefix不补造。

修订只缓存最初reference的on_time/late集合和总超期，新增候选仅模拟一次联合trial，按原理由顺序、原EPS执行相同三条件；槽末仍调用原ResearchAdapter/DDF全量重放逐项核对。人工枚举与调用计数验证等价。未改变任何状态特征、动作、奖励、hyperparameter、场景、顺序或seed；不是依据分数重新选配置。

工程继续运行保存为pilot-002，独立marker；pilot-001旧marker保持为interrupted_engineering且不删除。继续前固定旧初始checkpoint与20行原决策哈希，首同场景必须逐字段复现其taskID/48维状态/原动作/logprob/批准集/实际row及奖励（仅排除耗时、内部重算计数），且初始模型fingerprint相等，才允许训练。旧尝试不进入主矩阵均值，但其额外计算、完整与部分账本全部单列。
