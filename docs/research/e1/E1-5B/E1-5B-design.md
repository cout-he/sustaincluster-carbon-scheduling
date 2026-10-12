# E1-5B 普通 PPO-v0 设计与预注册

本阶段仅训练已有 40 维观测、3 个全局模式动作的普通 PPO。原 `engine.py`、`rl_env.py`、三种动作的调度规则、Reward 和物理公式全部按 E1-5A 字节保护。Test 仅引用既有覆盖审计，不生成快照、不训练、不评测。

新增 `rl_training_utils.py` 管理 Train/Validation 冻结包和禁止 Test 的入口，`rl_train.py` 管理 SB3 参数更新、保存和日志，`rl_evaluate.py` 管理完整 Drain、公平公共终点和模型选择。模拟器私有冻结输入不进入 actor；actor 只收到当前 40 维向量。选择成熟 SB3 而非自行实现 PPO，避免新增算法正确性问题。既有训练环境保持不动，正式评价复用执行器并补齐真实空闲尾段。

配置 `experiments/e1/configs/e1_5b_ppo.json` 在任何 Train 策略实验之前落盘。阈值 161.075 来自 4 月的 720 个小时 CI 中位数；不在 Validation/Test 拟合。默认 Reward、工程归一化保持 E1-5A。负载只使用预先声明的 1/1.5/2×，原 E1-2B 嵌套复制，不缩放单任务。选用出现容量等待且模式结果不同的最高已声明 Train 档；若所有档无信号，停止 PPO 并报告。

所有 RL seed 共用选定档 Train stress seed 510001；Validation 使用 520001。其余 stress seeds 仅保留，不将不同输入与 RL 初始化差异混淆。RL seeds 55101/55102/55103；smoke 55100。先 1024 步 smoke，再 20480 步 pilot，若有限数值、参数更新、墙钟 <=7200 秒且峰值 RSS <=8 GiB，另两 seed 同为 20480 步；pilot 55101 本身也是同预算候选，避免重复挑选。

采用 CPU、单 Torch 线程、确定性算法、SB3 2.9.0、MlpPolicy [64,64]。现有环境未安装 SB3，新增包用 `--no-deps --target` 隔离到本地 E1-5B/runtime-deps；不升级其他依赖。完整 episode 使用真实 Drain，30 天安全界限；学习步预算可能停在未结束 episode 中，该未完成 rollout 记录为采样预算尾部，不宣称完成工作量或减排。

正式 Validation 同一有限队列比较 EDF、EDF+Carbon、RiskOnly 及所有三个 PPO 候选，先各自排空，再将同一组公共终点设为最慢排空时刻，原物理模型零任务负载连续积分。未排空则公共终点为安全界限，结果保留删失，不作等工作量减排结论。

在看到任何候选表现前固定资格条件：全部完成且 DMR <=5%。仅在合格候选中按公共窗口碳排放、DMR、seed、checkpoint hash 排序；全部不合格则明确不选模型。5% 是本轮操作性筛选标准，不能视作安全保证或由结果反推的阈值。所有差模型保留。

报告真实原始奖励项尺度、动作退化、按当前 CI/压力/紧迫程度分层动作、40 特征有限性与饱和、不同 seed 变异。仅一个 Train 和一个 Validation 时间窗口，不做显著性或泛化主张。完整模型和日志只留本地，GitHub 上传新增代码、配置、测试、精简报告与小 CSV，旧研究档案不修改。
