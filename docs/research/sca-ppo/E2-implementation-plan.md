# E2最小准入与首次PPO pilot实施计划

基线`8e50b86fa2ea8ca5797605dc29318e5315a144be`，保持独立worktree和`research/sca-ppo-v0.2`。冻结设计位于原`object-model-a1/research_notes/sca-ppo-review/`，已核对；设计固定27维聚合观察、7动作、64×64 MLP、DDF/CG、增量碳及逐ID不可逆miss、gamma=1。具体PPO数值超参数及尺度在设计中明确留给E2预注册，本轮只选一套，不搜索。

1. 仅审计16个`train-...-r1-s1.5`基础窗口，保留原始身份与重复，不读取其他角色；按连续时间和原始内容隔离Train内部训练/留出，记录共享环境区间。新`e2_data.py`和人工测试。
2. 固定状态schema及物理归一化（不利用留出统计）；实现`ppo.py`的真实PyTorch actor-critic、gamma1/独立GAE lambda、带实际采样标志的策略更新、完整288槽价值训练、完整checkpoint及RNG/optimizer恢复。
3. `rl_environment.py`复用E0 ResearchAdapter。只从当前及历史观察编码；奖励仅用实际step碳减同槽公共空闲背景，和新不可逆miss ID。背景不作为观察，失败回合完整记账并停止进一步更新/训练，不改奖励或删回合。
4. 人工测试动作映射、无未来信息、奖励逐ID守恒、终止与bootstrap、反向传播及checkpoint一致性；输出E2-0准入报告并提交固定配置/源码后，才启动真实pilot。
5. 单seed、最多32个完整72h训练episode（9216训练step）；OM主容量；训练序列按固定训练组循环，无搜索。保存训练前与训练后确定性策略评估，使用同一Train内部留出组、OM、相同三方法及公共背景；留出不参与尺度/模型/超参数选择。
6. 所有检查通过则完成一次pilot及两份报告/轻量指标/曲线，checkpoint和大日志仅本地忽略目录。CG/未完成/censored/尾失效/NaN/动作或奖励失配即保存证据、停止后续训练，不重选seed/窗口。
7. 独立复核与受控测试后提交推送。本轮不做多seed正式训练或新增算法模块。

实施接口：`e2_data.audit_pool`给出固定train_ids/holdout_ids及输入哈希；`ppo.ActorCritic/PPOTrainer`仅接收数值轨迹；`rl_environment`负责27维当前观察、E0动作与实际奖励；pilot驱动器负责固定输入、背景、日志、一次性运行和停机，不重复实现V2调度器。
