# 54维/三动作独立Gym接口

`CarbonGainSchedulingEnv` schema=e1-7b-cgd-rl-v1，Discrete(3)，原40维/3动作与83维/4动作接口、checkpoint和旧训练入口保持原样。一个inner CarbonSchedulingEnv复用旧Observation/Reward和唯一LocalExecutor；真实非抢占资源状态与物理模型跨拍保留，reset才创建新模型。

Action0不主动延迟；Action1要求M2 Gain>P75；Action2要求Gain>0。两者相同Deadline保护；不可fit者由公共执行器记capacity_wait，不伪造启动。收益不可用执行并记录status。动作仅当拍有效，每次至多主动延迟15min。

基础40维单位/尺度沿用CI1000g、任务数1000、slack120min、资源work120min、时间24h。新增14维依次为预测1h均值、2h均值、2h趋势、Gain均值/p25/p50/p75、正收益/超P75候选数、估计不可用比例、保护允许比例、有效候选数、CI冷启动flag、gain presence。CI用1000、Gain用0.1kg、counts用1000，连续标量x/(scale+abs(x))，比例0..1；空统计为0，presence区分缺失。invalid_fraction不把保护拒绝当预测失败。终态新增块全0，不读下一CI。

Observation preview只使用当时已到达、可fit候选，发生在EDF遍历前；真实决定会看到之前真启动形成的新背景。日志明确此区别。原始/归一Reward来自实际碳、pending*15任务分钟、首次违约、截断剩余负债；同任务只罚一次。既有PPO的γ=0.99回报折扣、等待尺度和可变Drain终点仍可能偏离公共窗口目标；RewardConfig.gamma=10是deadline惩罚权重，与PPO折扣不同。本轮不调整这些参数、不以累计Reward替代论文指标。

真实Train1x同一5542任务集四次rollout均完整结束，三固定动作的ledger和intervals六个文件与对应EDF/M2moderate/M2规则byteexact。随机动作seed77101，单元测试另验证同seed完整确定性。动作统计：

| mode | steps | opportunity_steps | action1_2_distinct_steps | actual_mode1_2_checked_steps | actual_mode1_2_different_steps | actual_policy_deferred_steps | completed | deadline_missed_completed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 216 | 168 | 166 | 0 | 0 | 0 | 5542 | 0 |
| 1 | 216 | 175 | 170 | 216 | 172 | 145 | 5542 | 0 |
| 2 | 225 | 179 | 179 | 225 | 181 | 181 | 5542 | 0 |
| random | 216 | 168 | 165 | 135 | 103 | 94 | 5542 | 0 |

prepass列仅为遍历前机会集合。actual_mode1_2列在实际action1/2的同一遍历上比较两个阈值：首个不同决定必改变启动集合；没有任何不同则整拍相同（一次遍历归纳）。Action0未计算实际候选gain，故不冒称其每拍完整反事实等价性。它的任务/能碳行为已与EDF严格等价。完整actions、每拍分项、候选决定/首次miss和资源事件保留本地，未公开大账本。

Gym checker通过，toy SB3 PPO n_steps8/batch8/epochs1、16环境步/2次更新确认参数改变，不作为研究模型，未正式训练PPO。E1-8入口见reproduce，不能将54维env误标成旧40维trainer结果。
