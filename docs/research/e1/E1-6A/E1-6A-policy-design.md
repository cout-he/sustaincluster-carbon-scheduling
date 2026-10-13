# E1-6A 实施方案冻结

实施前已完成 code audit、mathematical method 和 experiment adaptation。
用户已确认 `no_history` 仅 `a_h=0`，共同 cold=2 保留；它不是完全移除历史冷启动保护。
本轮不再改参数，也不将另一 cold 消融混入结果。

新增 `task_risk_state.py` 提供可独立测试的纯预算函数；`task_risk.py` 是独立 EDF 策略注册。
函数输入限当前候选、当前已知 running/前序 pending、旧 ArrivalHistory；不引用未来任务或环境。
`LocalExecutor`、旧策略、RL 环境和功率模型全部不改。

先实现无缓存参考查询：每拍只排序一次，候选扫描已知严格前序。
只在 can-fit、高碳、有一拍网格 slack 的分支计算预算；其他 B=null。
如 profiling 证明需要优化，再逐候选与参考版核对，不能修改集合或 ceil 判定。
参数和公式完整定义见数学方法报告，默认值与用户预注册一致。

实验由独立 `task_risk_experiment.py` 执行，使用六个已有冻结包。
每组八策略（七个消融加 EDF），各自完整 Drain，持续物理对象空闲尾段延伸到组内最慢终点。
先 Train 三档，通过守恒、旧轨迹回归和实现正确性 gate，再运行开发 Validation。
gate 不要求新方法胜出。全部失败结果保留，Test 不访问，PPO 不训练。

实现步骤：

1. 保存历史源码、原数据、冻结输入、checkpoint、报告/归档和原 Git 字节身份。
2. 新微型行为测试先失败，再实现纯函数/独立策略；执行旧固定动作测试。
3. 驱动公平窗口/回放测试先失败，再实现并冻结源码、输入、参数。
4. 运行 Train，记录参考实现开销、缓冲分布和退化风险；通过功能 gate 后运行 Validation。
5. 汇总八策略结果、配对违约集合、首次 eligible 任务预算和代表任务，绘图及冻结回放。
6. 全部旧关键测试回归，核对历史保护；精选必要文件同步发布副本、测试、提交和正常推送。

发布只包含必要源码/配置/测试、精简报告与小汇总 CSV；完整输入、账本、诊断和 checkpoint 留本地。
