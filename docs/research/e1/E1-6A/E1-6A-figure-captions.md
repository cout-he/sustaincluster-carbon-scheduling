# E1-6A 正式图注与统计口径

每个 Train/开发 Validation × 倍率为一个确定性冻结任务集合。每组八策略处理同一任务集、完整 Drain、同一公共观察终点，提前排空者继续积分真实空闲 IT/HVAC。没有独立重复样本、置信区间或显著性结论；Test 未运行。

1. **Deadline miss rate**：全部已到达任务的完成违约数/总任务数；全部排空后无删失。虚线为预注册 5% 操作性目标，不能将高压目标改成其他数值。
2. **Carbon relative to paired EDF**：减碳百分比严格定义为 `100 × (C_EDF − C_policy) / C_EDF`，C 为同组公共窗口的 IT+HVAC kgCO₂e。正值减碳、负值增碳；不同窗口/倍率的绝对碳排放不构成算法对照。
3. **High-pressure Pareto**：同组 DMR 与上述减碳百分比的二维关系。仅报告八个被测策略中的非支配点，不是最优性、因果性或泛化证明。
4. **First-eligible buffer distribution**：每任务首次能容纳、当前 CI>161.075、网格 solo slack≥15min 时的 B，按 15min 拍数分布。每行标明该策略该组实际分母。低 CI、不足一拍 slack 的 B 为 null；容量失败不调用策略；这些均不能当作 B=0。
5. **Buffer by demand category**：按 q_r/C_r 的最大值划分 CPU/GPU/MEM 主导、tie、zero，统计上述首次 eligible B 的均值。无该类别时为 missing，不补零。类别表示容量比例需求主导，不等于实际资源瓶颈。

CSV 映射：图 1–3 使用 `analysis-001/figure-source-summary.csv`，图 4 使用 `figure-source-buffer.csv`，图 5 使用 `figure-source-resource.csv`。原始完整统计另见 summary、phase-comparison、budget-statistics、buffer-histogram、budget-by-resource、reason-counts 和 paired-miss-summary。

完整实验的自动生成 `analysis-001/figure-captions.md` 第 2 项使用了有歧义的变量文字；数值计算、图中轴标签和对应 CSV 使用上述正确公式。该封存原始文件不重写；本文件是交付及发布使用的澄清图注。

代表任务按预先固定的 first stable_id 规则选择 Full 相对 RiskOnly 的 added miss、avoided miss、changed start/same outcome；类别缺失如实记 false。任务级对照是轨迹关联，不能直接解释为单个 guard 的严格因果贡献。
