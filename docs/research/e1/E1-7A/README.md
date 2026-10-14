# E1-7A：调度空间与碳机会诊断

- [科学判断与唯一建议](E1-7A-scientific-decision.md)
- [代码审计](E1-7A-code-audit.md)、[预注册](E1-7A-pre-registration.md)、[五篇文献机制](E1-7A-literature-mechanism-map.md)
- [A离线排程](A-offline-scheduling-report.md)、[B真实能碳机会](B-carbon-opportunity-report.md)、[C单次干预](C-counterfactual-report.md)
- [测试](E1-7A-tests.md)、[复现与外部依赖](E1-7A-reproduce.md)、[交付边界](E1-7A-delivery.md)
- [三策略基线](tables/baseline-comparison.csv)、[A完整对照](tables/A-edf-vs-oracle.csv)、[B能碳](tables/B-energy-decomposition.csv)、[C十二对](tables/C-paired-outcomes.csv)
- [输入/选样摘要](public-selection-summary.json)、[科学输出哈希](public-evidence-manifest.json)

所有数值来自同一Train窗口，本轮不是独立泛化验证，不训练新的科研PPO。两组零效应是合成副本互换；无新增deadline失效病例则如实缺失，不重选。未将clairvoyant IT代理给在线调度器，也未证明全窗口最优碳/违约下界。

完整8幅PNG/PDF/SVG及底层绘图CSV保留本地figures-002/、figures-supplement-001/，原case/ledger/日志保持私有；公开图形说明及哈希，不分发完整图片或原输入。原执行器、功率公式和既有PPO/负面结果不变。
