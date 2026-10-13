# E1-6A 公布材料索引

完整八策略六档表与客观结论：[消融报告](E1-6A-ablation-report.md)。入口摘要：[交付说明](E1-6A-delivery.md)。

- [代码审计](E1-6A-code-audit.md)、[数学定义及伪代码](E1-6A-mathematical-method.md)、[接口设计](E1-6A-policy-design.md)。
- [执行器适配与冻结](E1-6A-experiment-adaptation.md)、[独立代码审阅](E1-6A-review.md)。
- [测试证据](E1-6A-tests.md)、[运行命令/外部数据依赖](E1-6A-reproduce.md)、[核查清单](E1-6A-reproducibility-checklist.md)。
- [正式图注](E1-6A-figure-captions.md)、[真实图形QA](E1-6A-figure-QA.md)、[修正DMR重绘脚本](render_corrected_dmr.py)。
- [48行完整汇总](tables/summary.csv)、[144行固定/排空/公共阶段核算](tables/phase-comparison.csv)。
- [预算统计](tables/budget-statistics.csv)、[预算分布](tables/buffer-histogram.csv)、[需求主导类别](tables/budget-by-resource.csv)、[决策原因](tables/reason-counts.csv)。
- [违约任务集合对照](tables/paired-miss-summary.csv)、[预注册案例选择](tables/representative-case-selection.csv)、[代表任务详细记录](tables/representative-tasks.csv)。
- [数据/源码/运行身份](campaign-provenance.json)、[输入核验](existing-input-check-001.json)、[独立科学结果核验](results-independent-verification.json)、[旧成果保护核验](preservation-final-001.json)。

报告中的 `analysis-001/` 和 `corrected-figures-001/` 链接指原科研目录本地归档，GitHub没有上传这些目录。公开表位于上面的 `tables/`；完整PNG/PDF/SVG、运行日志、冻结任务、逐任务大账本、checkpoint均留在本地。五图正式交付目录为 corrected-figures-001；原DMR图被标为不合格并保留，未改写任何封存哈希。

从这些小表重绘正式DMR（Python+matplotlib）：

```sh
python docs/research/e1/E1-6A/render_corrected_dmr.py --summary docs/research/e1/E1-6A/tables/summary.csv --output YOUR_NEW_FIGURE_DIRECTORY
python -m unittest discover -s docs/research/e1/E1-6A -p test_corrected_figures.py
```

真实实验需要外部冻结输入及旧参考账本，参见复现说明；本公布包不宣称真实数据自包含。上游来源与LICENSE仍保留。Full在六档无一档同时改善RiskOnly的违约与碳排放，Test未运行，本轮没有科研PPO训练或安全Shield。
