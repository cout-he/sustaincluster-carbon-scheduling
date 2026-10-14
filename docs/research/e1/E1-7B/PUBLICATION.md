# 公开材料布局与外部依赖

本文件仅说明当前E1-7B精简发布布局；不重写历史provenance。科研归档在原科研目录的 `research_notes/E1/E1-7B/`，GitHub精简报告在 `docs/research/e1/E1-7B/`。

- [交付判断](E1-7B-delivery.md)、[方法](E1-7B-method.md)、[实验](E1-7B-experiments.md)、[消融](E1-7B-ablation.md)、[接口](E1-7B-policy-and-rl-interface.md)、[测试](E1-7B-tests.md)、[复现](E1-7B-reproduce.md)保留科学归档原文与字节身份。
- 公开 `tables/` 对应本地 `analysis-002/tables/`，另加本地action-statistics、ablation-deltas和power-domain-clipping小表。[完整对照CSV](tables/comparison.csv)含48行，所有能碳为同场景统一公共终点核算。
- [冻结代理参数](parameters/cgd-calibration.json)是本地 `calibration-003/freeze.json` 的逐字节副本；其封装SHA和payload内容SHA定义不同，均保留原值。它包含简单回归系数及Train输入身份，不含原始任务、环境序列或PPO权重。[拟合诊断](parameters/fit-report.json)亦逐字节保留。
- `evidence/` 保存少量回归、图表与测试身份收据；完整原始测试日志含混合编码WSL前缀，公开test-receipt仅提取准确pytest末行，并记录完整原日志SHA。完整provenance、375个回放文件与大账本留在本地，公开manifest只记录其身份，不声称这些私有文件已上传。
- [六图脚本](plot_public_figures.py)只需公开CSV和numpy/pandas/matplotlib，按reproduce命令即可独立生成PNG/PDF/SVG。原图在本地 `figures-002/`；[图注与质量核验](E1-7B-figures.md)记录7pt最小字号、PNG300dpi、矢量PDF及3项明确警告。没有独立重复数据，未制造误差条。

真实物理实验仍依赖合法取得的原始workload、CI/weather，以及对应E1-5B/E1-5C冻结输入和历史物理基线。注册配置记录路径与SHA，身份不一致即拒绝。不得将新构造的输入冒称相同历史结果。发布目录不包含这些数据，因此不宣称完全自包含。

辅助分析脚本也在本目录：`build_analysis.py`、`audit_power_domain.py` 通过 `--root <完整本地E1-7B归档目录>` 读取科学归档；`validate_cgd_interface.py` 通过 `--freeze` 和 `--reference` 指定冻结参数及Train对照；`check_registered_pairs.py` 需要原E1-7A预注册案例归档。脚本依赖仓库源码可导入，默认目录是本脚本所在位置，公开精简目录本身不满足完整归档依赖。原科学报告中的 `research_notes/...` 命令是实际原科研运行路径；在发布布局调用辅助脚本时将脚本路径改为当前 `docs/research/e1/E1-7B/...`，并显式提供完整外部归档参数。所有输出使用新目录，禁止覆盖历史。

所有场景为既有Train和已用于开发的Validation；Test保持封存。没有正式新PPO训练，没有对Validation重拟合功率、阈值或保护参数，也不以测试通过代替算法效果证据。
