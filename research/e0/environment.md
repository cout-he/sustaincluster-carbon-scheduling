# E0 环境与输入完整性

## 实验身份

- 执行日期：2026-10-17，用户时区 Asia/Shanghai；机器记录采用 UTC。
- 独立目录：`D:\A-myprofile\SustainCluster\rl-scheduling-v1`。
- 仓库：<https://github.com/cout-he/sustaincluster-carbon-scheduling>。
- 分支：`research/rl-scheduling-v1`，直接从 main 快照 `749dc6c1868065c12a71bac71b3ae08a25c45f56` 创建。
- `SOURCE_SNAPSHOT.json` 指向官方版本 `3f6ea95cb835b89ba50b0ef76d66d14b8037643e`，2025-06-04。
- 未导入任何其他 research 分支源码、任务或物理模型。已有 checkout 仅作为该官方 commit 的 Git 对象缓存。

## 运行环境

Windows AMD64；Python 3.11.15；CPU 执行。精确版本见 `environment.json` 和 `requirements-tested.txt`。

本地 `.venv` 用 `D:\anaconda\envs\llama_factory\python.exe -m venv --system-site-packages .venv` 创建，因此复用了已安装的 numpy 2.4.4、pandas 2.3.3 和 PyTorch `2.14.0.dev20260620+cu130`。没有导入该环境中的模型、训练数据或 checkpoint；CUDA 未使用。venv 另安装 gymnasium 1.0.0、pytest 8.3.5、psychrolib 2.5.0、tensorboard 2.19.0、pytz 2025.2 及其依赖。

这是一份已执行环境记录，并非完全隔离的可移植 lockfile。特别是开发版 Torch 不能假定从通用 PyPI 重建；后续应另建干净、稳定版 CPU 环境并重复 E0 后再冻结正式依赖。本轮未把未经测试的替代版本写成已验证环境。上游 requirements 未固定版本，且缺少直接使用的 numpy/pytz 等依赖说明。

seaborn、Sphinx、furo 和 sphinx-autodoc-typehints 未安装；本轮未执行绘图与文档构建，它们不参与已执行的测试和短期仿真。禁用 pytest 自动加载外部插件；使用 Agg 绘图后端、UTF-8 输出和仓库内临时目录。

## 数据

恢复和逐文件校验依据：`data_manifest.json`；数据质量核验：`data_quality.json`；详情及命令：`data_recovery.md`。

从官方指定 commit 的本地 Git 对象提取 378 个原始输入 blob：64 个碳强度、230 个电价、3 个网络矩阵、80 个天气文件及 1 个 Alibaba ZIP。再从该官方 ZIP 解压唯一 workload PKL。合计 379 个本地输入文件、439,891,572 bytes，全部由 Git 忽略。没有复制旧研究目录中来源不明的 PKL，也未清洗或重写任务数据。

原配置所用的 15 个 2023 年外部时间序列均有 8,760 条小时记录，核验无重复、断点或非有限值；AWS 网络成本矩阵 20×20，无缺失。原 workload 有 37,552 行、1,300,144 个 task records；区间覆盖至 2021-01-26，存在 80 个 30 分钟间隔。文件名中的 full_year_2020 不能作为严格全年连续性的证据。上述问题保留原样；不能据此宣称全年任务映射已验证。

## 设置过程中的实际问题

1. 在顶层目录执行 Git 命令时确认其不是仓库；随后定位已有 checkout，并建立全新 clone。
2. 沙箱内默认网络代理不可连接；授权网络执行成功读取远端 main 并 clone。没有改变全局 Git/代理配置。
3. 首次 venv/ensurepip 因系统临时目录写权限失败。把 TEMP/TMP 指向 `research/e0/.tmp` 后创建成功。
4. 环境清单首次包含未安装的 seaborn，元数据查询报错；改为明确记录可选依赖缺失，没有把它写成已安装。
5. 数据质量脚本首次终端输出出现 GBK 编码错误，修正 E0 检查脚本后完成；原始数据和模拟器未改动。

## Git 边界与敏感信息

原始输入、`.venv`、临时文件和权重均不提交。提交的数据清单只含公开输入的路径、大小与摘要；实验产物仅为小型聚合统计、逐步统计和执行日志。

原始电价采集示例 `data/electricity_prices/scripts/test_prices.py` 含预置凭据。本分支仅将它改为要求外部环境变量，不运行该联网采集示例，不在报告或新提交内容中记录凭据值。历史 main 快照未改写；历史凭据的撤销需由其所有者处理。该例外不改变任务提取、调度、奖励或物理模型。

E0 自有 `.gitattributes` 保留 artifacts、evidence 和 CSV 的执行字节，避免 Git 换行转换使已记录的产物摘要漂移；已逐一比较暂存区的六个 baseline 产物与 verification.json 的大小和 SHA-256。
