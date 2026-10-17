# E0：原始 SustainCluster 实验基础报告

## Material Passport

- 实验：E0-original-sustaincluster，执行日期 2026-10-17（Asia/Shanghai）。
- 起始源码：`749dc6c1868065c12a71bac71b3ae08a25c45f56`，分支 `research/rl-scheduling-v1`。
- 官方来源：`HewlettPackard/sustain-cluster@3f6ea95cb835b89ba50b0ef76d66d14b8037643e`。
- 输入身份：`data_manifest.json`；环境版本：`environment.json`；结果：`baseline_results.csv`。
- 验证范围：真实执行与保存产物的复算；单个 seed、单日窗口，无新算法训练，无统计显著性或普遍优越性声明。

## 结论与完成范围

原始环境可以运行。已在全新 checkout 从指定 main 快照直接创建研究分支，恢复与该快照对应的官方输入，运行原有测试、最小 reset/step、24 小时短仿真和官方规则调度器对照。本轮未引入其他研究分支、ACP-PPO、新强化学习算法或训练权重，未修改 main。

五个官方规则基线、一个固定本地动作接口诊断、重复运行及直接 cluster 对照共 8 组仿真均完成 96 步。资源和统计账目可复算，但原始路径的时间及奖励语义不一致，**目前只能作为忠实复现的运行基础，尚不能作为公平的 RL 优势评估协议**。

核心任务提取、环境、调度器、奖励和物理模型保持官方源码。唯一原有源码改动是移除电价采集示例中的硬编码凭据，要求外部环境变量；该联网采集示例不在 E0 执行路径中。源码逐项对应关系及风险见 `code_audit.md`。

## 环境与输入

Windows AMD64、Python 3.11.15、CPU。venv 复用了本机系统包，包括开发版 PyTorch；它是已运行环境记录而非独立可移植 lockfile，后续需在稳定版干净环境复跑。详情见 `environment.md`。

原配置运行输入全部恢复：378 个官方 Git blob 加 ZIP 内原始 PKL，共 379 个文件、439,891,572 bytes，逐一核验摘要，全部被 Git 忽略。没有复制旧研究生成的任务；Alibaba 文件是官方预处理工作负载。五站使用的 2023 年 15 个小时序列各有 8,760 行；原 workload 自带 80 个缺失 15 分钟槽且时间范围延伸到 2021 年，未清洗或补造。详见 `data_recovery.md`、`data_quality.json`。

## 真实执行记录

所有仿真脚本自动定位源码根目录，统一使用原始 `TaskSchedulingEnv.step → DatacenterClusterManager.step → SustainDC.step`，另以直接 cluster 调用作对照；只读 observer 记录原始任务及 info，不改变调度决策、物理输出或输入任务。

| 验证 | 真实结果 | 时间/证据 |
|---|---|---|
| 原始 `pytest tests -q --tb=short` | exit 0，3 passed | 子进程 7.2411 秒；`evidence/original_tests.txt` |
| 原始测试可见输出复查 `pytest tests -s -vv --tb=short` | exit 0，3 passed；暴露训练测试假阳性 | 子进程 6.1828 秒；`evidence/original_tests_visible.txt` |
| 首次 1 步 smoke | 3 个运行失败，E0 observer 错用 price manager 属性 | `artifacts/smoke/summary.json` 保留 AttributeError 和耗时；没有有效物理汇总 |
| 修正 observer 属性后的 1 步 smoke | 3/3 完成，6 个相同任务；reset/step、规则与直接调用一致 | `artifacts/smoke_verified/`；execution 记录真实总耗时 |
| 96 步基线 | exit 0；8/8 完成，共 768 个 step、3,840 个站点记录 | 27.5727 秒；`artifacts/baseline/`、`artifacts/baseline_stdout.txt` |
| 最终源码验证 | 122 条官方清单、121 个导出源码文件核验通过；唯一内容例外为凭据外置 | 10.2554 秒；`source_integrity.json` |
| 保存产物独立复算 | 所有逐站→逐步→汇总账目和 repeat/direct 相等检查通过 | `evidence/verification.json`；0.0304 秒 |

“3 passed”不意味着训练可用：原测试调用不存在的 `train_SAC_single_action.py`，忽略 `os.system` 退出码，故没有执行训练。reset 测试调用的原 `make_env` 未传递 sim_config，使单动作和禁 defer 配置回退；它只证明该回退 reset 路径可运行。没有修改原测试来掩盖这些问题。

首次 smoke 是独立 E0 观察代码的属性名错误，修正为原 `ElectricityPrice_Manager.index` 后才成功；没有为跑通实验修改原模拟器。首次数据质量脚本的终端编码错误以及环境设置错误另见 `data_recovery.md` 和 `environment.md`。

## 真实短期基线数据

任务标签窗口：`2023-08-01 05:00:00+00:00` 至 `2023-08-02 05:00:00+00:00`，每步 15 分钟，seed 42，固定五站顺序，不洗牌；原始 task_scale=5、group_size=1。E0 明确采用多动作接口并禁 defer，便于逐任务回放；规则基线保留官方规则选择和排队行为。

各组均有 2,506 个相同需求与来源任务，96 步逐批指纹一致，完整序列 SHA-256 为 `b9f022a827d01737aa723edb1580211165e7d4351bbfd1c0b2449588d74f78f8`。表中能耗和碳排均为**五站加传输**，成本为站点电费加传输成本；分项精度与逐步数据保留在 CSV。

| 官方规则 | 总能耗 kWh | 总排放 kgCO2e | 总成本 USD | 已完成任务 | 已完成中的 SLA 违反 |
|---|---:|---:|---:|---:|---:|
| local_only | 154291.5298 | 48653.4668 | 14507.2708 | 2368 | 0 |
| priority_order | 146823.5988 | 44901.3197 | 13670.3970 | 2368 | 0 |
| round_robin | 154423.3491 | 48679.3172 | 14798.8097 | 2368 | 0 |
| lowest_carbon | 156899.3820 | 45745.8274 | 14672.5711 | 2368 | 47 |
| lowest_price | 157353.5614 | 47961.1868 | 14135.5606 | 2368 | 52 |

五个规则运行均启动 2,506 个任务，窗口末仍有 138 个运行中任务，无 pending。SLA 数值只覆盖已完成任务，不是全部到达任务最终违反率；本轮没有排空窗口外任务。

固定 `manual_local_replay` 只是将每个任务送回来源站的动作接口诊断，未训练策略。它启动 2,496、完成 2,342，窗口末运行中 154、在途 10；已完成任务中 SLA 违反 495，总能耗 154071.7814 kWh、总排放 48559.5975 kg。它与规则 local_only 的服务时间不同，**不能把这些差值解释为算法性能差异**。

## 一致性验证与边界

- 8 组运行逐步输入指纹完全一致；原始 PKL 和配置摘要记录在 execution.json。
- 每个站点、每步检查 CPU/GPU/内存可用量上下界与运行任务资源总和；检查任务计数守恒、开始/完成计数及 SLA 完成计数。
- 电池 kWh、站点 common kWh、cluster energy_usage 相等；碳排核验 `kWh × bat_avg_CI / 1000`，并与电池克单位结果交叉核对。原字段 `price_USD_kwh` 实际为 USD/MWh，沿用 `/1000` 成本换算。
- reward 仅验证原组件加权代数，不将其当作真实物理量；对副本故意加入 1 kg 碳排错误、1 核资源错误和 1 reward 错误，8 组共 24 项负向检查均拒绝篡改。
- local_only 重跑与原运行逐步完全一致；经 wrapper 与直接 cluster.step 的同一规则结果逐步完全一致。
- 另由独立审阅复算全部 CSV，逐站总和与逐步/最终表一致。上述内部一致性不等于官方物理假设已经经过外部实验验证。

## 下一阶段风险与建议顺序

1. **先明确时间合同。** 本轮任务钟为 8 月 1 日，原始随机物理钟实际从 8 月 27 日 18:00 UTC 起，相差 26 天 13 小时。近似年内日、随机 reset、站点时区与插值网格应单独验证；本轮没有静默修复。
2. **统一比较路径的服务语义。** 官方 RBC 即时入站；manual_rl 即使零延迟本地任务也要下一 tick 入站。物理功耗还在本步任务调度前计算。先确定统一的队列/传输/功耗时序，再设计公平对照。
3. **冻结计量与奖励合同。** wrapper 的 total 字段只有传输，官方表格脚本还会读取不存在的键而默认得到 0；原奖励是 CPU×时长代理，所有 RBC reward 为 0。本轮物理真值来自逐站 info，不应用 reward 宣称节碳。
4. **处理截断与健壮性。** 单日末仍有运行/在途任务；原规则遇到不可调度任务时存在 None 解引用风险。长窗口、资源紧张、超大任务及排空后的 SLA 需单列测试。
5. **补数据与依赖复现边界。** 原 workload 的缺槽、跨年范围与年份映射尚需完整审计；固定稳定依赖并建立干净 CPU 环境，随后扩大种子、季节和窗口。单 seed 的本表不能支持泛化或显著性声明。
6. **维护源码与文件边界。** 原测试存在假阳性，采集示例的历史凭据需所有者撤销。继续禁止原始大数据、权重和敏感文件进入 Git；后续修复应另分实验，保留 E0 冻结证据。

在这些合同明确之前，不建议启动 ACP-PPO 或其他新算法训练。

## 复现步骤

在本分支源码根目录执行。完整命令与解释器路径见 execution.json 和 evidence 下 command.json；以下 PowerShell 命令中的本机基础解释器路径应按实际安装调整。

```powershell
git rev-parse origin/main
git merge-base origin/main HEAD
# 两者应仍对应 749dc6c1868065c12a71bac71b3ae08a25c45f56

New-Item -ItemType Directory -Force research/e0/.tmp | Out-Null
$env:TEMP = (Resolve-Path research/e0/.tmp).Path
$env:TMP = $env:TEMP
& 'D:\anaconda\envs\llama_factory\python.exe' -m venv --system-site-packages .venv
.venv/Scripts/python.exe -m pip install gymnasium==1.0.0 pytest==8.3.5 psychrolib==2.5.0 tensorboard==2.19.0 pytz==2025.2
# 本机其余已测试版本见 requirements-tested.txt；不把开发版 Torch 假定为通用 PyPI 可安装。

.venv/Scripts/python.exe research/e0/restore_data.py --source-repo ..\upstream
.venv/Scripts/python.exe research/e0/inspect_data.py
.venv/Scripts/python.exe research/e0/verify_source.py --source-repo ..\upstream

$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:MPLBACKEND = 'Agg'
$env:PATH = (Resolve-Path .venv/Scripts).Path + ';' + $env:PATH
.venv/Scripts/python.exe -m pytest tests -s -vv --tb=short --basetemp=research/e0/.tmp/pytest-reproduce
.venv/Scripts/python.exe research/e0/run_baselines.py --steps 1 --strategies local_only manual_local_replay local_only_direct --output research/e0/artifacts/smoke_reproduce
.venv/Scripts/python.exe research/e0/run_baselines.py --steps 96 --output research/e0/artifacts/baseline_reproduce
.venv/Scripts/python.exe research/e0/collect_results.py
```

首次复现需先准备包含上述官方 commit 的 `HewlettPackard/sustain-cluster` 本地 clone，传给 `--source-repo`；不应指向其他研究分支生成的任务目录。`verify_evidence.py` 复算本次已保存的固定 `artifacts/baseline`，不把新复现覆盖成原证据。`baseline_results.csv` 中保留首次 smoke 的真实失败行；失败行的缺失物理数值为空，并未补造。
