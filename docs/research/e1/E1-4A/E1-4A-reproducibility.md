# E1-4A 复现与归档边界

本说明对应原科研工作区 `D:/A-myprofile/SustainCluster/upstream` 中的 E1-4A 正式实验；状态包括 2026-10-11 完成的最终测试。只使用 E1-3 `results-001` 的封存输入，不重新生成负载、不调参。公开清理仓库的适用范围见最后一节。

## 冻结环境与设计

- 参考档案：`research_notes/E1/E1-3/results-001`；原科研 HEAD 为 `3f6ea95cb835b89ba50b0ef76d66d14b8037643e`，公开研究分支的本轮父提交为 `ff09c06ebac789716416abc4771f238b629cf39d`。这两个角色与旧参数中的 E1-2B `reference_source_commit` 分开记录，不改写历史来源。
- 本轮 Python 源清单为 150 项，旧 142 项全部保留原字节，新增 8 项。`source-freeze.json` 的清单 SHA-256 为 `495b572c6f5fdbb3cdbd7987a0626d4309e3926fd6f2018410ac76cc272402d6`；设计配置文件 `experiments/e1/configs/e1_4a_feasibility.json` 的原始字节 SHA-256 为 `3b43f70ef499af6ad10f64ee5c84ca031c568df366cb3bef234a83acb1dfe21e`。配置也随档案保存为 `ablation_config.json`，其规范序列化哈希另进输入清单。
- 实际环境为 WSL Ubuntu 22.04，解释器 `/home/hwj/miniconda3/envs/sustaincluster/bin/python`；完整 `sys.version` 为 `3.10.22 (main, Oct 1 2026, 20:14:04) [GCC 14.3.0]`。各 `provenance.json` 保存完整 115 项依赖版本，复现检查要求完整版本映射一致；代表版本为 NumPy 2.2.6、pandas 2.3.3、pytest 9.1.1、gymnasium 1.4.0、psychrolib 2.5.0。
- 必跑倍率为 1.5、2.0，共 14,914、19,886 个冻结任务；可选倍率为 1.0、1.25。可选倍率须在运行前指定，不能按结果筛选；本轮正式档案只运行两档必跑倍率。
- 五策略为 `edf_carbon_threshold`、`cars_risk_only`、`cars_full`、`risk_cap_only`、`risk_shadow_only`。每档复用相同 `workload.json`、`environment.json`、`dc_config.json`、`selection.json` 和子档 `parameters.json`。历史 buffer、阈值、horizon、pending cap 严格沿用；子档阈值保留 `137.64999999999998` 的封存精度，不能用名义值 `137.65` 替换。
- `cap_condition` 为候选当时 `pending>max_pending`。CapOnly 在原 RiskOnly 可 Defer 且超限时 Execute；ShadowOnly 超限时不执行 planner，报告 uncertain 并保留 RiskOnly 决定。`future_ci=false`。这些规则与参数冻结分别由设计配置和封存参数约束。

## 原档案只读验证

在原科研工作区执行以下步骤；输出路径必须是未使用过的新路径。保持源、测试和设计配置不变直到实验与回放结束。

```bash
cd /mnt/d/A-myprofile/SustainCluster/upstream
export PYTHONDONTWRITEBYTECODE=1 GIT_OPTIONAL_LOCKS=0
E14A_PY=/home/hwj/miniconda3/envs/sustaincluster/bin/python

"$E14A_PY" - <<'PY'
from experiments.e1.feasibility_experiment import _verify_import_campaign
levels, config, provenance = _verify_import_campaign(
    "research_notes/E1/E1-3/results-001", [1.5, 2.0], True)
print("verified_original_sources", len(provenance["sources"]))
print("selected_frozen_levels", [(x["factor"], len(x["workload"].tasks)) for x in levels])
PY
```

这是现有只读验证 API，不运行策略。它验证原 campaign 及全部四个子档，而后选择两档：完整 mandatory provenance keys、科学输出文件集合与逐文件 SHA、独立 runtime SHA、父子完整 manifest、封存参数与 resolved config、所有策略配置、输入哈希、选择记录、任务数、环境与 DC identity、原公共终点均必须一致。批准档案模式还检查原 HEAD、固定窗口和容量，以及原 workload/carbon/weather CSV 的实际文件哈希。缺少输入键、遗漏文件或出现额外未声明科学输出均拒绝。

跨版本导入核对原源清单的每一项：旧归档 `source/` 的字节 SHA 必须与原清单相同，当前科研工作区对应文件也必须相同。当前源清单只允许声明的 8 个新增 Python 文件；新增文件不会要求旧 E1-3 清单具有新的整体 SHA。不能直接使用旧代码要求整份当前清单等于旧清单的 strict replay 来做此跨版本导入。

## 运行与严格回放

下列示例使用新的 `*-reproduce-001` 路径；已存在的正式档案无需重跑。

```bash
"$E14A_PY" -m experiments.e1.feasibility_experiment \
  --reference-root research_notes/E1/E1-3/results-001 \
  --config experiments/e1/configs/e1_4a_feasibility.json \
  --factors 1.5 2.0 \
  --output research_notes/E1/E1-4A/results-reproduce-001

"$E14A_PY" -m experiments.e1.feasibility_experiment \
  --replay research_notes/E1/E1-4A/results-reproduce-001 \
  --output research_notes/E1/E1-4A/replay-reproduce-001
```

CLI 正式导入要求同时包含 1.5 和 2.0；如预先决定补跑，可使用 `--factors 1.0 1.25 1.5 2.0`。同一 `--replay` 接口也接受单个已完成子档，如 `results-reproduce-001/load_1p50`，并须指定新的输出目录。

五策略共享输入和五策略各自排空边界的最大值作为公共观察终点。较早排空者使用同一持续物理模型继续空闲积分，CPU/GPU/MEM 为零，使用对应时刻真实 weather/CI。档案同时保留 fixed、own、common 三种核算，完整 task ledger、policy events、intervals、diagnostics 和公平性记录；碳比较适用性由 `fairness.json` 明示。

严格回放先核查已完成状态、当前全部源字节、归档源快照、完整 Python/依赖版本、冻结输入和完整输出清单，再使用已保存的 workload/environment/DC/selection/parameters 重执行。它不调用负载生成器，也不需要从原 CSV 重建冻结输入。重执行的全部科学文件逐字节哈希必须与参考 E1-4A 相同；新 clock 时间不要求相同。

## 三旧控制的字节回归

新五策略公共终点与原 E1-3 七策略公共终点可以不同。驱动另外在原终点 `2023-07-03T13:45:00+00:00` 保存三旧策略的 `legacy-controls/`，并处理旧终点早于或晚于新终点的两种顺序，避免改变已捕获的结果。

正式两档 `baseline_regression.json` 均为 `files_checked=36`、`matched=true`、`historical_provenance_untouched=true`。每档核对 EDF+Carbon、RiskOnly、Full 的科学文件和 diagnostics，共 **36 × 2 = 72 个字节级一致文件**。实际规则为 `byte-identical scientific and diagnostics files; no metadata exclusions needed`；这里没有把诊断差异或时间字段从科学回归中删掉。独立只读重算这 72 项 SHA 与旧归档及封存哈希也全部一致。

## 单动作干预与只读分析

```bash
"$E14A_PY" -m experiments.e1.feasibility_intervention \
  --reference research_notes/E1/E1-4A/results-reproduce-001/load_1p50 \
  --output research_notes/E1/E1-4A/intervention-reproduce-001

"$E14A_PY" -m experiments.e1.diagnose_feasibility \
  --reference-root research_notes/E1/E1-3/results-001 \
  --output research_notes/E1/E1-4A/reference-audit-reproduce-001

"$E14A_PY" -m experiments.e1.diagnose_feasibility \
  --campaign research_notes/E1/E1-4A/results-reproduce-001 \
  --output research_notes/E1/E1-4A/campaign-analysis-reproduce-001
```

干预只接受批准的 1.5×冻结 cohort（14,914 项，workload SHA `30cf0e805f1abbb38ba08ace92de84888bb5778e360459b7656bcf8367eb1e15`），严格使用设计配置中的历史候选 `3ba8a8e0067288f8/2023-07-01T10:15:00+00:00/17475/189~e1-2b-20261010-copy1`。它只把该候选在 `2023-07-01T10:15:00+00:00` 的原 Full Execute 改为一次 Defer，此后继续原 Full；要求恰好应用一次，并与未干预 Full 对照。干预独立归档，不进入公平五策略 comparison；当前模块没有干预 replay CLI。

两个分析入口只读输入、写新的 E1-4A 输出目录。旧档案分析检查完整明确风险集合及实际终态配对，并重建当时已知状态；新 campaign 分析保留各策略 common 指标、不同 guard 与 uncertain 动作、planner 实际执行分母、配对 miss 集合及启动时间差异。代表选择是预声明历史候选加每类 guard 的时间顺序首例，不能按成功与否挑选。方法细节见 `E1-4A-diagnostic-method.md`，正式机制结果见 `E1-4A-report.md`。

## Provenance 与历史保留

主 campaign、子档和干预均记录命令 argv、Git 状态、完整 Python/依赖映射、源路径→SHA 清单、源清单 SHA、输入/配置哈希和输出哈希；主 campaign 嵌入子档完整 provenance。`model_source_version` 分别记录原科研来源、公开父提交角色及复用的 142 项原来源，避免把发布提交当成历史模型源版本。

`runtime.json`、`run_status.json`、`provenance.json`、`source/**` 被明确排除在科学 `output_hashes` 之外。runtime 另存 `runtime_hashes` 并在验证和回放前核查，源快照另按完整 source manifest 核查；排除科学比较不表示不检查完整性。正式主档科学清单 7 项、两子档各 112 项；本轮严格回放 7+112+112 项全部相同。主/子档/干预 runtime 的独立 SHA 已只读复核一致。

旧 E1-3 子档的 `source/` 保存完整 142 项原 Python 字节，本轮主档和子档 `source/` 保存完整 150 项。需要保留历史执行环境时，应把这些快照按记录相对路径放入一个单独复现工作区，并配套记录的解释器、完整依赖和数据依赖；源快照本身不包含全部原始 CSV，也不等于可独立启动的完整 checkout。批准的历史导入仍需原 CSV 哈希一致；E1-4A replay 则使用其完整封存输入。原科研目录、历史来源快照及 provenance 不需要被改写。

`preservation-final.txt` 的只读保护检查覆盖 81,212 个旧文件，`changed=[]`、`newly_added_old_scope=[]`，原 Git 文件与元数据均相同。新结果、日志和本说明均写入 E1-4A 新路径。

## 当前验证记录

| 证据 | 实际结果或状态 |
|---|---|
| `results-001/run_status.json` | complete；结束 2026-10-11 03:06:44.860488 UTC |
| `replay-001/run_status.json` | complete；结束 2026-10-11 03:13:40.632850 UTC；同源 SHA、所有主/子科学清单一致 |
| `intervention-001/run_status.json` | complete；结束 2026-10-11 03:05:41.264989 UTC |
| `reference-audit-final/`、`campaign-analysis/` | 已生成只读分析输出；新分析绑定正式 `results-001` |
| 首轮 `tests-001.txt` | 4 failed、215 passed，1024.40s；四失败均为新测试 fixture 返回名义阈值与封存精度不同，修复为读取已封存 `parameters.json`，未放宽驱动校验 |
| `driver-fixed-002.txt` | 2 failed、2 passed、6 deselected，81.96s；诊断源并行编辑引发严格 source-stability 校验失败，保留失败日志 |
| `driver-fixed-003.txt` | 稳定源下相关两项 2 passed、8 deselected，75.88s |
| `diagnostic-tests/green-guards-final.log` | 13 passed，0.83s |
| `upstream-tests-001.txt` | 2 passed，5.56s；未执行禁跑的 RL training-loop |
| `publication-tests-001.txt` | 223 passed、2 skipped，648.60s；公开仓库缺少私有历史档案的两项依赖测试跳过 |
| `tests-final-002.txt` | 原科研目录最终 E1 测试 225 passed，1147.72s；进程退出码 0 |

这些失败与修复没有修改旧政策或物理逻辑。最终原科研测试以日志终态为准。复跑公开可适用测试使用记录的环境执行 `"$E14A_PY" -m pytest tests/e1 -q`；历史档案依赖是否可用需分别说明。

## 公开清理仓库的边界

`publication-source-comparison.json` 记录原科研目录与公开清理目录的实际 Python 字节比较：**56 项 exact（其中包括本轮全部 8 项新增 Python），93 项仅换行差异，1 项缺失，0 项其他差异**。缺失项是早先发布清理排除的 `docs_src/source/conf.py`，与物理逻辑无关；93 项旧生产源曾在发布清理时规范化为 LF。本轮新增 8 项 Python 与设计 JSON 配置在两目录均字节一致。

因此，公开清理 clone 的旧源字节并不等于原 E1-3 source manifest，且完整历史 frozen 输入与归档并未随公开 clone 发布。不能声称该 clone 可直接 consume 原历史字节清单，或认为公开 synthetic 测试通过已经证明私有历史数据端到端复现。上面的严格 E1-3 导入和已验证的正式运行/回放是在原科研工作区及其记录的字节源环境完成；需要在其他地方复现历史档案，应先构造独立的、与记录字节/解释器/依赖/数据一致的环境。

这个边界是既有发布清理造成的，本轮不改原源码、旧归档或旧 provenance 来使清理 clone 通过历史校验。公开 clone 适用于已发布代码与 synthetic tests；本说明及精简研究输出可以同步发布，完整任务账本、事件、冻结输入和私有来源仍留在本地科研档案。
