# E1-6B 测试、冻结回归与训练接口证据

本报告区分代码契约验证、真实训练运行和算法收益。测试通过不证明启发式风险过滤保证 SLA，不证明 PPO 已最优收敛，也不构成创新性结论。Test 数据仍然封存；角色拒绝测试使用合成输入验证拒绝行为，不读取真实 Test。

## 1. 已完成的原科研目录全量关键回归

冻结新增源码、测试和配置后，实际运行范围为 `tests/e1` 加 `tests/test_model.py`：

```text
532 passed in 1740.81s (0:29:00)
Exit code: 0
```

实际日志为本地 `full-tests-002.txt`。包含 **531 项 E1 测试 + 1 项原模型测试**；六个新增 adaptive 测试文件共有 **91 项**，其余 440 项为已有 E1 回归。不存在以单个新增模块通过替代旧机制回归的情况。原科研目录保留 dirty 状态；未重置、stash、删除、覆盖旧源码/数据/模型/历史归档。

实际命令（Windows PowerShell，离线 WSL runtime；无 Git configuration override）：

```powershell
wsl -d Ubuntu-22.04 --cd <WSL_RESEARCH_ROOT> -- `
  env PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 GIT_OPTIONAL_LOCKS=0 `
  PYTHONPATH=<WSL_RESEARCH_ROOT>/research_notes/E1/E1-5B/runtime-deps `
  <CONDA_ENV_PREFIX>/bin/python -m pytest `
  tests/e1 tests/test_model.py -q -p no:cacheprovider `
  --basetemp research_notes/E1/E1-6B/full-test-tmp-002
```

29 分钟墙钟包括旧 campaign/source snapshot/文件及 Git 检查的开销，不能当作 PPO 模型运行成本或模拟物理模型耗时。日志前部存在 WSL localhost 代理提示/编码前缀，Python 实际退出状态与 pytest 结果单独核对。

## 2. 首轮失败保留及冻结重跑

首轮 `full-tests-001.txt` 实际为 **517 passed、4 failed，1462.11s**，并非全量通过；当时新增源码仍在并发修改。四个失败分别为：

1. `test_cars_experiment.py::test_campaign_requires_child_manifest_and_records_failure`
2. `test_feasibility_experiment.py::test_cross_version_import_checks_complete_old_archive_before_execution[input]`
3. `test_feasibility_experiment.py::test_campaign_offline_replay_preserves_children_and_parent_manifest`
4. `test_feasibility_experiment.py::test_legacy_control_cutoff_can_be_later_than_five_policy_common_end`

栈中的拒绝原因分别是 campaign 执行期间 source changed、跨版本已归档 source hash 与当前发生变化的文件不一致等。它们说明**源码冻结保护有效拒绝了执行期间的修改**，不能据此声明旧调度模型有 bug，也不能略去首轮失败。没有修改旧 guard 来强行通过。

停止源码修改后，四项失败用例与六个新增文件的91项合并重跑：

```text
95 passed in 340.39s (0:05:40)
Exit code: 0
```

日志为 `frozen-key-tests-001.txt`。该回归自然完成；未中止或通过设置 Git status/untracked override 改变审计范围。随后上节全量532项亦完成，覆盖最终固定源版本。第一轮和重跑全部日志保留，不上传全文。

冻结关键重跑参数（与上节相同 WSL/env/Python 前缀）：

```bash
python -m pytest \
  tests/e1/test_cars_experiment.py::test_campaign_requires_child_manifest_and_records_failure \
  'tests/e1/test_feasibility_experiment.py::test_cross_version_import_checks_complete_old_archive_before_execution[input]' \
  tests/e1/test_feasibility_experiment.py::test_campaign_offline_replay_preserves_children_and_parent_manifest \
  tests/e1/test_feasibility_experiment.py::test_legacy_control_cutoff_can_be_later_than_five_policy_common_end \
  tests/e1/test_adaptive_risk.py tests/e1/test_adaptive_observation.py \
  tests/e1/test_adaptive_rl_env.py tests/e1/test_adaptive_train.py \
  tests/e1/test_adaptive_evaluate.py tests/e1/test_adaptive_analysis.py \
  -q -p no:cacheprovider --basetemp research_notes/E1/E1-6B/frozen-tests-tmp-001
```

以上 shell 里的 `python` 指实际前缀中的 SustainCluster Conda Python，不是任意系统 Python。

## 3. 新增91项的契约与测试先行证据

| 测试文件 | 新案例数 | 主要验证契约 |
|---|---:|---|
| `test_adaptive_risk.py` | 21 | 四动作精确门槛、CI等于阈值执行、grid solo/safe、v1预算冻结、真实start后同拍预算更新、容量回填与filter veto区分、动作/未来信息拒绝 |
| `test_adaptive_observation.py` | 11 | 43维顺序/手算值/归一化、全部已到达pending含capacity blocked、连续z与离散B一致、empty/unknown/tie/zero、fsum容量边界、future fork、快照不变 |
| `test_adaptive_rl_env.py` | 31 | 固定动作重现旧规则、唯一LocalExecutor、task/resource守恒、非抢占释放边界、持久IT/HVAC/idle、前40维不变、无未来泄漏、Reward首次计罚、terminated/truncated、Gym checker及随机完整rollout |
| `test_adaptive_train.py` | 13 | 四组83维/4动作/同参数量及预算、真实PPO参数/优化器更新、pre-action trace、Train双场景与reset、场景/manifest身份、Reward不变、角色/错误预算/覆盖拒绝 |
| `test_adaptive_evaluate.py` | 5 | 十九方法公共idle积分与任务守恒、checkpoint协议绑定、Test拒绝、记录动作精确回放、tamper拒绝 |
| `test_adaptive_analysis.py` | 10 | 完整12模型/三档/十九方法矩阵、公平gate与来源、成对效应算术、缺失seed/截断/能碳/角色/预算拒绝、不覆盖 |
| 合计 | **91** | 全部进入最终532项回归 |

TDD记录保留了功能不存在时的实际 RED，以及实现、接线和测试夹具修正过程：

- 风险策略首轮21项因新模块尚不存在失败；实现后19 passed/2 failed，再20 passed/1 failed，最终21 passed（3.17s）。中间涉及合成夹具原 MEM 默认需求、EDF deadline 排序和短有效历史压力，需要用符合预期容量/排序的任务实例，未改旧执行器以迁就断言。日志 `risk-red-001.txt`、`risk-green-001/002/003.txt` 保留。
- Observation 首轮11项因模块不存在失败（0.87s），实现后11 passed（2.66s）；与旧 task-risk/旧RL状态测试组合84 passed（2.28s）。详细记录见 `E1-6B-state-method.md`，不能将这84重复累加到最终532中。
- 新环境首轮31失败（ModuleNotFoundError）；首次新旧合测47 passed/7 failed，七项是测试把旧 interval 字段写成 `time`，实际字段是 `interval_start`。修正夹具后54 passed（1.73s）=31新增+23旧环境。旧执行器/物理公式没有因此修改。日志 `test-adaptive-env-red-001.txt`、`test-adaptive-env-green-001/002.txt` 保留。
- 新训练首轮11失败；增加 manifest/真实训练身份契约后11 passed/2 failed。完善新训练元数据和声明manifest校验后，与旧9项合测22 passed（17.98s）。日志 `adaptive-train-red-001/002.txt`、`adaptive-train-green-001.txt` 保留。
- 新评测首轮 RED 对缺失模块/契约提出要求；首次接线2 passed/2 failed，CSV首行启发式元数据缺少后续PPO字段，触发 `dict contains fields not in fieldnames`。新评测补齐统一metadata列后，新评测+风险合测25 passed（4.66s），加入loader/replay验收后5项全部通过（9.19s）。没有隐藏CSV丢字段或放宽科学回放断言。日志 `evaluate-green-001.txt`、`evaluate-risk-green-002.txt`、`replay-loader-green-001.txt` 等保留。
- 分析模块先以缺失模块得到RED，完成实现并修正合成测试的错误匹配文本后10项通过。最终冻结回归重新覆盖，不因局部测试通过而提前发布。

若需要复跑，使用全新 basetemp/日志，原TDD和测试记录不可覆盖。

## 4. 与 pytest 532分开记录的辅助审计

这些审计与532项有不同计数单位，**不得合并成一个更大的“pytest通过数”**：

| 辅助验证 | 实际证据 | 范围与限制 |
|---|---|---|
| B0算术自检 | 4项，OK，`B0-selftest-001.txt` | z区间、ceil网格决定等价、小需求瓶颈、全零任务；纯机制，非科研胜利样本 |
| 冻结v1预算独立复核 | 257,719 eligible预算逐项精确一致，输入哈希不变 | 保存状态的算术一致，不是安全保证或因果估计 |
| 真实Train固定动作回归 | 两档×三个动作=6轨迹核心文件/间隔结果字节一致，实际exit0 | 完成8313/11084，同旧物理执行；非新训练或Validation成绩 |
| 本地Validation驱动测试 | 13 passed in2.71s，exit0 | `test_validation_driver.py` 的合成矩阵/完整Drain/角色/能碳/资源/篡改/formal gate/replay/非覆盖拒绝；不包含真实Validation rollout |
| 独立科学账目审计测试 | 19 passed in1.85s，exit0 | `test_scientific_audit.py` 的手算微型实例与篡改反例：资源占用/释放、EDF顺序、任务特征、CI/能碳、Reward首次罚分、Drain；不代表57条真实新结果已经审计 |
| Factorial分析算术 | 6项，passed=true | 四个简单效应与交互项、加性零交互；不生成正式CSV或虚构成绩 |
| 补充分析自检 | `supplement_analysis.py --self-check` passed | schema、条件状态/过滤效应、交互与DMR百分点；纯算术，不证明收敛 |

本地Validation驱动实际命令（同WSL/env/Python前缀）：

```bash
python -m pytest research_notes/E1/E1-6B/test_validation_driver.py -q \
  -p no:cacheprovider --basetemp research_notes/E1/E1-6B/validation-driver-tests-tmp-001
python -m pytest research_notes/E1/E1-6B/test_scientific_audit.py -q \
  -p no:cacheprovider --basetemp research_notes/E1/E1-6B/scientific-audit-tests-tmp-001
```

Factorial与补充自检命令见 `E1-6B-reproduce.md`。完整真实结果出现后还需要 artifact/hash/资源小时/公共窗口与全量回放验收，不能用合成驱动测试替代。

## 5. Smoke与正式训练的独立判断

四组 Train-only 1024步smoke均实际完成、参数变化、19,397参数、83维/4动作、4 rollout/40epoch更新、256探针checkpoint回读一致；全部4096拍trace独立核验有限值、真实执行与首次违约，`smoke-gate-001.json` passed=true。四组成本满足预注册门槛，详见 `E1-6B-training-smoke.md`。

这是正式训练前的功能/成本门槛，不能预判正式训练收敛或Validation效果。随后完整 `formal-campaign-001/campaign-receipt.json` 已出现且passed=true：12个模型每个20,480步，总245,760步，80rollout/800epoch更新，实际参数/优化器/回读/有限trace与首次违约验收通过；详见主报告第3节及 `formal-training-report-001/receipt.json`。这证明正式训练真实完成，仍不证明最优收敛或方法优势。真实性能必须来自开发Validation公共窗口结果。

## 6. 发布副本验证：已完成关键测试

必要13个新核心文件（六源码、六测试、配置；合计133,758字节）已按字节同步独立发布副本，源/目标哈希一致收据为 `core-publication-sync-001.json`；此次同步本身不是commit或push。发布副本自己的 `tests/e1 + tests/test_model.py` 实际结果为：

```text
530 passed, 2 skipped in 1178.27s (0:19:38)
Exit code: 0
```

日志为本地 `publication-tests-001.txt`。两个已有测试自身明确声明的skip是公开仓库未提供的外部归档依赖：

1. `test_cars_experiment.py::test_approved_archive_head_preserves_research_provenance_not_publication_sha`（约第228行）：`Private historical archive is an external data dependency`，所需 E1-2B 私有历史归档不存在于发布副本。
2. `test_policy_runs.py::test_historical_e1_1a_fifo_legacy_fields_and_numbers_unchanged`（约第124行）：`Historical local artifact is not shipped with source; compare CLI checks it when supplied`，所需 E1-1A `smoke-007` 原始完整归档不存在于发布副本。

两项在具备外部材料的原科研目录已实际通过；没有删除用例、添加新skip或复制私有归档以掩盖依赖。发布结果须按530通过/2跳过报告，不能称为532全部通过或仓库完全自包含。

实际发布测试命令（stdout保留到原科研目录本轮的新 `publication-tests-001.txt`）：

```powershell
wsl -d Ubuntu-22.04 --cd <WSL_PUBLICATION_ROOT> -- `
  env PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 GIT_OPTIONAL_LOCKS=0 `
  PYTHONPATH=<WSL_RESEARCH_ROOT>/research_notes/E1/E1-5B/runtime-deps `
  <CONDA_ENV_PREFIX>/bin/python -m pytest `
  tests/e1 tests/test_model.py -q -p no:cacheprovider `
  --basetemp <WSL_RESEARCH_ROOT>/research_notes/E1/E1-6B/publication-test-tmp-001
```

此命令仅通过本地离线PYTHONPATH使用既有runtime，未把该目录纳入发布副本。

发布测试使用自己的新临时输出路径；不复制大型归档、runtime、模型、数据或私有文件进可提交目录。测试通过后仍需完成import、实际staged hash/敏感文件检查、精简结果同步与正常commit/push验收；本报告此版不提前宣称这些发布步骤完成。原 main 和原科研 Git 元数据保持不变。

## 7. 历史成果与原Git保护核验

`capture_preservation.py --verify preservation-final-001.json` 已实际运行并退出0，保护233,023个原文件：changed=[]、unexpected_additions=[]、git_metadata_equal=true、git_state_equal=true。因此原始源码/数据/历史归档/checkpoint和原科研 `.git` 状态相对开始时不变；本轮声明的13新文件及E1-6B全新notes按预注册范围排除。完整233k原始清单保持本地，公开只提供计数、布尔结论与必要身份SHA，不复制私有路径清单。

## 8. 有意未执行的上游训练入口

未使用裸 `pytest` 自动收集 `tests/test_training_loop.py`。该文件会导入上游训练入口，并通过 `os.system("python train_SAC_single_action.py ...")` 启动旧SAC训练；它不是当前独立E1四动作接口的无副作用单元测试，也不核验该子命令返回码。运行它可能开展本轮范围之外的生产/RL训练并产生输出。

因此选择完整 `tests/e1` 与不启动训练的 `tests/test_model.py`；未把这一选择描述为“仓库所有测试都通过”。旧E1关键执行/物理/冻结回放与三动作接口均在该范围内。当前测试证据不能替代真实数据覆盖、模型边界审计或新的未见Test验证。

## 9. 最终实际科学核验

三档19方法共57条开发Validation通过，全部任务完成、无未完成/删失；同档资源小时相等，公共终点均为2023-07-10 12:45 UTC。57条action-tape回放的完整scientific result哈希精确一致，含任务账本、资源、功耗及能碳。独立只读执行/能碳审计57条、旧基线回归12条通过；另对36条新RL逐候选复算冻结v1预算及过滤分母，读取文件前后SHA相同。

辅助测试另计，未混进532核心回归：Validation驱动15通过（含65位SHA录入错误的先RED后GREEN记录）、科学审计19通过、过滤诊断审计11通过。合成两任务单拍接口核验通过：同一45/60分钟solo slack，EDF启动两者、Carbon/RiskOnly延迟两者，新Action1无过滤只启动45任务；这是表达能力证据，不是性能单因素结论。该脚本初次遗漏start调用被执行器拒绝，修正notes调用后退出0；两日志均保留，没有修改执行器。

5张最终图PDF最小文字6.5pt，均通过5pt阈值审计及逐panel目视核验。初版supplement-001右刻度裁切保留，仅调整notes页边距，另存supplement-002；全部科学CSV字节哈希一致。模型/原数据/完整账本/日志/图片不上GitHub。
