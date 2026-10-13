# E1-6B 复现入口、外部依赖与冻结身份

本文给出已实现入口的运行方式，不预先声明正式 12 模型训练、57 条开发 Validation 轨迹或回放已经完成；完成证据须以最终报告及实际 campaign receipt 为准。命令中的科研根目录应替换为自己的本地路径。既有科研归档不可覆盖。

## 1. 公开仓库不是完全自包含的历史回放包

公开发布包含必要上游源码、E1 研究源码/测试、配置和精简结果，**不包含**原始 workload、CI/weather 数据、完整冻结 bundle、旧/新 PPO checkpoint、完整逐拍/逐任务账本、源码快照和训练/实验日志。代码可导入、微型测试可运行，不等于下载仓库后就能重放全部历史科学结果。

运行正式训练需要用户已具备的两个冻结 Train bundle；完整外部基线比较还需要三个冻结开发 Validation bundle 和三个原 E1-5B checkpoint。精确历史 action-tape replay 需要完整本地评测归档、对应 source-freeze 的源码集合及相同输入/依赖。若这些材料缺失或 SHA 不符，入口应拒绝；不得改写原 provenance、声明新 hash 代替旧 hash，或用自制数据冒充已有冻结身份。

重新从原始数据生成 bundle 的准备步骤见既有 `docs/research/e1/E1-5B/E1-5B-data-freeze.md` 与 `E1-5B-reproduce.md`。原始数据许可与获取责任仍由用户遵循上游说明；不得把数据或权重重新上传。新生成结果与历史精确回放应分开标记。

## 2. 依赖与运行位置

本轮实际 WSL 为 Ubuntu-22.04，Conda `sustaincluster`，CPU 单线程确定性；既有审计运行时为 Python 3.10.22、torch 2.14.1、Gymnasium 1.4.0、SB3 2.9.0、numpy 2.2.6、pandas 2.3.3、matplotlib 3.10.9。SB3 的追加依赖登记于 `requirements-e1-ppo.txt`；原 SustainCluster 科学依赖继续保持。仅有微型 mock power 测试通过不证明真实 IT/HVAC 环境已齐备。

下面的相对路径都从项目根目录运行。科研操作在本地科研工作区进行，发布副本只用于代码发布。若发布的辅助脚本位于 `docs/research/e1/E1-6B/`，应把**指定的辅助 `.py` 文件**按原字节放回本地 ignored `research_notes/E1/E1-6B/`，再执行：

```bash
# 在一个具备外部数据的全新本地科研副本根目录执行；目标文件存在则先停下核查。
mkdir -p research_notes/E1/E1-6B
for name in verify_smoke.py run_formal_campaign.py run_validation_campaign.py \
            supplement_analysis.py make_factorial_effects.py scientific_audit.py \
            filter_diagnostic_audit.py interface_capability_check.py; do
  test ! -e "research_notes/E1/E1-6B/$name" || exit 1
  cp "docs/research/e1/E1-6B/$name" "research_notes/E1/E1-6B/$name" || exit 1
  cmp "docs/research/e1/E1-6B/$name" "research_notes/E1/E1-6B/$name" || exit 1
done
```

只复制上述小脚本，不复制 docs/原归档整个目录。`run_formal_campaign.py`、`run_validation_campaign.py` 的 `ROOT=Path(__file__).resolve().parents[3]` 依赖脚本处在 `research_notes/E1/E1-6B/`；直接在 docs 位置运行会得到错误根目录。脚本内固定的 `train-<group>-<seed>-001`、`smoke-<group>-001` 身份也不能只靠更换顶层输出编号规避旧模型目录冲突。已有正式归档的科研根目录不重新跑这些固定目录命令；需要复跑时使用另一全新本地副本，并保留原档。

### 本轮实际 WSL 命令前缀

实际命令使用以下前缀，没有 `GIT_CONFIG_COUNT` 或 Git status coverage override：

```powershell
wsl -d Ubuntu-22.04 --cd <WSL_RESEARCH_ROOT> -- `
  env PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 GIT_OPTIONAL_LOCKS=0 `
  PYTHONPATH=<WSL_RESEARCH_ROOT>/research_notes/E1/E1-5B/runtime-deps `
  <CONDA_ENV_PREFIX>/bin/python <下文脚本或模块参数>
```

`runtime-deps` 是本地已有的离线依赖目录，公开仓库不包含它。若同版本依赖已经直接安装在 Conda 环境，可用该环境安装位置而无需这个目录；但应记录依赖版本/来源。不得把整个环境同步到 GitHub。

为减少后续示例的重复，WSL shell 在自己的项目根目录、已激活 `sustaincluster` 后可定义等价包装：

```bash
conda activate sustaincluster
E1_PYTHON="$CONDA_PREFIX/bin/python"
E1_RUNTIME_PATH="$PWD/research_notes/E1/E1-5B/runtime-deps"
e1_python() {
  env PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 GIT_OPTIONAL_LOCKS=0 \
    PYTHONPATH="$E1_RUNTIME_PATH" "$E1_PYTHON" "$@"
}
```

没有安装本地 runtime-deps 的其他环境应先核实同版本依赖。所有脚本路径和 bundle 路径仍以项目根目录为准；不能通过换运行位置改变数据角色。

## 3. 必需的五个有限任务集合

每个 bundle 包含 `bundle.json` 以及六个受 manifest SHA 保护的 payload：`workload.json`、`environment.json`、`dc_config.json`、`config.json`、`stress.json`、`parameters.json`。它们必须保留原字节和原 provenance。

| 角色 / 压力 | 相对路径 | 任务数 | bundle.json SHA256 |
|---|---|---:|---|
| Train 1.5× | `research_notes/E1/E1-5B/precheck-001/load-1.5/input` | 8,313 | `4e02f1a1525c2de262ce25970674badd2e620177820951cb7c4cf1c1d8f686b6` |
| Train 2× | `research_notes/E1/E1-5B/precheck-001/load-2/input` | 11,084 | `e3d76b2daff040e6d783cd0ce8a81766b98ea996cf02c1111f4fe3529576c77a` |
| 开发 Validation 1× | `research_notes/E1/E1-5C/diagnostics-001/inputs/validation-1` | 8,708 | `db65bf311e533991ef4cb2f0698c96570d2ebba0185b963c3bb3c16e656a0a18` |
| 开发 Validation 1.5× | `research_notes/E1/E1-5C/diagnostics-001/inputs/validation-1.5` | 13,062 | `55238b2cedba9326cc46921b2c786bff83350c2faf1e7dd69ab39083fda185ac` |
| 开发 Validation 2× | `research_notes/E1/E1-5B/validation-input-001` | 17,416 | `62658e53a8edcf1b4fe74f5b738f7f4628a8defdf271a1b8d848f23c7fc18f02` |

Train 到达窗口为 2023-05-01 05:00 至 May 3 05:00 UTC，真实环境覆盖至 June 2 05:00；开发 Validation 到达窗口为 July 8 05:00 至 July 10 05:00 UTC，真实环境覆盖至 August 9 05:00。阈值 161.075 源于已许可的 April 校准；SLA 1.5、资源倍率5、容量(50000 cores,1000 GPUs,80000 memory units)、22°C HVAC setpoint 不变。

`load_bundle(..., allowed_split=...)` 验证角色、manifest/payload、UTC边界、参数、真实数据来源 SHA、物理配置及任务数。四个原始来源 SHA 已冻结在 `rl_training_utils.py::APPROVED_SOURCE_SHA256`，不是从开发 Validation/Test 重新估计；本轮不重新拟合归一化。

旧外部 PPO 所需三个模型路径为 `research_notes/E1/E1-5B/train-seed-{55101,55102,55103}-001/model.zip`，并保留同目录 training_config/summary 及其受保护产物。模型SHA按seed分别为：

```text
55101 cd3d7b533fdfca455947ff00171d60da2c94dd114dcfc5b1ffbdeefa4a191d3e
55102 d2c0e6cd3765f7bd63814818011586de7ffc8f42d839bcb681049d62ecd024e3
55103 73d2cc8bb4548d59a57501d56aa3b832c4fb7aadc1fecbe32d883ec4ee3374ac
```

没有这些旧模型不能声称完成包含旧三动作 PPO 的十九方法公平矩阵。可以单独运行新环境或训练，但必须明示范围不完整，不编造缺失基线。

## 4. 测试与 B0 门槛

在全新测试输出路径运行，禁止写旧归档：

```bash
e1_python -m pytest tests/e1 tests/test_model.py -q -p no:cacheprovider \
  --basetemp research_notes/E1/E1-6B/reproduce-tests-tmp-001
```

本轮未用裸 `pytest` 自动启动仓库中的训练循环测试，避免该入口意外开展未授权上游训练。完整测试范围及实际通过数由最终测试报告给出。

B0 已保存状态预算诊断和真实固定动作接口回归不应改写旧文件。其完整本地输入/结果不公开；只有在具备对应旧归档时才可运行这些历史审计。新源码微型等价性测试包括固定0→EDF、无过滤固定3→EDF+Carbon、有过滤固定3→Task Full。测试通过不代表这些规则有性能收益。

## 5. 四组 1,024 步 smoke 与正式训练

以下 smoke 命令只在无同名目录的新本地科研副本使用；每组 seed55100、1,024步，不读取 Validation。先运行四组，再核验生成新的 gate：

```bash
for group in plain state shield combined; do
  e1_python -m experiments.e1.adaptive_train \
    --config experiments/e1/configs/e1_6b_ppo.json \
    --group "$group" --seed 55100 --steps 1024 \
    --output "research_notes/E1/E1-6B/smoke-$group-001" || exit 1
done
e1_python research_notes/E1/E1-6B/verify_smoke.py
```

gate 必须核验真实参数变化、有限值、checkpoint 往返、四组一致结构/输入预算、逐拍资源与首次违约、成本门槛；不能手写 passed=true 或把历史 smoke gate 当作自己新环境的证据。原已归档 gate 及模型不覆盖。

正式监督器预先保存完整协议与来源，然后运行四组×三seed，每模型20,480步、最多三个CPU单线程进程；任何 source/config 变化、失败、超限将停止调度并保存证据：

```bash
e1_python research_notes/E1/E1-6B/run_formal_campaign.py \
  --output research_notes/E1/E1-6B/formal-campaign-001
```

总预算245,760环境步。正式模型目录为 `train-<group>-<seed>-001`；源文件和测试/config需从开始到结束保持不变。监督器每模型核验80个 rollout、800次epoch更新、19,397参数、83维/4动作、实际场景/动作计数、checkpoint/参数哈希及功能门槛。场景50/50为episode选择规则，真实步占比另报。固定预算尾段允许critic bootstrap，不能伪装为完整排空结果。

若只需验证模块调用，正式单模型入口如下；它不能代替完整十二模型协议，也不得在已存在该输出时复跑：

```bash
e1_python -m experiments.e1.adaptive_train \
  --config experiments/e1/configs/e1_6b_ppo.json \
  --group plain --seed 55101 --steps 20480 \
  --output research_notes/E1/E1-6B/train-plain-55101-001
```

## 6. 完整开发 Validation 与冻结回放

正式 `campaign-receipt.json` passed=true 后才能运行完整开发 Validation；监督器在执行前核验12新模型、3旧模型、三个bundle和来源，保存十九方法完整计划，不进行模型选择：

```bash
e1_python research_notes/E1/E1-6B/run_validation_campaign.py \
  --mode evaluate --workers 2 \
  --formal research_notes/E1/E1-6B/formal-campaign-001 \
  --output research_notes/E1/E1-6B/validation-campaign-001
```

输出 `load-1`、`load-1.5`、`load-2`；每档19方法相同输入、完整Drain、最慢策略公共终点和持久空闲能耗。公共 endpoint/资源小时/全完成/无删失审计不通过，不能宣称等工作量减排。

完整 action-tape 回放命令：

```bash
e1_python research_notes/E1/E1-6B/run_validation_campaign.py \
  --mode replay --workers 2 \
  --archive research_notes/E1/E1-6B/validation-campaign-001 \
  --output research_notes/E1/E1-6B/validation-replay-001
```

该回放逐拍核对 Observation SHA，重新执行资源/任务/能碳与空闲尾段，并比较完整 scientific hashes；它是冻结动作轨迹复现，不是新在线算法或重新训练。源码集合、档案hash或输入身份不同应拒绝。发布仓库与完整历史源码快照的对象身份/文件集合可能不同，不能绕过guard后声称精确回放。

单档模块入口也公开，候选表是完整驱动生成的 `candidates.json`，不是依据该档成绩重新筛选：

```bash
e1_python -m experiments.e1.adaptive_evaluate \
  --bundle research_notes/E1/E1-5B/validation-input-001 --split validation \
  --candidates research_notes/E1/E1-6B/validation-campaign-001/candidates.json \
  --output research_notes/E1/E1-6B/single-validation2-001
```

所有公开训练/评测 CLI 都没有 Test 选项；后端也拒绝 Test角色。禁止添加 Test 参数、加载 Test bundle 或把 Validation 写成未见测试集。

## 7. 三档分析、训练行为补充和完整 2×2 效应

三档评测完成并通过公平gate后，收集12个正式模型摘要再分析；Bash数组仅展开注册模型名，不进行新的实验或参数扫描：

```bash
E1_TRAINING_DIRS=()
for group in plain state shield combined; do
  for seed in 55101 55102 55103; do
    E1_TRAINING_DIRS+=("research_notes/E1/E1-6B/train-$group-$seed-001")
  done
done

e1_python -m experiments.e1.adaptive_analysis \
  --groups research_notes/E1/E1-6B/validation-campaign-001/load-1 \
           research_notes/E1/E1-6B/validation-campaign-001/load-1.5 \
           research_notes/E1/E1-6B/validation-campaign-001/load-2 \
  --training "${E1_TRAINING_DIRS[@]}" \
  --output research_notes/E1/E1-6B/analysis-001

e1_python research_notes/E1/E1-6B/supplement_analysis.py \
  --analysis-dir research_notes/E1/E1-6B/analysis-001 \
  --training-dir "${E1_TRAINING_DIRS[@]}" \
  --output research_notes/E1/E1-6B/supplement-002

e1_python research_notes/E1/E1-6B/make_factorial_effects.py \
  --source research_notes/E1/E1-6B/analysis-001/all-methods.csv \
  --output research_notes/E1/E1-6B/factorial-001
```

主分析保留57行、全部3seed及分档均值/范围；补充分析保留更新数值、完整episode/partial episode、过滤分母和状态×过滤效应；factorial脚本保存45条同seed配对（四个简单对照+交互）及15条摘要。原始CSV与PNG/PDF/SVG图本地保存；公开只发布必要小CSV/报告，不上传完整模型/账本/日志/图片归档。

补充脚本算术检查（不生成虚构正式结果）：

```bash
e1_python research_notes/E1/E1-6B/supplement_analysis.py --self-check
e1_python research_notes/E1/E1-6B/make_factorial_effects.py --self-test
```

## 8. 完成证据与科学解释限制

应保存 Git/source/config/input/checkpoint/output SHA、完整真实命令与退出状态、12模型实际预算、测试报告、三档公共窗口审计及57轨迹冻结回放。公开精简报告只引用必要身份，完整私有文件清单和日志保持本地。原归档中的 provenance/hash/test记录绝不重写。

5%绝对目标不改；高压另报相对EDF额外违约与Carbon权衡。3seed是训练随机性，2× stress seed不创造新的真实到达形态。旧三动作PPO与新Plain同时改变动作/状态宽度/参数量/训练混合，是外部比较；严格2×2来自四组83维/四动作同网络同预算消融。Plain基础40维仍含历史压力和粗风险信号，新增43维是增量状态。

所有收益或失败结论须来自实际公共窗口成绩，不能由测试通过、参数变化、smoke return、动作频率或局部filter事件推出。本轮结束后停止，不追加某组预算、不更换风险公式、不访问Test。

最终只读审计命令（实际完整Validation通过后运行）：

```bash
e1_python research_notes/E1/E1-6B/scientific_audit.py
e1_python research_notes/E1/E1-6B/filter_diagnostic_audit.py
e1_python research_notes/E1/E1-6B/interface_capability_check.py
```

supplement-001保留初版图的右刻度裁切证据；最终supplement-002只调整页边距，全部科学CSV与001相同。现发布supplement绘图源为修正版本。在全新副本中直接生成002，不覆盖任何既有目录。辅助科学审计测试需先按字节复制所需test_scientific_audit.py/test_validation_driver.py/test_filter_diagnostic_audit.py到对应ignored notes路径；原微型tests/e1从公共仓库直接运行。
