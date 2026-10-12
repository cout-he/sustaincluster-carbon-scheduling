# E1-5B 复现命令与外部数据边界

普通 PPO-v0 基于原 E1-5A 环境，不含任务级预算或 Safety Shield。只允许 Train 和 Validation，Test 继续封存。所有输出目录必须不存在；入口拒绝覆盖已有成果。

## 环境

使用 `dependencies.json` 中实际验证的环境。现有 SustainCluster 科学依赖保持不变，只隔离增加 `stable-baselines3==2.9.0`：

```bash
python -m pip install --no-deps --target /outside-git/ppo-deps -r requirements-e1-ppo.txt
export PYTHONPATH="$PWD:/outside-git/ppo-deps"
export PYTHONDONTWRITEBYTECODE=1
export GIT_OPTIONAL_LOCKS=0
python -m pytest tests/e1 tests/test_model.py -q -p no:cacheprovider --basetemp=/outside-git/new-pytest-001
```

`--no-deps` 不安装或升级 Torch、Gymnasium、NumPy 等；须先具备匹配依赖。完整模型 ZIP（含优化器）、训练逐拍日志、冻结输入、账本及完整图集只保留本地，不加入 Git。

## 数据准备

发布仓库不是自包含的数据或模型包。需要合法取得并本地准备 `E1-5B-data-freeze.md` 所列 workload pickle、CI CSV、天气 JSON，原始字节 SHA 必须匹配。不能以另一个采样/重处理版本冒充本报告数据。pickle 只加载可信来源。物理 JSON 随源码发布，本次研究分支保留原 CRLF 字节以满足冻结 SHA；数值与之前发布的 LF 版本完全相同。

复制配置到 Git 外，并仅设置外部绝对数据路径（不更改阈值、窗口、参数或源身份）：

```python
import json
from pathlib import Path
cfg = json.loads(Path('experiments/e1/configs/e1_5b_ppo.json').read_text())
cfg['paths']['workload'] = '/external-data/result_df_full_year_2020.pkl'
cfg['paths']['carbon'] = '/external-data/US-CAL-CISO_2023_hourly.csv'
cfg['paths']['weather'] = '/external-data/2023.json'
# 保留仓库内 configs/dcs/dc_config.json。
Path('/outside-git/e1-5b-local.json').write_text(json.dumps(cfg, indent=2))
```

路径变化会产生新的配置/冻结包哈希；不可改写本轮历史产物或把新包标为旧哈希。应另存当前 Git SHA、工作树状态和源码哈希；本轮原始工作树为 dirty，历史源内容快照是完整证据，不能只用 upstream HEAD 描述研究代码版本。

## 预检查

对 1/1.5/2× 分别执行下列入口（下例为 2×，另外两档用独立新目录）：

```bash
python -m experiments.e1.rl_training_utils --config /outside-git/e1-5b-local.json --split train --factor 2 --stress-seed 510001 --output /outside-git/new-train2-input-001
python -m experiments.e1.rl_evaluate --config /outside-git/e1-5b-local.json --bundle /outside-git/new-train2-input-001 --split train --fixed-actions 0,1,2 --output /outside-git/new-train2-precheck-001
```

原预注册规则选最高的、所有模式排空、出现 capacity waiting 且 `(DMR, mean_wait_minutes_started)` 存在差异的档位。本轮结果选择 2×；不是按 PPO 胜负选负载。固定动作的账本、资源、能碳及事件全部字段与对应直接规则精确比较。

## 分阶段训练

```bash
python -m experiments.e1.rl_train --config /outside-git/e1-5b-local.json --bundle /outside-git/new-train2-input-001 --seed 55100 --steps 1024 --output /outside-git/new-smoke-001
python -m experiments.e1.rl_train --config /outside-git/e1-5b-local.json --bundle /outside-git/new-train2-input-001 --seed 55101 --steps 20480 --output /outside-git/new-train-55101-001
```

核验 pilot 参数更新、有限奖励/状态、墙钟 ≤7200s、进程峰值 RSS ≤8GiB，再运行 55102/55103 各 20480 步，使用同一 Train 包、参数和新输出目录。55101 pilot 已是正式候选，不重复训练后挑好结果。模型参数 canonical SHA 和确定性动作可以在同环境比较；ZIP 时间元数据与跨平台浮点不承诺逐字节一致。学习预算尾部单独记录，不作为完整任务集合评价；真实环境安全截断使正式训练报错，禁止静默继续。

## 仅 Validation 比较

```bash
python -m experiments.e1.rl_training_utils --config /outside-git/e1-5b-local.json --split validation --factor 2 --stress-seed 520001 --output /outside-git/new-validation-input-001
```

为全部三个模型生成候选清单，包含 name、seed、checkpoint、checkpoint_sha256、training_steps=20480、training_elapsed_seconds，值必须读取对应 `training_summary.json`；checkpoint 可用绝对路径或相对清单目录。加载器还验证相邻配置、摘要、模型和产物哈希，不能仅自填名称。示例单条：

```json
[{"name":"ppo_v0_55101","seed":55101,"checkpoint":"/outside-git/new-train-55101-001/model.zip","checkpoint_sha256":"<actual SHA256>","training_steps":20480,"training_elapsed_seconds":0}]
```

实际须填真实用时并保留 55102/55103。然后运行：

```bash
python -m experiments.e1.rl_evaluate --config /outside-git/e1-5b-local.json --bundle /outside-git/new-validation-input-001 --split validation --candidate-manifest /outside-git/new-candidates.json --output /outside-git/new-validation-001
```

该组自动包含 EDF、EDF+Carbon、RiskOnly，三候选均确定性推理、完整 Drain，统一最慢排空终点并连续计算早排空者的空闲 IT/HVAC。仅完成同一任务集合且资源小时一致才能解释公共碳差异；按预注册全部完成且 DMR≤5% 筛选，无合格者报告 none。不按训练回报选模型。

本轮本地分析入口：

```bash
python research_notes/E1/E1-5B/analyze_results.py --root research_notes/E1/E1-5B --output research_notes/E1/E1-5B/analysis-003
```

发布版对应 `docs/research/e1/E1-5B/analyze_results.py`，`--root` 需指向本地完整 E1-5B 产物目录；公开精简 CSV 不含逐拍日志，不能替代完整复分析所需输入。代码会拒绝缺失候选、改变 SHA 或 Test 角色。历史复现仍需原记录字节/依赖，不弱化严格回放契约。

## 本轮实际 WSL 命令

科学运行均在原科研目录中，统一前缀：

```powershell
wsl -d Ubuntu-22.04 --cd /mnt/d/A-myprofile/SustainCluster/upstream -- env PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 GIT_OPTIONAL_LOCKS=0 PYTHONPATH=/mnt/d/A-myprofile/SustainCluster/upstream/research_notes/E1/E1-5B/runtime-deps /home/hwj/miniconda3/envs/sustaincluster/bin/python <arguments>
```

实际 arguments：`research_notes/E1/E1-5B/run_precheck.py`；`research_notes/E1/E1-5B/run_training.py --stage smoke`；`research_notes/E1/E1-5B/run_training.py --stage pilot`；`research_notes/E1/E1-5B/run_candidate.py --seed 55102` 与 `--seed 55103`（独立进程并行）；`research_notes/E1/E1-5B/run_validation.py`；`research_notes/E1/E1-5B/analyze_results.py --root research_notes/E1/E1-5B --output research_notes/E1/E1-5B/analysis-003`；`research_notes/E1/E1-5B/validate_evidence.py --output research_notes/E1/E1-5B/evidence-audit-002.json --require-complete --validation-input research_notes/E1/E1-5B/validation-input-001`。脚本参数中的相对路径均相对原科研根目录。各命令日志和失败记录完整保留本地；不覆盖历史结果。

Test 的角色/边界在入口拒绝。本轮没有 Test 运行命令、模型选择结果或策略指标。

最终后处理在新 `analysis-003/` 完成。`analysis-001/` 标签错误及 `analysis-002/` 拥挤标签版本均保留，八份共同科学CSV逐字节一致；只修后处理，不重训或重跑Validation。`replay_validation_actions.py` 对643个保存观测重载模型得到完全相同动作，不推进模拟器。
