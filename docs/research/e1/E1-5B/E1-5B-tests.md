# E1-5B 测试证据与验证边界

本记录基于实际命令输出：原科研工作区完整测试 **324 passed，1172.81 秒**，退出码 0；新增行为测试与既有执行器/逐拍环境预检查 **79 passed，12.90 秒**。独立发布副本测试实际通过：320 passed、2 skipped、675.66秒，exit0。两个跳过用例需要未公开私有历史归档；不是新增PPO行为测试跳过。

| 检查 | 实际结果 | 本地证据 |
|---|---:|---|
| 新增输入冻结、PPO 更新及公平评测，加既有执行/逐拍/RL 环境回归 | 79 passed / 12.90 s | `precheck-tests-002.txt` |
| 原科研目录全部 `tests/` | 324 passed / 1172.81 s，exit 0 | `full-tests-001.txt` |
| 正式 Train 预检查 | 3 档 × 6 个运行；固定动作 0/1/2 全字段精确等价 | `precheck-001/decision.json`、各档 `equivalence.json` |
| 独立只读输出核验 | 18 个 Train 运行及 1024 步 smoke 产物通过 | `evidence-audit-001.json` |
| 发布副本全部E1加上游模型测试 | 320 passed / 2 skipped / 675.66s，exit0 | `publication-tests-001.txt` |

新增 45 个行为测试：输入冻结 12 项、PPO 训练 9 项、公平评测 24 项；并入既有 E1 276 项和上游 3 项测试，总计 324 项。本次旧测试通过不代表每一项旧测试都能证明机制正确。

## 实际运行命令

以下命令在原科研工作区根目录、WSL Ubuntu-22.04 执行；SB3 2.9.0 从独立本地目录加载，不升级 SustainCluster 原有 Conda 依赖。

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 GIT_OPTIONAL_LOCKS=0 \
  PYTHONPATH=/mnt/d/A-myprofile/SustainCluster/upstream/research_notes/E1/E1-5B/runtime-deps \
  /home/hwj/miniconda3/envs/sustaincluster/bin/python -m pytest \
  tests/e1/test_rl_training_utils.py tests/e1/test_rl_training.py \
  tests/e1/test_rl_evaluation.py \
  tests/e1/test_stepwise.py tests/e1/test_rl_env.py -q -p no:cacheprovider \
  --basetemp=research_notes/E1/E1-5B/precheck-unit-002

env PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 GIT_OPTIONAL_LOCKS=0 \
  PYTHONPATH=/mnt/d/A-myprofile/SustainCluster/upstream/research_notes/E1/E1-5B/runtime-deps \
  /home/hwj/miniconda3/envs/sustaincluster/bin/python -m pytest \
  tests -q -p no:cacheprovider \
  --basetemp=research_notes/E1/E1-5B/full-pytest-001
```

临时目录为本轮全新目录。复跑须换新名字，避免删除或覆盖历史测试证据。WSL 日志头部存在宿主代理提示及 NUL 字符，pytest 末尾结果和退出码均已实际核对。

## 新增机制测试覆盖

输入入口在读取 workload/environment 或创建物理模型前拒绝 Test 与角色错配。重新封存的错误 SLA、风险参数、物理配置也须被拒绝；Train 与 Validation 的冻结任务、UTC 时段及身份不同。PPO-v0 保持默认 Reward、工程归一化、新 CI 阈值 161.075，reset 新建物理对象，step 连续使用其状态。

真实 PPO 微型训练验证参数 SHA 改变、优化器状态随模型保存、同 seed/同依赖/单线程 CPU 的参数和动作奖励序列重现、保存加载后确定性推理一致，以及真实 rollout 的 Task/Resource 守恒。训练步数不能被 SB3 静默向上取整。正式完整 Drain 若出现安全截断则终止训练并保存失败证据；专门的微型截断诊断验证未完成资源负债与首次违约去重，不能混入正式完成工作量统计。

公平评测验证原始 EDF、EDF+Carbon、RiskOnly 与固定 Action 0/1/2 的完整 `ExecutionResult` 精确一致；所有策略先各自 Drain，再使用各自尚存的物理状态继续到共同终点。空闲尾段计入 IT/HVAC 电能和碳排放。安全截止时未完成、deadline 已违约与右删失分别保存，未排空结果不允许宣称等工作量减排。

同一当前历史、不同未来任务与 CI 的分叉测试，要求相同的当前 Observation、动作日志和本拍能碳/资源执行结果。actor 实際只收到 40 维当前传感向量。模型评测还要求显式 checkpoint SHA，并校验训练摘要、全部训练产物、配置 SHA、Train 身份、固定 Reward/归一化、实际 seed/步数、40 维输入及 Discrete(3)、训练后参数 SHA；角色和参数修改即使重新计算内部哈希也会被拒绝。

## 上游弱测试的边界

`tests/test_training_loop.py::test_train_loop_runs_short` 使用 `os.system`，未检查子进程退出码，也没有训练更新、模型文件或任务统计断言。临时算法配置在离开 `TemporaryDirectory` 后已经删除；调用的 `train_SAC_single_action.py` 在当前仓库不存在。因此该用例显示 passed 不能证明 SAC/PPO 训练或调度机制有效。本轮保留原文件，明确披露此弱覆盖，不据此声称完成训练。

其余上游模型前向形状及 environment reset 测试也只证明各自有限接口。资源分配、时序、SLA、能碳及 RL 参数更新的主证据来自本轮行为测试、旧 E1 关键回归和冻结实验账本核验。

## Smoke 外层日志错误

1024 步 smoke 已完成实际参数更新、检查点保存、优化器和确定性回放；随后笔记目录驱动在打印摘要时读取不存在的 `process_peak_rss_kib`，实际键名为 `peak_rss_kib`，使外层命令退出码为 1。原日志和原驱动副本保留；仅修正笔记驱动的字段引用，没有重复训练、修改模型或改变被冻结的研究源码。`smoke-recovery.json` 记录恢复核验及两个驱动身份。该命令的非零退出不隐瞒，也不应误写为 PPO 学习失败。

## 适用范围

同 seed 可复现承诺仅限相同输入、源码、依赖及确定性单线程 CPU，不承诺跨平台浮点位级一致。普通 PPO 不具备 SLA 安全保证；通过测试也不表示优于 RiskOnly。本轮 Test 不训练、不评测，最终论文指标仍使用公共观察窗口的碳排放、违约率和 JCT。

## 完整确定性动作日志核验

重新载入三个已认证checkpoint，对保存的40维Validation观测逐项预测：213/215/215拍，共643拍，全部与原动作日志精确一致。没有推进模拟器或生成新CI/任务，这验证推理日志可重现，并非再次完整物理rollout；证据 `validation-action-replay-001.json`。

## 独立发布副本最终验证

实际日志 SHA256=`9ae63cf8f8295d297aa855bdadf7a6b4c39cf45adabd305109695ab27bbbac7d`。全部E1测试及上游模型前向测试在发布源码运行，两个历史归档测试因私有输入缺失明确跳过。没有运行依赖原始多DC数据的上游env_setup，也没有用弱train_loop作为训练证明。

```bash
python -m pytest tests/e1 tests/test_model.py -q -p no:cacheprovider --basetemp=/mnt/d/A-myprofile/SustainCluster/upstream/research_notes/E1/E1-5B/publication-pytest-001
```

WSL cwd为 `/mnt/d/A-myprofile/SustainCluster/publication/sustaincluster-carbon-scheduling-001`，Python/PYTHONPATH沿用已记录兼容环境及隔离SB3。只读接口核验确认从发布副本导入模块，可加载外部同一11084-task Train包，并完成40维reset和一次Action0 step；并非新科研实验，也未读取Test。公开库仍依赖外部workload/CI/weather及本地checkpoint，不能宣称自包含。

DC JSON的raw SHA恢复为bbe439…；解析值与main完全相同。原文件第276行已有空格空行，为保留批准raw字节没有清除；该路径忽略行尾差异的diff为空，其余文件`git diff --check`通过。这是记录的字节兼容例外，没有改动物理值或原工作区。
