# E1-4A 验证记录

2026-10-11 实际运行，以下进程均退出 0，最终源码冻结后无 Python/测试/设计配置修改。

| 验证 | 最终结果 |
|---|---|
| 原科研目录全部 E1 测试 | **225 passed in 1147.72s (0:19:07)** |
| 独立发布副本全部适用 E1 测试 | **223 passed, 2 skipped in 648.60s (0:10:48)** |
| 上游非训练模型/环境测试 | **2 passed in 5.56s** |
| 原 E1-3 EDF+Carbon/RiskOnly/Full 科学与诊断文件 | 两档各 36，合计 **72 个文件字节一致** |
| 新两档五策略严格冻结回放 | campaign 7 + 两个子档各 112 = **231 个科学输出哈希一致** |
| 旧工作区/历史档案保护 | **81,212 个既有文件无变化，原 .git 字节及 Git 状态一致** |
| 五策略公平性/终态 | 同档相同输入/共享配置/资源小时，全任务完成；资源不超容量，公共终点含空闲积分 |

新增 38 个测试覆盖：原 RiskOnly 委托、不运行 Planner 的 CapOnly、256/257 边界、明确风险与 safe/uncertain fallback、历史已观察输入、资源不足回填与任务守恒、完整封存清单、篡改/缺失拒绝、旧/新公共终点两种顺序、冻结回放、诊断计数分母、一次干预绑定与恢复。旧 E1 测试继续通过。公开副本跳过两项需要未发布私有历史档案的测试，没有将跳过计为通过。

保留的中间失败：首轮 4 failed/215 passed，原因是新测试 fixture 将历史 `137.64999999999998` 写成名义 `137.65`；仅修正 fixture 读取已冻结参数，未放宽驱动约束。随后一次并行修改诊断源码导致 2 项 source-stability 校验失败；停止源码修改并冻结后重跑相关测试及最终整套通过。中间失败日志保留在原科研目录，不覆盖历史或当作最终结果。

实际最终测试命令（Windows PowerShell，原科研目录）：

```powershell
wsl -d Ubuntu-22.04 --cd /mnt/d/A-myprofile/SustainCluster/upstream -- env PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 GIT_OPTIONAL_LOCKS=0 /home/hwj/miniconda3/envs/sustaincluster/bin/python -m pytest -q tests/e1 --basetemp=research_notes/E1/E1-4A/tests-final-002 -p no:cacheprovider
```

实际发布副本命令：

```powershell
wsl -d Ubuntu-22.04 --cd /mnt/d/A-myprofile/SustainCluster/publication/sustaincluster-carbon-scheduling-001 -- env PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 GIT_OPTIONAL_LOCKS=0 /home/hwj/miniconda3/envs/sustaincluster/bin/python -m pytest -q tests/e1 --basetemp=/mnt/d/A-myprofile/SustainCluster/upstream/research_notes/E1/E1-4A/publication-tests-001 -p no:cacheprovider
```

上游仅执行 `tests/test_model.py` 与 `tests/test_env_setup.py`；没有执行会调用训练过程的 `tests/test_training_loop.py`，遵守不训练 RL 的任务边界。

原完整日志：`tests-final-002.txt`、`publication-tests-001.txt`、`upstream-tests-001.txt`、`run-001.txt`、`replay-001.txt`、`intervention-001.txt`、`preservation-final.txt`。均保留本地 `research_notes/E1/E1-4A/`；GitHub 仅发布本精简记录，不上传完整日志或测试临时档案。

算法计时另存 `runtime.csv`，与并行验证同时运行，不作为隔离硬件下的严格性能基准。测试与字节回放验证实现一致性，不能证明物理模型或研究结论的普适正确性。
