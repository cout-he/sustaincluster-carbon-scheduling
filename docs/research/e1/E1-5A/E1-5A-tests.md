# E1-5A 测试与保护证据

最终结果直接从已完成日志与新验证归档提取；未训练 RL。

| 检查 | 实际结果 |
|---|---|
| 原科研目录全部 E1 测试 | 276 passed in 1233.85s (0:20:33) |
| 独立发布副本全部适用 E1 测试 | 274 passed, 2 skipped in 685.14s (0:11:25) |
| 新逐拍/环境微型测试 | 34 passed in 1.70s，rl-green-003.txt |
| 封存真实输入历史数值回归 | 14/14 exact，12/14 八文件字节一致 |
| 随机完整 rollout | 14914/14914 complete，215拍，terminated=True |
| 随机奖励审计 | 53个 miss ID 唯一且等于账本全部missed；本拍奖励碳与物理积分一致 |
| 既有文件与原 Git 保护 | 164554个既有文件无变化，原.git字节/分支/状态一致；engine为单独授权改动 |

发布副本跳过两项需要未发布私有历史档案的测试，未将skip计为pass。旧E1-4A原科研目录有225项通过测试，本轮新增51项：11个共享内核、23个Gymnasium环境、17个独立验证驱动用例。两份完整suite均包含这些新增行为测试。

覆盖：修改前六策略黄金结构、batch/stepwise同内核、全队列EDF/回填/即时容量变化、资源与任务守恒、非抢占及精确finish/边界release、exclusive cutoff无新增到达、持续120min历史、未来workload/CI/weather分叉因果传感、首次违约与late-completion不重复罚分、真实kg/kWh/task-minutes奖励和censored负债、终止/截断/公共空闲积分、fresh persistent power/reset与幂等close、同seed重复交互、Gymnasium checker、随机完整cohort、完整封存清单/linked provenance/篡改拒绝。当前形式的局部测试和旧结果回归不构成安全保证或物理模型准确性的证明。

中间失败保留：未实现时的kernel/rl/driver RED；奖励fixture最初同时到达让EDF优先启动另一任务，随后修正fixture为运行后再到达，未修改算法迎合预期；一次并行源编辑触发source-stability拒绝，冻结后重跑；独立审查发现partial-cohort false drain、history120配置冲突、numeric-string canonicalization、零拍result及重复close，均以RED→GREEN修正；CSV的0/0.0仅用无容差Decimal核验，另有任意小真实数值差被拒绝的测试。旧回放器、历史expected输出和provenance未放宽或改写。

## 实际命令（Windows PowerShell）

原科研目录：

```powershell
wsl -d Ubuntu-22.04 --cd /mnt/d/A-myprofile/SustainCluster/upstream -- env PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 GIT_OPTIONAL_LOCKS=0 /home/hwj/miniconda3/envs/sustaincluster/bin/python -m pytest -q tests/e1 -p no:cacheprovider --basetemp research_notes/E1/E1-5A/full-tests-001
```

发布副本：

```powershell
wsl -d Ubuntu-22.04 --cd /mnt/d/A-myprofile/SustainCluster/publication/sustaincluster-carbon-scheduling-001 -- env PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 GIT_OPTIONAL_LOCKS=0 /home/hwj/miniconda3/envs/sustaincluster/bin/python -m pytest -q tests/e1 -p no:cacheprovider --basetemp /mnt/d/A-myprofile/SustainCluster/upstream/research_notes/E1/E1-5A/publication-tests-001
```

真实封存回归与smoke：

```powershell
wsl -d Ubuntu-22.04 --cd /mnt/d/A-myprofile/SustainCluster/upstream -- env PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 GIT_OPTIONAL_LOCKS=0 /home/hwj/miniconda3/envs/sustaincluster/bin/python -m experiments.e1.rl_validation --e1-3 research_notes/E1/E1-3/results-001 --e1-4a research_notes/E1/E1-4A/results-001 --output research_notes/E1/E1-5A/results-001 --seed 20261012
```

环境 Python3.10.22、Gymnasium1.4.0、NumPy2.2.6、pandas2.3.3、pytest9.1.1。Python及依赖完整清单、155文件源SHA和配置/数据哈希保留本地。新provenance的144个输出哈希全部复核；历史来源未重写。数据覆盖审计使用单独的Windows Python3.13.9/pandas2.3.3/NumPy2.3.5，只读数据统计，未用于策略仿真；两种环境角色不可混淆。

完整日志、临时测试档和真实账本仅保留本地E1-5A目录，GitHub发布本精简证据与[验证摘要](validation-summary.json)。复跑必须换全新输出目录；公开副本不包含重放旧真实实验所需的私有封存档。
