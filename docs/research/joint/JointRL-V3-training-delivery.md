# JointRL-V3训练交付记录

本文件由真实运行记录生成；当前流水线状态：`prepared`。未结束的训练、未生成的对照与消融均不计为完成。

## 冻结身份与预算

- 源码身份：`968d4650f11b3ec16beacc0b10ffd25f849265b15b94736963876d4ae65030c4`。
- 场景池manifest SHA256：`327e941eaa715aa9d25a30601502183cae27c1e7a83b0da7f31e2c0e251775ca`。
- 随机种子：20261017、20261018、20261019。
- 完整方法及五个机制消融分别训练，每组3个种子，共18个有限作业。每作业目标65,536物理步、上限21,600秒；数据加载、预测器拟合、辅助初始化、训练和开发选择均计时。仅提交完整Episode；余量不足时丢弃尾部，实际步数可能小于目标。
- 只用Train拟合CI预测器；只用两条未增强June Development场景选择模型。最终对照使用这两条场景的3压力×2 SLA预设敏感性矩阵；这些增强不是独立实测。
- July未用于本轮训练/选择/最终检查；September未读取。

| 训练版本 | 种子 | 当前阶段 | 已提交物理步/目标 | 完整Episode |
|---|---:|---|---:|---:|
| full | 20261017 | pending | 0/65,536 | 0 |
| full | 20261018 | pending | 0/65,536 | 0 |
| full | 20261019 | pending | 0/65,536 | 0 |
| no_queue_risk | 20261017 | pending | 0/65,536 | 0 |
| no_queue_risk | 20261018 | pending | 0/65,536 | 0 |
| no_queue_risk | 20261019 | pending | 0/65,536 | 0 |
| no_carbon_features | 20261017 | pending | 0/65,536 | 0 |
| no_carbon_features | 20261018 | pending | 0/65,536 | 0 |
| no_carbon_features | 20261019 | pending | 0/65,536 | 0 |
| no_robust | 20261017 | pending | 0/65,536 | 0 |
| no_robust | 20261018 | pending | 0/65,536 | 0 |
| no_robust | 20261019 | pending | 0/65,536 | 0 |
| independent_critic | 20261017 | pending | 0/65,536 | 0 |
| independent_critic | 20261018 | pending | 0/65,536 | 0 |
| independent_critic | 20261019 | pending | 0/65,536 | 0 |
| no_teacher | 20261017 | pending | 0/65,536 | 0 |
| no_teacher | 20261018 | pending | 0/65,536 | 0 |
| no_teacher | 20261019 | pending | 0/65,536 | 0 |

## 命令与恢复

在已审计的WSL Python环境、仓库根目录执行：

```bash
export PYTHONPATH=/mnt/d/A-myprofile/SustainCluster/upstream/research_notes/E1/E1-5B/runtime-deps
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m experiments.joint_v3.pipeline --run --protocol /mnt/d/A-myprofile/SustainCluster/publication/sustaincluster-carbon-scheduling-001/research_notes/joint_v3/formal-001/protocol.json --jobs 3
python -m experiments.joint_v3.report --campaign /mnt/d/A-myprofile/SustainCluster/publication/sustaincluster-carbon-scheduling-001/research_notes/joint_v3/formal-001 --docs docs/research/joint
```

流水线恢复拒绝变动源码/配置/数据身份，也拒绝重复启动仍存活的工作进程。单作业精确恢复命令为 `python -m experiments.joint_v3.train --pool research_notes/joint_v2/scenarios-001 --output <原training目录> --config <原冻结config文件> --resume`。不增加原累计时间/步数预算。

不可变Checkpoint保存在各作业`training/checkpoints`；指针包含权重字节、源码、配置与场景身份。完整状态包括两个Actor、四个条件价值输出及优化器、RNG、场景采样、六组约束系数、固定课程状态、教师衰减进度、预测器和完整Episode进度。当前采用固定学习率，配置与Adam状态共同保存其身份。

## 验收证据与限制

工程验收acceptance-001完成2个Train Episode、241物理步，处理7,892个任务，DMR0、无温度违规。此短程结果不属于正式训练，也不用于方法优越性或最终模型选择。验收预测器去重得到16条Train环境、1,536条CI观察；时间隔离拟合1,108条样本、校准281条完整残差向量，校准未来CI RMSE29.916 gCO₂/kWh。64个实际制冷验收状态均在18°C取得最低功率；正式初始化目标为4,096条去重状态，实际数量及不足记录在各作业pretraining-report.json。

风险/碳估计可信覆盖率可能很低；超出队列/时隙预算时保守回退会显著限制任务柔性。正式训练记录原因分布、估计耗时、教师覆盖及实际PPO更新；不能将这些回退掩饰为可靠预测。June Development已在V2多次使用，不是独立泛化证明。65,536步是有限预算目标，不代表已收敛或必然优于V2。

训练收敛、碳排/DMR、KL/梯度/熵、组系数、预测误差、制冷分布和最佳可行模型均以各作业CSV、episode-evidence.json和Checkpoint指针为准。当前未完成项见状态表，不补造结果。

## 图表与数据

`training/episodes.csv`为真实训练曲线数据，`python -m experiments.joint_v3.monitor --output <training目录>`生成PDF/SVG/PNG。最终对照图与源数据位于本地campaign/analysis。所有数值图只画真实记录；缺失种子不补值，只有3个真实种子均完成才给sample std。
