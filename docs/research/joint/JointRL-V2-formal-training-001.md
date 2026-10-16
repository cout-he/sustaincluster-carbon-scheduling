# JointRL-V2 第一次正式 B3 训练结果

## 1. 结果与结论

2026-10-16，按指定命令完成一次正式训练，进程退出码0。实际训练耗时 **3885.36秒（64分45.36秒）**，达到32768个联合训练采集步上限后正常停止，未追加第二次训练预算。完成271个完整episode，提交32757个物理步，累计完成884362次训练任务执行，3082次Deadline违约，按任务加权DMR **0.3485%**；已提交episode的完成率100%，未完成、右删失、温度违规均为0。

**最佳可行模型为第52轮**：两个预声明Development场景分别满足任务全部完成、DMR≤5%和零温度违规，合计5684/5684任务完成、DMR0、固定窗口总碳 **157708.40kgCO2e**、增量碳 **3340.86kgCO2e**。相对第1轮更新后的开发检查，总碳下降0.3947%、增量碳下降15.7582%。二者分母不同，不能把15.7582%写成系统总碳降幅。

**训练效果未持续改善。** 最后一次开发检查在第268轮，总碳158367.09kg，比第1轮还高0.0213%；最终第271轮模型没有额外开发评价。Task/Cooling四个网络组件持续更新，但这不等于收敛。最佳模型有开发集内部改善信号；**独立减碳效果尚未证实**：本轮没有未训练/规则基线对照，没有新增独立Validation或Test评价，开发集又参与反复模型选择。

## 2. 执行边界与实际命令

训练源码提交为[`3520819c1e6838257f4713158229552993de1c4f`](https://github.com/cout-he/sustaincluster-carbon-scheduling/commit/3520819c1e6838257f4713158229552993de1c4f)，工作分支`research/joint-v2-training`，启动前本地与远端一致，工作区干净。冻结场景池存在，`formal-001`此前不存在，因此本次为新输出，未覆盖或删除旧训练。

复用WSL Ubuntu-22.04、conda `sustaincluster`、Python3.10.22、PyTorch2.14.1+cu130、SB3 2.9.0及已有runtime-deps。PPO沿用CPU执行及指定单线程环境；没有安装、更换依赖。实际执行：

```bash
cd /mnt/d/A-myprofile/SustainCluster/publication/sustaincluster-carbon-scheduling-001
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPATH=/mnt/d/A-myprofile/SustainCluster/upstream/research_notes/E1/E1-5B/runtime-deps
set -o pipefail
/home/hwj/miniconda3/envs/sustaincluster/bin/python \
  -m experiments.joint_v2.pipeline \
  --pool research_notes/joint_v2/scenarios-001 \
  --output research_notes/joint_v2/formal-001 \
  --config experiments/joint_v2/training_config.yaml \
  --total-steps 32768 --wall-seconds 7200 --no-heldout-final \
  2>&1 | tee research_notes/joint_v2/formal-001.log
```

State、Action、Reward、PPO、Deadline保护、物理方程、配置和冻结数据均未修改；未运行额外测试、B0/B1/B2/RiskOnly或额外模型rollout。原7月Validation场景未加载，`heldout_validation=null`且本输出没有一次性检查标记；9月Test继续封存。6月Development按既有协议进行模型选择，其预先冻结的安全尾段维持原样。

预算口径：本次联合训练实际尝试32768步，最后一个场景`train-20230204T0500-r0.75-s1.5`只收集了11步，未更新参数，按原协议回滚模型/RNG/采样器。因此最终提交32757步，而不是伪造一个短成功episode；下文训练DMR、完成率及任务数只统计271个完整已提交episode，不声称该11步前缀的任务全部完成。另有Task示范249步、Cooling示范241步、Cooling初始化PPO32步、已完成Development评价16942步，分列而不混入联合训练量。整体pipeline耗时3885.50秒，低于7200秒。

## 3. 数据覆盖、初始化与课程

实际覆盖 **16个不同原始任务源片段、83/96个Train场景配置ID**，其中12个基础配置、71个额外资源/SLA增强配置。240/271轮带有额外合成增强标记。884362是跨场景和重复抽样的任务执行累计，不是884362条独立实测任务；16片段仍来自同一约7周原始轨迹。基础配置的Deadline本身也采用已声明的合成SLA。

池、种子及配置均沿用冻结版本，根/采样seed20261016，Task/Cooling为根seed+1/+2。未重新生成任务、打乱到达或变更压力因子。

Task预训练观察EDF示范后按既定规则跳过BC，原因是自由选择的示范缺少wait类别，并非训练中途改变初始化方案。Cooling使用128个真实轨迹驱动状态，完成2次BC更新和32拍/1次PPO初始化，随后进入联合训练。课程在第2轮后升至阶段1、第4轮后升至阶段2，最终仍为阶段2。自适应Deadline乘数从10出发，最终为0；259个episode提交后为0，全部271次更新均来自完整Train统计，没有利用开发或Test调系数。该动态并未保证每个Train场景DMR≤5%。

## 4. Deadline、任务完成与推迟行为

以下按episode顺序分成前三分之一、中三分之一、后三分之一；DMR以任务数加权，等待和JCT按任务数加权。三组场景混合不同，变化只能作为描述性监控，不能归因为学习进步。

| 阶段 | Episode | 完成/到达 | 违约数 | DMR | 完成率 | 平均等待min | 平均JCT min |
|---|---|---:|---:|---:|---:|---:|---:|
| 前段 | 1–90 | 298121/298121 | 821 | 0.2754% | 100% | 10.0573 | 105.1538 |
| 中段 | 91–180 | 292692/292692 | 1692 | 0.5781% | 100% | 8.7175 | 102.7858 |
| 后段 | 181–271 | 293549/293549 | 569 | 0.1938% | 100% | 9.9676 | 105.1334 |
| 全部 | 1–271 | 884362/884362 | 3082 | 0.3485% | 100% | 9.5841 | 104.3633 |

每组未完成、右删失、温度违规均为0。269/271轮DMR≤5%；两轮超标都来自`train-20230404T0500-r1.25-s1.5`：第127轮327/3873违约（8.4431%），第175轮332/3873违约（8.5722%，本轮最大）。不能以整体低DMR隐藏这些失败。

报告专用必要容量/期限下界对已访问配置均为0；零下界不证明可行，不能据此把上述违约全部判为算法可避免，也不能反过来宣称已证明容量不可行。本轮没有追加基线或离线实验来分解原因。Deadline预约保护触发和冲突均为0，因此本轮也没有提供该额外保护实际改善结果的证据。

Task共1428691个候选决策：start884362次、自由wait544329次、强制start266401次；自由决策共1162290次，**主动等待占自由决策46.8325%**，占全部候选决策38.0998%。前/中/后段自由等待率分别47.6346%/44.4603%/48.1858%，随机训练策略没有始终立即启动。容量等待累计20727个任务-拍，策略推迟累计544329个任务-拍（每拍15分钟）。

**唯一主动推迟任务数不可从现有产物还原。** 现有聚合日志保留动作次数和任务-拍，没有持久化可去重的推迟任务ID/计数；即使保护触发为0，也不能把544329次等待冒充544329个不同任务。轻量元数据将该指标记为null，Development CSV留空。本轮遵守源码冻结要求，没有补改日志或重跑训练。

## 5. 固定Development结果与最佳模型

共完成68次检查（第1轮及每4轮一次），每次使用同一对June05/June20场景，两个场景均满足可行性条件；所有检查的合计及单场景DMR均为0、完成率100%、零温度违规。没有使用Train分数替代开发筛选。

| 检查点 | Episode | 总碳kgCO2e | 增量碳kgCO2e | IT碳kgCO2e | HVAC碳kgCO2e | DMR |
|---|---:|---:|---:|---:|---:|---:|
| 首次开发检查，已更新1轮 | 1 | 158333.34 | 3965.80 | 95402.70 | 62930.64 | 0% |
| best_feasible / best_deadline | 52 | 157708.40 | 3340.86 | 94945.99 | 62762.41 | 0% |
| 最后已评价模型 | 268 | 158367.09 | 3999.56 | 95424.01 | 62943.08 | 0% |

相同运营窗口的背景碳固定为154367.54kg，因此最佳模型较首轮总碳和增量碳的绝对差均为−624.94kg；相对总碳为−0.3947%，相对增量为−15.7582%。第268轮相对首轮总碳为+0.0213%、增量为+0.8511%，相对最佳模型回升658.69kg。最终第271轮没有被另行评价，不能直接把第268轮成绩贴给final。

最佳模型的两个场景分别为：

| 固定场景 | 完成/到达 | DMR | 总碳kgCO2e | 增量碳kgCO2e | 相对首次总碳 |
|---|---:|---:|---:|---:|---:|
| June05 | 2869/2869 | 0% | 73401.73 | 1718.98 | −0.5097% |
| June20 | 2815/2815 | 0% | 84306.67 | 1621.88 | −0.2943% |

每个检查点的逐场景总碳、增量、IT/HVAC、DMR及等待统计见[Development轻量CSV](JointRL-V2-formal-training-001-development.csv)。首个参照已经训练1轮，既不是未训练模型，也不是EDF/B0基线。反复选择的小开发集不能当独立泛化证据；本轮没有证明与其他算法相比的减碳幅度，也没有证明随训练持续改善。

## 6. 两个智能体是否学到有效策略

271个episode中，Task/Cooling的Actor和Critic每轮都发生有效参数变化，四个组件各有271个不同的轮后参数哈希，最终均不同于初始化及预训练状态。联合训练更新数如下，不把初始化更新重复计入：

| Agent | 有效Actor更新 | 有效Critic更新 | 零有效Actor的轮数 | KL早停轮数 | 数值保护触发轮数 |
|---|---:|---:|---:|---:|---:|
| Task | 2193 | 2196 | 0/271 | 0/271 | 0/271 |
| Cooling | 2196 | 2196 | 0/271 | 0/271 | 0/271 |

Task只对有自由动作的batch更新Actor，因此比Critic少3次。Task轮级平均KL均值0.002096、最大0.014244，clip fraction均值0.021516、最大0.175094；Cooling KL均值0.000087、最大0.001256，clip fraction为0。联合log-ratio范围Task[−1.074,1.059]、Cooling[−0.152,0.139]。裁剪前全策略梯度范数峰值分别47.517和16.540，沿用原裁剪配置。持久化监控没有NaN/Inf，没有异常进程停止或频繁KL早停；这些数值诊断不能证明控制策略已经收敛。

固定开发场景提供了比随机Train动作更直接的行为证据：

- 第1轮：Task两个场景都没有主动等待，平均等待0；Cooling各降温1次后保持119次，主要停留21°C。
- 第52轮最佳模型：Task在June05/June20分别有1427/522次wait，平均等待7.4608/2.7815分钟，DMR仍为0；Cooling各降温4次、保持116次，到18°C后保持。联合策略确实形成了开发集上的不同执行行为，并伴随更低碳排。
- 第268轮：Task两个场景又都没有主动等待，平均等待回到0；Cooling各降温4次、升温4次、保持112次，June05/June20分别85/120和91/120个真实控制拍位于22°C。该变化伴随开发碳回升，表明后期策略质量回退；不能用参数持续更新替代有效学习结论。

本轮没有消融，因此不能把IT碳变化单独归给Task、把HVAC碳变化单独归给Cooling。较少等待、更低温度与总碳之间的联合贡献也未被单独识别。

## 7. 制冷分布与能碳核算

随机训练的Cooling动作并非固定温度：降温9824次、保持13227次、升温9223次。只统计真实Actor控制的32274拍，排除483拍确定性收尾及全部解析空闲尾段，得到：

| 设定值°C | Actor控制拍数 | 占比 |
|---:|---:|---:|
| 18 | 5747 | 17.81% |
| 19 | 7348 | 22.77% |
| 20 | 6838 | 21.19% |
| 21 | 7212 | 22.35% |
| 22 | 5129 | 15.89% |

前/中/后段18°C占比分别16.28%/17.99%/19.14%，整体仍较分散。不能使用含数千拍解析idle18尾段的`summary.setpoint_distribution`来声称Cooling已几乎固定18°C。最佳确定性开发策略趋于18°C边界，后期又更多处于22°C；没有形成持续改善的稳定低功率控制。最高机架入口温度27.15°C，低于冻结配置28°C限制，所有已提交轮次温度违规为0。

271个Train场景固定窗口核算总和为：IT碳16180892.65kg、HVAC碳10924487.87kg、总碳27105380.52kg、增量碳737510.29kg；IT电量83473846.83kWh、HVAC电量56192050.74kWh。它们是重复抽样和不同环境窗口的仿真核算累计，不是连续真实运营排放，也不是减碳证据。固定开发比较采用相同背景与相同任务，才适合解释检查点间差异。

## 8. Checkpoint、可恢复性与后续命令

实际保留283个SHA命名代文件，合计约1.13GiB；逐文件字节SHA均核对成功。四个命名指针可通过既有`CheckpointStore`安全加载，配置/数据/源码身份一致，两Agent策略、优化器、独立RNG、全局Python/NumPy/Torch RNG、采样器、课程、乘数、计数和最佳模型记录完整。

| 别名 | Episode | 代文件SHA256 |
|---|---:|---|
| latest / final | 271 | `14d432011ccbe9c3022918436fa6566df0e15fb0499824d8b1ae2e44d1a5ec2e` |
| best_feasible / best_deadline | 52 | `5f9b8fde4cc1e6713039e0fba1103dbb42ec42fd7a0517f299c8db0747fb122a` |

本轮确认了产物可加载、身份/哈希及恢复字段完整；没有额外启动恢复训练或新增恢复测试。最后11步前缀未提交，后续resume从其前面的完整状态继续，不能把该前缀当训练成果。

下面仅为**后续继续训练的示例，未在本轮执行**。65536是新的累计上限，7200是另一次调用预算；当前32768步这轮已结束，不会自动追加：

```bash
# 使用本报告第2节相同的WSL目录和环境变量
set -o pipefail
/home/hwj/miniconda3/envs/sustaincluster/bin/python \
  -m experiments.joint_v2.pipeline \
  --output research_notes/joint_v2/formal-001 --resume latest \
  --config experiments/joint_v2/training_config.yaml \
  --total-steps 65536 --wall-seconds 7200 --no-heldout-final \
  2>&1 | tee -a research_notes/joint_v2/formal-001.log
```

必须保留同一源码字节、配置和冻结pool。`--no-heldout-final`继续必需；不能因新输出/恢复而重新查看已经使用过的7月Validation。日志采用追加，保留已有记录；输出目录不删除、不覆盖。

## 9. 产物、图表与版本身份

本地完整产物根目录：

`D:/A-myprofile/SustainCluster/publication/sustaincluster-carbon-scheduling-001/research_notes/joint_v2/formal-001/`

- `training/run-result.json`、`pipeline-run-result.json`：实际停止原因与预算统计。
- `training/checkpoints/`：命名指针和模型代文件；`training/pretrain/`：初始化模型与统计。
- `training/monitor/episodes.csv/json`、`steps.csv/json`、`ppo_updates.csv`、`validation.json`、`selection.json`、`capacity_audits.json`、`tensorboard/`：完整监控。
- `training/monitor/training-curves.png`：实际生成且已打开核验的六面板图，包含DMR、完成率、固定总碳/增量、Task动作、制冷设定值、PPO损失。图中Train场景不同，碳曲线不可直接解释为学习减碳趋势；动作图也包含强制动作，准确自由等待率以上表为准。
- 上一级`research_notes/joint_v2/formal-001.log`：原始运行日志。

本地图表：[training-curves.png](D:/A-myprofile/SustainCluster/publication/sustaincluster-carbon-scheduling-001/research_notes/joint_v2/formal-001/training/monitor/training-curves.png)。图表、日志、模型、原始数据全部留在gitignored独立输出目录，本次Git仅提交本报告和[轻量JSON元数据](JointRL-V2-formal-training-001.json)、[Development CSV](JointRL-V2-formal-training-001-development.csv)。JSON包含逐轮摘要、逐开发检查结果、动作分布、更新诊断和checkpoint字节核验记录；不包含模型权重或原始任务/天气/CI序列。

重要身份：

- 训练源码Git：`3520819c1e6838257f4713158229552993de1c4f`。
- 源码manifest：`efd597aea9b786cffc95e0bde63913876b660e7d194f9b3573445f263a754a22`。
- 配置身份：`72d921b7b48205ed92c051a929a503a6f1d02555cb146bad8def78d248aaf606`。
- Train/Development数据身份：`cfffda6fa42a2d88340335e99cb807ccb384d238bfbd384d2da330fd229e13d9`。
- 场景池manifest SHA：`327e941eaa715aa9d25a30601502183cae27c1e7a83b0da7f31e2c0e251775ca`。

训练后与起始提交逐文件核对，实验源码和配置无变更。本轮获得了可恢复、可审核的首次正式训练结果：最佳开发候选合格，但后期效果回退，唯一推迟任务数缺失，独立减碳与收敛仍未得到证明。
