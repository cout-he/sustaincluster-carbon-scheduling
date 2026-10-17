# JointRL-V3 身份修复与正式训练恢复记录

2026-10-17，本轮继续实施 RAC-MAPPO。独立审查发现评价运行配置未与 checkpoint 内完整配置绑定，以及未来 September 冻结入口未充分检查实际训练终止和 Development 产物。两项缺口已修复。September payload 始终未访问；原始 Test 尝试标记不存在。

## 修复、验证与身份迁移

`evaluate.py` 在反序列化权重前检查配置内容哈希和源码身份，加载后比较完整内嵌配置。`test_once.py` 在冻结及授权时复核全部 18 项训练/Development 产物、预算终止、规范对照清单、模型字节、协议指定的 V2 第 52 轮模型及固定 Test 窗口；检查全部通过后才创建一次性标记。授权仍须由用户另行明确给出并绑定最终 freeze 哈希。

本次运行代码仅修改上述两个文件。训练、抽样、物理模型、奖励、PPO、网络、超参数和预算均未修改。协调器停止后，使用 `scripts/migrate_rac_v3_guard_identity.py` 实施一次显式迁移，原始不可变 `.pt` 代次完整保留。迁移前后权重、Critic、优化器、RNG、场景抽样、约束和训练计数的科学状态摘要一致。记录见 [迁移证书](JointRL-V3-guard-migration.json)。

- 原源码身份：`968d4650f11b3ec16beacc0b10ffd25f849265b15b94736963876d4ae65030c4`。
- 新源码身份：`571c0501d26153eb0f0a6013784ebd408a170493eedf569c426e733d2dfaced2`。
- 迁移保留的物理步/Episode：种子 20261017 为 2,452/21，20261018 为 2,730/23，20261019 为 1,874/16。
- 每种子已耗时均至少计入协调器记录的约 784 秒，再继续消耗原 21,600 秒上限。暂停修复期间没有训练计算，不计作新增训练。
- 修改前显式科研回归 `tests/e1 tests/joint tests/joint_v2 tests/joint_v3`：884 passed、2 skipped。修复后 `tests/joint tests/joint_v2 tests/joint_v3`：179 passed，含 18 项新增身份防护测试及 3 项迁移测试。未将两轮重叠测试相加。
- 三个迁移后的最佳可行模型已通过严格加载；恢复后的工作进程存活，后续步数增长，检查到的 PPO KL、梯度范数和 Critic loss 均为有限数值。

## 已验证进度与当前模型

以下为 **2026-10-17 15:14:20（UTC+8）** 从严格校验的不可变检查点提取的快照，训练仍在后台继续。完整轻量身份见 [运行快照](JointRL-V3-live-snapshot.json)。每作业目标仍为 65,536 物理步、最长 6 小时；完整方法与五个重新训练的消融共 18 项作业，最多 3 路 CPU 训练/评价。当前三个完整方法作业运行中，其余 15 项等待队列调度。尚无完成的最终敏感性矩阵评价。

| 种子 | 已提交物理步 | 完整 Episode | 已提交累计秒 |
|---|---:|---:|---:|
| 20261017 | 3,047 | 26 | 877.13 |
| 20261018 | 3,319 | 28 | 878.59 |
| 20261019 | 2,461 | 21 | 876.23 |

以下结果仅用于两条原始 June Development 场景的模型选择，不能代替最终 12 个压力/SLA 场景的 B0–B5 对照，也不是独立 Test 结果。所有列出的模型均完成 5,684/5,684 任务，DMR=0、无温度违规。

| 模型 | 总碳 kg | 相同 idle18 背景下增量碳 kg |
|---|---:|---:|
| V2 历史最佳（第 52 轮） | 157,708.40 | 3,340.86 |
| V3 种子 20261017 暂时最佳 | 157,743.32 | 3,375.78 |
| V3 种子 20261018 暂时最佳 | 157,740.82 | 3,373.28 |
| V3 种子 20261019 暂时最佳 | 157,740.82 | 3,373.28 |
| Immediate + 固定 18°C 的正式对照 | 待完整评价 | 待完整评价 |

当前暂时最佳 V3 的总碳和增量碳均高于 V2，尚未证明优于 V2 或强立即执行基线。不能据此提前判断有限预算结束后的结论。三种子训练未结束，不报告最终 mean±std，不把短程训练曲线称为已收敛。

各作业最佳模型由本地 `research_notes/joint_v3/formal-001/jobs/full-<seed>/training/checkpoints/best_feasible.json` 指向。两个种子的暂时最佳数值相同，保留各自独立权重与选择记录，不合并为单个模型。

## 已观察到的限制

三个种子的正式辅助初始化均采集 4,096 个不同的实际制冷输入状态；其中 4,024 个（98.24%）在 18°C 取得最低功率。这是当前物理模型和输入轨迹的实际结果，不能据此宣称学习到了独立制冷收益。Task 初始化仅有 5 条满足可靠性及非强制条件的教师标签，均为 Wait；其余 12,457 个候选未当作可靠监督。低教师覆盖及风险保守回退需要由完整训练和消融检验。

训练曲线位于各作业 `training/figures/training-progress.{pdf,svg,png}`，来源为真实 `episodes.csv`。框架图位于 `docs/research/joint/figures-v3/method-framework.{pdf,svg,png}`。最终碳排、DMR、IT/HVAC 分项和消融对照图须等待实际评价结果；当前没有用模拟值填补这些图。

## 运行与后续入口

当前协调器及三个工作进程已在隐藏后台恢复。若将来进程中断，在仓库根目录的已审计 WSL 环境执行下列恢复命令；存活进程会阻止重复启动：

```bash
export PYTHONPATH=/mnt/d/A-myprofile/SustainCluster/upstream/research_notes/E1/E1-5B/runtime-deps
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
/home/hwj/miniconda3/envs/sustaincluster/bin/python -m experiments.joint_v3.pipeline --run --protocol research_notes/joint_v3/formal-001/protocol.json --jobs 3
```

全部有限预算和 Development 对照/消融结束后，才可调用 `freeze_campaign` 生成最终冻结文件。September 仍需要之后的明确用户授权。以下命令本轮**没有执行**：

```bash
/home/hwj/miniconda3/envs/sustaincluster/bin/python -m experiments.joint_v3.test_once --freeze <final-evaluation-freeze.json> --authorization <explicit-human-authorization.json> --source-root /mnt/d/A-myprofile/SustainCluster/upstream --output <september-test-once>
```

授权 JSON 必须包含 `authorize_september_test: true` 和最终冻结文件的准确 `freeze_sha256`。当前不存在符合条件的最终冻结文件，也没有本次 Test 授权。原始数据、大体积模型、完整任务日志和训练中间文件均留在本地；提交仅包含源码、文档、配置及轻量身份记录。
