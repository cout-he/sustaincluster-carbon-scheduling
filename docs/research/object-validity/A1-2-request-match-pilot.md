# A1.2：真实任务申请量小样本匹配验证

日期：2026-10-19（Asia/Shanghai）。基于 A1.1 提交 `2fecc17850a9dcabe322416d11b56d7bb3e9fb34`；工作分支：`research/object-model-a1-2`。

**结论：本轮 30 组 Train 原始身份键全部唯一匹配，证明这些样本可以准确取回原始申请字段。A1.1 的“未验证”现可更新为 B：已验证部分样本可恢复，但尚不能宣布全量任务可靠恢复。** 主要限制从“原表身份是否能找到”转为缺失时间键、当前重复行的任务计数语义，以及多实例申请如何对应现有任务抽象。

## 1. 下载、样本和方法

仅从 [Alibaba 官方 README 列出的任务表入口](https://github.com/alibaba/clusterdata/blob/master/cluster-trace-gpu-v2020/README.md#2-data-files-in-the-trace) 下载 `pai_task_table.tar.gz`：35,514,117 字节。压缩包 SHA256 与官方值一致：`cd1d6dc3215d2a8607ccf6b6dd952b5db776df86926c73259fea7c1499ac40e5`。

以流式方式读取包内 `pai_task_table.csv`，共扫描 **1,261,050 行、113,398,956 字节**，未将 CSV 全量解压落盘。其 SHA256 `6954802b457305f8a9e480ef97c40060baee59649fd3adc62c5a1e048aa058de` 也与官方公布值一致。扫描完整任务表是为了排除后续行出现第二个匹配；只保留选定键及同 Job 的相关原始记录在本地。不下载实例表或其他原始数据表。

Train 来源为 `train-20230102T0500-r1-s1.5`，先核实 manifest 的 `role=train`、`test_access=false` 和 workload 文件哈希，再读取任务载荷。片段共 2,785 行，其中 **17 行原始 `end_time=null`**，无法组成完整匹配键，不纳入此次完整键抽样，也不改写或删除。其余完整键共有 2,756 组：2,744 组当前单行键、12 组当前重复键。

在查看原表记录前，按 `SHA256(JSON([job_name, 原始start_time, 原始end_time]))` 升序固定选择前 20 组单行键和前 10 组重复键；JSON 为紧凑 UTF-8，时间用 Decimal 精确规范化。得到 **30 组键、40 个当前行**。这里“唯一/重复”仅指该 Train 片段，不预先假定原表唯一。不是随机总体抽样，不据此估计全量恢复率。

匹配使用 `job_name + 原始数值 start_time + 原始数值 end_time`，时间按精确数值比较，没有容差、近邻补齐、15 分钟取整或使用年度平移后的到达时间。

## 2. 匹配结果

| 当前样本类型 | 身份键组数 | 当前行数 | 原表零匹配 | 原表唯一匹配 | 原表多匹配 |
|---|---:|---:|---:|---:|---:|
| 当前唯一键 U01–U20 | 20 | 20 | 0 | 20 | 0 |
| 当前重复键 R01–R10 | 10 | 20 | 0 | 10 | 0 |
| 合计 | **30** | **40** | **0** | **30** | **0** |

30 组对应 30 个不同的原始 `(job_name,task_name)` 身份。以下均为匹配到的原表值，不由当前 usage 推算：

| 字段 | 核验结果 |
|---|---|
| `task_name` | 30/30 非空；含 tensorflow、worker、PyTorchWorker、JupyterTask、TVMTuneMain、evaluator、ps 七类 |
| `inst_num` | 30/30 为正整数；范围 1–20；21 组为 1，9 组大于 1 |
| `plan_cpu` | 30/30 有限且非负；原始百分数 100–3200，对应每实例 1–32 vCPU |
| `plan_gpu` | 30/30 有限且非负；原始百分数 5–400，对应每实例 0.05–4 GPU 等价申请量 |
| `plan_mem` | 30/30 有限且非负；每实例 3.92578125–87.890625 GB |

单位及实例继承关系依据 [官方 Task 定义与层级示例](https://github.com/alibaba/clusterdata/blob/master/cluster-trace-gpu-v2020/README.md#pai_task_table)。恢复的 GPU 申请量仍可能为分数，不等于已经恢复独占实体卡数量。

## 3. 重复键究竟有什么歧义？

**这 10 组没有“多个原始 Task 候选”的身份歧义。** R01–R10 每组均为两个当前行精确匹配同一条原始 Task 记录，因而该组的 `task_name/inst_num/plan_*` 唯一。

但每组两个当前行的十二个原始字段中，**只有 `bandwidth_gb` 不同**。它们不能据此被认定为两个独立原始 Task。若把同一申请总量分别赋给两个当前行并都作为独立任务执行，就会按两份该原始 Task 的申请量计数。

源码存在与此现象一致的风险点：本地 `D:/A-myprofile/SustainCluster/upstream/data/workload/alibaba_2020_dataset/alibaba_analysis.ipynb` cell 56 仅按 `job_name` 连接带宽，再删除 `task_name`（原文本第 5521–5527 行；参见 [A1 字段证据](A1-field-evidence.md#2-原始字段在哪里丢失)）。不过，10 组所属 Job 中，原任务表行数分别为 1 行（5 组）、2 行（1 组）、3 行（4 组）；仅凭任务表不能还原带宽中间表的全部连接过程，**不能把所有重复的具体成因都认定为已证实**。本轮不选择哪行应保留，不合并、不删除、不改资源。

因此需区分：**原始申请记录已唯一找到；当前多个行如何对应一个真实任务的执行和核算，尚未解决。**

## 4. 能否建立准确对应关系，具体还卡在哪里？

**可以：对本轮 30 组完整原始键，申请字段对应关系已通过整张原任务表的唯一性扫描验证。** 不能由此直接完成现有训练数据替换，原因是：

1. 17 个缺失原始结束时间的当前行不满足本轮完整键规则；本轮不采用弱化键或猜测补齐，也未验证其是否可以另行消歧。
2. 重复组已恢复原始身份，却缺少当前多行的可靠计数与带宽来源解释；不能把原始申请广播后宣称重建完成。
3. 9 个样本是多实例 Task。`plan_*` 是实例继承请求；是否按 `inst_num` 汇总，以及怎样处理同步执行、重试与 Duration，仍需独立语义核验。现有 [alibaba_utils.py](../../../data/workload/alibaba_2020_dataset/alibaba_utils.py) 第 55–76 行以平均实例 runtime 生成 Duration，取回申请量不自动解决该抽象问题。本轮未下载实例表，未恢复实例并发。
4. 仅验证一个 Train 片段中的固定 30 键，尚未审计其余键或其他 Train 片段，不能称全量可恢复、全量无歧义或整体恢复率为 100%。

若继续，最小下一步应先由用户决定是否审计重复行来源和任务计数，再制定缺失键处理与新数据版本方案。本轮不实施完整恢复或任何对象调整。

## 5. 本地复核与执行边界

本地忽略目录：`D:/A-myprofile/SustainCluster/object-model-a1/research_notes/object-model-a1-2/`，保留原包、`selected-30-keys.json`、`match-details-local-only.json`、`summary.json`、下载元数据和一次性脚本；均不上传 GitHub。只提交本报告，不包含原始身份键或任务明细。

- 复核命令：在该工作树运行 `python research_notes/object-model-a1-2/match_pilot.py`；只扫描已下载任务包与指定 Train workload，不执行下载或修改输入。
- 匹配脚本 SHA256：`fad2387ce2d265df1e8ddf416b74320ed19c9774568dc6e4ee74d5a095030c7d`；预先保存的 30 键选择文件 SHA256：`ea2a17a7b704f67a9030a04760a2f47578cc6cc5b32843e5c619f446cbfb66a3`。
- Train 路径：`D:/A-myprofile/SustainCluster/publication/sustaincluster-carbon-scheduling-001/research_notes/joint_v2/scenarios-001/scenarios/train-20230102T0500-r1-s1.5/`；workload SHA256：`ba5b5f2cde71092f86f443edef4c0d5848899341cadae4290dd95d8d372260a7`。
- 实际扫描完成时间：北京时间 19:43:07。未读取现有 Development/Validation/Test 冻结场景载荷，未读 Train 环境载荷；未修改任务、容量、Deadline 或功率模型，未运行调度实验或训练。完成推送后停止。
