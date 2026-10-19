# A1.1：Alibaba 任务真实资源申请量可恢复性检查

核验日期：2026-10-19（Asia/Shanghai）。分支：`research/object-model-a1-1`；基准：`71cc74a191ca7f347f0a84280ba4f037352ef49d`。本轮仅检查可恢复性，不执行恢复。

**结论选 C：以当前本地资料和已完成的验证，不能可靠恢复现有 Train 任务的 CPU/GPU/MEM 申请量，应暂时保留观测使用量代理。** 这不意味着永久无法恢复：官方原表仍有获取入口，现有任务保留的原始时间也提供了候选匹配路径。但没有原表匹配证据，不能宣称已经可以恢复一部分，更不能给出恢复覆盖率。

## 1. 原始申请量数据是否存在或能够获取？

**本地未找到原表及相关中间数据。** 按文件名递归检索 `D:/A-myprofile`、`D:/Downloads`、`D:/edge_download`、`D:/hwj`、`D:/My_Profile`、`D:/python_projects`、`D:/git_project`、`D:/git`、用户 Desktop/Documents 及 WSL `/home/hwj`，未命中 `pai*table*`、`extracted*dfas*`、`cluster*trace*gpu*` 等名称；Windows 另查 Alibaba 压缩包及 `dfas/dfw/dfws/dft/dfi/dfj` 中间文件。`C:/Users/hwj/Downloads` 不存在。该结论限于已检索目录和名称模式，不排除其他位置或改名归档。

本地现有年度 PKL、冻结 Train 快照和预处理源码；源码提及原 CSV 路径不代表文件仍在。本轮没有反序列化年度 PKL。

**官方入口当前可响应。** 根据 [Alibaba 官方 README 的下载链接](https://github.com/alibaba/clusterdata/blob/master/cluster-trace-gpu-v2020/README.md#2-data-files-in-the-trace)，于北京时间 19:17:28 对 OSS 压缩包执行 HEAD：

| 文件 | HTTP | Content-Length（压缩字节） | 约合十进制 MB |
|---|---|---:|---:|
| `pai_task_table.tar.gz` | 200 | 35,514,117 | 35.51 |
| `pai_instance_table.tar.gz` | 200 | 694,839,139 | 694.84 |
| `pai_job_table.tar.gz` | 200 | 62,065,432 | 62.07 |

以上请求读取压缩包正文 **0 字节**；只能证明端点响应及声明大小，不能证明完整包可解压或校验通过。官方还列有 [GitHub 备用入口](https://github.com/qzweng/clusterdata-cluster-trace-gpu-v2020-data)。仅 GET 了 README 和两个小型字段 header，没有下载原始记录。

原 [任务 header](https://raw.githubusercontent.com/alibaba/clusterdata/master/cluster-trace-gpu-v2020/data/pai_task_table.header) 包含 `job_name,task_name,inst_num,status,start_time,end_time,plan_cpu,plan_mem,plan_gpu,gpu_type`；原 [实例 header](https://raw.githubusercontent.com/alibaba/clusterdata/master/cluster-trace-gpu-v2020/data/pai_instance_table.header) 包含 `job_name,task_name,inst_name,worker_name,inst_id,status,start_time,end_time,machine`。

## 2. 现有 Train 任务能否与原始任务正确匹配？

**尚未证实；存在可验证的路径，也存在身份丢失和连接多重性风险。**

源码链：

- [alibaba_utils.py](../../../data/workload/alibaba_2020_dataset/alibaba_utils.py) 第 45–76 行：实例与任务原本按 `(job_name, task_name)` 连接；`get_dfa` 保留任务表的原始 `start_time/end_time`，实例聚合列带 `_i` 后缀；`duration_min` 则来自实例 runtime 的均值。
- 本地 `D:/A-myprofile/SustainCluster/upstream/data/workload/alibaba_2020_dataset/alibaba_analysis.ipynb`：cell 12 按 `(job_name,task_name)` 汇总实例资源；cell 19 导出时不含 `task_name/plan_*`；cell 56 带宽只按 `job_name` 合并，随后删除 `task_name`。后一步可能产生一对多连接，缺原表时不能证明实际影响。
- [年度生成脚本](../../../data/workload/alibaba_2020_dataset/extract_dataset_from_dfas.py) 第 350–363 行和 [冻结加载器](../../../experiments/e1/workload.py) 第 14–24、153–160 行：最终只有十二个原始字段。`task_id/source_row/source_task_index` 是本地生成身份，不能代替原始 `task_name/worker_name`。

实际只读检查一个片段：`train-20230102T0500-r1-s1.5`。先核实场景角色为 Train，再读取 `workload.json.payload`，未读环境数据。

| 检查项 | 结果 |
|---|---:|
| 当前任务行数 | 2,785 |
| 不同 `job_name` | 2,770 |
| 有多行的 `job_name` / 所涉及行数 | 13 / 28 |
| 不同 `(job_name, 原始 start_time, 原始 end_time)` | 2,773 |
| 重复上述组合键数 / 所涉及行数 | 12 / 24 |
| 完全相同的十二字段重复组 | 0 |

十二列为 `job_name,start_time,end_time,start_dt,duration_min,cpu_usage,gpu_wrk_util,avg_mem,avg_gpu_wrk_mem,bandwidth_gb,weekday_name,weekday_num`。没有 `task_name`、实例身份、`inst_num` 或 `plan_*`；全部 `source_job_name` 与原始首列一致。本地生成任务 ID 唯一，并不证明原始任务唯一。

**可尝试的精确匹配键**是 `job_name + 原始数值 start_time + 原始数值 end_time`，对照原任务表后再取回 `task_name`。不能使用已平移到年度容器的到达时间，也不能根据名称语义猜测。即使某键在当前片段只出现一次，也不能据此断言它在原表中唯一；上述重复键也不能直接解释为重复任务或已证实的数据错误。

因此，现有字段不足以保证全量正确匹配，但也不能仅因丢失 `task_name` 就断言任何任务都不可恢复。**本轮没有执行原表连接；样本统计不是申请量恢复率。**

## 3. CPU/GPU/MEM 申请量能否可靠恢复？

**当前均不能直接恢复；若后续获得唯一且可审计的原始任务匹配，三者都有明确的原字段。** [官方说明](https://github.com/alibaba/clusterdata/blob/master/cluster-trace-gpu-v2020/README.md#pai_task_table) 支持 CPU 使用 `plan_cpu/100`（vCPU）、GPU 使用 `plan_gpu/100`（GPU 等价申请量，可为分数）、MEM 使用 `plan_mem`（GB）。不能由观测 usage 反推申请量，也不能强行将分数 GPU 向上取整为实体卡。

避免错配必须同时满足：

1. 先恢复并核实 `(job_name,task_name)`；多角色 Job 不能把整 Job 申请量广播给每个当前任务。
2. 按 [官方层级示例及实例定义](https://github.com/alibaba/clusterdata/blob/master/cluster-trace-gpu-v2020/README.md#pai_instance_table)，任务请求由实例继承。任务申请总量可按任务 `plan_* × inst_num` 计一次；如改为实例核算，则按去重、核实后的实例各计一次，不能再乘一次 `inst_num`。`inst_id` 官方实为 Job ID，不能单独作为实例唯一键；需核验 `worker_name/inst_name`、状态、重试及连接基数。
3. 当前 Duration 为平均实例 runtime；实例可能不同步启动。取回申请字段不等于已恢复真实并发或完成任务执行语义重建。若当前多行来自旧连接扩增，还必须区分年度重复、同源任务和真实不同任务，不能静默合并或重复计资源。

## 4. 是否建议后续改用真实申请量？

**建议有条件采用。** 研究提交时的资源预留与容量竞争时，原始申请字段比生命周期平均使用量更符合需求含义；但必须先通过身份和实例聚合审计，再建立独立数据/映射版本。当前 A1 仍应标为观测使用量代理，不能直接替换数值或沿用旧容量宣称新场景已验证。若唯一匹配仅覆盖一部分，应保留其余缺失和歧义，不填造申请量；若无法消除身份歧义，则需要从原表重建任务数据。

## 5. 下一步最小工作是什么？

待用户决定后，先取得**明确属于 Train 的少量原任务记录**，与本轮片段的身份白名单作精确连接；可由数据持有方预先提取。优先任务表，不必先取得完整实例表。完整文件的获取或跨数据集扫描均不属于本轮授权的恢复操作。

对该小样本报告零匹配、唯一匹配、多匹配、`plan_*` 缺失/冲突及 `inst_num` 分布，保留原 `task_name`，同时检查旧 job-only 带宽连接的多重性。仅在需要解释多实例并发或重复时补充对应 Train 实例子集。匹配失败或多义项保持未解决，不能近似填充。只有通过这些检查，才有证据将结论升级为 B 或 A；本轮不执行这一步。

### 核验身份与边界

- Train 文件位置：`D:/A-myprofile/SustainCluster/publication/sustaincluster-carbon-scheduling-001/research_notes/joint_v2/scenarios-001/scenarios/train-20230102T0500-r1-s1.5/`。
- manifest SHA256：`826ee325469860dd9ab81b5c64705f7559285ef9cbc271f5b76a41b62d8c7e0d`；workload 文件 SHA256：`ba5b5f2cde71092f86f443edef4c0d5848899341cadae4290dd95d8d372260a7`。
- 本地 notebook SHA256：`957ea2c6c56742e3a7fa72f1a08f93c997c4a1a00d9d25814eacdba13f53bd18`；本轮 README 正文 SHA256：`cc13cc69d12ab46c882a1dcc74614cab6956f85ebb0d47d9409ff340bf2e7c08`。两个 header 实测 SHA256 与官方 README 所列值一致；压缩包未校验。
- 汇总核验记录留在本地忽略目录 `research_notes/object-model-a1-1/{train-identity-probe,official-availability}.json`，不含任务明细。本次仅提交本报告。
- 未修改任务、配置、仿真器或功率模型；未启动调度实验或训练；未读取 June Development、July Validation、September Test 的任务或环境载荷。完成本轮后停止，等待用户决定。
