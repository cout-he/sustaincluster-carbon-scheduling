# A1：Alibaba 字段与需求映射证据附录

核验日期：2026-10-19（Asia/Shanghai）。代码基准：`a754c5ee4bd3922cecd6993d9abb6bfbacdda7e0`。本附录只讨论字段来源、处理及可支持的模型含义；不以规则实验结果选择字段解释。

## 1. 结论与证据等级

本轮找到 Alibaba 官方数据字典，补上了 A0 对 CPU/GPU 百分数分母和 `avg_mem` 单位的来源缺口。**已观测使用量的单位可以确认；当前冻结任务仍不能恢复原始申请量，也不能直接称为真实硬件预留。**

| 字段 | 已确认含义与单位 | 可直接支持的变换 | 不能据此声称的内容 |
|---|---|---|---|
| `cpu_usage` | 实例生命周期内的平均已使用 CPU 核数，以百分数表示；600 对应 6 vCPU 核 | `/100` 得到观测使用核数 | 申请核数、峰值需求或物理 CPU 插槽数量 |
| `gpu_wrk_util` | 实例生命周期内的平均已使用 GPU 等价量，以百分数表示；50 对应 0.5 GPU 等价量 | `/100` 得到观测 GPU 等价使用量 | 实际申请 GPU 数、独占卡数、MIG 切片数或已验证的共享隔离机制 |
| `avg_mem` | 平均已使用主存，官方标注 GB | 不再额外 `/100`、`/1000` 或转 GiB | 峰值/预留内存；官方没有在该字段定义中给出 GB 的底数细则 |
| `plan_cpu` / `plan_gpu` / `plan_mem` | 原始任务表中的申请量；前两者为百分数，内存为 GB | CPU/GPU `/100`，内存保留 GB；必须按实例身份正确聚合 | 这些字段已存在于当前冻结任务，或可以从 usage 反推 |

以上单位和使用量/申请量区分依据 [Alibaba 官方 GPU v2020 字典：sensor](https://github.com/alibaba/clusterdata/blob/master/cluster-trace-gpu-v2020/README.md#pai_sensor_table) 与 [task](https://github.com/alibaba/clusterdata/blob/master/cluster-trace-gpu-v2020/README.md#pai_task_table)。sensor 指标对应实例而非任务，取生命周期均值；task 中的请求由实例继承，整项任务可能包含多个实例。实例聚合必须依据身份及实例数量，不能把一项申请重复乘两次。

证据等级：**官方字段说明已确认**、**本地源码链已确认**、**一个冻结 Train 快照已运行核对**、**年度整包与原始 CSV 的逐行连接仍未复核**。不能用前三项替代最后一项。

## 2. 原始字段在哪里丢失

本地分析 notebook 的代码单元显示如下链路。仅读取代码单元，不运行 notebook，也不重新执行任何原始数据清洗：

1. `alibaba_analysis.ipynb` cell 8 / 10：建立实例表，再按 `worker_name` 连接传感器；cell 12 按 `(job_name, task_name)` 对 usage、memory 及 `plan_*` 求和。
2. cell 19：导出 `extracted_dfas.csv` 的列显式为 `job_name,start_time,end_time,start_date,duration_min,cpu_usage,gpu_wrk_util,avg_mem,avg_gpu_wrk_mem`，此处不含 `plan_*` 和 `task_name`。
3. cell 55 / 56：带宽按 `(job_name, task_name)` 聚合，却仅按 `job_name` 连接回提取结果，随后删去 `task_name`。这存在多角色 job 的连接多重性疑问；本轮缺原始表，未宣称已经发生重复或已经排除重复。
4. [年度生成脚本](../../../data/workload/alibaba_2020_dataset/extract_dataset_from_dfas.py)第 350–363 行指定最终十二列；第 346、372 行按 15 分钟桶生成 `tasks_matrix`；第 424–472 行把七周来源片段重复平移为年度容器。最终十二列不包含请求量、实例数量、实例身份、GPU 类型或设备身份。
5. [冻结任务加载器](../../../experiments/e1/workload.py)第 14–16 行声明十二列，153 行要求每项恰为十二字段。第 158–160 行把 usage 变成固定的三种逻辑占用。A1 必须从已冻结 Train 快照取值，不能调用会整体 `read_pickle` 的旧年度加载函数。

本地 notebook 的可定位原文件行：传感器连接 159；资源聚合 181–182；导出列及裁剪 1723–1731；带宽聚合 5465–5468；仅按 job 连接及删字段 5521–5527。notebook 在 `D:/A-myprofile/SustainCluster/upstream/data/workload/alibaba_2020_dataset/alibaba_analysis.ipynb`，未复制其输出或原始数据到 Git。

本轮实际检查 `train-20230102T0500-r1-s1.5`：先确认 manifest 的 `role=train`，再读其 `workload.json.payload`。其中 **2,785 项任务**全部 `raw_fields` 长度为 12，列名与加载器一致，无 `plan_*`、`task_name`、`worker_name` 或 `inst_num`。来源 SHA 为 `3ba8a8e0067288f8d2542752a292f7da04cb79305689ba145cc6060e96cb7f2e`。这直接验证该 Train 快照的缺失；其余 Train 输入由 A1 主实验的同一 schema 校验负责。

年度 PKL 本轮仅查文件元数据：142,336,499 bytes，与冻结协议身份记录一致。**没有 unpickle、反序列化扫描或重跑年度 EDA；不把源码列定义写成年度全量内容已经实检。** 在工作区按文件名搜索未发现 `pai_task_table`、`pai_sensor_table`、`pai_instance_table`、`pai_machine_spec` 原表/header，或 `extracted_dfas*` 中间 CSV。README 记载这些曾属于处理链，不等于它们现在本地可读。

## 3. 聚合与 Duration 仍有哪些模型假设

[本地工具函数](../../../data/workload/alibaba_2020_dataset/alibaba_utils.py)第 56–60 行先计算每个实例的 runtime，再按 `(job_name, task_name)` 取均值；第 76 行以该均值除以 60 得到 `duration_min`。notebook 则对同任务实例的平均资源使用量求和。

因此，冻结输入代表“已聚合的平均资源使用量 + 平均实例运行时间”的任务抽象。它不是实例级逐秒并发轨迹，也不是带完整 gang/placement 约束的任务申请表。A1 保持冻结 Duration 和到达桶不变，可以研究此抽象下的容量竞争与时间柔性，但应保留以下缺口：

- 连接是否一对一，以及同一 worker 是否有多行 sensor，需原表和主键唯一性报告。
- 多实例实际启动是否同步、资源平均值能否按一个固定块执行，需原实例轨迹核验。
- 内存均值是否足以作为容量预留，需峰值或申请字段；当前不能添加猜测的“安全放大倍数”。
- GPU 类型、显存、拓扑及共享执行是否改变 Duration，当前十二字段无法回答。

源脚本还包含既有清洗：notebook cell 16 过滤 Duration <15 min，年度生成脚本第 285–305 行按 IQR 过滤 Duration/CPU/MEM/带宽，第 399–405 行裁剪七周。**这些是已冻结输入的历史生成过程；A1 不再次过滤、不删除困难任务，也不把处理后 Train 宣称为未筛选的 Alibaba 全量原始任务。**

## 4. 为什么旧映射乘 5

[`utils/workload_utils.py`](../../../utils/workload_utils.py)第 50 行默认 `task_scale=5`，第 70 行明确说明其用途是模拟比原始更大的任务，第 79–81 行把 CPU/GPU/MEM 同乘。初始发布提交 `749dc6c` 已有这段默认值及说明；[`experiments/e1/configs/e1_5b_ppo.json`](../../../experiments/e1/configs/e1_5b_ppo.json)继承 `resource_multiplier=5`。这解释了**代码意图和继承关系**。

没有找到把 5 推导为实例数量、实际 GPU 卡数、峰值/申请转换系数、系统开销或设备校准系数的独立依据。不能把它解释成官方单位换算。V2 `resource_pressure_factor` 是在基础映射后另乘 0.75/1/1.25 的情景参数；不增加任务，也不改变 Duration/到达。

## 5. 可供选择的映射方案

| 方案 | CPU / GPU / MEM 公式 | 当前可执行性 | 使用边界 |
|---|---|---|---|
| Legacy ×5 | `5*cpu_usage/100` / `5*gpu_wrk_util/100` / `5*avg_mem` | 可从冻结 Train 实现，保持旧路线对照 | 合成需求放大；不能称原始真实申请 |
| Observed ×1 | `cpu_usage/100` / `gpu_wrk_util/100` / `avg_mem` | 官方单位支持，冻结 Train 可实现 | 聚合均值用量的固定资源、可分割 GPU 等价服务模型；尚未验证 reservation/placement |
| Requested（待补数据） | 正确实例聚合后的 `plan_cpu/100` / `plan_gpu/100` / `plan_mem` | 当前不可执行；缺失字段保持 `null` | 必须恢复 Train 对应原表及身份连接，审计唯一性/实例重复及缺失率后再版本化；不能由 usage 填充 |

Observed ×1 中建议将 GPU 明确称为 **GPU-equivalent 使用容量**，例如 0.25 代表四分之一全 GPU 的观测等价服务量。它不说明一项任务只需四分之一张实体卡。不能通过 `ceil(gpu_wrk_util/100)` 自动改成独占卡需求：这种变换丢失共享/显存/申请信息，并且改变原始任务需求语义。

Observed ×1 与 Legacy ×5 不能直接复用同一容量数值后比较算法。当需求及容量三维同时按 1/5 缩放，未作整数取整的确定性资源约束应完全等价。例如旧 M 的 `(50000,1000,80000)` 等压力参考为 `(10000,200,16000)`；L/H 也应重新推导。若对服务器核数或 GPU 数取整，需把它记录为独立的设计变更，重新实测。这个缩放等价性是算法无关的单位/压力核对，不给任何算法创造额外优势。

## 6. 来源身份与待补证据

| 来源 | 身份/位置 | 本轮核验方式 |
|---|---|---|
| Alibaba 官方字典 | `alibaba/clusterdata/cluster-trace-gpu-v2020/README.md`，读取 `master` 于 2026-10-19 | 官方网页字段规范；远端当前 commit SHA 未取得，机器清单保留 `null`，不伪造固定版本 |
| 官方请求字段 header | [pai_task_table.header](https://raw.githubusercontent.com/alibaba/clusterdata/master/cluster-trace-gpu-v2020/data/pai_task_table.header) | 独立确认原 schema 含 plan 字段；未下载数据 |
| 官方 sensor header | [pai_sensor_table.header](https://raw.githubusercontent.com/alibaba/clusterdata/master/cluster-trace-gpu-v2020/data/pai_sensor_table.header) | 独立确认 usage 与 memory 字段；未下载数据 |
| 本地 notebook | SHA256 `957ea2c6c56742e3a7fa72f1a08f93c997c4a1a00d9d25814eacdba13f53bd18` | 只读代码单元及文本定位；不执行 |
| 基准内 `alibaba_utils.py` | SHA256 `faaa571115cde969885b87ffdfb9bb6100abe1ea3b8d548e32676f822250d08f` | 与本地 upstream 文本相同；原始字节换行不同 |
| 基准内 `extract_dataset_from_dfas.py` | SHA256 `5aeae21d7adf8114a3290fcbb505efae13795731bdd81163f8918511138c8d0f` | 只读预处理源码；不调用读 CSV/PKL |
| 冻结年度来源 | PKL SHA256 `3ba8a8e0067288f8d2542752a292f7da04cb79305689ba145cc6060e96cb7f2e` | 引用基准冻结身份；本轮仅 stat |
| 本轮读取 Train 样本 | workload 文件 SHA256 `ba5b5f2cde71092f86f443edef4c0d5848899341cadae4290dd95d8d372260a7` | 先检查 role，后检查 schema；不输出任务明细 |

正式把请求资源或实体 GPU 清单接入前仍需：Train 对应原始任务/实例表；可靠身份连接及其一对一/一对多校验；硬件型号、CPU 的 vCPU/物理核关系、GPU 类型及共享能力、主存和显存配额、系统开销保留依据。上述缺失值在 [field_evidence.json](../../../experiments/object_model_a1/field_evidence.json) 以 `null` 或不可执行状态记录。A1 的 logical-only 结果不能据此转换为 kWh 或真实碳排放。
