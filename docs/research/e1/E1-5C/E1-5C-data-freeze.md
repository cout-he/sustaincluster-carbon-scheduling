# E1-5C 数据角色与身份

本轮发布父提交为 `958c14137bfeb30aaf6245a93a3cb693bcb60d20`。科研目录原始 HEAD/main 均为 `3f6ea95cb835b89ba50b0ef76d66d14b8037643e`，研究代码一直保留在 dirty 研究工作区；不能以该 HEAD 代表所有未提交研究源码。实际运行的源码身份为 `diagnostics-001/source-freeze.json` 的文件清单及 canonical SHA256：`8fbdc5efdfce419ea219f686bbb9a4536171478ed659a475b58e95aa67e06afc`。

## 窗口与冻结输入

| 角色 | UTC 到达区间（左闭右开） | 倍率/任务数 | stress seed | 来源 |
|---|---|---|---|---|
| Train | 2023-05-01 05:00 — 05-03 05:00 | 1×5542；1.5×8313；2×11084 | 510001 | E1-5B既有冻结输入/完整三基线结果 |
| Validation | 2023-07-08 05:00 — 07-10 05:00 | 1×8708；1.5×13062；2×17416 | 520001 | 2×引用E1-5B；1/1.5×本轮探索性诊断 |
| Test | 保留E1-5A/E1-5B既有封存定义 | 本轮不生成、分析或评测 | 本轮不使用 | 不参加本轮选择 |

Validation 两个新增档位在任何新策略执行前全部写入冻结输入；六组实际 workload/environment 哈希在 `input-identities.json`。完整冻结包和逐任务账本仅保留科研目录。历史四组的 artifact manifest 与各文件哈希逐一核验；新组显式标记 exploratory，既有 E1-5B `selected_policy=null`/5%准入结果未改。

共享条件为单DC US-CAL-CISO、15min、CPU50000/GPU1000/MEM80000、需求倍率5、SLA multiplier1.5、HVAC setpoint22°C。阈值161.075 gCO2e/kWh沿用E1-5B四月校准，未依据本轮结果调参。每档三策略共用任务和环境。Train 公共终点均为05-03 14:00UTC；Validation均为07-10 12:45UTC。历史组保留原终点；新组按三基线最慢Drain取公共终点，并积分较早排空后的相同物理空闲状态。各任务全部完成，无删失。

## 外部文件身份

| 文件（科研根目录下） | SHA256 |
|---|---|
| data/workload/alibaba_2020_dataset/result_df_full_year_2020.pkl | 3ba8a8e0067288f8d2542752a292f7da04cb79305689ba145cc6060e96cb7f2e |
| data/carbon_intensity/US-CAL-CISO/2023/US-CAL-CISO_2023_hourly.csv | 44bafab775c6b0d4762b1dc971eb293cfea154f80799a5230f9268d32e1ad55b |
| data/weather/US-CAL-CISO/2023.json | 090bddc8a5f4e9edb5f3f193c6c147ed03841337b4bb17c7740f3b63a03cdaf5 |
| configs/dcs/dc_config.json（原始物理配置字节） | bbe439176605512523a3a0ecd2bd22f18fdb9abf6ae6e782217157beedbf705b |

容器身份哈希读取完整原始文件；原有 loader 解析共享容器后只选择 Train/Validation。这不等同于提取或评测Test；本轮未提取Test episode、拟合Test统计或执行Test策略，驱动白名单拒绝Test。文件级保护哈希也不会计算Test窗口结果。发布副本不包含上述原始数据与历史完整运行档案，不能宣称完全自包含；复现需要用户自行合法获取/提供匹配数据，以及E1-5B历史冻结包。

## 独立性与分析边界

2×是原任务各复制一次，seed只改变副本身份/稳定排序，不改变资源到达形态，不得当作不同真实负载样本。1.5×的部分复制会随seed变化，但仍依赖同一trace。Train/Validation属于不同已注册时间窗，不代表统计独立性已经证明。

工作量必要条件为离线工具，能够使用全个开发窗口，禁止输入actor、策略或当前Reward。120min诊断压力分母固定；旧RiskOnly的warm/cold有效观察分母保持不变。本轮没有根据Test估计任何阈值或归一化。
