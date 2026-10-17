# E0 原始输入恢复记录

数据恢复成功。所有输入均来自 `SOURCE_SNAPSHOT.json` 指定的官方 [HewlettPackard/sustain-cluster](https://github.com/HewlettPackard/sustain-cluster/tree/3f6ea95cb835b89ba50b0ef76d66d14b8037643e) Commit `3f6ea95cb835b89ba50b0ef76d66d14b8037643e`，没有访问其他研究分支，没有引入研究生成的工作负载或修改后的物理模型。

## 来源和恢复方式

执行目录：`D:\A-myprofile\SustainCluster\rl-scheduling-v1`。本地官方对象库：`D:\A-myprofile\SustainCluster\upstream\.git`。本轮未使用网络下载；本地对象库包含与源码快照完全对应的官方输入。

恢复前，当前源码快照中未包含运行用 CSV/JSON/PKL/ZIP；本地 `upstream/data/` 中存在相应文件。378 个输入里，66 个本地工作树文件与官方 Git blob 字节一致，312 个因 CRLF 换行与官方 blob 不同；对后者进一步比较确认差异仅为 CRLF。恢复统一使用 **固定 Commit 的 Git blob**，没有直接复制工作树文件。本地既存的解压 PKL 也没有作为恢复来源。

实际执行：

```powershell
python research/e0/restore_data.py --source-repo D:\A-myprofile\SustainCluster\upstream
python research/e0/inspect_data.py
```

`restore_data.py` 通过 Python `subprocess.check_output` 获取二进制 `git cat-file blob <blob-id>` 输出，校验 Git blob SHA-1 后以 `Path.write_bytes` 写入文件，并记录 SHA-256 和字节数。没有使用 PowerShell 文本重定向写入二进制。PKL 仅从官方 ZIP 的唯一成员 `result_df_full_year_2020.pkl` 解压；`zipfile` 同时检查成员 CRC。

逐文件来源、官方 Commit、路径、Git blob、SHA-256、字节数及 Git 忽略状态见 `data_manifest.json`。`inspect_data.py` 又重新计算全部 379 个本地文件 SHA-256，与 manifest 全部相符。

## 恢复范围

| 类别 | 官方文件数 | 官方字节数 | 范围 |
|---|---:|---:|---|
| 碳强度 | 64 | 56,855,952 | 16 地区，2021–2024 |
| 电价 | 230 | 95,304,712 | processed、standardized 及 scripts 下官方 CSV |
| 天气 | 80 | 69,761,975 | 16 地区，2020–2024 |
| 网络成本 | 3 | 6,728 | AWS、Azure、GCP |
| Alibaba 2020 官方打包工作负载 | 1 | 75,625,706 | 官方预处理 ZIP |
| **官方 blobs 合计** | **378** | **297,555,073** | 全部与对应 Git 对象一致 |
| ZIP 内原始 PKL | 1 | 142,336,499 | 仅解压，无任务变换 |
| **本地合计** | **379** | **439,891,572** | 全部由现有 `.gitignore` 忽略 |

ZIP 的 Git blob 为 `0472ada5ee073f66b20c54a0d705133c1b8bcd66`，SHA-256 为 `907090a42863c46887cdf9c66f99ac77b2a3b351544362d8c6c34b9ace6754a9`。解压 PKL 的 SHA-256 为 `3ba8a8e0067288f8d2542752a292f7da04cb79305689ba145cc6060e96cb7f2e`。

这里的 Alibaba 输入是官方仓库分发的**预处理工作负载**，不是重新从 Alibaba 原始 trace 生成的数据。没有运行提取、扩增、分组、带宽生成或补齐任务数据的预处理脚本。运行时源码中原有的任务缩放等行为由 E0 运行审计单独说明。

## 原始配置对应的数据检查

检查读取原始 `configs/env/sim_config.yaml` 和 `configs/env/datacenters.yaml`。环境年为 2023，五地区为 US-CAL-CISO、DE-LU、CL-SIC、SG、AU-NSW，网络供应商为 AWS。以下均为对真实文件的只读检查，完整数值见 `data_quality.json`。

- 5 个碳强度、5 个标准化电价、5 个天气序列全部存在，各有 8,760 个小时，覆盖 2023-01-01 00:00 至 2023-12-31 23:00。时间单调，无重复时间戳，无非小时步长；所有实际使用数值字段无 NaN/Inf。天气四个原始字段长度均为 8,760。
- AWS 网络矩阵为 20 × 20，无缺值，最小值 0，最大值 0.147。原始加载器有 `fillna(0.0)` 行为，本轮恢复未执行这类数据改写。
- 工作负载有 37,552 行、1,300,144 个任务记录，列为 `interval_15m`、`tasks_matrix`；每个任务矩阵有 12 列。时间戳无重复、单调，任务矩阵无空行。原始 duration/CPU/GPU/MEM/bandwidth 五个数值字段无负数和非有限值。

## 原始数据的限制

1. `result_df_full_year_2020.pkl` 实际时间范围为 **2020-01-01 00:00 至 2021-01-26 23:45 UTC**，超出文件名暗示的 2020 年。相邻时间差有 80 个 30 分钟，其余 37,471 个为 15 分钟；范围内缺少 80 个 15 分钟槽。没有补齐或重排。本轮的统一仿真必须核查其位置索引如何映射环境年份，不能以“2020完整逐15分钟序列”作前提。
2. 官方碳强度输入不含 2020。原始默认仿真环境年 2023 所需输入齐全；把环境年改为 2020 会缺碳强度数据。
3. 已验证原始配置五地区的 2023 输入结构及数值完整性；其他年份、地区的文件仅做字节身份校验，未逐一声称语义完整。上游本身有 `GB_incompleted` 等命名，不应把“全部官方文件恢复”解释为“所有可选数据均无质量问题”。
4. 电价原始文件含负价及极值；恢复时原样保留。原始 `ElectricityPrice_Manager` 会按其既有 15/85 百分位 IQR 规则截断，属于源码行为。

## 真实执行时间和错误

时间为 2026-10-17，manifest/quality 保留 UTC 时间戳；当地时间为 Asia/Shanghai。

| 命令/检查 | 结果 | 时长 |
|---|---|---:|
| `python research/e0/restore_data.py --source-repo D:\A-myprofile\SustainCluster\upstream` | exit 0；378 blobs + 1 PKL 已恢复；379/379 Git 忽略 | 脚本测量 18.748447 秒 |
| 首次 `python research/e0/inspect_data.py` | exit 1；JSON 已写入，但 stdout 的 GBK 编码不能输出 CO₂ 下标字符，抛 `UnicodeEncodeError` | 工具 wall time 3.785985 秒 |
| 修正 E0 检查脚本 stdout 为 ASCII JSON 后，再次执行 `python research/e0/inspect_data.py` | exit 0；379/379 SHA-256 一致；得到上述真实质量统计 | 脚本测量 2.255624 秒；工具 wall time 3.487456 秒 |

系统 `python` 在启动时输出 `Failed to find real location of ...\Python314\python.exe` 提示，但上述成功命令返回 0，读取和校验完整。该提示及本段编码修正均没有更改 SustainCluster 原始核心源码或输入。

## 复现和 Git 控制

在新 checkout 中先准备包含上述官方 Commit 的官方仓库，随后从源码根目录运行：

```powershell
python research/e0/restore_data.py --source-repo <官方仓库本地路径>
python research/e0/inspect_data.py
git check-ignore data/workload/alibaba_2020_dataset/result_df_full_year_2020.zip data/workload/alibaba_2020_dataset/result_df_full_year_2020.pkl
git status --short
```

恢复脚本遇到已存在且与固定来源不同的文件会报错，不会覆盖它；可重跑验证已存在且相同的输入。所有 CSV、JSON、PKL、ZIP 仍留在忽略路径；提交范围仅含 E0 脚本、manifest、质量统计与报告，不应使用 `git add -f data/`。本轮未恢复 checkpoint、训练权重、notebook 输出或敏感文件。
