# E1-5A 数据覆盖审计与预设划分

## Material Passport

- Origin Skill：academic-research-suite / experiment-agent。
- Origin Mode：只读输入审计；为后续实验计划提供覆盖证据。
- Origin Date：2026-10-12。
- Verification Status：ANALYZED；源文件覆盖与哈希已实查，策略实验和训练均未执行。
- Version Label：e1-5a-data-split-proposal-v1。
- 范围：仅写入 `research_notes/E1/E1-5A/`；未修改旧源码、历史归档、原始数据、Git 元数据或发布副本。

## 结论与证据边界

现有 US-CAL-CISO 文件足以支持下列互不重叠的 48h 到达窗，以及各自到达结束后 30 天的真实对应环境文件覆盖。三个新窗与 E1-2A/E1-3 已见到达窗的原始任务身份交集均为零。这里确认的是**可用输入覆盖**；30 天是固定安全上限，未证明任一新策略一定在上限前排空，不能把旧窗口的 8.75h Drain 直接推广到新窗。

划分仍是 E1-5A 提案，后续冻结前必须锁定 workload/environment 快照、全部 seed、归一化与 API 可见字段。机器审计记录是 [coverage_audit.json](data-audit-002/coverage_audit.json)，划分契约是 [data_split.json](data_split.json)。未执行策略、训练、参数扫描或合成压力任务生成。

## 真实文件覆盖、缺口与时区

| 输入 | 实查范围与质量 |
| --- | --- |
| workload pickle | 37552 行；原始 dtype=`datetime64[ns, UTC]`，无 naive 时间、重复时间戳或网格偏移；2020-01-01 00:00 至 2021-01-26 23:45 UTC。包含范围内理论 37632 个 15min 桶中缺 80 桶；矩阵形状全部为 N×12，保存行无空矩阵。 |
| 2023 direct CI | 8760 个连续有限小时观测，2023-01-01 00:00 至 12-31 23:00 UTC；无重复或缺失。全年 2 行 `Data Estimated=true`，在 08-17 23:00 与 11-05 09:00 UTC；下列新窗完整环境段与 4 月校准均为 0 行。 |
| 2023 weather | 8760 个连续有限干球温度小时值；原始小时标签无时区。文件元数据为 `America/Los_Angeles`、`GMT-7`、`utc_offset_seconds=-25200`。按 E1 的固定元数据偏移解释，UTC 覆盖 2023-01-01 07:00 至 2024-01-01 06:00。 |
| 其他年环境库存 | 本地 CI 2021–2024、weather 2020–2024 已只读统计；各文件自身小时范围均无缺口、重复时间或非有限 direct CI/干球温度。未选择这些年作训练或测试。 |

CI 与天气单个 2023 文件的共同连续范围为 **2023-01-01 07:00 至 2023-12-31 23:00 UTC（包含两端小时样本）**。不得按文件名假定天气从 Jan 1 00UTC 开始，也不得将某个单年文件直接接到另一年后循环、复制尾点或外推。

`weather_time_basis=metadata_fixed_offset` 是已有 E1 语义，不是在本轮修复全年 DST。额外检查表明三个新窗从到达起点一直到安全终点的 IANA 当地偏移也均为 −25200 秒；所有 envelope 位于 DST 夏季段，未跨切换。全年冬季仍有固定偏移解释的真实性限制。天气来自已有历史气象文件，并非本轮测量；湿球 `0.9×干球` 是既有模型代理。详见 [integrity_checks.json](data-audit-002/integrity_checks.json)。

workload 生成脚本 `data/workload/alibaba_2020_dataset/extract_dataset_from_dfas.py:424–472` 把约 7 周原始块反复平移复制成扩展文件，矩阵内部 `start_dt` 没有同时更新。本轮仅验证已有文件，不重建原始 Alibaba trace。E1 将实验月/日/时/分映射到 trace 年 2020，**不能称为真实 2023 到达，不能把全年复制块当独立观测**。缺失桶在既有 loader 中会被解释为无到达，但它不能证明原始提交记录中确实没有任务；本提案选择没有缺桶的窗。

## 预设窗口与覆盖门槛

所有区间均为 UTC，arrival 使用左闭右开，15min 决策网格；到达终点不引入任务。环境检查包含安全终点对应的原始小时样本，以保证最后一拍 weather 插值所需上界。

| Split | 48h arrival `[start,end)` | 环境安全终点 `end+30d` | 原始任务数 | 到达桶 | 每源环境小时点 |
| --- | --- | --- | ---: | ---: | ---: |
| Train | 2023-05-01 05:00 → 05-03 05:00 | 2023-06-02 05:00 | 5542 | 192/192 | 769/769 |
| Validation | 2023-07-08 05:00 → 07-10 05:00 | 2023-08-09 05:00 | 8708 | 192/192 | 769/769 |
| Test | 2023-09-09 05:00 → 09-11 05:00 | 2023-10-11 05:00 | 6723 | 192/192 | 769/769 |

三个完整 arrival+Drain 上限 envelope 也不相交。三窗 12 列 schema、duration>0、所用资源/带宽非负且有限；以旧硬件 CPU50000/GPU1000/MEM80000 与资源倍率 5 审查，均无单任务超容量。最长 duration 分别为 392.9、393.23333333333335、393.26666666666665 分钟。最长 duration 和全窗任务数是离线可行性审计信息，不能作为 actor 的未来负载输入或自动调大 planner 时域的依据。

选窗依据是日期隔离、覆盖与原始身份隔离，不使用策略碳排、DMR、JCT、回报或胜负。初始 Test 候选是 Sep 8 05UTC；检查发现映射 trace 的 `2020-09-08 07:30 UTC` 缺桶，因此按覆盖优先顺移一天。保留 [初次审计](data-audit-001/coverage_audit.json)，不删改被否决记录。后续不能因为正式 Test 的性能或 CI 波动不理想再次换窗。

原始身份比较采用矩阵 `(job_name,start_time,end_time,start_dt)`，排除平移后的 arrival 和 stable ID。Train、Validation、Test、旧 Jul 1 窗的六个两两交集全部为 0。各新窗内部仍分别有 9、263、146 个重复身份计数；保留它们及不同 row/task_index 的任务身份，不擅自去重，亦不当作独立统计重复。零跨窗交集仅支持本次四个 cohort 的直接身份隔离，不能证明整个数据集或未来添加窗口独立。

## 已见信息及阈值角色

E1-2A/E1-3 已执行 arrival 为 `[2023-07-01 05:00,2023-07-03 05:00)`，环境冻到 Aug 2 05UTC。E1-2A 还预先扫描过 **Jul 1 至 Aug 31 每日 05UTC 的 48h 候选**；不能仅因没有正式跑策略，就把这些候选的 CI 当完全未见。该扫描最晚候选的 arrival 到 Sep 2 05UTC。Validation 放在可用于开发的已见季节段；正式 Test 从 Sep 9 开始，避开旧 arrival、旧 30 日环境 envelope 和旧选窗 CI 范围。

新 split 使用早期校准 `[2023-04-01 00:00,2023-05-01 00:00)` 的 720 个 direct CI 小时样本中位数 **161.075 gCO₂e/kWh**，二进制 hex 为 `0x1.4226666666666p+7`。校准无缺口、非有限值或 `Data Estimated=true`，在 Train 之前结束。它是后续新训练/Validation/Test 共用的预设阈值，不读取 Val/Test 的分位数。E1-3 的 June 校准阈值 `137.64999999999998` 仅保留作**历史固定动作/数值等价验证**，不得挪作新的 May Train 正式参数。

可预声明无需估计的工程归一化：资源除固定容量 `(50000,1000,80000)`，时间以分钟，CI scale=1000，计数 scale=1000，并对非负比值采用有界变换 `x/(1+x)`。这些值依据单位/工程常数，不依赖 Test 分布。如果后续需要 z-score、经验剪裁分位数或学习式归一化，只能拟合 Train；拟合后冻结，Validation 只用于允许的模型选择，Test 只作最后一次评估。不能用完整 2023 文件估计均值、方差或阈值。

## 可见信息、seed 与后续冻结

policy/actor 只能获得当前时钟、当前已可用 CI、已到达任务的声明特征、当前资源与运行任务状态，以及因果历史计数。CI 沿已有 E1 因果保持语义取 `timestamp<=now` 的最新样本；假定该小时值在标注时刻可用，未模拟发布延迟。weather 的真实线性插值及全段环境轨迹仅供环境功耗核算，不能进入 policy。Task duration/observed-usage 需求在 arrival 已知仍是既有仿真 oracle 假设，须明示，不能称为真实在线可观测保证。

离线审计/评估可读取而 policy 特征不可读取：未来 arrival 清单、全窗 task_count/max-duration、Drain 安全上限所需的后缀环境、未来 CI/weather、旧/新全窗覆盖统计、原始 start/end 时间与 weekday 字段、source row/index、源哈希、其他策略的配对试验 outcome、尚未发生的回报和最终排空信息。learner 可使用本策略已发生的训练 reward。任务 ID 只用于身份与稳定 tie-break；不得将内含 trace 行号/原始重复块信息编码进 actor 特征。

预留不同压力 seed 空间：Train=`510001,510002,510003`，Validation=`520001,520002,520003`，Test=`530001,530002,530003`；均不同于旧 stress seed `20261010`。这些是**待后续阶段正式冻结的提案**，本轮未生成 clone。相同 split/seed 下各策略共享完全相同冻结 cohort；同 seed 不同 load 档可保留嵌套复制。压力 seed 与 RL 初始化/采样 seed 必须分字段记录，固定物理 reset seed123 不能算独立压力或学习重复。各 stress seed 是同一 trace 的合成扰动，不扩大真实独立 trace 样本量。

后续冻结清单：采用的 windows/schema 与原始输入哈希；缺桶规则与 2020 映射；压力负载档/全部 seed/clone 清单；RL 初始化与训练随机种子；准确 threshold 字面值/归一化 transform；context API 和未来字段拒绝契约；15min 决策/非抢占与已知 duration 假设；48h 后禁止新到达；`end+30d` 固定 cap；未排空时删失字段和停止规则；共同终点及空闲同模型积分；Train-only 参数拟合与 Validation 模型选择规则；最后 Test 评估前全部选择锁定。

## 源哈希与审计复查

| 输入文件（相对 upstream） | SHA256 |
| --- | --- |
| `data/workload/alibaba_2020_dataset/result_df_full_year_2020.pkl` | `3ba8a8e0067288f8d2542752a292f7da04cb79305689ba145cc6060e96cb7f2e` |
| `data/carbon_intensity/US-CAL-CISO/2023/US-CAL-CISO_2023_hourly.csv` | `44bafab775c6b0d4762b1dc971eb293cfea154f80799a5230f9268d32e1ad55b` |
| `data/weather/US-CAL-CISO/2023.json` | `090bddc8a5f4e9edb5f3f193c6c147ed03841337b4bb17c7740f3b63a03cdaf5` |

上述三个实际输入在每次扫描前后 SHA256 相同。只读 Python 扫描使用 Windows `D:/anaconda/python.exe`（Python3.13.9 / pandas2.3.3），没有导入仿真器。审计源与日志分别留在 `data-audit-001/`、`data-audit-002/`。当前 [audit_data_coverage.py](audit_data_coverage.py) 拒绝覆盖已有审计 JSON；复查应先把脚本和输出目录复制为新的审计号，再以 `GIT_OPTIONAL_LOCKS=0`、`PYTHONDONTWRITEBYTECODE=1` 与 `python -B` 执行，禁止改旧审计。
