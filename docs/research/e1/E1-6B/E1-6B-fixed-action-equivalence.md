# E1-6B B0：真实冻结Train固定动作执行等价性

本项只验收新四动作环境是否复用旧执行语义，不是新Train性能实验、PPO训练或算法收益结论。没有读取Test或开发Validation输入，也没有修改E1-6A结果。实际命令最终退出码为0。

## 1. 输入、方法与核算终点

只加载既有Train1.5×/2× bundle（`load_bundle(..., allowed_split='train')`）：分别8313/11084个任务，同真实CI/weather、50000/1000/80000容量与旧物理模型。参考为 `research_notes/E1/E1-6A/campaign-001/train-{1.5,2}` 中原归档。每条新轨迹先完整Drain，再沿用同一个executor/power_model空闲积分到对应旧组公共终点 `2023-05-03T14:00:00Z`。若自身排空晚于旧终点，脚本会报差异并拒绝截断；本次未发生。

| Train倍率 | 新固定动作 / 过滤 | 风险状态 | 旧参考 | 完成任务 | 自身Drain终点UTC | 核心文件byte相等 |
|---:|---|---|---|---:|---|---|
| 1.5× | 0 / 关 | 关 | EDF | 8313 | May3 11:00 | 是 |
| 1.5× | 3 / 关 | 关 | EDF+Carbon | 8313 | May3 14:00 | 是 |
| 1.5× | 3 / 开 | 开 | Task Full | 8313 | May3 13:45 | 是 |
| 2× | 0 / 关 | 关 | EDF | 11084 | May3 11:15 | 是 |
| 2× | 3 / 关 | 关 | EDF+Carbon | 11084 | May3 14:00 | 是 |
| 2× | 3 / 开 | 开 | Task Full | 11084 | May3 13:45 | 是 |

全部pending/running终态为0。所选risk_state开关包含开启和关闭，未改变相应固定动作的核心行为；风险状态关闭只是附加43维置零，不改变共享分配或能碳机制。这里不是对开启/关闭风险状态训练效果的消融。

## 2. 比较对象与实际证据

每条轨迹分别检查以下5个完整文件SHA256相同，即字节相同：

- `ledger.jsonl`：完整任务特征、开始/完成时间、终态和deadline结果。
- `events.jsonl`：到达、入队、启动、完成释放事件。
- `policy_ledger.jsonl`：逐任务主动defer、capacity waiting与等待统计。
- `intervals.csv`：全部228拍公共区间的资源、IT/HVAC/总电能和carbon。
- `summary.json`：完成/违约/删失数量、Drain、观察终点和能碳；该文件不含policy名称。

完整 `policy_intervals.jsonl` 也字节相同，包括每拍排序、真实start/defer/capacity集合、pending与running。另按拍验证上述三类事件ID集合相同，并验证 `policy_events.jsonl` 的(time,task_id,event,decision_rank,CI,can_fit,available_before)逐行投影相同。

新策略解释性policy name/reason与旧基线不同，未要求这两个标签字节相同。threshold metadata也不纳入该投影：旧EDF没有threshold字段数值，新四动作policy带固定161.075字段，Action0仍从不主动defer；这只是标签差异。实际CI、排序、容量可容纳结果、可用资源与事件已比较。

`interface-regression.json` 确认：6条全部完成、exit_code=0、60个已读取原输入/历史结果文件SHA256前后不变、10个接口/执行/物理源码SHA256前后不变。新结果均写入全新 `b0-real-interface-001/`，没有覆盖参考归档。

| 证据 | SHA256 |
|---|---|
| `b0_real_interface.py` | `396c63960b185338511aef0c55300a7d443430cd181ceb1432faf127ffe343f9` |
| `b0-real-interface-001/interface-regression.json` | `bf8be23fa04e695fed078b6a5e668cbd6a11da9acea64c740e91483968ebfbc3` |
| `b0-real-interface-001/interface-summary.csv` | `c1f8c82be18d94a6b1dad8c95f49309d7ae4795ab0309206d0ab6828a15ffce5` |

完整manifest同时保存五类结果文件及policy文件输出hash与每条scientific core content hash。时钟runtime独立记录，不纳入核心科学一致性hash。

## 3. 实际命令与运行记录

```text
wsl -d Ubuntu-22.04 --cd <WSL_RESEARCH_ROOT> -- env PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=<WSL_RESEARCH_ROOT>:<WSL_RESEARCH_ROOT>/research_notes/E1/E1-5B/runtime-deps <CONDA_ENV_PREFIX>/bin/python research_notes/E1/E1-6B/b0_real_interface.py --output research_notes/E1/E1-6B/b0-real-interface-001
```

成功日志为 `b0-real-interface-002.log`。首次命令漏把项目根加入直接笔记脚本的PYTHONPATH，import `experiments` 失败，退出码1，尚未创建结果目录；错误原样保留在 `b0-real-interface-001.log`。修正启动路径后运行6条，未修改生产源码或旧输入。实际每条wall time约5.18–13.66秒，受同机其他进程影响，只用于复现成本记录，不作策略效率优势结论。

## 4. 验收边界

新环境固定Action0、无过滤Action3、过滤Action3在两个真实Train资源压力集合上精确复现旧EDF、EDF+Carbon、Task Full核心轨迹及物理积分。结合54项新旧环境微型测试，B0接口语义获得真实输入回归证据。

此证据不表明Task Full改善SLA、不保证风险过滤有效、不证明新PPO性能；更不替代12模型真实训练、57条公共开发Validation、后续模型回放或Test泛化。E1-6A负面结果保持冻结。
