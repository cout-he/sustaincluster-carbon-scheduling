# E1-7A 选样独立交叉审查

审查对象：`diagnostic_selection.py`、当前只实现 freeze 的 `diagnostic_experiment.py`、`configs/e1_7a.json`、预注册文档，以及 `selection-001/E1-7A-selection-manifest.json`。本次只读核验，没有运行新真实模拟、求解或干预；没有读取 Test/Validation。

## 当前冻结材料结论

当前选样符合预注册，无发现后验按违约或碳收益切换类别。selection payload SHA256：`e62b7eddcde17ee80c5eb3300b36c3a2ff4880a8899c24adc0f6fbd8b9b6c361`。独立 canonical JSON 重算匹配；配置 hash、selector 文件 hash、三档 bundle 文件 hash、每档六个冻结输入文件 hash、注册参考科学文件 hash、全部候选特征文件 hash均匹配。

冻结 Train1/1.5/2 任务数量分别5542/8313/11084，原 source/资源倍率5/SLA1.5/容量50000/1000/80000 与既有 bundle 绑定。`load_bundle(...allowed_split='train')` 在跟随 payload 之前拒绝 Test/角色不符，并核验窗口、物理配置、资源及任务源身份。

## 排序和任务选择复核

独立从完整候选特征重算四类 A 与六类 C 的阈值、类别顺序、排除重复和 tie-break，结果逐项相同。

- A：CPU/GPU降序、old pending降序、低压力利用率升序；均至少10个 pending，按最早UTC打破同分，按预注册类别顺序排除已选时刻。
- C：GPU/CPU降序、历史rho高/低排序、短/长grid slack排序；同分按UTC、stable task ID、EDF rank，排除已选 task ID。
- A11个快照，分档3/4/4；若每个快照执行A1/A2，则为22个有限子问题。40是上限而非强行填充：实际子集数量含18、22、27、36。全部 selected IDs 属于当前 pending，且正好是原 immutable task 特征按EDF排序的前40或全部，不生成新特征。
- C12个事件，1.5×/2×各6个。逐项匹配对应已保存 reference candidate、occurrence1、真实can-fit/started、arrival已知及solo grid slack至少15。
- 1×所有到达拍 `maximum_old_pending=0`，因此 queue_long 缺失确实不存在；不是被其他类别去重后隐藏，不补造类别。1.5×/2×最大 old pending为199/434。

历史 `common_end` 文本使用 `+08:00` 表示同一时刻；换算为预注册的2023-05-03 14:00 UTC。审查没有通过 PowerShell 日期自动转换重写原文本，独立 JSON hash 保留原始字符串。

## 在线信息边界

`build_selection_records` 仅按当前拍读取 arrival、参考候选可用资源、pending、任务已知 duration/deadline/需求及过去120分钟 arrival history。最新启动由原 deadline-duration 后 floor-grid取得。未将真实finish、最终DMR、窗口总carbon、未来CI下降或反事实delta用于排序。选样是离线遍历参考窗口，不声称是在线选窗；注册在线干预控制器仍只读取当前已到达任务。

`read_reference` 为哈希审计读取完整Train账本和未来参考占用，这是离线输入，当前 freeze 将 outcomes 读入但不用于选择。后续A2可使用离线固定背景；不得将该未来背景或oracle CI传入实际在线EDF/干预决策。

## 开发防护建议（不否定现有冻结选择）

当前 `validate_config` 已固定split、三档输入、容量、版本和主要求解预算，但未逐项验证配置中的所有解释字段，例如C factors/category阈值、occurrence、B delta/CI bins、historical common end。实际配置与硬编码规则一致，现有 hash封存未发生偏差。建议在最终驱动源码冻结前补充严格验证，以避免将来修改这些声明后仍执行原硬编码选择，或扩展C factors后与target_pairs=12声明不一致。保留本次selection-001，不通过重选修正。

尚未审查随后待扩展的A/B/C实际运行驱动、输出与公共窗口补齐；这些需要单独执行前审查及结果审计。本结论只适用于当前冻结材料与选样。
