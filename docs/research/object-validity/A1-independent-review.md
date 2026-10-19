# A1 独立实现审查

审查日期：2026-10-19（Asia/Shanghai）。基准：`a754c5ee4bd3922cecd6993d9abb6bfbacdda7e0`。范围为 `experiments/object_model_a1/config.py`、六个候选 JSON、`mapping-schemes.json`、`rules.py` 及其调用的原执行器/冻结加载器。审查者未修改这些实现文件，也未读取任何 Development/Validation/Test 数据。

## 审查结论

**未发现阻止 A1 确定性规则实验执行的缺陷。** 配置、资源竞争统计与 CI 暴露代理的边界一致；这一结论不构成物理模型校准、设备真实性或最终研究对象冻结的认可。

| 审查项 | 已核对实现 | 判断 |
|---|---|---|
| Train guard | 主入口 `load_pool(..., ('train',))`；旧加载器先检查角色，再读取 payload，并校验清单、payload 哈希及路径归属 | 当前入口不会装载非 Train payload |
| 映射适配 | 适配前用 raw fields 交叉检查原 `5*pressure` 映射；只改变三维需求，保留任务身份、原到达、Duration、SLA 和同一环境对象 | 旧资源占用不会被无提示地重复解释 |
| 版本与身份 | 候选文件 SHA、基础源码 SHA、原/派生 workload SHA、原环境及场景身份进入结果；源池清单 SHA 入元数据 | 可追溯来源和映射变更 |
| 六候选 | Legacy L/M/H 与 Observed OL/OM/OH 分开；OL/OH 按无放大映射重新取整；OM 是 M 的需求和容量同除 5 | 数值及解释一致；取整差异需在比较时保留说明 |
| 物理隔离 | 加载候选强制 `power_model=disabled`、`physical_carbon_output=false`；执行器只调用 ZeroPower 记账适配器；导出显式去掉功率、能耗、kg 碳字段 | 没有把新逻辑容量接到旧 IT/HVAC 计算 |
| 容量与主动等待 | 执行器在每项任务轮次先查容量；仅当前能放入的任务可以记为 policy defer；两者分开计数，同一任务可先后经历两类等待 | 不合并等待原因；是一种逐轮观察归因，非反事实因果分解 |
| 资源瓶颈 | 比较该轮任务需求与 `available_before`，允许一个事件包含多个资源；另报独占瓶颈计数、涉及任务数和时间并集 | 不用平均利用率替代瞬时竞争证据 |
| Deadline | 使用原 Duration 和合成 SLA，按实际 finish 判定；ceil15min 仅控制释放 | 不把释放时间误当作完成时间 |
| 完整任务集合 | 单项超容量直接报错；未 drain 完整则报错；不删除任务或补造完成 | 缺失实验不能伪装成完成 |
| CI 暴露 | 使用冻结 TemporalEnvironment 原 15 分钟阶梯 CI，按实际 Duration 精确积分；ceil 释放占用的积分另列 reserved proxy | 比较整段运行 CI，未只比较开始时刻 |
| 统计单位 | proxy 为资源单位×小时×gCO2e/kWh，且元数据声明不是 g/kg 排放；平均执行 CI 为资源小时加权 CI | 可以衡量相同资源工作量移动到何种 CI，不能称真实减碳率 |
| 在线规则信息 | 继承 Threshold 仅查看当前 CI 和单任务 latest-start；161.075 为原有固定常量 | 没有访问未来 CI 来决策；April 常量对年初并非严格历史先验，配置已标注 |
| 最终冻结与参数缺口 | 候选强制 `final_research_object_frozen=false`，开销、设备功率和制冷容量缺口为 null | 未将缺证参数猜测填入 |

## 执行验证

独立运行仅含手工合成任务和环境的测试：

```text
python -m pytest tests/object_model_a1/test_config.py tests/object_model_a1/test_rules.py -q
17 passed in 1.66s
```

测试覆盖需求与容量等比例缩放、角色拒绝、物理模式拒绝、声明不一致拒绝、16 分钟任务的完成/释放区别、GPU 容量等待与回填、EDF、Threshold 在单体 Slack 安全时仍因群体竞争违约、实际/保留 CI 暴露区分及单项超容量拒绝。

实际规则实验启动后的独立快照检查：已生成的前 **112 / 1,440** 项场景结果均完成全部任务，FIFO/EDF 主动等待为 0，未分类容量等待为 0，逐项容量等待分钟加主动等待分钟等于总观察等待分钟，结果无功率/能耗/kg 碳字段。运行时 `script_sha256` 与当时被审查的 `rules.py` 字节一致；源池清单 SHA 为 `327e941eaa715aa9d25a30601502183cae27c1e7a83b0da7f31e2c0e251775ca`。

随后对最终 `metadata.complete=true` 的 **1,440 / 1,440** 行做全量独立核对：所有到达任务均完成、unfinished全为0；FIFO/EDF主动等待全为0；未分类瓶颈事件为0；容量等待加主动等待与总观察等待逐场景一致；三种资源所有drain峰值均未超容量。完整聚合库存包含90组、270个资源利用率组。无缺失组合。

最终源码身份另作严格核对：真实 producer 为提交 `1620060cac1590047221614c761b79ff1e0a588d`。以 `git show` 取得该提交的文件字节再计算 SHA256，`rules.py` 精确等于运行时 `47a03d37a5b61b2510e8f937e5357307b05ffe44d310df3f374664e5343666d7`，适配器 `config.py` 精确等于 `28e491e4bdbc81b51e5e7b439687d3816f7477d41bf3eedd08356c01bc2832ee`。最终交付的 rules 增加了身份核验后处理，config 将继承的原统计放入 `source_statistics`，故不再声称其当前字节就是运行时字节；这些后续修改没有重放或改变实验指标。

`identity-repair.json` 记载核验1,440行、实际修复 **0行**，输入与输出CSV同为 SHA256 `68a9267e3d21bd30236771bdc3b9494d6369201d3e98e14084e9bccbd2ea0532`。原V2身份同时保留两个等值fragment别名，原输出已有完整身份。不能把这次核验写成补填或修复了结果。

新增的 `tables.py` 还在正式发布前拒绝 `complete=false`，核对每组16个片段、1,536个利用率槽、类型分区分母、FIFO/CT配对输入身份与各资源小时守恒。5项独立合成测试通过；最终全量内存预检生成288个场景配对、18组配对汇总及108个任务类型配对。此核验不重新运行调度，也不读取任务/环境payload。

## 必须保留的解释限制

- `long_slack_capacity_missed_tasks` 表示“发生过容量等待且最终违约”的交集，不证明所有违约仅由容量造成；Threshold 还可能先主动消耗 Slack。
- 不同策略的资源工作量相同且全部完成时，可以配对比较执行 CI 暴露；不可把 proxy 数值跨 CPU/GPU/MEM 相加当能量，或套旧物理功率转成排放。
- 不同映射的 proxy 绝对值会随单位/放大倍率改变；跨映射解释应看单位一致的归一化压力与加权执行 CI，Legacy/Observed 绝对 proxy 不能直接判优。
- 161.075 的规则是继承常量的诊断对照，并非重新通过 A1 结果挑选的最佳在线阈值。未来 CI 理想分析必须单列并标明其约束范围。
- 字段附录用中文概述官方字典并附主源链接，没有发布官方 README 全文或连续长段逐字引用；必要字段名与数值单位说明均用于事实核对。
