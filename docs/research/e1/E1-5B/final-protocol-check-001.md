# E1-5B 协议一致性与发布行尾兼容检查

## Material Passport

- 日期：2026-10-12。
- 范围：只读预注册 config、data-freeze/design/ppo-config、pilot产物、现有测试日志、原始与发布DC JSON及现有 `.gitattributes`。仅新增本记录，并修正本轮新 data-freeze/机器协议遗漏的 checkpoint SHA 最终 tie-break 文案；原预注册 config 已有该 tie-break，算法、权重、数据或参数未改变。
- 审核时状态：pilot已完成，55102/55103在按同预算运行，Validation/发布核验尚未完成。这里的“协议检查”不是整体交付完成。
- Test未执行；没有读取 Test episode 或发起任何新训练/科研实验。

## 协议和实施一致性

| 项目 | 审阅结论 |
|---|---|
| 数据窗口 | Train May1–3、Validation Jul8–10、Test Sep9–11，UTC05:00、48小时，Drain安全末端分别Jun2/Aug9/Oct11；一致 |
| CI/物理 | 固定161.075与4月720小时median来源；原容量50000/1000/80000、22°C、resource_multiplier5/SLA1.5、同一IT/HVAC；一致 |
| Action/Observation | 原40维、Discrete3模式语义及 RiskOnly120分钟/0.5与1.0/0,1,2拍；原kernel/Reward不改 |
| 输入控制 | Train stress510001固定所有episode与RLseeds；Validation520001；其他seed只预留；2×是相同父任务全量复制，不是独立realtrace |
| PPO | SB3 2.9.0、MlpPolicy64/64、CPU单线程确定性，参数与ppo-config一致；无VecNormalize |
| 预算 | smoke1024/55100；pilot=正式55101候选20480；55102/55103各20480；不因结果换预算/Reward |
| 终止 | 完整ownDrain训练；有限数据安全截断抛错；learn步预算尾部单独记录，没有伪造环境截断 |
| 评价 | 共同endpoint包含最慢candidate/heuristicDrain；早排空物理实例持续真实空闲积分；未排空禁止等工作量优势 |
| 选型 | 所有完成且DMR≤5%，再公共carbon/DMR/seed/checkpointSHA；无合格者none，保留全部candidate |

审阅原 `e1_5b_ppo.json` 的 SHA256仍为 `d9bc61383558d211aff2d7aa466b7396d12db8bda327b6cb8b519e20b6019e28`。只读Python字典比对：pilot的 `training_config.json` 去掉仅新增 `training_identity/source_identity/seed/total_timesteps` 后，与预注册 config **语义完全一致**，没有任何参数差异。pilot实际Train role、stress510001、factor2.0、RLseed55101、20480steps。

`E1-5B-design.md` 简略说“容量等待且模式结果不同”，`E1-5B-data-freeze.md` 与 `run_precheck.py` 给出精确定义：all_complete，并比较 `(DMR, mean_wait_minutes_started)`；这是详细化，没有事后改为另一判据。source/config哈希与实验驱动SHA分别记录，不通过修改旧回放检查适配新模块。

## 当前实际证据

- 原科研完整测试日志 `full-tests-001.txt`：**324 passed / 1172.81s**。发布副本对应测试仍须独立运行，不能据此直接推定通过。
- pilot summary：20480steps，94个完整episode，另65步partial episode，548.178202414s，peakRSS946292KiB（进程高水位）。参数SHA由 `0323509e39a0ec58d9a51d828fb192f172cff00ade62872dca3feb4aff4a582b` 变为 `8a172f4e75348bfad2e055759b6750be76d5858e4609ba8c3761ae8219e5a80e`，roundtrip=true，optimizer state12项。满足预注册7200s/8GiB资源gate。
- pilot checkpoint SHA=`cd3d7b533fdfca455947ff00171d60da2c94dd114dcfc5b1ffbdeefa4a191d3e`。这仅证明本候选真实更新并可加载，不是Validation性能或Test泛化结论。

## 发布兼容方案审阅

原DC JSON `bbe439176605512523a3a0ecd2bd22f18fdb9abf6ae6e782217157beedbf705b` 含CRLF；当前发布research中的DC JSON `b1675fbca01217565f469c1199584b88afcda2de8d8502cffcad19ae26b6ecd2` 含LF。移除CR后的文本精确一致，没有任何物理字段差异。

批准工作流要求原科学配置raw SHA不变。root拟在**独立发布副本的research分支**恢复原DC JSON原字节，并追加局部 `.gitattributes`：`configs/dcs/dc_config.json -text`。该方案与现有 `* text=auto eol=lf` 可以共存：更具体的后置规则关闭此路径的文本转换，其他source/tests/docs继续LF。也可显式增加 `-eol` 表明该路径不采用全局eol；无须调整生产模块或批准的科学参数。

这两文件属于发布字节身份兼容修改：不重新设计physics、不改变main、不修改原科研文件或历史provenance。现有旧提交仍保留旧LF blob；research新commit保留原CRLF blob，使clean checkout下strict raw SHA与批准原值一致。JSON的公开内容此前已存在，不会因此加入原始workload、模型或私有数据。

提交前必须验证：

1. explicit文件清单包含DC JSON与局部attributes，注明目的与前后SHA。
2. `git check-attr text eol -- configs/dcs/dc_config.json` 确认路径text为unset；Git staged blob与commit blob原字节SHA应为bbe439…，不能只检查工作树文件。
3. 原工作区文件不变、publication main仍749dc6c…，scope仅research；JSON解析值/去CR文本与旧published文件一致。
4. 发布副本相关测试（尤其bundle fixture与rawphysics hash）实际通过；读取新commit blob或fresh临时checkout确认checkout不会重新LF化。
5. 正常push仅research；远端history如出现未包含本地的变化则停；README仍明确原始workload/CI/weather及完整checkpoint是外部本地依赖。

这是一项可接受的最小兼容修复建议，实施与commit验证在本记录时尚未完成，不能写成已经成功发布。

## 保留的模型边界

只有一Train与一Validation窗口；workload来自重复平移trace，UTC/elapsed特征可能记忆训练周期；三个RLseeds测优化采样变异。CI小时值即刻可得、duration到达时已知、syntheticdeadline、wet-bulb代理及非抢占假设仍保留。新模型只是全局模式选择PPO-v0，不是任务级预算或SafetyShield。各策略ownReward不能替代共同窗口carbon/DMR/JCT。本轮不以Validation或任何单seed结果声称统计显著或Test泛化。
