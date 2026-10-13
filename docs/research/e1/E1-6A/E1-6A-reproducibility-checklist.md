# E1-6A 复现与交付核查清单

本清单于2026-10-13（Asia/Shanghai）整理，并按实际证据更新。已勾选条目完成对应核验，未勾选条目仍待最终验收；实际结果绑定campaign-001输出及其哈希。实现正确性不等于性能优势，新策略性能不用于筛选是否保留结果或是否运行开发Validation。

## 输入、源码及边界

- [x] 在所有新增源码/测试成形且最终微型测试通过后冻结源码，随后科研执行期间不再修改生产源码或测试。
- [x] 原科研目录只运行实验；发布目录只接收审核过的小体量源码/结果。不修改main、原dirty状态、旧checkpoint、旧归档或旧provenance。
- [x] 六组既有冻结包均核验payload SHA、workload/environment身份、任务数、SLA1.5、倍率5、容量50000/1000/80000、阈值161.075、真实CI/weather和UTC范围。
- [x] 原有Train seed510001、Validation520001；2×不是独立到达形态，不能将不同副本seed表述为独立真实负载样本。
- [x] 在线策略只有current snapshot、当前CI、已观察ArrivalHistory；离线SLA下界、完整未来工作负载、未来CI及其他策略结果不进入policy。
- [x] 明确Test继续封存，本轮没有正式PPO研究训练/旧权重修改或最终Safety Shield开发；旧PPO接口toy回归如实单独披露。

## 执行器、方法与阶段顺序

- [x] 唯一LocalExecutor执行内核、15分钟网格、原duration/deadline、非抢占聚合CPU/GPU/MEM、EDF全队列排序及一次遍历回填保持原语义。
- [x] 原IT/HVAC模型连续保留状态。所有策略在同一资源释放规则下计算占用；精确任务finish与下一网格释放不同。
- [x] 固定a_u=a_q=a_h=1、rho0=.5、k=1、K=8、cold2拍、H120分钟，Validation结果不触发v1系数修改或覆盖。
- [x] no_history仅去掉历史压力项，沿用共同cold；其他消融仅去掉所声明项或替换加权聚合为max-required。
- [x] 同拍start立即更新u与pending身份；主动defer和capacity等待仍在竞争集合，但不占运行资源；历史到达每任务只登记一次。
- [x] 微型正确性通过后，先依次运行Train1/1.5/2；完整Train功能gate通过并保存后，才运行Validation1/1.5/2。
- [x] Train gate核验完整Drain、资源/任务守恒、固定参数和源码、旧基线精确回归；不要求方法胜过RiskOnly或DMR<=5%才保留/继续。
- [x] 开发Validation已在E1-5C诊断中观察过，本轮仅为开发消融；不表述为独立泛化评测。

## 八策略组与历史回归

- [x] 每组包含EDF参考、旧EDF+Carbon/RiskOnly以及Full/no-history/no-urgent/no-utilization/max-required；总计六组×八策略。
- [x] 三个旧策略继续调用旧make_cars_policy并接收原bundle.parameters；没有新系数、风险阈值或规则注入旧策略。
- [x] 每组保存同一完整有限任务集合和相同环境身份；所有到达任务均有唯一终态。
- [x] 各策略先完整Drain，再选择本组max(observed_end)，早完成策略用原live物理模型积分真实空闲尾段；安全cap不足时完整记录未完成且不作等工作量减排宣称。
- [x] 固定到达、各自排空和公共窗口分别核算；所有已完成策略三维资源小时相等。
- [x] 每组baseline-regression.json为matched=true、checked=true、policies_checked=3；真实campaign没有使用microfixture的reference=None分支。
- [x] 旧任务ledger、任务事件、policy事件与policy ledger完整一致；到min(新公共终点,旧公共终点)的interval和policy queue前缀完全一致。
- [x] 新公共终点内重新积分旧基线，未直接减历史E1-5C不同范围总碳；各组保存新空闲尾段和公共终点。
- [x] 对六个真实cohort逐个核对config.reference有值、文件存在、旧run_status=complete及旧output_hashes完整匹配。

## 诊断、指标与科学解释

- [x] DMR/违约数、三种能耗kWh、碳kg、JCT/等待分钟、pending、capacity/policy等待、资源小时、排空小时和算法开销齐全。
- [x] 候选诊断仅在高CI且S>=15分钟时有budget；低CI/无slack早退不被误记为B=0，capacity失败不被误记为policy defer。
- [x] B分布至少包括所有eligible检查与每任务首次eligible检查；披露分母及重复候选。主导资源按q/C确定，零资源和tie单独说明。
- [x] 检查B是否多数相同、方法是否退化为EDF、是否丧失主动低碳等待、是否仅改善SLA而无碳收益。
- [x] 汇总理由触发、各资源分项贡献与CPU/GPU/MEM主导类别；解释削去各模块造成的轨迹差异，不能只看最终数字。
- [x] 同窗口同倍率比较，保持5%门槛；高压另报相对EDF的额外/避免违约、碳变化和Pareto权衡，不混合平均掩盖失效。
- [x] 配对任务证据包含成功和失败；对照差异只是轨迹关联，不能认定单个事件严格因果，也不能将启发式预算称为SLA保证。
- [x] max-required是预声明消融，保留全部负面结果，不在Validation搜索新的权重或择优改写v1。
- [x] 运行时间与科学输出分离；计时包括总run及order/decide/hook，报告候选数和静态缓存优化等价性证据。

## 回放、保护与发布

- [x] 保留实际命令、WSL/Python/依赖、原始输入哈希、bundle身份、源码版本、输出哈希、失败日志和阶段顺序证据。
- [x] 新目录拒绝覆盖，campaign仅允许跳过已经哈希核验且同source/input的完整组；失败组必须改新运行ID，不能覆盖。
- [x] 微型八策略严格科学回放已通过；真实Train1×八策略完整组回放也已退出码0、科学字节相同，挂钟与provenance分离。未声称六组均完整回放。
- [x] final保护核验对225228个真正旧普通科研文件及原.git元数据通过；保留原capture receipt原字节。
- [x] 原保护receipt在capture程序启动后写入了一份本轮新增tests/e1/test_task_risk_state.py；最终比较同时从before/current inventory排除此明确新增路径并说明分类，未修改原receipt或把新测试算作旧文件改动。
- [ ] 发布副本逐文件SHA匹配必要源码/配置/测试/精简报告/小CSV，检查暂存及忽略规则；完整workload、ledger、diagnostics、模型、日志和完整私有归档不上传。
- [ ] 发布目录研究分支正常commit/push，先核对远端历史没有未知新增提交，main不变，禁止force；推送后核查SHA与Commit链接。
- [x] 最终结论区分实现正确性、实际性能、未证新颖性及未知未来任务下的SLA限制；交付后停止，不自行进入E1-6B。

## 已做的只读先验核对（不等于实验完成）

独立脚本`check_existing_inputs.py`未导入策略/执行器/物理模型，用标准库核验六个现有Train/Validation包的6份payload及对应旧实验完整manifest、所需三个旧基线文件；`existing-input-check-001.json`记录实际检查结果。原始保护receipt SHA256为`28e6b9d3a7382b59b2a688a7d48e29cb304c0a0fca7d4a5b1db9bb3a8319b7c1`。该receipt的files中225229条只有上述一个本轮新测试属于新增排除项；真正旧文件保护范围因此为225228条，另有skipped/nonregular和已声明排除目录的边界。

驱动通用微型API允许reference=None以构造独立fixture，因而其regression会显示checked=false；真实run_campaign总是将configs/e1_6a.json六个非空reference传入run_group。必须从真实各组输出确认checked=true，不能将fixture gate用作真实Validation发布证据。


本清单在最终Git提交前封存；commit/push状态不提前勾选，随后由最终Git回执和远端Commit可核验。真实48结果与正式图表QA均已完成，原裁切DMR图标FAIL，独立修正图PASS。
