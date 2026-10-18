# A0 研究对象有效性诊断实施计划

> 执行模式：独立只读诊断脚本与确定性统计；不修改已有实现，不训练模型。使用 executing-plans / subagent-driven-development 的验证要求完成各自有界子任务。

**Goal:** 在有限预算内判断当前单DC的资源竞争、Deadline约束、低碳迁移空间与物理模型可信度。

**Architecture:** 从V2 `bf8a35ef647f19b5ef10ce1b99f723370b7a0732` 建立独立分支 `research/object-validity-audit`。资源诊断复用原执行器；碳机会分离容量无约束的理想指标与资源可执行规则比较；物理扫描只使用明确标注的手工输入。

**Tech Stack:** Python、现有冻结Train loader/LocalExecutor/JointPowerModel、pytest、matplotlib；CPU单线程。

**Spec:** 本轮用户“研究对象有效性诊断 A0”完整要求；报告直接回答A–E并在交付后停止。

## Global Constraints

- 全轮wall预算起点2026-10-18 13:03:54 UTC，截止13:33:54 UTC；到点不再开新计算。
- 不读取September payload或July Validation，不装载June Development，不用开发集调参。
- 只装载现有池的Train角色；环境尾段仅限对应已授权Train冻结environment。CI描述统计仅到达窗口，完整工作责任核算单列尾段。
- 所有原任务、到达、duration、需求、逻辑容量、冻结场景、生产算法与物理方程不变。
- FIFO/EDF均不主动推迟，容量不足正常等待并允许回填；capacity wait与policy defer分别记录。
- 无容量理想CI机会明确不等于可实现的最优调度，不等于完整系统碳收益。
- 阈值规则沿用既有161.075 g/kWh，不在本轮拟合/优化；不能宣称该旧April常数早于所有Jan–May Train。
- 只提交独立诊断代码、测试、报告、聚合CSV和图；任务明细与大日志留本地research_notes。
- 研究对象候选最多3套，参数范围由本轮数据导出；本轮不实施任何候选。

## Review Focus

1. 低均值不能掩盖短时峰值；报告逐资源分位数、峰值、阻塞任务和并发需求。
2. 15min采样、精确finish与网格release区分；完整任务CI积分覆盖ceil(duration/15)责任。
3. 同源增强与重复trace不是独立实测；pressure/SLA保持分组、保留场景ID。
4. 功率为零的资源专用探针不能输出为真实能耗；物理扫描手工输入不是观测数据。
5. 用现有函数能算出数值不等于硬件已标定；记录配置覆盖、单位疑点和可解释性风险。

## Tasks

- [x] 配置/身份：核对V2基线、源哈希、Train池、角色与代码覆盖链，生成协议与输入身份。
- [x] 资源：synthetic FIFO/EDF、backfill/释放/归因测试；预算内尽量96场景×2规则；输出利用率、瓶颈、需求、容量wait、JCT/DMR、slack。
- [x] 碳：合成完整区间CI积分测试；16原始Train×2SLA单任务机会；按Train元数据选择代表片段做固定22°C Immediate/Threshold成对仿真。
- [x] 物理：合成负载×天气×setpoint扫描及交叉检验；运行配置数量/分项和固定背景诊断。
- [x] 整合：核对不同统计分母；生成轻量CSV、说明图、A–E证据判断及最多3候选。
- [x] 验证/交付：仅新增文件diff检查、测试、图表QA、源码/数据身份与未完成项；提交审计分支，停止等待选择。

## 执行记录

资源192/192、碳规则54/54、物理1440/1440完成。26项测试通过；原145个源码/配置逐字节不变。会话额度中断后用户要求继续，只整理既有结果与合成验证，未扩大实验。跨中断wall超出原30分钟，报告明确披露。下一阶段未启动。
