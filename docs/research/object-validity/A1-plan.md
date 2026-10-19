# A1 研究对象重建实施计划

**Goal:** 建立可追溯的独立逻辑调度候选，实际验证资源竞争—Deadline—CI机会，交付后等待用户选择，不进入A2。

**Architecture:** 仅在新增 `experiments/object_model_a1/` 中实现版本化需求适配器、候选配置与只读Train规则实验；不调用功率模型。观测用量×1与Legacy×5明确隔离，申请量映射无字段时禁用。所有旧源码、数据、场景、checkpoint保持原样。

**Tech Stack:** Python3.10、原LocalExecutor资源账本、NumPy/pandas、pytest、matplotlib；两进程确定性规则回放，不训练模型。

**Spec:** 用户完整A1任务。候选只推荐、不最终冻结；A2物理修正仅出方案。本轮是A0既有诊断流程的有界扩展，执行范围由本轮用户任务明确授权。

## Global Constraints

- 基准 `a754c5ee4bd3922cecd6993d9abb6bfbacdda7e0`，分支 `research/object-model-a1`。
- 仅现有Train角色任务及环境；不整包unpickle年度PKL，防止加载禁用月份。原年度文件仅metadata和处理源码核对。
- 不读取June Development、July Validation、September Test payload，不训练PPO/MAPPO，不继续V3。
- 保留原到达、Duration、任务集合、SLA1.5/2与CI；倍率和容量有显式版本，不能反推预留资源。
- 不导出旧物理模型电能或碳排；CI积分代理标明非kgCO2e，未来CI理想分析忽略容量时必须标记。
- 实验固定：Observed OL/OM/OH×96Train×FIFO/EDF/Threshold=864；Legacy L/M/H×96Train×FIFO/EDF=576，共1440次，不按得分跳过。

## Review Focus

1. 官方使用量不是资源申请；多实例聚合与plan字段丢失必须披露。
2. ×1需求必须同步重推容量，不能仅改GPU分母；精确/5变换等价与64核/4GPU取整差异分别说明。
3. 角色验证先于payload；相同任务ID、原快照SHA、环境不变、映射后工作负载SHA均记录。
4. 单体超容必须报告；容量等待和主动推迟区分；完整排空，不删违约任务。
5. 总体DMR≤5%不代表每片段达标；比较最坏值、超阈片段数和长Slack受竞争影响群体。

## Tasks

- [x] 字段：Alibaba官方主源与本地预处理链交叉验证，形成字段证据与阻塞申请量方案。
- [x] 配置/映射：合成测试先红后绿；三Legacy与三Observed容量、SLA/CI/设备待校准项独立版本化。
- [x] 规则：合成等待/回填/释放/未来CI与结构不可行边界测试；全1440个固定实验，输出精确pooled与逐场景指标。
- [x] 机会：容量条件下剩余Slack、长Slack争用风险、规则CI代理、按任务Duration分组机会；不计算物理kg。
- [x] 报告：回答A–G，配置依据、inventory条件草案、原代码不变证明、轻量CSV/图/来源SHA。
- [x] 验证/交付：独立审查、测试、图QA、原分支/文件不变检查，提交推送后停止。

## Figure Contract

Python沿用既有偏好；两张quantitative grid比较容量的等待/DMR/局部风险、资源利用和CI代理/Deadline冲突。只用真实聚合CSV，16源片段是描述性重复，压力/SLA增强不是独立观测；不加统计显著性标记。输出可编辑PDF/SVG和600dpi PNG，字体≥6pt，静态检查、PDF字形审计及目检。

## 执行结果

固定1440回放全部完成，650.59秒；两种比例变换对照均通过。50项测试通过，字段/设备与结果已有独立审查，全部原653个tracked文件无修改。候选仅推荐，物理参数缺口保留，未训练/未进入A2。运行producer1620060，最终交付追加身份核验(0行修改)、来源统计命名隔离及报告。
