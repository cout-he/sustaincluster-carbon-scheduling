# E1-5C 独立源码复审 001

## 审阅范围与状态

只读审阅 `experiments/e1/sla_feasibility.py`、`sla_diagnostics.py`、`workload_pressure.py` 及对应新测试；交叉阅读原 `engine.py`、`policies.py`、`cars_state.py`、`rl_training_utils.py`、`rl_evaluate.py`、`temporal.py`。没有修改新旧源码、运行策略/模型、读取 Test输入或访问网络。本记录是数学与接口审阅，**不是独立执行 pytest 的通过声明**；实际测试与实验日志由主执行流程另行保存。

结论：当前批准的 Train/Validation 冻结数据与调用方式下，未发现阻塞实验的数学方向或历史覆盖问题。下界应明确作为**资源工作量必要条件的保守下界**，不称充分可行性、EDF理论最优或严格因果分解。

## 1. raw_deadline 的集合与区间方向正确

`task_timing()` 与真实执行器同样先将 duration构造为 `pd.Timedelta`，由纳秒值计算向上取整的资源占用拍数。实际 SLA为 `start+duration <= 原deadline`；释放为 `ceil_grid(start+duration)`。

raw模式取 `u` 为某个网格到达时刻，`v` 为某个原deadline的向上网格取整。对所检查的网格 `v`，`deadline<=v` 与 `ceil_grid(deadline)<=v` 等价，所以数组中的取整不会把原deadline晚于v的任务错误加入 `J(u,v)`。若集合内任务全部按时，它们开始不早于u，实际完成不晚于原deadline<=v，资源释放也不晚于网格v。因此完整占用工作 `q_ir * p_i` 均落在 `[u,v)`，必需不超过 `C_r*(v-u)`。

检查只枚举实际arrival × grid-ceiled deadline端点，未声称连续实数端点区间的完整分析；未发现超载仍不能证明非抢占多资源装箱可行。deadline精确完成语义没有被改成“资源释放前完成”。

## 2. occupied_deadline 是合法、单独标注的资源约束加强

按时的合法网格启动满足 `start<=L_i=floor_grid(D_i-duration_i)`，相应释放满足 `release<=L_i+p_i`。于是将这个**最晚按时占用释放上界**作为分析deadline，仍是每个按时任务必须满足的资源工作量必要条件。它可以早于 `ceil_grid(D_i)`，因此可能检出raw模式没有发现的网格启动限制。

这个值不是新SLA deadline，也不是修改duration；函数只读TaskSpec，输出 `interval_mode`、原SLA规则与占用规则。建议最终报告继续分别展示两个模式及其端点，而不把加强后的集合叫作原始 `J(u,v)`。

## 3. 最小违约基数与浮点认证方向正确

单一资源、单一区间超载时，违约任务必须至少使按时子集的工作减少超载量。按降序取最大的任务资源工作，给定k个移除任务能够释放的总工作不可能更大；最少需要的k因此是违约任务数的必要下界。它允许把一个miss任务的完整占用工作从区间全部删除，这是对真实miss任务可能仍占部分资源的乐观松弛，方向仍保守，不会因此夸大miss下界。

Fenwick树的浮点累计只用于选候选区间；最终每资源代表及top witness用原任务、`math.fsum`重新求总量与降序前缀，再认证基数。`1e-10*max(1,required,capacity)`容差增加可用工作预算，可能漏掉极小超载或降低下界，不能提高已认证下界；它未进入真实资源分配器。树的近似排序可能没有找出全体区间中的最强证据，但任何最终认证witness仍是合法下界。

跨CPU/GPU/MEM、多个区间与intrinsic唯一任务数取max，未相加。重复身份不会重复加入intrinsic计数。必要下界率**严格大于5%**才可在该模型下排除DMR<=5%；下界不足不能判定5%可行或不可行。EDF违约率不能替代这个下界。

## 4. 历史保护、角色与公共核算

`run_diagnostics()` 在创建输出或打开历史bundle前拒绝已存在输出、Test角色、训练开启、旧配置SHA改变及未批准历史组/新倍率。历史group只读，核验已完成状态、产物hash inventory、Workload/Environment身份、必需baseline账本与 `formal_test_evaluated=False`。所有新Validation 1×/1.5×输入先冻结并保存身份，再启动两组基线；不存在训练或checkpoint加载调用。

新group调用旧 `evaluate_group()`，默认三基线、无候选，完整Drain后使用同一持久物理对象延展到共同终点，不能通过新分析器改变旧算法。历史Train/Validation group的公共终点沿原归档保留：历史Validation可能包含六策略共同终点，分析仅筛出其中三基线，不能重写原账本。当前E1-5B记录中最慢启发式已经覆盖最慢PPO；即便将来不相等，同组保留的较长共同端点也应明确披露，而不是私自缩短。

必要条件分析仅在执行后离线调用，未输入policy、actor或Reward。`experiment_sources()` 保存源码身份并在结束核对不变；异常保存到新run_status而非覆盖旧档案。原数据/模型完整保护仍需本轮主流程的前后哈希证据，本审阅不能替代它。

## 5. 非阻塞说明与测试覆盖

离线 `workload_pressure._rolling()` 采用固定120min分母、cohort前零贡献并记录覆盖；旧RiskOnly采用有效已观测分钟数、t0 unknown/cold。前120min两者不是同一rho，报告必须区别，不能声称策略内部历史改变。近网格浮点duration的占用采用真实Timedelta(ns)语义；旧CARS历史的 `math.ceil(duration/15)` 若发生浮点边界差异应记录，不在本轮修旧策略。

新测试源码覆盖空集、largest-removal、非整拍duration、实际finish与后续release、deadline相等、集合端点、跨资源取max、intrinsic去重、零下界但不可装箱、occupied模式加强、排列不变性、浮点容量相等，以及30个随机四任务实例的穷举最优方向。diagnostic测试覆盖覆盖拒绝、路径逃逸、历史漂移、5%等号、近门槛不称已支持和真baseline archive只读核验。实际pytest通过情况须查看本轮日志，不能仅依据这些测试源码断言模型正确。
