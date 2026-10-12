# E1-5C 离线 SLA 必要条件与违约数下界

本模块只分析有限冻结任务集合，不创建调度器、不读取 CI/天气，不向 PPO 或现有在线策略提供分析结果。源码为 `experiments/e1/sla_feasibility.py`，行为测试为 `tests/e1/test_sla_feasibility.py`。旧执行器、任务、模型和历史结果均不修改。

## 首先区分两个时间

公共执行器只在 UTC 15 分钟边界启动任务。任务在开始时记录 `finish_time = start + pd.Timedelta(minutes=duration)`，上游 `release_resources()` 在随后第一个不早于 finish_time 的边界释放资源，但 SLA 按 **真实 finish_time <= 原始合成 deadline** 判断，并非按资源释放时刻判断。

记步长 Δ=15 分钟，真实时长 p_i，资源占用时长 P_i=Δ ceil(p_i/Δ)。例如 p_i=16、deadline-arrival=24 分钟，立即开始可在第 16 分钟按时完成，资源在第 30 分钟释放。因此不能以 `P_i > deadline-arrival` 把这个任务判为必然违约。

模块的 `task_timing()` 使用与 Task/Executor 相同的 Pandas Timedelta（纳秒）构造，计算：

- 原始到达 a_i 和原始 deadline d_i；
- 资源占用时长 P_i；
- 最晚合法网格启动 L_i = floor_grid(d_i-p_i)；
- 对所有按时启动都成立的最晚资源释放 D_i = L_i+P_i。

单任务必然不可按时完成的充分证据只有：某一维需求超过该维容量，或 L_i<a_i。按 task_id 取唯一集合，不重复计数。

## 默认检查：原要求的 J(u,v)

对于网格区间 [u,v)，集合为 J(u,v)={i:a_i>=u 且 d_i<=v}，原始 deadline 不变。对 r∈{CPU,GPU,MEM}，检查 W_r(u,v)=Σ_{i∈J} q_ir P_i，与 C_r(v-u) 比较。CPU 单位为 core-minute，GPU 为 GPU-minute，MEM 为与现有 mem_req 相同的 memory-unit-minute，不额外改变内存的输入口径。

必要性的原因：如果集合中任务全部按时完成，其网格启动 s_i>=u，而真实完成 f_i<=d_i<=v。因为 v 是网格边界，资源释放 ceil_grid(f_i)<=v，所以这些完整占用区间全部包含于 [u,v)。若某维 W_r>C_r(v-u)，该集合不能全部按时完成。跨区间边界的其他任务只能占用更多资源，本条件忽略它们，因而不会加强为错误的充分条件。

u 枚举不同的 arrival；v 枚举不同的 ceil_grid(raw deadline)。对于任意有非空 J 的网格区间，将 u 移至该 J 内最早的 arrival，将 v 移至该 J 内最晚的 ceil_grid(raw deadline)，集合不变而容量工作量不增（端点已经相等时不移动）。因此这些端点覆盖此类**网格区间完整任务集合工作量条件**的最强候选。这不是对任意实数区间的同一种 rounded 占用检查，也不包含跨边界的强制执行部分、非抢占装箱、三资源耦合等更强可行性条件。

## 独立强化：occupied_deadline

可选 `interval_mode='occupied_deadline'` 将集合筛选终点改成 D_i<=v，而不是 d_i<=v，单独保存结果与标签，不能把它称为原始 J 检查。证明是：按时网格启动必有 s_i<=L_i，故资源释放 s_i+P_i<=L_i+P_i=D_i。即使 D_i 晚于原始 deadline，或因网格可启动约束早于 ceil_grid(deadline)，这个资源释放上界仍成立。

例如 p=14、deadline=arrival+16 分钟，仅到达时那一拍能按时开始，所以 D=arrival+15。两个需求为满容量的这类任务，原始 J 在 v=30 的条件不超载，而独立的 occupied_deadline 条件能在 v=15 检出至少一个违约。本强化仍只是必要条件，不是实际违约概率，也没有改变任务 deadline。

## 保守最小违约数下界

固定某一个资源和某一个区间，将集合内资源工作量 w_i=q_ir P_i 按降序排为 w_(1)...。若超额为 E=W-C_r(v-u)，至少要从按时集合中移除足以覆盖 E 的工作量。删除 k 个任务最多移除前 k 大工作量，因而最小满足 Σ_{j=1}^k w_(j)>=E 的 k，是这个区间/资源的违约数必要下界。任务不会真的删除；这是对可能按时任务子集的离线数学推论。

最终仅取各单区间、单资源证书和单任务不可行唯一计数的 **最大值**。不直接相加，不把 CPU 和 GPU 的证书、重叠区间证书或 intrinsic 证书视为互不相交。没有实现任何未经证明的跨区间聚合。

数值计算采用相对 1e-10 的资源工作量容差，只弱化下界，不能成为执行器允许超容量的 epsilon。Fenwick 树用降序资源需求做计数/工作量查询；最终报告的 witness 会重新从原始任务集合，以 math.fsum 检查总量、降序前缀和删除数量。局部 witness 可保存 task_id 和最大工作量删除集合；公开小摘要默认只提供任务数与 task_id 集合哈希。删除集合是证书构造，不是唯一实际违约任务，也不等于 EDF 或 PPO 的违约集合。

只有当可靠下界/N **严格大于 5%**，才足以说明当前模型中 5% DMR 不可实现。下界不超过 5% 时，必须写“该必要条件不足以判定可行性”，不能用未检测超载宣称可行，也不能把 EDF DMR 当成理论最优下界。

## API 与计算成本

```python
from experiments.e1.sla_feasibility import analyze_sla_feasibility
raw = analyze_sla_feasibility(workload, capacity, top_witnesses=5)
strengthened = analyze_sla_feasibility(
    workload, capacity, interval_mode='occupied_deadline', top_witnesses=5)
```

返回 JSON 可序列化字典，包含冻结 workload SHA、原始任务数、区间模式、端点数、检查区间数、逐资源证书、全局保守下界、下界比例及解释限制。`include_task_ids=True` 仅用于私有任务级证书，默认 False。不可使用这个 API 的完整未来任务集合作为在线观测或动作输入。

预先排序资源工作量和 deadline，对每个 u 沿 deadline 端点单调加入任务。主复杂度约 O(3 U N log N + 3 U V log N)，内存 O(N+U+V) 及少量 top witness；避免每个区间重新排序全部任务。资源工作量证书使用降序前缀二分进行复核。48 小时 arrival 窗口最多 192 个 u，v 可延伸到窗口后的真实 deadline，不强行截在 arrival_end。

## 验证记录

先新增测试并真实运行，缺少模块时得到 `ModuleNotFoundError`，结果 `1 failed in 0.66s`；随后实现核心功能得到 `24 passed in 2.54s`。测试包括：真实 LocalExecutor 的 16 分钟完成/30 分钟资源释放、deadline 端点相等、三资源不能直接相加、输入不变和稳定 witness、个体不可行唯一计数、空任务、数值相等、原始 J 与独立强化区别，以及小规模离散非抢占枚举最优违约数。

还构造了“未检出超载却无法全部按时完成”的实例：3 个 CPU=0.6、duration=15、deadline=30 的任务，满容量 CPU=1；工作量 27<=30，但任一拍不能同放两个任务，所以最多按时完成 2 个，精确最优违约为 1，而必要条件返回 0。这直接验证了本模块不会把必要条件误写成充分条件。

补充固定 seed 510005 的 30 个四任务合成微型实例，对 raw_deadline 和 occupied_deadline 两个模式分别与独立穷举的非抢占最优违约数比较。完整最终测试记录由本轮运行日志保存，不依靠训练 Return 或旧 pytest 通过来证明模型正确。
