# E1-5C：既有 Reward 与 PPO 长期决策视野审计

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent，只读分析。
- Version: e1-5c-reward-horizon-audit-v1。
- Source: `rl_env.py`、`rl_train.py`、`rl_evaluate.py`、`configs/e1_5b_ppo.json` 与已有 E1-5B报告。
- Verification Status: ANALYZED；折扣数值用本地 Python标准库 `math` 实算。没有改变 Reward、折扣、模型或运行新 PPO。
- 数据角色：读取已有 Train/Validation报告，不读取 Test 输入或 Test性能。

## 1. 两个 gamma 与单位

必须区分 `RewardConfig.gamma=10`（违约项权重）和 `ppo.gamma=0.99`（未来回报折扣），它们不是同一参数。

对实际积分区间 `[t,t+15min)`，`rl_env.py:_reward()` 使用：

```text
r_t = -( carbon_kg_t/1000
       + pending_end_t*15min/15000 task-min
       + 10*new_deadline_misses_t/1000 tasks
       + 10*terminal_liability_t )
```

Carbon 来自本拍 IT+HVAC，`kW*0.25h=kWh`，再乘 `CI(gCO2e/kWh)/1000` 得 `kgCO2e`，不是传输指标。等待是策略/容量决策后的 pending 整拍任务分钟；任务开始于左边界则不计本拍等待。实际 deadline罚分在 `advance()` 返回的下一观测边界首次确定违约时产生，task ID集合去重；后来释放/完成不重复计罚。它不按违约分钟、资源需求或任务价值加权。因此“大 GPU 任务与小 CPU 任务一次 miss 同为 0.01成本”是当前目标定义，而非资源压力度量。

已完成任务按 `finish<=deadline` 判定；deadline后的第一个网格边界只对仍未完成或已确认晚完成的任务计罚。网格释放不会把在 deadline前实际完成的任务误报晚完成。`terminal_liability` 仅用于真实 `truncated`，包含已到达未完成任务数/1000及三资源剩余占用工作/(容量*120min)的最大值，不包括未到达任务。

## 2. Discount 0.99 在 15min 粒度下的含义

以一次当前奖励为权重 1，对 k 拍后的奖励权重是 `0.99**k`。由标准库实算：

| 时间量 | 拍数/公式 | 实际值 |
|---|---|---:|
| 120min后的权重 | `0.99**8` | 0.9227446944 |
| 24h后的权重 | `0.99**96` | 0.3810471181 |
| 48h后的权重 | `0.99**192` | 0.1451969062 |
| 55.75h后的权重 | `0.99**223` | 0.1063281837 |
| 半衰期 | `log(0.5)/log(0.99)*15min` | 68.96756394拍 =17.24189098h |
| e-folding时间 | `-1/log(0.99)*15min` | 24.87479062h |
| 几何权重总质量对应步数 | `1/(1-0.99)` | 100拍 =25h |

“100拍/25h”是常用有效尺度，不是硬截断视野：48h后仍有非零权重。当前碳/等待收益与几十小时后发生的容量拥堵或违约，可能获得较小权重；奖励统计的未折扣 Return 与 PPO 的折扣价值目标并不相同。不能直接据这些数值认定模型只会看 25h，critic 仍可学习更长期价值，但它的目标确实对远期成本折扣。

GAE 的 TD残差权重为 `(discount*gae_lambda)**k`，此处 `0.99*0.95=0.9405`，半衰期约 **2.82485393h**，几何质量尺度 **4.20168067h**。GAE残差跨度不等于实际可学习长期视野，残差已有 value bootstrap；它提示长延迟事件的信用分配更依赖价值估计与反复训练，不证明被截断为4h。`n_steps=256` 对应64h rollout，可跨 episode自动 reset，与环境48h到达窗或own Drain终点不同。

可复算代码，不加载数据、模型或仿真器：

```python
import math
g, lam, step_minutes = 0.99, 0.95, 15
for k in (8, 96, 192, 223):
    print(k, g ** k)
print('half_life_h', math.log(.5) / math.log(g) * step_minutes / 60)
print('efold_h', -1 / math.log(g) * step_minutes / 60)
print('geometric_mass_h', 1 / (1-g) * step_minutes / 60)
print('gae_half_life_h', math.log(.5) / math.log(g*lam) * step_minutes / 60)
```

## 3. Episode终点、真实截断与训练预算不可混淆

E1-5B 使用48h到达、随后完整 Drain，`stop_at_drain=True`，没有固定短episode人为停止任务。真实覆盖/安全cap达到且尚未完成时为 `truncated=True`，环境产生 backlog liability；`rl_train.py:_TrainingTrace` 正式模式随后报错并保存失败证据，拒绝将删失默默转换成普通 time-limit bootstrap。因此“有终罚”不意味着训练已经遇到且成功学会避免截断；本轮94完整episode/seed均排空，实际完整训练的terminal成本为0。

训练20,480采样步预算结束时，三seed的剩余尾段65/68/1步记录为**仍正常运行的未完成episode前缀**，没有让环境发出 `truncated`，不施加额外终罚，而按 PPO正常 critic bootstrap。这是价值估计边界，不是完整排空结果。用这些前缀的低碳或Return证明调度效果错误；E1-5B完整曲线和正式比较已明确排除这种推断。

真实 episode终止后，PPO reset新建物理对象并清除跨episode历史/罚分集合；跨拍的pending/running/物理内部状态一直保留。policy未知未来到达，但反复同一个Train有限轨迹、UTC/elapsed特征与可预测reset，仍可能形成时间记忆，不能把“代码没有未来字段”直接等同于独立负载泛化。

## 4. Reward与论文公平指标的不同目标

`rl_evaluate.py:evaluate_group()` 先运行各策略own Drain，再把较早结束策略的**同一个物理模型状态**延展到组内最慢共同终点；真实对应CI/weather、空闲IT/HVAC均计入论文指标。保存的 RL回报只到own Drain，公共空闲尾段没有 actor动作、也没有添加到own Reward。因此 `own carbon_cost*1000` 不等于公共窗口总碳。

策略会影响排空终点，所以own Reward与公共Carbon目标的边界不同。不能简单称公共尾段是所有策略的同一常数，更不能依据训练Return来代替等工作量碳排。未来训练若要改变能碳核算目标，应独立注册定义并验证，而不是在当前结果不理想时补一个权重。当前 waiting/deadline成本防止将少启动解释为纯低碳优势，但任何有限标量权重都不等于5%硬约束。

已冻结证据说明这种权衡实际存在：三个 Train采样中Carbon占总加权成本约47.64–49.31%，Wait约21.19–23.72%，Deadline约28.64–29.50%；它们均非零，不支持“某项完全淹没”判断。Validation中PPO55101 own Return −153.550627最高，却公共碳14,504.989kg，高于55102的14,431.177kg和55103的14,457.312kg；三者均未过5%准入。成本占比不能确定最优权重，也不能推断gamma是Train/Validation分布差异的唯一原因。

## 5. 后续协议建议，当前不实施

先用E1-5C必要条件与参考策略结果区分压力档位。只有参考支持的中压力场景才将绝对5%作为事先冻结的准入目标；高压力保留额外违约与Carbon–SLA权衡，不能为了达标改5%。Train/Validation/最终Test各角色、seed与校准规则继续隔离。

E1-6的主factorial对照应冻结同一Reward、discount、网络、动作接口与训练预算，把“风险状态”和“任务预算过滤”作为独立开关。若以后验证折扣/Reward定义，应作为单独、预注册敏感性研究，不与模块消融同时改动。可以记录未折扣与折扣的episode各成本、违约首次发生时刻、动作到成本滞后，以及公开完整Drain指标；不能把5%约束仅转换为较大违约权重后宣称安全。

下一阶段允许探索约束优化或明确的风险过滤，但必须报告未知未来下无法严格保证SLA、容量不可行时不能靠Shield创造资源、基于真实任务duration/资源已知的模型边界。本轮未提出已证实的安全RL、不访问Test、不重新训练或改权重。
