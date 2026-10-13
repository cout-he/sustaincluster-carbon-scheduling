# E1-6B：四动作在线接口与任务风险状态/过滤方法

本文记录已经实现并通过微型行为验证的接口与预注册协议，不报告尚未完成的正式 PPO 训练或开发 Validation 成绩。起点为研究提交 `d4be4d681bf04dfd2d2c7c9de341d2455270f89d`。E1-6A Task Full 的负面结论保持有效：开发 Validation 2× 比 RiskOnly 多341个deadline违约和33.714 kgCO₂e。以下风险过滤的效果待本轮公平实验验证。

## 1. 版本与唯一执行路径

| 组件 | 新接口/职责 |
|---|---|
| `adaptive_rl_env.py::AdaptiveCarbonSchedulingEnv` | schema=`e1-6b-adaptive-delay-env-v1`；83维 float32 Box([-1,1])，Discrete(4) |
| `adaptive_risk.py::AdaptiveDelayPolicy` | schema=`e1-6b-adaptive-delay-v1`；只定义全部候选共同使用的本拍档位与任务级defer条件 |
| `adaptive_observation.py::risk_observation` | schema=`e1-6b-continuous-risk-observation-v1`；只接收已观测 ResourceSnapshot 与 Pressure，输出43维风险块 |
| `adaptive_train.py` | schema=`e1-6b-adaptive-ppo-v1`；独立四动作PPO入口和Train双场景episode表 |
| `adaptive_evaluate.py` | schema=`e1-6b-common-evaluation-v1`；完整Drain与同组公共空闲尾段核算 |

新版 Gym 类组合旧 `CarbonSchedulingEnv`，保留其基础40维状态、奖励、违约去重、终态和CI历史辅助；每个step直接推进该旧对象的 `LocalExecutor.advance(new_policy)`，不调用旧三动作step。只有一组Task、资源池、pending/running/ledger及持久IT/HVAC模型，不存在另一套资源分配或物理内核。物理模型只在reset建立，在episode内及公共空闲尾段继续使用。

旧环境的 public schema、40维、Discrete(3)、三模式规则、旧训练入口和E1-5B checkpoint 保持原样。新类依赖旧类 `_observation`、`_reward`、`_observe_boundary`、`_info` 等辅助方法，这一组合依赖由新旧40维逐值回归保护。后续改变旧辅助接口需要先验证该依赖，不能假设继承或组合自动兼容。

## 2. 每拍时序与四动作精确定义

每15分钟决策边界：记录当拍arrival→释放完成资源→incoming入队→暂停供agent观察并选action→全部未启动任务EDF排序→can-fit容量判断→任务级policy decide→真实start/defer与即时资源更新→调度后running负载IT/HVAC积分→准备下一边界。所有旧backlog都参与EDF `(deadline,arrival,task_id)` 排序；容量放不下时继续回填后续可执行任务。任务不可抢占。

定义 `L_i=floor_15min(deadline_i−raw_duration_i)`，`solo_slack_i=(L_i−now)`，单位分钟。过滤关闭使用solo；过滤开启使用 `safe_slack_i=solo_slack_i−B_i`。

| Action | 在容量可容纳的候选上允许主动defer的条件 |
|---:|---|
| 0 | 不主动defer，EDF immediate |
| 1 | 当前CI>161.075，选用slack≥60分钟 |
| 2 | 当前CI>161.075，选用slack≥30分钟 |
| 3 | 当前CI>161.075，选用slack≥15分钟 |

CI单位gCO₂e/kWh；等于阈值execute。slack等于门槛允许defer；每次只延后一拍15分钟，下一拍重新判断。60/30/15不是一次sleep时长，也不是延期后必须仍保留的同数值slack。Action1在判定时solo=60允许延期一拍，随后可能剩45分钟。

capacity waiting由公共执行器先判断，不能算作风险过滤拒绝。过滤只将原本满足solo档位的defer机会改成execute；实际成功启动仍必须通过原资源池分配。它不能保证未知未来到达和竞争下的deadline。

## 3. 冻结v1启发式风险过滤

对资源r∈{CPU,GPU,MEM}：

```text
m_ir = q_ir / C_r
w_ir = m_ir / sum_r(m_ir)
u_r = 当前真实running需求 / C_r
Q_ir = 紧迫且仍pending的严格EDF前驱资源工作量 / (C_r × 120min)
rho_r = 已观测120min到达资源工作量 / (C_r × effective_history_minutes)
P_ir = u_r + Q_ir + max(0, rho_r−0.5)
z_i = sum_r(w_ir × P_ir)
B_i = 15min × ceil(cold + min(8,z_i))
safe_slack_i = solo_slack_i − B_i
```

全部系数沿用E1-6A：a_u=a_q=a_h=k=1，rho0=0.5，K=8，history=120min，cold=2仅在effective_history_minutes=0时成立。历史未知时历史数值项不伪造为已知rho，按原v1省略该项并使用cold保护；之后无到达的零压力是已知零。全零资源任务规定z=B=0。需求单位与原数据映射保持一致，资源工作量使用 `q×15ceil(raw_duration/15)`；solo deadline bound仍用raw duration。

紧迫前驱指本拍初始EDF顺序中候选之前、仍pending、`L_j−now≤120min` 的任务，包括overdue。真实启动立即从该集合移除并加入running；之前defer和capacity失败仍pending，保留为竞争者。因此同拍候选过滤会随先前真实start更新，不能复用拍前固定B数组。

E1-6A公式、权重、ceil规则和历史归档未改变。本轮Shield不是新理论可行性保证，也不能被描述为已经验证有效。weighted对小需求但紧缺资源的局限及量化压缩证据见 `B0-risk-diagnostics.md`；不能使用开发Validation搜索更有利系数。

## 4. 83维Observation定义

基础40维原样保留：当前/过去CI、三资源利用率、running/pending数、120分钟历史资源工作量与压力、历史有效性与粗buffer、solo slack统计、紧迫/过期任务数、pending需求/工作量，以及已知UTC时间/到达窗口标志。归一化继续使用旧工程常数，不重新拟合。

新增43维是当前全部pending的拍前EDF快照统计，包括capacity-blocked任务。它不是当前action下的预计启动集合，也不同于E1-6A的eligible/can-fit条件diagnostics分母。

| 字段组 | 维度 | 原始单位与网络变换 |
|---|---:|---|
| `risk_u_cpu/gpu/mem` | 3 | utilization，原比例[0,1] |
| `risk_rho_cpu/gpu/mem` | 3 | 无量纲历史压力，x/(1+abs(x)) |
| `risk_z_min/mean/p50/p90/max` | 5 | 连续无量纲z，x/(1+abs(x)) |
| CPU/GPU/MEM/tie/zero类别mean z与presence | 10 | 连续z同映射，presence为0/1 |
| solo/safe各min/mean/p10/max | 8 | 分钟，x/(120+abs(x))，保留负slack |
| solo/safe各≥60/30/15比例 | 6 | 全部pending中的比例[0,1] |
| pending present / history known | 2 | 0/1 |
| B mean/max | 2 | 分钟/150；B不替代连续z |
| z∈(0,1]、(1,2]、>2比例 | 3 | 全部pending中的比例[0,1] |
| 当前can-fit比例 | 1 | 加入当前running后的逐任务容量判断，未模拟队列分配 |

资源主导类别按需求/容量的最大值分类，等最大为tie，全零为zero。缺失类别均值0并presence=0；空队列任务统计0，pending_present=0，但真实running与历史字段可非零。未知历史rho数值为0并history_known=0，与已知零压力区分。风险块有界单调变换不hard clip连续z，但float32和有界映射仍有有限精度，原始字段与逐任务预算保留供审计。

观测构造器不接收workload、环境未来数组、action、最终结果或其他策略轨迹；策略只接收执行器的当前known snapshot。下一step返回的Observation可包含新边界已到达任务和新边界当前CI；前置Observation及本拍决策不能看到这些尚未到达信息。future-fork测试覆盖这一时点区分。

## 5. 四组消融的确切解释

| 组 | 新风险状态43维块 | frozen v1过滤 | 总维度/动作 |
|---|---|---|---|
| PPO-Plain | 全零 | 关 | 83 / 4 |
| PPO-State | 开 | 关 | 83 / 4 |
| PPO-Shield | 全零 | 开 | 83 / 4 |
| PPO-Combined | 开 | 开 | 83 / 4 |

四组输入层均保留全部83节点，关状态只是43块置零；同seed网络结构、参数量和初始化身份应核验。PPO-Plain仍有基础40维历史rho、利用率、RiskOnly粗buffer、solo slack和紧迫数量，不能称为完全无风险信息。State检验任务匹配连续z、类别与safe slack等新增块的边际信息，部分资源字段与公共基础重复。

State−Plain、Combined−Shield检验状态效果；Shield−Plain、Combined−State检验过滤效果；交互项可报告 `(Combined−State)−(Shield−Plain)`，须按同seed同压力场景成对计算。各组随后真实轨迹分叉，配对统计不等于对某单次过滤的严格因果归因。

旧40维三动作PPO仅是外部基线；与Plain同时改变动作空间、输入维数/参数量和Train场景分布，因此不能将其差值全部归因于新四动作。固定动作微型回归证明规则对应，但不证明训练性能收益。

## 6. 共用Reward与终态

本轮配置不改Reward：

```text
r_t = −[ carbon_kg_t/1000
         + pending_post_schedule_t×15/15000
         + 10×new_deadline_misses_t/1000
         + 10×terminal_liability_t ]
```

carbon来自本拍真实IT+HVAC电能乘当前CI/1000，不是传输能耗。wait为本拍调度后实际pending任务分钟。首次确定deadline违约只对一个任务计一次；真实完成违约或unfinished deadline≤观测边界时识别，并使用同一penalized集合。判定发生在step完成后的边界t+15min，不能把该事件误标为前置action时间t。

terminal_liability仅在环境真正truncated时启用：已到达未完成任务数/1000，加三资源剩余工作量/(容量×120min)中的最大值。pending按原duration向上取整，running按已知finish−now剩余时间向上取整。终点不加载新任务。

正式训练配置为完整Drain episode；到真实数据安全截止仍有工作时trainer保存失败并拒绝把它静默当作正常TimeLimit。固定PPO环境步预算可能结束在episode中间，这不是环境truncation；保留partial episode并按标准critic bootstrap，不能伪造完成或把尾段return当等工作量结果。

PPO gamma=0.99保持不变：约100拍后折扣0.366，192拍后约0.145，长期Drain/空闲碳与晚期违约的信用分配存在偏差风险。本轮记录这些局限，不调整权重制造开发Validation优势。训练return不替代正式公共窗口DMR、carbon、JCT。

## 7. 训练与公平评测协议

配置 `experiments/e1/configs/e1_6b_ppo.json` 登记Train1.5×（8313任务）/2×（11084任务）；reset按seed%2初始位置然后交替，episode选择比例1:1，实际环境步占比另报。PPO：[64,64]、lr0.0003、rollout256、batch64、epochs10、gamma0.99、GAE0.95、clip0.2、entropy0、vf0.5、maxgrad0.5；CPU单线程确定性。正式计划每组seed55101/55102/55103各20480步，先四组各1024步smoke55100，smoke只验功能/更新/回放/成本，不按效果选择。正式执行冻结记录与实际完成状态另见训练报告，本文不提前声称12模型训练完成。

开发Validation1×/1.5×/2×每档包含12个新PPO、3个既有三动作PPO及EDF、EDF+Carbon、RiskOnly、Task Full，共19方法/档、57条common结果。所有任务/CI/weather/容量/模型完全相同；各自Drain后取同档最慢终点，早完成策略按同一持久物理状态继续真实idle积分。任务features/身份、完成数、资源小时和公共终点均需审计。未完全排空时不宣称等工作量减排。

5% DMR保持预注册绝对目标，不提升为30%/35%；高压另报相对EDF额外违约与carbon权衡，三档不混合平均。当前Validation已经多轮开发使用，不是未见Test；Test继续拒绝。RL随机seed反映训练随机性，2× stress seed不生成独立到达形态。

## 8. 初版接口验收证据

已实际运行的新环境31例+旧环境23例：**54 passed in1.73s**，日志 `test-adaptive-env-green-002.txt`。验证固定Action0→EDF、无过滤Action3→EDF+Carbon、过滤Action3→Task Full的任务账本、真实start/defer集合、逐拍资源、电能与碳精确一致。future分叉、ceil/floor边界、阈值相等、capacity查询归因、首次违约不重复、非抢占释放、模型持续与idle、Gym checker、seed确定性均覆盖。完整TDD与首次测试字段修正证据保存在 `B0-interface-audit.md`。

另已完成两个真实Train档的6条固定动作接口回归，五类完整核心结果文件以及完整policy_intervals均与旧E1-6A归档字节相同；旧60个输入/结果文件与10个接口source hash前后不变，实际exit0。详见 `E1-6B-fixed-action-equivalence.md`。这只是接口回归，不是新的策略效果或训练结论。

本节记录最初的接口验收，完整旧E1回归、正式训练、57轨迹开发Validation与科学回放现已完成，实际效果见主报告及下方最终验收补充。测试通过只证明相应接口契约，不能证明风险过滤有效、PPO收敛到最优或学术创新成立。

## 最终验收补充

正式12模型训练、57条开发Validation、57条精确回放及资源/能碳/过滤分母独立审计已通过，详见E1-6B-report与tests。结果不支持新增状态稳定改善，过滤也未提供稳定联合优势；早期接口证据不替代这些实测结论。

另做一个不读取真实数据的单拍合成核验：duration90/120、SLA1.5给出solo45/60min，CPU/MEM各0.1且容量1，CI200。共享LocalExecutor中EDF启动两者，Carbon/RiskOnly延迟两者，Action1无过滤只启动45任务而延迟60任务；证明新接口能够表达旧三模式在这个相同状态下不能表达的决定集合。此微型表达能力不证明性能收益，后续仍需匹配网络/采样的动作空间控制组。
