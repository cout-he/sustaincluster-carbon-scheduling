# Task-PPO V1：训练前工程准入

2026-10-21。本报告在真实pilot之前提交，尚不包含学习收益。设计见[任务级MDP](task-level-MDP-design.md)，数值及完整32项场景序列见[pilot配置](pilot-config.json)。

## 输入与冻结边界

仅复用E2已审计的16个Train快照及完整manifest/workload/environment/任务身份哈希。12个Jan–Mar训练窗、4个Apr–May内部留出窗，跨组原始身份与72h环境覆盖无交叠；保留源重复观测，不代表独立泛化测试。复用既有角色/父目录/日期白名单加载器，不访问封存Development/Validation/Test或年度原始文件。

新分支从用户指定提交865f1ef建立，复用干净独立worktree，不修改旧分支。V1/V2、E0 EDF日历/DDF/CG/适配器和旧PPO/奖励/角色加载代码按配置中逐文件SHA256锁定；没有修改任何旧文件。

## 人工测试与独立复核

限定人工测试共 **177项通过**：新增任务级37项（joint PPO15、任务环境19、pilot集成3），旧SCA108项，显式V1/V2人工回归32项。没有执行全仓库discovery或真实工作负载工程试跑。

| 检查 | 证据 |
|---|---|
| 变长自回归联合概率 | 两候选四条proposal概率归一；sum条件LP、ratio1、joint clip手算；float64累加 |
| 物理时间/GAE | 一槽一reward/value/ratio；任务数变化不增加物理轨迹长度；真终止0bootstrap |
| 原proposal保留 | 全0和被拒1仍为actor样本；空候选/CG强制槽不伪造LP，保留critic |
| 当前任务与prefix | 48维手工值、未来历史项排除、冻结候选顺序、重复prepare幂等 |
| 联合DDF | 单独可行但联合冲突、拒绝→批准→拒绝、旧prefix复验及final完整重放逐条一致 |
| 真实执行 | 同槽propose不推进时钟；最终仅一次15min；全0路径逐字段等于旧EDF执行 |
| 回填与资源 | 人工CPU/GPU/MEM与回填；原0任务可能因非候选回填发生容量等待，不能错误记为主动等待 |
| 实际奖励责任 | 真实碳减同槽idle、逐ID一次miss、完整288槽；CG锁存及未完成/censored保留 |
| 真正训练与恢复 | 合成完整episode反传更新actor；完整Adam/RNG/model checkpoint连续更新等价 |

独立只读数学/执行复核未发现阻塞：teacher forcing保存原prefix而不重采样；固定DDF是参数无关的确定性变换；critic canonical query不依赖当前联合动作。entropy正则本pilot预注册为0，避免误用固定旧prefix条件熵的梯度；不添加奖励项。该变更明确排除旧7动作历史模型的严格单变量消融解释。

## 真实pilot门禁

提交设计、配置、源码和测试后才运行一次性入口。先32个initial/规则/历史留出评价（4窗×2容量×4方法），再恢复initial RNG/Adam，执行32个完整训练episode（16OM+16OH），最后8个final评价。另32个空闲背景独立核算。完整成功计划共104账本/29952物理槽，训练仅9216槽；任务内部选择不计为物理step。

同源同容量对照要求全部任务完成、无censored、CG有效、66–72h共同空闲、末温差≤0.05K、IT能量差≤1e-7kWh。失败回合先保存再停止；NaN/Inf、动作/预演/奖励不一致、无actor更新或checkpoint故障同样停止。不会换seed、删失败或改物理/Deadline/奖励。

初次训练前评价暴露重复核验批准前缀的二次调用瓶颈，已保留中断证据。修订使用等价的新增候选单次反事实及槽末原DDF全量复核；人工枚举、线性调用计数和前20真实槽继续一致性门禁约束，不删候选或改条件。任务最终动作分布/属性依赖与性能收益将在pilot后分别判断，工程测试不预言节碳优势。

```powershell
python -m unittest discover -s tests/task_ppo -t . -q
python -m unittest discover -s tests/sca_ppo -t . -q
python -m unittest tests.platform_v1.test_platform tests.platform_v1.test_saved_verification tests.platform_v2.test_platform tests.platform_v2.test_verification tests.platform_v2.test_acceptance -q
# 唯一真实pilot；已有marker后不得自动重试
python -u -m experiments.task_ppo.pilot --output research_notes/task-ppo-v1/pilot-002
```

状态：允许按预注册进入首次技术pilot。是否能够学习到有用的任务区分、OH是否产生风险反馈、是否值得扩大训练，均须以真实结果回答。

## 收尾验证补记

完整pilot后新增3项纯人工原始EDF排序审计回归，收尾限定测试为任务级40项、旧SCA108项、V1/V2人工回归32项，合计 **180项全部通过**。原177项为训练准入时的测试口径；新增审计未重新训练或执行调度评价。真实训练与评价结论见[首次pilot报告](first-pilot-report.md)。
