# E1-6A 交付与审核入口

本轮实现了独立的任务级资源匹配风险预算及四项预注册消融。所有策略仍使用唯一 LocalExecutor、原 IT/HVAC、EDF 排序及回填；旧研究源、PPO环境、main、M0和历史归档保持原状。No History仅a_h=0，保留共同cold2；尚未完全去掉历史冷启动保护。

**实际结果：Full没有在任一六档中同时改善RiskOnly的Carbon和DMR，不能宣称稳定优势。** Train2×少51次违约但多17.588kg；Validation2×多341次违约、多33.714kg。Required Max只在Validation1.5×对RiskOnly出现双目标改善，属于单个开发结果，不据此改选v1或重调系数。

| 场景 | N（八策略均完成） | RiskOnly违约 | Full违约 | RiskOnly kgCO₂e | Full kgCO₂e |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train1× | 5542 | 0 | 0 | 9533.353 | 9539.280 |
| Train1.5× | 8313 | 194 | 193 | 11408.937 | 11410.945 |
| Train2× | 11084 | 853 | 802 | 13408.320 | 13425.909 |
| Validation1× | 8708 | 439 | 440 | 9686.948 | 9701.993 |
| Validation1.5× | 13062 | 2575 | 2611 | 12076.348 | 12069.342 |
| Validation2× | 17416 | 5924 | 6265 | 14396.650 | 14430.364 |

每组同任务、同真实环境、完整Drain、同公共观察终点，三资源小时一致。Train终点2023-05-03 14:00 UTC，Validation终点2023-07-10 12:45 UTC。全部48运行未完成/运行/等待/删失终态均0；违约按实际finish与合成deadline判断。IT/HVAC单位kWh，逐拍kWh×gCO₂e/kWh÷1000为kgCO₂e，未使用传输电能代替总电能。

微型测试证明GPU/CPU压力下可产生旧统一缓冲没有的任务子集决定，但Full首次eligible预算15min占比为100%、92.43%、84.05%、98.93%、96.66%、74.15%。量化后很多任务仍同一缓冲档位，不能把表达能力当作性能优势。历史项/利用率项没有稳定独立收益，urgent项多数档影响小；高压删项方向混合。加权平均可能稀释小需求却紧缺的资源。仍是启发式，未知未来没有SLA保证，新颖性未验证。

报告入口：E1-6A-code-audit、mathematical-method、policy-design、ablation-report、tests、reproduce、figure-QA及reproducibility-checklist。完整48行指标包括能耗、DMR、等待/JCT、pending、两类等待、三资源利用率/小时、Drain和运行开销；本地在analysis-001，发布小表在tables/。代表任务按预注册first stable_id选择成功/失败/同结局启动变化，关联不等于单事件因果。

图表在本地corrected-figures-001（5×PNG/PDF/SVG及源CSV）。QA发现原analysis图1的sharey纵轴裁切高压点，原图保留并标为不合格；独立render_corrected_dmr.py在新目录修正到0–43.2143%，其余四图字节相同。3项显示边界测试通过；正式数学图注见E1-6A-figure-captions.md。GitHub不上传完整图片/账本/冻结任务，只上传renderer和小表，可由tables/summary.csv重绘DMR。

原E1 suite实际440 passed/989.24s；补充上游model forward实际1 passed/1.99s；独立显示测试3项通过。真实Train1八策略科学输出回放哈希完全一致；六组旧3基线精确回归通过。标准库独立48账本核验通过，225228旧文件保护及原Git元数据/状态一致。详细日志本地保留，发布复核结果及远端SHA另由最终回执确认，不提前声称push成功。旧suite的toy PPO短更新是接口单元测试，未训练正式科研PPO或改旧checkpoint。

复现入口：`python -m experiments.e1.task_risk_experiment --config experiments/e1/configs/e1_6a.json --split train --output NEW_DIR`，随后同输出的train-functional-gate控制Validation。分析使用task_risk_analysis，新目录拒绝覆盖。命令与依赖完整见E1-6A-reproduce.md。

发布副本不是含真实数据的自包含包：真实六组冻结输入、原环境数据和旧参考账本仍为外部本地依赖，config只记录身份/路径。新研究源码/配置/测试逐字节同步；清理发布基线旧上游93文件仅换行差异，Sphinx conf.py已排除，因此原封存manifest的严格回放必须使用原source快照，不能重写provenance来伪装相同。保留上游LICENSE与来源，只有personal研究分支提交推送，不更新main或原科研Git。

保留v1及所有负例，建议先审核预算的量化分辨率、需求匹配与瓶颈边界，再决定是否进入E1-6B。Test未访问，无正式PPO训练、无Safety Shield、无新阈值搜索。完成本轮发布后停止等待审核。
