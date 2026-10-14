# E1-7A0 预注册（求解/干预结果查看前）

## Material Passport

用户已给定完整A/B/C任务、阶段顺序与120秒求解预算。本文落实确定性选样和分析边界，不训练模型或开发预约在线算法。配置：experiments/e1/configs/e1_7a.json；唯一Train来源：E1-5B/precheck-001/load-{1,1.5,2}/input，TaskSpec集合继承E1-5C所引用的同一冻结集合，数量5542/8313/11084。stress seed510001，只读既有manifest，原需求倍率5、SLA1.5、资源容量50000/1000/80000、15min及April固定CI阈值161.075不变。Test和Validation不作为输入。

## 选样

源为同一Train集合已有EDF参考，先核对完整scientific文件SHA。全部选择只用当时可知字段，不读取后续违约、Carbon、干预结果进行排序。窗口只含固定48h到达拍；UTC明确。保存全部合格快照/候选特征的本地数据与哈希，并封存E1-7A-selection-manifest.json。

A每档各选四类互异时刻：pre-first-candidate CPU>=.8按CPU降序；GPU>=.8按GPU降序；old pending>=10按old pending降序；max利用率<=.5且old pending=0按max利用率升序。均需decision pending至少10，最多取40，按deadline/arrival/stableID抽取。各类tie最早UTC，类别顺序CPU/GPU/queue/control；已经选过的时刻排除。不存在则记空缺，不换类型制造拥堵。每快照分别构造A1、A2。

A1用原到达/任务特征、原容量、独立封闭子集，无背景；可能因此消除真实拥堵。A2候选not-before=快照，其他所有任务保留完整EDF固定启动/释放时间表，候选和背景ID不重叠。A2条件EDF保护固定背景未来承诺，名称与原EDF区分。它与CP-SAT在同固定背景/同任务/同release horizon上比较，是离线局部条件问题而非原集群全局最优。

C每档1.5/2各选最多六个互异taskID：GPU利用率>=.8降序；CPU>=.8降序；known历史max-rho>=1降序；已知rho<=.5升序；solo grid slack15–30升序；solo grid slack>=60降序。tie为UTC、stableID、原EDF rank。条件是reference真实can-fit/started、已到达、未开始、本拍solo>=15；每个目标occurrence1。总目标12pairs，缺类保留空缺。CI只保存而不按未来下降/收益选样。所有干预后严格回到同EDF后续控制器。

## 求解预算/数值

固定OR-Tools9.14.6206，solver seed710001，worker1，每案例120秒，不追加搜索。minimize misses_only，不用任意加权目标，不作secondobjective。OPTIMAL/FEASIBLE/INFEASIBLE/UNKNOWN分别保留目标、bound、实耗时；所有任务含迟到任务必须执行。

时间按原pd.Timedelta纳秒映射：真实finish=start+duration判断SLA，资源release=finish.ceil15，半开占用[start,release)。先尝试各维binary-float比例/gcd整数映射，只在int64安全时使用；否则scale1e6上取整需求/下取整容量。记录背景与候选的累积保守误差。数学binary精确映射也不等价原math.fsum全部舍入边界，因此求解器OPTIMAL仅对应记录模型，不称原全操作性浮点问题或完整集群理论最优。每一解以原E1资源语义再验并公共执行器回放。

## 碳诊断

先9条原Train基线EDF/EDF+Carbon/RiskOnly完整真实物理回放和历史数值回归。逐拍分IT/HVAC kWh、kgCO2e，active/zero-load及预先固定CI区间，算术/能源加权CI分开。小时重复、15min差、连续高低runlength、任务占用/solo时长按原数据统计。

IT局部代理在Train1 EDF全部公共窗口拍OLS校准，披露rank/RMSE；无合法预测时仅用未来真实Train CI做offline_clairvoyant_diagnostic，比较整个ceil-grid执行区间CI积分。代理不含HVAC/其他任务/背景截距，不能作为最终总碳或严格下界。实际可实现变化必须引用A合法排程和C完整分支原物理重放，所有同集合比较含公共idle。

## 反事实和解释

每pair从Train起点双重重放。目标前当前候选、ledger、队列、running、available、历史记录、已有事件/逐拍、currentCI、持久IT/HVAC对象树及累计能碳指纹必须完全一致，未知对象类型拒绝。目标仅改变一次can-fit动作；defer_once不保证下一拍启动。前缀如不一致或干预未出现作废保留错误，不能删除不利轨迹。

保存同状态RiskOnly rho/0-1-2buffer、v1连续z/B、需求权重/利用率/历史/紧迫前驱工作量。额外预注册known_release_screen只用strict EDF仍pending前驱，最新网格start在[now,now+120]：在各前驱最新启动边界，对已知running尚未release、candidate从now或now+15占用时与该前驱单独需求的三维fsum容量检查，分别count now/defer冲突。现有EDF没有实际预约，existing_reservations=0、reservation_conflict_assessed=false；该screen不分配其他pending、不预测未知到达，不是充分可行/概率保证，也不反馈策略。

两分支完整Drain，公共终点至少历史May3 14:00 UTC且覆盖两者较慢释放排空；同一持久物理实例补idle，真实对应CI/weather，不超覆盖。保存candidate/known-other/post-target-arrival影响、Δmisses=defer−execute、Δcarbon=defer−execute、资源小时守恒。

小样本只作条件干预和描述性关联，零变化/反例均保存，不拟合风险模型、不显著性检验、不将z称概率或将单特征称唯一原因。最终联合A/B/C，给一个有证据的下一阶段优先机制；若不足明确未知。

## 保护/发布

保护快照为旧224528文件metadata及当前源码/config/test/M0的158文件contentSHA，原Git关键元数据SHA及对象metadata；不读取封存Test或年度raw payload内容。legacy WSL pytest scratch reparse links不跟随。两次首次capture的scratch边界失败日志保留；capture-003封存既有文件边界，并行新增模块按显式白名单排除，原文件从未交给子代理修改。旧dirty工作树不得清理。全部新输出E1-7A，发布仅明确小文件白名单，保留完整大档本地；原数据、完整资料、原PDF及weights不上行。
