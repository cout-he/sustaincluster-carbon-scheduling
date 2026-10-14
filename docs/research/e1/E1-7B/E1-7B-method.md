# CGD-RL 候选算法 V1

版本 e1-7b-cgd-rl-v1。源码提交 [afe9d89](https://github.com/cout-he/sustaincluster-carbon-scheduling/commit/afe9d89b87b609c5bd69eb7ecd4050c5d1180e53)。这是可运行的候选假设，不是已证实创新或严格安全方法。单 DC 50000 CPU/1000 GPU/80000GB；15min；资源倍率5；合成SLA1.5；固定duration、不可抢占。相同场景使用同一冻结任务、CI/weather和公共终点，计入早排空后的同物理空闲尾段。Train五月窗口与开发Validation七月窗口保持原定义；Test继续封存。

CI预测 F0为当前值持续；F1为 `max(0, CI(tau-24h)+CI(t)-CI(t-24h))`，缺历史回退F0。在线缓存仅观测时间<=t，96个inclusive样本覆盖t至t+23h45，末区间终点t+24h。预热为窗口前24h的真实已知CI，按上一小时观测映射15min；原始文件SHA绑定且到start停止解析后续行。未来天气仅当前值持续，不读取未来真实天气。

目标任务占用N为实际 `finish=t+Timedelta(minutes=duration)` 后ceil到15min的释放距离，至少1拍；不是以预测替换真实finish的SLA。比较NOW的[0,N)与DEFER的[1,N+1)，共同H=N+1，静态HVAC尾段0。H>96返回unknown_horizon立即执行，不截短任务。已知running按释放时刻形成相同背景；其他pending后续启动和未知到达均不放入预测。两分支只改变目标安排，同拍真启动更新后续候选背景。这不是完整集群反事实。

IT代理为 `I(q)=b0+b_cpu*q_cpu+b_gpu*q_gpu+b_mem*q_mem`，用有/无目标差消除截距归因。M1比较完整网格占用积分，线性恒定增量下可等价望远镜消去为 `deltaI*.25*(CIhat[0]-CIhat[N])/1000` kg。所有中间CI仍属于完整两区间定义，不能只比较两个启动CI。

HVAC代理为五项 `clip(c0+c1*I/1000+c2*(I/1000)^2+c3*T/40+c4*(I/1000)*(T/40),0,Hmax)`。真实上游是当前负载/天气的代数响应，故V1固定alpha=0，不虚构热惯性。前一已完成区间功率通过只读hook更新，t决策不能读取[t,t+15)真实功率；alpha0不依赖它。两个预测分支共享初始状态/天气/CI/终点，M2比较IT+HVAC积分差。非负截断不证明响应或增量准确。

统一保护直接复用RiskOnly过去120min资源到达工作量压力、0/1/2拍和cold2：只有 `t+STEP*(1+B)<=deadline-duration` 才允许主动延迟。关闭风险buffer仍保留一拍solo边界。真实容量判断先于策略，EDF全队列排序和一次回填不变。保护不保证未知到达下的SLA，也不预约未来资源。

M0用当前CI>161.075；M1只用IT收益；M2用总收益，均共享保护/排序/容量/预测背景。aggressive要求Gain>0；moderate要求Gain>冻结Train正收益P75。P75=0.030622302469263302kgCO2e，来自实际EDF轨迹15376个保护允许候选，其中12852个M2正收益。M1也计算HVAC分解供诊断，但其决策不读取HVAC收益；本轮耗时包含共同诊断实现，不是最小化M1计算的成本消融。

PPO每拍选择0立即EDF、1 M2 moderate、2 M2 aggressive。1/2仅改变碳收益门槛，不改变保护。实际分项Reward仍来自原物理积分、真实等待、首次违约和截断负债。新方法没有Shadow Planner或Task Full必备模块；原E1-6A/6B负面结果保持不变。
