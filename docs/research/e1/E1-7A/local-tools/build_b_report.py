"""Derive a concise B report from accepted Train-only receipts, not new runs."""
import csv
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
def read(name):
    with (ROOT/name).open(encoding='utf8',newline='') as stream:
        return list(csv.DictReader(stream))
energy=read('B-analysis-001/B-energy-decomposition.csv')
opportunities=read('B-analysis-001/B-carbon-opportunity.csv')
baselines=read('baseline-replay-001/baseline-comparison.csv')
proxy=json.loads((ROOT/'B-analysis-001/receipt.json').read_text())['proxy']
zero=json.loads((ROOT/'B-zero-workload-001/receipt.json').read_text())['physical_summary']['totals']
ci=json.loads((ROOT/'B-analysis-001/load-1/ci-statistics.json').read_text())
table=['|负载|策略|完成|违约|DMR %|IT kWh|HVAC kWh|总 kWh|总碳 kg|Drain h|',
       '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for row,metrics in zip(energy,baselines):
    assert (row['factor'],row['policy'])==(metrics['factor'],metrics['policy'])
    table.append('|'+ '|'.join([f"{float(row['factor']):g}×",row['policy'],metrics['completed'],metrics['misses'],
        f"{float(metrics['dmr'])*100:.6f}",*[f"{float(row[k]):.6f}" for k in
        ['it_energy_kwh','hvac_energy_kwh','total_energy_kwh','carbon_kg']],f"{float(metrics['drain_hours']):g}"])+ '|')
op=['|负载|合法一拍候选|当前高CI|其中局部IT代理不获益|比例 %|相邻启动CI相同 %|',
    '|---|---:|---:|---:|---:|---:|']
for row in opportunities:
    op.append('|'+ '|'.join([f"{float(row['factor']):g}×",row['eligible'],row['high_current_ci'],
        row['high_ci_without_gain'],f"{int(row['high_ci_without_gain'])/int(row['high_current_ci'])*100:.3f}",
        f"{float(row['same_start_ci_fraction'])*100:.3f}"])+ '|')
report='''# B：真实能碳价值及完整执行区间机会

本报告使用已登记 Train 三档任务集合及原 EDF、EDF+Carbon、RiskOnly 的九次完整物理回放。固定到达窗为2023-05-01 05:00至05-03 05:00 UTC；公共核算截至05-03 14:00，共57小时（228拍），包括完整Drain及提前排空后的真实空闲功耗。三档分别5542/8313/11084任务全部完成，无删失。不是稳态持续到达研究，不涉及Validation/Test。

## 实际完整物理结果

'''+ '\n'.join(table)+'''

全部72个历史科学输出文件byte SHA相同；完成任务集合及三资源占用小时一致。总电能=IT+HVAC，逐拍碳=总kWh×真实gCO₂e/kWh÷1000。单DC传输能耗为零且从未代替总能耗。

相对EDF，Carbon三档公共窗口减碳分别为0.746%、1.259%、1.089%，对应新增违约0、83、59；RiskOnly为0.717%、1.112%、1.084%，对应新增违约0、41、74。这些是同工作量下的实际模型结果；高压减碳不能被称为SLA无代价优势。EDF 2× DMR约7.03%仍不能证明5%理论不可实现。

HVAC占EDF总电能的49.85%、54.77%、58.63%，是主要变化来源之一。由原模型积分得到，不能把HVAC视为固定底噪或去掉它解释任务碳。

## CI结构、duration与slack

48个完整小时均有四个完全相同CI样本，此冻结窗呈现小时块状CI，每小时四拍相同；这项统计不单独证明原数据采样或插值机制。到达窗平均CI为137.474167；高CI拍比例31.25%，高CI连续段720和180min。原数据真实低/高CI与天气持续用于Drain；公共57h算术CI为143.937719，能耗加权CI因策略负载时间不同而不同。

原任务平均duration103.095877min、平均占用111.284735min；1×中位duration74.866667min、90分位226.081667min、最大392.9min。SLA1.5对应原始solo slack中位37.433333min，而可用网格slack中位30min，25/75分位15/60min（详见B-duration-slack.csv；合成复制可能小幅改变1.5×分位数）。整段执行跨越多个小时，不能只比较两次启动时刻CI。

'''+ '\n'.join(op)+'''

候选分析覆盖EDF完整到达+Drain轨迹，与48h CI结构统计及A/C选样范围不同；2×有两个05-03 05:00的Drain候选，因此本表8076而48h选样合格候选8074。上表是离线已知真实未来CI的**局部IT机会代理**，不是在线预测或实际集群减排：取原EDF真实can-fit、solo grid slack≥15的候选，比较t启动与t+15启动的完整向上取整占用区间CI积分；不排放未来任务到在线策略，不改任务duration/需求。

高CI候选约45%–51%即使在这个clairvoyant局部代理下也没有正收益。相邻启动CI相同约74%–77%，但执行区间尾部CI可能不同，因此不能称这些候选全部没有机会。该代理不含下游容量等待、其他任务移位或HVAC，不能相加形成可实现全局碳收益、误称carbon oracle下界。

## IT代理的校准及模型边界

仅用Train1 EDF的228拍拟合IT kW截距+CPU/GPU/MEM，四列满秩；RMSE3.163229kW、R²0.997772。截距481.742195kW，CPU0.00605760kW/core，GPU0.20170297kW/GPU，MEM0.000304066kW/GB。局部任务机会只用需求相关部分，不把固定截距分配给各任务。没有拟合HVAC、Validation/Test、deadline或风险参数；本窗未产生负IT预测。

这仅是当前范围内的回归代理，无法证明逐任务真实边际功耗，CPU/GPU/MEM聚合与rack映射也可能影响泛化。真正科学碳结论全部使用持续上游IT/HVAC对象的完整重放。

## 零任务物理参考

运行前另行登记B-zero-workload-protocol.md，使用同一Train真实57h环境、容量和reset，仅建立独立空任务元组，不编辑原工作负载。CPU/GPU/MEM占用逐拍均为0。实际结果：IT27214.828600kWh、HVAC17288.548829kWh、总44503.377429kWh、碳6408.501391kg；IT空载功率约477.453133kW。

零任务碳相当于EDF总碳的约66.74%/55.55%/47.28%，说明这个固定大规模模型有较大持续空载开销。**这是参考，不是严格碳下界或可加任务边际分解**：HVAC非线性，内部Python对象持续复用，但当前step主要按给定负载/天气即时代数计算，不能据此声称已验证热惯性或动态热校准。没有修改服务器数量、容量、功率公式、休眠或关机语义来扩大收益。

## 合法调度与完整回放的交叉证据

A2真实40候选局部deadline改善11→7的重放，总碳反而+0.908028kg；它的目标为min misses，未优化碳。C的12对同状态单次干预，4减碳、6增碳、2精确0，任务违约集合均未改变。十个非零案例IT碳均降，六个被HVAC碳增加反转。最大单次减碳0.255147kg，最大增碳0.132059kg；这些不是全窗最优机会或可简单累加的策略收益。

因而当前CI阈值、IT代理符号和最终系统收益是不同量。值得优先验证完整执行区间及总模型收益估计；是否能在合法历史预测信息和独立窗口上获益仍未知。本轮未来CI仅是明确标记的离线诊断，尚无可直接交给RL的合法forecast。

数据依据：B-energy-decomposition.csv、B-ci-bin-energy-carbon.csv、B-active-idle-energy-carbon.csv、B-carbon-opportunity.csv、B-duration-slack.csv、C-carbon-components.csv。对应原始intervals、full opportunities、环境与输入身份、源码和output SHA、失败与成功日志仅保留本地。
'''
(ROOT/'B-carbon-opportunity-report.md').write_text(report,encoding='utf8')
(ROOT/'B-power-model-validation.md').write_text('''# B物理模型验证与核算边界

旧power.py、upstreamIT/HVAC方程、reset seed、make_dc_env(month=1)及逐拍执行器完全不改。每次实际回放都新建一次物理模型，随后整个57h持续复用其状态。此处复用历史设置，仅说明既有研究物理实现，未验证真实硬件校准或增加关机模型。

基线9×8历史文件byteexact；A全部44真实物理分支及固定背景守恒；C全部24分支相同任务、同资源小时、完整Drain+真实空闲尾段。独立逐拍验证kW×h=kWh、IT+HVAC=总电能、kWh×g/kWh÷1000=kg，并核对零传输语义。纳秒ceil只影响占用释放，rawfinish决定deadline；不把累计reward当能碳。

额外空任务参考保持真实CI/weather连续，不使用固定末值或重复数据。该参考不是等工作量策略，HVAC不能按常数或可加差额处理。实际碳改善与局部IT代理之间的差异见B报告和C-carbon-components.csv。没有通过测试宣称物理模型准确反映真实机房。
''',encoding='utf8')
print(json.dumps(dict(baselines=len(energy),zero_totals=zero,proxy=proxy),ensure_ascii=False))
