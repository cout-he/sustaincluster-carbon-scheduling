"""Read sealed diagnostics and export compact scientific summaries; no simulator."""
import csv
import argparse
import hashlib
import json
import math
import os
from pathlib import Path

HERE=Path(__file__).resolve().parent
RUN=HERE/'diagnostics-001'
def read(path):return json.loads(path.read_text(encoding='utf8'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def csvrows(path):
    with path.open(encoding='utf8',newline='') as stream:return list(csv.DictReader(stream))
def writecsv(path,rows):
    with path.open('x',encoding='utf8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
def markdown(rows,fields):
    result=['| '+' | '.join(title for key,title in fields)+' |','|'+'|'.join('---' for _ in fields)+'|']
    for row in rows:
        values=[]
        for key,title in fields:
            value=row[key]
            values.append(f'{value:.3f}' if isinstance(value,float) else str(value))
        result.append('| '+' | '.join(values)+' |')
    return '\n'.join(result)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--diagnostics', type=Path, default=RUN)
    parser.add_argument('--summary', type=Path, default=HERE/'summary-001')
    parser.add_argument('--report', type=Path, default=HERE/'E1-5C-report.md')
    parser.add_argument('--evidence', type=Path, default=HERE/'summary-evidence-001.json')
    args=parser.parse_args()
    RUN=args.diagnostics
    out=args.summary
    if out.exists():raise FileExistsError(out)
    if args.report.exists() or args.evidence.exists():raise FileExistsError('Report or evidence exists')
    status=read(RUN/'run_status.json')
    assert status['state']=='complete' and not status['formal_test_evaluated']
    outputs=read(RUN/'output-hashes.json')
    assert all(sha(RUN/name)==digest for name,digest in outputs.items())
    source=read(RUN/'source-freeze.json')
    inputs=read(RUN/'input-manifest.json')
    analyses=read(RUN/'analysis-summary.json')
    comparisons=csvrows(RUN/'baseline-comparison.csv')
    groups={(c['split'],float(c['factor'])):c for c in inputs['cohorts']}
    comparisons.sort(key=lambda row:(row['split'],float(row['factor']),
        ('edf_backfill','edf_carbon_threshold','cars_risk_only').index(row['policy'])))
    pressure_rows=[];resource_rows=[];deadline_rows=[];bound_rows=[];witness_rows=[];paired=[];baselines=[]
    for row in comparisons:
        assert int(row['arrived'])==int(row['completed']) and int(row['pending'])==int(row['running'])==0
        assert math.isclose(float(row['it_energy_kwh'])+float(row['hvac_energy_kwh']),float(row['total_energy_kwh']),rel_tol=1e-12)
        baselines.append(dict(split=row['split'],factor=float(row['factor']),policy=row['policy'],
            origin=row['origin'],tasks=int(row['arrived']),misses=int(row['deadline_missed_completed']),
            dmr_pct=100*float(row['deadline_violation_rate_observed']),carbon_kg=float(row['carbon_kg']),
            it_energy_kwh=float(row['it_energy_kwh']),hvac_energy_kwh=float(row['hvac_energy_kwh']),
            total_energy_kwh=float(row['total_energy_kwh']),jct_minutes=float(row['mean_jct_minutes_completed']),
            wait_minutes=float(row['mean_wait_minutes_started']),pending_mean=float(row['pending_mean']),
            pending_peak=int(float(row['pending_max'])),policy_defer_intervals=int(float(row['policy_defer_intervals'])),
            capacity_wait_intervals=int(float(row['capacity_wait_intervals'])),drain_hours=float(row['drain_hours']),
            common_end=row['common_end'],workload_sha256=row['workload_sha256'],environment_sha256=row['environment_sha256']))
    for a in analyses:
        split,factor=a['split'],float(a['factor']);label=f'{split}-{factor:g}'
        p=a['pressure'];edf=a['edf_alignment']['summary'];reference=next(r for r in baselines
            if r['split']==split and r['factor']==factor and r['policy']=='edf_backfill')
        pressure_rows.append(dict(split=split,factor=factor,tasks=p['tasks'],
            arrival_mean=p['arrivals']['mean'],arrival_p95=p['arrivals']['p95'],arrival_peak=p['arrivals']['max'],
            arrival_cv=p['arrivals']['std_population']/p['arrivals']['mean'],
            duration_mean_min=p['duration_minutes']['mean'],duration_p50_min=p['duration_minutes']['p50'],
            duration_p95_min=p['duration_minutes']['p95'],solo_slack_mean_min=p['solo_slack_minutes']['mean'],
            grid_slack_mean_min=p['usable_grid_slack_minutes']['mean'],urgent_arrival_pct=100*p['urgent_on_arrival_fraction'],
            cpu_window_load=p['cores']['arrival_window_load'],gpu_window_load=p['gpus']['arrival_window_load'],
            mem_window_load=p['mem']['arrival_window_load'],gpu_work_total_min=p['gpus']['window_work_minutes_occupied'],
            gpu_work_peak_min=p['gpus']['work_minutes_per_arrival_interval_occupied']['max'],
            gpu_pressure120_peak=p['gpus']['pressure_history']['max'],gpu_pressure120_above_one=p['gpus']['history_intervals_pressure_above_one'],
            edf_dmr_pct=reference['dmr_pct'],edf_capacity_wait_intervals=edf['capacity_wait_intervals'],
            edf_pending_mean=edf['pending']['mean'],edf_pending_peak=edf['pending']['max'],
            edf_cpu_saturated_ticks=edf['cores_near_saturated_intervals'],edf_gpu_saturated_ticks=edf['gpus_near_saturated_intervals'],
            edf_mem_saturated_ticks=edf['mem_near_saturated_intervals'],tier=a['diagnostic_tier']))
        deadline=dict(split=split,factor=factor,tasks=p['tasks'],urgent_tasks=p['urgent_on_arrival_tasks'],
            urgent_pct=100*p['urgent_on_arrival_fraction'])
        for name in ('duration_minutes','rounded_occupancy_minutes','solo_slack_minutes',
                     'usable_grid_slack_minutes','relative_solo_slack','latest_start_minutes_from_cohort'):
            for stat in ('min','p50','p95','max','mean'):deadline[name+'_'+stat]=p[name][stat]
        deadline_rows.append(deadline)
        for r in ('cores','gpus','mem'):
            data=p[r];entry=dict(split=split,factor=factor,resource=r,
                total_work_raw_resource_min=data['window_work_minutes_raw'],
                total_work_occupied_resource_min=data['window_work_minutes_occupied'],
                arrival_window_load_raw=data['arrival_window_load_raw'],arrival_window_load=data['arrival_window_load'])
            for name in ('work_minutes_per_arrival_interval_occupied','pressure_15m','pressure_history'):
                for stat in ('p50','p90','p95','p99','max','mean'):entry[name+'_'+stat]=data[name][stat]
            entry.update(instant_pressure_above_one=data['arrival_intervals_pressure_above_one'],
                history_pressure_above_one=data['history_intervals_pressure_above_one'])
            resource_rows.append(entry)
        for mode,f in a['feasibility'].items():
            bound=f['minimum_deadline_misses_lower_bound']
            bound_rows.append(dict(split=split,factor=factor,mode=mode,tasks=f['task_count'],
                lower_bound_misses=bound,lower_bound_dmr_pct=100*f['violation_rate_lower_bound'],
                intrinsically_impossible=f['intrinsically_impossible_count'],
                intervals_checked=f['checked_interval_count'],analysis_seconds=f['analysis_seconds'],
                cpu_bound=f['resource_results']['cpu']['minimum_deadline_misses_lower_bound'],
                gpu_bound=f['resource_results']['gpu']['minimum_deadline_misses_lower_bound'],
                mem_bound=f['resource_results']['mem']['minimum_deadline_misses_lower_bound'],
                five_pct_impossible_proven=f['violation_rate_lower_bound']>.05,
                feasibility_verdict='5pct_impossible_proven' if f['violation_rate_lower_bound']>.05 else 'undetermined_by_necessary_bound'))
            for r,values in f['resource_results'].items():
                w=values['witness']
                if w:witness_rows.append(dict(split=split,factor=factor,mode=mode,**w))
        group_name=groups[(split,factor)]['group_path']
        if os.name == 'nt' and group_name.startswith('/mnt/d/'):
            group_name='D:/'+group_name[7:]
        group=Path(group_name)
        if not group.is_absolute():group=args.root/group
        misssets={};waits={}
        for policy in ('edf_backfill','edf_carbon_threshold','cars_risk_only'):
            with (group/policy/'ledger.jsonl').open(encoding='utf8') as stream:
                tasks=[json.loads(line) for line in stream]
            assert len(tasks)==p['tasks'] and len({t['task_id'] for t in tasks})==p['tasks']
            misssets[policy]={t['task_id'] for t in tasks if t['deadline_outcome'] in ('missed','missed_unfinished')}
        for policy,missed in misssets.items():
            base=misssets['edf_backfill'];paired.append(dict(split=split,factor=factor,policy=policy,
                misses=len(missed),edf_misses=len(base),misses_common_with_edf=len(base&missed),
                additional_missed_tasks_vs_edf=len(missed-base),met_tasks_that_edf_missed=len(base-missed),
                net_additional_misses=len(missed)-len(base),
                interpretation='paired trajectory association; not per-event causal attribution'))
    out.mkdir()
    for filename,rows in [('baseline-comparison.csv',baselines),('workload-pressure-summary.csv',pressure_rows),
        ('resource-pressure-summary.csv',resource_rows),('deadline-slack-summary.csv',deadline_rows),
        ('sla-lower-bound-summary.csv',bound_rows),('sla-witness-summary.csv',witness_rows),('paired-miss-comparison.csv',paired)]:
        writecsv(out/filename,rows)
    train=next(p for p in pressure_rows if p['split']=='train' and p['factor']==2)
    val=next(p for p in pressure_rows if p['split']=='validation' and p['factor']==2)
    pressure_table=markdown(pressure_rows,[('split','窗口'),('factor','倍率'),('tasks','任务数'),
        ('gpu_window_load','GPU占用工作/48h容量'),('cpu_window_load','CPU占用工作/48h容量'),
        ('mem_window_load','MEM占用工作/48h容量'),('gpu_pressure120_peak','120min GPU压力峰值'),
        ('edf_dmr_pct','EDF DMR %'),('edf_capacity_wait_intervals','EDF容量等待次数')])
    baseline_table=markdown(baselines,[('split','窗口'),('factor','倍率'),('policy','策略'),
        ('tasks','完成任务'),('misses','违约'),('dmr_pct','DMR %'),('carbon_kg','kgCO2e'),
        ('total_energy_kwh','kWh'),('jct_minutes','JCT min'),('wait_minutes','等待 min'),('drain_hours','Drain h')])
    bound_table=markdown(bound_rows,[('split','窗口'),('factor','倍率'),('mode','区间类型'),
        ('lower_bound_misses','违约下界'),('lower_bound_dmr_pct','下界 %'),('cpu_bound','CPU'),('gpu_bound','GPU'),('mem_bound','MEM')])
    for filename,body in [('baseline-table.md',baseline_table),('pressure-table.md',pressure_table),('lower-bound-table.md',bound_table)]:
        with (out/filename).open('x',encoding='utf8') as stream:stream.write(body+'\n')
    report=f'''# E1-5C：负载压力、SLA边界与研究方案诊断

## Material Passport

证据为 E1-5B 既有 Train 三档与 Validation2× 的冻结任务/全部三基线轨迹，以及本轮先冻结再执行的 Validation1×/1.5×。发布父提交958c14137bfeb30aaf6245a93a3cb693bcb60d20。新增实验是开发用探索性诊断，不能作为未见窗口泛化验证，不重新选择 E1-5B模型。Test未读取为任务/CI切片或运行策略；没有新PPO训练、Shield或旧算法修改。

## 1. 完整三基线结果

全部18个三基线运行完成各自有限任务集合，pending/running/删失均为0；相同档位任务与CI/weather/资源/物理模型完全一致，保留原历史公共终点，本轮新组取三策略最慢终点并延展真实空闲积分。固定48h到达后Drain，不能称为持续到达稳态集群。不同档位能碳不能作等工作量优势比较。

{baseline_table}

完整IT/HVAC、总电能、pending、主动/容量等待、Drain和输入身份见summary-001/baseline-comparison.csv。原始完整 ledger、环境及归档只存本地；历史结果逐文件哈希核验后只读引用。

## 2. 为什么Train2×和Validation2×压力不同

到达工作量按每任务原CPU/GPU/MEM需求乘ceil15资源持有分钟；raw duration工作另列。压力为工作量/(容量×观察分钟)，是到达负债度量，不是该拍实际利用率，更不表示所有长任务必须在到达拍完成。192个到达拍含零拍。

{pressure_table}

2×任务数比为{val['tasks']/train['tasks']:.3f}，GPU占用工作比却为{val['gpu_work_total_min']/train['gpu_work_total_min']:.3f}。Train2× GPU负债/48h容量为{train['gpu_window_load']:.6f}，Validation2×为{val['gpu_window_load']:.6f}；CPU分别{train['cpu_window_load']:.6f}/{val['cpu_window_load']:.6f}，MEM分别{train['mem_window_load']:.6f}/{val['mem_window_load']:.6f}。因此不能仅按数量或同为2×认定压力相同。Validation GPU密集度更高，CPU压力反而更低。

每15min GPU到达工作峰值为{train['gpu_work_peak_min']:.3f}/{val['gpu_work_peak_min']:.3f} GPU-min，120min压力峰值{train['gpu_pressure120_peak']:.3f}/{val['gpu_pressure120_peak']:.3f}；其>1拍数{train['gpu_pressure120_above_one']}/{val['gpu_pressure120_above_one']}。全部分位数见resource-pressure-summary.csv。任务数CV为{train['arrival_cv']:.3f}/{val['arrival_cv']:.3f}，不能无证据称所有指标的突发性都增强；资源峰值、持续高压力及EDF积压需结合看。

平均duration为{train['duration_mean_min']:.3f}/{val['duration_mean_min']:.3f}min，紧迫到达占比为{train['urgent_arrival_pct']:.3f}%/{val['urgent_arrival_pct']:.3f}%。合成solo slack/duration均约0.5，同一SLA multiplier=1.5，未发生更改deadline规则；15min网格使部分短任务实际可用余量为0。该紧迫性、GPU负债、同时到达复制、EDF回填/资源耦合共同影响结果，不能认定唯一原因。

EDF公共窗口GPU>=99%利用率拍数为{train['edf_gpu_saturated_ticks']}/{val['edf_gpu_saturated_ticks']}，CPU为{train['edf_cpu_saturated_ticks']}/{val['edf_cpu_saturated_ticks']}，MEM为0/0；pending峰值{train['edf_pending_peak']:.0f}/{val['edf_pending_peak']:.0f}，容量等待事件{train['edf_capacity_wait_intervals']}/{val['edf_capacity_wait_intervals']}。EDF无主动延迟，Validation的大量违约在纯执行参考中已经存在。事件次数是重复候选尝试，不能当作违约任务数或容量不可避免违约数。具体资源阻塞mask、UTC峰值对应见pressure-evidence.md与pressure-timing-evidence.csv。

离线120min统计固定分母并以cohort前零填充；旧RiskOnly使用有效已观察时长与cold标志，所以前120min不是同一个policy rho。本轮不修改旧估计。2×复制每个任务一次；改seed主要改变副本ID与stable排序，不改变到达工作量曲线，不构成独立真实负载样本。1.5×部分复制会随seed选集变化，但仍是同一trace的相关合成压力。

## 3. SLA必要条件与5%是否不可实现

SLA判实际finish=start+raw duration<=deadline；资源只在ceil_grid(finish)释放。对于网格区间[u,v]，J={{arrival>=u,deadline<=v}}，所有按时任务的rounded占用工作必须落在该区间内。三资源分别比较sum(q*p)与C*(v-u)。本轮枚举arrival×grid-ceiled deadline端点；非连续所有实区间、非充分可行性算法。

超载时，假设可以最有利地删去工作量最大的任务，找到至少要删的任务数k，得到保守违约数下界。最终见证用原任务math.fsum复核k/k-1，并使用只弱化的数值容差；跨资源/重叠区间只取最大值，不相加。强化模式使用每任务最迟按时网格启动L=floor_grid(D-d)及释放上界L+ceil15(d)，没有把资源释放改作deadline，也没有修改任务。

{bound_table}

2×Validation原J界172、强化界177，177/17416=1.0163%，小于5%。**5%的可行性尚不能判定**。这已证明不可能让所有任务零违约，但不足以证明至少871个任务必然违约而排除5%。也不能说只有177是容量原因或其余5124一定可避免。EDF5301不是最优下界；仅凭必要条件未检出超载，不能证明零违约可行。Train1×的零违约由实际完整参考轨迹支持，Train1.5×/2×尚无零违约证明。CPU/MEM未发现此资源工作量超载不等于不会阻塞个别候选。

方法证明与精确微型验证见feasibility-method.md；25项测试含30个固定seed四任务非抢占穷举、真实执行器fractional finish/resource release边界、必要非充分反例。离线全窗检查不进入任何在线策略/actor。

## 4. 5%适用范围与下一版评价

Train1×低压力：三参考均0%，主要测试时间柔性。Train1.5×：EDF1.840%、Carbon2.839%、Risk2.334%，存在容量竞争且三参考均支持5%，优先中压力候选。Validation1×：EDF4.628%支持5%，Carbon5.248%、Risk5.041%略超目标，已经提供风险控制必须改善的开发场景；参考支持是存在具体调度，不是未来方法保证。它也是较接近边界的场景，不能把略超四舍五入为达标。

Train2× EDF7.028%只能称接近目标且尚不支持5%；Validation1.5×/2× EDF16.169%/30.438%属高压力。现有三参考未支持5%，理论仍未判定。高压报告相对EDF额外违约、配对miss集合与公共Carbon/Pareto，不把门槛改成35%。paired-miss-comparison.csv包含相同任务的新增/救回/共同miss；净差不能直接归因为某一次Carbon事件。

下一版按每窗口×每倍率分层报告，禁止混合DMR掩盖高压退化；完整Drain/统一任务量/公共空闲能碳保持。基于已用开发Validation分层的事实必须披露，不能称这些场景未见泛化；最终Test继续封存，未来只有在参数、种子、模型选择及档位协议全部冻结后获另行授权评测。E1-5B5%准入与selected_policy=null完全不变。

## 5. 两个模块如何产生并分离新的方法内容

新预算S_i=floor_grid(D_i-d_i)-t，扣除任务需求匹配的风险缓冲B_i，分别用q_ir/C_r匹配当前利用率、优先紧迫队列工作及历史到达压力。已知运行释放可选，未知未来禁止；不用复杂Shadow作为必需模块。相较RiskOnly的全资源max统一0/1/2拍，应产生对同拍不同资源/余量任务不同预算，参数与代价必须通过E1-6A微型与确定性对照检验。

下一版有限动作候选为0立即执行、1/2/3选择60/30/15min最低延迟前余量，仍每次只延一拍。风险过滤将S改为S-B判定。文档给出GPU-only执行而CPU-only延迟的同拍例子，旧三全局模式无法产生该子集；这证明接口表达能力，不证明学习或创新效果。公共EDF/非抢占/容量优先与回填不变。

详见next-stage-design.md：相同4动作、网络容量、Reward、预算下做风险状态×任务过滤的2×2消融，同时保留旧PPO-v0/RiskOnly，并另做A的确定性部件消融。本轮只有设计，没有实现算法或再训练。候选风险权重在下一轮三方审核及Train-only预验证后另行冻结，不假称已验证最优值。

gamma=.99在15min粒度的半衰期17.242h、48h权重.1452；它不是硬视野，但远期拥堵成本被折扣。own Drain Reward不包含公共空闲尾段，因此return不等于论文Carbon；有限违约权重不等于5%硬约束。具体Reward/GAE/截断审计见reward-horizon-audit.md，本轮不调权重或折扣。

## 6. 证据、保护与科研边界

config/source/input/output清单、实际WSL命令与新增探索性运行日志均保留。原工作区282291个可访问旧regular文件及Git元数据已在开始时SHA封存，另记录855个Linux重解析/非regular路径；结束核验另存，不清理或修改这些路径。原main/dirty文件、M0、E1-0至E1-5B报告/模型/provenance必须不变。

本轮固定单DC、分数GPU、已知duration与usage、合成SLA1.5、15min释放/整拍积分、真实环境及有限cohortDrain均限制外推。没有严格安全概率估计或真实线上集群验证，三个历史RLseed也不代表独立负载。代码和诊断正确不意味着后续创新已成立。
'''
    with args.report.open('x',encoding='utf8') as stream:stream.write(report)
    hashes={p.name:sha(p) for p in out.iterdir() if p.is_file()}
    with (out/'output-hashes.json').open('x',encoding='utf8') as stream:json.dump(hashes,stream,indent=2)
    with args.evidence.open('x',encoding='utf8') as stream:
        json.dump(dict(status='verified',diagnostic_files_verified=len(outputs),source_sha256=source['canonical_source_sha256'],
            baseline_rows=len(baselines),cohorts=len(analyses),all_complete=True,formal_test_evaluated=False,
            summary_output_hashes=hashes,report_sha256=sha(args.report)),stream,indent=2)
    print(json.dumps(dict(summary_rows=len(baselines),pressure_rows=len(pressure_rows),
        lower_bound_rows=len(bound_rows),report='E1-5C-report.md')))
