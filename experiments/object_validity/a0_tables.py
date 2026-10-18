"""Render exact A0 aggregate tables without reading any dataset payload."""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / 'docs/research/object-validity'


def read(name):
    with (ROOT/'data'/name).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def number(value, scale=1, digits=3):
    return f'{float(value)*scale:,.{digits}f}'


def main():
    lines = ['# A0 精确聚合统计表', '',
        '> 基线源码 bf8a35ef647f19b5ef10ce1b99f723370b7a0732；只读取本目录公开聚合 CSV。',
        '> 分位数采用 NumPy linear；利用率为相同15分钟槽的精确 pooled 分位数；任务指标按任务计数。',
        '> 96场景来自16原片段的压力/SLA增强，不是96个独立样本；无置信区间或显著性推断。', '']

    def table(title, source, headers, values):
        lines.extend([f'## {title}', '', f'数据：[{source}](data/{source})。', '',
                      '| '+' | '.join(headers)+' |', '| '+' | '.join(['---']*len(headers))+' |'])
        lines.extend('| '+' | '.join(map(str, row))+' |' for row in values)
        lines.append('')

    util = read('resources/utilization-pooled.csv')
    table('1. 到达24小时实际资源利用率（EDF，SLA1.5）', 'resources/utilization-pooled.csv',
        ['压力', '资源', 'Mean %', 'P50 %', 'P90 %', 'P95 %', 'P99 %', 'Peak %'],
        [[r['resource_pressure_factor'], r['resource'].upper()]+[
            number(r['utilization_'+key],100) for key in ('mean','p50','p90','p95','p99','max')]
         for r in util if r['policy']=='edf' and r['sla_multiplier']=='1.5'])
    groups = read('resources/pressure-policy-summary.csv')
    tasks = read('resources/task-distribution-pooled.csv')
    table('2. 容量等待、完成和 Deadline（每行52,339任务）', 'resources/pressure-policy-summary.csv',
        ['规则', '压力', 'SLA', '完成数', '容量等待数', '容量等待 %', 'DM数', 'DMR %', 'Mean JCT min', '队列峰值'],
        [[r['policy'],r['resource_pressure_factor'],r['sla_multiplier'],r['completed'],r['capacity_waited_tasks'],
          number(r['capacity_waited_tasks_fraction'],100),r['deadline_misses'],
          number(r['deadline_misses_fraction'],100),number(r['mean_jct_minutes']),number(r['queue_length_peak_any_scenario'],1,0)] for r in groups])
    table('3. 全任务容量等待分布（包含零等待；主动推迟均为0）', 'resources/task-distribution-pooled.csv',
        ['规则','压力','SLA','Mean min','P50','P90','P95','P99','Max','等待者均值 min','等待超过初始Slack任务数'],
        [[r['policy'],r['resource_pressure_factor'],r['sla_multiplier']]+[
          number(r['capacity_wait_minutes_'+key]) for key in ('mean','p50','p90','p95','p99','max')]+[
          number(r['capacity_wait_positive_mean_minutes']),r['wait_exceeds_initial_slack_tasks']] for r in tasks])
    base = next(r for r in tasks if r['policy']=='edf' and r['resource_pressure_factor']=='1.0' and r['sla_multiplier']=='1.5')
    caps = {'cpu':50000,'gpu':1000,'mem':80000}
    table('4. 单任务资源需求（基础压力；MEM为源码逻辑单位，尚未证实GB）','resources/task-distribution-pooled.csv',
        ['资源','Mean','P50','P90','P95','P99','Max','最大任务/容量 %'],
        [[resource.upper()]+[number(base[resource+'_demand_'+key]) for key in ('mean','p50','p90','p95','p99','max')]+[
            number(float(base[resource+'_demand_max'])/cap,100)] for resource,cap in caps.items()])
    table('5. Duration 与初始 Slack（分钟）','resources/task-distribution-pooled.csv',
        ['变量','Mean','P50','P90','P95','P99','Max'],
        [[label]+[number(row[prefix+'_'+key]) for key in ('mean','p50','p90','p95','p99','max')]
         for label,row,prefix in [('Duration',base,'duration_minutes')]+[
             ('Slack SLA'+r['sla_multiplier'],r,'initial_slack_minutes') for r in tasks if r['policy']=='edf' and r['resource_pressure_factor']=='1.0']])
    table('6. 合法等待15分钟槽数的任务比例','resources/task-distribution-pooled.csv',
        ['SLA','0槽 %','1槽 %','2槽 %','3槽 %','≥4槽 %'],
        [[r['sla_multiplier']]+[number(r['legal_wait_slots_'+key+'_fraction'],100) for key in ('0','1','2','3','ge4')]
         for r in tasks if r['policy']=='edf' and r['resource_pressure_factor']=='1.0'])
    table('7. 瓶颈尝试与到达工作量/容量时间（EDF，SLA1.5）','resources/pressure-policy-summary.csv',
        ['压力','资源','阻塞尝试','被阻塞独立任务','独占瓶颈尝试','需求×原Duration/容量时间 %','需求×ceil运行槽/容量时间 %'],
        [[r['resource_pressure_factor'],res.upper(),r[res+'_bottleneck_attempts'],r[res+'_bottleneck_unique_tasks'],
          r[res+'_exclusive_bottleneck_attempts'],number(r[res+'_arrival_work_to_capacity_time_scenario_mean'],100),
          number(r[res+'_rounded_arrival_work_to_capacity_time_scenario_mean'],100)]
         for r in groups if r['policy']=='edf' and r['sla_multiplier']=='1.5' for res in caps])
    scenarios = read('resources/scenario-summary.csv')
    # Maximum temporal duration across scenes; not a synthetic joined time series.
    qcols = ['queue_nonempty_minutes_arrival','queue_nonempty_longest_minutes_arrival',
             'queue_high_longest_minutes_arrival','queue_at_peak_longest_minutes_arrival']
    available = [c for c in qcols if c in scenarios[0]]
    if not available:
        available = [c for c in scenarios[0] if c.startswith('queue_') and 'longest' in c and 'arrival' in c]
    table('8. 队列持续时间：各组中最严重单场景（分钟）','resources/scenario-summary.csv',
        ['规则','压力','SLA']+available,
        [[g['policy'],g['resource_pressure_factor'],g['sla_multiplier']]+[
            number(max(float(r[c]) for r in scenarios if all(r[k]==g[k] for k in ('policy','resource_pressure_factor','sla_multiplier'))))
            for c in available] for g in groups])
    demand = read('resources/unconstrained-demand-pooled.csv')
    table('9. 无容量立即并发需求（非可执行调度）','resources/unconstrained-demand-pooled.csv',
        ['压力','资源','Mean','P50','P90','P95','P99','Peak','超当前容量槽数/1536'],
        [[r['resource_pressure_factor'],r['resource'].upper()]+[
            number(r['demand_'+key]) for key in ('mean','p50','p90','p95','p99','max')]+[r['over_current_capacity_slots']] for r in demand])
    opportunity = read('carbon/opportunity_groups.csv')
    table('10. 未来CI理想机会：无资源约束、非物理碳减排','carbon/opportunity_groups.csv',
        ['SLA','任务数','可等待数','有改善数','占全部 %','占可等待 %','CPU加权降低 %','GPU加权降低 %','MEM加权降低 %'],
        [[r['group'],r['tasks'],r['waitable_tasks'],r['positive_opportunity_tasks'],
          number(r['positive_fraction_of_all'],100),number(r['positive_fraction_of_waitable'],100)]+[
          number(r[res+'_resource_weighted_ci_integral_reduction_percent']) for res in caps]
         for r in opportunity if ';' not in r['group']])
    pairs = read('carbon/rule_pairs.csv')
    table('11. 匹配FIFO阈值规则结果（现有功率模型条件下）','carbon/rule_pairs.csv',
        ['场景ID','总碳降低 %','相对同温度空闲基准增量碳降低 %','Immediate DMR %','Threshold DMR %','JCT增加 min'],
        [[r['scenario_id'],number(r['total_carbon_reduction_percent'],1,5),
          number(r['incremental_carbon_reduction_percent']),number(r['dmr_immediate'],100),
          number(r['dmr_threshold'],100),number(r['jct_change_minutes'])]
         for r in pairs if r['baseline_policy']=='immediate_fifo'])
    probe = read('physics/physical-probes.csv')
    fields = ['it_kw','crac_fan_kw','ct_fan_kw','compressor_kw','cw_pump_kw','ct_pump_kw','system_kw']
    table('12. 手工25°C室外温度的电功率分项（kW）','physics/physical-probes.csv',
        ['CPU=GPU=MEM负载率','设定°C']+fields,
        [[r['cpu_utilization'],r['setpoint_c']]+[number(r[key]) for key in fields]
         for r in probe if r['family']=='factorial' and float(r['ambient_dry_bulb_c'])==25 and
         float(r['setpoint_c']) in (18,22) and float(r['cpu_utilization'])==float(r['gpu_utilization'])==float(r['mem_utilization'])])
    lines += ['## 统计说明', '',
        '- 所有任务均保留；容量等待、主动推迟事件分别统计。上表即时规则主动推迟为零。',
        '- 容量等待次数是任务×决策槽事件，不能当作独立任务数；多资源可同时阻塞，逐资源次数不能直接相加解释任务总数。',
        '- 高队列阈值为单场景 `max(1,ceil(P95调度后队列长度))`，连续时长和峰值连续时长分别记录；完整原字段保留于逐场景CSV。',
        '- 到达窗口为每片段24小时；排空窗口与完整31天冻结环境窗口另列CSV，不能混为利用率分母。',
        '- 第9表将任务按到达立即运行且忽略容量，只估计所需并发；并非实际资源利用率。',
        '- 第10表采用完整ceil(Duration/15)槽的CI积分；不含资源竞争、IT/HVAC功率或固定背景。',
        '- 第11表相同任务、窗口、固定22°C，阈值161.075 g/kWh继承旧常量；不据此证明真实设备节碳。',
        '- 第12表是现有代数方程输出，不是实测设备；其异常正是审计对象。', '']
    (ROOT/'A0-statistical-evidence.md').write_bytes('\n'.join(lines).encode('utf8'))


if __name__ == '__main__':
    main()
