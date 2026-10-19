"""Publish A1 aggregate tables only after the complete Train rule run.

Consumes aggregate CSV/metadata, never task/environment payload or physics.
Resource-CI exposure is a scheduling proxy and is never converted to emissions.
"""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path


INPUT_FILES = ('scenario-summary.csv', 'candidate-summary.csv',
               'utilization-pooled.csv', 'task-types.csv', 'metadata.json')
RESOURCES = ('cpu', 'gpu', 'mem')
CANDIDATES = ('L', 'M', 'H', 'OL', 'OM', 'OH')
OBSERVED = ('OL', 'OM', 'OH')
POLICY_LABELS = {'fifo': 'FIFO', 'edf': 'EDF', 'carbon_threshold_fifo': 'CT/FIFO'}
PRESSURES = (.75, 1., 1.25)
SLAS = (1.5, 2.)


def number(row, key):
    value = row[key]
    return None if value in ('', None) else float(value)


def integer(row, key):
    value = number(row, key)
    if value is None or not value.is_integer():
        raise ValueError('Expected integer count: ' + key)
    return int(value)


def group_key(row):
    return (row['candidate_id'], number(row, 'resource_pressure_factor'),
            number(row, 'sla_multiplier'), row['policy'])


def read_csv(path):
    with Path(path).open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(path, rows):
    if not rows:
        raise ValueError('Refuse an empty aggregate publication: ' + str(path))
    with Path(path).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def reduction_percent(baseline, comparison):
    """Signed change: negative means higher CI exposure; do not clamp."""
    if baseline == 0:
        return None
    return 100 * (baseline - comparison) / baseline


def validate_complete(metadata, scenarios, groups, utilization, types):
    """Reject partial or inconsistent aggregate inventories before writing."""
    if metadata.get('complete') is not True:
        raise ValueError('Refuse partial A1 results: metadata.complete must be true')
    if metadata.get('allowed_roles') != ['train'] or metadata.get('physics_model_called') is not False:
        raise ValueError('Expected Train-only, physics-disabled result metadata')
    if metadata.get('trained_models') is not False:
        raise ValueError('Expected no model training')
    if metadata.get('completed_runs') != metadata.get('planned_runs') or len(scenarios) != metadata['planned_runs']:
        raise ValueError('Rule result row count differs from completed inventory')
    if set(g['candidate_id'] for g in groups) != set(CANDIDATES):
        raise ValueError('Expected six explicitly declared candidates')
    expected = {(c, p, s, policy) for c in CANDIDATES for p in PRESSURES for s in SLAS
                for policy in (('fifo', 'edf', 'carbon_threshold_fifo') if c in OBSERVED else ('fifo', 'edf'))}
    grouped = defaultdict(list)
    for row in scenarios:
        if row['role'] != 'train':
            raise ValueError('Non-Train aggregate row')
        if integer(row, 'completed') != integer(row, 'arrived') or integer(row, 'unfinished') != 0:
            raise ValueError('Incomplete cohort cannot enter matched exposure comparison')
        if any('energy' in key or 'carbon_kg' in key or 'power_kw' in key for key in row):
            raise ValueError('Unexpected physical output fields')
        if not math.isclose(number(row, 'capacity_wait_minutes_total') + number(row, 'policy_defer_minutes_total'),
                            number(row, 'observed_wait_minutes_total'), abs_tol=1e-7):
            raise ValueError('Waiting attribution does not conserve observed wait')
        grouped[group_key(row)].append(row)
    group_map = {group_key(row): row for row in groups}
    if len(group_map) != len(groups) or set(group_map) != expected or set(grouped) != expected:
        raise ValueError('Missing or duplicate candidate/pressure/SLA/policy group')
    for key, rows in grouped.items():
        summary = group_map[key]
        if len(rows) != 16 or len({r['source_fragment_id'] for r in rows}) != 16:
            raise ValueError('Expected 16 distinct source fragments per group')
        if sum(integer(r, 'arrived') for r in rows) != integer(summary, 'arrived'):
            raise ValueError('Group denominator differs from constituent fragments')
        if sum(integer(r, 'deadline_misses') for r in rows) != integer(summary, 'deadline_misses'):
            raise ValueError('Group Deadline count mismatch')
    util_keys = [(*group_key(r), r['resource']) for r in utilization]
    if len(set(util_keys)) != len(util_keys) or set(util_keys) != {(*key, r) for key in expected for r in RESOURCES}:
        raise ValueError('Missing or duplicate utilization aggregate')
    for row in utilization:
        if integer(row, 'slots') != 1536:
            raise ValueError('Expected 16 x 96 equal arrival-window utilization slots')
    type_counts = defaultdict(int)
    type_keys = set()
    for row in types:
        key = (*group_key(row), row['task_type'], row['duration_group'])
        if key in type_keys:
            raise ValueError('Duplicate task type aggregate')
        type_keys.add(key)
        type_counts[group_key(row)] += integer(row, 'tasks')
    if set(type_counts) != expected or any(type_counts[k] != integer(group_map[k], 'arrived') for k in expected):
        raise ValueError('Task type denominators do not partition complete cohorts')
    return group_map


def matched_threshold_pairs(scenarios):
    """Match FIFO/CT on the identical candidate and original frozen scenario."""
    index = {}
    for row in scenarios:
        if row['candidate_id'] not in OBSERVED or row['policy'] not in ('fifo', 'carbon_threshold_fifo'):
            continue
        key = (row['candidate_id'], row['original_scenario_id'], row['policy'])
        if key in index:
            raise ValueError('Duplicate scenario-policy pairing key')
        index[key] = row
    result = []
    keys = sorted((c, original) for c, original, policy in index if policy == 'fifo')
    for candidate, original in keys:
        base = index[candidate, original, 'fifo']
        try:
            comparison = index[candidate, original, 'carbon_threshold_fifo']
        except KeyError as error:
            raise ValueError('Missing matched Threshold scenario') from error
        identity_fields = ('original_scenario_identity_sha256', 'source_fragment_id',
                           'original_workload_sha256', 'mapped_workload_sha256',
                           'original_environment_sha256', 'candidate_config_sha256',
                           'mapping_version', 'resource_pressure_factor', 'sla_multiplier')
        if any(base[field] != comparison[field] for field in identity_fields):
            raise ValueError('Matched policy input identity differs')
        n = integer(base, 'arrived')
        if n != integer(comparison, 'arrived') or integer(base, 'completed') != n or integer(comparison, 'completed') != n:
            raise ValueError('Matched cohort is incomplete or changed')
        row = {field: base[field] for field in ('candidate_id', 'mapping_version',
               'candidate_config_sha256', 'original_scenario_id', 'original_scenario_identity_sha256',
               'source_fragment_id', 'mapped_workload_sha256', 'original_workload_sha256',
               'original_environment_sha256', 'resource_pressure_factor', 'sla_multiplier')}
        row.update(fifo_scenario_id=base['scenario_id'], threshold_scenario_id=comparison['scenario_id'],
                   tasks=n, baseline_policy='fifo', comparison_policy='carbon_threshold_fifo',
                   fifo_deadline_misses=integer(base, 'deadline_misses'),
                   threshold_deadline_misses=integer(comparison, 'deadline_misses'),
                   fifo_dmr=number(base, 'dmr'), threshold_dmr=number(comparison, 'dmr'),
                   dmr_change_percentage_points=100*(number(comparison, 'dmr')-number(base, 'dmr')),
                   fifo_mean_jct_minutes=number(base, 'mean_jct_minutes'),
                   threshold_mean_jct_minutes=number(comparison, 'mean_jct_minutes'),
                   jct_change_minutes=number(comparison, 'mean_jct_minutes')-number(base, 'mean_jct_minutes'),
                   fifo_capacity_waited_tasks=integer(base, 'capacity_waited_tasks'),
                   threshold_capacity_waited_tasks=integer(comparison, 'capacity_waited_tasks'),
                   threshold_policy_deferred_tasks=integer(comparison, 'policy_deferred_tasks'),
                   fifo_dmr_le5=number(base, 'dmr') <= .05,
                   threshold_dmr_le5=number(comparison, 'dmr') <= .05)
        for resource in RESOURCES:
            hours = number(base, resource+'_resource_hours')
            other_hours = number(comparison, resource+'_resource_hours')
            if not math.isclose(hours, other_hours, rel_tol=1e-11, abs_tol=1e-7):
                raise ValueError('Matched completed resource-hours differ')
            b = number(base, resource+'_ci_exposure_proxy')
            t = number(comparison, resource+'_ci_exposure_proxy')
            rb = number(base, resource+'_reserved_ci_exposure_proxy')
            rt = number(comparison, resource+'_reserved_ci_exposure_proxy')
            row.update({resource+'_resource_hours': hours,
                        resource+'_fifo_ci_exposure_proxy': b,
                        resource+'_threshold_ci_exposure_proxy': t,
                        resource+'_ci_exposure_reduction_percent': reduction_percent(b, t),
                        resource+'_fifo_mean_execution_ci_g_per_kwh': b/hours if hours else None,
                        resource+'_threshold_mean_execution_ci_g_per_kwh': t/hours if hours else None,
                        resource+'_fifo_reserved_ci_exposure_proxy': rb,
                        resource+'_threshold_reserved_ci_exposure_proxy': rt,
                        resource+'_reserved_exposure_reduction_percent': reduction_percent(rb, rt)})
        result.append(row)
    if len(index) != 2*len(result):
        raise ValueError('Unpaired Threshold scenario')
    return result


def summarize_pairs(pairs):
    grouped = defaultdict(list)
    for row in pairs:
        grouped[row['candidate_id'], number(row, 'resource_pressure_factor'), number(row, 'sla_multiplier')].append(row)
    result = []
    for (candidate, pressure, sla), rows in sorted(grouped.items()):
        n = sum(r['tasks'] for r in rows)
        row = dict(candidate_id=candidate, mapping_version=rows[0]['mapping_version'],
                   candidate_config_sha256=rows[0]['candidate_config_sha256'],
                   resource_pressure_factor=pressure, sla_multiplier=sla, fragments=len(rows), tasks=n,
                   baseline_policy='fifo', comparison_policy='carbon_threshold_fifo',
                   fifo_deadline_misses=sum(r['fifo_deadline_misses'] for r in rows),
                   threshold_deadline_misses=sum(r['threshold_deadline_misses'] for r in rows),
                   fifo_fragments_dmr_gt5=sum(not r['fifo_dmr_le5'] for r in rows),
                   threshold_fragments_dmr_gt5=sum(not r['threshold_dmr_le5'] for r in rows),
                   newly_above5_fragments=sum(r['fifo_dmr_le5'] and not r['threshold_dmr_le5'] for r in rows),
                   dmr_increased_fragments=sum(r['threshold_dmr'] > r['fifo_dmr'] for r in rows),
                   fifo_capacity_waited_tasks=sum(r['fifo_capacity_waited_tasks'] for r in rows),
                   threshold_capacity_waited_tasks=sum(r['threshold_capacity_waited_tasks'] for r in rows),
                   threshold_policy_deferred_tasks=sum(r['threshold_policy_deferred_tasks'] for r in rows))
        row['fifo_dmr'] = row['fifo_deadline_misses']/n
        row['threshold_dmr'] = row['threshold_deadline_misses']/n
        row['dmr_change_percentage_points'] = 100*(row['threshold_dmr']-row['fifo_dmr'])
        row['jct_change_minutes'] = math.fsum(r['jct_change_minutes']*r['tasks'] for r in rows)/n
        for resource in RESOURCES:
            hours = math.fsum(r[resource+'_resource_hours'] for r in rows)
            b = math.fsum(r[resource+'_fifo_ci_exposure_proxy'] for r in rows)
            t = math.fsum(r[resource+'_threshold_ci_exposure_proxy'] for r in rows)
            rb = math.fsum(r[resource+'_fifo_reserved_ci_exposure_proxy'] for r in rows)
            rt = math.fsum(r[resource+'_threshold_reserved_ci_exposure_proxy'] for r in rows)
            row.update({resource+'_resource_hours': hours,
                        resource+'_fifo_ci_exposure_proxy': b,
                        resource+'_threshold_ci_exposure_proxy': t,
                        resource+'_ci_exposure_reduction_percent': reduction_percent(b, t),
                        resource+'_fifo_mean_execution_ci_g_per_kwh': b/hours if hours else None,
                        resource+'_threshold_mean_execution_ci_g_per_kwh': t/hours if hours else None,
                        resource+'_fifo_reserved_ci_exposure_proxy': rb,
                        resource+'_threshold_reserved_ci_exposure_proxy': rt,
                        resource+'_reserved_exposure_reduction_percent': reduction_percent(rb, rt)})
        result.append(row)
    return result


def matched_type_pairs(types):
    """Task-type means support duration weighting, not GPU-demand weighting."""
    index = {(*group_key(r), r['task_type'], r['duration_group']): r for r in types}
    result = []
    for key, base in sorted(index.items()):
        candidate, pressure, sla, policy, task_type, duration_group = key
        if candidate not in OBSERVED or policy != 'fifo':
            continue
        comparison = index.get((candidate, pressure, sla, 'carbon_threshold_fifo', task_type, duration_group))
        if comparison is None:
            raise ValueError('Missing matched task type')
        n = integer(base, 'tasks')
        duration = number(base, 'duration_minutes_mean')
        if n != integer(comparison, 'tasks') or not math.isclose(duration, number(comparison, 'duration_minutes_mean'), rel_tol=1e-11):
            raise ValueError('Task type count or original duration differs')
        b = number(base, 'execution_ci_integral_g_per_kwh_hours_mean')*n
        t = number(comparison, 'execution_ci_integral_g_per_kwh_hours_mean')*n
        hours = duration*n/60
        result.append(dict(candidate_id=candidate, mapping_version=base['mapping_version'],
            resource_pressure_factor=pressure, sla_multiplier=sla, task_type=task_type,
            duration_group=duration_group, tasks=n, original_runtime_hours=hours,
            fifo_execution_ci_integral_sum_g_per_kwh_hours=b,
            threshold_execution_ci_integral_sum_g_per_kwh_hours=t,
            fifo_duration_weighted_execution_ci_g_per_kwh=b/hours,
            threshold_duration_weighted_execution_ci_g_per_kwh=t/hours,
            duration_weighted_ci_reduction_percent=reduction_percent(b, t),
            fifo_dmr=number(base, 'dmr'), threshold_dmr=number(comparison, 'dmr'),
            fifo_capacity_waited_tasks=integer(base, 'capacity_waited_tasks'),
            threshold_capacity_waited_tasks=integer(comparison, 'capacity_waited_tasks'),
            threshold_policy_deferred_tasks=integer(comparison, 'policy_deferred_tasks'),
            fifo_observed_wait_mean_minutes=number(base, 'observed_wait_minutes_mean'),
            threshold_observed_wait_mean_minutes=number(comparison, 'observed_wait_minutes_mean'),
            weighting='task runtime hours; NOT CPU/GPU/memory demand and NOT energy'))
    return result


def fragment_competition(scenarios):
    fields = ('candidate_id', 'mapping_version', 'candidate_config_sha256', 'scenario_id',
        'original_scenario_id', 'original_scenario_identity_sha256', 'source_fragment_id',
        'mapped_workload_sha256', 'original_environment_sha256', 'policy',
        'resource_pressure_factor', 'sla_multiplier', 'arrival_start', 'arrival_end',
        'active_observation_end', 'arrived', 'completed', 'dmr', 'capacity_waited_tasks',
        'policy_deferred_tasks', 'capacity_competition_minutes', 'capacity_competition_longest_minutes',
        'policy_defer_active_minutes', 'policy_defer_longest_minutes', 'capacity_waiting_tasks_peak',
        'queue_length_max_through_drain', 'queue_nonempty_minutes_through_drain',
        'cpu_bottleneck_active_minutes', 'gpu_bottleneck_active_minutes', 'mem_bottleneck_active_minutes',
        'cpu_bottleneck_attempts', 'gpu_bottleneck_attempts', 'mem_bottleneck_attempts')
    return [dict((k, r[k]) for k in fields) for r in scenarios]


def f(value, digits=2):
    return '—' if value is None else f'{float(value):.{digits}f}'


def pct(value):
    return f(100*float(value))


def table(headers, rows):
    lines = ['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join('---' for _ in headers) + ' |']
    lines += ['| ' + ' | '.join(str(v).replace('|', '\\|') for v in row) + ' |' for row in rows]
    return '\n'.join(lines)


def render_report(metadata, group_map, utilization, pair_summary, type_pairs):
    ordered = sorted(group_map.values(), key=lambda r: (CANDIDATES.index(r['candidate_id']),
                     number(r, 'sla_multiplier'), number(r, 'resource_pressure_factor'), r['policy']))
    text = ['# A1：确定性规则实验统计附表', '',
        '本文件由 `python -m experiments.object_model_a1.tables` 从完整的公开聚合 CSV 生成。'
        '它不读取任务明细或年度数据，也不调用 IT/HVAC 模型。原始场景身份、候选配置及逐片段统计均保留在配套 CSV。', '',
        f"实际完成 **{metadata['completed_runs']:,} / {metadata['planned_runs']:,}** 次规则仿真；"
        f"协议基准 `{metadata['source_sha']}`；实际仿真 producer 提交 "
        f"`{metadata.get('diagnostic_source_commit', '未提供；以逐文件运行SHA为准')}`；"
        f"源池清单 SHA256 `{metadata['pool_manifest_sha256']}`。", '',
        '统计单位：每个候选×压力×SLA×规则包含16个不同来源片段，任务数见各表 n；'
        '相同片段的压力/SLA/候选增强不是独立实测样本。DMR分母为同一到达任务全集；'
        '所有完成率须按该表及原 CSV 核对，未删任务。分位数用 numpy linear 方法；资源利用率使用16×96=1,536个等权15分钟到达窗口槽。', '',
        '## 1. 六候选：压力1、FIFO/EDF', '',
        '等待率仅指经历过容量不足的任务；平均等待为全体任务的容量等待，正等待均值只以容量等待>0的任务为分母。'
        '最坏片段DMR与汇总DMR同时报告，避免合并平均掩盖高峰。', '']
    rows = []
    for r in ordered:
        if number(r, 'resource_pressure_factor') != 1 or r['policy'] not in ('fifo', 'edf'):
            continue
        rows.append([r['candidate_id'], r['sla_multiplier'], POLICY_LABELS[r['policy']], r['arrived'],
            pct(r['capacity_waited_fraction']), f(r['capacity_wait_minutes_mean']),
            f(r['capacity_wait_positive_mean_minutes']), f(r['capacity_wait_minutes_p95']),
            f(r['queue_peak'], 0), pct(r['dmr']), pct(r['worst_fragment_dmr']),
            str(r['fragments_with_dmr_gt5_percent'])+'/16', pct(r['completion_fraction']),
            f(r['jct_minutes_mean']), r['worst_fragment_id']])
    text += [table(['候选','SLA','规则','n','容量等待%','均等min','正等待均min','等待P95min',
                    '队峰','DMR%','最坏DMR%','片段>5%','完成%','平均JCTmin','最坏片段ID'], rows), '',
        '最坏片段的完整原场景ID见 `rules/candidate-summary.csv` 的 `worst_fragment_id`，'
        '完整比较见 `fragment-competition.csv`。汇总达标不等于每个片段均达标。', '',
        '## 2. Observed三候选：所有压力、SLA与规则', '',
        'CT/FIFO为同FIFO顺序的固定碳阈值规则。主动等待和容量等待分开；队列长度包含两者。'
        '竞争总分钟为16个片段各自容量等待时间并集之和，不表示连续16天的一条轨迹。', '']
    for candidate in OBSERVED:
        rows=[]
        for r in ordered:
            if r['candidate_id'] != candidate:
                continue
            rows.append([r['resource_pressure_factor'],r['sla_multiplier'],POLICY_LABELS[r['policy']],r['arrived'],
                pct(r['capacity_waited_fraction']),pct(r['policy_deferred_fraction']),
                f(r['observed_wait_minutes_mean']),f(r['queue_peak'],0),pct(r['dmr']),
                pct(r['worst_fragment_dmr']),str(r['fragments_with_dmr_gt5_percent'])+'/16',
                pct(r['completion_fraction']),f(r['jct_minutes_mean']),
                f(r['capacity_competition_minutes'],0),f(r['longest_competition_minutes'],0)])
        text += [f'### {candidate}', '', table(['压力','SLA','规则','n','容量等待%','主动等待%',
            '总等均min','队峰','DMR%','最坏DMR%','片段>5%','完成%','JCTmin','竞争总min','最长竞争min'],rows), '']
    text += ['## 3. CPU/GPU/MEM利用率：压力1、EDF', '',
        '全部数值为候选逻辑容量百分比。均值、P50/P90/P95/P99和峰值来自完整的1,536个到达窗口槽，'
        '不包含31天长空闲尾段；完整drain的峰值另保留在规则CSV。', '']
    rows=[]
    for r in sorted(utilization,key=lambda r:(CANDIDATES.index(r['candidate_id']),number(r,'sla_multiplier'),RESOURCES.index(r['resource']))):
        if number(r,'resource_pressure_factor') == 1 and r['policy']=='edf':
            rows.append([r['candidate_id'],r['sla_multiplier'],r['resource'].upper(),r['slots']]+[
                pct(r['utilization_'+k]) for k in ('mean','p50','p90','p95','p99','max')])
    text += [table(['候选','SLA','资源','槽n','均值%','P50%','P90%','P95%','P99%','峰值%'],rows), '',
        '## 4. 理论时间柔性与竞争后的剩余余量', '',
        '理论Slack=(SLA−1)×原Duration；合法等待槽=floor(Slack/15min)。下表只取OM、压力1、FIFO，'
        '因为改变容量或规则不会改变任务自身的理论Slack。', '']
    rows=[]
    for sla in SLAS:
        r=group_map['OM',1.,sla,'fifo']
        n=integer(r,'arrived')
        rows.append([sla,n,f(r['duration_minutes_p50']),f(r['duration_minutes_p95']),
            f(r['initial_slack_minutes_p50']),f(r['initial_slack_minutes_p95']),
            pct(r['legal_wait_slots_0_fraction'])]+[
            f(100*integer(r,'can_wait_ge'+str(m)+'_tasks')/n) for m in (15,30,60)])
    text += [table(['SLA','n','Duration P50min','P95min','Slack P50min','P95min','不能等15min%',
                    '可等≥15min%','≥30min%','≥60min%'],rows), '',
        '以下比较压力1的剩余单体Slack。余量=初始Slack−实际总等待；负余量与零合法槽分开。'
        '“长Slack”指初始≥60min；“容量等待且违约”为统计交集，不能解释为容量独立造成的违约。', '']
    rows=[]
    for r in ordered:
        if r['candidate_id'] not in OBSERVED or number(r,'resource_pressure_factor') != 1:
            continue
        n=integer(r,'arrived')
        rows.append([r['candidate_id'],r['sla_multiplier'],POLICY_LABELS[r['policy']],n,
            r['observed_remaining_legal_wait_slots_negative_tasks'],r['observed_remaining_legal_wait_slots_0_tasks'],
            r['observed_remaining_legal_wait_slots_ge4_tasks'],r['long_slack_tasks'],
            r['long_slack_capacity_waited_tasks'],r['long_slack_capacity_missed_tasks'],r['long_slack_all_missed_tasks']])
    text += [table(['候选','SLA','规则','n','余量负n','余量0槽n','余量≥4槽n','长Slack n',
                    '长且容量等n','长且容量等且miss n','长且miss n'],rows), '',
        '## 5. 同源FIFO与CT/FIFO：完整运行CI暴露代理', '',
        '每一对使用同一候选、原场景、映射后任务、环境及完整完成任务集合，并验证CPU/GPU/MEM各自资源小时守恒。'
        '执行暴露 `X_r=Σ_i demand_ir × ∫start_i^finish_i CI(t)dt`，以实际Duration积分；'
        '保留/释放尾槽的暴露另列 `reserved`，不混用。表中下降%=100×(FIFO−CT)/FIFO，负值为暴露上升；'
        '每组先分别求暴露总和再求比例，不平均片段百分比。**这些百分比不是真实减碳率。**', '']
    rows=[]
    for r in pair_summary:
        rows.append([r['candidate_id'],r['resource_pressure_factor'],r['sla_multiplier'],r['tasks'],
            f(r['cpu_ci_exposure_reduction_percent']),f(r['gpu_ci_exposure_reduction_percent']),
            f(r['mem_ci_exposure_reduction_percent']),pct(r['fifo_dmr']),pct(r['threshold_dmr']),
            f(r['dmr_change_percentage_points']),r['newly_above5_fragments'],f(r['jct_change_minutes'])])
    text += [table(['候选','压力','SLA','n','CPU暴露下降%','GPU暴露下降%','MEM暴露下降%',
                    'FIFO DMR%','CT DMR%','ΔDMR百分点','新超5%片段n','ΔJCTmin'],rows), '',
        '`matched-threshold-pairs.csv` 保留288个逐场景配对及原始场景ID/输入SHA；'
        '`matched-threshold-summary.csv` 保存18组资源小时加权CI与暴露。'
        '不跨资源相加这些proxy，不跨Legacy×5和Observed×1比较proxy绝对值。', '',
        '## 6. 任务类型与CI利用：压力1', '',
        '`gpu_positive`仅指该任务观测GPU用量>0，`cpu_only`指该值=0；'
        '它们不是原始训练/推理标签。Duration组为<60、60–<240、≥240min。'
        '现有类型CSV保留每任务CI积分的均值，所以此表使用Σ任务CI积分/Σ任务运行小时；'
        '这是**按任务运行时长加权CI**，没有按GPU需求或功耗加权。类型N及原Duration在FIFO/CT间核对相等。', '']
    rows=[]
    for r in type_pairs:
        if r['resource_pressure_factor'] != 1:
            continue
        rows.append([r['candidate_id'],r['sla_multiplier'],r['task_type'],r['duration_group'],r['tasks'],
            f(r['fifo_duration_weighted_execution_ci_g_per_kwh']),
            f(r['threshold_duration_weighted_execution_ci_g_per_kwh']),
            f(r['duration_weighted_ci_reduction_percent']),pct(r['fifo_dmr']),pct(r['threshold_dmr'])])
    text += [table(['候选','SLA','用量类','时长类','n','FIFO运行CI','CT运行CI','运行CI下降%',
                    'FIFO DMR%','CT DMR%'],rows), '',
        '全压力类型配对见 `matched-task-type-ci.csv`。运行CI单位gCO2e/kWh，仅说明工作发生时的电网强度；'
        '本轮没有建立每单位CPU/GPU/内存的能耗系数，不能生成kgCO2e、kWh、PUE或制冷节能结论。', '',
        '## 7. 可复现性与限制', '',
        '全部输入文件SHA、表格生成脚本SHA、结果计算口径见 `data/a1/tables-metadata.json`。'
        '逐片段竞争持续时间与规则指标见 `data/a1/fragment-competition.csv`；'
        '原始场景行汇总仍以 `data/a1/rules/scenario-summary.csv` 为完整证据。', '',
        '此处不据DMR结果删除任务，不把EDF表现当作可实现最优上界，也不把Threshold表现当作问题不存在的证据。'
        '16个来源片段不构成统计独立性证明，本附表不补造置信区间。研究对象仍待用户确认，A2未实施。', '']
    return '\n'.join(text)


def generate(input_dir, output_dir, report_path):
    input_dir, output_dir, report_path = map(Path, (input_dir, output_dir, report_path))
    metadata = json.loads((input_dir/'metadata.json').read_text(encoding='utf-8'))
    # Check this before reading partial CSV or creating public files.
    if metadata.get('complete') is not True:
        raise ValueError('Refuse partial A1 results: metadata.complete must be true')
    scenarios, groups, utilization, types = (read_csv(input_dir/name) for name in INPUT_FILES[:4])
    group_map = validate_complete(metadata, scenarios, groups, utilization, types)
    pairs = matched_threshold_pairs(scenarios)
    pair_summary = summarize_pairs(pairs)
    type_pairs = matched_type_pairs(types)
    if len(pairs) != 288 or len(pair_summary) != 18:
        raise ValueError('Expected 3 Observed candidates x 3 pressure x 2 SLA x 16 paired fragments')
    report = render_report(metadata, group_map, utilization, pair_summary, type_pairs)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    outputs = {'matched-threshold-pairs.csv': pairs,
               'matched-threshold-summary.csv': pair_summary,
               'matched-task-type-ci.csv': type_pairs,
               'fragment-competition.csv': fragment_competition(scenarios)}
    for name, rows in outputs.items():
        write_csv(output_dir/name, rows)
    report_path.write_bytes(report.encode('utf-8'))
    provenance = dict(schema='a1-aggregate-evidence-v1', source_sha=metadata['source_sha'],
        base_source_sha=metadata['base_source_sha'], generator_sha256=file_sha(__file__),
        diagnostic_source_commit=metadata.get('diagnostic_source_commit'),
        producer_run_rules_sha256=metadata.get('producer_run_rules_sha256',metadata['script_sha256']),
        producer_run_adapter_sha256=metadata.get('producer_run_adapter_sha256'),
        current_reproduction_rules_sha256=metadata.get('current_reproduction_rules_sha256'),
        input_sha256={name: file_sha(input_dir/name) for name in INPUT_FILES},
        input_complete=True, input_scenario_runs=len(scenarios), matched_scenario_pairs=len(pairs),
        matched_summary_groups=len(pair_summary), physics_model_called=False,
        source_pool_manifest_sha256=metadata['pool_manifest_sha256'],
        output_sha256={name: file_sha(output_dir/name) for name in outputs},
        report_sha256=file_sha(report_path),
        proxy_units='separate resource-unit * hour * gCO2e/kWh; NOT physical carbon or energy',
        type_weighting='sum individual task CI integrals / sum original task runtime hours; no resource demand weighting',
        comparison='within-candidate same original scenario and full completed cohort; FIFO versus CT/FIFO',
        ratio_aggregation='ratio of summed separate-resource exposures; no averaging of fragment percentage changes',
        output_contains_task_details=False)
    (output_dir/'tables-metadata.json').write_text(json.dumps(provenance, indent=2)+'\n', encoding='utf-8')
    return provenance


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, help='Complete rule aggregate directory')
    parser.add_argument('--output', required=True, help='Lightweight public aggregate output directory')
    parser.add_argument('--report', required=True)
    args = parser.parse_args(argv)
    result = generate(args.input, args.output, args.report)
    print(json.dumps({k: result[k] for k in ('input_scenario_runs', 'matched_scenario_pairs', 'matched_summary_groups')}))


if __name__ == '__main__':
    main()
