"""Analyze generated E1 result files without loading any input datasets."""
import argparse
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path


ARMS = ('A', 'B', 'C')
COMPARISONS = (('B', 'A'), ('C', 'A'), ('C', 'B'))
CONDITIONS = ('all_states', 'candidate_opportunity', 'candidate_and_carbon_trigger')


def _read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def _lines(path):
    if not path.exists():
        return []
    with path.open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _cell(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':')) if isinstance(value, (dict, list, tuple)) else value


def _csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: _cell(value) for key, value in row.items()} for row in rows)


def _mean(values):
    values = [v for v in values if v is not None and math.isfinite(v)]
    return sum(values)/len(values) if values else None


def _case(result, group):
    summary = result['summary']
    tasks = result['tasks']
    ledger_misses = sum(bool(t['deadline_missed']) for t in tasks)
    consistent = (len(tasks) == summary['arrived'] and ledger_misses == summary['deadline_missed']
                  and len({t['task_id'] for t in tasks}) == len(tasks))
    valid = bool(group['passed'] and summary.get('verification_pass')
        and not summary.get('failure_latched', True)
        and summary.get('reference_guard_failure_count') == 0
        and summary.get('actual_unfinished_count') == 0
        and summary.get('actual_common_idle_tail_valid') and consistent
        and [s['minute'] for s in result['steps']] == list(range(0, 4320, 15)))
    return dict(**summary, group_passed=bool(group['passed']), group_errors=group.get('errors', []),
        ledger_tasks=len(tasks), ledger_deadline_missed=ledger_misses,
        summary_ledger_consistent=consistent, valid_case=valid)


def _pair(fragment, capacity, left, right, cases, results):
    a, b = cases[left], cases[right]
    delta_temperature = a['inlet_final_c']-b['inlet_final_c']
    same_mass = abs(a['thermal_capacity_kj_k']-b['thermal_capacity_kj_k']) <= 1e-9
    same_tasks = ({t['task_id'] for t in results[left]['tasks']}
                  == {t['task_id'] for t in results[right]['tasks']})
    terminal_pass = abs(delta_temperature) <= .05+1e-9 and same_mass
    same_background = (a.get('common_idle_carbon_kg') is not None
        and b.get('common_idle_carbon_kg') is not None
        and abs(a['common_idle_carbon_kg']-b['common_idle_carbon_kg']) <= 1e-9)
    row = dict(fragment=fragment, capacity=capacity, comparison=left+'-'+right,
        valid_pair=bool(a['valid_case'] and b['valid_case'] and same_tasks
                        and same_background and terminal_pass and a.get('source_sha') == b.get('source_sha')),
        same_task_ids=same_tasks, same_common_idle_background=same_background,
        terminal_pair_check_pass=terminal_pass,
        delta_inlet_final_c=delta_temperature,
        delta_thermal_storage_kwh=a['thermal_capacity_kj_k']*delta_temperature/3600 if same_mass else None)
    for field in ('carbon_kg', 'common_incremental_carbon_kg', 'dmr', 'mean_jct_minutes',
                  'policy_wait_minutes', 'capacity_wait_minutes', 'it_kwh', 'hvac_kwh', 'total_kwh'):
        row['delta_'+field] = a[field]-b[field] if field in a and field in b else None
    row['delta_dmr_percentage_points'] = 100*row['delta_dmr']
    return row


def _divergence(a, b):
    indexed = [{row['minute']: row for row in rows} for rows in (a, b)]
    for minute in sorted(set(indexed[0]) & set(indexed[1])):
        signatures = [tuple(tuple(sorted(rows[minute].get(key, [])))
                            for key in ('actual_started', 'effective_policy_wait', 'capacity_wait'))
                      for rows in indexed]
        if signatures[0] != signatures[1]:
            return minute
    return None


def _arrival(task):
    if task is None:
        return None
    if task.get('arrival_minute') is not None:
        return task['arrival_minute']
    finish, jct = task.get('finish_minute'), task.get('jct_minutes')
    return finish-jct if finish is not None and jct is not None else None


def _task_diagnosis(fragment, capacity, results, decisions, checks):
    first = _divergence(decisions['A'], decisions['B'])
    task_maps = {arm: {t['task_id']: t for t in results[arm]['tasks']} for arm in ARMS}
    harm = defaultdict(list)
    for check in checks['B']:
        if check['approved']:
            for victim in check.get('lost_on_time_ids', []):
                harm[victim].append((check['minute'], check['task_id'], 'approved_candidate'))
    for record in decisions['B']:
        if record.get('approved'):
            for victim in record.get('final_lost_on_time_ids', []):
                harm[victim].append((record['minute'], None, 'final_joint_set'))
    rejected = [c for c in checks['C'] if not c['approved']]
    rejected_ids = {c['task_id'] for c in rejected}
    rejected_victims = {victim for c in rejected for victim in c.get('lost_on_time_ids', [])}
    rows = []
    for identity in sorted(set().union(*(set(t) for t in task_maps.values()))):
        tasks = {arm: task_maps[arm].get(identity) for arm in ARMS}
        complete = all(tasks.values())
        new_b = complete and not tasks['A']['deadline_missed'] and tasks['B']['deadline_missed']
        new_c = complete and not tasks['A']['deadline_missed'] and tasks['C']['deadline_missed']
        rescued = bool(new_b and not tasks['C']['deadline_missed'])
        own = bool(tasks['B'] and tasks['B']['policy_wait_minutes'] > 0)
        evidence = harm[identity]
        candidates = sorted({event[1] for event in evidence if event[1] is not None})
        c_related = identity in rejected_victims or bool(set(candidates) & rejected_ids)
        arrival = next((v for arm in ARMS if (v := _arrival(tasks[arm])) is not None), None)
        uncertain = bool(new_b and not evidence)
        row = dict(fragment=fragment, capacity=capacity, task_id=identity,
            all_three_task_rows_present=complete, arrival_minute=arrival,
            earliest_AB_actual_divergence_minute=first,
            future_at_first_AB_divergence=arrival > first if arrival is not None and first is not None else None,
            B_new_miss_vs_A=bool(new_b), C_new_miss_vs_A=bool(new_c), C_rescues_B_new_miss=rescued,
            B_own_policy_wait=own, B_indirect_effect_candidate=bool(new_b and not own),
            B_known_reference_harm=bool(evidence),
            B_harm_first_minute=min((e[0] for e in evidence), default=None),
            B_harm_candidate_ids=candidates, B_harm_proof_types=sorted({e[2] for e in evidence}),
            C_actual_reject_related=c_related,
            C_rejection_directly_lists_victim=identity in rejected_victims,
            C_rejected_B_harm_candidate_ids=sorted(set(candidates) & rejected_ids),
            same_state_C_counterfactual_proven=False,
            causal_link_uncertain=uncertain,
            evidence_level=('local_known_reference_harm_and_paired_trajectory' if new_b and evidence
                else 'paired_trajectory_only_causal_link_uncertain' if new_b else 'descriptive_task_comparison'))
        for arm, task in tasks.items():
            for field in ('start_minute', 'finish_minute', 'deadline_minute', 'deadline_missed',
                          'policy_wait_minutes', 'capacity_wait_minutes', 'jct_minutes', 'state'):
                row[arm+'_'+field] = task.get(field) if task else None
        rows.append(row)
    totals = dict(B_new_misses_vs_A=sum(r['B_new_miss_vs_A'] for r in rows),
        C_new_misses_vs_A=sum(r['C_new_miss_vs_A'] for r in rows),
        C_rescues_B_new_misses=sum(r['C_rescues_B_new_miss'] for r in rows),
        B_new_misses_with_local_reference_harm=sum(r['B_new_miss_vs_A'] and r['B_known_reference_harm'] for r in rows),
        B_new_misses_with_own_policy_wait=sum(r['B_new_miss_vs_A'] and r['B_own_policy_wait'] for r in rows),
        B_new_misses_without_own_policy_wait=sum(r['B_new_miss_vs_A'] and not r['B_own_policy_wait'] for r in rows),
        B_new_misses_future_at_first_divergence=sum(r['B_new_miss_vs_A'] and r['future_at_first_AB_divergence'] is True for r in rows),
        B_new_misses_with_C_actual_reject_relation=sum(r['B_new_miss_vs_A'] and r['C_actual_reject_related'] for r in rows),
        B_new_misses_causal_link_uncertain=sum(r['causal_link_uncertain'] for r in rows))
    return rows, dict(fragment=fragment, capacity=capacity,
        earliest_AB_actual_divergence_minute=first,
        AB_decision_logs_complete=all([d['minute'] for d in decisions[a]] == list(range(0, 4320, 15)) for a in ('A', 'B')),
        **totals)


def _eligible(row, condition):
    return condition == 'all_states' or (row['candidate_count'] > 0
        and (condition == 'candidate_opportunity' or row['carbon_trigger']))


def _action_reports(probes):
    indexed = defaultdict(list)
    for row in probes:
        indexed[row['fragment'], row['capacity'], row['minute']].append(row)
    states = []
    for (fragment, capacity, minute), rows in sorted(indexed.items()):
        valid = len(rows) == 7 and {r['action_id'] for r in rows} == set(range(7))
        signatures = [_cell(r['signature']) for r in rows]
        unique = len(set(signatures))
        noop = next((_cell(r['signature']) for r in rows if r['action_id'] == 0), None)
        states.append(dict(fragment=fragment, capacity=capacity, minute=minute,
            candidate_count=max(r['candidate_count'] for r in rows),
            carbon_trigger=any(r['carbon_trigger'] for r in rows), complete_seven_actions=valid,
            unique_execution_signatures=unique,
            alias_ratio=(7-unique)/6 if valid else None,
            noop_equivalence_fraction=sum(r['action_id'] != 0 and _cell(r['signature']) == noop for r in rows)/6 if valid else None,
            unique_approved_signatures=len({_cell(r['approved_signature']) for r in rows}),
            probe_elapsed_ms=sum(r['elapsed_ms'] for r in rows)))
    aliases, coverage = [], []
    scopes = [('ALL', states, probes)]
    for fragment, capacity in sorted({(r['fragment'], r['capacity']) for r in states}):
        scopes.append((fragment+'__'+capacity,
            [r for r in states if r['fragment'] == fragment and r['capacity'] == capacity],
            [r for r in probes if r['fragment'] == fragment and r['capacity'] == capacity]))
    for scope, state_rows, action_rows in scopes:
        for condition in CONDITIONS:
            selected = [r for r in state_rows if _eligible(r, condition)]
            aliases.append(dict(scope=scope, condition=condition, states=len(selected),
                complete_seven_action_states=sum(r['complete_seven_actions'] for r in selected),
                mean_unique_execution_signatures=_mean([r['unique_execution_signatures'] for r in selected]),
                mean_alias_ratio=_mean([r['alias_ratio'] for r in selected]),
                mean_noop_equivalence_fraction=_mean([r['noop_equivalence_fraction'] for r in selected])))
            for action in range(7):
                chosen = [r for r in action_rows if r['action_id'] == action and _eligible(r, condition)]
                row = dict(scope=scope, condition=condition, action_id=action, states=len(chosen))
                for field in ('power_coverage', 'cpu_coverage', 'gpu_coverage', 'mem_coverage',
                              'proposed_count', 'approved_count', 'effective_policy_wait_count',
                              'capacity_wait_count', 'started_count', 'elapsed_ms'):
                    row['mean_'+field] = _mean([r[field] for r in chosen])
                coverage.append(row)
    return states, aliases, coverage


def _dmr(rows):
    tasks = sum(row['ledger_tasks'] for row in rows)
    misses = sum(row['ledger_deadline_missed'] for row in rows)
    return dict(cases=len(rows), tasks=tasks, deadline_missed=misses,
        task_weighted_dmr=misses/tasks if tasks else None,
        invalid_cases=sum(not row['valid_case'] for row in rows))


def analyze(output, publish):
    """Read local generated results and write small tables plus a local full ledger."""
    output, publish = Path(output).resolve(), Path(publish).resolve()
    cases, pairs, diagnosis, groups, probes, errors = [], [], [], [], [], []
    folders = sorted(p for p in output.iterdir() if p.is_dir() and '__' in p.name)
    for folder in folders:
        group_file = folder/'group.json'
        if group_file.exists():
            group = _read(group_file)
        else:
            fragment, capacity = folder.name.rsplit('__', 1)
            group = dict(fragment=fragment, capacity=capacity, passed=False,
                         errors=['Group did not finish; final group.json absent'])
        fragment, capacity = group['fragment'], group['capacity']
        results, summaries, decisions, checks = {}, {}, {}, {}
        for arm in ARMS:
            result_file = folder/arm/'result.json'
            error_file = folder/arm/'error.json'
            if error_file.exists():
                failure = _read(error_file)
                error_row = dict(fragment=fragment, capacity=capacity, arm=arm,
                    minute=failure.get('minute'), error=failure.get('error'))
                errors.append(error_row)
                if not result_file.exists():
                    cases.append(dict(fragment=fragment, capacity=capacity, arm=arm,
                        case_id=fragment+'__'+capacity+'__'+arm, valid_case=False,
                        group_passed=False, group_errors=group.get('errors', []),
                        ledger_tasks=0, ledger_deadline_missed=0,
                        missing_result=True, error=error_row['error']))
            if not result_file.exists():
                continue
            results[arm] = _read(result_file)
            summaries[arm] = _case(results[arm], group); cases.append(summaries[arm])
            decisions[arm] = _lines(folder/arm/'decisions.jsonl')
            checks[arm] = _lines(folder/arm/'candidates.jsonl')
        for left, right in COMPARISONS:
            if left in results and right in results:
                pairs.append(_pair(fragment, capacity, left, right, summaries, results))
        if set(results) == set(ARMS):
            task_rows, group_diagnosis = _task_diagnosis(fragment, capacity, results, decisions, checks)
            diagnosis.extend(task_rows); groups.append(dict(**group_diagnosis, passed=group['passed']))
        else:
            groups.append(dict(fragment=fragment, capacity=capacity, passed=False,
                missing_arms=sorted(set(ARMS)-set(results))))
        probes.extend(dict(row, fragment=fragment, capacity=capacity)
                      for row in _lines(folder/'C/action-probes.jsonl'))
    cases.sort(key=lambda r: (r['fragment'], r['capacity'], r['arm']))
    aggregate = []
    by_arm = {arm: _dmr([r for r in cases if r['arm'] == arm]) for arm in ARMS}
    for capacity in sorted({row['capacity'] for row in cases}):
        for arm in ARMS:
            aggregate.append(dict(capacity=capacity, arm=arm,
                **_dmr([r for r in cases if r['capacity'] == capacity and r['arm'] == arm])))
    aggregate.extend(dict(capacity='ALL_CAPACITIES', arm=arm, **by_arm[arm]) for arm in ARMS)
    aggregate.append(dict(capacity='ALL_CAPACITIES', arm='ALL_ARMS_DESCRIPTIVE', **_dmr(cases)))
    states, aliases, coverage = _action_reports(probes)
    representative = [r for r in diagnosis if r['B_new_miss_vs_A'] or r['C_new_miss_vs_A']
                      or r['C_actual_reject_related'] or r['B_own_policy_wait']]
    representative.sort(key=lambda r: (not r['B_new_miss_vs_A'], not r['B_known_reference_harm'],
                                       r['fragment'], r['capacity'], r['task_id']))
    _csv(output/'task-diagnosis-all.csv', diagnosis)
    for filename, rows in [('case-summary.csv', cases), ('paired-results.csv', pairs),
            ('dmr-aggregate.csv', aggregate), ('task-diagnosis-cases.csv', representative[:30]),
            ('action-state-summary.csv', states), ('action-alias-summary.csv', aliases),
            ('action-coverage.csv', coverage), ('failed-episodes.csv', errors)]:
        _csv(publish/filename, rows)
    victim_counts = Counter()
    for group in groups:
        victim_counts.update({key: value for key, value in group.items()
                              if key.startswith(('B_new_misses_', 'C_new_misses_', 'C_rescues_'))})
    decision_totals = {}
    for arm in ARMS:
        selected = [row for row in cases if row['arm'] == arm]
        reasons = Counter()
        for row in selected:
            reasons.update(row.get('rejection_reason_counts', {}))
        decision_totals[arm] = dict(
            **{key: sum(row.get(key, 0) for row in selected) for key in ('attempts', 'approved', 'rejected')},
            rejection_reason_counts=dict(reasons),
            maximum_decision_ms=max((row.get('decision_max_ms', 0) for row in selected), default=None),
            maximum_case_p95_ms=max((row.get('decision_p95_ms', 0) for row in selected), default=None))
    report = dict(schema='sca-ppo-e1-generated-result-analysis-v1',
        local_output=output.as_posix(), expected_case_count=27, actual_case_count=len(cases),
        matrix_complete=len(cases) == 27 and len(groups) == 9,
        valid_case_count=sum(row['valid_case'] for row in cases),
        invalid_case_count=sum(not row['valid_case'] for row in cases),
        valid_pair_count=sum(row['valid_pair'] for row in pairs), pair_count=len(pairs),
        error_episode_count=len(errors), error_episodes=errors,
        dmr_by_arm=by_arm, victims=dict(victim_counts), groups=groups,
        full_task_diagnosis_rows=len(diagnosis), representative_task_rows=min(30, len(representative)),
        decision_totals_by_arm=decision_totals,
        action_sampling=dict(states=len(states), expected_states=324,
            complete_seven_action_states=sum(row['complete_seven_actions'] for row in states),
            sampled_actual_trajectory='C', hours='integer hours 0 through 35',
            sampled_states_are_not_full_trajectory_census=True,
            alias_formula='(7 - unique_execution_signatures) / 6',
            noop_equivalence_formula='nonzero actions equivalent to action 0 / 6'),
        action_alias_aggregate=[row for row in aliases if row['scope'] == 'ALL'],
        caveats=[
            'DMR aggregates weight tasks, not fragment means; repeated capacities and arms are not independent samples.',
            'Invalid arms and groups remain in descriptive tables and denominators; inspect validity flags before interpretation.',
            'An episode without a saved task ledger has unknown task outcomes; DMR denominators count only preserved task rows.',
            'An actual policy wait alone does not prove that a particular deferral caused an observed miss.',
            'B known-reference harm is a local known-task counterfactual; paired A/B trajectories can later diverge.',
            'C actual rejection evidence comes from its own trajectory and is not a matched-state B/C causal experiment.',
            'Future-at-first-divergence flags require an observed or reconstructible arrival time; unknown remains blank.',
            'The fixed hourly action sample includes empty states; opportunity-conditional denominators are reported separately.'])
    publish.mkdir(parents=True, exist_ok=True)
    (publish/'analysis-summary.json').write_text(json.dumps(report, sort_keys=True, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, help='Generated local E1 result directory')
    parser.add_argument('--publish', required=True, help='Directory for lightweight analysis tables')
    args = parser.parse_args()
    report = analyze(args.output, args.publish)
    print(json.dumps(dict(cases=report['actual_case_count'], invalid=report['invalid_case_count'],
                          matrix_complete=report['matrix_complete'])))


if __name__ == '__main__':
    main()
