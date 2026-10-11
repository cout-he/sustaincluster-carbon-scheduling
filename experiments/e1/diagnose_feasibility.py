"""Read-only E1-4A checks of archived CARS-v1 decisions and final outcomes.

An incremental shadow miss is a counterfactual schedule set membership, not a
probability forecast. Pairing it with an actual terminal outcome is descriptive
only: the actual executor applies carbon guards again, and sees later arrivals.
"""
from collections import Counter
from bisect import bisect_right
import csv
import json
import math
from pathlib import Path
import statistics
from types import SimpleNamespace

import pandas as pd

from .contracts import STEP
from .cars_feasibility import ShadowPlanner
from .cars_state import ResourceSnapshot, RunningTask

CAMPAIGN_POLICIES = ('edf_carbon_threshold', 'cars_risk_only', 'cars_full',
                     'risk_cap_only', 'risk_shadow_only')


def _risk_only_eligible(row):
    if 'risk_only_defer' in row:
        return bool(row['risk_only_defer'])
    if row['reason'] in ('low_ci_execute', 'no_grid_slack', 'risk_buffer_guard'):
        return False
    # EDF-carbon controls recorded the same observed history but did not apply
    # its guard. Reconstruct RiskOnly eligibility for this denominator only.
    if row.get('risk_enabled') is False and all(key in row for key in ('time', 'latest_start', 'buffer_steps')):
        if pd.Timestamp(row['time'])+STEP*(1+row['buffer_steps']) > pd.Timestamp(row['latest_start']):
            return False
    return row['reason'] in ('high_ci_defer', 'known_deadline_risk',
        'known_feasibility_uncertain', 'cap_execute_guard', 'shadow_uncertain_risk_fallback')


def _planner_executed(row):
    if 'planner' not in row:
        return False
    if 'shadow_planner_executed' in row:
        return bool(row['shadow_planner_executed'])
    planner = row['planner']
    if 'planner_calls' in planner:
        return planner['planner_calls'] > 0
    return planner['reason'] != 'pending_limit'


def planner_status_counts(diagnostics):
    """Separate synthetic status reports from actual bounded planning work."""
    reports = [row for row in diagnostics if 'planner' in row]
    actual = [row for row in reports if _planner_executed(row)]
    reported = Counter(row['planner']['status'] for row in reports)
    executed = Counter(row['planner']['status'] for row in actual)
    result = dict(planner_reported_status_decisions=len(reports),
        planner_compared_decisions=len(reports),  # compatibility; reports include bypass
        planner_status_bypass_count=len(reports)-len(actual),
        planner_actual_execution_decisions=len(actual),
        planner_actual_scheduling_passes=sum(row['planner'].get('planner_calls', 2) for row in actual))
    for status in ('safe', 'risk', 'uncertain'):
        result['planner_'+status+'_count'] = reported[status]
        result['planner_'+status+'_proportion'] = reported[status]/len(reports) if reports else None
        result['planner_actual_'+status+'_count'] = executed[status]
        result['planner_actual_'+status+'_proportion'] = executed[status]/len(actual) if actual else None
    return result


def decision_counts(diagnostics, max_pending=256):
    reasons = Counter(row['reason'] for row in diagnostics)
    uncertain = Counter(row['planner']['reason'] for row in diagnostics
                        if row.get('planner', {}).get('status') == 'uncertain')
    cap = sum(not row.get('defer', False) and
        (row['reason']=='cap_execute_guard' or (row['reason']=='known_feasibility_uncertain'
            and row.get('planner', {}).get('reason')=='pending_limit')) for row in diagnostics)
    other = sum(row['reason']=='known_feasibility_uncertain' and not row.get('defer', False)
        and row.get('planner', {}).get('reason')!='pending_limit' for row in diagnostics)
    eligible = [_risk_only_eligible(row) for row in diagnostics]
    over = [bool(row.get('pending_limit_exceeded', row.get('pending_count', 0)>max_pending))
            for row in diagnostics]
    return dict(candidate_checks=len(diagnostics),
        history_risk_buffer_guards=reasons['risk_buffer_guard'],
        explicit_shadow_risk_guards=reasons['known_deadline_risk'],
        pending_cap_fallbacks=cap, cap_execute_guards=cap,
        other_uncertain_fallbacks=other,
        uncertain_risk_fallbacks=reasons['shadow_uncertain_risk_fallback'],
        planner_uncertain_status_count=sum(uncertain.values()),
        pending_limit_status_count=uncertain['pending_limit'],
        planner_actual_execution_decisions=sum(_planner_executed(row) for row in diagnostics),
        queue_limit_all_candidate_checks=sum(over),
        risk_only_eligible_candidate_checks=sum(eligible),
        queue_limit_eligible_candidate_checks=sum(a and b for a, b in zip(over, eligible)),
        reason_counts=dict(reasons), planner_uncertain_reasons=dict(uncertain))


def pair_known_risk(diagnostics, ledger):
    """One row per (explicit-risk decision, incremental predicted miss ID)."""
    paired = []
    for row in diagnostics:
        planner = row.get('planner', {})
        if planner.get('status') != 'risk':
            continue
        new_ids = sorted(set(planner['missed_defer'])-set(planner['missed_now']))
        for task_id in new_ids:
            if task_id not in ledger:
                raise ValueError(f'Predicted task absent from actual ledger: {task_id}')
            actual = ledger[task_id]
            outcome = actual['deadline_outcome']
            paired.append(dict(time=row['time'], candidate_id=row['task_id'],
                risk_task_id=task_id, missed_now_count=len(planner['missed_now']),
                missed_defer_count=len(planner['missed_defer']),
                actual_outcome=outcome, actual_miss=outcome in ('missed', 'missed_unfinished'),
                actual_start_time=actual.get('start_time'),
                actual_finish_time=actual.get('finish_time'),
                sla_deadline=actual.get('sla_deadline')))
    return paired


def risk_outcome_counts(paired):
    missed = [row for row in paired if row['actual_miss']]
    met = sum(row['actual_outcome'] == 'met' for row in paired)
    return dict(prediction_occurrences=len(paired),
        unique_predicted_tasks=len({row['risk_task_id'] for row in paired}),
        actual_miss_occurrences=len(missed),
        unique_actual_miss_tasks=len({row['risk_task_id'] for row in missed}),
        actual_met_occurrences=met, other_outcome_occurrences=len(paired)-len(missed)-met)


def select_cases(diagnostics):
    """Fixed, outcome-independent rules; absent selections remain explicit."""
    ordered = sorted(diagnostics, key=lambda row: (pd.Timestamp(row['time']).value, row['task_id']))
    def first(predicate):
        return next((row for row in ordered if predicate(row)), None)
    historical_time = pd.Timestamp('2023-07-01T10:15:00Z')
    return dict(
        first_known_risk=first(lambda r: r['reason'] == 'known_deadline_risk'),
        first_pending_cap_fallback=first(lambda r:
            r.get('planner', {}).get('reason') == 'pending_limit'),
        first_history_buffer_guard=first(lambda r: r['reason'] == 'risk_buffer_guard'),
        historical_17475_189_copy1_1015=first(lambda r:
            pd.Timestamp(r['time']) == historical_time
            and '/17475/189~' in r['task_id'] and r['task_id'].endswith('copy1')))


def _boundary_shadow_starts(candidate, snapshot):
    """First NOW-plan boundary only; no physical executor or future input."""
    blocked = dict(snapshot.blocked_until)
    holds = [r.demand for r in snapshot.running if r.release_time > snapshot.now]
    started = []
    def start(spec):
        hold = (spec.cores_req, spec.gpu_req, spec.mem_req)
        if any(math.fsum([hold[d]]+[item[d] for item in holds]) > snapshot.capacity[d]
               for d in range(3)):
            return False
        holds.append(hold)
        started.append(spec.task_id)
        return True
    if not start(candidate):
        raise ValueError('Candidate does not fit archived decision state')
    ordered = sorted(snapshot.pending, key=lambda s: (s.sla_deadline, s.arrival_time, s.task_id))
    for spec in ordered:
        if spec.task_id == candidate.task_id or blocked.get(spec.task_id, snapshot.now) > snapshot.now:
            continue
        start(spec)
    return started


def compare_same_step(candidate, snapshot, step_events, candidate_rank):
    """Compare shadow NOW starts with real actions at/after candidate's turn.

    Current-boundary arrival count here is among pending jobs still in the
    candidate snapshot. Archive cases also report ALL arrivals at that boundary.
    """
    shadow = _boundary_shadow_starts(candidate, snapshot)
    remaining = [event for event in step_events if event['decision_rank'] >= candidate_rank]
    actual = [event['task_id'] for event in remaining if event['event'] == 'started']
    events_by_id = {event['task_id']: event['event'] for event in remaining}
    return dict(shadow_now_started_ids=shadow, actual_remaining_started_ids=actual,
        shadow_start_actual_deferred_ids=[key for key in shadow if events_by_id.get(key) == 'policy_deferred'],
        shadow_start_actual_capacity_waited_ids=[key for key in shadow if events_by_id.get(key) == 'capacity_waited'],
        actual_start_shadow_waited_ids=[key for key in actual if key not in shadow],
        current_boundary_arrival_count=sum(s.arrival_time == snapshot.now for s in snapshot.pending),
        same_step_start_set_mismatch=set(shadow) != set(actual))


def _read_jsonl(path):
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n', encoding='utf-8')


def _csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, sort_keys=True) if isinstance(value, (list, dict)) else value
                             for key, value in row.items()})


def _check_output(reference, output):
    reference, output = Path(reference).resolve(), Path(output).resolve()
    if output == reference or reference in output.parents:
        raise ValueError('Output must be outside read-only reference/campaign root')
    if 'E1-4A' not in output.parts:
        raise ValueError('Diagnostic output must stay under E1-4A')
    return reference, output


def _terminal(row):
    fields = ('task_id', 'arrival_time', 'start_time', 'finish_time', 'sla_deadline',
              'deadline_outcome', 'duration', 'cores_req', 'gpu_req', 'mem_req')
    return {key: row.get(key) for key in fields}


def _compact_case(value):
    """Bound long ID lists in representative exports, with explicit totals."""
    if not isinstance(value, dict):
        return value
    result = {}
    for key, item in value.items():
        if isinstance(item, list) and len(item) > 32:
            result[key] = item[:32]
            result[key+'_total_count'] = len(item)
            result[key+'_truncated'] = True
        else:
            result[key] = _compact_case(item)
    return result


class _ArchiveState:
    """Reconstruct candidate-observed state only from completed archive logs."""
    def __init__(self, folder):
        self.folder = Path(folder)
        self.diagnostics = _read_jsonl(self.folder/'diagnostics.jsonl')
        self.ledger = {row['task_id']: row for row in _read_jsonl(self.folder/'ledger.jsonl')}
        self.specs = {key: SimpleNamespace(task_id=key,
            arrival_time=pd.Timestamp(row['arrival_time']), duration=row['duration'],
            cores_req=row['cores_req'], gpu_req=row['gpu_req'], mem_req=row['mem_req'],
            sla_deadline=pd.Timestamp(row['sla_deadline'])) for key, row in self.ledger.items()}
        self.events = {}
        for event in _read_jsonl(self.folder/'policy_events.jsonl'):
            self.events.setdefault(event['time'], []).append(event)
        self.intervals = {row['time']: row for row in _read_jsonl(self.folder/'policy_intervals.jsonl')}
        config = json.loads((self.folder/'resolved_config.json').read_text(encoding='utf-8'))
        self.capacity = tuple(config['datacenter'][key] for key in ('total_cores', 'total_gpus', 'total_mem'))
        self.planner = ShadowPlanner(config['policy_params'].get('horizon_steps', 8),
                                     config['policy_params'].get('max_pending', 256))
        self.running_times = [(key, pd.Timestamp(row['start_time']), pd.Timestamp(row['finish_time']).ceil('15min'))
                              for key, row in self.ledger.items() if row.get('start_time') is not None]
        self.running_cache = {}
        arrived = sorted((spec.arrival_time.value, key) for key, spec in self.specs.items())
        self.arrival_times = [time for time, _ in arrived]
        self.arrival_work_prefix = [(0., 0., 0.)]
        for _, key in arrived:
            spec = self.specs[key]
            occupied = math.ceil(spec.duration/15)*15
            old = self.arrival_work_prefix[-1]
            self.arrival_work_prefix.append(tuple(total+demand*occupied for total, demand in
                zip(old, (spec.cores_req, spec.gpu_req, spec.mem_req))))

    def snapshot(self, row):
        now = pd.Timestamp(row['time'])
        events = self.events[row['time']]
        match = [event for event in events if event['task_id'] == row['task_id']]
        if len(match) != 1:
            raise ValueError('Archived candidate event must occur exactly once per step')
        rank = match[0]['decision_rank']
        prior = [event for event in events if event['decision_rank'] < rank]
        starts = {event['task_id'] for event in prior if event['event'] == 'started'}
        pending_ids = [key for key in self.intervals[row['time']]['ordered_waiting_ids'] if key not in starts]
        if row['time'] not in self.running_cache:
            self.running_cache[row['time']] = tuple(key for key, start, release in self.running_times
                                                    if start < now and release > now)
        running_ids = list(self.running_cache[row['time']])+sorted(starts)
        running = tuple(RunningTask(key,
            (self.specs[key].cores_req, self.specs[key].gpu_req, self.specs[key].mem_req),
            self.ledger[key]['finish_time'], self.specs[key].sla_deadline) for key in running_ids)
        blocked = tuple((event['task_id'], now+STEP) for event in prior if event['event'] == 'policy_deferred')
        snapshot = ResourceSnapshot(now, self.capacity, tuple(row['available']),
            tuple(self.specs[key] for key in pending_ids), running, blocked_until=blocked)
        if len(snapshot.pending) != row['pending_count'] or len(snapshot.running) != row['running_count']:
            raise ValueError('Reconstructed pending/running counts disagree with archive')
        return snapshot, events, rank

    def later_arrivals(self, time, endpoint):
        if endpoint is None:
            return dict(later_arrival_count_before_actual_start=None, later_arrival_resource_work_minutes=None)
        lower = bisect_right(self.arrival_times, pd.Timestamp(time).value)
        upper = bisect_right(self.arrival_times, pd.Timestamp(endpoint).value)
        upper = max(lower, upper)
        old, new = self.arrival_work_prefix[lower], self.arrival_work_prefix[upper]
        return dict(later_arrival_count_before_actual_start=upper-lower,
                    later_arrival_resource_work_minutes=[a-b for a, b in zip(new, old)])

    def evidence(self, row):
        snapshot, events, rank = self.snapshot(row)
        candidate = self.specs[row['task_id']]
        replay = self.planner.compare(candidate, snapshot)
        planner = row.get('planner')
        agrees = None if planner is None else all(
            getattr(replay, key) == (tuple(planner[key]) if isinstance(getattr(replay, key), tuple) else planner[key])
            for key in ('status', 'missed_now', 'missed_defer', 'unknown_now', 'unknown_defer', 'reason'))
        same = compare_same_step(candidate, snapshot, events, rank)
        same.update(candidate_rank=rank, pending_count=len(snapshot.pending),
            running_count=len(snapshot.running), current_boundary_all_arrival_count=
            bisect_right(self.arrival_times, snapshot.now.value)-bisect_right(self.arrival_times, snapshot.now.value-1),
            planner_replay_agrees=agrees,
            shadow_replay=dict(status=replay.status, reason=replay.reason,
                missed_now=list(replay.missed_now), missed_defer=list(replay.missed_defer),
                unknown_now=list(replay.unknown_now), unknown_defer=list(replay.unknown_defer)),
            same_step_later_policy_defer_count=sum(event['event']=='policy_deferred'
                for event in events if event['decision_rank'] > rank),
            same_step_later_capacity_wait_count=sum(event['event']=='capacity_waited'
                for event in events if event['decision_rank'] > rank))
        return same


def analyze_reference(reference_root, output):
    """Read old results-001; create only small descriptive tables under E1-4A."""
    reference, output = _check_output(reference_root, output)
    counts, pairs, risk_decisions, missed_tasks, cases, case_rows = [], [], [], [], {}, []
    summary = dict(schema='e1-4a-feasibility-audit-v1', reference_root=str(reference),
        interpretation='Counterfactual incremental miss IDs paired to actual Full outcomes; not probability calibration.',
        later_arrivals_definition='arrival > decision time and arrival <= affected task actual Full start; descriptive only',
        loads={})
    for load in ('load_1p50', 'load_2p00'):
        state = _ArchiveState(reference/load/'cars_full')
        risk_ledger = {row['task_id']: row for row in _read_jsonl(reference/load/'cars_risk_only'/'ledger.jsonl')}
        decision = decision_counts(state.diagnostics)
        paired = pair_known_risk(state.diagnostics, state.ledger)
        known = [row for row in state.diagnostics if row.get('planner', {}).get('status') == 'risk']
        evidence = {(row['time'], row['task_id']): state.evidence(row) for row in known}
        for row in paired:
            item = evidence[(row['time'], row['candidate_id'])]
            row.update(load=load, **state.later_arrivals(row['time'], row['actual_start_time']),
                same_step_start_set_mismatch=item['same_step_start_set_mismatch'],
                shadow_start_actual_deferred_count=len(item['shadow_start_actual_deferred_ids']),
                shadow_start_actual_capacity_waited_count=len(item['shadow_start_actual_capacity_waited_ids']),
                planner_replay_agrees=item['planner_replay_agrees'])
        for row in known:
            item = evidence[(row['time'], row['task_id'])]
            risk_decisions.append(dict(load=load, time=row['time'], task_id=row['task_id'],
                missed_now_count=len(row['planner']['missed_now']),
                missed_defer_count=len(row['planner']['missed_defer']),
                incremental_risk_ids=sorted(set(row['planner']['missed_defer'])-set(row['planner']['missed_now'])),
                actual_candidate_outcome=state.ledger[row['task_id']]['deadline_outcome'],
                candidate_rank=item['candidate_rank'], pending_count=item['pending_count'],
                running_count=item['running_count'],
                current_boundary_pending_arrival_count=item['current_boundary_arrival_count'],
                current_boundary_all_arrival_count=item['current_boundary_all_arrival_count'],
                planner_replay_agrees=item['planner_replay_agrees'],
                shadow_now_started_count=len(item['shadow_now_started_ids']),
                actual_remaining_started_count=len(item['actual_remaining_started_ids']),
                shadow_start_actual_deferred_count=len(item['shadow_start_actual_deferred_ids']),
                shadow_start_actual_capacity_waited_count=len(item['shadow_start_actual_capacity_waited_ids']),
                actual_start_shadow_waited_count=len(item['actual_start_shadow_waited_ids']),
                same_step_start_set_mismatch=item['same_step_start_set_mismatch'],
                same_step_later_policy_defer_count=item['same_step_later_policy_defer_count'],
                same_step_later_capacity_wait_count=item['same_step_later_capacity_wait_count']))
        missed = Counter(row['risk_task_id'] for row in paired if row['actual_miss'])
        for key, occurrences in sorted(missed.items()):
            missed_tasks.append(dict(load=load, task_id=key, predicted_risk_occurrences=occurrences,
                **{field: value for field, value in _terminal(state.ledger[key]).items() if field != 'task_id'},
                risk_only_outcome=risk_ledger[key]['deadline_outcome'], risk_only_start=risk_ledger[key].get('start_time')))
        selected = select_cases(state.diagnostics)
        cases[load] = {}
        for rule, row in selected.items():
            if row is None:
                cases[load][rule] = None
                case_rows.append(dict(load=load, rule=rule, selected=False))
                continue
            item = evidence.get((row['time'], row['task_id'])) or state.evidence(row)
            associated = next((key for key, actual in state.ledger.items()
                if actual.get('source_row') == 17470 and actual.get('source_task_index') == 35 and '~' not in key), None)
            cases[load][rule] = _compact_case(dict(diagnostic=row, full_candidate=_terminal(state.ledger[row['task_id']]),
                risk_only_candidate=_terminal(risk_ledger[row['task_id']]), evidence=item,
                associated_17470_35=None if associated is None else dict(
                    full=_terminal(state.ledger[associated]), risk_only=_terminal(risk_ledger[associated]))))
            case_rows.append(dict(load=load, rule=rule, selected=True, time=row['time'], task_id=row['task_id'],
                reason=row['reason'], full_outcome=state.ledger[row['task_id']]['deadline_outcome'],
                risk_only_outcome=risk_ledger[row['task_id']]['deadline_outcome'],
                same_step_start_set_mismatch=item['same_step_start_set_mismatch'],
                planner_replay_agrees=item['planner_replay_agrees']))
        outcomes = risk_outcome_counts(paired)
        summary['loads'][load] = dict(decision_counts=decision, risk_outcomes=outcomes,
            explicit_risk_decisions=len(known),
            planner_replay_disagreements=sum(item['planner_replay_agrees'] is False for item in evidence.values()),
            same_step_start_set_mismatch_decisions=sum(item['same_step_start_set_mismatch'] for item in evidence.values()),
            same_step_shadow_start_real_defer_decisions=sum(bool(item['shadow_start_actual_deferred_ids']) for item in evidence.values()),
            same_step_shadow_start_real_capacity_wait_decisions=sum(bool(item['shadow_start_actual_capacity_waited_ids']) for item in evidence.values()),
            predicted_actual_miss_with_later_arrivals_occurrences=sum(row['actual_miss'] and
                bool(row['later_arrival_count_before_actual_start']) for row in paired))
        counts.append(dict(load=load, **{key: value for key, value in decision.items() if not isinstance(value, dict)}, **outcomes))
        pairs.extend(paired)
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in [('decision_categories', counts), ('known_risk_outcomes', pairs),
                       ('known_risk_decisions', risk_decisions), ('predicted_risk_actual_miss', missed_tasks),
                       ('representative_cases', case_rows)]:
        _csv(output/f'{name}.csv', rows)
    _json(output/'summary.json', summary)
    _json(output/'representative_cases.json', cases)
    return summary


def _compare_ledgers(baseline, ledger):
    if set(baseline) != set(ledger):
        raise ValueError('All campaign policies must have the same task IDs')
    changes = []
    counts = Counter()
    deltas = []
    for key in sorted(baseline, key=lambda key: (baseline[key].get('arrival_time', ''), key)):
        base, actual = baseline[key], ledger[key]
        bmiss = base['deadline_outcome'] in ('missed', 'missed_unfinished')
        amiss = actual['deadline_outcome'] in ('missed', 'missed_unfinished')
        bmet, amet = base['deadline_outcome']=='met', actual['deadline_outcome']=='met'
        if bmiss and amiss:
            outcome_category = 'both_missed'
        elif bmet and amiss:
            outcome_category = 'risk_only_met_policy_missed'
        elif bmiss and amet:
            outcome_category = 'risk_only_missed_policy_met'
        elif bmet and amet:
            outcome_category = 'both_met'
        else:
            outcome_category = 'other_outcome_pair'
        counts[outcome_category] += 1
        delta = None
        if base.get('start_time') is not None and actual.get('start_time') is not None:
            delta = (pd.Timestamp(actual['start_time'])-pd.Timestamp(base['start_time'])).total_seconds()/60
            deltas.append(delta)
            start_category = 'earlier' if delta < 0 else 'later' if delta > 0 else 'same'
        else:
            start_category = 'start_unobserved'
        counts[start_category] += 1
        changes.append(dict(task_id=key, arrival_time=base.get('arrival_time'),
            outcome_category=outcome_category, start_category=start_category, delta_start_minutes=delta,
            risk_only_outcome=base['deadline_outcome'], policy_outcome=actual['deadline_outcome'],
            risk_only_start=base.get('start_time'), policy_start=actual.get('start_time')))
    paired = {key: counts[key] for key in ('both_missed', 'risk_only_met_policy_missed',
        'risk_only_missed_policy_met', 'both_met', 'other_outcome_pair')}
    start = dict(task_count=len(ledger), comparable_start_count=len(deltas),
        earlier_count=counts['earlier'], later_count=counts['later'], same_count=counts['same'],
        unobserved_start_count=counts['start_unobserved'],
        mean_delta_minutes=statistics.fmean(deltas) if deltas else None,
        median_delta_minutes=statistics.median(deltas) if deltas else None,
        min_delta_minutes=min(deltas) if deltas else None,
        max_delta_minutes=max(deltas) if deltas else None)
    rules = dict(policy_earlier_start=lambda r: r['start_category']=='earlier',
        policy_later_start=lambda r: r['start_category']=='later',
        risk_met_policy_miss=lambda r: r['outcome_category']=='risk_only_met_policy_missed',
        risk_miss_policy_met=lambda r: r['outcome_category']=='risk_only_missed_policy_met')
    representatives = {rule: [row for row in changes if predicate(row)][:3] for rule, predicate in rules.items()}
    return paired, start, representatives


def analyze_campaign(campaign, output):
    """Read five-policy driver outputs; export small common-horizon summaries."""
    campaign, output = _check_output(campaign, output)
    result = dict(schema='e1-4a-campaign-diagnosis-v1', campaign_root=str(campaign),
        baseline='cars_risk_only', policies=list(CAMPAIGN_POLICIES),
        representative_rule='First three within each category by (arrival_time, task_id); no full-ledger export',
        proportion_denominators=dict(
            guard_per_candidate='all policy decide() candidate checks; excludes engine capacity-wait turns',
            queue_limit_all_per_candidate='all policy decide() candidate checks',
            queue_limit_eligible_per_eligible='RiskOnly defer-eligible candidate checks before extra guards',
            eligibility='ablation risk_only_defer flag; historical reason and saved buffer/latest-start rule for controls',
            reported_status_proportions='all diagnostics planner.status reports, including queue-limit bypass',
            actual_status_proportions='actual bounded planner execution decisions; excludes queue-limit bypass'),
        loads={})
    common_rows, guard_rows, miss_rows, start_rows, representative_rows = [], [], [], [], []
    for load in ('load_1p50', 'load_2p00'):
        folder = campaign/load
        fairness = json.loads((folder/'fairness.json').read_text(encoding='utf-8'))
        with (folder/'comparison.csv').open(encoding='utf-8', newline='') as stream:
            common = [dict(load=load, **row) for row in csv.DictReader(stream)
                      if row['phase']=='common' and row['policy'] in CAMPAIGN_POLICIES]
        if len(common) != len(CAMPAIGN_POLICIES) or {row['policy'] for row in common} != set(CAMPAIGN_POLICIES):
            raise ValueError('Expected exactly one common comparison row per campaign policy')
        common_rows.extend(common)
        baseline = {row['task_id']: row for row in _read_jsonl(folder/'cars_risk_only'/'ledger.jsonl')}
        result['loads'][load] = dict(fairness=fairness, policies={})
        for policy in CAMPAIGN_POLICIES:
            ledger = {row['task_id']: row for row in _read_jsonl(folder/policy/'ledger.jsonl')}
            diagnostics = _read_jsonl(folder/policy/'diagnostics.jsonl')
            guards = decision_counts(diagnostics)
            status_fields = planner_status_counts(diagnostics)
            status_fields.update({key+'_per_candidate': guards[key]/guards['candidate_checks']
                if guards['candidate_checks'] else None for key in ('history_risk_buffer_guards',
                    'explicit_shadow_risk_guards', 'pending_cap_fallbacks', 'other_uncertain_fallbacks',
                    'cap_execute_guards', 'uncertain_risk_fallbacks', 'queue_limit_all_candidate_checks')})
            status_fields['queue_limit_eligible_per_eligible'] = guards['queue_limit_eligible_candidate_checks']/guards[
                'risk_only_eligible_candidate_checks'] if guards['risk_only_eligible_candidate_checks'] else None
            paired, starts, representatives = _compare_ledgers(baseline, ledger)
            result['loads'][load]['policies'][policy] = dict(guard_counts=guards,
                planner_status=status_fields, paired_misses=paired, start_differences=starts,
                representatives=representatives)
            guard_rows.append({'load': load, 'policy': policy, **guards, **status_fields})
            miss_rows.append(dict(load=load, policy=policy, **paired))
            start_rows.append(dict(load=load, policy=policy, **starts))
            for rule, rows in representatives.items():
                for row in rows:
                    representative_rows.append(dict(load=load, policy=policy, rule=rule, **row))
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in [('common_metrics', common_rows), ('campaign_guard_counts', guard_rows),
                       ('paired_miss_sets', miss_rows), ('start_difference_summary', start_rows),
                       ('campaign_representatives', representative_rows)]:
        _csv(output/f'{name}.csv', rows)
    _json(output/'campaign_summary.json', result)
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--reference-root')
    inputs.add_argument('--campaign')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = analyze_reference(args.reference_root, args.output) if args.reference_root else analyze_campaign(args.campaign, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
