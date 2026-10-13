"""Independent offline audit of the 36 adaptive candidate/filter trajectories.

No new policy rollouts, model loading, parameter fitting or Test access. Frozen
task ledgers and current action tapes reconstruct already-observed states.
Risk arithmetic is independently restated from frozen E1-6A v1, not imported
from the policy under audit. Hashes bind every read artifact before and after.
"""
import argparse
from collections import defaultdict
import csv
from functools import lru_cache
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
NOTES = Path(__file__).resolve().parent
GROUPS = ('plain', 'state', 'shield', 'combined')
SEEDS = (55101, 55102, 55103)
CAPACITY_KEYS = ('cores', 'gpus', 'mem')
DEMAND_KEYS = ('cores_req', 'gpu_req', 'mem_req')
ACTION_SLACK = (None, 60, 30, 15)
THRESHOLD = 161.075
STEP = pd.Timedelta(minutes=15)
COUNTERS = ('candidate_checks', 'solo_eligible', 'filter_veto', 'deferred')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def jsonl(path):
    with Path(path).open(encoding='utf8') as stream:
        return [json.loads(row) for row in stream if row.strip()]


def csv_rows(path):
    with Path(path).open(encoding='utf8', newline='') as stream:
        return list(csv.DictReader(stream))


@lru_cache(maxsize=131072)
def stamp(value):
    result = pd.Timestamp(value)
    require(result.tzinfo is not None, 'Explicit UTC timestamps required')
    return result.tz_convert('UTC')


def demand(spec):
    return tuple(spec[key] for key in DEMAND_KEYS)


@lru_cache(maxsize=131072)
def _deadline(arrival, duration, multiplier):
    return stamp(arrival) + pd.Timedelta(minutes=duration * multiplier)


def deadline(spec):
    return _deadline(spec['arrival_time'], spec['duration'], spec['sla_multiplier'])


@lru_cache(maxsize=131072)
def _grid_start(arrival, duration, multiplier):
    return (_deadline(arrival, duration, multiplier) - pd.Timedelta(minutes=duration)).floor('15min')


def grid_start(spec):
    return _grid_start(spec['arrival_time'], spec['duration'], spec['sla_multiplier'])


def order_key(spec):
    return deadline(spec), stamp(spec['arrival_time']), spec['task_id']


def near(actual, expected, field):
    require(math.isfinite(float(actual)) and math.isclose(float(actual), float(expected),
            rel_tol=1e-12, abs_tol=1e-9), field + ' differs')


def compare(actual, expected, field):
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), field + ' keys differ')
        for key in expected:
            compare(actual[key], expected[key], field + '.' + key)
    elif isinstance(expected, (tuple, list)):
        require(isinstance(actual, (tuple, list)) and len(actual) == len(expected), field + ' shape differs')
        for index, (a, e) in enumerate(zip(actual, expected)):
            compare(a, e, field + '.' + str(index))
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        if field.endswith(('buffer_minutes', 'cold_steps')):
            require(actual == expected, field + ' integer boundary differs')
        else:
            near(actual, expected, field)
    else:
        require(actual == expected, field + ' differs')


def history_pressure(now, start, capacity, observed_history):
    """Compute once per current step, preserving original open-left semantics."""
    now, start = stamp(now), stamp(start)
    observed_history = tuple(observed_history)
    require(all(start <= stamp(s['arrival_time']) <= now for s in observed_history),
            'History contains prehistory or future task')
    elapsed = min(120., (now - start).total_seconds() / 60)
    history = sorted((s for s in observed_history if now - pd.Timedelta(minutes=120) < stamp(s['arrival_time']) <= now),
                     key=lambda s: (stamp(s['arrival_time']), s['task_id']))
    rho = (tuple(math.fsum(demand(s)[k] * (15 * math.ceil(s['duration'] / 15)) for s in history)
                 / (capacity[k] * elapsed) for k in range(3)) if elapsed else (None, None, None))
    return rho, 2 if elapsed == 0 else 0


def risk_budget(spec, now, start, capacity, running, predecessors, observed_history, *, _observed_pressure=None):
    """Frozen weighted v1, using only supplied current/past observed task facts."""
    now, start = stamp(now), stamp(start)
    running, predecessors, observed_history = tuple(running), tuple(predecessors), tuple(observed_history)
    require(len(capacity) == 3 and all(c > 0 and math.isfinite(c) for c in capacity), 'Invalid capacity')
    require(start <= now, 'History starts after current time')
    require(all(stamp(s['arrival_time']) <= now for s in (spec,) + running + predecessors + observed_history),
            'Budget cannot include future tasks/history')
    require(all(stamp(s['arrival_time']) >= start for s in observed_history), 'No prehistory allowed')
    matched = tuple(q / c for q, c in zip(demand(spec), capacity))
    total = math.fsum(matched)
    weights = tuple(q / total for q in matched) if total else (0., 0., 0.)
    utilization = tuple(math.fsum(demand(s)[k] for s in running) / capacity[k] for k in range(3))
    urgent = [s for s in predecessors if s['task_id'] != spec['task_id'] and
              (grid_start(s) - now).total_seconds() / 60 <= 120]
    queue = tuple(math.fsum(demand(s)[k] * (15 * math.ceil(s['duration'] / 15)) for s in urgent)
                  / (capacity[k] * 120) for k in range(3))
    # Only audit_step supplies the private cache, after verifying all known IDs.
    # It contains no candidate/running-dependent aggregate and changes no math.
    rho, cold = (_observed_pressure if _observed_pressure is not None else
                 history_pressure(now, start, capacity, observed_history))
    pressure = tuple(u + q + (max(0., h - .5) if h is not None else 0.)
                     for u, q, h in zip(utilization, queue, rho))
    if total:
        score = math.fsum(w * x for w, x in zip(weights, pressure))
        buffer = 15 * math.ceil(cold + min(8, score))
        leaders = [r for r, q in zip(('cpu', 'gpu', 'mem'), matched) if q == max(matched)]
        dominant = leaders[0] if len(leaders) == 1 else 'tie'
    else:
        score, buffer, dominant = 0., 0, 'zero'
    slack = (grid_start(spec) - now).total_seconds() / 60
    return dict(weights=weights, utilization=utilization, urgent_pressure=queue, history_pressure=rho,
                resource_pressure=pressure, score=score, buffer_minutes=buffer,
                solo_slack_minutes=slack, safe_slack_minutes=slack - buffer,
                cold_steps=cold, dominant_resource=dominant)


def audit_step(*, specs, now, start, capacity, action, ci, risk_filter,
               pending_ids, running_ids, history_ids, events, diagnostics):
    """Capacity-first candidate population; update allocation before next check."""
    require(isinstance(action, int) and not isinstance(action, bool) and 0 <= action < 4, 'Invalid action')
    require(isinstance(risk_filter, bool), 'Invalid filter flag')
    now = stamp(now)
    require(all(task_id in specs for task_id in tuple(pending_ids) + tuple(running_ids) + tuple(history_ids)),
            'Unknown observed task ID')
    require(all(stamp(specs[task_id]['arrival_time']) <= now for task_id in
                tuple(pending_ids) + tuple(running_ids) + tuple(history_ids)), 'Observed state includes future task')
    require(len(pending_ids) == len(set(pending_ids)) and len(running_ids) == len(set(running_ids)) and
            not set(pending_ids).intersection(running_ids), 'Duplicate/current task ownership differs')
    ordered = sorted(pending_ids, key=lambda task_id: order_key(specs[task_id]))
    require(len(events) == len(ordered) and [r['task_id'] for r in events] == ordered and
            [r['decision_rank'] for r in events] == list(range(len(ordered))), 'Whole-queue EDF event order differs')
    result = dict(candidate_checks=0, solo_eligible=0, filter_veto=0, deferred=0,
                  capacity_waiting=0, started=0, same_step_updated_budget_checks=0)
    running = list(running_ids); pending = set(ordered); index = 0; started_this_step = False
    observed_pressure = history_pressure(now, start, capacity, [specs[k] for k in history_ids])
    for rank, task_id in enumerate(ordered):
        event = events[rank]; spec = specs[task_id]
        used = tuple(math.fsum(demand(specs[k])[dimension] for k in running) for dimension in range(3))
        available = tuple(c - u for c, u in zip(capacity, used))
        can_fit = all(math.fsum([demand(specs[k])[dimension] for k in running] + [demand(spec)[dimension]]) <= limit
                      for dimension, limit in enumerate(capacity))
        require(stamp(event['time']) == now and event['can_fit_at_decision'] is can_fit,
                'Capacity event flag/time differs')
        for name, value in zip(CAPACITY_KEYS, available):
            near(event['available_before'][name], value, 'event available_before.' + name)
        if 'ci_g_per_kwh' in event:
            near(event['ci_g_per_kwh'], ci, 'event current CI')
            near(event['threshold_g_per_kwh'], THRESHOLD, 'event CI threshold')
        if not can_fit:
            require(event['event'] == 'capacity_waited', 'Capacity waiting must precede policy checks')
            result['capacity_waiting'] += 1
            continue
        require(index < len(diagnostics), 'Missing can-fit candidate diagnostic')
        row = diagnostics[index]; index += 1
        require(row['task_id'] == task_id and stamp(row['time']) == now, 'Candidate diagnostic order/time differs')
        threshold = ACTION_SLACK[action]
        slack = (grid_start(spec) - now).total_seconds() / 60
        eligible = action != 0 and ci > THRESHOLD and slack >= threshold
        budget = None; veto = False
        if action == 0:
            reason = 'adaptive_edf_execute'
        elif ci <= THRESHOLD:
            reason = 'low_ci_execute'
        elif not eligible:
            reason = 'below_action_slack'
        else:
            if risk_filter:
                predecessors = [specs[k] for k in ordered[:rank] if k in pending]
                budget = risk_budget(spec, now, start, capacity, [specs[k] for k in running],
                                     predecessors, (), _observed_pressure=observed_pressure)
                veto = budget['safe_slack_minutes'] < threshold
                if started_this_step:
                    result['same_step_updated_budget_checks'] += 1
            reason = 'adaptive_risk_veto' if veto else 'adaptive_high_ci_defer'
        defer = eligible and not veto
        require(row['action'] == action and row['slack_threshold_minutes'] == threshold and
                row['risk_filter'] is risk_filter and row['solo_eligible'] is eligible and
                row['filter_veto'] is veto and row['defer'] is defer and row['reason'] == reason,
                'Action/eligibility/veto/defer semantics differ')
        near(row['ci_g_per_kwh'], ci, 'diagnostic current CI')
        near(row['solo_slack_minutes'], slack, 'diagnostic current grid slack')
        compare(row['available'], available, 'diagnostic available')
        compare(row['budget'], budget, 'budget')
        require(row['pending_count'] == len(pending) and row['running_count'] == len(running),
                'Same-step observed queue state differs')
        require(event['event'] == ('policy_deferred' if defer else 'started'),
                'Diagnostic does not match real defer/start')
        if 'reason' in event:
            require(event['reason'] == reason, 'Policy/diagnostic reason differs')
        result['candidate_checks'] += 1; result['solo_eligible'] += eligible
        result['filter_veto'] += veto; result['deferred'] += defer
        if not defer:
            pending.remove(task_id); running.append(task_id); started_this_step = True; result['started'] += 1
    require(index == len(diagnostics), 'Extra capacity-failed/noncandidate diagnostic')
    return result


def audit_method(bundle, folder, common_row):
    specs = {s.task_id: s.to_dict() for s in bundle.workload.tasks}
    ledger = {r['task_id']: r for r in jsonl(folder / 'ledger.jsonl')}
    require(set(ledger) == set(specs), 'Frozen task set differs')
    require(all({k: row[k] for k in specs[task_id]} == specs[task_id] for task_id, row in ledger.items()),
            'Frozen task features changed')
    require(all(row['state'] == 'completed' for row in ledger.values()), 'Full Drain required')
    actions = csv_rows(folder / 'actions.csv'); events = jsonl(folder / 'policy_events.jsonl')
    intervals = csv_rows(folder / 'intervals.csv'); detail = jsonl(folder / 'policy_intervals.jsonl')
    by_time = defaultdict(list); diagnostics = defaultdict(list)
    for row in events:
        by_time[stamp(row['time'])].append(row)
    with gzip.open(folder / 'policy-diagnostics.jsonl.gz', 'rt', encoding='utf8') as stream:
        last = bundle.workload.start
        for line in stream:
            row = json.loads(line); now = stamp(row['time'])
            require(now >= last, 'Diagnostic times are not monotonic')
            last = now; diagnostics[now].append(row)
    own_end = stamp(common_row['own_observation_end']); common_end = stamp(common_row['observation_end'])
    require(len(actions) == (own_end - bundle.workload.start) // STEP and
            len(intervals) == len(detail) == (common_end - bundle.workload.start) // STEP,
            'Actor/common observation coverage differs')
    require(own_end <= common_end <= bundle.environment.end, 'Observation endpoint outside real coverage')
    samples = {s.time: s for s in bundle.environment.samples}
    starts = {task_id: stamp(row['start_time']) for task_id, row in ledger.items()}
    releases = {task_id: stamp(row['finish_time']).ceil('15min') for task_id, row in ledger.items()}
    totals = dict(candidate_checks=0, solo_eligible=0, filter_veto=0, deferred=0,
                  capacity_waiting=0, started=0, same_step_updated_budget_checks=0)
    seen = set()
    for index, action_row in enumerate(actions):
        now = bundle.workload.start + index * STEP
        require(stamp(action_row['time']) == now and stamp(detail[index]['time']) == now and
                stamp(intervals[index]['interval_start']) == now, 'Current step boundaries differ')
        near(action_row['ci_g_per_kwh'], samples[now].ci_g_per_kwh, 'Current frozen CI')
        near(intervals[index]['ci_g_per_kwh'], samples[now].ci_g_per_kwh, 'Interval frozen CI')
        # Ledger start/release times are used only for offline reconstruction;
        # no prediction or unarrived task is passed into budget arithmetic.
        pending = [task_id for task_id, spec in specs.items()
                   if stamp(spec['arrival_time']) <= now <= starts[task_id]]
        running = sorted((task_id for task_id in specs if starts[task_id] < now < releases[task_id]),
                         key=lambda k: (starts[k], order_key(specs[k])))
        history = [task_id for task_id, spec in specs.items() if stamp(spec['arrival_time']) <= now]
        require(set(detail[index]['decision_pending_ids']) == set(pending), 'Current known pending population differs')
        require(int(action_row['pending']) == len(pending) and int(action_row['running']) == len(running),
                'Current observed queue counts differ')
        risk_filter = common_row['group'] in ('shield', 'combined')
        counted = audit_step(specs=specs, now=now, start=bundle.workload.start, capacity=bundle.capacity,
            action=int(action_row['action']), ci=samples[now].ci_g_per_kwh, risk_filter=risk_filter,
            pending_ids=pending, running_ids=running, history_ids=history,
            events=by_time[now], diagnostics=diagnostics[now])
        for field in COUNTERS:
            require(int(action_row[field]) == counted[field], 'Per-step saved filter denominator differs: ' + field)
        for key in totals:
            totals[key] += counted[key]
        seen.add(now)
    require(not set(diagnostics).difference(seen), 'Diagnostic exists outside actor own-Drain scope')
    require(not any(rows for time, rows in by_time.items() if time >= own_end), 'Idle tail contains policy candidates')
    for key in COUNTERS:
        require(int(common_row[key]) == totals[key], 'Common saved filter denominator differs: ' + key)
    require(int(common_row['policy_defer_intervals']) == totals['deferred'] and
            int(common_row['capacity_wait_intervals']) == totals['capacity_waiting'], 'Real wait totals differ')
    require(totals['started'] == len(specs), 'Every cohort task must start exactly once')
    return dict(method=folder.name, group=common_row['group'], seed=int(common_row['seed']),
        factor=bundle.factor, task_count=len(specs), actor_intervals=len(actions), **totals,
        veto_fraction=totals['filter_veto'] / totals['solo_eligible'] if totals['solo_eligible'] else None,
        defer_fraction=totals['deferred'] / totals['solo_eligible'] if totals['solo_eligible'] else None,
        denominator='can-fit solo-eligible candidate opportunities, excludes capacity failures',
        current_ci_action_solo_safe_exact=True, frozen_v1_budget_verified=True,
        same_step_allocation_updates_verified=True, no_candidate_diagnostics_during_idle=True)


def run(campaign, output):
    campaign, output = Path(campaign), Path(output)
    require(not output.exists(), 'Fresh audit output required; never overwrite')
    manifest = read_json(campaign / 'campaign.json'); receipt = read_json(campaign / 'campaign-receipt.json')
    require(receipt.get('passed') is True and manifest.get('mode') == 'evaluate' and
            manifest.get('formal_test_evaluated') is False and manifest.get('validation_training') is False and
            manifest.get('total_core_results') == 57, 'Completed development Validation required; Test sealed')
    require(len(manifest['inputs']) == 3 and {r['factor'] for r in manifest['inputs']} == {1., 1.5, 2.} and
            all(r['split'] == 'validation' for r in manifest['inputs']), 'Development roles/factors differ; Test sealed')
    require(receipt['campaign_manifest_sha256'] == digest(campaign / 'campaign.json'), 'Campaign identity hash differs')
    sys.path.insert(0, str(ROOT))
    from experiments.e1.rl_training_utils import load_bundle
    hashes = {str(campaign / name): digest(campaign / name) for name in ('campaign.json', 'campaign-receipt.json')}
    jobs = []
    for record in manifest['inputs']:
        path = Path(record['path'])
        for name, expected in record['files'].items():
            require(digest(path / name) == expected, 'Frozen input hash differs')
            hashes[str(path / name)] = expected
        bundle = load_bundle(path, allowed_split='validation')
        require(bundle.factor == record['factor'] and bundle.workload.sha256 == record['workload_sha256'] and
                bundle.environment.sha256 == record['environment_sha256'] and
                bundle.parameters['threshold_g_per_kwh'] == THRESHOLD, 'Frozen Validation identity/threshold differs')
        folder = campaign / ('load-' + record['label'])
        output_hashes = read_json(folder / 'output_hashes.json')
        for name in ('output_hashes.json', 'common_summary.csv', 'fairness.json'):
            hashes[str(folder / name)] = digest(folder / name)
            if name != 'output_hashes.json':
                require(output_hashes[name] == hashes[str(folder / name)], 'Saved evaluation summary hash differs')
        fairness = read_json(folder / 'fairness.json')
        require(fairness['all_tasks_completed'] is True and fairness['equal_resource_hours'] is True and
                fairness['idle_included'] is True and fairness['formal_test_evaluated'] is False and
                fairness['policy_future_ci_access'] is False, 'Evaluation fairness/online role differs')
        rows = csv_rows(folder / 'common_summary.csv')
        adaptive = [row for row in rows if row['group'] in GROUPS]
        require(len(rows) == 19 and len(adaptive) == 12 and
                {(r['group'], int(r['seed'])) for r in adaptive} == {(g, s) for g in GROUPS for s in SEEDS},
                'Nineteen-method/twelve-adaptive matrix differs')
        for row in adaptive:
            require(row['split'] == 'validation' and float(row['factor']) == bundle.factor and
                    row['phase'] == 'common' and row['observation_end'] == fairness['common_end'],
                    'Common development observation scope differs')
            method = folder / row['policy']
            for name in ('ledger.jsonl', 'policy_events.jsonl', 'policy_intervals.jsonl',
                         'actions.csv', 'intervals.csv', 'policy-diagnostics.jsonl.gz'):
                expected = output_hashes[row['policy'] + '/' + name]
                require(digest(method / name) == expected, 'Saved trajectory hash differs')
                hashes[str(method / name)] = expected
            jobs.append((bundle, method, row))
    require(len(jobs) == 36, 'Exactly 36 adaptive offline audits required')
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'audit-plan.json', dict(schema='e1-6b-independent-filter-audit-v1',
        methods=36, candidates='capacity-first; all pending sorted EDF', budget_future_task_access=False,
        offline_posthoc_ledger_reconstruction=True, test_access=False,
        arithmetic='frozen weighted v1 independently restated; u+urgent+max(0,rho-.5); cold2,K8,ceil15',
        input_hashes=hashes, driver_sha256=digest(__file__)))
    results = []
    try:
        for bundle, method, row in jobs:
            result = audit_method(bundle, method, row); results.append(result)
            print(json.dumps(dict(method=method.name, factor=bundle.factor, **{k: result[k] for k in COUNTERS})), flush=True)
        require(all(digest(path) == expected for path, expected in hashes.items()), 'Read-only source/input artifact changed')
        with (output / 'method-filter-audit.csv').open('x', encoding='utf8', newline='') as stream:
            writer = csv.DictWriter(stream, list(results[0])); writer.writeheader(); writer.writerows(results)
        final = dict(passed=True, methods=36, no_rollouts_or_training=True, test_access=False,
            all_read_files_unchanged=True, all_candidate_denominators_exact=True,
            frozen_v1_budgets_independently_verified=True,
            files={p.name: digest(p) for p in output.iterdir() if p.is_file()})
        write_json(output / 'audit-receipt.json', final)
        print(json.dumps(final, sort_keys=True), flush=True)
        return final
    except Exception as exc:
        write_json(output / 'failure.json', dict(error_type=type(exc).__name__, message=str(exc),
                                               completed_methods=results, passed=False))
        raise


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n', encoding='utf8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', default=str(NOTES / 'validation-campaign-001'))
    parser.add_argument('--output', default=str(NOTES / 'filter-audit-001'))
    args = parser.parse_args()
    run(args.campaign, args.output)


if __name__ == '__main__':
    main()
