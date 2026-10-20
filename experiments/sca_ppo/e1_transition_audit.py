"""Post hoc replay of saved E1 calendars; no input loaders or environment steps."""
import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from experiments.platform_v1.data import Task
from experiments.platform_v2.inventory import load_config
from experiments.sca_ppo.edf_calendar import Snapshot, Running, simulate, grid_release, edf_key


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "research_notes/sca-ppo/e1-001"
HERE = OUT / "analysis-checks"
EPS = 1e-9


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(a, b):
    return math.isclose(a, b, rel_tol=0, abs_tol=EPS)


def calendar_times(plan):
    return {r.task_id: (r.start, r.finish, r.release) for r in plan.records}


def write_csv(name, rows, fields):
    with (HERE / name).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    global OUT, HERE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(OUT))
    args = parser.parse_args()
    OUT = Path(args.output).resolve()
    HERE = OUT / "analysis-checks"
    HERE.mkdir(parents=True, exist_ok=True)
    execution = json.loads((OUT / 'execution.json').read_text(encoding='utf-8'))
    cfg = load_config()
    assert execution['status'] == 'completed'
    inputs = {'execution.json': sha(OUT / 'execution.json')}
    for relative in ('experiments/sca_ppo/edf_calendar.py',):
        assert sha(ROOT / relative) == execution['source_sha256'][relative], relative
    groups, events, classifications, blockers = [], [], [], []
    for group in sorted(p for p in OUT.iterdir() if p.is_dir() and p.name.startswith('train-')):
        loaded = {}
        for relative in ('A/result.json', 'B/result.json', 'C/result.json', 'C/decisions.jsonl'):
            path = group / relative
            identity = f'{group.name}/{relative}'
            inputs[identity] = sha(path)
            assert inputs[identity] == execution['evidence_sha256'][identity], identity
            if path.suffix == '.json':
                loaded[relative] = json.loads(path.read_text(encoding='utf-8'))
            else:
                loaded[relative] = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
        result = loaded['C/result.json']
        capacity = tuple(cfg['capacities'][result['summary']['capacity']])
        rows = {r['task_id']: r for r in result['tasks']}
        assert len(rows) == len(result['tasks'])
        assert all(r['start_minute'] is not None and r['finish_minute'] is not None for r in rows.values())
        tasks = {identity: Task(identity, r['arrival_minute'], r['duration_minutes'],
                               (r['cpu'], r['gpu'], r['mem_gb']), 'SAVED_TRACE',
                               r['source_row'], r['source_task_index']) for identity, r in rows.items()}
        for identity, r in rows.items():
            assert close(r['deadline_minute'], r['arrival_minute'] + 1.5 * r['duration_minutes'])
            assert close(r['finish_minute'], r['start_minute'] + r['duration_minutes'])
        a_miss = {r['task_id'] for r in loaded['A/result.json']['tasks'] if r['deadline_missed']}
        c_miss = {r['task_id'] for r in result['tasks'] if r['deadline_missed']}
        new_miss = c_miss - a_miss
        decisions = loaded['C/decisions.jsonl']
        assert len(decisions) == 288
        first_reference_on_time, ever_accepted_on_time, first_loss = {}, set(), {}
        recovered_after_first_loss = set()
        steps = {r['minute']: r for r in result['steps']}
        group_events, group_classes = [], []
        boundary_counts = Counter()
        previous = None

        def snapshot_at(now, remove_new=False):
            # Actual future start/finish values never enter a pending Task.
            known = [(identity, r) for identity, r in rows.items() if r['arrival_minute'] <= now]
            pending = tuple(tasks[identity] for identity, r in known
                            if r['start_minute'] >= now
                            and (not remove_new or r['arrival_minute'] < now))
            active = [(identity, r) for identity, r in known
                      if r['start_minute'] < now and grid_release(r['finish_minute']) > now]
            active.sort(key=lambda pair: (pair[1]['start_minute'], *edf_key(tasks[pair[0]], 1.5)))
            running = tuple(Running(tasks[identity], r['finish_minute']) for identity, r in active)
            return Snapshot(now, capacity, 1.5, pending, running)

        for index, decision in enumerate(decisions):
            now = decision['minute']
            assert now == index * 15
            snapshot = snapshot_at(now)
            reference = simulate(snapshot)
            accepted = simulate(snapshot, frozenset(decision['approved']))
            known_ids = set(reference.by_id)
            if tuple(decision['reference_start_ids']) != reference.first_slot.started:
                blockers.append(dict(case=group.name, minute=now, code='REFERENCE_START_REPLAY_MISMATCH'))
            if len(reference.on_time_ids) != decision['reference_on_time_count']:
                blockers.append(dict(case=group.name, minute=now, code='REFERENCE_ON_TIME_COUNT_MISMATCH'))
            for field, actual in (('actual_started', accepted.first_slot.started),
                                  ('effective_policy_wait', accepted.first_slot.policy_wait),
                                  ('capacity_wait', accepted.first_slot.capacity_wait)):
                if tuple(decision[field]) != actual:
                    blockers.append(dict(case=group.name, minute=now, code=field.upper() + '_REPLAY_MISMATCH'))
            if any(not close(predicted, steps[now]['used_' + axis])
                   for predicted, axis in zip(accepted.first_slot.used, ('cpu', 'gpu', 'mem'))):
                blockers.append(dict(case=group.name, minute=now, code='USED_RESOURCES_REPLAY_MISMATCH'))
            if not close(reference.total_lateness(reference.late_ids), decision['reference_late_total']):
                blockers.append(dict(case=group.name, minute=now, code='REFERENCE_LATENESS_REPLAY_MISMATCH'))
            if not close(accepted.total_lateness(reference.late_ids), decision['final_late_total']):
                blockers.append(dict(case=group.name, minute=now, code='ACCEPTED_LATENESS_REPLAY_MISMATCH'))
            lost_within_decision = reference.on_time_ids - accepted.on_time_ids
            if lost_within_decision or set(decision['final_lost_on_time_ids']) != lost_within_decision:
                blockers.append(dict(case=group.name, minute=now, code='DDF_LOCAL_ON_TIME_PROTECTION_FAILED',
                                     task_ids=sorted(lost_within_decision)))
            for identity in known_ids:
                first_reference_on_time.setdefault(identity, identity in reference.on_time_ids)
            ever_accepted_on_time.update(accepted.on_time_ids)

            if previous is not None:
                boundary_counts['all'] += 1
                previous_minute, previous_plan = previous
                previous_ids = set(previous_plan.by_id)
                common = previous_ids & known_ids
                arrivals = tuple(identity for identity, r in rows.items() if r['arrival_minute'] == now)
                boundary_counts['with_arrivals' if arrivals else 'without_arrivals'] += 1
                old_only = simulate(snapshot_at(now, remove_new=True)) if arrivals else reference
                if set(old_only.by_id) != common:
                    blockers.append(dict(case=group.name, minute=now, code='OLD_ONLY_KNOWN_SET_MISMATCH'))
                old_times, prior_times = calendar_times(old_only), calendar_times(previous_plan)
                changed = [identity for identity in common
                           if any(not close(a, b) for a, b in zip(old_times[identity], prior_times[identity]))]
                if changed:
                    blockers.append(dict(case=group.name, minute=now, code='OLD_ONLY_SUFFIX_MISMATCH',
                                         task_ids=sorted(changed)))
                else:
                    boundary_counts['old_only_suffix_pass'] += 1
                    if not arrivals:
                        boundary_counts['no_arrival_suffix_pass'] += 1
                lost = (previous_plan.on_time_ids & known_ids) - reference.on_time_ids
                if lost and not arrivals:
                    blockers.append(dict(case=group.name, minute=now, code='FORECAST_LOSS_WITHOUT_NEW_ARRIVALS',
                                         task_ids=sorted(lost)))
                if lost:
                    boundary_counts['with_forecast_loss'] += 1
                for identity in sorted(lost):
                    old = previous_plan.by_id[identity]
                    new = reference.by_id[identity]
                    event = dict(fragment=result['summary']['fragment'], capacity=result['summary']['capacity'],
                        task_id=identity, previous_minute=previous_minute, boundary_minute=now,
                        new_arrival_count=len(arrivals), previous_accepted_start=old.start,
                        previous_accepted_finish=old.finish, new_reference_start=new.start,
                        new_reference_finish=new.finish, old_only_finish=old_only.by_id[identity].finish,
                        deadline_minute=new.deadline, previous_approved_count=len(decisions[index-1]['approved']),
                        actual_c_finish=rows[identity]['finish_minute'], actual_c_missed=identity in c_miss,
                        actual_a_missed=identity in a_miss, c_new_miss_vs_a=identity in new_miss,
                        first_forecast_loss=identity not in first_loss)
                    group_events.append(event)
                    first_loss.setdefault(identity, event)
            recovered_after_first_loss.update(set(first_loss) & accepted.on_time_ids)
            previous = now, accepted

        for identity in sorted(new_miss):
            loss = first_loss.get(identity)
            if identity in ever_accepted_on_time and loss is not None and loss['new_arrival_count'] > 0:
                label = 'earlier_accepted_on_time_then_lost_at_known_set_augmentation'
            elif identity not in ever_accepted_on_time and first_reference_on_time.get(identity) is False:
                label = 'initial_reference_already_late_never_accepted_on_time'
            else:
                label = 'other_uncertain'
            r = rows[identity]
            group_classes.append(dict(fragment=result['summary']['fragment'], capacity=result['summary']['capacity'],
                task_id=identity, classification=label, arrival_minute=r['arrival_minute'],
                deadline_minute=r['deadline_minute'], actual_c_start=r['start_minute'],
                actual_c_finish=r['finish_minute'], first_reference_on_time=first_reference_on_time.get(identity),
                ever_accepted_on_time=identity in ever_accepted_on_time,
                recovered_after_first_loss=identity in recovered_after_first_loss,
                first_loss_boundary=None if loss is None else loss['boundary_minute'],
                first_loss_arrival_count=None if loss is None else loss['new_arrival_count']))
        events.extend(group_events)
        classifications.extend(group_classes)
        groups.append(dict(fragment=result['summary']['fragment'], capacity=result['summary']['capacity'],
            task_count=len(rows), a_deadline_missed=len(a_miss), c_deadline_missed=len(c_miss),
            b_deadline_missed=loaded['B/result.json']['summary']['deadline_missed'],
            c_new_miss_vs_a=len(new_miss), a_miss_prevented_by_c=len(a_miss-c_miss),
            transition_events=len(group_events), unique_forecast_loss_tasks=len(first_loss),
            forecast_loss_tasks_later_recovered=len(recovered_after_first_loss),
            new_miss_classification=dict(Counter(r['classification'] for r in group_classes)),
            boundary_checks=dict(boundary_counts)))
        print(json.dumps(groups[-1], ensure_ascii=False), flush=True)

    assert len(groups) == 9
    totals = {key: sum(g[key] for g in groups) for key in
              ('task_count', 'a_deadline_missed', 'b_deadline_missed', 'c_deadline_missed',
               'c_new_miss_vs_a', 'a_miss_prevented_by_c', 'transition_events', 'unique_forecast_loss_tasks',
               'forecast_loss_tasks_later_recovered')}
    totals['new_miss_classification'] = dict(Counter(r['classification'] for r in classifications))
    totals['boundary_checks'] = dict(sum((Counter(g['boundary_checks']) for g in groups), Counter()))
    selected = [e for e in events if e['c_new_miss_vs_a'] and e['first_forecast_loss']]
    examples = []
    for event in selected:
        if len(examples) == 2:
            break
        if not examples or event['fragment'] != examples[0]['fragment'] or event['capacity'] != examples[0]['capacity']:
            examples.append(event)
    summary = dict(schema='sca-ppo-e1-posthoc-forecast-transition-audit-v1',
        status='BLOCKER' if blockers else 'PASS',
        scope='Deterministic diagnosis of saved C trajectories, no new performance runs or policy changes',
        reconstruction='At t include arrival<=t only; pending actualstart>=t; running actualstart<t and grid_release(actualfinish)>t. Pending future starts/finishes never supplied to calendar.',
        interpretation='Local reference protection does not imply global or cross-policy per-ID/DMR dominance. Arrival-pruned next-state replay isolates known-set augmentation at a boundary; no single prior defer is identified as a cause.',
        totals=totals, groups=groups, compact_examples=examples, blockers=blockers,
        source_and_config_sha256={relative:sha(ROOT / relative) for relative in
                                 ('experiments/sca_ppo/edf_calendar.py', 'experiments/platform_v2/config.json')},
        saved_inputs_sha256=inputs, script_sha256=sha(Path(__file__).resolve()))
    write_csv('forecast-loss-transitions.csv', events, list(events[0]) if events else [])
    write_csv('c-new-miss-classification.csv', classifications,
              list(classifications[0]) if classifications else [])
    (HERE / 'forecast-transition-audit.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    hashes = {p.name:sha(p) for p in sorted(HERE.iterdir()) if p.is_file() and p.name != 'audit-evidence-sha256.json'}
    (HERE / 'audit-evidence-sha256.json').write_text(json.dumps(hashes, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status':summary['status'], 'totals':totals, 'compact_examples':examples,
                      'blocker_count':len(blockers), 'sha256':hashes}, ensure_ascii=False, indent=2), flush=True)
    if blockers:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
