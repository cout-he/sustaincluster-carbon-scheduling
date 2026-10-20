"""E2 admission of the sixteen authorized Train base windows; no scheduling.

The split rule is fixed before any audit: January--March training, April--May
holdout. Overlapping 72-hour inputs or shared original job/start/end identities
join components; their chronological interval closure stays together. A
component touching April/May belongs wholly to holdout. No task is deduplicated.
"""
import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
import itertools
import json
from pathlib import Path

from experiments.platform_v1.data import canonical_sha, file_sha, load_train, stamp
from experiments.platform_v2.inventory import build_inventory, load_config

ROOT = Path(__file__).resolve().parents[2]
AUTHORIZED_SCENARIOS = Path('D:/A-myprofile/SustainCluster/publication/'
    'sustaincluster-carbon-scheduling-001/research_notes/joint_v2/scenarios-001/scenarios')
DATES = ('20230102', '20230103', '20230104', '20230105',
         '20230201', '20230202', '20230203', '20230204',
         '20230301', '20230302', '20230303', '20230304',
         '20230403', '20230404', '20230501', '20230503')
FRAGMENTS = tuple('train-'+day+'T0500-r1-s1.5' for day in DATES)
CONFIG = ROOT/'experiments/platform_v2/config.json'
SPLIT_RULE = dict(training_months=['2023-01', '2023-02', '2023-03'],
    holdout_months=['2023-04', '2023-05'],
    grouping='72-hour environment overlap OR shared original job/start/end; chronological interval closure',
    boundary_component_assignment='entire component to holdout if it touches April or May',
    metric_selection=False, task_deduplication=False)


def task_identity(scenario):
    return canonical_sha([(t.task_id, t.arrival_minute, t.duration_minutes, t.demand)
                          for t in scenario.tasks])


def load_authorized(root, expected=None):
    """Reject path/ID/role/test_access before any workload/environment access."""
    root = Path(root).resolve()
    if root.parent != AUTHORIZED_SCENARIOS.resolve() or root.name not in FRAGMENTS:
        raise ValueError('Only authorized E2 Train base window paths may be loaded')
    manifest = json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    if (manifest.get('role') != 'train' or manifest.get('test_access') is not False
            or manifest.get('scenario_id') != root.name):
        raise ValueError('Only authorized Train role/test_access and matching scenario identity may be loaded')
    scenario = load_train(root, hours=72, expected=expected)
    id_start = datetime.strptime(root.name[6:19], '%Y%m%dT%H%M').replace(tzinfo=timezone.utc)
    if stamp(scenario.provenance['start']) != id_start:
        raise ValueError('Scenario ID date differs from the workload arrival start')
    if expected is not None:
        if (len(scenario.tasks) != expected['tasks']
                or task_identity(scenario) != expected['task_identity_sha256']):
            raise ValueError('Frozen E2 task identity/count changed')
        provenance = {k: v for k, v in expected.items()
                      if k not in ('tasks', 'task_identity_sha256')}
        if scenario.provenance != provenance:
            raise ValueError('Frozen E2 input provenance changed')
    return scenario


def _repeats(values):
    counts = Counter(values)
    return dict(occurrences=sum(counts.values()), unique=len(counts),
        duplicate_excess=sum(n-1 for n in counts.values()),
        repeat_groups=sum(n > 1 for n in counts.values()),
        repeated_occurrences=sum(n for n in counts.values() if n > 1),
        maximum_multiplicity=max(counts.values(), default=0))


def _environment(scenario, environment):
    start = stamp(scenario.provenance['start']); end = start+timedelta(hours=72)
    if stamp(environment['start']) > start or stamp(environment['end']) < end:
        raise ValueError('Environmental snapshot does not declare full 72-hour coverage')
    samples = [s for s in environment['samples'] if start <= stamp(s['time']) < end]
    times = [stamp(s['time']) for s in samples]
    if times != [start+timedelta(minutes=15*i) for i in range(288)]:
        raise ValueError('Incomplete or duplicate 72-hour quarter-hour environmental coverage')
    by_time = {stamp(s['time']): s for s in samples}
    for index, observation in enumerate(scenario.observations):
        minute = index*15; boundary = minute//60*60
        source = by_time[start+timedelta(minutes=boundary)]
        if (observation.minute, observation.observed_minute, observation.ci_g_per_kwh,
                observation.outdoor_c) != (minute, boundary, source['ci_g_per_kwh'], source['temperature_c']):
            raise ValueError('Environmental observation violates causal previous-hour hold')
    if len(scenario.observations) != 288:
        raise ValueError('Expected 288 loaded environmental observations')
    return dict(start=start.isoformat(), end_exclusive=end.isoformat(),
                responsibility_hours=72, samples_in_72_hours=288,
                unique_contiguous_quarter_hours=True, causal_previous_hour_hold=True), by_time


def _one(folder, capacities):
    scenario = load_authorized(folder)
    workload = json.loads((folder/'workload.json').read_text(encoding='utf-8'))['payload']
    environment = json.loads((folder/'environment.json').read_text(encoding='utf-8'))['payload']
    if (workload['resource_multiplier'], workload['sla_multiplier'], workload['trace_year']) != (5, 1.5, 2020):
        raise ValueError('Expected legacy x5/r1/s1.5 2020 source before Observed x1 mapping')
    rows = workload['tasks']
    if not rows:
        raise ValueError('Empty Train arrival cohort')
    for row in rows:
        stable = (f"{workload['source_sha256'][:16]}/{row['arrival_time']}/"
                  f"{row['source_row']}/{row['source_task_index']}")
        if (row['task_id'] != stable or row['source_job_name'] != row['raw_fields'][0]
                or row['sla_multiplier'] != 1.5 or row['dc_id'] != 1):
            raise ValueError('Noncanonical stable source identity or task configuration')
    for name, capacity in capacities.items():
        if any(any(d > c for d, c in zip(t.demand, capacity)) for t in scenario.tasks):
            raise ValueError('Intrinsic task demand exceeds '+name+' capacity')
    env_audit, samples = _environment(scenario, environment)
    content = dict(raw=[canonical_sha(r['raw_fields']) for r in rows],
                   original=[canonical_sha(r['raw_fields'][:3]) for r in rows],
                   jobs=[r['source_job_name'] for r in rows],
                   stable=[t.task_id for t in scenario.tasks])
    expected = dict(scenario.provenance, tasks=len(scenario.tasks),
                    task_identity_sha256=task_identity(scenario))
    report = dict(fragment_id=folder.name, tasks=len(rows), provenance=scenario.provenance,
        task_identity_sha256=expected['task_identity_sha256'],
        arrival=dict(hours=24, start=scenario.provenance['start'],
            end_exclusive=scenario.provenance['arrival_end'],
            recorded_arrival_slots=len({r['arrival_time'] for r in rows}),
            all_arrivals_within_24_hours=True),
        legacy_x5_mapping_verified=True, observed_x1_mapping_verified=True,
        stable_source_identity_verified=True, original_rows_retained=True,
        raw_content=_repeats(content['raw']), original_job_start_end=_repeats(content['original']),
        original_job_name=_repeats(content['jobs']), environment=env_audit,
        maximum_observed_demand=[max(t.demand[i] for t in scenario.tasks) for i in range(3)],
        all_tasks_individually_fit_each_capacity=True)
    return report, expected, content, samples


def _overlap(a, b):
    return max(0., (min(stamp(a['end_exclusive']), stamp(b['end_exclusive']))
                    - max(stamp(a['start']), stamp(b['start']))).total_seconds()/60)


def _components(ids, pairs):
    """Connected components, closed over chronological intervals."""
    parent = list(range(len(ids)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    def join(a, b):
        parent[find(b)] = find(a)
    for pair in pairs:
        if pair['overlapping_environment_minutes'] or pair['shared_original_job_start_end']:
            join(ids.index(pair['fragments'][0]), ids.index(pair['fragments'][1]))
    # Closing spans may merge several graph components; repeat to a fixed point.
    while True:
        before = [find(i) for i in range(len(ids))]
        groups = {}
        for i in range(len(ids)):
            groups.setdefault(find(i), []).append(i)
        for members in groups.values():
            for i in range(min(members), max(members)+1):
                join(members[0], i)
        if before == [find(i) for i in range(len(ids))]:
            break
    groups = {}
    for i, fragment in enumerate(ids):
        groups.setdefault(find(i), []).append(fragment)
    return list(groups.values())


def audit_pool(pool):
    """Read only the explicit 16 Train IDs; return aggregate admission evidence."""
    pool = Path(pool).resolve()
    scenarios = pool if pool.name == 'scenarios' else pool/'scenarios'
    if scenarios.resolve() != AUTHORIZED_SCENARIOS.resolve():
        raise ValueError('Only the authorized E2 scenarios parent path may be audited')
    # Freeze the outcome-independent rule before opening any real-input manifest.
    split_rule = dict(SPLIT_RULE)
    cfg = load_config(CONFIG)
    capacities = {name: build_inventory(cfg, name).capacity for name in ('OM', 'OH', 'OL')}
    available = sorted(p.name for p in scenarios.glob('train-*-r1-s1.5') if p.is_dir())
    fragments, inputs, contents, samples, invalid = [], {}, {}, {}, []
    for fragment in FRAGMENTS:
        try:
            row, expected, content, environment = _one(scenarios/fragment, capacities)
        except (ValueError, OSError, KeyError, TypeError) as exc:
            invalid.append(dict(fragment_id=fragment, error=type(exc).__name__+': '+str(exc)))
            continue
        fragments.append(row); inputs[fragment] = expected
        contents[fragment] = content; samples[fragment] = environment
    by_id = {r['fragment_id']: r for r in fragments}
    pairs, blockers = [], []
    for a, b in itertools.combinations(inputs, 2):
        overlap = _overlap(by_id[a]['environment'], by_id[b]['environment'])
        shared_times = samples[a].keys() & samples[b].keys()
        conflicts = sum(samples[a][t] != samples[b][t] for t in shared_times)
        pair = dict(fragments=[a, b], overlapping_environment_minutes=overlap,
            conflicting_shared_environment_samples=conflicts,
            shared_raw_content=len(set(contents[a]['raw']) & set(contents[b]['raw'])),
            shared_original_job_start_end=len(set(contents[a]['original']) & set(contents[b]['original'])),
            shared_original_job_name=len(set(contents[a]['jobs']) & set(contents[b]['jobs'])),
            shared_stable_task_ids=len(set(contents[a]['stable']) & set(contents[b]['stable'])))
        pairs.append(pair)
        if conflicts or pair['shared_stable_task_ids']:
            blockers.append('Conflicting environment samples or stable task IDs: '+a+' / '+b)
    components = []
    for number, members in enumerate(_components(list(inputs), pairs)):
        split = 'holdout' if any(f[6:12] >= '202304' for f in members) else 'train'
        components.append(dict(component_id=f'component-{number+1:02d}', fragment_ids=members,
            start=by_id[members[0]]['environment']['start'],
            end_exclusive=max(by_id[f]['environment']['end_exclusive'] for f in members), split=split))
    train = [f for c in components if c['split'] == 'train' for f in c['fragment_ids']]
    holdout = [f for c in components if c['split'] == 'holdout' for f in c['fragment_ids']]
    crossing = [p for p in pairs if (p['fragments'][0] in train) != (p['fragments'][1] in train)]
    cross_split = dict(shared_raw_content=sum(p['shared_raw_content'] for p in crossing),
        shared_original_job_start_end=sum(p['shared_original_job_start_end'] for p in crossing),
        shared_stable_task_ids=sum(p['shared_stable_task_ids'] for p in crossing),
        overlapping_environment_minutes=sum(p['overlapping_environment_minutes'] for p in crossing))
    if invalid:
        blockers.append('All sixteen windows must be verified; invalid windows retained for review, no automatic exclusion')
        train, holdout = [], []
    if not train or not holdout:
        blockers.append('No nonempty isolated chronological training/holdout split is admitted')
    if any(cross_split.values()):
        blockers.append('Training/holdout input overlap remains')
    aggregate = {key: [item for content in contents.values() for item in content[key]]
                 for key in ('raw', 'original', 'jobs', 'stable')}
    return dict(schema='sca-ppo-e2-train-data-admission-v1', passed=not blockers,
        pool_path=scenarios.as_posix(), declared_window_count=16, usable_window_count=len(inputs),
        total_tasks=len(aggregate['stable']), train_ids=train, holdout_ids=holdout,
        split_rule=split_rule, inputs=inputs, inputs_canonical_sha256=canonical_sha(inputs),
        fragments=fragments, invalid_windows=invalid, blockers=blockers,
        time_components=components, cross_fragment=pairs, cross_split=cross_split,
        aggregate=dict(raw_content=_repeats(aggregate['raw']),
            original_job_start_end=_repeats(aggregate['original']),
            original_job_name=_repeats(aggregate['jobs']), stable_task_ids=_repeats(aggregate['stable'])),
        capacities=capacities, v2_configuration_sha256=file_sha(CONFIG),
        audits=dict(role_and_test_access_checked_before_payload=True,
            legacy_x5_and_observed_x1_verified=True, arrival_hours=24, environment_hours=72,
            task_deduplication=False, performance_metrics_computed=False),
        available_train_base_directory_names=available,
        read_boundary=dict(payload_fragment_ids=list(FRAGMENTS), other_window_manifests_read=False,
            annual_workload_read=False, original_csv_read=False, dev_validation_test_read=False,
            schedules_or_learning_run=False),
        limitations=['Original processed duplicate observations are retained unchanged.',
            'Static individual task fit does not prove scheduling feasibility.',
            'The holdout consists of authorized Train windows, not sealed Validation or Test.',
            'Repeated use of one processed annual trace is development evidence.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pool', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if not Path(args.pool).is_absolute():
        parser.error('--pool must be an absolute path')
    report = audit_pool(args.pool)
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, allow_nan=False,
                                 sort_keys=True, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(passed=report['passed'], usable_windows=report['usable_window_count'],
        tasks=report['total_tasks'], train_windows=len(report['train_ids']),
        holdout_windows=len(report['holdout_ids']), output=str(output))))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
