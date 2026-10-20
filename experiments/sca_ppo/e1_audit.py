"""Deterministic E1-0 audit of three frozen Train snapshots; no scheduling.

Only manifests and workload/environment payloads for the explicit whitelist are
read. Other Train window availability is counted from directory names alone.
"""
import argparse
from collections import Counter, defaultdict
from datetime import timedelta
import itertools
import json
from pathlib import Path

from experiments.platform_v1.data import COLUMNS, canonical_sha, file_sha, load_train, stamp
from experiments.platform_v2.inventory import build_inventory, load_config

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT/'docs/research/platform-v2-final/acceptance-protocol.json'
CONFIG = ROOT/'experiments/platform_v2/config.json'
FRAGMENTS = ('train-20230302T0500-r1-s1.5', 'train-20230404T0500-r1-s1.5',
             'train-20230501T0500-r1-s1.5')


def _repeats(values):
    counts = Counter(values)
    return dict(occurrences=sum(counts.values()), unique=len(counts),
        duplicate_excess=sum(n-1 for n in counts.values()),
        repeat_groups=sum(n > 1 for n in counts.values()),
        repeated_occurrences=sum(n for n in counts.values() if n > 1),
        maximum_multiplicity=max(counts.values(), default=0))


def _extent(values):
    present = [v for v in values if v is not None]
    return dict(minimum=min(present) if present else None,
                maximum=max(present) if present else None,
                missing=len(values)-len(present))


def _content(rows):
    return dict(raw=[canonical_sha(t['raw_fields']) for t in rows],
        original=[canonical_sha(t['raw_fields'][:3]) for t in rows],
        jobs=[t['source_job_name'] for t in rows])


def _identity_repeats(rows):
    groups = defaultdict(list)
    for task in rows:
        groups[canonical_sha(task['raw_fields'][:3])].append(task)
    repeated = [g for g in groups.values() if len(g) > 1]
    differences = Counter()
    for group in repeated:
        for index, name in enumerate(COLUMNS):
            if len({canonical_sha(t['raw_fields'][index]) for t in group}) > 1:
                differences[name] += 1
    return dict(groups=len(repeated), differing_fields=dict(sorted(differences.items())),
        same_source_row_groups=sum(len({t['source_row'] for t in g}) == 1 for g in repeated),
        all_observed_rows_retained=True)


def _environment_audit(scenario, environment):
    origin = stamp(scenario.provenance['start'])
    end = origin+timedelta(hours=72)
    samples = [s for s in environment['samples'] if origin <= stamp(s['time']) < end]
    times = [stamp(s['time']) for s in samples]
    if times != [origin+timedelta(minutes=15*i) for i in range(288)]:
        raise ValueError('Incomplete or duplicate 72-hour environmental coverage')
    hourly = {stamp(s['time']): s for s in samples if stamp(s['time']).minute == 0}
    causal = len(scenario.observations) == 288
    for index, observation in enumerate(scenario.observations):
        minute = index*15
        observed = minute//60*60
        source = hourly[origin+timedelta(minutes=observed)]
        causal = causal and (observation.minute, observation.observed_minute,
            observation.ci_g_per_kwh, observation.outdoor_c) == (
            minute, observed, source['ci_g_per_kwh'], source['temperature_c'])
    if not causal:
        raise ValueError('Loaded observations violate causal previous-hour hold')
    return dict(responsibility_hours=72, samples_in_snapshot=len(environment['samples']),
        samples_in_72_hours=len(samples), hourly_anchors=len(hourly),
        unique_contiguous_quarter_hours=True, causal_previous_hour_hold=True,
        loaded_observations=len(scenario.observations),
        observation_minute_range=[0, 4305], observed_minute_range=[0, 4260],
        transform=scenario.provenance['environment_transform'],
        source_interpolation=environment['interpolation'],
        weather_source_time_basis=scenario.provenance['weather_source_time_basis'],
        weather_source_timezone=scenario.provenance['weather_source_timezone'],
        weather_source_utc_offset_seconds=scenario.provenance['weather_source_utc_offset_seconds'])


def audit_pool(pool, *, protocol=None):
    """Validate the frozen boundary and return aggregate evidence without mutation.

    ``protocol`` permits isolated synthetic fixtures; the CLI always uses the
    repository's frozen V2 protocol and provides no override for it.
    """
    pool = Path(pool).resolve()
    protocol = protocol if protocol is not None else json.loads(PROTOCOL.read_text(encoding='utf-8'))
    if set(protocol['inputs']) != set(FRAGMENTS):
        raise ValueError('Frozen protocol differs from the three-fragment whitelist')
    if file_sha(CONFIG) != protocol['source_hashes']['experiments/platform_v2/config.json']:
        raise ValueError('Frozen V2 configuration changed')
    config = load_config(CONFIG)
    capacities = {name: build_inventory(config, name).capacity for name in ('OM', 'OH', 'OL')}
    results, contents, all_rows, all_ids = [], {}, [], []
    for fragment in FRAGMENTS:
        folder = pool/'scenarios'/fragment
        expected = protocol['inputs'][fragment]
        # This loader role/schema/test_access check precedes ALL payload reads
        # and hashes, then verifies manifest/file/content hashes against freeze.
        scenario = load_train(folder, hours=72, expected=expected)
        digest = canonical_sha([(t.task_id, t.arrival_minute, t.duration_minutes, t.demand)
                                for t in scenario.tasks])
        if scenario.scenario_id != fragment or digest != expected['task_identity_sha256']:
            raise ValueError('Frozen task identity mismatch')
        if len(scenario.tasks) != expected['tasks']:
            raise ValueError('Frozen task identity count mismatch')
        frozen_provenance = {k: v for k, v in expected.items()
                             if k not in ('tasks', 'task_identity_sha256')}
        if scenario.provenance != frozen_provenance:
            raise ValueError('Frozen Train provenance changed')
        workload = json.loads((folder/'workload.json').read_text(encoding='utf-8'))['payload']
        environment = json.loads((folder/'environment.json').read_text(encoding='utf-8'))['payload']
        if (workload['resource_multiplier'], workload['sla_multiplier'], workload['trace_year']) != (5, 1.5, 2020):
            raise ValueError('Frozen workload configuration differs from legacy mapping')
        rows = workload['tasks']
        source_ids = []
        for task in rows:
            source_id = (task['arrival_time'], task['source_row'], task['source_task_index'])
            stable_id = f"{workload['source_sha256'][:16]}/{source_id[0]}/{source_id[1]}/{source_id[2]}"
            if (task['task_id'] != stable_id or task['source_job_name'] != task['raw_fields'][0]
                    or task['sla_multiplier'] != 1.5 or task['dc_id'] != 1):
                raise ValueError('Noncanonical stable source identity or task configuration')
            source_ids.append(source_id)
        if len(set(source_ids)) != len(source_ids):
            raise ValueError('Duplicate stable source identity')
        for name, capacity in capacities.items():
            if any(any(d > c for d, c in zip(task.demand, capacity)) for task in scenario.tasks):
                raise ValueError('Intrinsic task demand exceeds '+name+' capacity')
        content = _content(rows); contents[fragment] = content
        all_rows.extend(rows); all_ids.extend(t.task_id for t in scenario.tasks)
        results.append(dict(fragment_id=fragment, tasks=len(rows),
            task_identity_sha256=digest, provenance=scenario.provenance,
            frozen_hashes_and_task_identity_match=True, stable_source_identity_verified=True,
            raw_content=_repeats(content['raw']),
            original_job_start_end=_repeats(content['original']),
            original_job_name=_repeats(content['jobs']),
            original_identity_repeats=_identity_repeats(rows),
            source=dict(trace_year=workload['trace_year'],
                source_row=_extent([t['source_row'] for t in rows]),
                recorded_arrival_slots=len({t['arrival_time'] for t in rows}),
                original_start_time=_extent([t['raw_fields'][1] for t in rows]),
                original_end_time=_extent([t['raw_fields'][2] for t in rows]),
                original_start_dt=_extent([t['raw_fields'][3] for t in rows])),
            maximum_observed_demand=[max(t.demand[i] for t in scenario.tasks) for i in range(3)],
            all_tasks_individually_fit_each_capacity=True,
            environment=_environment_audit(scenario, environment)))
    if len(set(all_ids)) != len(all_ids):
        raise ValueError('Stable task IDs collide across authorized fragments')
    pairs = []
    for a, b in itertools.combinations(FRAGMENTS, 2):
        pairs.append(dict(fragments=[a, b],
            shared_raw_content=len(set(contents[a]['raw']) & set(contents[b]['raw'])),
            shared_original_job_start_end=len(set(contents[a]['original']) & set(contents[b]['original'])),
            shared_original_job_name=len(set(contents[a]['jobs']) & set(contents[b]['jobs']))))
    content = _content(all_rows)
    # Names only: no manifest or payload belonging to another window is opened.
    available = sum(p.is_dir() for p in (pool/'scenarios').glob('train-*-r1-s1.5'))
    return dict(schema='sca-ppo-e1-train-data-audit-v1', passed=True,
        pool_path=pool.as_posix(), v2_protocol_sha256=file_sha(PROTOCOL),
        expected_inputs_canonical_sha256=canonical_sha(protocol['inputs']),
        v2_configuration_sha256=file_sha(CONFIG), capacities=capacities,
        total_tasks=len(all_rows), all_stable_task_ids_unique=True, fragments=results,
        aggregate=dict(raw_content=_repeats(content['raw']),
            original_job_start_end=_repeats(content['original']), original_job_name=_repeats(content['jobs'])),
        cross_fragment=pairs, optional_train_base_window_directory_count=available,
        read_boundary=dict(payload_fragment_ids=list(FRAGMENTS),
            other_window_manifests_read=False, annual_workload_read=False,
            original_csv_read=False, schedules_or_learning_run=False),
        original_schema=dict(columns=list(COLUMNS), identity_fields=[
            'source_job_name', 'source_row', 'source_task_index', 'raw_fields[0:4]'],
            stable_id='source_sha256[:16]/arrival_time/source_row/source_task_index'),
        limitations=[
            'Known processed duplicate observations are retained; job identities are not unique task rows.',
            'Original raw timestamps are relative epoch values; 2020 trace buckets are aligned to 2023 arrivals.',
            'Cross-fragment source comparisons ignore assigned arrival dates and use original job/start/end or all twelve raw fields.',
            'Repeated exposure to these fixed Train snapshots is development evidence, not independent annual collection.',
            'Only static individual task fit is checked; this audit does not prove scheduling feasibility.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pool', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if not Path(args.pool).is_absolute():
        parser.error('--pool must be an absolute path')
    report = audit_pool(args.pool)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(passed=report['passed'], tasks=report['total_tasks'], output=str(output))))


if __name__ == '__main__':
    main()
