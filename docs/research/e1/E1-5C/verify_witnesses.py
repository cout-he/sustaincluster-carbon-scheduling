"""Independent offline verification; does not import the feasibility module."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(first, second):
    assert math.isclose(float(first), float(second), rel_tol=2e-12, abs_tol=1e-8), (first, second)


def verify(root, diagnostic, destination):
    if destination.exists():
        raise FileExistsError('Refusing to replace a previous verification')
    manifest_path, summary_path = diagnostic / 'input-manifest.json', diagnostic / 'analysis-summary.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    summaries = json.loads(summary_path.read_text(encoding='utf-8'))
    assert manifest['test_evaluation'] is False and manifest['training'] is False
    by_key = {(entry['split'], float(entry['factor'])): entry for entry in manifest['cohorts']}
    results = []
    for cohort in summaries:
        split, factor = cohort['split'], float(cohort['factor'])
        assert split in ('train', 'validation'), 'Test access is forbidden'
        source = by_key[(split, factor)]
        input_name = source['input_path']
        if os.name == 'nt' and input_name.startswith('/mnt/d/'):
            input_name = 'D:/' + input_name[len('/mnt/d/'):]
        input_path = Path(input_name)
        if not input_path.is_absolute():
            input_path = root / input_path
        workload_path = input_path / 'workload.json'
        assert file_hash(workload_path) == source['identity']['files']['workload.json']
        document = json.loads(workload_path.read_text(encoding='utf-8'))
        payload = document['payload']
        assert document['kind'] == 'e1-workload-v1'
        assert digest(payload) == document['sha256'] == source['identity']['workload_sha256']
        tasks = payload['tasks']
        capacity = np.asarray(source['identity']['capacity'], dtype=float)
        arrival = np.asarray([pd.Timestamp(task['arrival_time']).value for task in tasks], dtype=np.int64)
        duration_ns = np.asarray([pd.Timedelta(minutes=task['duration']).value for task in tasks], dtype=np.int64)
        assert np.all(duration_ns > 0), 'Zero-nanosecond duration is outside this bound semantic'
        deadline = np.asarray([(pd.Timestamp(task['arrival_time']) + pd.Timedelta(
            minutes=task['sla_multiplier'] * task['duration'])).value for task in tasks], dtype=np.int64)
        step_ns = 900_000_000_000
        occupied_steps = (duration_ns + step_ns - 1) // step_ns
        float_steps = np.asarray([math.ceil(task['duration'] / 15.0) for task in tasks], dtype=np.int64)
        latest_grid = ((deadline - duration_ns) // step_ns) * step_ns
        demand = np.asarray([[task[name] for name in ('cores_req', 'gpu_req', 'mem_req')]
                              for task in tasks], dtype=float)
        work = demand * (15 * occupied_steps[:, None])
        intrinsic = ((latest_grid < arrival) | np.any(demand > capacity, axis=1))
        modes = {}
        for mode in ('raw_deadline', 'occupied_deadline'):
            reported = cohort['feasibility'][mode]
            assert reported['task_count'] == len(tasks)
            assert reported['workload_sha256'] == document['sha256']
            assert reported['intrinsically_impossible_count'] == int(intrinsic.sum())
            effective = (((deadline + step_ns - 1) // step_ns) * step_ns
                         if mode == 'raw_deadline' else latest_grid + occupied_steps * step_ns)
            starts, arrival_index = np.unique(arrival, return_inverse=True)
            ends, deadline_index = np.unique(effective, return_inverse=True)
            cell = np.zeros((len(starts), len(ends), 3), dtype=float)
            np.add.at(cell, (arrival_index, deadline_index), work)
            required = np.cumsum(np.cumsum(cell[::-1], axis=0)[::-1], axis=1)
            minutes = (ends[None, :] - starts[:, None]) / 60e9
            valid = minutes > 0
            assert int(valid.sum()) == reported['checked_interval_count']
            resource_checks = {}
            for index, resource in enumerate(('cpu', 'gpu', 'mem')):
                cap_work = capacity[index] * minutes
                tolerance = 1e-10 * np.maximum(1, np.maximum(np.abs(required[:, :, index]), np.abs(cap_work)))
                overloaded = valid & (required[:, :, index] > cap_work + tolerance)
                ratio = np.divide(required[:, :, index], cap_work,
                                  out=np.zeros_like(cap_work), where=valid)
                expected = reported['resource_results'][resource]
                assert int(overloaded.sum()) == expected['overloaded_interval_count']
                close(float(ratio[valid].max()), expected['maximum_workload_capacity_ratio'])
                resource_checks[resource] = dict(overloaded_interval_count=int(overloaded.sum()),
                    maximum_workload_capacity_ratio=float(ratio[valid].max()),
                    reported_minimum_misses_lower_bound=expected['minimum_deadline_misses_lower_bound'])
            witnesses = {}
            supplied = list(reported['witnesses']) + [item['witness']
                for item in reported['resource_results'].values() if item['witness'] is not None]
            for witness in supplied:
                u, v = pd.Timestamp(witness['start']), pd.Timestamp(witness['end'])
                assert u.value % step_ns == v.value % step_ns == 0
                cutoff = deadline if mode == 'raw_deadline' else latest_grid + occupied_steps * step_ns
                selected = np.flatnonzero((arrival >= u.value) & (cutoff <= v.value))
                resource = ('cpu', 'gpu', 'mem').index(witness['resource'])
                ids = sorted(tasks[index]['task_id'] for index in selected)
                assert len(selected) == witness['task_count']
                assert digest(ids) == witness['task_id_sha256']
                weights = sorted((float(work[index, resource]) for index in selected), reverse=True)
                required_sum = math.fsum(weights)
                capacity_sum = capacity[resource] * (v.value - u.value) / 60e9
                numerical_tolerance = 1e-10 * max(1, abs(required_sum), abs(capacity_sum))
                close(required_sum, witness['required_resource_minutes'])
                close(capacity_sum, witness['capacity_resource_minutes'])
                close(numerical_tolerance, witness['numerical_tolerance_resource_minutes'])
                k = int(witness['minimum_misses_lower_bound'])
                removed_k = math.fsum(weights[:k])
                removed_previous = math.fsum(weights[:max(0, k-1)])
                assert required_sum - removed_k <= capacity_sum + numerical_tolerance
                assert k > 0 and required_sum - removed_previous > capacity_sum + numerical_tolerance
                close(removed_k, witness['greedy_removal_resource_minutes'])
                key = (u.value, v.value, resource)
                witnesses[key] = dict(resource=witness['resource'],
                    start_utc=u.tz_convert('UTC').isoformat(), end_utc=v.tz_convert('UTC').isoformat(),
                    task_count=len(selected), task_id_sha256=digest(ids),
                    required_resource_minutes=required_sum, capacity_resource_minutes=capacity_sum,
                    minimum_misses_lower_bound=k, largest_k_removed_resource_minutes=removed_k,
                    largest_k_minus_one_removed_resource_minutes=removed_previous,
                    remaining_after_k_minus_one_resource_minutes=required_sum-removed_previous,
                    k_and_k_minus_one_certificate_passed=True)
            lower = max([int(intrinsic.sum())] +
                        [entry['reported_minimum_misses_lower_bound'] for entry in resource_checks.values()])
            assert lower == reported['minimum_deadline_misses_lower_bound']
            close(lower / len(tasks), reported['violation_rate_lower_bound'])
            modes[mode] = dict(independent_grid_matrix_checks=resource_checks,
                verified_witness_count=len(witnesses), witnesses=list(witnesses.values()),
                minimum_deadline_misses_lower_bound=lower, violation_rate_lower_bound=lower/len(tasks))
        results.append(dict(split=split, factor=factor, tasks=len(tasks),
            workload_sha256=document['sha256'], workload_file_sha256=file_hash(workload_path),
            all_duration_nanoseconds_positive=True, minimum_duration_nanoseconds=int(duration_ns.min()),
            float_ceil_vs_nanosecond_ceil_difference_count=int(np.sum(float_steps != occupied_steps)),
            intrinsic_impossible_count=int(intrinsic.sum()), modes=modes))
    result = dict(schema='e1-5c-independent-witness-verification-v1', passed=True,
        no_test_access=True, no_training=True, feasibility_module_imported=False,
        verification_method='Independent JSON/UTC timing, 2D suffix-prefix workload matrices and sorted witness k/k-1 certificates',
        input_manifest_sha256=file_hash(manifest_path), analysis_summary_sha256=file_hash(summary_path), cohorts=results)
    destination.write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(dict(passed=True, cohorts=len(results), duration_ceil_differences=sum(
        entry['float_ceil_vs_nanosecond_ceil_difference_count'] for entry in results),
        verified_witnesses=sum(mode['verified_witness_count'] for entry in results for mode in entry['modes'].values()),
        output_sha256=file_hash(destination)), sort_keys=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--diagnostics', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    arguments = parser.parse_args()
    verify(arguments.root, arguments.diagnostics, arguments.output)
