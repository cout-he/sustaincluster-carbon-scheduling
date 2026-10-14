"""Fail-closed state fingerprints and independent E1 counterfactual audits.

These tools are offline diagnostics. They never provide future information to
the policy, nor replace the upstream IT/HVAC model with a power approximation.
"""
from collections import Counter, deque
import datetime
import hashlib
import math
import types

import numpy as np
import pandas as pd

from .contracts import STEP, canonical_bytes, content_hash, utc_time

VERSION = 'e1-7a-counterfactual-audit-v1'


def serializable_state(value):
    """Serialize object state without addresses, with stable alias paths.

    Arrays retain dtype/shape and exact byte digests. RNG states and the entire
    physical object tree are included. Unsupported opaque objects fail closed.
    This verifies inspected Python state, not arbitrary external side effects.
    """
    seen = {}

    def visit(item, path):
        if item is None or isinstance(item, (str, bool, int)):
            return item
        if isinstance(item, (float, np.floating)):
            number = float(item)
            return number if math.isfinite(number) else {'nonfinite_float': str(number)}
        if isinstance(item, np.integer):
            return int(item)
        if isinstance(item, np.bool_):
            return bool(item)
        if isinstance(item, (pd.Timestamp, datetime.datetime)):
            return {'timestamp': item.isoformat()}
        if isinstance(item, (pd.Timedelta, datetime.timedelta)):
            return {'timedelta_seconds': item.total_seconds()}
        if isinstance(item, np.dtype):
            return {'numpy_dtype': item.str, 'description': str(item.descr)}
        if isinstance(item, type):
            return {'python_type': item.__module__ + '.' + item.__qualname__}
        if isinstance(item, (types.FunctionType, types.BuiltinFunctionType, np.ufunc)):
            # The qualified implementation is static; source hashes are separate.
            return {'callable': getattr(item, '__module__', 'numpy') + '.' + item.__name__}
        identity = id(item)
        if identity in seen:
            return {'reference_path': seen[identity]}
        seen[identity] = path
        if isinstance(item, np.ndarray):
            if item.dtype.hasobject:
                return {'array_dtype': str(item.dtype), 'shape': list(item.shape),
                        'objects': visit(item.tolist(), path + '.objects')}
            return {'array_dtype': item.dtype.str, 'shape': list(item.shape),
                    'bytes_sha256': hashlib.sha256(np.ascontiguousarray(item).tobytes()).hexdigest()}
        if isinstance(item, np.random.Generator):
            return {'numpy_generator': visit(item.bit_generator.state, path + '.rng')}
        if isinstance(item, np.random.RandomState):
            return {'numpy_random_state': visit(item.get_state(), path + '.rng')}
        if isinstance(item, dict):
            keys = sorted(item, key=lambda key: canonical_bytes(visit_key(key)))
            return {'mapping': [[visit_key(key), visit(item[key], path + '.' + str(index))]
                                for index, key in enumerate(keys)]}
        if isinstance(item, (list, tuple, deque)):
            return {'sequence_type': type(item).__name__,
                    'items': [visit(child, path + '.' + str(index)) for index, child in enumerate(item)]}
        if isinstance(item, (set, frozenset)):
            # Current observer sets contain task-ID strings; sorting is explicit.
            children = [visit_key(child) for child in item]
            return {'set_type': type(item).__name__, 'items': sorted(children, key=canonical_bytes)}
        if hasattr(item, '__dict__'):
            return {'object_type': type(item).__module__ + '.' + type(item).__qualname__,
                    'attributes': visit(vars(item), path + '.attributes')}
        raise TypeError('Unsupported state fingerprint type: ' + type(item).__module__ + '.' + type(item).__qualname__)

    def visit_key(key):
        if key is None or isinstance(key, (str, bool, int)):
            return {'key_type': type(key).__name__, 'value': key}
        if isinstance(key, float) and math.isfinite(key):
            return {'key_type': 'float', 'value': key}
        if isinstance(key, pd.Timestamp):
            return {'key_type': 'timestamp', 'value': key.isoformat()}
        if isinstance(key, tuple):
            return {'key_type': 'tuple', 'value': [visit_key(child) for child in key]}
        raise TypeError('Unsupported state mapping key: ' + type(key).__name__)

    return visit(value, '$')


def state_fingerprint(value):
    return content_hash(serializable_state(value))


def executor_prefix(executor, observer, candidate, context):
    """Only currently observed state; never inspect future cohort/sample lists."""
    ledger = {}
    for task_id, entry in executor.ledger.items():
        task = entry['task']
        task_state = {key: value for key, value in vars(task).items() if key not in ('dest_dc', 'origin_dc')}
        task_state['destination_is_current_pool'] = task.dest_dc is executor.pool
        task_state['origin_reference_is_none'] = task.origin_dc is None
        ledger[task_id] = {'spec': entry['spec'].to_dict(), 'task': task_state,
                           **{key: value for key, value in entry.items() if key not in ('spec', 'task')}}
    components = dict(candidate=candidate.to_dict(), current=context,
        current_sample=executor.sample, ledger=ledger,
        queue={'pending': [task.job_name for task in executor.pool.pending_tasks],
               'running': [task.job_name for task in executor.pool.running_tasks],
               'capacity': executor.pool.capacity,
               'available': (executor.pool.available_cores, executor.pool.available_gpus, executor.pool.available_mem)},
        events=executor.events, steps=executor.steps, policy_events=executor.policy_events,
        policy_steps=executor.policy_steps,
        observer={'history': observer.history, 'snapshot': observer.snapshot,
                  'ordered': observer.ordered, 'pending': observer.pending,
                  'running': observer.running, 'available': observer.available,
                  'occurrences': observer.occurrences},
        accumulated={name: math.fsum(step[name] for step in executor.steps)
                     for name in ('it_energy_kwh', 'hvac_energy_kwh', 'total_energy_kwh', 'carbon_kg')},
        physical_model=executor.power_model)
    hashes = {key: state_fingerprint(value) for key, value in components.items()}
    return {'version': VERSION, 'components': hashes, 'sha256': content_hash(hashes)}


def _close(actual, expected, label):
    if not math.isclose(float(actual), float(expected), rel_tol=1e-12, abs_tol=1e-9):
        raise RuntimeError(label + ' audit mismatch')


def audit_result(result, workload, environment, capacity):
    """Reconstruct occupancy from task rows, not executor aggregate counters."""
    capacity = tuple(capacity)
    tasks = {row['task_id']: row for row in result.tasks}
    specs = {spec.task_id: spec for spec in workload.tasks}
    if len(tasks) != len(result.tasks) or set(tasks) != set(specs):
        raise RuntimeError('Task identity or complete cohort audit mismatch')
    events = Counter((event['task_id'], event['event']) for event in result.events)
    event_times = {(event['task_id'], event['event']): event['time'] for event in result.events}
    resources = ('cores', 'gpus', 'mem')
    occupancy = []
    for task_id, spec in specs.items():
        row = tasks[task_id]
        if any(row[key] != spec.to_dict()[key] for key in spec.to_dict()):
            raise RuntimeError('Immutable task input changed')
        if row['state'] != 'completed' or row['start_time'] is None or row['finish_time'] is None:
            raise RuntimeError('Every task must be completed after drain')
        start, finish = utc_time(row['start_time'], True), utc_time(row['finish_time'])
        if start < spec.arrival_time or finish != start + pd.Timedelta(minutes=spec.duration):
            raise RuntimeError('Task arrival/start/actual finish audit mismatch')
        if utc_time(row['completion_observed_at'], True) != finish.ceil('15min'):
            raise RuntimeError('Grid resource release audit mismatch')
        occupancy.append((start.value, finish.ceil('15min').value,
                          (spec.cores_req, spec.gpu_req, spec.mem_req)))
        if row['deadline_outcome'] != ('met' if finish <= spec.sla_deadline else 'missed'):
            raise RuntimeError('Actual finish deadline audit mismatch')
        if utc_time(row['sla_deadline']) != spec.sla_deadline or row['sla_met'] != (finish <= spec.sla_deadline):
            raise RuntimeError('Deadline or SLA field audit mismatch')
        for kind in ('arrived', 'enqueued', 'started', 'completed_resources_released'):
            if events[task_id, kind] != 1:
                raise RuntimeError('Task arrival/start/release must occur exactly once')
        expected_times = {'arrived': spec.arrival_time, 'enqueued': spec.arrival_time,
                          'started': start, 'completed_resources_released': finish.ceil('15min')}
        for kind, expected_time in expected_times.items():
            if utc_time(event_times[task_id, kind], True) != expected_time:
                raise RuntimeError('Task arrival/start/release event time audit mismatch')
    samples = {sample.time: sample for sample in environment.samples}
    times = [utc_time(step['interval_start'], True) for step in result.steps]
    if len(set(times)) != len(times) or not times or times[0] != workload.start:
        raise RuntimeError('Unique interval/observation start audit mismatch')
    if any(right - left != pd.Timedelta(minutes=15) for left, right in zip(times, times[1:])):
        raise RuntimeError('Contiguous interval audit mismatch')
    for step, now in zip(result.steps, times):
        if now not in samples or utc_time(step['interval_end'], True) != now + pd.Timedelta(minutes=15):
            raise RuntimeError('Environmental coverage audit mismatch')
        sample = samples[now]
        if any(step[key] != getattr(sample, field) for key, field in (
                ('ci_g_per_kwh', 'ci_g_per_kwh'), ('temperature_c', 'temperature_c'), ('wet_bulb_c', 'wet_bulb_c'))):
            raise RuntimeError('CI/weather audit mismatch')
        occupying = [requirements for start, release, requirements in occupancy if start <= now.value < release]
        for index, resource in enumerate(resources):
            used = math.fsum(requirements[index] for requirements in occupying)
            if used > capacity[index]:
                raise RuntimeError('Resource capacity audit violation')
            _close(step['used_' + resource], used, 'resource')
            _close(step['available_' + resource], capacity[index] - used, 'resource')
        _close(step['interval_hours'], .25, 'energy interval')
        _close(step['it_energy_kwh'], step['it_power_kw'] * .25, 'IT energy')
        _close(step['hvac_energy_kwh'], step['hvac_power_kw'] * .25, 'HVAC energy')
        _close(step['total_energy_kwh'], step['it_energy_kwh'] + step['hvac_energy_kwh'], 'total energy')
        _close(step['total_power_kw'], step['it_power_kw'] + step['hvac_power_kw'], 'total energy power')
        _close(step['carbon_kg'], step['total_energy_kwh'] * sample.ci_g_per_kwh / 1000, 'carbon')
    for field in ('it_energy_kwh', 'hvac_energy_kwh', 'total_energy_kwh', 'carbon_kg'):
        _close(result.summary[field], math.fsum(step[field] for step in result.steps), 'aggregate ' + field)
    if result.summary['deadline_missed_completed'] != sum(row['deadline_outcome'] == 'missed' for row in tasks.values()):
        raise RuntimeError('Aggregate deadline count audit mismatch')
    resource_hours = [math.fsum(step['used_' + resource] * .25 for step in result.steps) for resource in resources]
    # Actual Task finish uses pandas Timedelta nanoseconds. At a float immediately
    # above a grid boundary this can differ from the frozen v1 history proxy's
    # math.ceil(duration/15). Audit the executor, never change that legacy proxy.
    expected = [math.fsum(getattr(spec, field) * ((pd.Timedelta(minutes=spec.duration).value + STEP.value - 1) // STEP.value) * .25
                         for spec in workload.tasks) for field in ('cores_req', 'gpu_req', 'mem_req')]
    for actual, wanted in zip(resource_hours, expected):
        _close(actual, wanted, 'resource hours')
    return {'passed': True, 'arrived': len(tasks), 'completed': len(tasks), 'resource_hours': resource_hours,
            'misses': sum(row['deadline_outcome'] == 'missed' for row in tasks.values()),
            'observation_end': result.steps[-1]['interval_end']}


def audit_pair(paired, workload, environment, capacity):
    if not paired.prefix_equal or paired.prefixes['force_execute'] != paired.prefixes['force_defer_once']:
        raise RuntimeError('Counterfactual prefix mismatch')
    reports = {name: audit_result(result, workload, environment, capacity)
               for name, result in paired.results.items()}
    if any(report['observation_end'] != paired.common_end.isoformat() for report in reports.values()):
        raise RuntimeError('Common observation endpoint mismatch')
    for name, rows in paired.interventions.items():
        if len(rows) != 1 or rows[0]['changed_defer'] != (name == 'force_defer_once'):
            raise RuntimeError('Single intervention action audit mismatch')
        actual = [event for event in paired.results[name].policy_events
                  if event['task_id'] == paired.target.task_id and event['time'] == paired.target.time.isoformat()]
        expected = 'started' if name == 'force_execute' else 'policy_deferred'
        if len(actual) != 1 or actual[0]['event'] != expected or actual[0]['reason'] != name or not actual[0]['can_fit_at_decision']:
            raise RuntimeError('Actual target decision differs from registered intervention')
    execute, defer = paired.results['force_execute'], paired.results['force_defer_once']
    policy_difference = defer.policy_summary['policy_defer_intervals'] - execute.policy_summary['policy_defer_intervals']
    if policy_difference != 1:
        raise RuntimeError('Only one actual policy defer is permitted')
    for left, right in zip(reports['force_execute']['resource_hours'], reports['force_defer_once']['resource_hours']):
        _close(left, right, 'paired resource hours')
    return {'version': VERSION, 'passed': True, 'prefix_equal': True,
            'arrived': len(workload.tasks), 'completed_per_branch': len(workload.tasks),
            'resource_hours_equal': True, 'delta_policy_defer_intervals': policy_difference, 'branches': reports}
