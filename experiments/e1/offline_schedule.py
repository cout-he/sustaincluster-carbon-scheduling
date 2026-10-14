"""Bounded mandatory offline oracle and common-executor conditional replay.

All future background inspection is explicitly offline. These wrappers are not
online policy registrations and are never offered to PPO or an observation.
"""
import math
import time
from dataclasses import dataclass, replace
from fractions import Fraction
from functools import reduce

import pandas as pd

from .contracts import STEP, content_hash, finite, utc_time
from .offline_schedule_cases import RESOURCE_FIELDS, RESOURCE_NAMES
from .offline_schedule_verify import normalize_starts, verify_schedule
from .policies import EDFBackfill, PolicyDecision

SOLUTION_SCHEMA = 'e1-7a-offline-solution-v1'
OPTIMALITY_SCOPE = ('Only the mandatory finite integer-resource subproblem is optimized; '
                    'conservative quantization and original fsum rounding boundaries prevent '
                    'an unconditional original E1 floating-resource optimum claim, even for '
                    'the finite selected case. This is never a full-cluster optimum.')
_SAFE_INT = ((1 << 63) - 1) // 4


@dataclass(frozen=True)
class SolverBudget:
    max_seconds: float = 120
    workers: int = 1
    seed: int = 710001
    conservative_scale: int = 1_000_000

    def __post_init__(self):
        if finite(self.max_seconds, 'solver budget', True) > 120:
            raise ValueError('E1-7A per-case solver budget must not exceed 120 seconds')
        if any(isinstance(v, bool) or not isinstance(v, int)
               for v in (self.workers, self.seed, self.conservative_scale)):
            raise ValueError('Solver workers, seed and resource scale must be integers')
        if self.workers != 1 or self.seed != 710001 or self.conservative_scale != 1_000_000:
            raise ValueError('E1-7A uses worker=1, seed=710001 and fixed scale=1000000')


@dataclass(frozen=True)
class ScheduleSolution:
    status: str
    starts: tuple
    miss_count: object
    best_bound: object
    wall_seconds: float
    solver_version: str
    precision: tuple
    verification: object
    case_sha256: str
    optimality_scope: str = OPTIMALITY_SCOPE
    objective: str = 'minimum_deadline_miss_count_only'

    def to_dict(self):
        return dict(schema=SOLUTION_SCHEMA, status=self.status,
                    starts=[[key, value.isoformat()] for key, value in self.starts],
                    miss_count=self.miss_count, best_bound=self.best_bound,
                    wall_seconds=self.wall_seconds, solver_version=self.solver_version,
                    precision=list(self.precision), verification=self.verification,
                    case_sha256=self.case_sha256, optimality_scope=self.optimality_scope,
                    objective=self.objective)

    @classmethod
    def from_dict(cls, value):
        data = dict(value)
        if data.pop('schema', None) != SOLUTION_SCHEMA:
            raise ValueError('Offline solution schema mismatch')
        data['starts'] = tuple((key, utc_time(time, True)) for key, time in data['starts'])
        data['precision'] = tuple(data['precision'])
        return cls(**data)


def integer_resources(case, budget):
    """Try the minimal exact binary lattice, otherwise round conservatively.

    Exact fractions avoid a floating ceil which could round demand downward.
    The int64 guard includes the sum of all possible cumulative demands, not
    just an individual coefficient. The CP-SAT model is validated separately.
    """
    maps = []
    for resource, (field, name, capacity) in enumerate(zip(RESOURCE_FIELDS, RESOURCE_NAMES, case.capacity)):
        ids = [s.task_id for s in case.tasks] + [b.task_id for b in case.background]
        values = [float(getattr(s, field)) for s in case.tasks] + [b.demand[resource] for b in case.background]
        fractions = [Fraction(v) for v in values] + [Fraction(capacity)]
        denominator = max((q.denominator for q in fractions), default=1)
        numerators = [q.numerator * (denominator // q.denominator) for q in fractions]
        divisor = reduce(math.gcd, numerators, 0) or 1
        exact = [n // divisor for n in numerators]
        safe = max(exact, default=0) <= _SAFE_INT and sum(exact[:-1]) <= _SAFE_INT
        if safe:
            integers = exact
            record = dict(resource=name, mode='exact_binary', unit_numerator=divisor,
                          unit_denominator=denominator, scale=None,
                          maximum_demand_error=0.0, sum_demand_errors=0.0, capacity_loss=0.0)
        else:
            scale = budget.conservative_scale
            integers = [(q.numerator * scale + q.denominator - 1) // q.denominator
                        for q in fractions[:-1]]
            integers.append((fractions[-1].numerator * scale) // fractions[-1].denominator)
            if max(integers, default=0) > _SAFE_INT or sum(integers[:-1]) > _SAFE_INT:
                raise ValueError('Fixed conservative mapping still exceeds safe int64 range')
            errors = [float(Fraction(n, scale) - q) for n, q in zip(integers[:-1], fractions[:-1])]
            record = dict(resource=name, mode='conservative_decimal', unit_numerator=1,
                          unit_denominator=scale, scale=scale,
                          maximum_demand_error=max(errors, default=0.0),
                          sum_demand_errors=math.fsum(errors),
                          capacity_loss=float(fractions[-1] - Fraction(integers[-1], scale)))
        record.update(capacity_integer=integers[-1],
                      candidate_demands=dict(zip(ids[:len(case.tasks)], integers[:len(case.tasks)])),
                      background_demands=dict(zip(ids[len(case.tasks):], integers[len(case.tasks):-1])),
                      original_capacity=capacity,
                      int64_guard='coefficients and all cumulative demands <= INT64_MAX/4')
        maps.append(record)
    return tuple(maps)


def _precision_summary(mapping):
    result = {key: value for key, value in mapping.items()
              if key not in ('candidate_demands', 'background_demands')}
    result.update(candidate_count=len(mapping['candidate_demands']),
                  background_count=len(mapping['background_demands']),
                  coefficient_sha256=content_hash(dict(candidate=mapping['candidate_demands'],
                                                       background=mapping['background_demands'])))
    return result


def solve_deadline_case(case, budget=None):
    """Minimize misses only; no dropped jobs and no arbitrary weighted target."""
    import ortools
    from ortools.sat.python import cp_model
    budget = SolverBudget() if budget is None else budget
    mappings = integer_resources(case, budget)
    horizon = int((case.end - case.start) / STEP)
    model = cp_model.CpModel()
    variables, intervals, misses = {}, {}, []
    for spec in case.tasks:
        raw_duration = pd.Timedelta(minutes=spec.duration)
        occupied = (raw_duration.value + STEP.value - 1) // STEP.value
        lower = max(0, int((spec.arrival_time - case.start) // STEP),
                    int((case.candidate_not_before - case.start) // STEP))
        upper = horizon - occupied
        # Explicitly infeasible domains are represented by a false constraint;
        # all jobs still remain mandatory in the model.
        start = model.NewIntVar(lower, max(lower, upper), f'start:{spec.task_id}')
        release = model.NewIntVar(lower + occupied, max(lower, upper) + occupied,
                                  f'release:{spec.task_id}')
        model.Add(release <= horizon)
        intervals[spec.task_id] = model.NewIntervalVar(start, occupied, release, f'hold:{spec.task_id}')
        variables[spec.task_id] = start
        latest = (spec.sla_deadline.value - raw_duration.value - case.start.value) // STEP.value
        missed = model.NewBoolVar(f'missed:{spec.task_id}')
        model.Add(start > latest).OnlyEnforceIf(missed)
        model.Add(start <= latest).OnlyEnforceIf(missed.Not())
        misses.append(missed)
    background_intervals = {}
    for b in case.background:
        first = max(0, int((b.start - case.start) // STEP))
        last = min(horizon, int((b.release - case.start) // STEP))
        if first < last:
            background_intervals[b.task_id] = model.NewIntervalVar(first, last - first, last,
                                                                   f'fixed-background:{b.task_id}')
    for mapping in mappings:
        model.AddCumulative(list(intervals.values()) + list(background_intervals.values()),
            [mapping['candidate_demands'][key] for key in intervals] +
            [mapping['background_demands'][key] for key in background_intervals], mapping['capacity_integer'])
    model.Minimize(sum(misses))
    invalid = model.Validate()
    if invalid:
        raise ValueError(f'CP-SAT model validation failed: {invalid}')
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = budget.max_seconds
    solver.parameters.num_search_workers = budget.workers
    solver.parameters.random_seed = budget.seed
    status_code = solver.Solve(model)
    status = solver.StatusName(status_code)
    starts, checked, count = (), None, None
    if status in ('OPTIMAL', 'FEASIBLE'):
        starts = tuple(sorted((key, case.start + solver.Value(value) * STEP)
                              for key, value in variables.items()))
        checked = verify_schedule(case, starts)
        count = int(round(solver.ObjectiveValue()))
        if count != checked['miss_count']:
            raise RuntimeError('Solver miss count differs from original-clock verification')
    return ScheduleSolution(status, starts, count, solver.BestObjectiveBound(), solver.WallTime(),
                            ortools.__version__, tuple(_precision_summary(m) for m in mappings),
                            checked, case.sha256)


class OfflineSchedulePolicy(EDFBackfill):
    """Offline start commitments; capacity allocation remains LocalExecutor's."""
    name = 'offline_fixed_schedule'

    def __init__(self, case, starts, conditional_edf=False):
        self.case, self.starts = case, normalize_starts(starts)
        self.background = {b.task_id: b for b in case.background}
        self.candidates = {s.task_id: s for s in case.tasks}
        self.conditional_edf = conditional_edf
        self.selected_running = []

    def order(self, waiting):
        return tuple(sorted(waiting, key=lambda s: (
            0 if s.task_id in self.background else 1, s.sla_deadline, s.arrival_time, s.task_id)))

    def on_step_start(self, snapshot):
        self.now = snapshot.now
        self.selected_running = [r for r in snapshot.running if r.task_id in self.candidates]

    def on_task_started(self, running):
        if running.task_id in self.candidates:
            self.selected_running.append(running)

    def _future_background_fits(self, spec, now):
        release = (now + pd.Timedelta(minutes=spec.duration)).ceil('15min')
        if release > self.case.end:
            return False
        demand = tuple(getattr(spec, field) for field in RESOURCE_FIELDS)
        for t in pd.date_range(now, release, freq=STEP, inclusive='left'):
            active = [b.demand for b in self.case.background if b.start <= t < b.release]
            active += [r.demand for r in self.selected_running if t < r.release_time]
            if any(math.fsum([d[resource] for d in active] + [demand[resource]]) > capacity
                   for resource, capacity in enumerate(self.case.capacity)):
                return False
        return True

    def decide(self, spec, context):
        if spec.task_id in self.background:
            promised = self.background[spec.task_id].start
        elif spec.task_id not in self.candidates:
            raise ValueError('Task not present in offline candidates/background')
        elif self.conditional_edf:
            if context.now < self.case.candidate_not_before:
                return PolicyDecision(True, 'offline_before_candidate_anchor')
            return PolicyDecision(not self._future_background_fits(spec, context.now),
                                  'offline_conditional_edf_background_reservation')
        else:
            promised = self.starts[spec.task_id]
        if context.now > promised:
            raise RuntimeError('Offline fixed start commitment was missed')
        return PolicyDecision(context.now < promised, 'offline_fixed_start_commitment')


class _NoEnergyDiagnostic:
    """A-only clock/resource tests do not claim physical energy measurements."""
    def measure(self, cpu, gpu, mem, sample):
        from .power import PowerReading
        return PowerReading(0, 0, 0)


def _default_inputs(case):
    from .environment import EnvironmentalSample, FrozenEnvironment
    from .workload import FrozenWorkload
    if case.mode != 'A1':
        raise ValueError('A2 requires complete original workload and corresponding environment')
    end_arrivals = max((s.arrival_time + STEP for s in case.tasks), default=case.start + STEP)
    workload = FrozenWorkload(case.start, end_arrivals, case.source_workload_sha256,
                              5, 1.5, 2020, case.tasks)
    environment = FrozenEnvironment(case.start, case.end, 'offline-no-ci', 'offline-no-weather', tuple(
        EnvironmentalSample(t, 0, 20, 18) for t in pd.date_range(case.start, case.end, freq=STEP, inclusive='left')))
    return workload, environment


def _check_replay_inputs(case, workload, environment):
    specs = {s.task_id: s for s in workload.tasks}
    candidate_ids, background_ids = {s.task_id for s in case.tasks}, {b.task_id for b in case.background}
    if set(specs) != candidate_ids | background_ids:
        raise ValueError('Replay requires exactly all original candidates plus fixed background')
    for s in case.tasks:
        if specs[s.task_id].to_dict() != s.to_dict():
            raise ValueError('Replay candidate input differs from frozen original')
    for b in case.background:
        spec = specs[b.task_id]
        if b.start < spec.arrival_time or (b.start + pd.Timedelta(minutes=spec.duration)).ceil('15min') != b.release:
            raise ValueError('Fixed background does not match original arrival/duration')
        if b.demand != tuple(getattr(spec, f) for f in RESOURCE_FIELDS):
            raise ValueError('Fixed background demand differs from original task')
    if environment.end < case.end:
        raise ValueError('Replay environmental coverage shorter than case horizon')


def _run(case, starts, workload, environment, power_model, conditional_edf):
    from .engine import LocalExecutor
    _check_replay_inputs(case, workload, environment)
    policy = OfflineSchedulePolicy(case, starts, conditional_edf)
    executor = LocalExecutor(workload, environment, case.capacity, power_model, policy, drain=True)
    executor.start(until=case.end)
    while not executor.finished:
        now = executor.sample.time
        record = executor.advance()
        actual = set(executor.policy_steps[-1]['started_ids'])
        required = {b.task_id for b in case.background if b.start == now}
        if not conditional_edf:
            required |= {key for key, start in policy.starts.items() if start == now}
        if not required <= actual:
            raise RuntimeError('A fixed offline start could not execute under original resources')
    result = executor.result()
    if len(result.tasks) != len(workload.tasks) or any(row['state'] != 'completed' for row in result.tasks):
        raise ValueError('Offline reference horizon did not complete all mandatory original work')
    if isinstance(power_model, _NoEnergyDiagnostic):
        result = replace(result, summary=dict(result.summary,
            offline_zero_power_diagnostic=True, is_cluster_carbon_result=False,
            energy_accounting='unmodeled_zero_power_clock_and_resource_test_only'))
    return result


def replay_schedule(case, starts, workload=None, environment=None, power_model=None):
    """Physically replay an already verified schedule using the single executor.

    Real energy requires an explicitly supplied real frozen environment and
    UpstreamPowerModel. The A-only defaults deliberately measure no energy.
    """
    verify_schedule(case, starts)
    if (workload is not None or environment is not None) and power_model is None:
        raise ValueError('Explicit frozen replay inputs require an explicit power model')
    if workload is None and environment is None:
        workload, environment = _default_inputs(case)
    if workload is None or environment is None:
        raise ValueError('Replay workload and environment must be supplied together')
    return _run(case, starts, workload, environment,
                power_model if power_model is not None else _NoEnergyDiagnostic(), False)


def run_conditional_edf(case, *, workload=None, environment=None, power_model=None):
    """Same fixed background; EDF candidates may not break future commitments.

    For A1 this is the normal EDF rule through LocalExecutor. A2 adds an
    explicitly offline whole-hold reservation check, not an online algorithm.
    """
    began = time.perf_counter()
    if (workload is not None or environment is not None) and power_model is None:
        raise ValueError('Explicit frozen EDF inputs require an explicit power model')
    if workload is None and environment is None:
        workload, environment = _default_inputs(case)
    if workload is None or environment is None:
        raise ValueError('EDF workload and environment must be supplied together')
    result = _run(case, {}, workload, environment,
                  power_model if power_model is not None else _NoEnergyDiagnostic(), True)
    ids = {s.task_id for s in case.tasks}
    starts = tuple(sorted((row['task_id'], utc_time(row['start_time'], True))
                          for row in result.tasks if row['task_id'] in ids))
    checked = verify_schedule(case, starts)
    return ScheduleSolution('EDF_REFERENCE', starts, checked['miss_count'], None,
                            time.perf_counter() - began, 'LocalExecutor',
                            (), checked, case.sha256,
                            'Reference schedule only; not an optimality bound', 'conditional_edf_reference')
