"""Pure ordering/eligibility rules; no execution, mutable Tasks or future CI."""
from dataclasses import dataclass
from typing import ClassVar, Protocol

import pandas as pd

from .contracts import STEP, finite, utc_time
from .workload import TaskSpec

POLICY_SCHEMA = 'e1-1b-policy-v1'
POLICY_NAMES = ('immediate_fifo', 'edf_backfill', 'carbon_threshold_fifo')


@dataclass(frozen=True)
class DecisionContext:
    now: pd.Timestamp
    ci_g_per_kwh: float

    def __post_init__(self):
        object.__setattr__(self, 'now', utc_time(self.now, True))
        object.__setattr__(self, 'ci_g_per_kwh', finite(self.ci_g_per_kwh, 'current CI'))


@dataclass(frozen=True)
class PolicyDecision:
    defer: bool
    reason: str


def latest_start(spec):
    """Solo theoretical start bound; does not reserve future capacity."""
    return spec.sla_deadline - pd.Timedelta(minutes=spec.duration)


class SchedulingPolicy(Protocol):
    name: str

    def order(self, waiting: tuple[TaskSpec, ...]) -> tuple[TaskSpec, ...]: ...
    def decide(self, spec: TaskSpec, context: DecisionContext) -> PolicyDecision: ...


class ImmediateFIFO:
    name = 'immediate_fifo'

    def order(self, waiting):
        return waiting

    def decide(self, spec, context):
        return PolicyDecision(False, 'eligible')


class EDFBackfill(ImmediateFIFO):
    name = 'edf_backfill'

    def order(self, waiting):
        return tuple(sorted(waiting, key=lambda s: (s.sla_deadline, s.arrival_time, s.task_id)))


@dataclass(frozen=True)
class CarbonThresholdFIFO(ImmediateFIFO):
    threshold_g_per_kwh: float
    name: ClassVar[str] = 'carbon_threshold_fifo'

    def __post_init__(self):
        object.__setattr__(self, 'threshold_g_per_kwh', finite(self.threshold_g_per_kwh, 'carbon threshold'))

    def decide(self, spec, context):
        if context.ci_g_per_kwh <= self.threshold_g_per_kwh:
            return PolicyDecision(False, 'ci_at_or_below_threshold')
        bound = latest_start(spec)
        if context.now >= bound:
            return PolicyDecision(False, 'latest_start_reached_or_passed')
        if context.now + STEP > bound:
            return PolicyDecision(False, 'insufficient_one_step_slack')
        return PolicyDecision(True, 'high_ci_and_safe_one_step_slack')


def make_policy(config):
    """Resolve only literal policy configuration; never inspect input data."""
    name, params = config.get('policy'), config.get('policy_params', {})
    if name not in POLICY_NAMES or not isinstance(params, dict):
        raise ValueError('Unsupported policy or policy_params')
    if name == 'carbon_threshold_fifo':
        if set(params) != {'threshold_g_per_kwh'}:
            raise ValueError('Carbon policy requires only a fixed threshold_g_per_kwh')
        return CarbonThresholdFIFO(params['threshold_g_per_kwh'])
    if params:
        raise ValueError('Threshold parameters are valid only for Carbon-Threshold/FIFO')
    return EDFBackfill() if name == 'edf_backfill' else ImmediateFIFO()
