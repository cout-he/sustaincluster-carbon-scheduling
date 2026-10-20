"""Fixed E2 aggregate observation and actual-step reward accounting.

Only V2 observations enter the policy state. The public idle background is
read at the matching slot after execution and never enters the encoder.
This module has no tensor library, input loader, training, or scenario I/O.
"""
import math
import statistics

from experiments.platform_v2.environment import PlatformEnv
from .adapter import ResearchAdapter
from .completion_guard import CompletionGuard, CLEAR_MINUTE, HORIZON_MINUTE, FAILURE
from .edf_calendar import STEP, grid_release, simulate


SLA_MULTIPLIER = 1.5
STATE_NAMES = (
    'time_sin', 'time_cos', 'elapsed_fraction', 'completion_budget_fraction', 'phase',
    'ci_current', 'ci_mean_4h', 'ci_median_24h', 'ci_relative_median', 'ci_change_1h',
    'cpu_used_fraction', 'gpu_used_fraction', 'mem_used_fraction',
    'cpu_next_release_fraction', 'gpu_next_release_fraction',
    'pending_count', 'reference_start_count', 'pending_cpu_grid_hours',
    'pending_gpu_grid_hours', 'candidate_dynamic_power', 'reference_slack_min',
    'reference_slack_median', 'reference_slack_below_slot_fraction',
    'reference_late_fraction', 'inlet_temperature', 'outdoor_temperature',
    'failure_latched',
)
STATE_SCHEMA = {
    'version': 'sca-ppo-e2-state-v1',
    'names': STATE_NAMES,
    'fixed_scales': {
        'day_minutes': 1440., 'horizon_minutes': HORIZON_MINUTE,
        'completion_minutes': CLEAR_MINUTE, 'ci_g_per_kwh': 1000.,
        'queue_count': 1000., 'resource_hours_days': 24.,
        'slack_minutes': 1440., 'temperature_c': 40.,
        'cpu_dynamic_kw_per_equivalent': .005,
        'gpu_dynamic_kw_per_equivalent': .225, 'psu_efficiency': .94,
    },
    'ci_history': 'strictly minute < now; inclusive 4h/24h lower boundaries; '
                  'empty mean/median use current; change uses latest minute <= now-60',
    'phase': '0 before 1440; 0.5 before 3960; 1 thereafter',
    'resource_hours': 'ceil(duration/15 - 1e-12) * 0.25 * demand / (capacity * 24)',
    'candidate_power': 'sum Pdyn(reference current-slot starters) / Pdyn(logical capacity)',
    'reference_slack': 'deadline minus raw reference finish, pending tasks only',
    'reference_late_fraction': 'reference late IDs / pending plus running reference records',
    'clipping': 'remaining completion budget has a zero floor; no other clipping',
}


def _dynamic_power(cpu, gpu):
    return (.005 * cpu + .225 * gpu) / .94


def state_vector(obs, reference_plan, failure_latched=False):
    """Return the versioned 27 floats using current/past public observations.

    Reference future finishes are predictions for already arrived tasks, not
    future task or environmental data. Empty queue aggregates are zero.
    """
    now = obs['minute']
    current = obs['ci_g_per_kwh']
    history = [(minute, ci) for minute, ci in obs['ci_history'] if minute < now]
    recent = [ci for minute, ci in history if minute >= now - 240]
    day = [ci for minute, ci in history if minute >= now - 1440]
    mean = statistics.mean(recent) if recent else current
    median = statistics.median(day) if day else current
    old = [(minute, ci) for minute, ci in history if minute <= now - 60]
    change = current - max(old, key=lambda pair: pair[0])[1] if old else 0.
    capacity = obs['capacity']
    used = [(cap - available) / cap for cap, available in zip(capacity, obs['available'])]
    release = [math.fsum(demand[axis] for _, demand, finish in obs['running']
                         if grid_release(finish) <= now + STEP) / capacity[axis]
               for axis in (0, 1)]
    pending = obs['pending']
    by_id = reference_plan.by_id
    candidates = set(reference_plan.first_slot.started)
    work = [math.fsum(math.ceil(t.duration_minutes / STEP - 1e-12) * .25 * t.demand[axis]
                      for t in pending) / (capacity[axis] * 24.) for axis in (0, 1)]
    candidate_power = math.fsum(_dynamic_power(t.demand[0], t.demand[1])
                                for t in pending if t.task_id in candidates)
    full_power = _dynamic_power(capacity[0], capacity[1])
    slack = [by_id[t.task_id].deadline - by_id[t.task_id].finish for t in pending]
    phase = 0. if now < 1440 else .5 if now < CLEAR_MINUTE else 1.
    values = [math.sin(2 * math.pi * now / 1440.), math.cos(2 * math.pi * now / 1440.),
              now / HORIZON_MINUTE, max(CLEAR_MINUTE - now, 0) / CLEAR_MINUTE, phase,
              current / 1000., mean / 1000., median / 1000.,
              (current - median) / max(abs(median), 1.), change / 1000.,
              *used, *release, len(pending) / 1000., len(candidates) / 1000., *work,
              candidate_power / full_power,
              min(slack) / 1440. if slack else 0.,
              statistics.median(slack) / 1440. if slack else 0.,
              sum(value < STEP for value in slack) / len(slack) if slack else 0.,
              len(reference_plan.late_ids) / len(reference_plan.records) if reference_plan.records else 0.,
              obs['inlet_c'] / 40., obs['outdoor_c'] / 40., float(failure_latched)]
    values = [float(value) for value in values]
    if len(values) != len(STATE_NAMES) or not all(math.isfinite(value) for value in values):
        raise ValueError('State must contain exactly 27 finite scalars')
    return values


class RewardLedger:
    """Actual incremental carbon and irreversible miss IDs, with no duplicates."""

    def __init__(self, c_scale, lambda_miss):
        if (not math.isfinite(c_scale) or c_scale <= 0 or
                not math.isfinite(lambda_miss) or lambda_miss < 0):
            raise ValueError('Reward requires positive finite carbon scale and nonnegative miss penalty')
        self.c_scale = float(c_scale)
        self.lambda_miss = float(lambda_miss)
        self._seen_miss_ids = set()
        self._carbon_rewards = []
        self._miss_rewards = []

    @property
    def seen_miss_ids(self):
        return frozenset(self._seen_miss_ids)

    @property
    def carbon_reward_sum(self):
        return math.fsum(self._carbon_rewards)

    @property
    def miss_reward_sum(self):
        return math.fsum(self._miss_rewards)

    @property
    def total_reward(self):
        return math.fsum((self.carbon_reward_sum, self.miss_reward_sum))

    def add(self, carbon_kg, idle_carbon_kg, new_miss_ids):
        identities = tuple(new_miss_ids)
        if not math.isfinite(carbon_kg) or not math.isfinite(idle_carbon_kg):
            raise ValueError('Step carbon must be finite')
        if len(identities) != len(set(identities)) or self._seen_miss_ids.intersection(identities):
            raise ValueError('Irreversible miss ID was already penalized')
        carbon_reward = -(carbon_kg - idle_carbon_kg) / self.c_scale
        miss_reward = -self.lambda_miss * len(identities)
        reward = math.fsum((carbon_reward, miss_reward))
        if not all(math.isfinite(value) for value in (carbon_reward, miss_reward, reward)):
            raise ValueError('Reward must be finite')
        self._seen_miss_ids.update(identities)
        self._carbon_rewards.append(carbon_reward)
        self._miss_rewards.append(miss_reward)
        return dict(reward=reward, carbon_reward=carbon_reward, miss_reward=miss_reward,
                    new_miss_count=len(identities))


class RLEpisode:
    """One original V2 72-hour episode under E0 DDF/CG execution semantics.

    Eligible slots select one of the seven ordinary actions, including a
    deterministic policy's selected action. All forced slots require 0/False.
    The caller owns policy log probabilities; this wrapper never creates one.
    """

    def __init__(self, scenario, inv, cfg, idle_result, c_scale, lambda_miss):
        self.env = PlatformEnv(scenario, inv, cfg, SLA_MULTIPLIER, 'fixed')
        if self.env.horizon != HORIZON_MINUTE:
            raise ValueError('E2 requires the full original 72-hour responsibility window')
        if len(idle_result['steps']) != HORIZON_MINUTE // STEP:
            raise ValueError('Common idle background must cover all 288 slots')
        self.adapter = ResearchAdapter()
        self.idle_result = idle_result
        self.c_scale = c_scale
        self.lambda_miss = lambda_miss
        self.ledger = RewardLedger(c_scale, lambda_miss)
        self.records = []
        self.obs = None
        self._prepared = None

    def reset(self):
        self.obs = self.env.reset()
        self.adapter.reset()
        self.ledger = RewardLedger(self.c_scale, self.lambda_miss)
        self.records = []
        self._prepared = None
        return self.obs

    def prepare(self):
        if self.obs is None or self.env.minute >= self.env.horizon:
            raise RuntimeError('Prepare requires a reset, nonterminal episode')
        if self._prepared is None:
            snapshot = self.adapter.observe(self.obs)
            reference = simulate(snapshot)
            reference_pass = CompletionGuard.accepts(reference)
            failure = self.adapter.guard.failure_latched or not reference_pass
            state = state_vector(self.obs, reference, failure)
            eligible = bool(reference.first_slot.started) and not failure
            # accepts is pure: actual decide inspects this slot exactly once.
            self._prepared = (state, eligible)
        state, eligible = self._prepared
        return list(state), eligible

    def step(self, action_id, policy_sampled):
        state, eligible = self.prepare()
        if type(action_id) is not int or action_id not in range(7):
            raise ValueError('Action ID must be an integer in 0..6')
        if type(policy_sampled) is not bool or policy_sampled != eligible or (not eligible and action_id != 0):
            raise ValueError('Eligible slots require selected=True; forced slots require action 0/False')
        decision = self.adapter.decide(self.obs, action_id=action_id)
        obs, done, row = self.adapter.execute(self.env, decision)
        # No common idle value is consulted until the actual slot has executed.
        idle_row = self.idle_result['steps'][len(self.records)]
        if idle_row['minute'] != row['minute']:
            raise AssertionError('Common idle background slot is misaligned')
        terms = self.ledger.add(row['carbon_kg'], idle_row['carbon_kg'],
                                self.adapter.records[-1]['new_irreversible_miss_ids'])
        record = dict(self.adapter.records[-1])
        record.update(terms, action_id=action_id, proposed_action=action_id,
                      policy_sampled=policy_sampled, state=state, row=dict(row),
                      idle_carbon_kg=idle_row['carbon_kg'],
                      incremental_carbon_kg=row['carbon_kg'] - idle_row['carbon_kg'])
        self.records.append(record)
        self.obs = obs
        self._prepared = None
        return terms['reward'], done, record

    def finish(self):
        if self.obs is None or self.env.minute != HORIZON_MINUTE or len(self.records) != HORIZON_MINUTE // STEP:
            raise RuntimeError('Complete and account for all 288 actual slots before finish')
        result = self.env.result()
        summary = result['summary']
        adapter_summary = self.adapter.summary(self.env)
        summary.update(adapter_summary)
        idle = math.fsum(record['idle_carbon_kg'] for record in self.records)
        incremental = math.fsum(record['incremental_carbon_kg'] for record in self.records)
        reasons = []
        if adapter_summary['failure_latched']:
            reasons.append(FAILURE)
        if summary['unfinished']:
            reasons.append('UNFINISHED_TASKS')
        if summary['deadline_censored']:
            reasons.append('CENSORED_DEADLINES')
        if not adapter_summary['actual_common_idle_tail_valid']:
            reasons.append('COMMON_IDLE_TAIL_INVALID')
        if adapter_summary['actual_last_release_minute'] is not None and adapter_summary['actual_last_release_minute'] > CLEAR_MINUTE:
            reasons.append('COMPLETION_AFTER_CLEAR_LIMIT')
        expected = -incremental / self.c_scale - self.lambda_miss * len(self.ledger.seen_miss_ids)
        conservation_error = abs(self.ledger.total_reward - expected)
        if not reasons:
            official_miss_ids = {row['task_id'] for row in result['tasks'] if row['deadline_missed']}
            assert self.ledger.seen_miss_ids == official_miss_ids, 'Actual reward miss IDs disagree with complete V2 ledger'
            expected = -incremental / self.c_scale - self.lambda_miss * summary['deadline_missed']
            conservation_error = abs(self.ledger.total_reward - expected)
            assert math.isclose(self.ledger.total_reward, expected, rel_tol=1e-10, abs_tol=1e-10), 'Episode reward does not conserve actual carbon and miss counts'
        summary.update(common_idle_carbon_kg=idle, common_incremental_carbon_kg=incremental,
                       reward_sum=self.ledger.total_reward, carbon_reward_sum=self.ledger.carbon_reward_sum,
                       miss_reward_sum=self.ledger.miss_reward_sum,
                       reward_conservation_error=conservation_error,
                       protocol_valid=not reasons, failure_reasons=tuple(reasons))
        return result
