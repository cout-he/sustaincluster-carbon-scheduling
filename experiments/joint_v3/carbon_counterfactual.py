"""Complete-work IT+HVAC carbon estimates from isolated canonical model probes.

These are predicted gains under a frozen continuation, never measured realized
reductions. Both paths use the same persistent current weather and lower the
CRAC setpoint by one degree per slot toward 18 C. The physical model is algebraic;
the probe's sensor writes cannot alter the executing environment.
"""
from collections import OrderedDict
from dataclasses import asdict, dataclass
import math
from time import perf_counter

import numpy as np

from .uncertainty import robust_gain


@dataclass(frozen=True)
class CarbonEstimate:
    mean_carbon_gain_kg: float
    std_carbon_gain_kg: float
    robust_carbon_gain_kg: float
    forecast_confidence: float
    counterfactual_valid: bool
    tail_liability_kg: float
    estimator_runtime_ms: float
    reason: str
    start_settled: bool
    wait_settled: bool
    evaluated_slots: int
    work_outside_forecast: bool
    accounting_boundary: str = 'joint_full_hvac_v1'
    continuation: str = 'lower_1c_per_slot_toward_18c_persistent_current_weather'
    tail_liability_known: bool = True

    def to_dict(self):
        return asdict(self)


class CarbonCounterfactual:
    def __init__(self, power_factory, config=None):
        self.power_factory = power_factory
        config = config or {}

        def option(name, default):
            return config.get(name, default) if isinstance(config, dict) else getattr(config, name, default)

        self.kappa = float(option('kappa', 1.))
        self.max_slots = int(option('max_slots', 192))
        self.default_capacity = option('capacity', None)
        self.cache_limit = int(option('cache_limit', 65536))
        if (not np.isfinite(self.kappa) or self.kappa < 0 or self.max_slots < 16
                or self.max_slots > 192 or self.cache_limit < 1):
            raise ValueError('Invalid frozen counterfactual uncertainty/planning limits')
        self._probe = None
        self._cache = OrderedDict()
        self.cache_hits = self.cache_misses = 0

    @property
    def cache_hit_rate(self):
        return self.cache_hits/max(1, self.cache_hits+self.cache_misses)

    def _power(self, utilization, sample, setpoint):
        key = (*map(float, utilization), float(sample.temperature_c),
               float(sample.wet_bulb_c), float(setpoint))
        if key in self._cache:
            self.cache_hits += 1
            self._cache.move_to_end(key)
            return self._cache[key]
        self.cache_misses += 1
        if self._probe is None:
            self._probe = self.power_factory()
        self._probe.set_setpoint(float(setpoint))
        reading = self._probe.measure(*utilization, sample)
        if getattr(reading, 'accounting_boundary', None) != 'joint_full_hvac_v1':
            raise ValueError('Counterfactual requires canonical joint_full_hvac_v1 accounting')
        total = float(reading.total_kw)
        if not np.isfinite(total) or total < 0:
            raise ValueError('Canonical power returned invalid total kW')
        self._cache[key] = total
        if len(self._cache) > self.cache_limit:
            self._cache.popitem(last=False)
        return total

    @staticmethod
    def _loads(schedule):
        loads = np.asarray(schedule.loads, dtype=float)
        if loads.size == 0:
            loads = np.zeros((0, 3))
        if loads.ndim != 2 or loads.shape[1] != 3 or not np.isfinite(loads).all() or (loads < 0).any():
            raise ValueError('Counterfactual schedule must contain finite nonnegative raw resource loads')
        return loads

    def estimate(self, start_schedule, wait_schedule, forecast, current_sample, setpoint_c, *, capacity=None):
        started = perf_counter()
        raw_capacity = capacity if capacity is not None else self.default_capacity
        if raw_capacity is None:
            raw_capacity = getattr(start_schedule, 'capacity', None)
        cap = np.asarray(raw_capacity, dtype=float)
        if cap.shape != (3,) or not np.isfinite(cap).all() or (cap <= 0).any():
            raise ValueError('CPU/GPU/MEM capacities are required for physical utilization conversion')
        if not np.isfinite(setpoint_c) or not 18 <= setpoint_c <= 22:
            raise ValueError('Counterfactual setpoint must be within 18..22 C')
        start, wait = self._loads(start_schedule), self._loads(wait_schedule)
        if (start > cap*(1+1e-10)).any() or (wait > cap*(1+1e-10)).any():
            raise ValueError('Counterfactual raw resource loads exceed physical capacities')
        horizon = len(forecast.mean)
        scenarios = np.asarray(forecast.scenarios, dtype=float)
        if (horizon < 1 or scenarios.ndim != 2 or scenarios.shape[1] != horizon
                or not scenarios.shape[0] or not np.isfinite(scenarios).all() or (scenarios < 0).any()):
            raise ValueError('Counterfactual requires finite nonnegative CI scenario trajectories')
        common_length = max(horizon, len(start), len(wait))
        slots = min(self.max_slots, common_length)
        start_settled = bool(start_schedule.settled and len(start) <= self.max_slots)
        wait_settled = bool(wait_schedule.settled and len(wait) <= self.max_slots)
        start_u, wait_u = np.zeros((slots, 3)), np.zeros((slots, 3))
        start_u[:min(slots, len(start))] = np.minimum(1., start[:slots]/cap)
        wait_u[:min(slots, len(wait))] = np.minimum(1., wait[:slots]/cap)
        work_outside = bool((start[horizon:] > 0).any() or (wait[horizon:] > 0).any())
        reasons = []
        if not forecast.valid:
            reasons.append('forecast_invalid:' + forecast.reason)
        if not start_schedule.valid or not wait_schedule.valid:
            reasons.append('schedule_invalid:' + str(start_schedule.reason) + '/' + str(wait_schedule.reason))
        if not start_settled or not wait_settled:
            reasons.append('unsettled_known_work')
        if work_outside:
            reasons.append('known_work_outside_ci_forecast')
        if set(start_schedule.finishes) != set(wait_schedule.finishes):
            reasons.append('different_known_work_sets')
        if common_length > self.max_slots:
            reasons.append('planning_window_capped')
        z = np.maximum(18., float(setpoint_c)-np.arange(1, slots+1))
        start_power = np.array([self._power(u, current_sample, temp) for u, temp in zip(start_u, z)])
        wait_power = np.array([self._power(u, current_sample, temp) for u, temp in zip(wait_u, z)])
        delta = start_power-wait_power
        # The same conservative, constant tail CI is used by both paths.
        tail_ci = max(float(current_sample.ci_g_per_kwh), float(np.max(forecast.mean)),
                      float(np.max(scenarios)))
        ci = np.full((len(scenarios), slots), tail_ci)
        inside = min(slots, horizon)
        ci[:, :inside] = scenarios[:, :inside]
        gains = ci@delta*.25/1000
        mean, std, robust = robust_gain(gains, self.kappa)
        liability = 0.
        liability_known = True
        if work_outside:
            idle = np.array([self._power(np.zeros(3), current_sample, temp) for temp in z])
            liability = max(float(np.maximum(0., start_power[horizon:]-idle[horizon:]).sum()),
                            float(np.maximum(0., wait_power[horizon:]-idle[horizon:]).sum()))*tail_ci*.25/1000
        if not start_settled or not wait_settled:
            durations = [getattr(plan, 'unresolved_duration_minutes', None)
                         for plan in (start_schedule, wait_schedule)]
            liability_known = all(value is not None and np.isfinite(value) and value >= 0
                                  for value in durations)
            remaining_hours = max(durations)/60 if liability_known else 0.
            # The planner provides a serial-duration upper bound of known work
            # beyond its cap. Include it rather than silently dropping long jobs.
            # Early invalid snapshots cannot establish a finite remaining bound;
            # their finite planning penalty is explicitly labelled unknown.
            full_kw = self._power(np.ones(3), current_sample, 18.)
            liability = max(liability, full_kw*tail_ci*(self.max_slots*.25+remaining_hours)/1000)
            if not liability_known:
                reasons.append('unknown_remaining_work_liability')
        valid = not reasons
        if not valid:
            robust = min(0., robust)
        confidence = float(forecast.confidence) if valid else 0.
        return CarbonEstimate(mean, std, robust, confidence, valid, float(liability),
                              (perf_counter()-started)*1000, 'ok' if valid else ';'.join(reasons),
                              start_settled, wait_settled, slots, work_outside,
                              tail_liability_known=liability_known)

    def fast_estimate(self, candidate, forecast, current_sample, setpoint_c, *, capacity, current_used):
        """Two-probe marginal-power proxy, without a known-queue safety claim.

        Hold the observed background load, current setpoint and weather fixed.
        Both options execute the candidate's entire ceil-15-minute occupancy.
        Their overlapping slots cancel algebraically, leaving CI0 minus CI at
        its final shifted slot. This keeps computation independent of duration
        while preserving equal work. The caller separately caps proxy calls.
        """
        started = perf_counter()
        cap, used = np.asarray(capacity, dtype=float), np.asarray(current_used, dtype=float)
        demand = np.asarray((candidate.cores_req, candidate.gpu_req, candidate.mem_req), dtype=float)
        if (cap.shape != (3,) or used.shape != (3,) or demand.shape != (3,)
                or not np.isfinite(cap).all() or not np.isfinite(used).all()
                or not np.isfinite(demand).all() or (cap <= 0).any()
                or (used < 0).any() or (demand < 0).any()
                or (used+demand > cap).any()):
            raise ValueError('Fast marginal proxy requires a physically feasible raw resource capacity snapshot')
        if not np.isfinite(setpoint_c) or not 18 <= setpoint_c <= 22:
            raise ValueError('Fast marginal proxy setpoint must be within 18..22 C')
        duration = float(candidate.duration)
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError('Fast marginal proxy duration must be positive and finite')
        occupancy = math.ceil(duration/15.)
        ci = np.asarray(forecast.scenarios, dtype=float)
        horizon = len(forecast.mean)
        if (ci.ndim != 2 or ci.shape[1] != horizon or not ci.shape[0] or horizon < 1
                or not np.isfinite(ci).all() or (ci < 0).any()):
            raise ValueError('Fast marginal proxy requires nonnegative finite CI scenario trajectories')
        base_kw = self._power(used/cap, current_sample, float(setpoint_c))
        with_candidate_kw = self._power((used+demand)/cap, current_sample, float(setpoint_c))
        marginal_kw = with_candidate_kw-base_kw
        outside = occupancy >= horizon
        tail_ci = max(float(current_sample.ci_g_per_kwh), float(np.max(forecast.mean)), float(ci.max()))
        # Equal-work constant-power trajectories telescope, including any common
        # constant-CI tail. No unknown future observation is accepted or queried.
        last_ci = np.full(len(ci), tail_ci) if outside else ci[:, occupancy]
        gains = marginal_kw*(ci[:, 0]-last_ci)*.25/1000
        mean, std, robust = robust_gain(gains, self.kappa)
        tail_slots = max(0, occupancy-horizon+1)
        liability = max(0., marginal_kw)*tail_ci*.25*tail_slots/1000
        if not np.isfinite(liability):
            raise ValueError('Fast marginal proxy tail liability exceeds finite numerical range')
        reasons = ['fast_marginal_proxy_no_queue_counterfactual']
        if not forecast.valid:
            reasons.append('forecast_invalid:' + forecast.reason)
        if outside:
            reasons.append('known_work_outside_ci_forecast')
        # Even a within-horizon proxy does not settle the entire known queue.
        # Preserve its quantitative mean/std, but never supply positive guidance.
        return CarbonEstimate(mean, std, min(0., robust), 0., False, float(liability),
            (perf_counter()-started)*1000, ';'.join(reasons), False, False,
            min(self.max_slots, max(horizon, occupancy+1)), outside,
            continuation='frozen_current_background_load_setpoint_weather_equal_candidate_work',
            tail_liability_known=True)

    def close(self):
        if self._probe is not None:
            self._probe.close()
            self._probe = None
