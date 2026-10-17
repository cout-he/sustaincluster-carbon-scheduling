"""Independent ex-post calibration from signals the physical episode observed.

Known-queue counterfactual calibration is not a causal estimate of the carbon
effect of the executing global policy. Jobs remain frozen to each paired
reservation hypothesis; only its forecast CI/persistent-weather inputs are
replaced by signals observed subsequently by the real physical episode.
No result from this module enters actors, critics, masks, teachers or returns.
"""
from collections import Counter
import math
import time

import numpy as np

from experiments.e1.contracts import STEP, utc_time
from experiments.e1.environment import EnvironmentalSample


SCOPE = 'offline_known_queue_counterfactual_calibration_not_realized_global_policy_carbon_effect'


def _errors(values):
    values = list(values)
    return dict(count=len(values), mae=math.fsum(abs(v) for v in values)/len(values) if values else None,
                rmse=math.sqrt(math.fsum(v*v for v in values)/len(values)) if values else None)


def forecast_errors(records):
    """Score h0..15 only where both origin and label were physically observed."""
    observed = {utc_time(row['interval_start']): row for row in records}
    errors = [[] for _ in range(16)]
    valid_errors = [[] for _ in range(16)]
    for origin, row in observed.items():
        predictions = row.get('ci_forecast_mean', ())
        for horizon, prediction in enumerate(predictions[:16]):
            actual = observed.get(origin + horizon*STEP)
            if actual is None:
                continue
            error = float(prediction)-float(actual['ci_g_per_kwh'])
            if math.isfinite(error):
                errors[horizon].append(error)
                if row.get('ci_forecast_valid', False):
                    valid_errors[horizon].append(error)
    report = []
    for horizon, (all_values, valid_values) in enumerate(zip(errors, valid_errors)):
        all_stats, valid_stats = _errors(all_values), _errors(valid_values)
        report.append(dict(horizon_slots=horizon, count=all_stats['count'],
            mae_g_per_kwh=all_stats['mae'], rmse_g_per_kwh=all_stats['rmse'],
            valid_forecast_count=valid_stats['count'], valid_forecast_mae_g_per_kwh=valid_stats['mae'],
            valid_forecast_rmse_g_per_kwh=valid_stats['rmse']))
    future = _errors(v for row in errors[1:] for v in row)
    return dict(scope='observed_physical_prefix_only; h0_is_current_observation_not_future_prediction',
        units='gCO2e_per_kWh', by_horizon=report,
        overall_future=dict(count=future['count'], mae_g_per_kwh=future['mae'], rmse_g_per_kwh=future['rmse']))


def audit_probes(env):
    """Score complete paired hypotheses after done, without reading future data.

    A probe is unscored when any part of its sixteen-slot CI horizon or full
    known-work trajectories lies outside the actual physical prefix. Neither
    analytic idle report tails nor the environment's unvisited samples supply
    labels. The same one-degree cooling continuation applies to both paths.
    """
    if not env.done:
        raise RuntimeError('Mechanism labels are available only after a completed physical episode')
    if getattr(env, '_mechanism_audit', None) is not None:
        return env._mechanism_audit
    started = time.perf_counter()
    observations = {utc_time(row['interval_start']): row for row in env.records}
    unscored, results = Counter(), []
    probe = None
    powers = {}

    def power(loads, capacity, row, stamp, setpoint):
        nonlocal probe
        utilization = tuple(float(v)/float(c) for v, c in zip(loads, capacity))
        key = (*utilization, float(row['temperature_c']), float(row['wet_bulb_c']), float(setpoint))
        if key not in powers:
            if probe is None:
                probe = env.power_factory()
            sample = EnvironmentalSample(stamp, float(row['ci_g_per_kwh']),
                                         float(row['temperature_c']), float(row['wet_bulb_c']))
            probe.set_setpoint(setpoint)
            reading = probe.measure(*utilization, sample)
            if reading.accounting_boundary != 'joint_full_hvac_v1':
                raise ValueError('Mechanism calibration requires the unchanged physical accounting boundary')
            powers[key] = float(reading.total_kw)
        return powers[key]

    try:
        for index, item in enumerate(getattr(env, 'mechanism_probes', ())):
            if not item['start_settled'] or not item['wait_settled']:
                unscored['unsettled_known_work'] += 1
                continue
            if item['known_task_ids_start'] != item['known_task_ids_wait']:
                unscored['different_known_work_sets'] += 1
                continue
            start = np.asarray(item['start_loads'], dtype=float).reshape((-1, 3))
            wait = np.asarray(item['wait_loads'], dtype=float).reshape((-1, 3))
            capacity = np.asarray(item['capacity'], dtype=float)
            if (capacity.shape != (3,) or not np.isfinite(capacity).all() or (capacity <= 0).any()
                    or not np.isfinite(start).all() or not np.isfinite(wait).all()
                    or (start < 0).any() or (wait < 0).any()
                    or (start > capacity*(1+1e-10)).any() or (wait > capacity*(1+1e-10)).any()):
                unscored['invalid_physical_loads'] += 1
                continue
            if not np.allclose(start.sum(axis=0), wait.sum(axis=0), rtol=1e-10, atol=1e-9):
                unscored['unequal_known_resource_work'] += 1
                continue
            length = max(int(item['forecast_horizon']), len(start), len(wait))
            origin = utc_time(item['initial_timestamp'])
            times = [origin + h*STEP for h in range(length)]
            if any(stamp not in observations for stamp in times):
                unscored['unobserved_physical_prefix'] += 1
                continue
            delta_carbon = []
            for horizon, stamp in enumerate(times):
                row = observations[stamp]
                z = max(18., float(item['initial_setpoint_c'])-horizon-1.)
                first_load = start[horizon] if horizon < len(start) else np.zeros(3)
                second_load = wait[horizon] if horizon < len(wait) else np.zeros(3)
                difference = power(first_load, capacity, row, stamp, z)-power(second_load, capacity, row, stamp, z)
                delta_carbon.append(difference*.25*float(row['ci_g_per_kwh'])/1000.)
            gain = math.fsum(delta_carbon)
            prediction = float(item['mean_carbon_gain_kg'])
            robust = float(item['robust_carbon_gain_kg'])
            results.append(dict(probe_index=index, initial_timestamp=item['initial_timestamp'],
                task_id=item['task_id'], evaluated_slots=length,
                predicted_mean_gain_kg=prediction, predicted_std_gain_kg=float(item['std_carbon_gain_kg']),
                predicted_robust_gain_kg=robust, observed_signal_gain_kg=gain,
                mean_gain_error_kg=prediction-gain, robust_gain_error_kg=robust-gain,
                sign_correct=int(np.sign(prediction)) == int(np.sign(gain)),
                robust_sign_correct=int(np.sign(robust)) == int(np.sign(gain))))
    finally:
        if probe is not None:
            probe.close()
    mean_errors = _errors(row['mean_gain_error_kg'] for row in results)
    robust_errors = _errors(row['robust_gain_error_kg'] for row in results)
    positives = sum(row['observed_signal_gain_kg'] > 0 for row in results)
    negatives = sum(row['observed_signal_gain_kg'] < 0 for row in results)
    report = dict(scope=SCOPE, signal_scope='physically_observed_CI_and_weather_only',
        continuation='same_lower_1c_per_slot_toward_18c_for_both_complete_known_work_paths',
        probes=len(getattr(env, 'mechanism_probes', ())), scored=len(results), unscored=sum(unscored.values()),
        unscored_reasons=dict(unscored), records=results,
        mean_gain_mae_kg=mean_errors['mae'], mean_gain_rmse_kg=mean_errors['rmse'],
        robust_gain_mae_kg=robust_errors['mae'], robust_gain_rmse_kg=robust_errors['rmse'],
        sign_accuracy=sum(row['sign_correct'] for row in results)/len(results) if results else None,
        robust_sign_accuracy=sum(row['robust_sign_correct'] for row in results)/len(results) if results else None,
        observed_positive_count=positives, observed_negative_count=negatives,
        observed_zero_count=len(results)-positives-negatives,
        predicted_positive_count=sum(row['predicted_mean_gain_kg'] > 0 for row in results),
        predicted_negative_count=sum(row['predicted_mean_gain_kg'] < 0 for row in results),
        observed_positive_sign_accuracy=(sum(row['sign_correct'] for row in results if row['observed_signal_gain_kg'] > 0)/positives
                                         if positives else None),
        observed_negative_sign_accuracy=(sum(row['sign_correct'] for row in results if row['observed_signal_gain_kg'] < 0)/negatives
                                         if negatives else None),
        independent_probe_power_calls=len(powers), runtime_ms=(time.perf_counter()-started)*1000.)
    env._mechanism_audit = report
    return report
