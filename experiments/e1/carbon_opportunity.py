"""Offline E1-7A carbon diagnostics, with no scheduler or physics replacement.

The IT proxy and execution-window CI below are deliberately clairvoyant *analysis*
tools. They may read later real CI to diagnose opportunities, and must never be
passed to an online policy or described as full-cluster emissions. Actual carbon
comparisons require the unchanged LocalExecutor and persistent IT/HVAC model.
"""
import math
from collections import Counter

import numpy as np
import pandas as pd

from .contracts import STEP


VERSION = 'e1-7a-carbon-opportunity-v1'
THRESHOLD = 161.075
RESOURCE_FIELDS = ('used_cores', 'used_gpus', 'used_mem')
COEFFICIENT_FIELDS = ('cpu_kw_per_core', 'gpu_kw_per_gpu', 'mem_kw_per_gb')
ENERGY_FIELDS = ('it_energy_kwh', 'hvac_energy_kwh', 'total_energy_kwh')
CARBON_FIELDS = ('it_carbon_kg', 'hvac_carbon_kg', 'carbon_kg')


def _records(values):
    if isinstance(values, pd.DataFrame):
        return values.to_dict('records')
    return list(values)


def _get(value, key, default=None):
    return value.get(key, default) if isinstance(value, dict) else getattr(value, key, default)


def _number(value, name, nonnegative=True):
    if isinstance(value, bool):
        raise ValueError(f'{name} must be numeric, not Boolean')
    value = float(value)
    if not math.isfinite(value) or (nonnegative and value < 0):
        raise ValueError(f'{name} must be finite' + (' and nonnegative' if nonnegative else ''))
    return value


def _time(value):
    time = pd.Timestamp(value)
    if time.tzinfo is None or pd.isna(time):
        raise ValueError('Diagnostic times must have explicit UTC timezone')
    time = time.tz_convert('UTC')
    if time.floor('15min') != time:
        raise ValueError('Diagnostic times must be on the original 15-minute grid')
    return time


def _threshold(value):
    value = _number(value, 'CI threshold')
    if not 0 < value < 200:
        raise ValueError('Fixed diagnostic CI bins require 0 < threshold < 200')
    return value


def _sample_rows(samples):
    rows = []
    for sample in _records(getattr(samples, 'samples', samples)):
        time = _time(_get(sample, 'time', _get(sample, 'interval_start')))
        ci = _number(_get(sample, 'ci_g_per_kwh'), 'CI')
        rows.append(dict(time=time, ci_g_per_kwh=ci))
    rows.sort(key=lambda row: row['time'])
    if any(b['time'] - a['time'] != STEP for a, b in zip(rows, rows[1:])):
        raise ValueError('Real CI samples must be unique and contiguous at 15 minutes')
    return rows


def _totals(rows):
    return dict(interval_count=len(rows), **{
        key: math.fsum(row[key] for row in rows) for key in ENERGY_FIELDS + CARBON_FIELDS})


def energy_decomposition(intervals, threshold=THRESHOLD):
    """Decompose genuine interval records, kWh and kgCO2e; ignore transfer energy.

    Idle means zero CPU/GPU/MEM *and* no running task, not zero IT power. Every
    supplied accounting field is checked against kW times the real 15-minute
    interval and IT+HVAC times CI. No model is instantiated or re-estimated.
    """
    threshold = _threshold(threshold)
    rows = []
    for record in _records(intervals):
        start, end = _time(_get(record, 'interval_start')), _time(_get(record, 'interval_end'))
        if end - start != STEP:
            raise ValueError('Energy intervals must retain the original 15-minute length')
        hours = STEP.total_seconds()/3600
        ci = _number(_get(record, 'ci_g_per_kwh'), 'CI')
        it = _number(_get(record, 'it_power_kw'), 'IT kW')
        hvac = _number(_get(record, 'hvac_power_kw'), 'HVAC kW')
        row = dict(interval_start=start.isoformat(), interval_end=end.isoformat(),
                   interval_hours=hours, ci_g_per_kwh=ci, it_power_kw=it,
                   hvac_power_kw=hvac, total_power_kw=it+hvac,
                   it_energy_kwh=it*hours, hvac_energy_kwh=hvac*hours,
                   total_energy_kwh=(it+hvac)*hours,
                   it_carbon_kg=it*hours*ci/1000, hvac_carbon_kg=hvac*hours*ci/1000,
                   carbon_kg=(it+hvac)*hours*ci/1000)
        for key in ('interval_hours', 'total_power_kw') + ENERGY_FIELDS + ('carbon_kg',):
            actual = _get(record, key)
            if actual is not None and not math.isclose(_number(actual, key), row[key],
                                                       rel_tol=1e-10, abs_tol=1e-9):
                raise ValueError(f'Interval {key} does not match original IT/HVAC integration')
        demands = tuple(_number(_get(record, key, 0), key) for key in RESOURCE_FIELDS)
        running = _number(_get(record, 'running', 0), 'running count')
        row.update(dict(zip(RESOURCE_FIELDS, demands)))
        row['running'] = running
        row['activity'] = 'active' if running > 0 or any(demands) else 'zero_load_idle'
        rows.append(row)
    ordered = sorted(rows, key=lambda row: row['interval_start'])
    if any(pd.Timestamp(b['interval_start']) != pd.Timestamp(a['interval_end'])
           for a, b in zip(ordered, ordered[1:])):
        raise ValueError('Interval accounting must be unique and contiguous')
    totals = _totals(rows)
    groups = {name: _totals([row for row in rows if row['activity'] == name])
              for name in ('active', 'zero_load_idle')}
    bins = []
    for i, (lower, upper) in enumerate(zip((0, threshold, 200, 300), (threshold, 200, 300, math.inf))):
        selected = [row for row in rows if (row['ci_g_per_kwh'] >= lower if i == 0
                    else row['ci_g_per_kwh'] > lower) and row['ci_g_per_kwh'] <= upper]
        bins.append(dict(lower_g_per_kwh=lower, upper_g_per_kwh=upper if math.isfinite(upper) else None,
                         lower_inclusive=i == 0, upper_inclusive=math.isfinite(upper), **_totals(selected)))
    return dict(version=VERSION, energy_unit='kWh', carbon_unit='kgCO2e',
                threshold_g_per_kwh=threshold, totals=totals, interval_rows=rows,
                activity_groups=groups, ci_bins=bins,
                ci_means=dict(arithmetic_g_per_kwh=math.fsum(row['ci_g_per_kwh'] for row in rows)/len(rows)
                              if rows else None,
                              energy_weighted_g_per_kwh=totals['carbon_kg']*1000/totals['total_energy_kwh']
                              if totals['total_energy_kwh'] else None),
                transfer_energy_used=False, idle_is_zero_load_not_zero_power=True)


def ci_temporal_statistics(samples, arrival_start, arrival_end, threshold=THRESHOLD):
    """Describe actual arrival-window CI, end exclusive; no cycles or last hold."""
    start, end = _time(arrival_start), _time(arrival_end)
    threshold = _threshold(threshold)
    if end <= start:
        raise ValueError('Arrival window must be positive')
    rows = [row for row in _sample_rows(samples) if start <= row['time'] < end]
    if len(rows) != (end-start)//STEP or not rows or rows[0]['time'] != start or rows[-1]['time']+STEP != end:
        raise ValueError('Actual CI does not cover every arrival-window interval')
    values = [row['ci_g_per_kwh'] for row in rows]
    high = [value > threshold for value in values]
    runs = {'high': [], 'low': []}
    last, count = high[0], 0
    for flag in high:
        if flag != last:
            runs['high' if last else 'low'].append(count*15)
            count = 0
        last, count = flag, count+1
    runs['high' if last else 'low'].append(count*15)
    hour_groups = {}
    for row in rows:
        hour_groups.setdefault(row['time'].floor('h'), []).append(row)
    hour_rows = [dict(hour_start=hour.isoformat(), sample_count=len(items),
                      complete_hour=len(items) == 4, distinct_ci_count=len({item['ci_g_per_kwh'] for item in items}),
                      mean_ci_g_per_kwh=math.fsum(item['ci_g_per_kwh'] for item in items)/len(items))
                 for hour, items in hour_groups.items()]
    complete = [row for row in hour_rows if row['complete_hour']]
    lag = lambda k: [abs(b-a) for a,b in zip(values, values[k:])]
    lag_one, lag_four = lag(1), lag(4)
    return dict(version=VERSION, arrival_start=start.isoformat(), arrival_end=end.isoformat(),
                end_exclusive=True, native_step_minutes=15, sample_count=len(rows),
                threshold_g_per_kwh=threshold, arithmetic_mean_g_per_kwh=math.fsum(values)/len(values),
                minimum_g_per_kwh=min(values), maximum_g_per_kwh=max(values),
                high_ci_fraction=sum(high)/len(high),
                same_adjacent_fraction=sum(a == b for a,b in zip(values,values[1:]))/(len(values)-1)
                if len(values) > 1 else None,
                lag_one_step_mean_abs_delta_g_per_kwh=math.fsum(lag_one)/len(lag_one) if lag_one else None,
                lag_four_steps_mean_abs_delta_g_per_kwh=math.fsum(lag_four)/len(lag_four) if lag_four else None,
                complete_hours=len(complete), constant_complete_hours=sum(row['distinct_ci_count'] == 1 for row in complete),
                high_run_lengths_minutes=runs['high'], low_run_lengths_minutes=runs['low'], hour_rows=hour_rows,
                sample_rows=[dict(time=row['time'].isoformat(),ci_g_per_kwh=row['ci_g_per_kwh']) for row in rows])


def calibrate_it_proxy(intervals, *, role='train'):
    """Fit a descriptive Train-only OLS proxy in physical resource units.

    ``role='train'`` is an explicit caller assertion when records have no role
    metadata. Passing another role or any non-Train labelled record fails. This
    cannot infer a file's data role and does not authorize reading other windows.
    Rank-deficient coefficients are flagged, never presented as causal effects.
    """
    if str(role).lower() != 'train':
        raise ValueError('IT proxy fitting is Train-only')
    rows = _records(intervals)
    if not rows:
        raise ValueError('IT proxy requires observed Train intervals')
    for row in rows:
        label = _get(row, 'role', _get(row, 'data_role', 'train'))
        if str(label).lower() != 'train':
            raise ValueError('IT proxy cannot fit Validation or Test-labelled records')
    demand = np.asarray([[_number(_get(row,key),key) for key in RESOURCE_FIELDS] for row in rows], dtype=float)
    target = np.asarray([_number(_get(row,'it_power_kw'),'IT kW') for row in rows], dtype=float)
    scale = np.maximum(np.max(np.abs(demand),axis=0),1.)
    design = np.column_stack((np.ones(len(rows)),demand/scale))
    scaled, _, rank, singular = np.linalg.lstsq(design,target,rcond=None)
    coefficients = np.concatenate((scaled[:1],scaled[1:]/scale))
    residual = target-design@scaled
    sse, variance = float(residual@residual), float(np.sum((target-np.mean(target))**2))
    return dict(version=VERSION, fit_role='train', coefficients=dict(zip(('intercept_kw',)+COEFFICIENT_FIELDS,
                map(float,coefficients))), resource_units=['core','GPU','GB'], power_unit='kW',
                fit=dict(sample_count=len(rows),rank=int(rank), coefficients_identifiable=int(rank)==4,
                         rmse_kw=float(math.sqrt(sse/len(rows))), mean_absolute_error_kw=float(np.mean(np.abs(residual))),
                         maximum_absolute_error_kw=float(np.max(np.abs(residual))),
                         r_squared=1-sse/variance if variance > 0 else None,
                         singular_values=list(map(float,singular)), design_scale=list(map(float,scale))),
                models_hvac=False, is_cluster_carbon_result=False,
                interpretation='descriptive Train OLS; resource coefficients are not causal task power attribution')


def _can_fit(event):
    value = _get(event,'can_fit_at_decision',False)
    if value in (True,'True','true',1,'1'):
        return True
    if value in (False,'False','false',0,'0'):
        return False
    raise ValueError('Candidate capacity flag must be explicit Boolean')


def execution_window_opportunities(tasks, events, samples, coefficients, delta_minutes=15):
    """Compare one-step shift over the *whole grid occupancy* using future CI.

    Candidates are genuine can-fit, arrived and not-yet-started events with enough
    solo slack. The local IT proxy excludes the idle intercept. It does not include
    HVAC, background load, endogenous later starts, or reservation feasibility.
    Missing real CI yields an explicit coverage rejection, never extrapolation.
    """
    if isinstance(delta_minutes,bool) or int(delta_minutes) != delta_minutes or delta_minutes <= 0 or delta_minutes % 15:
        raise ValueError('Shift must be a positive multiple of the original 15-minute grid')
    delta = pd.Timedelta(minutes=int(delta_minutes))
    specs = _records(getattr(tasks,'tasks',tasks))
    by_id = {_get(spec,'task_id'):spec for spec in specs}
    if len(by_id) != len(specs) or any(not task_id for task_id in by_id):
        raise ValueError('Diagnostic task IDs must be nonempty and unique')
    samples = _sample_rows(samples)
    # Timestamp.value and Timedelta.value are both nanoseconds; do not divide one
    # by seconds or treat a floating duration as a timestamp-array offset.
    ci = {row['time'].value:row['ci_g_per_kwh'] for row in samples}
    coefficients = dict(coefficients.get('coefficients',coefficients))
    intercept = _number(coefficients.get('intercept_kw',0),'IT intercept',False)
    slopes = tuple(_number(coefficients[key],key,False) for key in COEFFICIENT_FIELDS)
    rows, rejected, counts, started = [], [], Counter(), set()
    ordered = [(_time(_get(event,'time')),i,event) for i,event in enumerate(_records(events))]
    ordered.sort(key=lambda value:(value[0],value[1]))
    for now, index, event in ordered:
        kind, task_id = _get(event,'event'),_get(event,'task_id')
        if kind not in ('started','policy_deferred','capacity_waited'):
            continue
        if task_id not in by_id:
            raise ValueError('Candidate event references a task outside the frozen cohort')
        spec = by_id[task_id]
        arrival = _time(_get(spec,'arrival_time'))
        duration = _number(_get(spec,'duration'),'task duration')
        if duration <= 0:
            raise ValueError('Task duration must be positive')
        deadline = _get(spec,'sla_deadline')
        if deadline is None:
            deadline = arrival+pd.Timedelta(minutes=_number(_get(spec,'sla_multiplier'),'SLA multiplier')*duration)
        else:
            deadline = pd.Timestamp(deadline)
            if deadline.tzinfo is None:
                raise ValueError('Deadline must have explicit timezone')
            deadline = deadline.tz_convert('UTC')
        solo = (deadline-now-pd.Timedelta(minutes=duration)).total_seconds()/60
        reason = ('already_started' if task_id in started else 'not_arrived' if arrival > now
                  else 'capacity_unavailable' if not _can_fit(event)
                  else 'insufficient_solo_slack' if solo < delta_minutes else None)
        if reason:
            counts[reason] += 1
        else:
            occupied = int(math.ceil(pd.Timedelta(minutes=duration).value/STEP.value))
            now_slots = [now.value+k*STEP.value for k in range(occupied)]
            defer_slots = [now.value+delta.value+k*STEP.value for k in range(occupied)]
            missing = sorted(set(now_slots+defer_slots)-ci.keys())
            if missing:
                rejected.append(dict(task_id=task_id,time=now.isoformat(),event_index=index,
                                     reason='missing_real_ci_for_execution_window',
                                     missing_times=[pd.Timestamp(t,tz='UTC').isoformat() for t in missing]))
            else:
                integral_now = math.fsum(ci[t]*.25 for t in now_slots)
                integral_defer = math.fsum(ci[t]*.25 for t in defer_slots)
                demands = tuple(_number(_get(spec,key),key) for key in ('cores_req','gpu_req','mem_req'))
                dynamic = math.fsum(c*r for c,r in zip(slopes,demands))
                gain = dynamic*(integral_now-integral_defer)/1000
                rows.append(dict(task_id=task_id,time=now.isoformat(),event_index=index,candidate_event=kind,
                    arrival=arrival.isoformat(),deadline=deadline.isoformat(),solo_slack_minutes=solo,
                    grid_solo_slack_minutes=math.floor(solo/15)*15,delta_minutes=int(delta_minutes),
                    actual_duration_minutes=duration,occupied_steps=occupied,occupied_minutes=15*occupied,
                    hypothetical_start_now=now.isoformat(),hypothetical_start_defer=(now+delta).isoformat(),
                    release_now=(now+occupied*STEP).isoformat(),release_defer=(now+delta+occupied*STEP).isoformat(),
                    start_ci_now_g_per_kwh=ci[now.value],start_ci_defer_g_per_kwh=ci[now.value+delta.value],
                    high_ci_at_candidate=ci[now.value] > THRESHOLD,same_source_hour=now.floor('h')==(now+delta).floor('h'),
                    ci_integral_now_g_per_kwh_hours=integral_now,ci_integral_defer_g_per_kwh_hours=integral_defer,
                    ci_mean_now_g_per_kwh=integral_now/(occupied*.25),ci_mean_defer_g_per_kwh=integral_defer/(occupied*.25),
                    cores_req=demands[0],gpu_req=demands[1],mem_req=demands[2],task_dynamic_it_proxy_kw=dynamic,
                    excluded_idle_intercept_kw=intercept,it_proxy_nonnegative=dynamic >= 0,
                    local_it_proxy_gain_kg=gain,delta_proxy_carbon_kg=-gain,
                    offline_clairvoyant_diagnostic=True,is_cluster_carbon_result=False))
        if kind == 'started':
            started.add(task_id)
    return dict(version=VERSION,offline_clairvoyant_diagnostic=True,is_cluster_carbon_result=False,
                threshold_g_per_kwh=THRESHOLD,delta_minutes=int(delta_minutes),rows=rows,
                coverage_rejections=rejected,eligibility_rejections=dict(counts),
                proxy_excludes=['idle_intercept','HVAC','background_load','subsequent_task_interference'],
                selection='genuine candidate; arrived; not started; can-fit; solo slack permits one requested grid shift')
