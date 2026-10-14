"""Offline opportunity diagnostics: accounting and current-event eligibility."""
import copy
import importlib

import pandas as pd
import pytest

class _OpportunityModule:
    def __getattr__(self, name):
        return getattr(importlib.import_module('experiments.e1.carbon_opportunity'), name)


opportunity = _OpportunityModule()


T0 = pd.Timestamp('2023-05-01 05:00:00', tz='UTC')


def interval(k, ci=200.0, it=100.0, hvac=20.0, used=(0.0, 0.0, 0.0), running=0):
    return dict(interval_start=(T0 + pd.Timedelta(minutes=15*k)).isoformat(),
                interval_end=(T0 + pd.Timedelta(minutes=15*(k+1))).isoformat(),
                interval_hours=.25, ci_g_per_kwh=ci, it_power_kw=it,
                hvac_power_kw=hvac, total_power_kw=it+hvac,
                it_energy_kwh=it*.25, hvac_energy_kwh=hvac*.25,
                total_energy_kwh=(it+hvac)*.25, carbon_kg=(it+hvac)*.25*ci/1000,
                used_cores=used[0], used_gpus=used[1], used_mem=used[2],
                running=running, transmission_energy_kwh=999999.)


def samples(cis):
    return [dict(time=(T0 + pd.Timedelta(minutes=15*k)).isoformat(),
                 ci_g_per_kwh=value) for k, value in enumerate(cis)]


def task(task_id='a', duration=16, arrival=T0, multiplier=3):
    return dict(task_id=task_id, arrival_time=arrival.isoformat(), duration=duration,
                sla_multiplier=multiplier, cores_req=2., gpu_req=1., mem_req=3.)


def candidate(k=0, task_id='a', can_fit=True, event='started'):
    return dict(time=(T0 + pd.Timedelta(minutes=15*k)).isoformat(),
                task_id=task_id, can_fit_at_decision=can_fit, event=event)


COEFFICIENTS = dict(intercept_kw=100., cpu_kw_per_core=2.,
                    gpu_kw_per_gpu=3., mem_kw_per_gb=4.)


def test_energy_it_hvac_units_activity_ci_mean_and_transfer_exclusion():
    rows = [interval(0, ci=100, it=100, hvac=20),
            interval(1, ci=300, it=200, hvac=40, used=(1, 0, 0)),
            interval(2, ci=161.075, it=100, hvac=20, running=1)]
    original = copy.deepcopy(rows)
    value = opportunity.energy_decomposition(rows)
    assert value['totals']['it_energy_kwh'] == 100
    assert value['totals']['hvac_energy_kwh'] == 20
    assert value['totals']['total_energy_kwh'] == 120
    assert value['totals']['it_carbon_kg'] == pytest.approx(21.526875)
    assert value['totals']['hvac_carbon_kg'] == pytest.approx(4.305375)
    assert value['totals']['carbon_kg'] == pytest.approx(25.83225)
    assert value['activity_groups']['zero_load_idle']['interval_count'] == 1
    assert value['activity_groups']['active']['interval_count'] == 2
    assert value['ci_means']['arithmetic_g_per_kwh'] == pytest.approx(561.075/3)
    assert value['ci_means']['energy_weighted_g_per_kwh'] == pytest.approx(25.83225*1000/120)
    assert value['ci_bins'][0]['interval_count'] == 2
    assert value['ci_bins'][2]['interval_count'] == 1
    assert rows == original


@pytest.mark.parametrize('field', ['total_energy_kwh', 'total_power_kw', 'carbon_kg', 'interval_hours'])
def test_energy_rejects_inconsistent_physical_accounting(field):
    row = interval(0); row[field] += 1
    with pytest.raises(ValueError):
        opportunity.energy_decomposition([row])


def test_energy_empty_and_zero_energy_have_explicit_denominators():
    empty = opportunity.energy_decomposition([])
    assert empty['totals']['total_energy_kwh'] == 0
    assert empty['ci_means']['energy_weighted_g_per_kwh'] is None
    zero = opportunity.energy_decomposition([interval(0, it=0, hvac=0)])
    assert zero['ci_means']['arithmetic_g_per_kwh'] == 200
    assert zero['ci_means']['energy_weighted_g_per_kwh'] is None


def test_energy_bins_threshold_boundary_and_all_energy_conserved():
    value = opportunity.energy_decomposition([interval(k, ci=ci) for k,ci in
        enumerate([0,161.075,161.075001,200,200.001,300,300.001])])
    assert [row['interval_count'] for row in value['ci_bins']] == [2,2,2,1]
    assert sum(row['total_energy_kwh'] for row in value['ci_bins']) == value['totals']['total_energy_kwh']


@pytest.mark.parametrize('case', ['duplicate', 'gap', 'naive', 'negative_ci'])
def test_environment_rejects_invalid_temporal_sources(case):
    rows = samples([100, 200])
    if case == 'duplicate': rows[1]['time'] = rows[0]['time']
    elif case == 'gap': rows[1]['time'] = (T0 + pd.Timedelta(minutes=30)).isoformat()
    elif case == 'naive': rows[0]['time'] = '2023-05-01 05:00:00'
    else: rows[0]['ci_g_per_kwh'] = -1
    with pytest.raises(ValueError):
        opportunity.ci_temporal_statistics(rows, T0, T0+pd.Timedelta(minutes=30), 161.075)


def test_ci_statistics_exclusive_arrival_end_and_hourly_repetition():
    value = opportunity.ci_temporal_statistics(samples([200]*4+[100]*4+[999]),
        T0,T0+pd.Timedelta(hours=2),161.075)
    assert value['sample_count'] == 8
    assert value['high_ci_fraction'] == .5
    assert value['complete_hours'] == 2
    assert value['constant_complete_hours'] == 2
    assert value['same_adjacent_fraction'] == pytest.approx(6/7)
    assert value['high_run_lengths_minutes'] == [60]
    assert value['low_run_lengths_minutes'] == [60]
    assert value['lag_one_step_mean_abs_delta_g_per_kwh'] == pytest.approx(100/7)
    assert value['lag_four_steps_mean_abs_delta_g_per_kwh'] == 100


def test_ci_statistics_does_not_call_variable_hour_constant():
    value = opportunity.ci_temporal_statistics(samples([200,200,100,100]),
        T0,T0+pd.Timedelta(hours=1),161.075)
    assert value['constant_complete_hours'] == 0
    assert value['hour_rows'][0]['distinct_ci_count'] == 2


def test_ols_train_only_correct_units_and_prediction_error():
    rows = [interval(k, it=100+2*c+3*g+4*m, used=(c,g,m))
            for k,(c,g,m) in enumerate([(0,0,0),(1,0,0),(0,1,0),(0,0,1),(2,3,4)])]
    original = copy.deepcopy(rows)
    value = opportunity.calibrate_it_proxy(rows)
    assert value['coefficients'] == pytest.approx(COEFFICIENTS)
    assert value['fit']['rank'] == 4
    assert value['fit']['rmse_kw'] < 1e-10
    assert value['fit']['r_squared'] == pytest.approx(1)
    assert value['fit_role'] == 'train'
    assert value['models_hvac'] is False
    assert rows == original


@pytest.mark.parametrize('role', ['validation','test','dev_validation'])
def test_ols_refuses_non_train_fit(role):
    with pytest.raises(ValueError):
        opportunity.calibrate_it_proxy([interval(0)], role=role)


def test_ols_refuses_non_train_record_even_if_default_role_train():
    row=interval(0);row['role']='test'
    with pytest.raises(ValueError):
        opportunity.calibrate_it_proxy([row])


def test_rank_deficiency_is_explicit_not_fabricated_identifiability():
    value = opportunity.calibrate_it_proxy([interval(k, it=100+2*k, used=(k,0,0)) for k in range(4)])
    assert value['fit']['rank'] == 2
    assert value['fit']['coefficients_identifiable'] is False


def test_opportunity_full_grid_duration_and_local_proxy_excludes_intercept():
    value = opportunity.execution_window_opportunities([task()], [candidate()],
        samples([300,200,100,100]), COEFFICIENTS)
    row = value['rows'][0]
    assert row['occupied_steps'] == 2  # 16min occupies two grid intervals
    assert row['actual_duration_minutes'] == 16
    assert row['ci_integral_now_g_per_kwh_hours'] == 125
    assert row['ci_integral_defer_g_per_kwh_hours'] == 75
    assert row['task_dynamic_it_proxy_kw'] == 19
    assert row['local_it_proxy_gain_kg'] == pytest.approx(.95)
    assert row['delta_proxy_carbon_kg'] == pytest.approx(-.95)
    assert row['excluded_idle_intercept_kw'] == 100
    assert value['offline_clairvoyant_diagnostic'] is True
    assert value['is_cluster_carbon_result'] is False


def test_same_hour_start_does_not_imply_whole_execution_ci_unchanged():
    # Start 05:15 vs05:30 both CI300; 45min interval moves last slot into06:00 CI100.
    value = opportunity.execution_window_opportunities([task(duration=45)], [candidate(k=1)],
        samples([300,300,300,300,100,100]), COEFFICIENTS)
    row = value['rows'][0]
    assert row['start_ci_now_g_per_kwh'] == row['start_ci_defer_g_per_kwh'] == 300
    assert row['same_source_hour'] is True
    assert row['local_it_proxy_gain_kg'] == pytest.approx(.95)


@pytest.mark.parametrize('cis,expected', [([300,300,300],0),([200,250,300],-.2375)])
def test_high_current_ci_can_have_no_gain_or_worse_gain(cis, expected):
    value = opportunity.execution_window_opportunities([task(duration=15)], [candidate()],
        samples(cis), COEFFICIENTS)
    row=value['rows'][0]
    assert row['high_ci_at_candidate'] is True
    assert row['local_it_proxy_gain_kg'] == pytest.approx(expected)


def test_candidates_require_arrived_unstarted_capacity_and_exact_slack_boundary():
    ts=[task('eligible',duration=30,multiplier=1.5),task('tight',duration=29,multiplier=1.5),
        task('future',arrival=T0+pd.Timedelta(minutes=15)),task('no_fit'),task('already')]
    events=[candidate(task_id='already'),candidate(task_id='already',k=1),
            candidate(task_id='eligible'),candidate(task_id='tight'),candidate(task_id='future'),
            candidate(task_id='no_fit',can_fit=False,event='capacity_waited')]
    value=opportunity.execution_window_opportunities(ts,events,samples([200]*10),COEFFICIENTS)
    assert [row['task_id'] for row in value['rows']] == ['already','eligible']
    assert value['eligibility_rejections']['insufficient_solo_slack'] == 1
    assert value['eligibility_rejections']['not_arrived'] == 1
    assert value['eligibility_rejections']['capacity_unavailable'] == 1
    assert value['eligibility_rejections']['already_started'] == 1


def test_missing_drain_ci_rejected_without_wrap_or_hold_last():
    value=opportunity.execution_window_opportunities([task()], [candidate()],
        samples([200,200]),COEFFICIENTS)
    assert value['rows'] == []
    assert value['coverage_rejections'][0]['task_id'] == 'a'
    assert value['coverage_rejections'][0]['reason'] == 'missing_real_ci_for_execution_window'


def test_ci_opportunity_uses_timestamp_nanoseconds_not_seconds_index():
    value=opportunity.execution_window_opportunities([task(duration=15)], [candidate()],
        samples([200,100,999]),COEFFICIENTS)
    assert value['rows'][0]['local_it_proxy_gain_kg'] == pytest.approx(.475)


def test_opportunity_dataframe_and_no_input_mutation():
    ts=[task()];ev=[candidate()];ss=samples([200,100,100]);original=copy.deepcopy((ts,ev,ss))
    value=opportunity.execution_window_opportunities(pd.DataFrame(ts),pd.DataFrame(ev),
        pd.DataFrame(ss),dict(COEFFICIENTS))
    assert len(value['rows']) == 1
    assert (ts,ev,ss) == original


def test_duplicate_task_ids_and_off_grid_candidates_rejected():
    with pytest.raises(ValueError):
        opportunity.execution_window_opportunities([task(),task()], [candidate()],samples([200]*5),COEFFICIENTS)
    ev=candidate();ev['time']=(T0+pd.Timedelta(minutes=1)).isoformat()
    with pytest.raises(ValueError):
        opportunity.execution_window_opportunities([task()], [ev],samples([200]*5),COEFFICIENTS)
