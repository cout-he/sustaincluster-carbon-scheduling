"""Causal forecast and complete-work carbon counterfactual contracts."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from experiments.e1.environment import EnvironmentalSample
from experiments.joint_v3.ci_forecaster import CIForecast, CIForecaster
from experiments.joint_v3.carbon_counterfactual import CarbonCounterfactual
from experiments.joint_v3.uncertainty import robust_gain


NOW = pd.Timestamp('2023-01-01T00:00:00Z')
CAPACITY = (50000, 1000, 80000)
SAMPLE = EnvironmentalSample(NOW, 300, 20, 18)


def train_scenario(count=500):
    times = pd.date_range(NOW, periods=count, freq='15min')
    values = 300 + 80 * np.sin(np.arange(count) * 2 * np.pi / 96)
    samples = tuple(EnvironmentalSample(t, float(v), 20, 18) for t, v in zip(times, values))
    return SimpleNamespace(role='train', environment=SimpleNamespace(samples=samples))


def schedule(loads, settled=True):
    return SimpleNamespace(loads=np.asarray(loads, dtype=float).reshape(-1, 3),
                           starts={'job': NOW}, finishes={'job': NOW + pd.Timedelta(minutes=15)},
                           valid=settled, settled=settled, reason='ok' if settled else 'planning_horizon_exceeded',
                           capacity=CAPACITY, unresolved_task_ids=() if settled else ('tail',))


def forecast(values=None, residuals=None):
    return CIForecast(np.full(16, 300.) if values is None else np.asarray(values, dtype=float),
                      np.zeros((8, 16)) if residuals is None else np.asarray(residuals, dtype=float),
                      0.9, True, 'ok')


@pytest.fixture
def physical_factory():
    from experiments.e1.config import ROOT
    from experiments.joint.power import JointPowerModel
    config = dict(total_cores=50000, total_gpus=1000, total_mem=80000,
                  location='US-CAL-CISO', hvac_setpoint_c=22, seed=123)
    return lambda: JointPowerModel(config, ROOT / 'configs/dcs/dc_config.json')


def test_robust_gain_uses_sample_standard_deviation():
    mean, std, robust = robust_gain([1., 3.], kappa=1.)
    assert mean == 2.
    assert std == pytest.approx(2 ** .5)
    assert robust == pytest.approx(2 - 2 ** .5)


def test_fit_rejects_non_train_before_touching_environment():
    class HeldOut:
        role = 'validation'

        @property
        def environment(self):
            raise AssertionError('held-out payload read')

    with pytest.raises(ValueError, match='Train'):
        CIForecaster().fit([train_scenario(), HeldOut()])


def test_causal_forecast_ignores_future_and_round_trips_identity():
    scenario = train_scenario()
    model = CIForecaster().fit([scenario])
    history = scenario.environment.samples[:200]
    current = history[-1].time
    predicted = model.predict(history, current)
    with_future = model.predict(scenario.environment.samples, current)
    assert predicted.valid
    assert predicted.mean.shape == (16,)
    assert predicted.residual_scenarios.shape == (8, 16)
    assert predicted.mean[0] == history[-1].ci_g_per_kwh
    np.testing.assert_array_equal(predicted.mean, with_future.mean)
    np.testing.assert_array_equal(predicted.scenarios[:, 0], np.full(8, history[-1].ci_g_per_kwh))
    restored = CIForecaster.from_state(model.export_state())
    np.testing.assert_array_equal(restored.predict(history, current).scenarios, predicted.scenarios)
    assert restored.identity_sha256 == model.identity_sha256


def test_duplicate_train_environments_do_not_reweight_fit_or_calibration():
    scenario = train_scenario()
    single = CIForecaster().fit([scenario])
    repeated = CIForecaster().fit([scenario, scenario])
    assert single.identity_sha256 == repeated.identity_sha256
    audit = single.export_state()['audit']
    assert audit['fit_label_end'] < audit['calibration_origin_start']
    assert audit['calibration_vectors'] >= 2


def test_training_environment_drain_tail_is_excluded_before_reading_ci():
    scenario = train_scenario(500)
    cutoff = scenario.environment.samples[350].time
    scenario.workload = SimpleNamespace(end=cutoff)
    bounded = CIForecaster().fit([scenario])
    trimmed = SimpleNamespace(role='train', workload=SimpleNamespace(end=cutoff),
                              environment=SimpleNamespace(samples=scenario.environment.samples[:350]))
    clean = CIForecaster().fit([trimmed])
    assert bounded.identity_sha256 == clean.identity_sha256
    assert bounded.export_state()['audit']['unique_observations'] == 350


def test_short_history_is_causal_persistence_and_invalid_for_teacher():
    scenario = train_scenario()
    model = CIForecaster().fit([scenario])
    current = scenario.environment.samples[1]
    predicted = model.predict(scenario.environment.samples[:2], current.time)
    assert not predicted.valid
    assert predicted.confidence == 0
    np.testing.assert_array_equal(predicted.mean, np.full(16, current.ci_g_per_kwh))


def test_calibration_labels_cannot_change_fitted_mean_or_normalization():
    scenario = train_scenario()
    changed_samples = scenario.environment.samples[:400] + tuple(
        replace(s, ci_g_per_kwh=s.ci_g_per_kwh+10000) for s in scenario.environment.samples[400:])
    changed = SimpleNamespace(role='train', environment=SimpleNamespace(samples=changed_samples))
    original = CIForecaster().fit([scenario])
    shifted = CIForecaster().fit([changed])
    history = scenario.environment.samples[:200]
    np.testing.assert_array_equal(original.predict(history, history[-1].time).mean,
                                  shifted.predict(history, history[-1].time).mean)
    assert shifted.predict(history, history[-1].time).confidence < original.predict(history, history[-1].time).confidence


def test_uncertainty_samples_whole_calibration_trajectories_and_rejects_tampering():
    scenario = train_scenario()
    model = CIForecaster().fit([scenario])
    state = model.export_state()
    predicted = model.predict(scenario.environment.samples[:200], scenario.environment.samples[199].time)
    bank = {tuple(row) for row in state['residuals']}
    assert all(tuple(row) in bank for row in predicted.residual_scenarios)
    state['residuals'][0][1] += 1
    with pytest.raises(ValueError, match='identity'):
        CIForecaster.from_state(state)


def test_identical_counterfactuals_have_zero_gain_with_real_full_hvac(physical_factory):
    estimator = CarbonCounterfactual(physical_factory)
    try:
        plan = schedule([[25000, 500, 40000], [25000, 500, 40000]])
        result = estimator.estimate(plan, plan, forecast(), SAMPLE, 22, capacity=CAPACITY)
        assert result.counterfactual_valid
        assert result.mean_carbon_gain_kg == result.std_carbon_gain_kg == result.robust_carbon_gain_kg == 0
        assert result.start_settled and result.wait_settled
        assert result.accounting_boundary == 'joint_full_hvac_v1'
    finally:
        estimator.close()


def test_physical_resource_normalization_and_full_energy_gain(physical_factory):
    estimator = CarbonCounterfactual(physical_factory)
    probe = physical_factory()
    try:
        probe.set_setpoint(18)
        active = probe.measure(.5, .5, .5, SAMPLE).total_kw
        idle = probe.measure(0, 0, 0, SAMPLE).total_kw
        immediate = schedule([[25000, 500, 40000]])
        delayed = schedule([[0, 0, 0], [25000, 500, 40000]])
        result = estimator.estimate(immediate, delayed, forecast([400, 100] + [100]*14),
                                    SAMPLE, 18, capacity=CAPACITY)
        # Same two slots carry all work and idle overhead: 0.25 h * 300 g/kWh / 1000.
        assert result.mean_carbon_gain_kg == pytest.approx((active-idle) * .075)
        assert result.robust_carbon_gain_kg > 0
    finally:
        estimator.close()
        probe.close()


def test_uncertainty_penalty_is_based_on_batch_carbon_scenarios(physical_factory):
    estimator = CarbonCounterfactual(physical_factory)
    probe = physical_factory()
    try:
        probe.set_setpoint(18)
        marginal_kw = probe.measure(.5, .5, .5, SAMPLE).total_kw-probe.measure(0, 0, 0, SAMPLE).total_kw
        residuals = np.zeros((8, 16))
        residuals[:, 1] = [-70, -50, -30, -10, 10, 30, 50, 70]
        immediate = schedule([[25000, 500, 40000]])
        delayed = schedule([[0, 0, 0], [25000, 500, 40000]])
        result = estimator.estimate(immediate, delayed, forecast([400, 100]+[100]*14, residuals),
                                    SAMPLE, 18, capacity=CAPACITY)
        assert result.mean_carbon_gain_kg == pytest.approx(marginal_kw*.075)
        assert result.std_carbon_gain_kg == pytest.approx(marginal_kw*.00025*np.sqrt(2400))
        assert result.robust_carbon_gain_kg == pytest.approx(result.mean_carbon_gain_kg-result.std_carbon_gain_kg)
        misses = estimator.cache_misses
        repeated = estimator.estimate(immediate, delayed, forecast([400, 100]+[100]*14, residuals),
                                      SAMPLE, 18, capacity=CAPACITY)
        assert repeated.mean_carbon_gain_kg == result.mean_carbon_gain_kg
        assert estimator.cache_misses == misses
    finally:
        estimator.close()
        probe.close()


def test_full_work_beyond_forecast_cannot_manufacture_positive_guidance(physical_factory):
    estimator = CarbonCounterfactual(physical_factory)
    try:
        work = [25000, 500, 40000]
        immediate = schedule([work] * 16)
        delayed = schedule([[0, 0, 0]] + [work] * 16)
        result = estimator.estimate(immediate, delayed, forecast(), SAMPLE, 18, capacity=CAPACITY)
        assert result.mean_carbon_gain_kg == pytest.approx(0, abs=1e-10)
        assert not result.counterfactual_valid
        assert result.robust_carbon_gain_kg <= 0
        assert result.tail_liability_kg > 0
        assert result.evaluated_slots == 17
    finally:
        estimator.close()


def test_unsettled_schedule_never_supplies_positive_teacher_gain(physical_factory):
    estimator = CarbonCounterfactual(physical_factory)
    try:
        result = estimator.estimate(schedule([[25000, 500, 40000]]), schedule([[0, 0, 0]], False),
                                    forecast(), SAMPLE, 18, capacity=CAPACITY)
        assert not result.counterfactual_valid
        assert result.robust_carbon_gain_kg <= 0
        assert not result.wait_settled
        assert result.tail_liability_kg > 0
    finally:
        estimator.close()


def test_known_unsettled_duration_is_charged_beyond_the_planning_cap(physical_factory):
    estimator = CarbonCounterfactual(physical_factory)
    probe = physical_factory()
    try:
        probe.set_setpoint(18)
        full_kw = probe.measure(1, 1, 1, SAMPLE).total_kw
        immediate = schedule([[25000, 500, 40000]], False)
        delayed = schedule([[0, 0, 0]], False)
        immediate.unresolved_duration_minutes = 6000
        delayed.unresolved_duration_minutes = 6015
        result = estimator.estimate(immediate, delayed, forecast(), SAMPLE, 18, capacity=CAPACITY)
        assert result.tail_liability_known
        assert result.tail_liability_kg >= full_kw*300*(48+100.25)/1000
        assert result.robust_carbon_gain_kg <= 0
    finally:
        estimator.close()
        probe.close()


def test_fast_proxy_uses_real_marginal_power_and_equal_candidate_work(physical_factory):
    estimator = CarbonCounterfactual(physical_factory)
    probe = physical_factory()
    candidate = SimpleNamespace(cores_req=12500, gpu_req=250, mem_req=20000, duration=31.)
    try:
        probe.set_setpoint(18)
        base_kw = probe.measure(.5, .5, .5, SAMPLE).total_kw
        total_kw = probe.measure(.75, .75, .75, SAMPLE).total_kw
        # 31 minutes occupies three complete slots; the shift exchanges CI0/CI3.
        prediction = forecast([400., 350., 200., 100.]+[100.]*12)
        result = estimator.fast_estimate(candidate, prediction, SAMPLE, 18,
            capacity=CAPACITY, current_used=(25000, 500, 40000))
        assert result.mean_carbon_gain_kg == pytest.approx((total_kw-base_kw)*.075)
        assert result.std_carbon_gain_kg == 0
        assert not result.counterfactual_valid
        assert result.forecast_confidence == 0
        assert result.robust_carbon_gain_kg <= 0
        assert 'fast_marginal_proxy' in result.reason
    finally:
        estimator.close()
        probe.close()


def test_fast_proxy_uses_only_causal_forecast_and_marks_tail_untrusted(physical_factory):
    estimator = CarbonCounterfactual(physical_factory)
    scenario = train_scenario()
    forecaster = CIForecaster().fit([scenario])
    history = scenario.environment.samples[:200]
    sample = history[-1]
    causal = forecaster.predict(history, sample.time)
    with_future = forecaster.predict(scenario.environment.samples, sample.time)
    candidate = SimpleNamespace(cores_req=12500, gpu_req=250, mem_req=20000, duration=240.)
    try:
        a = estimator.fast_estimate(candidate, causal, sample, 18,
            capacity=CAPACITY, current_used=(25000, 500, 40000))
        b = estimator.fast_estimate(candidate, with_future, sample, 18,
            capacity=CAPACITY, current_used=(25000, 500, 40000))
        assert a.mean_carbon_gain_kg == b.mean_carbon_gain_kg
        assert a.work_outside_forecast
        assert a.tail_liability_kg > 0
        assert not a.counterfactual_valid
        assert a.robust_carbon_gain_kg <= 0
        with pytest.raises(ValueError, match='capacity'):
            estimator.fast_estimate(candidate, causal, sample, 18,
                capacity=CAPACITY, current_used=CAPACITY)
    finally:
        estimator.close()
