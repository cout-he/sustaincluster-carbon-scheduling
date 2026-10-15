"""Acceptance contracts for the real upstream joint IT/HVAC adapter."""

import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from experiments.e1.config import ROOT
from experiments.e1.environment import EnvironmentalSample
from experiments.e1.power import UpstreamPowerModel
from experiments.joint.power import ACCOUNTING_BOUNDARY, COMPONENTS, JointPowerModel


CONFIG = dict(total_cores=50000, total_gpus=1000, total_mem=80000,
              location='US-CAL-CISO', hvac_setpoint_c=22, seed=123)
MODEL = ROOT / 'configs/dcs/dc_config.json'
SAMPLE = EnvironmentalSample(pd.Timestamp('2023-01-01T05:00:00Z'), 100, 20, 18)


@pytest.fixture
def model():
    instance = JointPowerModel(CONFIG, MODEL)
    yield instance
    instance.close()


def test_full_boundary_preserves_22c_historical_equations(model):
    legacy_model = UpstreamPowerModel(CONFIG, MODEL)
    legacy = legacy_model.measure(0.5, 0.4, 0.3, SAMPLE)
    reading = model.measure(0.5, 0.4, 0.3, SAMPLE)
    assert reading.it_kw == legacy.it_kw
    assert reading.legacy_hvac_kw == legacy.hvac_kw
    assert reading.legacy_total_kw == legacy.total_kw
    info = model.dc_env.info
    assert info['dc_HVAC_total_power_kW'] == legacy.hvac_kw
    assert info['dc_total_power_kW'] == legacy.total_kw
    assert info['dc_CW_pump_power_kW'] == model.dc_env.CW_pump_load
    assert info['dc_CT_pump_power_kW'] == model.dc_env.CT_pump_load
    assert reading.hvac_kw == pytest.approx(info['dc_joint_full_hvac_v1_power_kW'])
    assert reading.total_kw == pytest.approx(info['dc_joint_full_total_v1_power_kW'])
    assert reading.total_kw - legacy.total_kw == pytest.approx(
        reading.components_kw['crac_fan'] + reading.components_kw['cw_pump']
        + reading.components_kw['ct_pump'])
    legacy_model.dc_env.close()


@pytest.mark.parametrize('utilization,ambient', [(0.0, 4.0), (0.5, 20.0), (1.0, 40.0)])
def test_electrical_components_and_units_conserve_without_thermal_heat(model, utilization, ambient):
    reading = model.measure(utilization, utilization, utilization,
                            replace(SAMPLE, temperature_c=ambient, wet_bulb_c=ambient * 0.9))
    parts = reading.components_kw
    assert set(parts) == set(COMPONENTS)
    assert all(np.isfinite(v) and v >= 0 for v in parts.values())
    assert reading.accounting_boundary == ACCOUNTING_BOUNDARY
    assert reading.it_kw == parts['it']
    assert reading.hvac_kw == pytest.approx(sum(parts[k] for k in COMPONENTS[1:]))
    assert reading.total_kw == pytest.approx(sum(parts.values()))
    assert parts['crac_fan'] == pytest.approx(model.dc_env.CRAC_Fan_load / 1000)
    assert parts['cw_pump'] == pytest.approx((300000 * 0.0011 / 0.87) / 1000)
    assert parts['ct_pump'] == pytest.approx((300000 * 0.0011 / 0.87) / 1000)
    assert reading.crac_thermal_kw == pytest.approx(model.dc_env.CRAC_Cooling_load / 1000)
    assert 'crac_thermal' not in parts
    energy = {k: v * 0.25 for k, v in parts.items()}
    carbon = {k: v * SAMPLE.ci_g_per_kwh / 1000 for k, v in energy.items()}
    assert sum(energy.values()) == pytest.approx(reading.total_kw * 0.25)
    assert sum(carbon.values()) == pytest.approx(reading.total_kw * 0.25 * 100 / 1000)
    assert json.loads(json.dumps(reading.to_dict()))['components_kw'] == dict(parts)
    with pytest.raises(TypeError):
        parts['it'] = 0


def test_temperature_audit_action_masks_and_real_physical_control(model):
    assert model.supply_approach_c == pytest.approx(5.15)
    assert (model.inlet_model_min_c, model.inlet_model_max_c) == (16, 28)
    assert model.setpoint == 22
    assert model.action_mask() == (True, True, False)
    with pytest.raises(ValueError, match='masked'):
        model.apply_action(2)
    warm = model.measure(0.5, 0.5, 0.5, SAMPLE)
    assert warm.rack_inlet_max_c == pytest.approx(27.15)
    for _ in range(4):
        model.apply_action(0)
    assert model.setpoint == 18
    assert model.action_mask() == (False, True, True)
    with pytest.raises(ValueError, match='masked'):
        model.apply_action(0)
    cool = model.measure(0.5, 0.5, 0.5, SAMPLE)
    assert cool.rack_inlet_max_c == pytest.approx(23.15)
    assert warm.it_kw != pytest.approx(cool.it_kw)
    assert warm.hvac_kw != pytest.approx(cool.hvac_kw)
    assert warm.rack_outlet_max_c != pytest.approx(cool.rack_outlet_max_c)
    assert max(warm.rack_inlet_max_c, cool.rack_inlet_max_c) <= model.rack_inlet_limit_c
    for invalid in (17, 23, float('nan'), True):
        with pytest.raises(ValueError):
            model.set_setpoint(invalid)
    model.reset()
    assert model.setpoint == 22


def test_cooling_observation_is_causal_current_load_and_preserves_physical_sensors(model):
    initial = model.cooling_observation(0.1, 0.2, 0.3, SAMPLE)
    assert not initial['thermal_sensor_available']
    assert initial['previous_it_kw'] == initial['previous_hvac_kw'] == 0
    assert model._last_reading is None
    assert not model.dc_env.dc.rackwise_inlet_temp
    last = model.measure(0.1, 0.2, 0.3, SAMPLE)
    info_before = model.dc_env.info.copy()
    inlet_before = list(model.dc_env.dc.rackwise_inlet_temp)
    fan_before = model.dc_env.dc.racks_list[0].v_fan_rack.copy()
    observed = model.cooling_observation(0.8, 0.7, 0.6, SAMPLE)
    assert observed['thermal_sensor_available']
    assert observed['cpu_utilization'] == 0.8
    assert observed['current_it_kw'] > last.it_kw
    assert observed['previous_it_kw'] == last.it_kw
    assert observed['previous_hvac_kw'] == last.hvac_kw
    assert observed['previous_rack_outlet_max_c'] == last.rack_outlet_max_c
    assert observed['previous_rack_inlet_max_c'] == last.rack_inlet_max_c
    assert model.dc_env.info == info_before
    assert model.dc_env.dc.rackwise_inlet_temp == inlet_before
    np.testing.assert_array_equal(model.dc_env.dc.racks_list[0].v_fan_rack, fan_before)
    current = model.measure(0.8, 0.7, 0.6, SAMPLE)
    assert observed['current_it_kw'] == pytest.approx(current.it_kw, rel=1e-12)
    model.reset()
    assert not model.cooling_observation(0.0, 0.0, 0.0, SAMPLE)['thermal_sensor_available']


def test_invalid_utilization_and_extrapolated_cooling_ranges_fail(model):
    for values in ((-0.01, 0, 0), (0, 1.01, 0), (0, 0, float('nan'))):
        with pytest.raises(ValueError):
            model.measure(*values, SAMPLE)
        with pytest.raises(ValueError):
            model.cooling_observation(*values, SAMPLE)
    for override in ({'max_setpoint_c': 27}, {'initial_setpoint_c': 21},
                     {'action_step_c': 0.5}, {'rack_inlet_limit_c': 29}):
        with pytest.raises(ValueError):
            JointPowerModel(CONFIG, MODEL, cooling_config=override)
    with pytest.raises(ValueError):
        replace(model.measure(0.5, 0.5, 0.5, SAMPLE), total_kw=1)
