import pytest

from .helpers import execute, task


def test_interval_power_uses_current_load_and_carbon_units():
    result = execute([task('a', duration=5)])
    # Busy [05:00,05:15): IT 4 kW, HVAC 2 kW, CI 100 g/kWh.
    # Idle [05:15,05:30): IT 2 kW, HVAC 2 kW, CI 200 g/kWh.
    assert [s['it_power_kw'] for s in result.steps] == [4, 2]
    assert [s['total_energy_kwh'] for s in result.steps] == [1.5, 1]
    assert [s['carbon_kg'] for s in result.steps] == pytest.approx([0.15, 0.2])
    assert result.summary['it_energy_kwh'] == 1.5
    assert result.summary['hvac_energy_kwh'] == 1
    assert result.summary['total_energy_kwh'] == 2.5
    assert result.summary['carbon_kg'] == pytest.approx(0.35)
    assert all(s['transmission_energy_kwh'] == 0 for s in result.steps)


def test_invalid_power_cannot_be_silently_integrated():
    from experiments.e1.power import PowerReading
    for values in [(1, 2, 4), (float('nan'), 0, 1), (1, -1, 0)]:
        with pytest.raises(ValueError): PowerReading(*values)


def test_real_upstream_power_adapter_has_no_first_step_load_lag():
    from experiments.e1.power import UpstreamPowerModel
    from experiments.e1.config import ROOT
    from experiments.e1.environment import EnvironmentalSample
    from .helpers import START
    cfg = dict(total_cores=50000, total_gpus=1000, total_mem=80000, location='US-CAL-CISO',
               hvac_setpoint_c=22)
    model = UpstreamPowerModel(cfg, ROOT / 'configs/dcs/dc_config.json')
    sample = EnvironmentalSample(START, 100, 20, 18)
    idle = model.measure(0, 0, 0, sample)
    busy = model.measure(0.5, 0.5, 0.5, sample)
    assert busy.it_kw > idle.it_kw
    assert busy.total_kw == pytest.approx(busy.it_kw + busy.hvac_kw)
    assert model.dc_env.info['dc_cpu_workload_fraction'] == 0.5
    assert model.dc_env.info['dc_exterior_ambient_temp'] == 20
