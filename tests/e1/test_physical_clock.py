"""Audit real make_dc_env month and reset RNG effects without new physics."""
from .helpers import START


def test_month_and_reset_seed_do_not_change_supplied_load_weather_power(monkeypatch):
    from experiments.e1.config import ROOT
    from experiments.e1.environment import EnvironmentalSample
    from experiments.e1.power import UpstreamPowerModel
    import utils.make_envs as factory
    cfg = dict(total_cores=50000, total_gpus=1000, total_mem=80000,
               location='US-CAL-CISO', hvac_setpoint_c=22)
    january = UpstreamPowerModel(cfg, ROOT / 'configs/dcs/dc_config.json')
    original = factory.make_dc_env
    def august_factory(*args, **kwargs):
        kwargs['month'] = 8
        return original(*args, **kwargs)
    monkeypatch.setattr(factory, 'make_dc_env', august_factory)
    august = UpstreamPowerModel(cfg, ROOT / 'configs/dcs/dc_config.json')
    august.dc_env.seed = 98765
    august.dc_env.reset(seed=98765)
    for cpu, gpu, mem, temp in ((0, 0, 0, 20), (0.25, 0.5, 0.75, 30), (0.5, 0.1, 0.2, 15)):
        sample = EnvironmentalSample(START, 200, temp, temp * 0.9)
        assert january.measure(cpu, gpu, mem, sample) == august.measure(cpu, gpu, mem, sample)
