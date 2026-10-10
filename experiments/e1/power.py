"""Adapter to upstream equations; only the order of supplying inputs changes."""
import math
from dataclasses import dataclass

from .contracts import finite


@dataclass(frozen=True)
class PowerReading:
    it_kw: float
    hvac_kw: float
    total_kw: float

    def __post_init__(self):
        for name in ('it_kw', 'hvac_kw', 'total_kw'):
            finite(getattr(self, name), name)
        if not math.isclose(self.total_kw, self.it_kw + self.hvac_kw, rel_tol=1e-10, abs_tol=1e-9):
            raise ValueError('Total power must equal IT + upstream HVAC power')


class UpstreamPowerModel:
    def __init__(self, config, model_path):
        from utils.make_envs import make_dc_env
        # Same resource-based sizing as SustainDC. No battery or transfer model.
        capacity_mw = (6 * config['total_cores'] + 500 * config['total_gpus']
                       + 2.5 * config['total_mem']) / 1e6
        self.dc_env, *_ = make_dc_env(
            month=1, location=config['location'], dc_config_file=str(model_path),
            datacenter_capacity_mw=capacity_mw, max_bat_cap_Mw=0,
            use_ls_cpu_load=True, add_cpu_usage=False,
            total_cores=config['total_cores'], total_gpus=config['total_gpus'],
            dc_memory_GB=config['total_mem'])
        self.setpoint = config['hvac_setpoint_c']
        self.dc_env.reset(seed=123)

    def measure(self, cpu, gpu, mem, sample):
        for name, value in (('cpu', cpu), ('gpu', gpu), ('mem', mem)):
            if finite(value, name) > 1:
                raise ValueError(f'{name} utilization exceeds capacity')
        # The interval load is supplied BEFORE step(), removing upstream load lag.
        self.dc_env.update_workloads(cpu, mem, gpu)
        self.dc_env.set_ambient_temp(sample.temperature_c, sample.wet_bulb_c)
        _, _, _, _, info = self.dc_env.step(self.setpoint)
        # Missing/invalid fields and model errors must fail, never use stale zeros.
        return PowerReading(float(info['dc_ITE_total_power_kW']),
                            float(info['dc_HVAC_total_power_kW']),
                            float(info['dc_total_power_kW']))
