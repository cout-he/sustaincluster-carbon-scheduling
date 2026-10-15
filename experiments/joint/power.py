"""One upstream physical model with the explicit joint_full_hvac_v1 boundary.

The historical dc_gym totals deliberately remain unchanged.  Its pump info keys
are historically labelled kW but contain W, so this adapter consumes the raw W
attributes instead.  CRAC cooling load is heat, not an electrical component.
"""

import math
from dataclasses import dataclass
from numbers import Integral
from types import MappingProxyType
from typing import Mapping

from experiments.e1.contracts import finite


ACCOUNTING_BOUNDARY = 'joint_full_hvac_v1'
COMPONENTS = ('it', 'crac_fan', 'ct_fan', 'compressor', 'cw_pump', 'ct_pump')
HVAC_COMPONENTS = COMPONENTS[1:]
DEFAULT_COOLING = dict(min_setpoint_c=18.0, max_setpoint_c=22.0,
                       initial_setpoint_c=22.0, action_step_c=1.0,
                       rack_inlet_limit_c=28.0)


def _temperature(value, name):
    if isinstance(value, bool):
        raise ValueError(f'{name} must be a finite temperature')
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f'{name} must be a finite temperature')
    return result


@dataclass(frozen=True)
class JointPowerReading:
    it_kw: float
    hvac_kw: float
    total_kw: float
    components_kw: Mapping[str, float]
    rack_inlet_max_c: float
    rack_outlet_max_c: float
    setpoint_c: float
    crac_thermal_kw: float
    legacy_hvac_kw: float
    legacy_total_kw: float
    accounting_boundary: str = ACCOUNTING_BOUNDARY

    def __post_init__(self):
        if self.accounting_boundary != ACCOUNTING_BOUNDARY:
            raise ValueError('Unsupported joint power accounting boundary')
        for name in ('it_kw', 'hvac_kw', 'total_kw', 'crac_thermal_kw',
                     'legacy_hvac_kw', 'legacy_total_kw'):
            finite(getattr(self, name), name)
        for name in ('rack_inlet_max_c', 'rack_outlet_max_c', 'setpoint_c'):
            _temperature(getattr(self, name), name)
        if set(self.components_kw) != set(COMPONENTS):
            raise ValueError('Power components must contain exactly the six joint electrical terms')
        components = {name: finite(self.components_kw[name], name) for name in COMPONENTS}
        for actual, expected, name in (
            (self.it_kw, components['it'], 'IT'),
            (self.hvac_kw, math.fsum(components[k] for k in HVAC_COMPONENTS), 'HVAC'),
            (self.total_kw, math.fsum(components.values()), 'total'),
            (self.total_kw, self.it_kw + self.hvac_kw, 'IT + HVAC'),
            (self.legacy_hvac_kw, components['ct_fan'] + components['compressor'], 'legacy HVAC'),
            (self.legacy_total_kw, self.it_kw + self.legacy_hvac_kw, 'legacy total'),
        ):
            if not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-9):
                raise ValueError(f'Inconsistent {name} power accounting')
        object.__setattr__(self, 'components_kw', MappingProxyType(components))

    def to_dict(self):
        return {name: (dict(value) if name == 'components_kw' else value)
                for name in self.__dataclass_fields__
                for value in (getattr(self, name),)}


# A concise compatibility name for callers that expect an adapter reading.
PowerReading = JointPowerReading


class JointPowerModel:
    """Canonical IT/HVAC equations with one physical advance per interval.

    The checked-in configuration has a 5.15 C averaged supply approach. Its
    rack-inlet fitting domain is 16..28 C; 18..22 C CRAC setpoints therefore
    give 23.15..27.15 C inlets. This conservative range contains the historical
    22 C baseline and requires no extrapolation. The 28 C inlet ceiling denotes
    the model's fitted domain, not a hardware certification or SLA guarantee.
    """

    def __init__(self, config, model_path, *, cooling_config=None):
        from utils.make_envs import make_dc_env

        cooling = dict(DEFAULT_COOLING)
        cooling.update(cooling_config or {})
        for name in DEFAULT_COOLING:
            cooling[name] = _temperature(cooling[name], name)
        self.cooling_config = cooling
        self.min_setpoint_c = cooling['min_setpoint_c']
        self.max_setpoint_c = cooling['max_setpoint_c']
        self.action_step_c = cooling['action_step_c']
        self.rack_inlet_limit_c = cooling['rack_inlet_limit_c']
        if (not self.min_setpoint_c < self.max_setpoint_c
                or not self.min_setpoint_c <= 22.0 <= self.max_setpoint_c
                or cooling['initial_setpoint_c'] != 22.0
                or self.action_step_c != 1.0):
            raise ValueError('Joint cooling needs a nontrivial range containing initial 22 C and 1 C actions')
        # Keep historical equipment sizing, IT equations and capacity semantics.
        # In particular, do not resize the plant as a function of policy actions.
        capacity_mw = (6 * config['total_cores'] + 500 * config['total_gpus']
                       + 2.5 * config['total_mem']) / 1e6
        self.dc_env, *_ = make_dc_env(
            month=1, location=config['location'], dc_config_file=str(model_path),
            datacenter_capacity_mw=capacity_mw, max_bat_cap_Mw=0,
            use_ls_cpu_load=True, add_cpu_usage=False,
            total_cores=config['total_cores'], total_gpus=config['total_gpus'],
            dc_memory_GB=config['total_mem'])
        self._seed = int(config.get('seed', 123))
        rack = self.dc_env.dc.racks_list[0]
        # The canonical model uses the first rack and broadcasts it uniformly.
        self.supply_approach_c = float(rack.clamp_supply_approach_temp(
            self.dc_env.dc.rack_supply_approach_temp_list[0]))
        self.inlet_model_min_c, self.inlet_model_max_c = map(
            float, self.dc_env.DC_Config.INLET_TEMP_RANGE)
        if (self.min_setpoint_c + self.supply_approach_c < self.inlet_model_min_c
                or self.max_setpoint_c + self.supply_approach_c > self.inlet_model_max_c
                or not self.inlet_model_min_c < self.rack_inlet_limit_c <= self.inlet_model_max_c
                or self.max_setpoint_c + self.supply_approach_c > self.rack_inlet_limit_c):
            raise ValueError('Cooling range exceeds the upstream rack inlet domain or safety ceiling')
        self.reset()

    @property
    def setpoint(self):
        return self._setpoint

    @property
    def setpoint_c(self):
        return self._setpoint

    def reset(self):
        self.dc_env.reset(seed=self._seed)
        self._setpoint = self.cooling_config['initial_setpoint_c']
        self._last_reading = None

    def set_setpoint(self, setpoint):
        setpoint = _temperature(setpoint, 'setpoint')
        if not self.min_setpoint_c <= setpoint <= self.max_setpoint_c:
            raise ValueError('Cooling setpoint is outside the legal range')
        self._setpoint = setpoint

    def action_mask(self):
        return (self.setpoint - self.action_step_c >= self.min_setpoint_c,
                True,
                self.setpoint + self.action_step_c <= self.max_setpoint_c)

    def apply_action(self, action):
        if isinstance(action, bool) or not isinstance(action, Integral) or action not in (0, 1, 2):
            raise ValueError('Cooling action must be 0 (lower), 1 (hold), or 2 (raise)')
        if not self.action_mask()[int(action)]:
            raise ValueError('Cooling action is masked at the setpoint boundary')
        self.set_setpoint(self.setpoint + (int(action) - 1) * self.action_step_c)
        return self.setpoint

    @staticmethod
    def _loads(cpu, gpu, mem):
        result = tuple(finite(v, name) for name, v in (('cpu', cpu), ('gpu', gpu), ('mem', mem)))
        if any(v > 1 for v in result):
            raise ValueError('Utilization exceeds resource capacity')
        return result

    def _current_it_kw(self, cpu, gpu, mem):
        """Evaluate existing rack IT equations at current load and old setpoint.

        These algebraic equations have no elapsed-time state. Preserve the
        canonical routine's fan-cache side effect so observing cannot advance
        or overwrite the last physical thermal/power sensors. No outlet or
        post-action HVAC calculation is performed here.
        """
        rack = self.dc_env.dc.racks_list[0]
        previous_fan = rack.v_fan_rack
        try:
            cpu_w, fan_w, gpu_w = rack.compute_instantaneous_pwr_vecd(
                inlet_temp=self.setpoint + self.supply_approach_c,
                ITE_load_pct=cpu * 100, GPU_load_pct=gpu * 100,
                MEMORY_load_pct=mem * 100)
        finally:
            rack.v_fan_rack = previous_fan
        memory_w = 0.07 * self.dc_env.dc_memory_GB
        return finite((float(cpu_w + fan_w + gpu_w) * len(self.dc_env.dc.racks_list)
                       + memory_w) / 1000, 'current IT power')

    def cooling_observation(self, cpu, gpu, mem, sample):
        cpu, gpu, mem = self._loads(cpu, gpu, mem)
        last = self._last_reading
        hour = sample.time.hour + sample.time.minute / 60
        day = sample.time.dayofyear - 1 + hour / 24
        return dict(
            cpu_utilization=cpu, gpu_utilization=gpu, mem_utilization=mem,
            current_it_kw=self._current_it_kw(cpu, gpu, mem),
            previous_it_kw=last.it_kw if last else 0.0,
            previous_hvac_kw=last.hvac_kw if last else 0.0,
            setpoint_c=self.setpoint,
            previous_rack_inlet_max_c=last.rack_inlet_max_c if last else self.setpoint + self.supply_approach_c,
            previous_rack_outlet_max_c=last.rack_outlet_max_c if last else 0.0,
            thermal_sensor_available=bool(last),
            temperature_c=_temperature(sample.temperature_c, 'temperature_c'),
            wet_bulb_c=_temperature(sample.wet_bulb_c, 'wet_bulb_c'),
            ci_g_per_kwh=finite(sample.ci_g_per_kwh, 'carbon intensity'),
            hour_sin=math.sin(2 * math.pi * hour / 24),
            hour_cos=math.cos(2 * math.pi * hour / 24),
            day_sin=math.sin(2 * math.pi * day / 365.25),
            day_cos=math.cos(2 * math.pi * day / 365.25))

    def measure(self, cpu, gpu, mem, sample):
        cpu, gpu, mem = self._loads(cpu, gpu, mem)
        self.dc_env.update_workloads(cpu, mem, gpu)
        self.dc_env.set_ambient_temp(
            _temperature(sample.temperature_c, 'temperature_c'),
            _temperature(sample.wet_bulb_c, 'wet_bulb_c'))
        _, _, _, _, info = self.dc_env.step(self.setpoint)
        # All six calculate_HVAC_power returns are W (third is W thermal).
        # IT already includes server fans, CPU, GPU and background DRAM.
        components = dict(it=float(info['dc_ITE_total_power_kW']),
                          crac_fan=float(self.dc_env.CRAC_Fan_load) / 1000,
                          ct_fan=float(self.dc_env.CT_Cooling_load) / 1000,
                          compressor=float(self.dc_env.Compressor_load) / 1000,
                          cw_pump=float(self.dc_env.CW_pump_load) / 1000,
                          ct_pump=float(self.dc_env.CT_pump_load) / 1000)
        hvac_kw = math.fsum(components[k] for k in HVAC_COMPONENTS)
        reading = JointPowerReading(
            it_kw=components['it'], hvac_kw=hvac_kw,
            total_kw=components['it'] + hvac_kw, components_kw=components,
            rack_inlet_max_c=max(map(float, self.dc_env.dc.rackwise_inlet_temp)),
            rack_outlet_max_c=max(map(float, self.dc_env.rackwise_outlet_temp)),
            setpoint_c=self.setpoint,
            crac_thermal_kw=float(self.dc_env.CRAC_Cooling_load) / 1000,
            legacy_hvac_kw=float(info['dc_HVAC_total_power_kW']),
            legacy_total_kw=float(info['dc_total_power_kW']))
        self._last_reading = reading
        return reading

    def close(self):
        self.dc_env.close()
