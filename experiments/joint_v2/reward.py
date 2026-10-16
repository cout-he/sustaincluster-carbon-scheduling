"""V2 interval costs: real incremental carbon and bounded external DMR weight.

Only the current sample's independent idle-18 reference is subtracted. The
future fixed-window background is a reporting identity, never a terminal cost.
"""
import math
from dataclasses import fields

from experiments.joint.reward import JointRewardConfig, SystemReward


class V2SystemReward(SystemReward):
    def __init__(self, config, reference_provider, stage_provider):
        config = dict(config)
        if config.get('carbon_mode', 'incremental_idle18') != 'incremental_idle18':
            raise ValueError('V2 requires its versioned incremental_idle18 carbon objective')
        self.deadline_multiplier = float(config.get('deadline_multiplier', config.get('gamma', 10.)))
        if not math.isfinite(self.deadline_multiplier) or self.deadline_multiplier < 0:
            raise ValueError('Deadline multiplier must be finite and nonnegative')
        super().__init__({f.name: config[f.name] for f in fields(JointRewardConfig) if f.name in config})
        self.reference_provider = reference_provider
        self.stage_provider = stage_provider

    def set_deadline_multiplier(self, value):
        value = float(value)
        if not math.isfinite(value) or value < 0:
            raise ValueError('Deadline multiplier must be finite and nonnegative')
        self.deadline_multiplier = value

    def calculate(self, executor, interval, *, terminal, inlet_limit_c):
        _, terms = super().calculate(executor, interval, terminal=terminal, inlet_limit_c=inlet_limit_c)
        reference = self.reference_provider()
        difference_kw = interval['total_power_kw'] - reference.total_kw
        tolerance_kw = 1e-8 * max(1., reference.total_kw)
        if difference_kw < -tolerance_kw:
            raise RuntimeError('Canonical model violates the assumed idle18 lower bound; cannot train V2 objective')
        increment_kw = max(0., difference_kw)
        hours, ci = interval['interval_hours'], interval['ci_g_per_kwh']
        background_kg = reference.total_kw * hours * ci / 1000
        increment_kg = increment_kw * hours * ci / 1000
        interval.update(background_carbon_kg=background_kg, incremental_carbon_kg=increment_kg,
                        background_energy_kwh=reference.total_kw * hours,
                        incremental_energy_kwh=increment_kw * hours,
                        stage=self.stage_provider(), carbon_objective='incremental_idle18')
        for name, power in reference.components_kw.items():
            interval['background_' + name + '_power_kw'] = power
            interval['background_' + name + '_energy_kwh'] = power * hours
            interval['background_' + name + '_carbon_kg'] = power * hours * ci / 1000
        terms.update(actual_carbon_kg=interval['carbon_kg'], background_carbon_kg=background_kg,
                     incremental_carbon_kg=increment_kg, training_carbon_kg=increment_kg,
                     carbon_mode='incremental_idle18', deadline_multiplier=self.deadline_multiplier,
                     carbon_cost=self.config.alpha * increment_kg / self.config.carbon_scale_kg,
                     deadline_cost=self.deadline_multiplier * terms['new_deadline_misses'] /
                     self.config.deadline_scale_tasks)
        reward = -math.fsum(terms[k] for k in (
            'carbon_cost', 'wait_cost', 'deadline_cost', 'thermal_safety_cost', 'terminal_cost'))
        return reward, terms
