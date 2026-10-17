"""Three cost channels with one label per physical slot, six Train multipliers."""
import copy
import math
import numpy as np


def episode_costs(terms, total_tasks):
    if isinstance(total_tasks, bool) or int(total_tasks) != total_tasks or total_tasks < 0:
        raise ValueError('Final cohort task count must be a nonnegative integer')
    result = np.asarray([[t['incremental_carbon_kg'] / 1000.,
        t['new_deadline_misses'] / max(1, total_tasks),
        math.fsum(t.get(k, 0.) for k in ('wait_cost', 'thermal_safety_cost', 'terminal_cost'))]
        for t in terms], dtype=np.float64).reshape(-1, 3)
    if not np.isfinite(result).all() or (result < -1e-12).any():
        raise ValueError('Nonfinite or negative physical cost')
    return result


class GroupConstraints:
    def __init__(self, config=None):
        self.config = dict(initial_lambda=10., lambda_min=10., lambda_max=5000.,
                           eta_lambda=20., beta=.1, target_dmr=.05, **(config or {})) if not config else dict(config)
        cfg = self.config
        if not (0 < cfg['lambda_min'] <= cfg['initial_lambda'] <= cfg['lambda_max']
                and cfg['eta_lambda'] >= 0 and 0 < cfg['beta'] <= 1 and cfg['target_dmr'] == .05):
            raise ValueError('Invalid positive-floor grouped Deadline configuration')
        self.values = {self.key(p, s): float(cfg['initial_lambda'])
                       for p in (.75, 1., 1.25) for s in (1.5, 2.)}
        self.ema, self.counts = {}, {k: 0 for k in self.values}

    @staticmethod
    def key(pressure, sla):
        if pressure not in (.75, 1., 1.25) or sla not in (1.5, 2.):
            raise ValueError('Unknown pressure/SLA group')
        return f'{pressure:g}|{sla:g}'

    def update(self, key, dmr, *, role, complete):
        if role != 'train':
            raise ValueError('Constraint updates require Train episodes')
        if key not in self.values or not math.isfinite(dmr) or not 0 <= dmr <= 1:
            raise ValueError('Invalid group or observed DMR')
        if not complete:
            return dict(applied=False, reason='incomplete_cohort', value=self.values[key])
        cfg = self.config
        self.ema[key] = ((1-cfg['beta'])*self.ema[key]+cfg['beta']*dmr
                         if key in self.ema else float(dmr))
        old = self.values[key]
        self.values[key] = float(np.clip(old+cfg['eta_lambda']*(self.ema[key]-cfg['target_dmr']),
                                        cfg['lambda_min'], cfg['lambda_max']))
        self.counts[key] += 1
        return dict(applied=True, group=key, previous=old, value=self.values[key],
                    ema_dmr=self.ema[key], observed_dmr=dmr)

    def export_state(self):
        return copy.deepcopy(dict(config=self.config, values=self.values, ema=self.ema, counts=self.counts))

    @classmethod
    def from_state(cls, state):
        result = cls(state['config'])
        if set(state['values']) != set(result.values):
            raise ValueError('Saved constraint groups differ')
        result.values, result.ema, result.counts = (copy.deepcopy(state[k]) for k in ('values', 'ema', 'counts'))
        if any(not result.config['lambda_min'] <= v <= result.config['lambda_max'] for v in result.values.values()):
            raise ValueError('Saved constraint coefficient violates bounds')
        return result
