"""One system cost per physical interval; deadline misses are charged once."""
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class JointRewardConfig:
    alpha: float = 1.
    beta: float = 1.
    gamma: float = 10.
    carbon_scale_kg: float = 1000.
    wait_scale_task_minutes: float = 15000.
    deadline_scale_tasks: float = 1000.
    terminal_scale_tasks: float = 1000.
    terminal_weight: float = 10.
    thermal_weight: float = 10.
    thermal_scale_degree_steps: float = 1.
    work_minutes: float = 120.

    def __post_init__(self):
        for name, value in vars(self).items():
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError('Reward coefficients/scales must be positive finite: ' + name)


class SystemReward:
    def __init__(self, config=None):
        self.config = config if isinstance(config, JointRewardConfig) else JointRewardConfig(**(config or {}))
        self.penalized = set()

    def calculate(self, executor, interval, *, terminal, inlet_limit_c):
        now, config = executor.observed_end, self.config
        newly, unfinished, remaining = [], [], [0., 0., 0.]
        for key, entry in executor.ledger.items():
            spec, task = entry['spec'], entry['task']
            missed = ((entry['state'] == 'completed' and not task.sla_met) or
                      (entry['state'] != 'completed' and spec.sla_deadline <= now))
            if missed and key not in self.penalized:
                newly.append(key)
                self.penalized.add(key)
            if terminal and entry['state'] != 'completed':
                unfinished.append(key)
                minutes = spec.duration if entry['state'] == 'pending' else max(
                    0., (task.finish_time - now).total_seconds() / 60)
                for i, demand in enumerate((spec.cores_req, spec.gpu_req, spec.mem_req)):
                    remaining[i] += demand * math.ceil(minutes / 15) * 15
        carbon = interval['carbon_kg'] / config.carbon_scale_kg
        wait_minutes = interval['pending'] * 15
        wait = wait_minutes / config.wait_scale_task_minutes
        deadline = len(newly) / config.deadline_scale_tasks
        work = max(v / (c * config.work_minutes) for v, c in zip(remaining, executor.pool.capacity))
        liability = len(unfinished) / config.terminal_scale_tasks + work
        excess = max(0., interval['rack_inlet_max_c'] - inlet_limit_c)
        terms = dict(carbon_kg=interval['carbon_kg'], wait_task_minutes=wait_minutes,
                     new_deadline_misses=len(newly), new_deadline_miss_ids=sorted(newly),
                     unfinished_tasks=len(unfinished), remaining_resource_work_minutes=remaining,
                     terminal_liability_normalized=liability, thermal_excess_c=excess,
                     carbon_cost=config.alpha * carbon, wait_cost=config.beta * wait,
                     deadline_cost=config.gamma * deadline,
                     thermal_safety_cost=config.thermal_weight * excess / config.thermal_scale_degree_steps,
                     terminal_cost=config.terminal_weight * liability)
        reward = -math.fsum(terms[k] for k in (
            'carbon_cost', 'wait_cost', 'deadline_cost', 'thermal_safety_cost', 'terminal_cost'))
        return reward, terms
