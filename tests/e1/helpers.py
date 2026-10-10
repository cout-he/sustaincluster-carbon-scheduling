import pandas as pd

START = pd.Timestamp('2023-08-01T05:00:00Z')


def task(task_id, arrival=0, duration=15, cpu=1, gpu=0, mem=1, sla=1.5):
    from experiments.e1.workload import TaskSpec
    return TaskSpec(task_id, task_id, 0, ord(task_id[0]),
                    START + pd.Timedelta(minutes=arrival), duration, cpu, gpu,
                    mem, 123, sla, 1, tuple(range(12)))


def inputs(tasks, minutes=30):
    from experiments.e1.workload import FrozenWorkload
    from experiments.e1.environment import FrozenEnvironment, EnvironmentalSample
    end = START + pd.Timedelta(minutes=minutes)
    workload = FrozenWorkload(START, end, 'fixture-source', 1, 1.5, 2020, tuple(tasks))
    environment = FrozenEnvironment(START, end, 'fixture-ci', 'fixture-weather', tuple(
        EnvironmentalSample(START + pd.Timedelta(minutes=i), 100 + 100 * (i // 15), 20, 18)
        for i in range(0, minutes, 15)))
    return workload, environment


class KnownPower:
    """Isolate the physical dependency, retain the real Task/resource executor."""
    def measure(self, cpu, gpu, mem, sample):
        from experiments.e1.power import PowerReading
        return PowerReading(2 + 2 * cpu, 2, 4 + 2 * cpu)


def execute(tasks, minutes=30, capacity=(1, 1, 1), power=None):
    from experiments.e1.engine import LocalExecutor
    workload, environment = inputs(tasks, minutes)
    return LocalExecutor(workload, environment, capacity, power or KnownPower()).run()
