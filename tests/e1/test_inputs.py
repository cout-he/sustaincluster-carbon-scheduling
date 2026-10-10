"""Input contracts: catch column shifts, RNG coupling and accidental extrapolation."""
import json
import random
from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pytest

START = pd.Timestamp('2023-08-01T05:00:00Z')
END = pd.Timestamp('2023-08-01T05:30:00Z')


def trace_file(tmp_path, tasks=None):
    if tasks is None:
        tasks = [['same-job', 1, 2, '1970-01-01T00:00:00Z', 10,
                  200, 50, 3, 99, 7, 'Thursday', 3]]
    path = tmp_path / 'trace.pkl'
    pd.DataFrame([
        {'interval_15m': pd.Timestamp('2020-08-01T05:00:00Z'),
         'tasks_matrix': np.array(tasks, dtype=object)},
        {'interval_15m': pd.Timestamp('2020-08-01T05:30:00Z'),
         'tasks_matrix': np.array(tasks, dtype=object)},
    ]).to_pickle(path)
    return path


def load(path, **kwargs):
    from experiments.e1.workload import load_workload
    return load_workload(path, START, END, resource_multiplier=kwargs.get('scale', 5),
                         sla_multiplier=kwargs.get('sla', 1.5), dc_id=1, trace_year=2020)


def test_columns_scale_and_synthetic_deadline(tmp_path):
    task = load(trace_file(tmp_path)).tasks[0]
    assert (task.cores_req, task.gpu_req, task.mem_req) == (10, 2.5, 15)
    assert task.bandwidth_gb == 7  # field 9, never GPU memory field 8=99
    assert task.arrival_time == START
    assert task.duration == 10
    assert task.sla_deadline == pd.Timestamp('2023-08-01T05:15:00Z')
    other = load(trace_file(tmp_path), scale=2, sla=2).tasks[0]
    assert (other.cores_req, other.gpu_req, other.mem_req) == (4, 1, 6)
    assert other.duration == 10 and other.bandwidth_gb == 7
    assert other.sla_deadline == pd.Timestamp('2023-08-01T05:20:00Z')


def test_freeze_roundtrip_is_immutable_and_excludes_end_arrival(tmp_path):
    from experiments.e1.workload import FrozenWorkload
    inputs = load(trace_file(tmp_path))
    assert len(inputs.tasks) == 1
    with pytest.raises(FrozenInstanceError):
        inputs.tasks[0].duration = 100
    path = tmp_path / 'workload.json'
    inputs.save(path)
    assert FrozenWorkload.load(path) == inputs
    original = json.loads(path.read_text())
    original['payload']['tasks'][0]['duration'] = 90
    path.write_text(json.dumps(original))
    with pytest.raises(ValueError, match='hash'):
        FrozenWorkload.load(path)


def test_duplicate_job_names_have_unique_ids_and_do_not_consume_rng(tmp_path):
    rows = [['same', 1, 2, '1970-01-01', 10, 100, 0, 1, 88, 2, 'x', 1]] * 2
    path = trace_file(tmp_path, rows)
    random.seed(999); np.random.seed(999)
    a = load(path)
    random.random(); np.random.random(10)
    b = load(path)
    assert a == b
    assert len({t.task_id for t in a.tasks}) == 2
    saved = random.getstate()
    task = a.tasks[0].make_task()
    assert random.getstate() == saved
    assert task.job_name == a.tasks[0].task_id
    assert task.origin_dc_id == 1 and task.sla_multiplier == 1.5


@pytest.mark.parametrize('column,value', [(4, 0), (5, -1), (6, float('nan')),
                                         (7, float('inf')), (9, -1)])
def test_invalid_trace_demands_fail(tmp_path, column, value):
    row = ['job', 1, 2, '1970-01-01', 10, 100, 0, 1, 99, 2, 'x', 1]
    row[column] = value
    with pytest.raises(ValueError):
        load(trace_file(tmp_path, [row]))


def test_wrong_schema_and_duplicate_buckets_fail(tmp_path):
    path = trace_file(tmp_path, [['job'] * 11])
    with pytest.raises(ValueError, match='12'):
        load(path)
    path = trace_file(tmp_path)
    frame = pd.read_pickle(path)
    pd.concat([frame, frame]).to_pickle(path)
    with pytest.raises(ValueError, match='duplicate'):
        load(path)


def test_missing_unused_end_metadata_freezes_as_null_without_changing_execution_inputs(tmp_path):
    from experiments.e1.workload import FrozenWorkload
    row = ['job', 1, float('nan'), '1970-01-01', 10, 100, 0, 1, 99, 2, 'x', 1]
    frozen = load(trace_file(tmp_path, [row]))
    path = tmp_path / 'frozen.json'; frozen.save(path)
    restored = FrozenWorkload.load(path)
    assert restored == frozen
    assert restored.tasks[0].raw_fields[2] is None
    assert restored.tasks[0].duration == 10 and restored.tasks[0].cores_req == 5


def environment_files(tmp_path):
    ci = tmp_path / 'ci.csv'
    pd.DataFrame({'Datetime (UTC)': ['2023-08-01T05:00:00Z', '2023-08-01T06:00:00Z'],
                  'Carbon Intensity gCO₂eq/kWh (direct)': [100, 200]}).to_csv(ci, index=False)
    weather = tmp_path / 'weather.json'
    weather.write_text(json.dumps({'utc_offset_seconds': 0, 'hourly': {
        'time': ['2023-08-01T05:00', '2023-08-01T06:00'],
        'temperature_2m': [20, 24]}}))
    return ci, weather


def test_environment_uses_actual_utc_interpolation_without_noise(tmp_path):
    from experiments.e1.environment import load_environment, FrozenEnvironment
    ci, weather = environment_files(tmp_path)
    inputs = load_environment(ci, weather, START, END)
    assert [s.ci_g_per_kwh for s in inputs.samples] == [100, 125]
    assert [s.temperature_c for s in inputs.samples] == [20, 21]
    assert [s.wet_bulb_c for s in inputs.samples] == pytest.approx([18, 18.9])
    assert [s.time for s in inputs.samples] == [START, START + pd.Timedelta(minutes=15)]
    path = tmp_path / 'environment.json'; inputs.save(path)
    assert FrozenEnvironment.load(path) == inputs


def test_environment_out_of_range_or_invalid_weather_offset_fails(tmp_path):
    from experiments.e1.environment import load_environment
    ci, weather = environment_files(tmp_path)
    with pytest.raises(ValueError, match='coverage'):
        load_environment(ci, weather, START - pd.Timedelta(hours=1), END)
    document = json.loads(weather.read_text()); document['utc_offset_seconds'] = '3600'
    weather.write_text(json.dumps(document))
    with pytest.raises(ValueError, match='offset'):
        load_environment(ci, weather, START, END)


def test_local_weather_uses_declared_fixed_offset_before_utc_interpolation(tmp_path):
    from experiments.e1.environment import load_environment, FrozenEnvironment
    ci, weather = environment_files(tmp_path)
    document = json.loads(weather.read_text())
    document.update(utc_offset_seconds=-25200, timezone='America/Los_Angeles')
    document['hourly']['time'] = ['2023-07-31T22:00', '2023-07-31T23:00']
    weather.write_text(json.dumps(document))
    frozen = load_environment(ci, weather, START, END)
    assert [s.temperature_c for s in frozen.samples] == [20, 21]
    assert frozen.payload()['weather_source_utc_offset_seconds'] == -25200
    assert frozen.payload()['weather_time_basis'] == 'metadata_fixed_offset'
    path = tmp_path / 'environment.json'; frozen.save(path)
    assert FrozenEnvironment.load(path) == frozen


def test_weather_naive_labels_without_offset_are_not_assumed_utc(tmp_path):
    from experiments.e1.environment import load_environment
    ci, weather = environment_files(tmp_path)
    document = json.loads(weather.read_text()); del document['utc_offset_seconds']
    weather.write_text(json.dumps(document))
    with pytest.raises(ValueError, match='offset'):
        load_environment(ci, weather, START, END)


def test_config_rejects_unaligned_naive_windows_and_unsupported_policies():
    from experiments.e1.config import validate_config
    from pathlib import Path
    import yaml
    config = yaml.safe_load((Path(__file__).parents[2] / 'experiments/e1/configs/single_dc.yaml').read_text())
    for key, value in [('start', '2023-08-01T05:01:00Z'),
                       ('start', '2023-08-01T05:00:00'),
                       ('end', config['start']), ('timestep_minutes', 5), ('policy', 'edf')]:
        bad = json.loads(json.dumps(config)); bad[key] = value
        with pytest.raises(ValueError): validate_config(bad)
