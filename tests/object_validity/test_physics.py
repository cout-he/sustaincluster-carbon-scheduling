"""Read-only A0 probes test physical identity and electrical accounting."""

from pathlib import Path

import pytest

from experiments.joint.power import JointPowerModel
from experiments.object_validity.a0_physics import effective_mapping, probe, source_hashes


ROOT = Path(__file__).resolve().parents[2]
CONFIG = dict(total_cores=50000, total_gpus=1000, total_mem=80000,
              location='US-CAL-CISO', seed=123)


@pytest.fixture
def model():
    instance = JointPowerModel(CONFIG, ROOT / 'configs/dcs/dc_config.json')
    yield instance
    instance.close()


def test_effective_mapping_counts_instantiated_objects_not_declared_gpu_list(model):
    mapping = effective_mapping(model)
    assert mapping['logical_cpu_cores'] == 50000
    assert mapping['logical_gpus'] == 1000
    assert mapping['rack_count'] == 32
    assert mapping['declared_gpu_configs'] == 1024
    assert mapping['instantiated_cpu_server_objects'] == 800
    assert mapping['represented_cpu_cores'] == 51200
    assert mapping['instantiated_gpu_power_objects'] == 800
    assert mapping['gpu_power_object_to_logical_capacity_ratio'] == pytest.approx(.8)


def test_power_parts_sum_and_probe_does_not_modify_production_sources(model):
    before = source_hashes(ROOT)
    row = probe(model, .5, 1, .0, 19, 25, 'test')
    assert row['system_kw'] == pytest.approx(sum(row[k + '_kw'] for k in (
        'it', 'crac_fan', 'ct_fan', 'compressor', 'cw_pump', 'ct_pump')))
    assert row['it_kw'] == pytest.approx(sum(row[k + '_kw'] for k in (
        'cpu', 'gpu', 'server_fan', 'memory_background')))
    assert row['rack_inlet_max_c'] == pytest.approx(24.15)
    assert row['cw_pump_kw'] == pytest.approx(300000 * .0011 / .87 / 1000)
    assert source_hashes(ROOT) == before


def test_algebraic_probe_repeat_and_temperature_constraint(model):
    a = probe(model, 0, .5, 1, 22, 35, 'a')
    probe(model, 1, 1, 1, 18, -5, 'other')
    b = probe(model, 0, .5, 1, 22, 35, 'b')
    assert a['system_kw'] == pytest.approx(b['system_kw'], rel=1e-12)
    assert a['rack_inlet_max_c'] == pytest.approx(27.15)
    assert a['rack_inlet_margin_c'] == pytest.approx(.85)
