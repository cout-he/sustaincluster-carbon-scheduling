"""Independent original-resource/finish/release/integral and prefix checks."""
from dataclasses import replace
import numpy as np
import pytest
from .helpers import START, KnownPower, task
from .test_counterfactual import cohort


def pair():
    from experiments.e1.counterfactual import InterventionTarget, run_paired_case
    w, e = cohort([task('a', duration=16, cpu=.5, mem=.5, sla=3), task('b', arrival=15)])
    p = run_paired_case(w, e, (1, 1, 1), KnownPower, InterventionTarget('x', START, 'a'), minimum_common_end=e.end)
    return w, e, p


def test_independent_audit_conservation_finish_release_energy_and_once():
    from experiments.e1.counterfactual_audit import audit_pair
    w, e, p = pair()
    report = audit_pair(p, w, e, (1, 1, 1))
    assert report['passed'] and report['prefix_equal']
    assert report['arrived'] == report['completed_per_branch'] == 2
    assert report['delta_policy_defer_intervals'] == 1
    assert report['resource_hours_equal']


def test_tampered_power_resource_deadline_and_arrival_fail_independent_audit():
    from experiments.e1.counterfactual_audit import audit_result
    w, e, p = pair()
    r = p.results['force_execute']
    for field, value in [('carbon_kg', 999), ('used_cores', 999), ('it_energy_kwh', 999)]:
        steps = (dict(r.steps[0], **{field: value}),) + r.steps[1:]
        with pytest.raises((ValueError, RuntimeError), match='carbon|resource|energy|capacity'):
            audit_result(replace(r, steps=steps), w, e, (1, 1, 1))
    tasks = (dict(r.tasks[0], deadline_outcome='missed'),) + r.tasks[1:]
    with pytest.raises((ValueError, RuntimeError), match='deadline'):
        audit_result(replace(r, tasks=tasks), w, e, (1, 1, 1))
    with pytest.raises((ValueError, RuntimeError), match='arrival|once'):
        audit_result(replace(r, events=r.events + (r.events[0],)), w, e, (1, 1, 1))


def test_physical_object_state_fingerprint_covers_arrays_rng_alias_and_unknown():
    from experiments.e1.counterfactual_audit import state_fingerprint
    class Model:
        def __init__(self):
            self.values = np.array([1., 2.]); self.same = self.values
            self.rng = np.random.default_rng(123); self.infinity = float('inf')
    a, b = Model(), Model()
    assert state_fingerprint(a) == state_fingerprint(b)
    b.values[0] = 2
    assert state_fingerprint(a) != state_fingerprint(b)
    b = Model(); b.rng.random()
    assert state_fingerprint(a) != state_fingerprint(b)
    with pytest.raises(TypeError, match='Unsupported'):
        state_fingerprint(iter([1]))


def test_real_upstream_power_state_pair_and_persistent_idle_are_auditable():
    from pathlib import Path
    from experiments.e1.counterfactual import InterventionTarget, run_paired_case
    from experiments.e1.counterfactual_audit import audit_pair
    from experiments.e1.power import UpstreamPowerModel
    config = dict(location='US-CAL-CISO', total_cores=50000, total_gpus=1000,
                  total_mem=80000, hvac_setpoint_c=22)
    model_path = Path(__file__).resolve().parents[2] / 'configs/dcs/dc_config.json'
    w, e = cohort([task('a', duration=16, cpu=500, gpu=10, mem=800, sla=3)])
    paired = run_paired_case(w, e, (50000, 1000, 80000),
        lambda: UpstreamPowerModel(config, model_path), InterventionTarget('real_power', START, 'a'),
        minimum_common_end=e.end)
    assert audit_pair(paired, w, e, (50000, 1000, 80000))['passed']
    assert paired.prefixes['force_execute']['components']['physical_model'] == paired.prefixes['force_defer_once']['components']['physical_model']
    assert len(paired.results['force_execute'].steps) == 8


def test_resource_hours_follow_actual_timedelta_ns_near_float_grid_boundary():
    import math
    from experiments.e1.counterfactual import InterventionTarget, run_paired_case
    from experiments.e1.counterfactual_audit import audit_pair
    duration = math.nextafter(15., math.inf)
    w, e = cohort([task('a', duration=duration, cpu=.5, gpu=.1, mem=.5, sla=3)])
    # pandas duration truncation maps this input to exactly one occupied step;
    # using math.ceil(duration/15) would incorrectly infer two steps.
    paired = run_paired_case(w, e, (1, 1, 1), KnownPower, InterventionTarget('boundary', START, 'a'))
    report = audit_pair(paired, w, e, (1, 1, 1))
    assert report['branches']['force_execute']['resource_hours'] == [.125, .025, .125]
