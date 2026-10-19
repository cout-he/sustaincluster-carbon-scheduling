"""A1 aggregate comparison contracts; synthetic dictionaries only."""
import json
import pytest

from experiments.object_model_a1.tables import (
    generate, matched_threshold_pairs, matched_type_pairs, reduction_percent, summarize_pairs,
)


def scenario(policy, exposure, misses=0, n=2, name='s1'):
    row = dict(candidate_id='OM', original_scenario_id=name, policy=policy,
        scenario_id='a1-OM-'+name, original_scenario_identity_sha256='same-scenario',
        source_fragment_id=name, original_workload_sha256='original', mapped_workload_sha256='mapped',
        original_environment_sha256='same-env', candidate_config_sha256='config',
        mapping_version='observed_usage_x1_pressure_v1', resource_pressure_factor='1.0',
        sla_multiplier='1.5', arrived=str(n), completed=str(n), deadline_misses=str(misses),
        dmr=str(misses/n), mean_jct_minutes='30', capacity_waited_tasks='0', policy_deferred_tasks='0')
    for resource in ('cpu', 'gpu', 'mem'):
        row[resource+'_resource_hours'] = '2'
        row[resource+'_ci_exposure_proxy'] = str(exposure)
        row[resource+'_reserved_ci_exposure_proxy'] = str(exposure+10)
    return row


def test_partial_results_rejected_before_csv_access_or_output_creation(tmp_path):
    source = tmp_path/'input'; source.mkdir()
    (source/'metadata.json').write_text(json.dumps({'complete': False}))
    with pytest.raises(ValueError, match='partial'):
        generate(source, tmp_path/'output', tmp_path/'report.md')
    assert not (tmp_path/'output').exists()
    assert not (tmp_path/'report.md').exists()


def test_matched_exposure_is_signed_and_mismatched_environment_fails():
    base=scenario('fifo',100)
    comparison=scenario('carbon_threshold_fifo',110,misses=1)
    pairs=matched_threshold_pairs([base,comparison])
    assert pairs[0]['gpu_ci_exposure_reduction_percent']==pytest.approx(-10)
    assert pairs[0]['dmr_change_percentage_points']==50
    comparison['original_environment_sha256']='different'
    with pytest.raises(ValueError, match='identity'):
        matched_threshold_pairs([base,comparison])


def test_aggregate_ratio_uses_sum_of_exposures_and_keeps_denominator():
    pairs=matched_threshold_pairs([scenario('fifo',100),scenario('carbon_threshold_fifo',90),
        scenario('fifo',900,n=8,name='s2'),scenario('carbon_threshold_fifo',450,n=8,name='s2')])
    summary=summarize_pairs(pairs)[0]
    assert summary['tasks']==10
    assert summary['gpu_ci_exposure_reduction_percent']==pytest.approx(46)
    assert summary['gpu_fifo_mean_execution_ci_g_per_kwh']==250


def test_resource_work_mismatch_and_zero_denominator_not_hidden():
    base=scenario('fifo',100); comparison=scenario('carbon_threshold_fifo',90)
    comparison['gpu_resource_hours']='3'
    with pytest.raises(ValueError, match='resource-hours'):
        matched_threshold_pairs([base,comparison])
    assert reduction_percent(0,0) is None


def test_types_use_runtime_weighting_and_reject_changed_duration():
    base=dict(candidate_id='OM',mapping_version='observed_usage_x1_pressure_v1',
        resource_pressure_factor='1',sla_multiplier='1.5',policy='fifo',task_type='gpu_positive',
        duration_group='lt60min',tasks='2',duration_minutes_mean='30',
        execution_ci_integral_g_per_kwh_hours_mean='100',dmr='0',capacity_waited_tasks='0',
        policy_deferred_tasks='0',observed_wait_minutes_mean='0')
    comparison=dict(base,policy='carbon_threshold_fifo',execution_ci_integral_g_per_kwh_hours_mean='90')
    pair=matched_type_pairs([base,comparison])[0]
    assert pair['original_runtime_hours']==1
    assert pair['fifo_duration_weighted_execution_ci_g_per_kwh']==200
    assert pair['duration_weighted_ci_reduction_percent']==pytest.approx(10)
    comparison['duration_minutes_mean']='35'
    with pytest.raises(ValueError,match='duration'):
        matched_type_pairs([base,comparison])
