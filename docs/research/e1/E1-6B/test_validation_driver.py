"""Small archival-driver contract tests; no real Validation execution."""
import csv
import importlib.util
import json
from pathlib import Path
import re

import pytest

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('e1_6b_validation_driver',HERE/'run_validation_campaign.py')
driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)


def fixture_output(tmp_path):
    record=dict(tasks=2,factor=1.,workload_sha256='work',environment_sha256='env',capacity=[1,1,1])
    fairness=dict(split='validation',formal_test_evaluated=False,policy_future_ci_access=False,
        all_tasks_completed=True,equal_resource_hours=True,equal_workload_carbon_comparison_valid=True,
        idle_included=True,development_validation=True,arrived=2,workload_sha256='work',
        environment_sha256='env',capacity=[1,1,1],common_end='2020-07-10T12:45:00+00:00')
    rows=[dict(policy=name,split='validation',phase='common',factor=1,arrived=2,completed=2,
        pending=0,running=0,deadline_right_censored=0,termination='all_completed',
        observation_end=fairness['common_end'],deadline_missed_completed=1,deadline_missed_unfinished=0,
        deadline_violation_rate_observed=.5,it_energy_kwh=3.,hvac_energy_kwh=2.,total_energy_kwh=5.,
        cores_hours=1.,gpus_hours=1.,mem_hours=1.) for name in driver.expected_methods()]
    folder=tmp_path/'result';folder.mkdir()
    def save():
        driver.write_json(folder/'fairness.json',fairness)
        driver.write_json(folder/'scientific_hashes.json',{name:'hash' for name in driver.expected_methods()})
        driver.write_json(folder/'source-freeze.json',{})
        with (folder/'common_summary.csv').open('w',encoding='utf8',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        driver.write_json(folder/'output_hashes.json',{p.name:driver.file_hash(p) for p in folder.iterdir()
            if p.name!='output_hashes.json'})
    save()
    return folder,record,fairness,rows,save


def test_complete_nineteen_method_contract(tmp_path):
    folder,record,_,_,_=fixture_output(tmp_path)
    report=driver.validate_output(folder,record)
    assert report['methods']==19 and report['completed_tasks']==2 and report['idle_included']


@pytest.mark.parametrize('bad',('test','idle','incomplete','energy','misses','methods','resources'))
def test_scientific_gate_rejects_incomparable_outputs(tmp_path,bad):
    folder,record,fairness,rows,save=fixture_output(tmp_path)
    if bad=='test':fairness['split']='test'
    elif bad=='idle':fairness['idle_included']=False
    elif bad=='incomplete':rows[0]['completed']=1
    elif bad=='energy':rows[0]['total_energy_kwh']=100
    elif bad=='misses':rows[0]['deadline_violation_rate_observed']=0
    elif bad=='methods':rows.pop()
    elif bad=='resources':rows[0]['gpus_hours']=2
    save()
    with pytest.raises(ValueError):driver.validate_output(folder,record)


def test_hash_tamper_is_rejected(tmp_path):
    folder,record,_,_,_=fixture_output(tmp_path)
    (folder/'fairness.json').write_text('{}')
    with pytest.raises(ValueError,match='hash mismatch'):driver.validate_output(folder,record)


def test_formal_gate_is_required(tmp_path):
    driver.write_json(tmp_path/'campaign-receipt.json',dict(passed=False,total_actual_steps=0))
    with pytest.raises(ValueError,match='Train gate'):driver.candidate_records(tmp_path,'frozen')


def test_exact_replay_flag_required(tmp_path):
    folder,record,_,_,_=fixture_output(tmp_path)
    driver.write_json(folder/'replay-verification.json',dict(scientific_outputs_exact=False))
    with pytest.raises(ValueError,match='not exact'):driver.validate_output(folder,record,replay=True)


def test_cli_refuses_existing_output_before_data_reads(tmp_path):
    with pytest.raises(FileExistsError):driver.main(['--output',str(tmp_path)])


def test_method_matrix_and_roles_are_fixed():
    assert len(set(driver.expected_methods()))==19
    assert [(case[0],case[1]) for case in driver.CASES]==[('1',1.),('1.5',1.5),('2',2.)]
    assert all('test' not in case[2].lower() for case in driver.CASES)


def test_registered_hashes_have_exact_sha256_shape():
    for case in driver.CASES:
        assert re.fullmatch(r'[0-9a-f]{64}',case[3]),case[0]
    for seed,digest in driver.LEGACY_SHA.items():
        assert re.fullmatch(r'[0-9a-f]{64}',digest),seed


def test_registered_manifests_and_legacy_checkpoints_match_existing_archives():
    # Read metadata and model bytes only; no task/environment or checkpoint loading.
    for label,_,relative,digest in driver.CASES:
        assert driver.file_hash(driver.ROOT/relative/'bundle.json')==digest,label
    for seed,digest in driver.LEGACY_SHA.items():
        folder=driver.ROOT/f'research_notes/E1/E1-5B/train-seed-{seed}-001'
        assert driver.file_hash(folder/'model.zip')==digest,seed
        summary=driver.read(folder/'training_summary.json')
        assert summary['checkpoint_sha256']==digest,seed
