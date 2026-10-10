import json

import pytest

from experiments.e1.contracts import content_hash, file_hash, write_json


def campaign(tmp_path,monkeypatch):
    from experiments.e1 import contention as c
    root=tmp_path/'original';root.mkdir()
    child=root/'load_1p00';child.mkdir()
    write_json(child/'selection.json',None);write_json(child/'provenance.json',{})
    write_json(root/'run_status.json',{'state':'complete'})
    config=dict(factors=[1],seed=123,reference_config={},threshold_g_per_kwh=150)
    write_json(root/'config.json',config)
    names=['config.json','comparison.csv','diagnostics.json','input_identities.json','baseline_regression.json','load_1p00_replication.json']
    for name in names[1:]:write_json(root/name,{})
    monkeypatch.setattr(c,'experiment_sources',lambda:{})
    monkeypatch.setattr(c,'dependencies',lambda:{})
    monkeypatch.setattr(c,'verify_archive',lambda *args:(None,None))
    prov=dict(version=c.VERSION,sources={},source_sha256=content_hash({}),python=c.sys.version,dependencies={},
              output_hashes={n:file_hash(root/n) for n in names},child_output_manifests={'load_1p00':{}})
    write_json(root/'provenance.json',prov)
    return c,root,prov


@pytest.mark.parametrize('damage',['missing_output','version','source_aggregate','missing_child'])
def test_campaign_manifest_rejects_incomplete_identity_before_running(tmp_path,monkeypatch,damage):
    c,root,prov=campaign(tmp_path,monkeypatch)
    if damage=='missing_output':prov['output_hashes'].pop('diagnostics.json')
    elif damage=='version':prov['version']='wrong'
    elif damage=='source_aggregate':prov['source_sha256']='wrong'
    else:prov['child_output_manifests'].clear()
    write_json(root/'provenance.json',prov)
    calls=[]
    monkeypatch.setattr(c,'run_levels',lambda *args,**kwargs:calls.append(True))
    with pytest.raises(ValueError):c.replay_levels(root,tmp_path/'replay')
    assert not calls


def test_failed_numerical_replay_has_failed_status(tmp_path,monkeypatch):
    c,root,prov=campaign(tmp_path,monkeypatch)
    out=tmp_path/'replay'
    def divergent(*args,**kwargs):
        def action(folder):
            for name in prov['output_hashes']:write_json(folder/name,{'different':True})
            if kwargs.get('verify_replay'):kwargs['verify_replay'](folder)
        return c._fresh_run(out,action)
    monkeypatch.setattr(c,'run_levels',divergent)
    with pytest.raises(RuntimeError,match='replay numerical mismatch'):c.replay_levels(root,out)
    assert json.loads((out/'run_status.json').read_text())['state']=='failed'


def test_timeline_clips_unfinished_execution_at_observation_end():
    from experiments.e1.plot_contention import timeline_execution
    task=dict(state='running',start_time='2023-07-01T05:15:00Z',finish_time='2023-07-01T07:15:00Z')
    end,censored=timeline_execution(task,'2023-07-01T06:00:00Z')
    assert end.isoformat()=='2023-07-01T06:00:00+00:00' and censored
    task['state']='completed';task['finish_time']='2023-07-01T05:45:00Z'
    end,censored=timeline_execution(task,'2023-07-01T06:00:00Z')
    assert end.isoformat()=='2023-07-01T05:45:00+00:00' and not censored
