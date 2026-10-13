"""Synthetic summary fixtures verify analysis gates; never scientific runs."""
import csv
import hashlib
import json
from pathlib import Path

import pytest

SEEDS=(55101,55102,55103)
GROUPS=('plain','state','shield','combined')
BASELINES=('edf_backfill','edf_carbon_threshold','cars_risk_only','task_risk_full')


def digest(data):return hashlib.sha256(data).hexdigest()


def js(path,data):path.write_text(json.dumps(data,sort_keys=True),encoding='utf8')


def csv_out(path,rows):
    with path.open('w',encoding='utf8',newline='') as stream:
        writer=csv.DictWriter(stream,list(rows[0]));writer.writeheader();writer.writerows(rows)


def fixture(root):
    training=[];checkpoints={}
    config=dict(ppo=dict(gamma=.99,n_steps=256,n_epochs=10),policy_kwargs=dict(net_arch=[64,64]),
                reward=dict(alpha=1.,beta=1.,gamma=10.),normalization=dict(slack_minutes=120),
                risk_parameters=dict(a_u=1,a_q=1,a_h=1),threshold_g_per_kwh=161.075)
    for group in GROUPS:
        for seed in SEEDS:
            folder=root/(group+'_'+str(seed));folder.mkdir(parents=True);training.append(folder)
            checkpoint=digest((group+str(seed)).encode());checkpoints[(group,seed)]=checkpoint
            js(folder/'training_config.json',dict(config,group=group,seed=seed))
            updates=[{'environment_steps':min(20480,(i+1)*256),'train/n_updates':(i+1)*10,
                      'train/loss':1/(i+1)} for i in range(80)]
            csv_out(folder/'updates.csv',updates)
            episodes=[dict(episode=i,ending_step=(i+1)*10240,factor=factor,length=10240,
                episode_return=-10-i,terminated=True,truncated=False,carbon_kg=100+i,
                wait_task_minutes=10, new_deadline_misses=1,carbon_cost=.1,wait_cost=.1,
                deadline_cost=.01,terminal_cost=0) for i,factor in enumerate((1.5,2.))]
            csv_out(folder/'episodes.csv',episodes)
            files={p.name:digest(p.read_bytes()) for p in folder.iterdir()}
            summary=dict(schema='e1-6b-adaptive-ppo-v1',group=group,seed=seed,num_timesteps=20480,
                observation_size=83,action_size=4,parameter_count=19397,elapsed_seconds=50.,
                parameter_sha256_before=digest(('initial'+str(seed)).encode()),
                parameter_sha256_after=digest(('trained'+group+str(seed)).encode()),
                checkpoint_sha256=checkpoint,optimizer_state_entries=12,deterministic_roundtrip=True,
                source_sha256=digest(b'source'),completed_episodes=2,current_partial_episode_steps=0,
                budget_ends_mid_episode=False, action_counts=[5120]*4,
                scene_steps={'1.5':10240,'2.0':10240},scene_episodes={'1.5':1,'2.0':1},
                training_identities=[dict(split='train',factor=1.5),dict(split='train',factor=2.)],files=files)
            js(folder/'training_summary.json',summary)
    folders=[]
    for factor in (1.,1.5,2.):
        folder=root/('validation-'+str(factor));folder.mkdir();folders.append(folder)
        methods=[(name,'heuristic',None,base) for name,base in zip(BASELINES,(10,12,15,16))]
        methods += [(group+'_'+str(seed),group,seed,base+index)
                    for group,base in zip(GROUPS,(20,18,22,21)) for index,seed in enumerate(SEEDS)]
        methods += [('legacy_'+str(seed),'legacy_3action',seed,14+index) for index,seed in enumerate((1,2,3))]
        rows=[]
        for name,group,seed,base in methods:
            n=int(100*factor);miss=int(base*factor);energy=2000*factor+base
            rows.append(dict(policy=name,phase='common',split='validation',factor=factor,group=group,seed=seed,
                arrived=n,completed=n,pending=0,running=0,deadline_missed_completed=miss,
                deadline_missed_unfinished=0,deadline_right_censored=0,
                deadline_violation_rate_observed=miss/n,carbon_kg=1000*factor+base*.5,
                it_energy_kwh=energy*.8,hvac_energy_kwh=energy*.2,total_energy_kwh=energy,
                observation_end='2023-07-10T12:45:00+00:00',termination='all_completed',
                mean_jct_minutes_completed=60+base,mean_wait_minutes_started=base,
                pending_mean=base,pending_max=base*2,policy_defer_intervals=base*3,
                capacity_wait_intervals=base*2,cores_hours=12*factor,gpus_hours=3*factor,mem_hours=6*factor,
                cores_fraction_mean=.1,cores_fraction_max=.2,gpus_fraction_mean=.3,gpus_fraction_max=.9,
                mem_fraction_mean=.2,mem_fraction_max=.3,drain_hours=7.75,
                candidate_checks=100,solo_eligible=50,filter_veto=10 if group in ('shield','combined') else 0,
                deferred=40,run_seconds=2,inference_seconds=.1,
                checkpoint_sha256=checkpoints.get((group,seed),'external'),
                training_steps=20480 if group in GROUPS else '',
                action_0_count=25,action_1_count=25,action_2_count=25,action_3_count=25))
        csv_out(folder/'common_summary.csv',rows)
        fairness=dict(schema='e1-6b-common-evaluation-v1',split='validation',all_tasks_completed=True,
            equal_resource_hours=True,equal_workload_carbon_comparison_valid=True,idle_included=True,
            policy_future_ci_access=False,formal_test_evaluated=False,development_validation=True,
            common_end='2023-07-10T12:45:00+00:00',arrived=int(100*factor),capacity=[1,1,1],
            environment_sha256=digest(b'environment'),workload_sha256=digest(str(factor).encode()),
            same_frozen_input_sha256=digest(('input'+str(factor)).encode()))
        js(folder/'fairness.json',fairness)
        js(folder/'scientific_hashes.json',{r['policy']:digest(r['policy'].encode()) for r in rows})
        refresh(folder)
    return folders,training


def refresh(folder):
    js(folder/'output_hashes.json',{name:digest((folder/name).read_bytes())
                                  for name in ('common_summary.csv','fairness.json','scientific_hashes.json')})


def alter_csv(folder,change):
    with (folder/'common_summary.csv').open(encoding='utf8') as stream:rows=list(csv.DictReader(stream))
    change(rows);csv_out(folder/'common_summary.csv',rows);refresh(folder)


def test_complete_matrix_and_same_seed_paired_differences(tmp_path):
    from experiments.e1.adaptive_analysis import analyze_campaign
    groups,training=fixture(tmp_path/'inputs')
    result=analyze_campaign(groups,training,tmp_path/'analysis',render=False)
    assert len(result['methods'])==57 and len(result['paired'])==27
    assert len(result['groups'])==12 and len(result['training'])==12
    assert len(result['updates'])==960 and len(result['episodes'])==24
    row=next(r for r in result['paired'] if r['factor']==1 and r['seed']==55101 and r['group']=='state')
    assert row['delta_misses']==-2 and row['delta_dmr_percentage_points']==pytest.approx(-2)
    assert row['delta_carbon_kg']==pytest.approx(-1)
    assert result['manifest']['all_equal_workload_carbon_comparisons_valid'] is True
    assert (tmp_path/'analysis'/'all-methods.csv').is_file()


@pytest.mark.parametrize('mutation',[lambda r:r.pop(4),lambda r:r.append(dict(r[4]))])
def test_missing_or_duplicate_method_rejected(tmp_path,mutation):
    from experiments.e1.adaptive_analysis import analyze_campaign
    groups,training=fixture(tmp_path/'inputs');alter_csv(groups[0],mutation)
    with pytest.raises(ValueError):analyze_campaign(groups,training,tmp_path/'analysis',render=False)
    assert not (tmp_path/'analysis').exists()


def test_smoke_budget_cannot_be_presented_as_formal_training(tmp_path):
    from experiments.e1.adaptive_analysis import analyze_campaign
    groups,training=fixture(tmp_path/'inputs');p=training[0]/'training_summary.json'
    summary=json.loads(p.read_text());summary['num_timesteps']=1024;js(p,summary)
    with pytest.raises(ValueError,match='formal'):analyze_campaign(groups,training,tmp_path/'analysis',render=False)


def test_incomplete_drain_disallows_carbon_comparison(tmp_path):
    from experiments.e1.adaptive_analysis import analyze_campaign
    groups,training=fixture(tmp_path/'inputs');p=groups[0]/'fairness.json'
    fairness=json.loads(p.read_text());fairness['all_tasks_completed']=False;js(p,fairness);refresh(groups[0])
    with pytest.raises(ValueError,match='[Cc]omplete'):analyze_campaign(groups,training,tmp_path/'analysis',render=False)


@pytest.mark.parametrize('field,value',[('deadline_violation_rate_observed',.999),('checkpoint_sha256','wrong')])
def test_inconsistent_metrics_or_checkpoint_rejected(tmp_path,field,value):
    from experiments.e1.adaptive_analysis import analyze_campaign
    groups,training=fixture(tmp_path/'inputs');alter_csv(groups[0],lambda rows:rows[4].update({field:value}))
    with pytest.raises(ValueError):analyze_campaign(groups,training,tmp_path/'analysis',render=False)


def test_existing_output_is_never_overwritten(tmp_path):
    from experiments.e1.adaptive_analysis import analyze_campaign
    groups,training=fixture(tmp_path/'inputs');out=tmp_path/'analysis';out.mkdir();(out/'sentinel').write_text('retain')
    with pytest.raises(FileExistsError):analyze_campaign(groups,training,out,render=False)
    assert (out/'sentinel').read_text()=='retain'


def test_test_split_rejected_before_reporting(tmp_path):
    from experiments.e1.adaptive_analysis import analyze_campaign
    groups,training=fixture(tmp_path/'inputs');p=groups[0]/'fairness.json'
    fairness=json.loads(p.read_text());fairness['split']='test';js(p,fairness);refresh(groups[0])
    with pytest.raises(ValueError,match='Validation'):analyze_campaign(groups,training,tmp_path/'analysis',render=False)


def test_plot_limits_include_all_baselines_and_signed_paired_values(tmp_path):
    from experiments.e1.adaptive_analysis import analyze_campaign,plot_results
    groups,training=fixture(tmp_path/'inputs');result=analyze_campaign(groups,training,tmp_path/'analysis',render=False)
    result['methods'][0]['deadline_violation_rate_observed']=.8
    result['methods'][0]['carbon_kg']=100000
    audit=plot_results(result['methods'],result['paired'],tmp_path/'figures')
    assert audit['deadline-miss-rate'][0]==0 and audit['deadline-miss-rate'][1]>=80
    assert audit['carbon-emissions'][0]==0 and audit['carbon-emissions'][1]>=100000
    assert audit['paired-dmr'][0]<-2 and audit['paired-dmr'][1]>2
    for stem in ('deadline-miss-rate','carbon-emissions','paired-effects'):
        for extension in ('png','pdf','svg'):
            assert (tmp_path/'figures'/(stem+'.'+extension)).stat().st_size>100
