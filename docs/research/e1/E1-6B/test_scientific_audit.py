"""Hand-calculated audit contracts; no simulator, workload loader, PPO or Test."""
import copy
import csv
import importlib.util
from pathlib import Path

import pandas as pd
import pytest

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('e1_6b_scientific_audit',HERE/'scientific_audit.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)


def simple(tmp_path):
    start=pd.Timestamp('2023-07-08T00:00:00Z');arrival_end=start+audit.STEP;end=start+3*audit.STEP
    # A deliberately tight synthetic SLA exercises first miss before release.
    # This is not a change to the study's frozen 1.5 multiplier.
    task=dict(task_id='task',arrival_time=start.isoformat(),duration=15.1,sla_multiplier=.5,
              cores_req=1.,gpu_req=1.,mem_req=1.,dc_id=1)
    deadline=start+pd.Timedelta(minutes=7.55);finish=start+pd.Timedelta(minutes=15.1)
    ledger=[dict(**task,state='completed',start_time=start.isoformat(),finish_time=finish.isoformat(),
        completion_observed_at=(start+2*audit.STEP).isoformat(),sla_deadline=deadline.isoformat(),
        sla_met=False,deadline_outcome='missed',dest_dc_id=1,wait_intervals=0,
        transmission_delay_seconds=0.,transmission_energy_kwh=0.,transmission_carbon_kg=0.)]
    events=[dict(time=start.isoformat(),task_id='task',event=kind) for kind in ('arrived','enqueued','started')]
    events.append(dict(time=(start+2*audit.STEP).isoformat(),task_id='task',event='completed_resources_released'))
    decisions=[dict(time=start.isoformat(),task_id='task',event='started',decision_rank=0,
        can_fit_at_decision=True,available_before=dict(cores=2.,gpus=2.,mem=2.))]
    intervals=[];policy=[];samples={};actions=[]
    for i in range(3):
        time=start+i*audit.STEP;running=i<2;initial=i==0
        intervals.append(dict(interval_start=time.isoformat(),interval_end=(time+audit.STEP).isoformat(),
            interval_hours=.25,pending=0,running=int(running),ci_g_per_kwh=200.,temperature_c=20.,wet_bulb_c=18.,
            it_power_kw=2.,hvac_power_kw=1.,total_power_kw=3.,it_energy_kwh=.5,hvac_energy_kwh=.25,
            total_energy_kwh=.75,carbon_kg=.15,transmission_energy_kwh=0.,
            **{name:int(running) if name.startswith('used_') else (2-int(running)) if name.startswith('available_') else int(running)/2
                for resource in ('cores','gpus','mem') for name in ('used_'+resource,'available_'+resource,resource+'_fraction')}))
        policy.append(dict(time=time.isoformat(),decision_pending_ids=['task'] if initial else [],
            ordered_waiting_ids=['task'] if initial else [],started_ids=['task'] if initial else [],
            policy_deferred_ids=[],capacity_waiting_ids=[],pending_end_ids=[],running_ids=['task'] if running else []))
        samples[time]=dict(ci_g_per_kwh=200.,temperature_c=20.,wet_bulb_c=18.)
        if i<2:
            actions.append(dict(time=time.isoformat(),reward_event_time=(time+audit.STEP).isoformat(),
                pending=int(initial),running=int(i==1),ci_g_per_kwh=200.,action=0,
                reward=-.00015-(.01 if i==0 else 0.),started_ids=['task'] if initial else [],
                deferred_ids=[],capacity_waiting_ids=[],new_deadline_miss_ids=['task'] if i==0 else []))
    summary=dict(arrived=1,completed=1,pending=0,running=0,deadline_missed_completed=1,deadline_met=0,
        deadline_missed_unfinished=0,deadline_right_censored=0,intervals=3,observation_end=end.isoformat(),
        drained_at=(start+2*audit.STEP).isoformat(),it_energy_kwh=1.5,hvac_energy_kwh=.75,total_energy_kwh=2.25,
        carbon_kg=.45,carbon_unit='kgCO2e',energy_unit='kWh',power_unit='kW')
    common=dict(**summary,own_observation_end=(start+2*audit.STEP).isoformat(),drain_hours=.25,
        deadline_violation_rate_observed=1.,mean_jct_minutes_completed=15.1,mean_wait_minutes_started=0.,
        cores_hours=.5,gpus_hours=.5,mem_hours=.5,pending_mean=0.,pending_max=0,capacity_wait_intervals=0,
        policy_defer_intervals=0,**{name:1/3 if name.endswith('_mean') else .5 for resource in ('cores','gpus','mem')
            for name in (resource+'_fraction_mean',resource+'_fraction_max')})
    payload=dict(ledger=ledger,events=events,policy_events=decisions,policy_intervals=policy,
        policy_ledger=[dict(task_id='task',capacity_wait_intervals=0,policy_defer_intervals=0,observed_pending_minutes=0.)],
        intervals=intervals,summary=summary,policy_summary={},actions=actions)
    folder=tmp_path/'result';folder.mkdir()
    def save():
        for key in ('ledger','events','policy_events','policy_intervals','policy_ledger'):
            with (folder/(key+'.jsonl')).open('w',encoding='utf8') as stream:
                for row in payload[key]:stream.write(__import__('json').dumps(row)+'\n')
        for key in ('summary','policy_summary'):audit.write_json(folder/(key+'.json'),payload[key])
        for key in ('intervals','actions'):
            data=copy.deepcopy(payload[key])
            for row in data:
                for name,value in row.items():
                    if isinstance(value,list):row[name]=__import__('json').dumps(value)
            with (folder/(key+'.csv')).open('w',encoding='utf8',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    save()
    def run():return audit.audit_method({'task':task},samples,(2.,2.,2.),start,arrival_end,end,folder,common)
    return payload,common,save,run


def test_fractional_duration_holds_two_slots_and_idle_tail(tmp_path):
    _,_,_,run=simple(tmp_path)
    result=run()
    assert result['carbon_kg']==pytest.approx(.45)
    assert result['first_miss_penalties']==1 and result['grid_release_correct']


@pytest.mark.parametrize('bad',('duplicate','duration','release','deadline','miss','occupancy','capacity',
    'energy','carbon','ci','pending','order','firstmiss','repeatedmiss','reward','drain','units'))
def test_hand_calculated_counterexamples_are_rejected(tmp_path,bad):
    payload,common,save,run=simple(tmp_path)
    if bad=='duplicate':payload['ledger'].append(copy.deepcopy(payload['ledger'][0]))
    elif bad=='duration':payload['ledger'][0]['finish_time']='2023-07-08T00:15:00+00:00'
    elif bad=='release':payload['ledger'][0]['completion_observed_at']='2023-07-08T00:15:00+00:00'
    elif bad=='deadline':payload['ledger'][0]['sla_deadline']='2023-07-08T00:30:00+00:00'
    elif bad=='miss':payload['ledger'][0]['sla_met']=True
    elif bad=='occupancy':payload['intervals'][1]['used_gpus']=0.
    elif bad=='capacity':payload['policy_events'][0]['can_fit_at_decision']=False
    elif bad=='energy':payload['intervals'][0]['total_energy_kwh']=3.
    elif bad=='carbon':payload['intervals'][0]['carbon_kg']=150.
    elif bad=='ci':payload['intervals'][0]['ci_g_per_kwh']=999.
    elif bad=='pending':payload['policy_intervals'][1]['pending_end_ids']=['task']
    elif bad=='order':payload['policy_intervals'][0]['ordered_waiting_ids']=[]
    elif bad=='firstmiss':payload['actions'][0]['new_deadline_miss_ids']=[]
    elif bad=='repeatedmiss':payload['actions'][1]['new_deadline_miss_ids']=['task']
    elif bad=='reward':payload['actions'][1]['reward']=0.
    elif bad=='drain':common['own_observation_end']='2023-07-08T00:45:00+00:00'
    elif bad=='units':payload['summary']['carbon_unit']='gCO2e'
    save()
    with pytest.raises(ValueError):run()


def test_existing_audit_output_refused(tmp_path):
    with pytest.raises(FileExistsError):audit.main(['--output',str(tmp_path)])
