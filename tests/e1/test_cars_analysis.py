"""Analysis preserves all seven methods and clips unobserved execution."""
import pandas as pd
import pytest
from .helpers import START


def test_observed_timeline_clips_unfinished_predicted_finish():
    from experiments.e1.analyze_cars import timeline_record
    end=START+pd.Timedelta(minutes=30)
    row=dict(task_id='a',arrival_time=START.isoformat(),sla_deadline=end.isoformat(),
             start_time=START.isoformat(),finish_time=(end+pd.Timedelta(minutes=60)).isoformat(),
             state='running',deadline_outcome='missed_unfinished')
    item=timeline_record('cars_full',row,end)
    assert item['observed_execution_end']==end.isoformat() and item['censored']


def test_summary_keeps_all_methods_and_completed_denominator():
    from experiments.e1.analyze_cars import summarize_rows
    from experiments.e1.cars import CARS_POLICY_NAMES
    rows=[dict(policy=p,phase='common',arrived=10,completed=10,
        deadline_missed_completed=i,deadline_missed_unfinished=0,carbon_kg=100+i)
        for i,p in enumerate(CARS_POLICY_NAMES)]
    result=summarize_rows(1.5,rows)
    assert len(result)==7 and [r['policy'] for r in result]==list(CARS_POLICY_NAMES)
    assert result[-1]['deadline_miss_rate']==.6
    assert result[-1]['carbon_reduction_vs_fifo_pct']==pytest.approx(-6)


def test_analysis_reads_driver_diagnostics_instead_of_silently_skipping(tmp_path,monkeypatch):
    import json
    from experiments.e1 import analyze_cars as module
    from experiments.e1.cars import CARS_POLICY_NAMES
    from experiments.e1.temporal import write_csv
    campaign=tmp_path/'campaign';level=campaign/'load_1p50';level.mkdir(parents=True)
    (level/'fairness.json').write_text(json.dumps({'common_end':START.isoformat()}))
    data=[]
    for p in CARS_POLICY_NAMES:
        folder=level/p;folder.mkdir()
        (folder/'ledger.jsonl').write_text(json.dumps({'task_id':'a','deadline_outcome':'met'})+'\n')
        (folder/'policy_events.jsonl').write_text('')
        data.append(dict(policy=p,phase='common',arrived=1,completed=1,
            deadline_missed_completed=0,deadline_missed_unfinished=0,carbon_kg=100))
    (level/'cars_full/diagnostics.jsonl').write_text(json.dumps({'reason':'known_deadline_risk',
        'planner':{'status':'risk','missed_now':['shared'],'missed_defer':['shared','new']}})+'\n')
    write_csv(level/'comparison.csv',data)
    monkeypatch.setattr(module,'plot',lambda *args:None)
    output=tmp_path/'analysis';module.analyze(campaign,output)
    mechanism=json.loads((output/'mechanisms.json').read_text())['1.5']['reasons']['cars_full']
    assert mechanism['reason_counts']=={'known_deadline_risk':1}
    assert mechanism['shared_miss_checks']==mechanism['additional_miss_checks']==1
