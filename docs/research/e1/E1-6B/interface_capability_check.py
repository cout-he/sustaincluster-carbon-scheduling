"""Synthetic one-step expression test, not a performance experiment."""
from dataclasses import replace
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[2]))
from tests.e1.helpers import START, KnownPower, inputs, task
from experiments.e1.engine import LocalExecutor
from experiments.e1.cars import make_cars_policy
from experiments.e1.policies import EDFBackfill
from experiments.e1.adaptive_risk import AdaptiveDelayPolicy

tasks=[task('t45',duration=90,cpu=.1,mem=.1),task('t60',duration=120,cpu=.1,mem=.1)]
workload,environment=inputs(tasks,15)
environment=replace(environment,samples=tuple(replace(s,ci_g_per_kwh=200) for s in environment.samples))
params={'threshold_g_per_kwh':161.075}
policies={'EDF':EDFBackfill(),
          'EDF+Carbon':make_cars_policy('edf_carbon_threshold',params,START,(1,1,1)),
          'RiskOnly':make_cars_policy('cars_risk_only',params,START,(1,1,1)),
          'Action1':AdaptiveDelayPolicy(params,START,(1,1,1),risk_filter=False)}
policies['Action1'].set_action(1)
results={}
for name,policy in policies.items():
    live=LocalExecutor(workload,environment,(1,1,1),KnownPower(),policy=policy)
    live.start();live.advance();live.check_ownership();live.pool.inventory()
    s=live.policy_steps[-1]
    results[name]={k:s[k] for k in ('started_ids','policy_deferred_ids','capacity_waiting_ids')}
assert results['EDF']['started_ids']==['t45','t60']
assert results['EDF+Carbon']['policy_deferred_ids']==['t45','t60']
assert results['RiskOnly']['policy_deferred_ids']==['t45','t60']
assert results['Action1']['started_ids']==['t45'] and results['Action1']['policy_deferred_ids']==['t60']
assert all(not r['capacity_waiting_ids'] for r in results.values())
receipt=dict(passed=True,synthetic=True,steps=1,real_data_or_test_access=False,
    grid_solo_slack_minutes={'t45':45,'t60':60},ci_g_per_kwh=200,threshold=161.075,
    results=results,interpretation='new action expresses a different task decision set; no isolated performance-effect claim')
out=HERE/'interface-capability-001';out.mkdir(exist_ok=False)
with (out/'receipt.json').open('x',encoding='utf8') as f:json.dump(receipt,f,indent=2)
print(json.dumps(receipt))
