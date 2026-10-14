"""Score online estimates on ALL twelve previously selected Train interventions.

Actual full-system labels are read only after the causal candidate replay. They
never enter any policy, forecast, model fitting, threshold or observation.
"""
import argparse,json
from pathlib import Path
import pandas as pd
from experiments.e1.config import ROOT
from experiments.e1.contracts import content_hash,file_hash,write_json
from experiments.e1.carbon_gain_experiment import CONFIG,load_frozen,load_registered_bundle
from experiments.e1.carbon_gain_policy import CarbonGainEDF
from experiments.e1.engine import LocalExecutor
from experiments.e1.online_power_surrogate import PowerSurrogate
from experiments.e1.online_ci_forecast import extract_prehistory
from experiments.e1.run import _fresh_run
from experiments.e1.temporal import write_csv

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--freeze',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    cfg=json.loads(CONFIG.read_text());frozen=load_frozen(a.freeze);model=PowerSurrogate.from_dict(frozen['model'])
    selected=ROOT/'research_notes/E1/E1-7A/selection-001/E1-7A-selection-manifest.json'
    doc=json.loads(selected.read_text());assert content_hash(doc['payload'])==doc['sha256']
    cases=doc['payload']['C_cases'];assert len(cases)==12 and doc['payload']['split']=='train'
    def action(out):
        estimates=[]
        for label in ('1.5','2'):
            bundle=load_registered_bundle(cfg,'train',label)
            targets={(pd.Timestamp(c['time']),c['task_id']):c for c in cases if c['factor']==float(label)}
            prior=extract_prehistory(ROOT/'data/carbon_intensity/US-CAL-CISO/2023/US-CAL-CISO_2023_hourly.csv',bundle.workload.start,expected_sha256=bundle.environment.carbon_source_sha256)
            class Observer(CarbonGainEDF):
                def decide(self,spec,ctx):
                    if (ctx.now,spec.task_id) in targets:
                        c=targets[(ctx.now,spec.task_id)];g=self._estimate(spec);guard=self.guard.check(spec,ctx.now)
                        estimates.append(dict(case_id=c['case_id'],factor=bundle.factor,task_id=spec.task_id,time=ctx.now.isoformat(),
                            ci=ctx.ci_g_per_kwh,m0_high_ci=ctx.ci_g_per_kwh>161.075,guard_allowed=guard['allowed'],
                            buffer_steps=guard['buffer_steps'],**{k:v for k,v in g.items() if k!='elapsed_seconds'}))
                    return super().decide(spec,ctx)
            p=Observer('m2',bundle.workload.start,bundle.capacity,model,prehistory=prior);p.set_action(0)
            LocalExecutor(bundle.workload,bundle.environment,bundle.capacity,bundle.power_factory(),p,True).run(stop_at_drain=True)
        if len(estimates)!=12:raise RuntimeError('Missing registered candidate')
        # This join is explicitly offline evaluation, after all predictions.
        labels=ROOT/'research_notes/E1/E1-7A/tables/C-carbon-components.csv'
        actual=pd.read_csv(labels);rows=[]
        for estimate in estimates:
            r=actual[actual.case_id==estimate['case_id']].iloc[0]
            rows.append(dict(**estimate,actual_gain_it_kg=-r.delta_it_carbon_kg,
                actual_gain_hvac_kg=-r.delta_hvac_carbon_kg,actual_gain_total_kg=-r.delta_total_carbon_kg,
                it_error_kg=estimate['gain_it_kg']+r.delta_total_carbon_kg,
                total_error_kg=estimate['gain_total_kg']+r.delta_total_carbon_kg,
                label_scope='full cohort, force once then EDF; prediction target shift ideal15min/known running only'))
        write_csv(out/'registered-predictions.csv',rows)
        write_json(out/'provenance.json',dict(selection_sha256=file_hash(selected),actual_label_sha256=file_hash(labels),
            frozen_sha256=file_hash(a.freeze),script_sha256=file_hash(Path(__file__)),all_preselected_12=True,
            no_online_label_access=True,test_access=False))
    _fresh_run(a.output,action)
if __name__=='__main__':main()
