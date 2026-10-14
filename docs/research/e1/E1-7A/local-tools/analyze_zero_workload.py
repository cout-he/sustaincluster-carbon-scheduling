"""Registered supplementary physical reference; no policy comparison or Test."""
import csv
from dataclasses import replace
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.e1.carbon_opportunity import energy_decomposition
from experiments.e1.contracts import content_hash, file_hash, write_json
from experiments.e1.diagnostic_experiment import load_selection_manifest, load_cohort
from experiments.e1.engine import LocalExecutor
from experiments.e1.policies import EDFBackfill
from experiments.e1.run import _fresh_run, _write_result, source_hashes


def main():
    root=Path(__file__).resolve().parent
    payload=load_selection_manifest(root/'selection-001/E1-7A-selection-manifest.json')
    sources=source_hashes()
    def action(out):
        bundle,_=load_cohort(payload['config'],'1')
        work=replace(bundle.workload,tasks=())
        executor=LocalExecutor(work,bundle.environment,bundle.capacity,bundle.power_factory(),EDFBackfill(),drain=True)
        executor.run(stop_at_drain=True)
        end=pd.Timestamp(payload['cohorts'][0]['common_end'])
        result=executor.extend_idle(end)
        assert not result.tasks
        assert len(result.steps)==228
        assert all(step['used_'+key]==0 for step in result.steps for key in ('cores','gpus','mem'))
        _write_result(out,result)
        decomposition=energy_decomposition(result.steps)
        rows=[]
        tick=pd.Timedelta(minutes=15).value
        for label in ('1','1.5','2'):
            actual,_=load_cohort(payload['config'],label)
            variables={
                'duration_minutes':[spec.duration for spec in actual.workload.tasks],
                'raw_solo_slack_minutes':[(spec.sla_deadline-spec.arrival_time).total_seconds()/60-spec.duration for spec in actual.workload.tasks],
                'grid_solo_slack_minutes':[((spec.sla_deadline.value-pd.Timedelta(minutes=spec.duration).value-actual.workload.start.value)//tick*tick+actual.workload.start.value-spec.arrival_time.value)/60e9 for spec in actual.workload.tasks],
                'occupied_minutes':[((pd.Timedelta(minutes=spec.duration).value+tick-1)//tick)*15 for spec in actual.workload.tasks]}
            for name,values in variables.items():
                rows.append(dict(factor=actual.factor,quantity=name,n=len(values),
                    mean=float(np.mean(values)),minimum=float(np.min(values)),
                    p25=float(np.quantile(values,.25)),p50=float(np.quantile(values,.5)),
                    p75=float(np.quantile(values,.75)),p90=float(np.quantile(values,.9)),
                    p99=float(np.quantile(values,.99)),maximum=float(np.max(values))))
        with (out/'B-duration-slack.csv').open('x',encoding='utf8',newline='') as stream:
            writer=csv.DictWriter(stream,list(rows[0]),lineterminator='\n')
            writer.writeheader();writer.writerows(rows)
        for name,digest in sources.items():
            assert file_hash(Path(__file__).resolve().parents[3]/name)==digest
        receipt=dict(passed=True,split='train',test_access=False,
            zero_workload_reference_only=True,is_equal_work_strategy_comparison=False,
            original_input_identity=bundle.identity,zero_workload_sha256=work.sha256,
            common_end=end.isoformat(),sources=sources,source_sha256=content_hash(sources),
            script_sha256=file_hash(Path(__file__)),command=sys.argv,python=sys.version,
            physical_summary=decomposition,
            output_hashes={p.name:file_hash(p) for p in out.iterdir() if p.is_file() and p.name!='run_status.json'})
        write_json(out/'receipt.json',receipt)
        print(json.dumps(dict(passed=True,physical_summary=decomposition)))
    _fresh_run(root/'B-zero-workload-001',action)


if __name__=='__main__':
    main()
