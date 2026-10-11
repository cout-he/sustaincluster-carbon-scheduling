"""Predeclared single Execute→Defer action; excluded from fair baselines."""
import argparse
import math
import shutil
import sys
import time
from pathlib import Path

import pandas as pd

from .cars import CarsPolicy
from .cars_experiment import TimingPolicy, _input_hashes, _snapshot_sources
from .contention import read_json
from .contracts import content_hash, file_hash, utc_time, write_json
from .engine import LocalExecutor
from .feasibility_experiment import _verify_level, _write_policy
from .policies import PolicyDecision
from .power import UpstreamPowerModel
from .run import _fresh_run, _git, dependencies
from .temporal import experiment_sources, write_csv

VERSION='e1-4a-single-action-intervention-v1'
APPROVED_WORKLOAD_HASH='30cf0e805f1abbb38ba08ace92de84888bb5778e360459b7656bcf8367eb1e15'


def validate_intervention_cohort(workload,design):
    if (design.get('factor')!=1.5 or workload.sha256!=APPROVED_WORKLOAD_HASH
            or len(workload.tasks)!=14914):
        raise ValueError('Intervention requires the approved 1.5x frozen cohort')


class SingleDeferFull(CarsPolicy):
    """Change one eligible candidate action; all later decisions use original Full."""
    def __init__(self, parameters, start, capacity, task_id, at):
        super().__init__('cars_full',parameters,start,capacity)
        self.target_id,self.target_time=task_id,utc_time(at,True)
        self.interventions=[]

    def decide(self,spec,context):
        original=super().decide(spec,context)
        if spec.task_id!=self.target_id or context.now!=self.target_time:
            return original
        if self.interventions or original.defer:
            raise ValueError('Intervention requires exactly one original Execute action')
        record=dict(time=context.now.isoformat(),task_id=spec.task_id,
            original_defer=original.defer,original_reason=original.reason,
            changed_defer=True,changed_reason='intervention_defer_once')
        self.interventions.append(record)
        self.diagnostics[-1].update(intervention=record,defer=True,reason='intervention_defer_once')
        # Guard counters retain what original Full computed. Count the actual
        # action separately so it cannot be mistaken for an extra guard benefit.
        self.reason_counts[original.reason]-=1
        if self.reason_counts[original.reason]==0:del self.reason_counts[original.reason]
        self.reason_counts['intervention_defer_once']+=1
        return PolicyDecision(True,'intervention_defer_once')

    def assert_applied_once(self):
        if len(self.interventions)!=1:
            raise RuntimeError('Predeclared intervention must be applied exactly once')


def _execute(out,reference):
    started=time.perf_counter()
    item=_verify_level(reference)
    cfg,w,e,params=(item[k] for k in ('cfg','workload','environment','parameters'))
    design=item['ablation_config']['intervention']
    validate_intervention_cohort(w,design)
    if read_json(Path(reference)/'reference.json')['input_hashes']['workload_snapshot']!=w.sha256:
        raise ValueError('Intervention cohort differs from reference')
    if design['task_id'] not in {t.task_id for t in w.tasks}:
        raise ValueError('Predeclared task missing from this frozen load')
    hashes=experiment_sources();_snapshot_sources(out,hashes)
    dc=cfg['datacenter'];capacity=dc['total_cores'],dc['total_gpus'],dc['total_mem']
    configs=read_json(Path(reference)/'cars_full/resolved_config.json')
    executions={};own={};policies={};timing={}
    for name in ('full_control','full_single_defer'):
        policy=(CarsPolicy('cars_full',params,w.start,capacity) if name=='full_control'
                else SingleDeferFull(params,w.start,capacity,design['task_id'],design['time']))
        timed=TimingPolicy(policy)
        ex=LocalExecutor(w,e,capacity,UpstreamPowerModel(dc,item['model_path']),timed,drain=True)
        tick=time.perf_counter();own[name]=ex.run(stop_at_drain=True)
        timing[name]=dict(own_run_seconds=time.perf_counter()-tick,scheduler=timed.timings)
        executions[name],policies[name]=ex,policy
        if isinstance(policy,SingleDeferFull):policy.assert_applied_once()
    reference_end=pd.Timestamp(read_json(Path(reference)/'fairness.json')['common_end'])
    common=max(reference_end,*(ex.observed_end for ex in executions.values()))
    results={}
    for name,ex in executions.items():
        results[name]=ex.extend_idle(common) if ex.drained_at is not None else own[name]
        _write_policy(out/name,results[name],own[name],ex,configs,w,common,policies[name].diagnostics)
    # The unchanged control must reproduce the original complete task trajectories.
    for name in ('ledger.jsonl','events.jsonl','policy_ledger.jsonl','policy_events.jsonl','diagnostics.jsonl'):
        if file_hash(out/'full_control'/name)!=file_hash(Path(reference)/'cars_full'/name):
            raise RuntimeError('Unchanged Full control differs: '+name)
    b={r['task_id']:r for r in results['full_control'].tasks}
    changed=[]
    for row in results['full_single_defer'].tasks:
        base=b[row['task_id']]
        if any(row[k]!=base[k] for k in ('start_time','finish_time','deadline_outcome')):
            changed.append(dict(task_id=row['task_id'],deadline=row['sla_deadline'],
                baseline_start=base['start_time'],intervened_start=row['start_time'],
                baseline_finish=base['finish_time'],intervened_finish=row['finish_time'],
                baseline_deadline_outcome=base['deadline_outcome'],intervened_deadline_outcome=row['deadline_outcome']))
    comparison=[dict(trajectory=name,**r.summary) for name,r in results.items()]
    write_csv(out/'comparison.csv',comparison)
    write_csv(out/'changed_tasks.csv',changed)
    states={name:dict(completed=r.summary['completed'],resource_hours=[
        math.fsum(s[k] for s in r.steps)*.25 for k in ('used_cores','used_gpus','used_mem')])
        for name,r in results.items()}
    write_json(out/'intervention.json',dict(version=VERSION,additional_intervention=True,
        included_in_fair_baselines=False,design=design,actions=policies['full_single_defer'].interventions,
        changed_task_count=len(changed),common_end=common.isoformat(),states=states,
        baseline_source_provenance_sha256=content_hash(item['provenance']),
        note='One action intervention, then original Full. Guard computations retained; only one action changed.'))
    for name,value in (('resolved_config.json',configs),('parameters.json',params),('selection.json',item['selection'])):
        write_json(out/name,value)
    w.save(out/'workload.json');e.save(out/'environment.json');shutil.copyfile(item['model_path'],out/'dc_config.json')
    write_json(out/'runtime.json',dict(total_seconds=time.perf_counter()-started,policies=timing))
    if hashes!=experiment_sources():raise RuntimeError('Sources changed during intervention')
    scientific={p.relative_to(out).as_posix():file_hash(p) for p in out.rglob('*')
                if p.is_file() and 'source' not in p.relative_to(out).parts
                and p.name not in ('run_status.json','provenance.json','runtime.json')}
    write_json(out/'provenance.json',dict(version=VERSION,git=dict(head=_git('rev-parse','HEAD'),
        branch=_git('branch','--show-current')),command=sys.argv,python=sys.version,dependencies=dependencies(),
        sources=hashes,source_sha256=content_hash(hashes),
        input_hashes=_input_hashes(configs,w,e,out/'dc_config.json',params,item['selection']),
        baseline_source_provenance_sha256=content_hash(item['provenance']),output_hashes=scientific,
        runtime_hashes={'runtime.json':file_hash(out/'runtime.json')}))
    return comparison


def run_intervention(reference,output):
    return _fresh_run(output,lambda out:_execute(out,reference))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();run_intervention(args.reference,args.output)
