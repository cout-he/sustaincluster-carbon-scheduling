"""A0 read-only Train carbon opportunities and bounded rule comparisons.

No model is fitted. Future-CI opportunities are separate single-task diagnostics,
not feasible schedules. Full electrical comparisons use the existing executor
and unchanged JointPowerModel at fixed 22 C over the entire frozen window.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import subprocess
import time

import numpy as np
import pandas as pd


def integral_opportunity(ci, start, duration_minutes, slack_minutes):
    """Full rounded-running-slot integral, with legal grid-aligned start offsets.

    Existing executor holds each running task's entire demand for a 15-minute
    interval and releases at the first boundary at/after its true finish.
    CI is the frozen environment's left-boundary sample for that interval.
    """
    ci = np.asarray(ci, dtype=float)
    if (duration_minutes <= 0 or slack_minutes < 0 or start < 0
            or not np.isfinite(ci).all()):
        raise ValueError('Invalid CI, duration, slack or start')
    n = max(1, int(math.ceil(duration_minutes / 15 - 1e-12)))
    wait = int(math.floor(slack_minutes / 15 + 1e-12))
    observed = max(0, min(wait + 1, len(ci) - n - start + 1))
    if observed == 0:
        return dict(runtime_slots=n, legal_wait_slots=wait, window_complete=False,
                    candidates_observed=0, candidates_required=wait + 1)
    prefix = np.concatenate(([0.], np.cumsum(ci)))
    starts = np.arange(start, start + observed)
    integrals = (prefix[starts+n] - prefix[starts]) * .25
    best = int(np.argmin(integrals))
    immediate = float(integrals[0])
    saving = max(0., immediate - float(integrals[best]))
    return dict(runtime_slots=n, legal_wait_slots=wait,
                candidates_observed=observed, candidates_required=wait+1,
                window_complete=observed == wait + 1, best_offset_slots=best,
                immediate_ci_integral_g_per_kwh_h=immediate,
                best_ci_integral_g_per_kwh_h=float(integrals[best]),
                saving_ci_integral_g_per_kwh_h=saving,
                saving_fraction=saving / immediate if immediate else 0.,
                current_start_ci_g_per_kwh=float(ci[start]),
                best_start_ci_g_per_kwh=float(ci[start+best]))


def segmented_acf(segments, lag):
    """Pearson lag correlation formed only from pairs inside each fragment."""
    if lag < 1:
        raise ValueError('Lag must be positive')
    left = [s[:-lag] for s in segments if len(s) > lag]
    right = [s[lag:] for s in segments if len(s) > lag]
    if not left:
        return dict(lag_slots=lag, lag_minutes=lag*15, pairs=0, correlation=None)
    a, b = np.concatenate(left), np.concatenate(right)
    corr = float(np.corrcoef(a, b)[0,1]) if np.std(a) and np.std(b) else None
    return dict(lag_slots=lag, lag_minutes=lag*15, pairs=len(a), correlation=corr)


def stats(values, prefix=''):
    a = np.asarray(values, dtype=float)
    if not len(a):
        return {prefix+'count': 0}
    row = {prefix+'count': len(a), prefix+'mean': float(a.mean()),
           prefix+'std_population': float(a.std()), prefix+'min': float(a.min()),
           prefix+'max': float(a.max())}
    row.update({prefix+'p'+str(q): float(np.percentile(a,q)) for q in (5,25,50,75,90,95,99)})
    return row


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    names = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='', encoding='utf8') as stream:
        writer = csv.DictWriter(stream, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')


def _base_key(scenario):
    return scenario.identity['source_fragment_id']


def opportunity_summary(rows, label):
    full = [r for r in rows if r['window_complete']]
    waitable = [r for r in full if r['legal_wait_slots'] > 0]
    positive = [r for r in waitable if r['saving_ci_integral_g_per_kwh_h'] > 1e-8]
    out = dict(group=label, tasks=len(rows), full_windows=len(full),
               censored_windows=len(rows)-len(full), waitable_tasks=len(waitable),
               positive_opportunity_tasks=len(positive),
               positive_fraction_of_all=len(positive)/len(rows) if rows else 0.,
               positive_fraction_of_waitable=len(positive)/len(waitable) if waitable else 0.)
    out.update(stats([r['saving_fraction']*100 for r in waitable], 'saving_percent_waitable_'))
    out.update(stats([r['saving_fraction']*100 for r in positive], 'saving_percent_positive_'))
    # Resource-weighted CI integrals are not physical carbon or full energy.
    for resource in ('cpu', 'gpu', 'mem'):
        denominator = sum(r[resource] * r['immediate_ci_integral_g_per_kwh_h'] for r in full)
        numerator = sum(r[resource] * r['saving_ci_integral_g_per_kwh_h'] for r in full)
        out[resource+'_resource_weighted_ci_integral_reduction_percent'] = 100*numerator/denominator if denominator else 0.
    return out


def diagnose_opportunities(scenarios, output):
    bases = {}
    for s in scenarios:
        if s.identity['resource_pressure_factor'] == 1.:
            bases[(_base_key(s), s.identity['sla_multiplier'])] = s
    ci_bases = {key: s for (key, sla),s in bases.items() if sla == 1.5}
    ci_rows, segments = [], []
    for key,s in sorted(ci_bases.items(), key=lambda kv: kv[1].scenario_id):
        arrivals = [x for x in s.environment.samples if s.workload.start <= x.time < s.workload.end]
        a = np.asarray([x.ci_g_per_kwh for x in arrivals])
        segments.append(a)
        ci_rows.append(dict(scenario_id=s.scenario_id, source_fragment_id=key,
                            arrival_start=s.workload.start.isoformat(), arrival_end=s.workload.end.isoformat(),
                            **stats(a,'ci_g_per_kwh_'), peak_trough_g_per_kwh=float(a.max()-a.min())))
    all_ci = np.concatenate(segments)
    ci_summary = dict(arrival_only=True, distinct_source_fragments=len(ci_bases),
                      **stats(all_ci, 'ci_g_per_kwh_'),
                      peak_trough_g_per_kwh=float(all_ci.max()-all_ci.min()),
                      acf=[segmented_acf(segments, lag) for lag in (1,4,16,32,48,92)])
    write_csv(output/'ci_by_fragment.csv', ci_rows)
    write_json(output/'ci_summary.json', ci_summary)
    all_rows, fragment_rows = [], []
    for (key,sla),s in sorted(bases.items(), key=lambda kv: kv[1].scenario_id):
        ci = np.asarray([sample.ci_g_per_kwh for sample in s.environment.samples])
        rows=[]
        for task in s.workload.tasks:
            start=int((task.arrival_time-s.environment.start).total_seconds()/900)
            slack=(task.sla_multiplier-1)*task.duration
            row=integral_opportunity(ci,start,task.duration,slack)
            row.update(scenario_id=s.scenario_id, source_fragment_id=key, sla_multiplier=sla,
                       duration_minutes=task.duration, initial_slack_minutes=slack,
                       cpu=task.cores_req,gpu=task.gpu_req,mem=task.mem_req)
            rows.append(row)
        fragment_rows.append(dict(scenario_id=s.scenario_id,source_fragment_id=key,
                                  sla_multiplier=sla,environment_end=s.environment.end.isoformat(),
                                  **opportunity_summary(rows,'fragment')))
        all_rows.extend(rows)
    write_csv(output/'opportunity_by_fragment.csv',fragment_rows)
    summaries=[]
    for sla in (1.5,2.):
        rows=[r for r in all_rows if r['sla_multiplier']==sla]
        summaries.append(opportunity_summary(rows,'sla='+str(sla)))
        for lo,hi,name in ((0,15,'0-15m'),(15,60,'15-60m'),(60,240,'1-4h'),(240,float('inf'),'>=4h')):
            summaries.append(opportunity_summary([r for r in rows if lo<=r['duration_minutes']<hi],f'sla={sla};duration={name}'))
        for lo,hi,name in ((0,1,'0slots'),(1,4,'1-3slots'),(4,16,'4-15slots'),(16,float('inf'),'>=16slots')):
            summaries.append(opportunity_summary([r for r in rows if lo<=r['legal_wait_slots']<hi],f'sla={sla};slack={name}'))
    write_csv(output/'opportunity_groups.csv',summaries)
    return ci_summary, summaries


class CachedFixedPower:
    """Cache identical fixed-setpoint algebraic idle readings only.

    Validate repeatability first; every distinct weather tuple still calls the
    unchanged model. Active readings always call the unchanged model.
    """
    def __init__(self, scenario):
        from experiments.joint.power import JointPowerModel
        self.raw=JointPowerModel(scenario.datacenter,scenario.model_path)
        self.cache={}
        self.calls=0
        sample=scenario.environment.samples[0]
        a=self.raw.measure(0,0,0,sample)
        b=self.raw.measure(0,0,0,sample)
        if any(not math.isclose(getattr(a,name),getattr(b,name),rel_tol=1e-12,abs_tol=1e-9)
               for name in ('it_kw','hvac_kw','total_kw','rack_inlet_max_c','rack_outlet_max_c')):
            raise RuntimeError('Fixed idle is stateful; cache is invalid')
        self.raw.reset()

    def measure(self,cpu,gpu,mem,sample):
        if cpu or gpu or mem:
            self.calls+=1
            return self.raw.measure(cpu,gpu,mem,sample)
        key=(sample.temperature_c,sample.wet_bulb_c)
        if key not in self.cache:
            self.calls+=1
            self.cache[key]=self.raw.measure(0,0,0,sample)
        return self.cache[key]


def rule_row(s,policy,power):
    from experiments.e1.engine import LocalExecutor
    ex=LocalExecutor(s.workload,s.environment,s.capacity,power,policy=policy,drain=True)
    result=ex.run(stop_at_drain=True)
    active_end=ex.observed_end
    active_steps=list(result.steps)
    completed=result.summary['completed']
    if completed==len(s.workload.tasks):
        result=ex.extend_idle(s.environment.end)
    baseline_carbon=baseline_it=baseline_hvac=0.
    for sample in s.environment.samples:
        reading=power.measure(0,0,0,sample)
        baseline_carbon+=reading.total_kw*.25*sample.ci_g_per_kwh/1000
        baseline_it+=reading.it_kw*.25
        baseline_hvac+=reading.hvac_kw*.25
    total=result.summary
    misses=total['deadline_missed_completed']+total['deadline_missed_unfinished']
    jct=[(pd.Timestamp(t['finish_time'])-pd.Timestamp(t['arrival_time'])).total_seconds()/60
         for t in result.tasks if t['state']=='completed']
    active_carbon=sum(x['carbon_kg'] for x in active_steps)
    active_idle=sum(power.measure(0,0,0,x).total_kw*.25*x.ci_g_per_kwh/1000
                    for x in s.environment.samples if x.time<active_end)
    return dict(scenario_id=s.scenario_id,source_fragment_id=_base_key(s),
        pressure=s.identity['resource_pressure_factor'],sla=s.identity['sla_multiplier'],
        workload_sha256=s.identity['workload_sha256'],environment_sha256=s.identity['environment_sha256'],
        policy=policy.name,threshold_g_per_kwh=getattr(policy,'threshold_g_per_kwh',None),fixed_setpoint_c=22,
        window_start=s.environment.start.isoformat(),window_end=s.environment.end.isoformat(),
        active_end=active_end.isoformat(),full_fixed_window=ex.observed_end==s.environment.end,
        arrived=total['arrived'],completed=completed,completion_fraction=completed/total['arrived'],
        dmr=misses/total['arrived'],mean_jct_minutes=float(np.mean(jct)) if jct else None,
        total_carbon_kg=total['carbon_kg'],it_energy_kwh=total['it_energy_kwh'],
        hvac_energy_kwh=total['hvac_energy_kwh'],total_energy_kwh=total['total_energy_kwh'],
        idle_fixed22_carbon_kg=baseline_carbon,incremental_carbon_kg=total['carbon_kg']-baseline_carbon,
        idle_it_energy_kwh=baseline_it,idle_hvac_energy_kwh=baseline_hvac,
        active_prefix_carbon_kg=active_carbon,active_prefix_idle_carbon_kg=active_idle,
        active_prefix_incremental_carbon_kg=active_carbon-active_idle,
        **{k:result.policy_summary[k] for k in ('policy_defer_intervals','capacity_wait_intervals',
            'policy_deferred_tasks','capacity_waited_tasks','pending_max')})


def select_rules(scenarios):
    # Selection is fixed before outcomes, using only inherited Train arrival work.
    bases={_base_key(s):s for s in scenarios
           if s.identity['resource_pressure_factor']==1. and s.identity['sla_multiplier']==1.5}
    ranked=sorted(bases,key=lambda key:(bases[key].statistics['capacity_pressure'],bases[key].scenario_id))
    chosen=[ranked[0],ranked[(len(ranked)-1)//2],ranked[-1]]
    return sorted([s for s in scenarios if _base_key(s) in chosen],
                  key=lambda s:(chosen.index(_base_key(s)),s.identity['resource_pressure_factor'],s.identity['sla_multiplier']))


def read_existing_rules(path, scenarios):
    """Reuse only identity-matching local results from this exact A0 design."""
    path=Path(path)
    if not path.exists():
        return []
    by_id={s.scenario_id:s for s in scenarios}
    rows=[]
    with path.open(newline='',encoding='utf8') as stream:
        for row in csv.DictReader(stream):
            for key,value in row.items():
                if value=='':
                    row[key]=None
                elif value in ('True','False'):
                    row[key]=value=='True'
                else:
                    try:
                        row[key]=float(value)
                    except ValueError:
                        pass
            s=by_id.get(row['scenario_id'])
            if (s is None or row['workload_sha256']!=s.identity['workload_sha256']
                    or row['environment_sha256']!=s.identity['environment_sha256']
                    or row['window_end']!=s.environment.end.isoformat()
                    or row['fixed_setpoint_c']!=22
                    or row['arrived']!=len(s.workload.tasks)
                    or row['policy'] not in ('edf_backfill','immediate_fifo','carbon_threshold_fifo')):
                raise ValueError('Stored rule result identity/design mismatch')
            rows.append(row)
    if len({(r['scenario_id'],r['policy']) for r in rows})!=len(rows):
        raise ValueError('Duplicated stored rule identity')
    return rows


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pool',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--max-seconds',type=float,default=900)
    parser.add_argument('--opportunity-only',action='store_true')
    parser.add_argument('--reuse-rules',action='store_true',help='Add missing rules after exact local identity checks')
    args=parser.parse_args(argv)
    if not 0<args.max_seconds<=900:
        parser.error('A0 carbon maximum budget is 900 seconds')
    started=time.monotonic()
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    from experiments.joint_v2.scenarios import load_pool
    from experiments.e1.contracts import file_hash,content_hash
    from experiments.e1.policies import EDFBackfill,ImmediateFIFO,CarbonThresholdFIFO
    scenarios=load_pool(args.pool,('train',))
    ci_summary,opportunity=diagnose_opportunities(scenarios,output)
    chosen=select_rules(scenarios)
    rows=read_existing_rules(output/'rule_comparison.csv',chosen) if args.reuse_rules else []
    for s in ([] if args.opportunity_only else chosen):
        if time.monotonic()-started>args.max_seconds-30:
            break
        power=CachedFixedPower(s)
        for policy in (EDFBackfill(),CarbonThresholdFIFO(161.075),ImmediateFIFO()):
            if any(r['scenario_id']==s.scenario_id and r['policy']==policy.name for r in rows):
                continue
            rows.append(rule_row(s,policy,power))
            write_csv(output/'rule_comparison.csv',rows)
            print(json.dumps({k:rows[-1][k] for k in ('scenario_id','policy','completed','dmr','total_carbon_kg','incremental_carbon_kg')}),flush=True)
            power.raw.reset()
        power.raw.close()
    pairs=[]
    for s in chosen:
        pair=[r for r in rows if r['scenario_id']==s.scenario_id]
        by_policy={r['policy']:r for r in pair}
        if 'carbon_threshold_fifo' not in by_policy:
            continue
        b=by_policy['carbon_threshold_fifo']
        for baseline in ('immediate_fifo','edf_backfill'):
            if baseline not in by_policy:
                continue
            a=by_policy[baseline]
            pairs.append(dict(scenario_id=s.scenario_id,pressure=a['pressure'],sla=a['sla'],
                baseline_policy=baseline,comparison_policy='carbon_threshold_fifo',
                carbon_saved_kg=a['total_carbon_kg']-b['total_carbon_kg'],
                total_carbon_reduction_percent=100*(a['total_carbon_kg']-b['total_carbon_kg'])/a['total_carbon_kg'],
                incremental_carbon_reduction_percent=100*(a['incremental_carbon_kg']-b['incremental_carbon_kg'])/a['incremental_carbon_kg'] if a['incremental_carbon_kg'] else None,
                dmr_immediate=a['dmr'],dmr_threshold=b['dmr'],
                jct_change_minutes=b['mean_jct_minutes']-a['mean_jct_minutes']))
    write_csv(output/'rule_pairs.csv',pairs)
    root=Path(__file__).resolve().parents[2]
    try:
        sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True,stderr=subprocess.DEVNULL).strip()
    except subprocess.CalledProcessError:
        # Windows-managed worktree .git paths are not understood by Linux git.
        # This is the explicitly frozen A0 production baseline, not an inferred
        # identity of new diagnosis code (whose own file hash is also recorded).
        sha='bf8a35ef647f19b5ef10ce1b99f723370b7a0732'
    metadata=dict(schema='a0-carbon-diagnosis-v1',source_sha=sha,
        script_sha256=file_hash(__file__),pool_manifest_sha256=file_hash(Path(args.pool)/'manifest.json'),
        role='train',loaded_scenarios=len(scenarios),base_fragments=len({ _base_key(s) for s in scenarios}),
        data_identity_sha256=content_hash([s.identity for s in scenarios]),
        source_hashes=scenarios[0].identity['original_source_hashes'],
        arrival_start_min=min(s.workload.start for s in scenarios).isoformat(),
        arrival_end_max=max(s.workload.end for s in scenarios).isoformat(),
        frozen_environment_end_max=max(s.environment.end for s in scenarios).isoformat(),
        ci_description='16 nonduplicated Train 24-hour arrival fragments; ACF within each fragment only',
        source_independence='Distinct source fragments are deduplicated portions of the same 49-day Alibaba trace, not independent statistical samples.',
        opportunity='perfect future-CI per-task legal 15-minute starts; full ceil(duration/15) left-held CI integral; ignores all capacities, competition, power nonlinearity and full HVAC; not achievable optimum',
        opportunity_resource_weighting='CPU/GPU/MEM demand times CI integral; each reported separately, no claim of physical kgCO2e',
        rules='existing LocalExecutor, capacity-first one-pass backfill; ImmediateFIFO/ImmediateEDF/CarbonThresholdFIFO(161.075 inherited, not tuned); same tasks/environment/window; fixed22 C existing JointPowerModel throughout',
        threshold_source='experiments/e1/configs/e1_5b_ppo.json:threshold_g_per_kwh inherited literal, unchanged for A0',
        note='Primary threshold effect uses ImmediateFIFO versus CarbonThresholdFIFO (same FIFO ordering). EDF versus ThresholdFIFO is secondary and combines ordering and defer eligibility effects.',
        units=dict(ci='gCO2e/kWh',ci_integral='gCO2e*h/kWh',power='kW',energy='kWh',carbon='kgCO2e',jct='minutes'),
        selection='sort pressure=1,SLA=1.5 bases by inherited arrival resource-work pressure; choose ranks0,7,15 before outcomes',
        selected_scenarios=[s.scenario_id for s in chosen],planned_rule_runs=54,
        completed_rule_runs=len(rows),complete_pairs=len(pairs),budget_seconds=args.max_seconds,
        last_invocation_elapsed_seconds=time.monotonic()-started,
        timing_scope='Last invocation includes opportunity summaries and any newly simulated missing rules; identity-reused prior rule runs are excluded. Cumulative A0 work is bounded and reported by the parent A0 wall clock, not this field.',
        test_access=False,validation_access=False,development_access=False,
        ci_summary=ci_summary,opportunity_summary=[r for r in opportunity if ';' not in r['group']])
    write_json(output/'metadata.json',metadata)
    print(json.dumps(dict(completed_rule_runs=len(rows),last_invocation_elapsed_seconds=metadata['last_invocation_elapsed_seconds'])),flush=True)


if __name__=='__main__':
    main()
