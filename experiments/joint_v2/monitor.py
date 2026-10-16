"""Episode, system-carbon and optimizer monitoring, rebuilt from committed state."""
import csv
import json
import math
from pathlib import Path

from .checkpoint import atomic_json


def step_record(outcome, *, episode, global_step, scenario_id):
    row = dict(episode=episode, global_step=global_step, scenario_id=scenario_id,
        reward=outcome.reward, setpoint_c=outcome.interval['setpoint_c'],
        total_power_kw=outcome.interval['total_power_kw'], carbon_kg=outcome.interval['carbon_kg'],
        incremental_carbon_kg=outcome.terms['incremental_carbon_kg'],
        task_candidates=len(outcome.task_actions), task_free_actions=sum(all(x['mask']) for x in outcome.task_actions),
        task_start=sum(x['action']==0 for x in outcome.task_actions),
        task_wait=sum(x['action']==1 for x in outcome.task_actions),
        cooling_actions=len(outcome.cooling_actions),
        thermal_excess_c=outcome.terms['thermal_excess_c'], stage=outcome.interval['stage'])
    return row


def episode_record(index, scenario, summary, updates, constraint, curriculum, counters, elapsed):
    result = dict(episode=index, scenario_id=scenario.scenario_id,
        source_fragment_id=scenario.identity.get('source_fragment_id'),
        synthetic=bool(scenario.statistics.get('synthetic', False)),
        augmentation=scenario.statistics.get('augmentation', {}), scenario_statistics=scenario.statistics,
        summary=summary, ppo=updates, constraint=constraint, curriculum=dict(curriculum), counters=dict(counters),
        elapsed_seconds=elapsed)
    return result


def capacity_audit(scenario):
    """A report-only necessary lower bound, never a policy/critic input.

    Any task completed by a deadline must consume its actual execution resource
    work inside [scenario start,deadline]. Even allowing fractional/preemptive
    work cannot exceed capacity*time. Largest-work removals certify a minimal
    miss cardinality for these prefix intervals; no overload proves nothing.
    """
    from experiments.e1.contracts import STEP
    from experiments.e1.policies import latest_start
    tasks = tuple(scenario.workload.tasks)
    intrinsic = sum(any(getattr(t,n)>c for n,c in zip(('cores_req','gpu_req','mem_req'),scenario.capacity))
                    or latest_start(t).floor('15min') < t.arrival_time for t in tasks)
    lower = intrinsic
    # All original deadlines are known only to this OFFLINE reporting audit.
    ordered = sorted(tasks,key=lambda t:t.sla_deadline)
    work = [[],[],[]]
    for i,t in enumerate(ordered):
        for j,n in enumerate(('cores_req','gpu_req','mem_req')):
            work[j].append(getattr(t,n)*t.duration)
        if i+1<len(ordered) and ordered[i+1].sla_deadline==t.sla_deadline:
            continue
        minutes=(t.sla_deadline-scenario.workload.start).total_seconds()/60
        for weights,capacity in zip(work,scenario.capacity):
            excess=math.fsum(weights)-capacity*minutes
            if excess <= 1e-9*max(1.,capacity*minutes):
                continue
            removed=0
            for value in sorted(weights,reverse=True):
                excess-=value; removed+=1
                if excess <= 1e-9*max(1.,capacity*minutes): break
            lower=max(lower,removed)
    return dict(role='offline_report_only_no_policy_input', intrinsic_impossible_tasks=intrinsic,
        necessary_deadline_misses_lower_bound=lower,
        necessary_dmr_lower_bound=lower/len(tasks) if tasks else 0.,
        scope='actual-resource-work prefix intervals plus individual latest grid start',
        limitation='Zero bound does not prove feasible; observed misses above bound are not all certified avoidable.')


class Monitor:
    def __init__(self, output, *, tensorboard=True):
        self.output=Path(output); self.output.mkdir(parents=True,exist_ok=True)
        self.writer=None; self.logged=0
        self.logged_validations=0
        if tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            # Saved records are authoritative; purge/replay prevents duplicate
            # episode scalars if interruption occurred after checkpoint commit.
            self.writer=SummaryWriter(str(self.output/'tensorboard'),purge_step=0)

    def flush(self,state):
        records=state['episodes']; steps=state['step_history']
        atomic_json(self.output/'episodes.json',records)
        atomic_json(self.output/'steps.json',steps)
        fields=('episode','scenario_id','source_fragment_id','synthetic','arrived','completed','unfinished',
            'dmr','temperature_violations','carbon_kg','incremental_carbon_kg','background_carbon_kg',
            'reward','wait_cost','deadline_cost','terminal_cost','thermal_safety_cost',
            'lambda','curriculum_stage','physical_steps','task_actions','cooling_actions','elapsed_seconds',
            'waiting_mean_minutes','jct_completed_mean_minutes','capacity_wait_intervals','policy_defer_intervals',
            'deadline_guard_triggers','it_energy_kwh','hvac_energy_kwh','it_carbon_kg','hvac_carbon_kg',
            'crac_fan_energy_kwh','ct_fan_energy_kwh','compressor_energy_kwh','cw_pump_energy_kwh','ct_pump_energy_kwh')
        with (self.output/'episodes.csv').open('w',encoding='utf8',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields); writer.writeheader()
            for row in records:
                summary=row['summary']; reward=summary['reward']
                flat={k:summary.get(k) for k in ('arrived','completed','unfinished','dmr','temperature_violations',
                                               'carbon_kg','incremental_carbon_kg','background_carbon_kg')}
                flat.update({k:row[k] for k in ('episode','scenario_id','source_fragment_id','synthetic','elapsed_seconds')})
                flat.update({k:reward[k] for k in ('reward','wait_cost','deadline_cost','terminal_cost','thermal_safety_cost')})
                flat.update({'lambda':row['constraint']['value'],'curriculum_stage':row['curriculum']['stage'],
                    'physical_steps':row['counters']['physical_steps'],'task_actions':row['counters']['task_actions'],
                    'cooling_actions':row['counters']['cooling_actions']})
                for key in fields:
                    if key not in flat and key in summary:flat[key]=summary[key]
                writer.writerow(flat)
        if steps:
            with (self.output/'steps.csv').open('w',encoding='utf8',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(steps[0]));writer.writeheader();writer.writerows(steps)
        optimizer_rows=[]
        def flatten(prefix,value,result):
            for key,item in value.items():
                name=prefix+key
                if isinstance(item,dict):flatten(name+'/',item,result)
                elif isinstance(item,(int,float)) and not isinstance(item,bool):result[name]=item
        for row in records:
            for role,metrics in row['ppo'].items():
                flat=dict(episode=row['episode'],agent=role);flatten('',metrics,flat);optimizer_rows.append(flat)
        if optimizer_rows:
            columns=sorted(set().union(*(row.keys() for row in optimizer_rows)))
            with (self.output/'ppo_updates.csv').open('w',encoding='utf8',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader();writer.writerows(optimizer_rows)
        atomic_json(self.output/'validation.json',state['validations'])
        atomic_json(self.output/'capacity_audits.json',state.get('capacity_audits',{}))
        atomic_json(self.output/'selection.json',dict(best_feasible=state['best_feasible'],
            best_deadline=state['best_deadline'],status='feasible_candidate' if state['best_feasible'] else 'no_feasible_model'))
        if self.writer:
            for row in records[self.logged:]:
                index=row['episode']; summary=row['summary']
                for key in ('dmr','carbon_kg','incremental_carbon_kg','background_carbon_kg',
                            'completed','unfinished','temperature_violations','deadline_guard_triggers'):
                    self.writer.add_scalar('system/'+key,summary[key],index)
                self.writer.add_scalar('system/completion_rate',summary['completed']/max(1,summary['arrived']),index)
                self.writer.add_scalar('constraint/lambda',row['constraint']['value'],index)
                for key,value in summary['reward'].items():
                    self.writer.add_scalar('reward/'+key,value,index)
                for agent,metrics in row['ppo'].items():
                    scalars={};flatten('',metrics,scalars)
                    for key,value in scalars.items():
                        if math.isfinite(value):
                            self.writer.add_scalar(agent+'/'+key,value,index)
                for action,count in summary['task_action_distribution'].items():
                    self.writer.add_scalar('task_actions/'+action,count,index)
            for row in state['validations'][self.logged_validations:]:
                for key in ('dmr','carbon_kg','incremental_carbon_kg','completed','unfinished','temperature_violations'):
                    self.writer.add_scalar('development/'+key,row[key],row['episode'])
            self.writer.flush(); self.logged=len(records);self.logged_validations=len(state['validations'])

    def close(self):
        if self.writer:self.writer.close()


def plot(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output=Path(output)
    records=json.loads((output/'episodes.json').read_text(encoding='utf8'))
    steps=json.loads((output/'steps.json').read_text(encoding='utf8'))
    if not records:return None
    fig,axes=plt.subplots(2,3,figsize=(14,7),layout='constrained')
    x=[r['episode'] for r in records]; summaries=[r['summary'] for r in records]
    axes[0,0].plot(x,[s['dmr'] for s in summaries],marker='.'); axes[0,0].axhline(.05,color='r',ls='--')
    axes[0,0].set(title='Deadline miss rate',xlabel='Completed episode',ylabel='DMR')
    axes[0,1].plot(x,[s['completed']/max(1,s['arrived']) for s in summaries],marker='.')
    axes[0,1].set(title='Task completion rate',xlabel='Completed episode',ylabel='Completed / arrived',ylim=(0,1.05))
    axes[0,2].plot(x,[s['carbon_kg'] for s in summaries],marker='.',label='Fixed-window total')
    axes[0,2].plot(x,[s['incremental_carbon_kg'] for s in summaries],marker='.',label='Task/control increment')
    axes[0,2].set(title='System carbon (scenarios vary)',xlabel='Completed episode',ylabel='kg CO2e');axes[0,2].legend()
    validations=json.loads((output/'validation.json').read_text(encoding='utf8'))
    if validations:
        vx=[r['episode'] for r in validations]
        axes[0,0].plot(vx,[r['dmr'] for r in validations],marker='x',label='Development');axes[0,0].legend()
        axes[0,1].plot(vx,[r['completed']/max(1,r['arrived']) for r in validations],marker='x',label='Development');axes[0,1].legend()
    for action,label in [('0','Start'),('1','Wait')]:
        axes[1,0].plot(x,[s['task_action_distribution'].get(action,0)/max(1,sum(s['task_action_distribution'].values())) for s in summaries],label=label)
    axes[1,0].set(title='Task action distribution',xlabel='Completed episode',ylabel='Decision fraction');axes[1,0].legend()
    axes[1,1].plot([s['global_step'] for s in steps],[s['setpoint_c'] for s in steps],lw=.8)
    axes[1,1].set(title='CRAC setting, including settling',xlabel='Committed physical step',ylabel='Setpoint (C)',ylim=(17.5,22.5))
    for agent in ('task','cooling'):
        axes[1,2].plot(x,[r['ppo'][agent].get('policy_loss',0.) for r in records],label=agent)
    axes[1,2].set(title='PPO policy loss',xlabel='Completed episode',ylabel='Loss');axes[1,2].legend()
    for ax in axes.flat:ax.grid(alpha=.2)
    path=output/'training-curves.png';fig.savefig(path,dpi=160);plt.close(fig)
    return path
