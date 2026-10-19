"""15-minute task order/defer and HVAC actions; complete immutable task ledger."""
from dataclasses import dataclass, replace
import math
import statistics

from .physics import PhysicalModel


@dataclass(frozen=True)
class Action:
    order: tuple
    defer: frozenset
    supply_c: float


class PlatformEnv:
    def __init__(self,scenario,inventory,cfg,sla_multiplier):
        if not math.isfinite(sla_multiplier) or sla_multiplier<1:raise ValueError('Invalid synthetic SLA')
        self.scenario=scenario;self.inv=inventory;self.cfg=cfg;self.sla=sla_multiplier
        self.model=PhysicalModel(inventory,cfg)
        self.tasks=tuple(replace(t,demand=tuple(inventory.mapping_factor*x for x in t.demand)) for t in scenario.tasks)
        if len({t.task_id for t in self.tasks})!=len(self.tasks):raise ValueError('Duplicate task ID')
        if not scenario.observations or any(o.minute!=i*15 or o.observed_minute>o.minute for i,o in enumerate(scenario.observations)):
            raise ValueError('Incomplete grid or future environmental observation')
        for task in self.tasks:
            if (not math.isfinite(task.arrival_minute) or task.arrival_minute<0 or task.arrival_minute%15
                    or not math.isfinite(task.duration_minutes) or task.duration_minutes<=0 or len(task.demand)!=3
                    or any(not math.isfinite(x) or x<0 or x>c for x,c in zip(task.demand,inventory.capacity))):
                raise ValueError('Invalid or intrinsically infeasible task; never drop it')
        self.horizon=15*len(scenario.observations)
        self.arrivals={}
        for task in self.tasks:self.arrivals.setdefault(task.arrival_minute,[]).append(task)
        self._initialized=False

    def reset(self):
        self.minute=0.;self.inlet=self.cfg['thermal']['initial_inlet_c'];self.pending=[];self.running=[]
        self.ledger={t.task_id:dict(task=t,state='future',start_minute=None,finish_minute=None,
                                  capacity_wait_minutes=0,policy_wait_minutes=0) for t in self.tasks}
        self.steps=[];self.history=[];self._initialized=True;self._prepare()
        return self.observation()

    def _prepare(self):
        retained=[]
        for task in self.running:
            entry=self.ledger[task.task_id]
            if entry['release_minute']<=self.minute:entry['state']='completed'
            else:retained.append(task)
        self.running=retained
        if self.minute<self.horizon:
            for task in self.arrivals.get(self.minute,[]):
                self.ledger[task.task_id]['state']='pending';self.pending.append(task)

    def observation(self):
        if self.minute>=self.horizon:return {'minute':self.minute,'terminated':True}
        sample=self.scenario.observations[int(self.minute/15)]
        used=tuple(math.fsum(t.demand[i] for t in self.running) for i in range(3))
        return dict(minute=self.minute,ci_g_per_kwh=sample.ci_g_per_kwh,outdoor_c=sample.outdoor_c,
                    observed_minute=sample.observed_minute,inlet_c=self.inlet,capacity=self.inv.capacity,
                    available=tuple(c-u for c,u in zip(self.inv.capacity,used)),pending=tuple(self.pending),
                    running=tuple((t.task_id,t.demand,self.ledger[t.task_id]['finish_minute']) for t in self.running),
                    ci_history=tuple(self.history),sla_multiplier=self.sla)

    def step(self,action):
        if not self._initialized or self.minute>=self.horizon:raise RuntimeError('Reset before step; no stepping past responsibility window')
        pending={t.task_id:t for t in self.pending}
        if not isinstance(action,Action) or len(action.order)!=len(pending) or len(set(action.order))!=len(pending) or set(action.order)!=set(pending) or not set(action.defer)<=set(pending):
            raise ValueError('Action must preserve every pending task exactly once')
        # Validate physical action before mutating the task ledger.
        if not math.isfinite(action.supply_c) or not self.cfg['thermal']['supply_min_c']<=action.supply_c<=self.cfg['thermal']['supply_max_c']:
            raise ValueError('Invalid HVAC action')
        sample=self.scenario.observations[int(self.minute/15)]
        used=[math.fsum(t.demand[i] for t in self.running) for i in range(3)]
        remaining=[];blocked=set();starts=[];policy_wait=0;capacity_wait=0
        bottlenecks=[0,0,0]
        for identity in action.order:
            task=pending[identity];entry=self.ledger[identity]
            resources=[i for i in range(3) if math.fsum([used[i],task.demand[i]])>self.inv.capacity[i]]
            if resources:
                entry['capacity_wait_minutes']+=15;capacity_wait+=1;blocked.add(identity)
                for i in resources:bottlenecks[i]+=1
                remaining.append(task)
            elif identity in action.defer:
                entry['policy_wait_minutes']+=15;policy_wait+=1;remaining.append(task)
            else:
                entry.update(state='running',start_minute=self.minute,finish_minute=self.minute+task.duration_minutes,
                             release_minute=15*math.ceil((self.minute+task.duration_minutes)/15-1e-12))
                self.running.append(task);starts.append(task)
                used=[math.fsum([used[i],task.demand[i]]) for i in range(3)]
        self.pending=remaining
        if any(u>c+1e-9 for u,c in zip(used,self.inv.capacity)):raise RuntimeError('Capacity conservation failed')
        parts=self.model.it_power(tuple(used));request=action.supply_c
        if self.minute>=self.horizon-self.cfg['evaluation']['terminal_common_hours']*60:
            request=self.cfg['evaluation']['fixed_supply_c']
        applied,thermal,override=self.model.safe_transition(self.inlet,parts['it_kw'],sample.outdoor_c,request)
        row=dict(minute=self.minute,ci_g_per_kwh=sample.ci_g_per_kwh,outdoor_c=sample.outdoor_c,
                 observation_minute=sample.observed_minute,inlet_start_c=self.inlet,supply_requested_c=request,
                 supply_applied_c=applied,safety_override=int(override),used_cpu=used[0],used_gpu=used[1],used_mem=used[2],
                 pending=len(remaining),running=len(self.running),started=len(starts),
                 started_cpu=sum(t.demand[0] for t in starts),started_gpu=sum(t.demand[1] for t in starts),
                 capacity_wait_tasks=capacity_wait,policy_wait_tasks=policy_wait,
                 blocked_cpu=bottlenecks[0],blocked_gpu=bottlenecks[1],blocked_mem=bottlenecks[2],**parts,**thermal)
        row['total_kw']=row['it_kw']+row['hvac_kw']
        row['carbon_kg']=row['total_kw']*.25*sample.ci_g_per_kwh/1000
        self.steps.append(row);self.inlet=thermal['inlet_end_c']
        self.history.append((self.minute,sample.ci_g_per_kwh));self.minute+=15;self._prepare()
        return self.observation(),self.minute>=self.horizon,dict(row)

    def result(self):
        if self.minute!=self.horizon:raise RuntimeError('Fixed responsibility window not fully evaluated')
        rows=[]
        for task in self.tasks:
            e=self.ledger[task.task_id];finish=e['finish_minute'];deadline=task.arrival_minute+self.sla*task.duration_minutes
            complete=e['state']=='completed'
            missed=(finish>deadline+1e-9) if complete else self.horizon>=deadline
            rows.append(dict(task_id=task.task_id,state=e['state'],start_minute=e['start_minute'],finish_minute=finish,
                             deadline_minute=deadline,deadline_missed=missed,deadline_censored=not complete and self.horizon<deadline,
                             capacity_wait_minutes=e['capacity_wait_minutes'],policy_wait_minutes=e['policy_wait_minutes'],
                             jct_minutes=finish-task.arrival_minute if complete else None))
        n=len(rows);completed=[r for r in rows if r['state']=='completed']
        steps=self.steps;arrival_steps=[s for s in steps if s['minute']<self.scenario.arrival_window_minutes]
        summary=dict(arrived=n,completed=len(completed),unfinished=n-len(completed),completion_rate=len(completed)/n if n else 1.,
                     deadline_missed=sum(r['deadline_missed'] for r in rows),deadline_censored=sum(r['deadline_censored'] for r in rows),
                     dmr=sum(r['deadline_missed'] for r in rows)/n if n else 0.,
                     mean_jct_minutes=statistics.mean(r['jct_minutes'] for r in completed) if completed else 0.,
                     capacity_wait_minutes=sum(r['capacity_wait_minutes'] for r in rows),policy_wait_minutes=sum(r['policy_wait_minutes'] for r in rows),
                     capacity_wait_tasks=sum(r['capacity_wait_minutes']>0 for r in rows),policy_wait_tasks=sum(r['policy_wait_minutes']>0 for r in rows),
                     queue_peak=max((s['pending'] for s in steps),default=0),competition_minutes=15*sum(s['capacity_wait_tasks']>0 for s in steps),
                     start_cpu_peak=max(s['started_cpu'] for s in steps),start_gpu_peak=max(s['started_gpu'] for s in steps),
                     it_kwh=sum(s['it_kw']*.25 for s in steps),hvac_kwh=sum(s['hvac_kw']*.25 for s in steps),
                     total_kwh=sum(s['total_kw']*.25 for s in steps),carbon_kg=sum(s['carbon_kg'] for s in steps),
                     inlet_max_c=max(s['inlet_peak_c'] for s in steps),inlet_final_c=self.inlet,
                     thermal_storage_kwh=self.model.c_kj_k*(self.inlet-self.cfg['thermal']['initial_inlet_c'])/3600,
                     safety_overrides=sum(s['safety_override'] for s in steps),
                     max_balance_residual_kj=max(abs(s['balance_residual_kj']) for s in steps),
                     horizon_hours=self.horizon/60.,arrival_hours=self.scenario.arrival_window_minutes/60.,
                     carbon_recalculation_error_kg=abs(sum(s['total_kw']*.25*s['ci_g_per_kwh']/1000 for s in steps)-sum(s['carbon_kg'] for s in steps)))
        for i,name in enumerate(['cpu','gpu','mem']):
            values=[s['used_'+name]/self.inv.capacity[i] for s in arrival_steps]
            summary[name+'_mean_util']=statistics.mean(values) if values else 0.
            summary[name+'_peak_util']=max(values,default=0.)
            summary['blocked_'+name]=sum(s['blocked_'+name] for s in steps)
        for key in ['cpu_kw','gpu_kw','dimm_kw','host_other_kw','psu_loss_kw','rack_switch_kw','supply_fan_kw','pump_kw','compressor_kw','condenser_fan_kw','coil_kw','heat_rejection_kw','envelope_kw']:
            summary[key.removesuffix('_kw')+'_kwh']=sum(s[key]*.25 for s in steps)
        return dict(summary=summary,tasks=rows,steps=steps)


def baseline_action(obs,policy,cooling,cfg):
    if policy not in ['fifo','edf','threshold_fifo','threshold_edf'] or cooling not in ['fixed','variable']:
        raise ValueError('Unknown baseline')
    tasks=list(obs['pending']);sla=obs['sla_multiplier']
    if policy.endswith('edf'):tasks.sort(key=lambda t:(t.arrival_minute+sla*t.duration_minutes,t.arrival_minute,t.task_id))
    else:tasks.sort(key=lambda t:(t.arrival_minute,t.source_row,t.source_index,t.task_id))
    deferred=set();e=cfg['evaluation'];now=obs['minute']
    history=[ci for minute,ci in obs['ci_history'] if minute>=now-e['threshold_history_hours']*60]
    if policy.startswith('threshold') and now>=e['threshold_warmup_hours']*60 and history:
        threshold=statistics.median(history)
        if obs['ci_g_per_kwh']>threshold:
            for task in tasks:
                if now+15<=task.arrival_minute+(sla-1)*task.duration_minutes+1e-9:deferred.add(task.task_id)
    return Action(tuple(t.task_id for t in tasks),frozenset(deferred),
                  e['fixed_supply_c'] if cooling=='fixed' else cfg['thermal']['supply_max_c'])


def run(scenario,inventory,cfg,task_policy,cooling,sla_multiplier,idle=False):
    if idle:scenario=replace(scenario,tasks=())  # separate explicitly labeled physical background, not a workload policy
    env=PlatformEnv(scenario,inventory,cfg,sla_multiplier);obs=env.reset();done=False
    while not done:obs,done,_=env.step(baseline_action(obs,task_policy,cooling,cfg))
    return env.result()
