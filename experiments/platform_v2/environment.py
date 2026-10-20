"""V1 queue/release/ledger semantics with injected V2 physics and authority.

Only initialization is adapted to inject the model; core lifecycle methods
remain inherited unchanged from platform_v1.environment.PlatformEnv.
"""
from dataclasses import replace
import math
from experiments.platform_v1.environment import PlatformEnv as BaseEnv,Action,baseline_action
from .physics import PhysicalModel


class PlatformEnv(BaseEnv):
    def __init__(self,scenario,inventory,cfg,sla_multiplier,authority='joint'):
        if authority not in ['fixed','independent','joint']:raise ValueError('Unknown HVAC authority')
        if not math.isfinite(sla_multiplier) or sla_multiplier<1:raise ValueError('Invalid synthetic SLA')
        self.scenario=scenario;self.inv=inventory;self.cfg=cfg;self.sla=sla_multiplier;self.authority=authority
        self.model=PhysicalModel(inventory,cfg);self.tasks=tuple(scenario.tasks)
        if len({t.task_id for t in self.tasks})!=len(self.tasks):raise ValueError('Duplicate task ID')
        if not scenario.observations or any(o.minute!=i*15 or o.observed_minute>o.minute for i,o in enumerate(scenario.observations)):
            raise ValueError('Incomplete grid or future environmental observation')
        for task in self.tasks:
            if (not math.isfinite(task.arrival_minute) or task.arrival_minute<0 or task.arrival_minute%15
                    or not math.isfinite(task.duration_minutes) or task.duration_minutes<=0 or len(task.demand)!=3
                    or any(not math.isfinite(x) or x<0 or x>c for x,c in zip(task.demand,inventory.capacity))):
                raise ValueError('Invalid or intrinsically infeasible task; never drop it')
        self.horizon=15*len(scenario.observations);self.arrivals={}
        for task in self.tasks:self.arrivals.setdefault(task.arrival_minute,[]).append(task)
        self._initialized=False

    def observation(self):
        out=super().observation();out['hvac_authority']=self.authority
        return out

    def step(self,action):
        if isinstance(action,Action) and self.authority!='joint':
            locked=self.cfg['evaluation']['fixed_supply_c'] if self.authority=='fixed' else self.cfg['thermal']['supply_max_c']
            if action.supply_c!=locked:raise ValueError('HVAC action locked by research question/control authority')
        return super().step(action)

    def result(self):
        result=super().result();summary=result['summary'];steps=result['steps']
        for prefix,subset in [('arrival',[s for s in steps if s['minute']<1440]),('drain',[s for s in steps if s['minute']>=1440])]:
            summary[prefix+'_hours']=len(subset)*.25
            for label in ['it','hvac','total']:
                summary[prefix+'_'+label+'_kwh']=math.fsum(s[label+'_kw']*.25 for s in subset)
            summary[prefix+'_carbon_kg']=math.fsum(s['carbon_kg'] for s in subset)
            for axis,name in enumerate(['cpu','gpu','mem']):
                summary[prefix+'_'+name+'_mean_util']=math.fsum(s['used_'+name]/self.inv.capacity[axis] for s in subset)/len(subset) if subset else 0.
                summary[prefix+'_'+name+'_peak_util']=max([0]+[s['used_'+name]/self.inv.capacity[axis] for s in subset])
        summary.update(cooling_infeasible_steps=sum(not s['cooling_feasible'] for s in steps),
                       performance_domain_invalid_steps=sum(not s['curve_domain_valid'] for s in steps),
                       low_ambient_boundary_steps=sum(s['low_ambient_boundary'] for s in steps),
                       unmet_cooling_kwh_th=math.fsum(s['unmet_cooling_kw']*.25 for s in steps),
                       max_unmet_cooling_kw=max(s['max_unmet_cooling_kw'] for s in steps),
                       max_chiller_balance_residual_kj=max(abs(s['chiller_balance_residual_kj']) for s in steps),
                       thermal_capacity_kj_k=self.model.c_kj_k,ua_kw_k=self.model.ua_kw_k,
                       last_task_finish_minute=max([0]+[t['finish_minute'] or 0 for t in result['tasks']]))
        for k in ['evaporator_kw','package_kw']:
            summary[k.removesuffix('_kw')+'_kwh']=math.fsum(s[k]*.25 for s in steps)
        return result


def run(scenario,inventory,cfg,policy,cooling,sla_multiplier,idle=False):
    if idle:scenario=replace(scenario,tasks=())
    authority='fixed' if cooling=='fixed' else 'independent'
    env=PlatformEnv(scenario,inventory,cfg,sla_multiplier,authority);obs=env.reset();done=False
    while not done:obs,done,_=env.step(baseline_action(obs,policy,cooling,cfg))
    return env.result()
