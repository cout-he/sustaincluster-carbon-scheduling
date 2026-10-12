"""Online Gymnasium mode selection over the single E1 execution kernel.

No RL trainer, task-level learned budget or predictive planner is implemented.
The frozen inputs belong to the simulator, never to observations or policy info.
"""
from dataclasses import dataclass, fields
from collections import deque
import math

import gymnasium as gym
import numpy as np

from .cars import make_cars_policy
from .contracts import STEP, finite
from .engine import LocalExecutor
from .policies import latest_start

RL_SCHEMA='e1-5a-mode-env-v1'
ACTION_POLICIES=('edf_backfill','edf_carbon_threshold','cars_risk_only')


@dataclass(frozen=True)
class ObservationNormalization:
    """Engineering scales, declared independently of held-out input data."""
    ci_g_per_kwh: float=1000.
    count_tasks: float=1000.
    slack_minutes: float=120.
    work_minutes: float=120.
    elapsed_hours: float=24.

    def __post_init__(self):
        for field in fields(self):
            object.__setattr__(self,field.name,finite(getattr(self,field.name),field.name,True))


@dataclass(frozen=True)
class RewardConfig:
    alpha: float=1.
    beta: float=1.
    gamma: float=10.
    carbon_scale_kg: float=1000.
    wait_scale_task_minutes: float=15000.
    deadline_scale_tasks: float=1000.
    terminal_scale_tasks: float=1000.
    terminal_weight: float=10.

    def __post_init__(self):
        for field in fields(self):
            object.__setattr__(self,field.name,finite(getattr(self,field.name),field.name,
                   field.name not in ('alpha','beta','gamma')))


OBSERVATION_FIELDS=(
    'ci_current','ci_lag1','ci_lag2','ci_lag3','ci_lag4','ci_lags_valid_fraction','ci_delta',
    'cpu_utilization','gpu_utilization','mem_utilization','running_count','pending_count',
    'history_cpu_work','history_gpu_work','history_mem_work',
    'history_cpu_rho','history_gpu_rho','history_mem_rho',
    'history_effective_fraction','history_cold','risk_buffer_fraction',
    'solo_slack_min','solo_slack_mean','solo_slack_p10','solo_slack_max',
    'urgent_count','deadline_due_count','solo_latest_start_passed_count',
    'pending_cpu_demand','pending_gpu_demand','pending_mem_demand',
    'pending_cpu_work','pending_gpu_work','pending_mem_work',
    'utc_day_sin','utc_day_cos','utc_weekday_sin','utc_weekday_cos','elapsed_time','arrivals_open')


def _bounded(value,scale=1.):
    return value/(scale+abs(value))


class CarbonSchedulingEnv(gym.Env):
    """One action selects existing eligibility rules for every candidate this step.

    `power_factory` MUST return a fresh physical adapter at reset, never per step.
    Full evaluation should use an externally selected common end with
    stop_at_drain=False. max_episode_steps is an exclusive training cutoff;
    terminal backlog liability covers arrived unfinished work only.
    """
    metadata={'render_modes':[]}
    render_mode=None

    def __init__(self,workload,environment,capacity,power_factory,parameters,*,
                 reward_config=None,normalization=None,max_episode_steps=None,stop_at_drain=False):
        super().__init__()
        if not callable(power_factory): raise ValueError('A fresh power_factory is required')
        if parameters.get('future_ci',False): raise ValueError('Future CI is forbidden')
        finite(parameters['threshold_g_per_kwh'],'fixed carbon threshold')
        if finite(parameters.get('history_minutes',120),'history_minutes',True)!=120:
            raise ValueError('E1-5A requires the unchanged 120-minute history window')
        self._workload,self._environment=workload,environment
        self.capacity=tuple(finite(v,'capacity',True) for v in capacity)
        if len(self.capacity)!=3: raise ValueError('CPU/GPU/MEM required')
        self.power_factory,self.parameters=power_factory,dict(parameters)
        self.reward_config=reward_config or RewardConfig()
        self.normalization=normalization or ObservationNormalization()
        if isinstance(self.reward_config,dict): self.reward_config=RewardConfig(**self.reward_config)
        if isinstance(self.normalization,dict): self.normalization=ObservationNormalization(**self.normalization)
        self._validate_limit(max_episode_steps)
        if not isinstance(stop_at_drain,bool): raise ValueError('stop_at_drain must be bool')
        self.max_episode_steps,self.stop_at_drain=max_episode_steps,stop_at_drain
        self.action_space=gym.spaces.Discrete(3)
        self.observation_space=gym.spaces.Box(-1.,1.,shape=(len(OBSERVATION_FIELDS),),dtype=np.float32)
        self.executor=None
        self._model_closed=True
        self._done=True

    @staticmethod
    def _validate_limit(value):
        if value is not None and (isinstance(value,bool) or not isinstance(value,int) or value<1):
            raise ValueError('max_episode_steps must be a positive integer')

    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed)
        self.action_space.seed(seed)
        options={} if options is None else dict(options)
        if set(options)-{'max_episode_steps','stop_at_drain'}:
            raise ValueError('Unsupported reset options')
        limit=options.get('max_episode_steps',self.max_episode_steps)
        stop=options.get('stop_at_drain',self.stop_at_drain)
        self._validate_limit(limit)
        if not isinstance(stop,bool): raise ValueError('stop_at_drain must be bool')
        self.close()
        self._policies=tuple(make_cars_policy(name,self.parameters,self._workload.start,self.capacity)
                             for name in ACTION_POLICIES)
        self._risk=self._policies[2]
        self.executor=LocalExecutor(self._workload,self._environment,self.capacity,
                                    self.power_factory(),self._policies[0],drain=True)
        self._model_closed=False
        until=self._environment.end
        if limit is not None: until=min(until,self._workload.start+STEP*limit)
        self.executor.start(stop_at_drain=stop,until=until)
        self._penalized=set()
        self._ci_history=deque(maxlen=5)
        self._done=False
        self._observe_boundary()
        return self._observation(),self._info()

    def _observe_boundary(self):
        now=self.executor.observed_end
        # Always observe arrivals, including steps whose action was EDF/Carbon.
        for policy in self._policies:
            if hasattr(policy,'history'):
                policy.history.observe(now,self.executor.current_arrivals)
        self._pressure=self._risk.history.pressure(now)
        if not self.executor.finished:
            self._ci_history.append(float(self.executor.sample.ci_g_per_kwh))

    def _info(self):
        ledger=self.executor.ledger.values()
        counts={state:sum(e['state']==state for e in ledger) for state in ('pending','running','completed')}
        counts['arrived']=sum(counts.values())
        return dict(time=self.executor.observed_end.isoformat(),observed_counts=counts,
                    observation_schema=RL_SCHEMA)

    def _observation(self):
        now=self.executor.observed_end
        pending=tuple(self.executor.ledger[t.job_name]['spec'] for t in self.executor.pool.pending_tasks)
        pressure=self._pressure
        norm=self.normalization
        used=tuple(math.fsum(getattr(t,key) for t in self.executor.pool.running_tasks)
                   for key in ('cores_req','gpu_req','mem_req'))
        slack=[(latest_start(s)-now).total_seconds()/60 for s in pending]
        demands=tuple(math.fsum(getattr(s,key) for s in pending)
                      for key in ('cores_req','gpu_req','mem_req'))
        work=tuple(math.fsum(getattr(s,key)*math.ceil(s.duration/15)*15 for s in pending)
                   for key in ('cores_req','gpu_req','mem_req'))
        urgent=sum(now+STEP*(1+pressure.buffer_steps)>latest_start(s) for s in pending)
        due=sum(s.sla_deadline<=now for s in pending)
        late=sum(latest_start(s)<now for s in pending)
        ci=self._ci_history[-1]
        lags=[self._ci_history[-1-i] if len(self._ci_history)>i else self._ci_history[0]
              for i in range(1,5)]
        values=[_bounded(ci,norm.ci_g_per_kwh),*[_bounded(x,norm.ci_g_per_kwh) for x in lags],
                min(len(self._ci_history)-1,4)/4,_bounded(ci-lags[0],norm.ci_g_per_kwh)]
        values += [u/c for u,c in zip(used,self.capacity)]
        values += [_bounded(len(self.executor.pool.running_tasks),norm.count_tasks),
                   _bounded(len(pending),norm.count_tasks)]
        values += [_bounded(v/(c*norm.work_minutes)) for v,c in zip(pressure.resource_work_minutes,self.capacity)]
        values += [_bounded(0. if r is None else r) for r in pressure.rho_resources]
        values += [pressure.effective_minutes/120,float(pressure.rho is None),pressure.buffer_steps/2]
        stats=(min(slack),math.fsum(slack)/len(slack),float(np.quantile(slack,.1)),max(slack)) if slack else (0.,)*4
        values += [_bounded(x,norm.slack_minutes) for x in stats]
        values += [_bounded(x,norm.count_tasks) for x in (urgent,due,late)]
        values += [_bounded(v/c) for v,c in zip(demands,self.capacity)]
        values += [_bounded(v/(c*norm.work_minutes)) for v,c in zip(work,self.capacity)]
        day=(now.hour*60+now.minute)/1440*2*math.pi
        weekday=now.weekday()/7*2*math.pi
        values += [math.sin(day),math.cos(day),math.sin(weekday),math.cos(weekday),
                   _bounded((now-self._workload.start).total_seconds()/3600,norm.elapsed_hours),
                   float(now<self._workload.end)]
        self.observation_raw=dict(ci_g_per_kwh=ci,ci_lags_g_per_kwh=lags,
            utilization=tuple(u/c for u,c in zip(used,self.capacity)),
            pending_tasks=len(pending),running_tasks=len(self.executor.pool.running_tasks),
            history_resource_work_minutes=pressure.resource_work_minutes,
            history_rho_resources=pressure.rho_resources,history_observed_tasks=pressure.observed_tasks,
            history_effective_minutes=pressure.effective_minutes,history_cold=pressure.rho is None,
            buffer_steps=pressure.buffer_steps,solo_slack_minutes=slack,
            urgent_tasks=urgent,deadline_due_tasks=due,solo_latest_start_passed_tasks=late,
            pending_resource_demand=demands,pending_resource_work_minutes=work)
        vector=np.asarray(values,dtype=np.float32)
        if not self.observation_space.contains(vector): raise RuntimeError('Invalid online observation')
        return vector

    def _reward(self,interval,truncated):
        now=self.executor.observed_end
        newly=[]
        unfinished=[]
        remaining=[0.,0.,0.]
        for task_id,entry in self.executor.ledger.items():
            spec,task=entry['spec'],entry['task']
            miss=(entry['state']=='completed' and not task.sla_met) or (
                entry['state']!='completed' and spec.sla_deadline<=now)
            if miss and task_id not in self._penalized:
                self._penalized.add(task_id);newly.append(task_id)
            if entry['state']!='completed':
                unfinished.append(task_id)
                minutes=spec.duration if entry['state']=='pending' else max(0.,(task.finish_time-now).total_seconds()/60)
                occupied=math.ceil(minutes/15)*15
                for i,v in enumerate((spec.cores_req,spec.gpu_req,spec.mem_req)):
                    remaining[i]+=v*occupied
        cfg=self.reward_config
        carbon=interval['carbon_kg']/cfg.carbon_scale_kg
        wait=interval['pending']*15/cfg.wait_scale_task_minutes
        deadline=len(newly)/cfg.deadline_scale_tasks
        remaining_work=max(v/(c*self.normalization.work_minutes) for v,c in zip(remaining,self.capacity)) if truncated else 0.
        liability=(len(unfinished)/cfg.terminal_scale_tasks+remaining_work) if truncated else 0.
        reward=-(cfg.alpha*carbon+cfg.beta*wait+cfg.gamma*deadline+cfg.terminal_weight*liability)
        return reward,dict(carbon_kg=interval['carbon_kg'],wait_task_minutes=interval['pending']*15,
            new_deadline_misses=len(newly),new_deadline_miss_ids=sorted(newly),
            carbon_normalized=carbon,wait_normalized=wait,deadline_normalized=deadline,
            unfinished_tasks=len(unfinished) if truncated else 0,
            remaining_resource_work_minutes=remaining if truncated else [0.,0.,0.],
            remaining_work_normalized=remaining_work,terminal_liability_normalized=liability,
            carbon_cost=cfg.alpha*carbon,wait_cost=cfg.beta*wait,deadline_cost=cfg.gamma*deadline,
            terminal_cost=cfg.terminal_weight*liability)

    def step(self,action):
        if self._done: raise RuntimeError('reset() is required before another interaction')
        if isinstance(action,(bool,np.bool_)) or not self.action_space.contains(action):
            raise ValueError('Action must be 0, 1 or 2')
        interval=self.executor.advance(self._policies[int(action)])
        finished=self.executor.finished
        completed=(self.executor.observed_end>=self._workload.end and
                   not (self.executor.pool.pending_tasks or self.executor.pool.running_tasks))
        terminated,truncated=bool(finished and completed),bool(finished and not completed)
        self._done=terminated or truncated
        self._observe_boundary()
        reward,terms=self._reward(interval,truncated)
        observation=self._observation()
        info=self._info()
        info.update(action=int(action),mode=ACTION_POLICIES[int(action)],interval=dict(interval),reward=terms)
        if self._done:
            info['final_summary']=dict(self.executor.result().summary)
            info['termination_reason']='all_completed_at_accounting_end' if terminated else 'episode_or_coverage_limit'
        return observation,float(reward),terminated,truncated,info

    def close(self):
        if self.executor is not None and not self._model_closed:
            model=self.executor.power_model
            closer=getattr(model,'close',None)
            if closer is not None: closer()
            self._model_closed=True
        self._done=True
