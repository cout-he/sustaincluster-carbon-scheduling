"""Independent CGD-RL interface: threshold actions over the sole E1 kernel."""
import math
import gymnasium as gym
import numpy as np
from .cars_state import ResourceSnapshot, RunningTask, demand
from .carbon_gain_policy import CarbonGainEDF
from .rl_env import CarbonSchedulingEnv, OBSERVATION_FIELDS as BASE_FIELDS, _bounded
from .contracts import finite

RL_SCHEMA='e1-7b-cgd-rl-v1'
GAIN_FIELDS=('forecast_ci_1h','forecast_ci_2h','forecast_ci_trend','gain_mean','gain_p25','gain_p50','gain_p75',
    'positive_count','moderate_count','invalid_fraction','guard_allowed_fraction','gain_candidates',
    'forecast_cold','gain_present')
OBSERVATION_FIELDS=BASE_FIELDS+GAIN_FIELDS

class CarbonGainSchedulingEnv(gym.Env):
    """0 EDF immediate, 1 M2 gain>Train-P75, 2 M2 gain>0.

    Actions 1/2 share guard strength; threshold changes only carbon eligibility.
    Actual reward, completion/terminal debt and physics come from the legacy
    environment and one persistent LocalExecutor. Added features use frozen
    physical scales: CI1000, counts1000, gain.1kg; signed x/(scale+abs(x)).
    """
    metadata={'render_modes':[]};render_mode=None
    def __init__(self,workload,environment,capacity,power_factory,parameters,model,*,
                 prehistory=(),moderate_threshold_kg=None,forecast_method='seasonal',reward_config=None,
                 normalization=None,max_episode_steps=None,stop_at_drain=False):
        super().__init__()
        self.inner=CarbonSchedulingEnv(workload,environment,capacity,power_factory,parameters,
            reward_config=reward_config,normalization=normalization,max_episode_steps=max_episode_steps,
            stop_at_drain=stop_at_drain)
        self.model,self.prehistory=model,tuple(prehistory)
        self.moderate_threshold_kg=(None if moderate_threshold_kg is None else finite(moderate_threshold_kg,'moderate threshold'))
        self.forecast_method=forecast_method
        self.action_space=gym.spaces.Discrete(3)
        self.observation_space=gym.spaces.Box(-1.,1.,shape=(len(OBSERVATION_FIELDS),),dtype=np.float32)
        self.policy=None;self._done=True

    @property
    def executor(self):return self.inner.executor

    def _snapshot(self):
        ex=self.executor
        pending=self.policy.order(tuple(ex.ledger[t.job_name]['spec'] for t in ex.pool.pending_tasks))
        running=tuple(RunningTask(t.job_name,demand(ex.ledger[t.job_name]['spec']),t.finish_time,
            ex.ledger[t.job_name]['spec'].sla_deadline) for t in ex.pool.running_tasks)
        available=(ex.pool.available_cores,ex.pool.available_gpus,ex.pool.available_mem)
        return ResourceSnapshot(ex.observed_end,self.inner.capacity,available,pending,running,ex.current_arrivals)

    def _observation(self,base=None):
        base=self.inner._observation() if base is None else base
        if self.executor.finished:
            block=np.zeros(len(GAIN_FIELDS));raw=dict(terminal=True,candidates=[])
        else:
            snapshot=self._snapshot();rows=self.policy.preview(snapshot,self.executor.sample)
            f=self.policy.forecaster.forecast(snapshot.now,9)
            usable=[r['gain_total_kg'] for r in rows if r['status']=='ok']
            fit=[r for r in rows if r['can_fit']];n=len(fit)
            stats=(math.fsum(usable)/len(usable),*np.quantile(usable,(.25,.5,.75))) if usable else (0.,)*4
            threshold=self.moderate_threshold_kg
            values=[_bounded(math.fsum(f.ci[1:5])/4,1000),_bounded(math.fsum(f.ci[1:9])/8,1000),
                _bounded(f.ci[8]-f.ci[0],1000),*[_bounded(float(v),.1) for v in stats],
                _bounded(sum(g>0 for g in usable),1000),
                _bounded(sum(g>threshold for g in usable) if threshold is not None else 0,1000),
                sum(r['status'] not in ('ok','deadline_guard') for r in fit)/n if n else 0.,
                sum(r['guard_allowed'] for r in fit)/n if n else 0.,
                _bounded(len(usable),1000),float(f.cold),float(bool(usable))]
            block=np.asarray(values)
            # Online info exposes only known candidates and causal predictions.
            raw=dict(terminal=False,candidates=rows,forecast_ci=list(f.ci),moderate_threshold_kg=threshold,
                preview='before EDF pass; later real starts update actual decision background')
        self.observation_raw=dict(self.inner.observation_raw,carbon_gain=raw)
        obs=np.concatenate((base,block)).astype(np.float32,copy=False)
        if not self.observation_space.contains(obs):raise RuntimeError('Invalid CGD observation')
        return obs

    def _info(self):
        info=self.inner._info();info.update(observation_schema=RL_SCHEMA,moderate_threshold_kg=self.moderate_threshold_kg)
        return info

    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed);self.action_space.seed(seed)
        base,_=self.inner.reset(seed=seed,options=options)
        self.policy=CarbonGainEDF('m2',self.inner._workload.start,self.inner.capacity,self.model,
            prehistory=self.prehistory,forecast_method=self.forecast_method,moderate_threshold_kg=self.moderate_threshold_kg)
        self._done=False
        return self._observation(base),self._info()

    def step(self,action):
        if self._done:raise RuntimeError('reset required')
        if isinstance(action,(bool,np.bool_)) or not self.action_space.contains(action):raise ValueError('Action must be 0/1/2')
        action=int(action);self.policy.set_action(action);first=len(self.policy.diagnostics)
        interval=self.executor.advance(self.policy)
        completed=(self.executor.observed_end>=self.inner._workload.end and
                   not(self.executor.pool.pending_tasks or self.executor.pool.running_tasks))
        terminated=bool(self.executor.finished and completed);truncated=bool(self.executor.finished and not completed)
        self._done=self.inner._done=terminated or truncated
        self.inner._observe_boundary()
        reward,terms=self.inner._reward(interval,truncated)
        obs=self._observation();diagnostics=self.policy.diagnostics[first:]
        # Wallclock is useful offline profiling, never policy/observation or gym determinism info.
        clean=[{k:v for k,v in r.items() if k not in ('elapsed_seconds','decision_seconds')} for r in diagnostics]
        info=self._info();info.update(action=action,interval=dict(interval),reward=terms,policy_diagnostics=clean)
        if self._done:
            info['final_summary']=dict(self.executor.result().summary)
            info['termination_reason']='all_completed_at_accounting_end' if terminated else 'episode_or_coverage_limit'
        return obs,float(reward),terminated,truncated,info

    def close(self):self.inner.close();self._done=True
