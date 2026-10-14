"""EDF/backfill eligibility using M0/M1/M2 with one common deadline guard."""
import math
import time
from collections import Counter
from dataclasses import replace
from .carbon_gain import CarbonGainEstimator
from .cars_state import ResourceSnapshot, demand
from .contracts import STEP, finite, utc_time
from .deadline_guard import DeadlineGuard
from .online_ci_forecast import OnlineCIForecast
from .policies import EDFBackfill, PolicyDecision

VERSION='e1-7b-carbon-gain-policy-v1'

class CarbonGainEDF(EDFBackfill):
    def __init__(self,mode,start,capacity,model,*,prehistory=(),forecast_method='seasonal',
                 gain_threshold_kg=0.,moderate_threshold_kg=None,guard_enabled=True):
        if mode not in ('m0','m1','m2'):raise ValueError('Unknown gain mode')
        if tuple(capacity)!=model.capacity:raise ValueError('Proxy and executor capacity differ')
        if any(utc_time(r['time'],True)>=utc_time(start,True) for r in prehistory):
            raise ValueError('Prehistory must precede first decision')
        self.mode=mode;self.name='cgd_'+mode+'_'+forecast_method
        self.threshold_g_per_kwh=161.075
        self.gain_threshold_kg=finite(gain_threshold_kg,'gain threshold')
        self.moderate_threshold_kg=(None if moderate_threshold_kg is None else finite(moderate_threshold_kg,'moderate gain'))
        self.guard=DeadlineGuard(start,capacity,guard_enabled);self.history=self.guard.history
        self.forecaster=OnlineCIForecast(forecast_method,prehistory)
        self.estimator=CarbonGainEstimator(model,self.forecaster)
        self.past_hvac_kw=0.;self.completed_interval_end=None
        self.diagnostics=[];self.reason_counts=Counter();self.action=None
        self.counters=Counter();self.current_sample=None

    def set_action(self,action):
        if isinstance(action,bool) or action not in (0,1,2):raise ValueError('CGD action must be 0/1/2')
        self.action=int(action)

    def on_step_environment(self,sample):
        self.current_sample=sample
        self.forecaster.observe(sample.time,sample.ci_g_per_kwh)

    def on_interval_completed(self,record):
        end=utc_time(record['interval_end'],True)
        if self.completed_interval_end is not None and end<=self.completed_interval_end:
            raise ValueError('Completed power must advance once per interval')
        self.completed_interval_end=end;self.past_hvac_kw=finite(record['hvac_power_kw'],'observed past HVAC')

    def on_step_start(self,snapshot):
        if self.current_sample is None or self.current_sample.time!=snapshot.now:
            raise ValueError('Current environment must be observed before candidate decisions')
        if self.completed_interval_end is not None and self.completed_interval_end>snapshot.now:
            raise ValueError('Cannot access current unfinished interval power')
        self.snapshot=snapshot;self.running=list(snapshot.running)
        self.pending={s.task_id:s for s in snapshot.pending};self.available=snapshot.available
        self.guard.observe(snapshot)

    def on_candidate_resources(self,now,available):
        if now!=self.snapshot.now:raise ValueError('Stale candidate snapshot')
        self.available=tuple(available)

    def on_task_started(self,running):
        del self.pending[running.task_id];self.running.append(running)

    def on_task_deferred(self,task_id,until):
        if task_id not in self.pending or until!=self.snapshot.now+STEP:raise ValueError('Only defer one step')

    def _estimate(self,spec):
        state=ResourceSnapshot(self.snapshot.now,self.snapshot.capacity,self.available,
            tuple(self.pending.values()),tuple(self.running))
        return self.estimator.estimate(spec,state,self.current_sample.temperature_c,self.past_hvac_kw)

    def preview(self,snapshot,sample):
        """Current-known candidates, no hypothetical allocations or actual actions.

        Snapshot represents BEFORE the EDF pass. Sequential real decisions may
        have different background after preceding real starts; this is logged.
        """
        self.on_step_environment(sample);self.on_step_start(snapshot)
        rows=[]
        used=tuple(math.fsum(r.demand[k] for r in snapshot.running) for k in range(3))
        for s in snapshot.pending:
            can_fit=all(math.fsum((u,q))<=c for u,q,c in zip(used,demand(s),snapshot.capacity))
            guard=self.guard.check(s,snapshot.now)
            if can_fit and guard['allowed']:
                estimate=self._estimate(s)
            else:estimate=dict(status='capacity_unavailable' if not can_fit else 'deadline_guard',
                gain_it_kg=None,gain_hvac_kg=None,gain_total_kg=None,forecast_cold=False)
            rows.append(dict(task_id=s.task_id,can_fit=can_fit,guard_allowed=guard['allowed'],**estimate))
        return rows

    def decide(self,spec,context):
        clock=time.perf_counter()
        if context.now!=self.snapshot.now or spec.task_id not in self.pending:raise ValueError('Unobserved candidate')
        self.counters['candidate_checks']+=1
        guard=self.guard.check(spec,context.now)
        threshold=(self.moderate_threshold_kg if self.action==1 else 0.) if self.action is not None else self.gain_threshold_kg
        estimate=dict(status='not_evaluated',gain_it_kg=None,gain_hvac_kg=None,gain_total_kg=None,
                      forecast_cold=False,forecast_fallback_count=0,elapsed_seconds=0.)
        if self.action==0:
            decision=PolicyDecision(False,'execute_now')
        elif not guard['allowed']:
            decision=PolicyDecision(False,guard['reason'])
        elif self.mode=='m0':
            decision=PolicyDecision(context.ci_g_per_kwh>self.threshold_g_per_kwh,
                'carbon_threshold_high' if context.ci_g_per_kwh>self.threshold_g_per_kwh else 'carbon_gain_insufficient')
        elif self.action==1 and threshold is None:
            decision=PolicyDecision(False,'moderate_threshold_unavailable')
        else:
            estimate=self._estimate(spec);self.counters['prediction_calls']+=1
            if estimate['status']!='ok':
                self.counters['estimate_unavailable']+=1
                decision=PolicyDecision(False,estimate['status'])
            else:
                gain=estimate['gain_it_kg'] if self.mode=='m1' else estimate['gain_total_kg']
                allowed=gain>threshold
                decision=PolicyDecision(allowed,'carbon_gain_positive' if allowed else 'carbon_gain_insufficient')
        self.reason_counts[decision.reason]+=1
        self.diagnostics.append(dict(time=context.now.isoformat(),task_id=spec.task_id,mode=self.mode,action=self.action,
            ci_g_per_kwh=context.ci_g_per_kwh,duration_minutes=spec.duration,demand=list(demand(spec)),
            deadline=spec.sla_deadline.isoformat(),threshold_kg=threshold,
            guard_reason=guard['reason'],**{k:v for k,v in guard.items() if k!='reason'},**estimate,
            running_count=len(self.running),available=list(self.available),
            completed_interval_end=self.completed_interval_end.isoformat() if self.completed_interval_end is not None else None,
            defer=decision.defer,reason=decision.reason,decision_seconds=time.perf_counter()-clock))
        return decision
