"""CARS-v1 ablations. Online eligibility only; shared executor owns resources."""
from collections import Counter
from dataclasses import asdict

from .contracts import STEP, finite
from .policies import (POLICY_NAMES, CarbonThresholdFIFO, EDFBackfill, ImmediateFIFO,
                       PolicyDecision, latest_start)
from .cars_state import ArrivalHistory, ResourceSnapshot

CARS_SCHEMA='e1-3-cars-v1'
CARS_POLICY_NAMES=POLICY_NAMES+('edf_carbon_threshold','cars_known_only','cars_risk_only','cars_full')


class CarsPolicy(EDFBackfill):
    def __init__(self,name,parameters,experiment_start,capacity):
        if name not in CARS_POLICY_NAMES[3:]: raise ValueError('Unknown CARS policy')
        self.name=name
        self.threshold_g_per_kwh=finite(parameters['threshold_g_per_kwh'],'carbon threshold')
        self.known=name in ('cars_known_only','cars_full')
        self.risk=name in ('cars_risk_only','cars_full')
        self.history=ArrivalHistory(experiment_start,capacity,parameters.get('history_minutes',120),
            parameters.get('risk_breakpoints',(.5,1)),parameters.get('risk_buffer_steps',(0,1,2)),
            parameters.get('cold_start_buffer_steps',2))
        if self.known:
            from .cars_feasibility import ShadowPlanner
            self.planner=ShadowPlanner(parameters.get('horizon_steps',8),parameters.get('max_pending',256))
        self.diagnostics=[]
        self.counters=dict(candidate_checks=0,planner_calls=0,conservative_fallbacks=0,
                           known_risk_guards=0,risk_buffer_guards=0)
        self.reason_counts=Counter()

    def on_step_start(self,snapshot):
        self.snapshot=snapshot
        # dict insertion order is the shared EDF order, reused throughout this step.
        self.pending={s.task_id:s for s in snapshot.pending}
        self.running=list(snapshot.running);self.blocked={}
        self.available=snapshot.available
        self.history.observe(snapshot.now,snapshot.arrivals)
        self.pressure=self.history.pressure(snapshot.now)

    def on_candidate_resources(self,now,available):
        if now!=self.snapshot.now: raise ValueError('Stale policy state')
        self.available=tuple(available)

    def on_task_started(self,running):
        if running.task_id not in self.pending: raise ValueError('Unknown start notification')
        del self.pending[running.task_id]
        self.running.append(running)

    def on_task_deferred(self,task_id,until):
        self.blocked[task_id]=until

    def decide(self,spec,context):
        if spec.task_id not in self.pending or context.now!=self.snapshot.now:
            raise ValueError('Candidate missing from current observed state')
        self.counters['candidate_checks']+=1
        pressure=self.pressure
        diag=dict(time=context.now.isoformat(),task_id=spec.task_id,policy=self.name,
            ci_g_per_kwh=context.ci_g_per_kwh,latest_start=latest_start(spec).isoformat(),
            latest_grid_start=latest_start(spec).floor('15min').isoformat(),
            available=list(self.available),pending_count=len(self.pending),running_count=len(self.running),
            risk_enabled=self.risk,known_enabled=self.known,**asdict(pressure))
        if context.ci_g_per_kwh<=self.threshold_g_per_kwh:
            decision=PolicyDecision(False,'low_ci_execute')
        elif context.now+STEP>latest_start(spec):
            decision=PolicyDecision(False,'no_grid_slack')
        elif self.risk and context.now+STEP*(1+pressure.buffer_steps)>latest_start(spec):
            self.counters['risk_buffer_guards']+=1
            decision=PolicyDecision(False,'risk_buffer_guard')
        else:
            decision=PolicyDecision(True,'high_ci_defer')
            if self.known:
                snapshot=ResourceSnapshot(context.now,self.snapshot.capacity,self.available,
                    tuple(self.pending.values()),tuple(self.running),blocked_until=tuple(self.blocked.items()))
                result=self.planner.compare(spec,snapshot)
                self.counters['planner_calls']+=result.planner_calls
                planner=asdict(result)
                for key in ('candidate_start_now','candidate_start_defer','horizon_end'):
                    if planner[key] is not None: planner[key]=planner[key].isoformat()
                diag['planner']=planner
                if result.status=='uncertain':
                    self.counters['conservative_fallbacks']+=1
                    decision=PolicyDecision(False,'known_feasibility_uncertain')
                elif result.status=='risk':
                    self.counters['known_risk_guards']+=1
                    decision=PolicyDecision(False,'known_deadline_risk')
        diag.update(defer=decision.defer,reason=decision.reason)
        self.reason_counts[decision.reason]+=1
        self.diagnostics.append(diag)
        return decision


def make_cars_policy(name,parameters,experiment_start,capacity):
    """Independent registry; old policy classes and schema remain unchanged."""
    if name=='immediate_fifo': return ImmediateFIFO()
    if name=='edf_backfill': return EDFBackfill()
    if name=='carbon_threshold_fifo': return CarbonThresholdFIFO(parameters['threshold_g_per_kwh'])
    return CarsPolicy(name,parameters,experiment_start,capacity)
