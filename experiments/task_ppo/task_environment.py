"""Autoregressive task proposals inside one unchanged physical V2 time slot.

The proposal path is recorded before the one real15-minute execution. DDF and
CG remain the original pure reference filters; no future input is loaded here.
"""
import math

from experiments.sca_ppo.completion_guard import CompletionGuard, HORIZON_MINUTE
from experiments.sca_ppo.deadline_filter import LATENESS_EPS_MINUTES
from experiments.sca_ppo.edf_calendar import STEP, edf_key, simulate
from experiments.sca_ppo.rl_environment import (
    RLEpisode, STATE_NAMES as GLOBAL_NAMES, STATE_SCHEMA as GLOBAL_SCHEMA, state_vector,
)


TASK_NAMES = (
    'task_cpu_fraction', 'task_gpu_fraction', 'task_mem_fraction', 'task_duration',
    'task_age', 'task_deadline_remaining', 'task_personal_slack', 'task_dynamic_power',
    'task_reference_remaining', 'task_grid_duration', 'task_arrival_fraction',
    'task_deadline_fraction',
)
PREFIX_NAMES = (
    'candidate_count', 'candidate_position', 'approved_candidate_fraction',
    'approved_cpu_fraction', 'approved_gpu_fraction', 'approved_mem_fraction',
    'approved_gpu_grid_hours', 'rejected_candidate_fraction',
)
STATE_NAMES = (*GLOBAL_NAMES, *TASK_NAMES, *PREFIX_NAMES, 'actor_query')
STATE_SCHEMA = {
    'version': 'task-ppo-v1-48-physical-scales-v1',
    'names': STATE_NAMES,
    'global_schema': GLOBAL_SCHEMA,
    'task_scales': {
        'resource_demand': 'CPU/GPU/MEM divided by original logical capacity',
        'duration_age_deadline_slack_reference_grid': 'minutes divided by1440',
        'arrival_and_absolute_deadline': 'minutes divided by4320',
        'power': '(.005CPU+.225GPU)/.94 divided by same expression for capacity',
    },
    'prefix_scales': {
        'candidate_count': 1000.,
        'position_approved_rejected_counts': 'divide by frozen candidate count',
        'approved_demands': 'divide by original logical CPU/GPU/MEM capacities',
        'approved_gpu_grid_hours': 'divide by logicalGPU capacity times24h',
    },
    'critic_query': 'frozen27global values followed by21zeros, including query flag0',
    'actor_query': 'frozen27global, current12task, approved8prefix, query flag1',
    'proposal_semantics': '0=no voluntary defer;1=request defer, including a DDF rejection',
    'clipping': 'only global remaining completion budget is floored, as in frozen E2',
}


def _power(cpu,gpu):
    return (.005*cpu+.225*gpu)/.94


class TaskEpisode(RLEpisode):
    """One physical episode with variable-length conditional task proposals.

    ``prepare_slot`` is idempotent and freezes its candidate set. Each call to
    ``propose`` consumes exactly one fixed EDF position without advancing the
    environment. ``execute_slot`` is the only entry that invokes real V2.step.
    Parent ``finish`` conserves full actual carbon, irreversible misses and CG.
    """

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self._slot=None

    def reset(self):
        obs=super().reset()
        self._slot=None
        return obs

    def prepare_slot(self):
        if self.obs is None or self.env.minute>=HORIZON_MINUTE:
            raise RuntimeError('Prepare requires reset and a nonterminal physical episode')
        if self._slot is None:
            snapshot=self.adapter.observe(self.obs)
            reference=simulate(snapshot)
            failure=self.adapter.guard.failure_latched or not CompletionGuard.accepts(reference)
            global_state=tuple(state_vector(self.obs,reference,failure))
            tasks={task.task_id:task for task in snapshot.pending}
            candidates=(() if failure else tuple(sorted(reference.first_slot.started,
                key=lambda identity:edf_key(tasks[identity],snapshot.sla))))
            self._slot=dict(snapshot=snapshot,reference=reference,tasks=tasks,
                global_state=global_state,value_state=(*global_state,*([0.]*21)),
                candidates=candidates,index=0,approved=[],proposed=[],decisions=[],
                internal_candidate_checks=0,internal_calendar_plans=1,
                reference_on_time=reference.on_time_ids,reference_late=reference.late_ids,
                reference_late_total=reference.total_lateness(reference.late_ids))
        if self.env.minute!=self._slot['snapshot'].minute:
            raise AssertionError('Physical time changed during conditional proposal sequence')
        return list(self._slot['value_state']),self._slot['candidates']

    def next_state(self):
        self.prepare_slot()
        slot=self._slot;index=slot['index'];candidates=slot['candidates']
        if index>=len(candidates):
            raise RuntimeError('All frozen task candidates are already processed')
        task=slot['tasks'][candidates[index]]
        snapshot=slot['snapshot'];now=snapshot.minute;capacity=snapshot.capacity
        deadline=edf_key(task,snapshot.sla)[0]
        grid_minutes=STEP*math.ceil(task.duration_minutes/STEP-1e-12)
        task_values=[*(demand/cap for demand,cap in zip(task.demand,capacity)),
            task.duration_minutes/1440.,(now-task.arrival_minute)/1440.,
            (deadline-now)/1440.,(deadline-now-task.duration_minutes)/1440.,
            _power(task.demand[0],task.demand[1])/_power(capacity[0],capacity[1]),
            (slot['reference'].by_id[task.task_id].finish-now)/1440.,grid_minutes/1440.,
            task.arrival_minute/4320.,deadline/4320.]
        approved=[slot['tasks'][identity] for identity in slot['approved']]
        count=len(candidates)
        rejected=sum(row['action']==1 and not row['approved'] for row in slot['decisions'])
        prefix_values=[count/1000.,index/count,len(approved)/count,
            *(math.fsum(task.demand[axis] for task in approved)/capacity[axis] for axis in range(3)),
            math.fsum(math.ceil(task.duration_minutes/STEP-1e-12)*.25*task.demand[1]
                      for task in approved)/(capacity[1]*24.),rejected/count]
        values=[*slot['global_state'],*task_values,*prefix_values,1.]
        if len(values)!=48 or not all(math.isfinite(value) for value in values):
            raise ValueError('Actor task observation must contain48 finite current-state scalars')
        return [float(value) for value in values]

    def propose(self,action_id):
        if type(action_id) is not int or action_id not in (0,1):
            raise ValueError('Task proposal must be integer0 or1')
        state=self.next_state()
        slot=self._slot;identity=slot['candidates'][slot['index']]
        old_prefix=tuple(slot['approved']);approved=False;reasons=();lost=()
        if action_id:
            slot['proposed'].append(identity)
            # The previous prefix was already checked against this identical
            # immutable reference. Check only its new joint extension here;
            # execute_slot still invokes original DDF over every original1.
            trial=simulate(slot['snapshot'],frozenset((*old_prefix,identity)))
            slot['internal_candidate_checks']+=1;slot['internal_calendar_plans']+=1
            lost=tuple(sorted(slot['reference_on_time']-trial.on_time_ids))
            trial_late=trial.total_lateness(slot['reference_late'])
            reasons=[]
            if lost:reasons.append('REFERENCE_ON_TIME_SET_LOST')
            if trial_late>slot['reference_late_total']+LATENESS_EPS_MINUTES:
                reasons.append('ORIGINAL_LATE_TOTAL_INCREASED')
            if not CompletionGuard.accepts(trial):reasons.append('COMPLETION_GUARD_REJECTED')
            reasons=tuple(reasons);approved=not reasons
            if approved:
                assert len(trial.late_ids)<=len(slot['reference_late'])
            if approved:slot['approved'].append(identity)
        record=dict(task_id=identity,state=state,action=action_id,approved=approved,
                    reasons=tuple(reasons),lost_on_time_ids=tuple(lost),
                    approved_prefix_before=old_prefix,approved_prefix_after=tuple(slot['approved']))
        slot['decisions'].append(record);slot['index']+=1
        return dict(record)

    def execute_slot(self):
        self.prepare_slot()
        slot=self._slot
        if slot['index']!=len(slot['candidates']):
            raise RuntimeError('Finish every frozen task proposal before one physical execution')
        decision=self.adapter.decide(self.obs,proposed=tuple(slot['proposed']))
        if decision.snapshot!=slot['snapshot'] or decision.reference.records!=slot['reference'].records:
            raise AssertionError('Real current state/reference changed during the joint proposal')
        if decision.filter.approved!=tuple(slot['approved']):
            raise AssertionError('Complete proposal replay differs from cumulative DDF prefix')
        request_records=[row for row in slot['decisions'] if row['action']==1]
        if len(request_records)!=len(decision.filter.checks):
            raise AssertionError('Final DDF must count each original request exactly once')
        for row,check in zip(request_records,decision.filter.checks):
            if (row['task_id'],row['approved'],row['reasons'],row['lost_on_time_ids'])!=(
                    check.task_id,check.approved,check.reasons,check.lost_on_time_ids):
                raise AssertionError('Final DDF revalidation changed a conditional request result')
            if check.trial_defer!=(*row['approved_prefix_before'],row['task_id']):
                raise AssertionError('Final DDF used a different joint defer prefix')
        obs,done,actual=self.adapter.execute(self.env,decision)
        idle=self.idle_result['steps'][len(self.records)]
        if idle['minute']!=actual['minute']:
            raise AssertionError('Actual and public idle reward slots are misaligned')
        terms=self.ledger.add(actual['carbon_kg'],idle['carbon_kg'],
                              self.adapter.records[-1]['new_irreversible_miss_ids'])
        record=dict(self.adapter.records[-1])
        record.update(terms,policy_sampled=bool(slot['candidates']),state=list(slot['value_state']),
            candidate_ids=slot['candidates'],task_decisions=tuple(slot['decisions']),
            row=dict(actual),idle_carbon_kg=idle['carbon_kg'],
            incremental_carbon_kg=actual['carbon_kg']-idle['carbon_kg'],
            internal_candidate_checks=slot['internal_candidate_checks'],
            internal_calendar_plans=slot['internal_calendar_plans']+len(decision.filter.checks)+
                (1 if decision.filter.reference_guard_failed else 2))
        self.records.append(record);self.obs=obs;self._slot=None;self._prepared=None
        return terms['reward'],done,record
