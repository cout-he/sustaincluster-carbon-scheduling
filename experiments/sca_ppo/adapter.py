"""Observation-only policy adapter; execution verification reads arrived IDs only.

No scenario loader, future task registry, future CI or weather access. Reset
this adapter together with V2. It does not own or alter frozen environment code.
"""
from dataclasses import dataclass

from experiments.platform_v1.environment import Action
from .edf_calendar import Snapshot, Running, simulate, edf_key, grid_release, DEADLINE_EPS
from .deadline_filter import filter_requests, ranked_request
from .completion_guard import CompletionGuard, CLEAR_MINUTE, HORIZON_MINUTE, FAILURE


@dataclass(frozen=True)
class Decision:
    snapshot: Snapshot
    reference: object
    filter: object
    proposed: tuple
    action: Action
    reference_cg_pass: bool
    failure_latched: bool


class ResearchAdapter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.known = {}
        self.guard = CompletionGuard()
        self.records = []
        self.irreversible_miss_ids = set()

    def observe(self, obs):
        now = obs['minute']
        for t in obs['pending']:
            if t.arrival_minute > now: raise ValueError('Future task in pending observation')
            if t.task_id in self.known and self.known[t.task_id] != t:
                raise ValueError('Observed task identity changed')
            self.known[t.task_id] = t
        running = []
        for identity, demand, finish in obs['running']:
            if identity not in self.known:
                raise ValueError('Reset adapter with env; running identity has never been observed')
            t = self.known[identity]
            if tuple(demand) != tuple(t.demand): raise ValueError('Running demand changed')
            running.append(Running(t, finish))
        return Snapshot(now, tuple(obs['capacity']), obs['sla_multiplier'],
                        tuple(obs['pending']), tuple(running))

    def decide(self, obs, action_id=0, proposed=None):
        s = self.observe(obs)
        reference = simulate(s)
        cg_pass = self.guard.inspect_reference(reference)
        request = ranked_request(s, reference, action_id) if proposed is None else tuple(proposed)
        if len(set(request)) != len(request) or not set(request) <= set(reference.first_slot.started):
            raise ValueError('Request must be a fixed unique prefix of reference starters')
        # Retain proposed IDs for diagnostics, but never approve after a latch.
        result = filter_requests(s, reference, () if self.guard.failure_latched else request)
        order = tuple(t.task_id for t in sorted(s.pending, key=lambda t: edf_key(t, s.sla)))
        action = Action(order, frozenset(result.approved), 20.0)
        return Decision(s, reference, result, request, action, cg_pass, self.guard.failure_latched)

    def execute(self, env, decision):
        if env.horizon != HORIZON_MINUTE:
            raise ValueError('Research protocol requires unmodified 72-hour responsibility window')
        current = self.observe(env.observation())
        if current != decision.snapshot:
            raise AssertionError('Stale decision: real V2 state differs from calendar snapshot')
        pending = decision.action.order
        before = {i: (env.ledger[i]['policy_wait_minutes'], env.ledger[i]['capacity_wait_minutes'])
                  for i in pending}
        obs, done, row = env.step(decision.action)
        started = tuple(i for i in pending if env.ledger[i]['start_minute'] == current.minute)
        policy = tuple(i for i in pending if env.ledger[i]['policy_wait_minutes'] > before[i][0])
        capacity = tuple(i for i in pending if env.ledger[i]['capacity_wait_minutes'] > before[i][1])
        preview = decision.filter.preview
        predicted = decision.filter.plan.by_id
        try:
            if (started, policy, capacity) != (preview.started, preview.policy_wait, preview.capacity_wait):
                raise AssertionError('V2 actual per-ID start/wait differs from final joint preview')
            if tuple(row['used_'+a] for a in ('cpu', 'gpu', 'mem')) != preview.used:
                raise AssertionError('V2 used resources differ from preview')
            if (row['started'] != len(started) or row['policy_wait_tasks'] != len(policy)
                    or row['capacity_wait_tasks'] != len(capacity)):
                raise AssertionError('V2 row counts disagree with per-ID ledger deltas')
            for axis, name in enumerate(('cpu', 'gpu', 'mem')):
                if row['blocked_'+name] != sum(axis in axes for _, axes in preview.blocked_resources):
                    raise AssertionError('V2 resource bottlenecks differ from preview')
            for i in started:
                entry = env.ledger[i]; p = predicted[i]
                if (entry['start_minute'], entry['finish_minute'], entry['release_minute']) != (p.start, p.finish, p.release):
                    raise AssertionError('V2 raw finish or grid release differs from preview')
        except AssertionError as exc:
            # Preserve evidence after the real step; never silently repair or retry.
            self.records.append(dict(minute=current.minute, error_code='PREVIEW_EXECUTION_MISMATCH',
                error=str(exc), proposed=decision.proposed, approved=decision.filter.approved,
                expected_preview=preview, effective_policy_wait=policy, capacity_wait=capacity,
                actual_started=started, actual_row=dict(row)))
            raise
        # Step-end irreversible evidence only; EDF predictions are never events.
        tau = env.minute
        new_miss = []
        for i, t in self.known.items():
            entry = env.ledger[i]
            deadline = t.arrival_minute + current.sla*t.duration_minutes
            bound = entry['finish_minute'] if entry['finish_minute'] is not None else tau+t.duration_minutes
            if bound > deadline + DEADLINE_EPS and i not in self.irreversible_miss_ids:
                new_miss.append(i); self.irreversible_miss_ids.add(i)
        record = dict(minute=current.minute, proposed=decision.proposed,
                      approved=decision.filter.approved, effective_policy_wait=policy,
                      capacity_wait=capacity, actual_started=started,
                      reference_cg_pass=decision.reference_cg_pass,
                      failure_latched=decision.failure_latched,
                      failure_code=FAILURE if decision.failure_latched else None,
                      new_irreversible_miss_ids=tuple(sorted(new_miss)),
                      checks=decision.filter.checks)
        self.records.append(record)
        if not done: self.observe(obs)  # register ONLY tasks now exposed by V2
        return obs, done, row

    def summary(self, env):
        if env.minute != HORIZON_MINUTE: raise RuntimeError('Complete all 288 slots before summary')
        entries = [env.ledger[i] for i in self.known]
        unfinished = sum(e['state'] != 'completed' for e in entries)
        releases = [e['release_minute'] for e in entries if e['state'] == 'completed']
        tail = [r for r in env.steps if r['minute'] >= CLEAR_MINUTE]
        tail_valid = (len(tail) == 24 and all(r['pending'] == 0 and r['running'] == 0
                       and r['supply_requested_c'] == 20 for r in tail))
        if not unfinished:
            official = {i for i, t in self.known.items()
                        if env.ledger[i]['finish_minute'] > t.arrival_minute+env.sla*t.duration_minutes+DEADLINE_EPS}
            assert self.irreversible_miss_ids == official, 'Per-ID irreversible events disagree with complete V2 ledger'
        return dict(failure_code=FAILURE if self.guard.failure_latched else None,
                    failure_latched=self.guard.failure_latched,
                    reference_guard_failure_count=self.guard.reference_guard_failure_count,
                    first_failure_minute=self.guard.first_failure_minute,
                    actual_last_release_minute=None if unfinished else max(releases, default=0),
                    actual_unfinished_count=unfinished,
                    actual_common_idle_tail_valid=tail_valid,
                    irreversible_miss_ids=tuple(sorted(self.irreversible_miss_ids)))
