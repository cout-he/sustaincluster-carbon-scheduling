"""Artificial jobs only. No workload/data loader is called."""
import copy
import math
import random
import unittest
from dataclasses import replace

from experiments.platform_v1.data import Task, Observation, Scenario
from experiments.platform_v1.environment import Action
from experiments.platform_v2.environment import PlatformEnv
from experiments.platform_v2.inventory import load_config, build_inventory
from experiments.sca_ppo.edf_calendar import Snapshot, Running, simulate, grid_release
from experiments.sca_ppo.deadline_filter import filter_requests, ranked_request
from experiments.sca_ppo.completion_guard import CompletionGuard, FAILURE
from experiments.sca_ppo.adapter import ResearchAdapter


def task(name, duration, demand=(1, 0, 0), arrival=0):
    return Task(name, arrival, duration, demand, 'ARTIFICIAL', 0, 0)


def snapshot(tasks, capacity=(2, 2, 2), sla=2, minute=0, running=()):
    return Snapshot(minute, capacity, sla, tuple(tasks), tuple(running))


def environment(tasks, sla=2):
    cfg = load_config()
    sc = Scenario('E0_ARTIFICIAL', tuple(tasks), tuple(
        Observation(15*n, 400, 25, (15*n//60)*60) for n in range(288)),
        {'synthetic': True, 'units': 'minutes/CPU-equivalent/GPU-equivalent/GB'}, 1440)
    return PlatformEnv(sc, build_inventory(cfg, 'OM'), cfg, sla, 'fixed')


class CalendarTests(unittest.TestCase):
    def test_raw_finish_and_grid_release(self):
        p = simulate(snapshot([task('a', 16)], sla=1))
        a = p.by_id['a']
        self.assertEqual((a.start, a.finish, a.release), (0, 16, 30))
        self.assertIn('a', p.on_time_ids)
        self.assertEqual(grid_release(30), 30)
        self.assertEqual(grid_release(30 + 1e-8), 45)

    def test_each_resource_and_backfill(self):
        for axis in range(3):
            d = [0, 0, 0]; d[axis] = 2
            small = [0, 0, 0]; small[(axis+1) % 3] = 1
            jobs = [task('a', 15, tuple(d)), task('b', 30, tuple(d)), task('c', 45, tuple(small))]
            p = simulate(snapshot(jobs))
            self.assertEqual(p.first_slot.started, ('a', 'c'))
            self.assertEqual(p.first_slot.capacity_wait, ('b',))
            self.assertEqual(dict(p.first_slot.blocked_resources)['b'], (axis,))

    def test_running_grid_padding_blocks_capacity(self):
        a = task('a', 16, (2, 0, 0))
        p = simulate(snapshot([task('b', 15, (2, 0, 0), 15)], minute=15,
                              running=(Running(a, 16),)))
        self.assertEqual(p.by_id['b'].start, 30)
        self.assertEqual(p.first_slot.capacity_wait, ('b',))

    def test_pure_repeated_calls_and_invalid_input(self):
        s = snapshot([task('a', 30), task('b', 20)])
        before = copy.deepcopy(s)
        p, q = simulate(s), simulate(s)
        self.assertEqual(s, before)
        self.assertEqual(p.records, q.records)
        with self.assertRaises(ValueError): simulate(snapshot([task('x', 1, (3, 0, 0))]))
        with self.assertRaises(ValueError): simulate(s, frozenset({'missing'}))
        with self.assertRaises(ValueError): simulate(snapshot([task('x', math.nan)]))


class FilterTests(unittest.TestCase):
    def test_power_and_slack_rankings_distinguish_unequal_scores(self):
        s = snapshot([task('cpu', 60, (2, 0, 0)), task('gpu', 15, (0, 1, 0)),
                      task('long', 90)], capacity=(3, 1, 1))
        ref = simulate(s)
        self.assertEqual(ranked_request(s, ref, 2), ('gpu', 'cpu', 'long'))
        self.assertEqual(ranked_request(s, ref, 5), ('long', 'cpu', 'gpu'))

    def test_sixteen_candidates_are_checked_without_action_mask(self):
        s = snapshot([task(f'a{i:02}', 120) for i in range(20)], capacity=(20, 1, 1))
        ref = simulate(s)
        request = ranked_request(s, ref, 3)
        decision = filter_requests(s, ref, request)
        self.assertEqual(len(decision.checks), 16)
        self.assertEqual(decision.approved, request)
        self.assertEqual(len(decision.preview.started), 4)

    def test_self_safe_but_other_known_job_late_is_rejected(self):
        s = snapshot([task('c', 20, (2, 0, 0)), task('a', 30, (4, 0, 0)),
                      task('b', 30, (2, 0, 0))], capacity=(4, 1, 1))
        ref = simulate(s)
        trial = simulate(s, frozenset({'b'}))
        self.assertEqual((ref.by_id['a'].finish, trial.by_id['a'].finish), (60, 75))
        self.assertIn('b', trial.on_time_ids)
        d = filter_requests(s, ref, ('b',))
        self.assertEqual(d.approved, ())
        self.assertEqual(d.checks[0].lost_on_time_ids, ('a',))

    def test_individually_safe_but_joint_defer_conflicts(self):
        s = snapshot([task('a', 15), task('b', 15), task('c', 30, (4, 0, 0)),
                      task('d', 45, (2, 0, 0))], capacity=(4, 1, 1))
        ref = simulate(s)
        self.assertEqual(ref.late_ids, frozenset({'c'}))
        self.assertEqual(ref.total_lateness(ref.late_ids), 15)
        for identity in ('a', 'b'):
            self.assertEqual(filter_requests(s, ref, (identity,)).approved, (identity,))
        joint = filter_requests(s, ref, ('a', 'b'))
        self.assertEqual(joint.approved, ('a',))
        self.assertEqual(joint.checks[1].trial_defer, ('a', 'b'))
        self.assertEqual(joint.checks[1].lost_on_time_ids, ('a', 'b'))
        self.assertEqual(joint.checks[1].trial_late_total, 0)  # old late job improves, still reject

    def test_fixed_prefix_does_not_replace_rejection(self):
        s = snapshot([task('a', 15), task('b', 30)], sla=1)
        ref = simulate(s)
        request = ranked_request(s, ref, 1)
        self.assertEqual(request, ('a',))
        d = filter_requests(s, ref, request)
        self.assertEqual(tuple(c.task_id for c in d.checks), ('a',))
        self.assertEqual(d.approved, ())
        self.assertEqual(ref.first_slot.started, ('a', 'b'))

    def test_edf_tie_order_and_all_seven_action_prefixes(self):
        jobs = [task(f't{i:02}', 120, (1, 0, 0)) for i in range(20)]
        s = snapshot(reversed(jobs), capacity=(20, 1, 1), sla=2)
        p = simulate(s)
        self.assertEqual(p.first_slot.started, tuple(t.task_id for t in jobs))
        self.assertEqual([len(ranked_request(s, p, a)) for a in range(7)], [0, 1, 4, 16, 1, 4, 16])

    def test_negative_slack_original_late_is_not_rejected(self):
        s = snapshot([task('a', 15, arrival=15), task('b', 15, arrival=15)],
                     capacity=(1, 1, 1), sla=1.5, minute=60)
        p = simulate(s)
        d = filter_requests(s, p, ('a',))
        self.assertEqual(d.approved, ('a',))
        self.assertEqual(p.total_lateness(p.late_ids), d.plan.total_lateness(p.late_ids))

    def test_original_late_total_cannot_increase(self):
        s = snapshot([task('a', 15, arrival=15)], sla=1.5, minute=60)
        d = filter_requests(s, simulate(s), ('a',))
        self.assertEqual(d.approved, ())
        self.assertIn('ORIGINAL_LATE_TOTAL_INCREASED', d.checks[0].reasons)
        self.assertEqual(d.preview.started, ('a',))

    def test_all_rejected_is_normal_edf_not_guard_failure(self):
        s = snapshot([task('a', 15)], sla=1)
        d = filter_requests(s, simulate(s), ('a',))
        self.assertEqual(d.approved, ())
        self.assertEqual(d.preview.started, ('a',))
        self.assertFalse(d.reference_guard_failed)

    def test_only_reference_starters_and_no_refill(self):
        s = snapshot([task('a', 15), task('b', 30), task('c', 45)], capacity=(1, 1, 1), sla=1)
        p = simulate(s)
        with self.assertRaises(ValueError): filter_requests(s, p, ('b',))
        req = ranked_request(s, p, 1)
        d = filter_requests(s, p, req)
        self.assertEqual(len(req), 1)
        self.assertEqual(len(d.checks), 1)
        self.assertEqual(d.approved, ())

    def test_approved_is_not_always_policy_wait(self):
        s = snapshot([task('a', 15), task('c', 30, (2, 0, 0)), task('b', 45)], sla=4)
        p = simulate(s)
        d = filter_requests(s, p, ('b', 'a'))
        self.assertEqual(set(d.approved), {'a', 'b'})
        self.assertEqual(d.preview.started, ('c',))
        self.assertEqual(d.preview.policy_wait, ('a',))
        self.assertEqual(d.preview.capacity_wait, ('b',))

    def test_guard_exact_boundary_and_candidate_crossing(self):
        s = snapshot([task('a', 15)], minute=3945, sla=400)
        p = simulate(s)
        self.assertEqual(p.max_release, 3960)
        self.assertTrue(CompletionGuard.accepts(p))
        d = filter_requests(s, p, ('a',))
        self.assertEqual(d.approved, ())
        self.assertIn('COMPLETION_GUARD_REJECTED', d.checks[0].reasons)
        q = simulate(snapshot([task('a', 15.000001)], minute=3945, sla=400))
        self.assertEqual(q.max_release, 3975)
        self.assertFalse(CompletionGuard.accepts(q))


class AdapterTests(unittest.TestCase):
    def test_seeded_synthetic_calendar_v2_parity(self):
        rng = random.Random(20261020)
        for case in range(40):
            with self.subTest(case=case):
                jobs = [task(f'x{i:02}', rng.choice((0.1, 15, 15.01, 16.2, 31.5, 60.01, 90)),
                             tuple(cap*rng.choice((0, .125, .25, .5, 1)) for cap in (10000, 200, 16000)),
                             rng.choice((0, 15, 30))) for i in range(rng.randint(1, 12))]
                env = environment(jobs, sla=rng.choice((1, 1.5, 2, 4)))
                obs = env.reset(); adapter = ResearchAdapter()
                while env.minute < 30:
                    obs, _, _ = adapter.execute(env, adapter.decide(obs))
                s = adapter.observe(obs)
                deferred = frozenset(t.task_id for t in s.pending if rng.random() < .5)
                plan = simulate(s, deferred)
                while env.pending or env.running:
                    now = env.observation()
                    order = tuple(t.task_id for t in sorted(now['pending'], key=lambda t:
                                  (t.arrival_minute+env.sla*t.duration_minutes, t.arrival_minute, t.task_id)))
                    _, _, row = env.step(Action(order, deferred if env.minute == 30 else frozenset(), 20))
                    if row['minute'] == 30:
                        self.assertEqual(row['policy_wait_tasks'], len(plan.first_slot.policy_wait))
                        self.assertEqual(row['capacity_wait_tasks'], len(plan.first_slot.capacity_wait))
                for identity, p in plan.by_id.items():
                    e = env.ledger[identity]
                    self.assertEqual((e['start_minute'], e['finish_minute'], e['release_minute']),
                                     (p.start, p.finish, p.release))

    def test_fractional_duration_running_start_cancellation(self):
        env = environment([task('x', 60.01, arrival=15)]); obs = env.reset(); adapter = ResearchAdapter()
        for _ in range(2):
            obs, _, _ = adapter.execute(env, adapter.decide(obs))
        d = adapter.decide(obs)
        self.assertEqual(d.reference.by_id['x'].start, 15)
        self.assertEqual(d.reference.by_id['x'].finish, env.ledger['x']['finish_minute'])

    def test_preview_mismatch_raises_and_preserves_failure_evidence(self):
        env = environment([task('a', 60)]); obs = env.reset(); adapter = ResearchAdapter()
        d = adapter.decide(obs)
        corrupt_preview = replace(d.filter.preview, started=())
        corrupt = replace(d, filter=replace(d.filter, preview=corrupt_preview))
        with self.assertRaises(AssertionError): adapter.execute(env, corrupt)
        self.assertEqual(adapter.records[-1]['error_code'], 'PREVIEW_EXECUTION_MISMATCH')
        self.assertEqual(adapter.records[-1]['actual_started'], ('a',))
        self.assertEqual(env.minute, 15)  # no rollback/retry that hides the failed real step

    def test_guard_prediction_failure_but_actual_clear_by_66h_is_preserved(self):
        # EDF packing anomaly: an unknown future job changes backfill packing.
        jobs = [task('a', 990, (10000, 0, 0)), task('b', 1650, (2500, 0, 0)),
                task('c', 1320, (7500, 0, 0)), task('d', 1980, (2500, 0, 0)),
                task('z', 330, (5000, 0, 0), 660)]
        env = environment(jobs); obs = env.reset(); adapter = ResearchAdapter()
        first = adapter.decide(obs, action_id=1)
        self.assertEqual(first.reference.max_release, 4290)
        self.assertEqual(first.proposed, ('a',))
        obs, _, _ = adapter.execute(env, first)
        self.assertEqual(adapter.records[-1]['proposed'], ('a',))
        while env.minute < env.horizon:
            obs, _, _ = adapter.execute(env, adapter.decide(obs, action_id=0))
        s = adapter.summary(env)
        self.assertTrue(s['failure_latched'])
        self.assertEqual(s['actual_last_release_minute'], 3960)
        self.assertTrue(s['actual_common_idle_tail_valid'])
        self.assertEqual(s['actual_unfinished_count'], 0)
        self.assertEqual(s['irreversible_miss_ids'], ('c',))

    def test_no_new_arrival_preserves_accepted_plan_suffix(self):
        env = environment([task('a', 15, (2500, 0, 0)), task('b', 15, (2500, 0, 0)),
                           task('c', 30, (10000, 0, 0)), task('d', 45, (5000, 0, 0))])
        obs = env.reset(); adapter = ResearchAdapter()
        decision = adapter.decide(obs, proposed=('a', 'b'))
        self.assertEqual(decision.filter.approved, ('a',))
        accepted = decision.filter.plan.by_id
        obs, _, _ = adapter.execute(env, decision)
        while env.minute < 90:
            next_decision = adapter.decide(obs, action_id=0)
            for identity, timing in next_decision.reference.by_id.items():
                self.assertEqual(timing.finish, accepted[identity].finish)
                self.assertEqual(timing.release, accepted[identity].release)
            obs, _, _ = adapter.execute(env, next_decision)
        for identity, timing in accepted.items():
            self.assertEqual(env.ledger[identity]['finish_minute'], timing.finish)

    def test_calendar_parity_all_resources_and_numeric_boundaries(self):
        # Frozen OM capacities, artificial demands only. No configuration mutation.
        for axis, cap in enumerate((10000, 200, 16000)):
            def demand(fraction):
                values = [0, 0, 0]; values[axis] = cap*fraction
                return tuple(values)
            env = environment([task('a', 15.000001, demand(0.5)),
                               task('b', 29.999999, demand(1)),
                               task('c', 45, demand(0.5)), task('d', 16, demand(0.25))])
            obs = env.reset(); adapter = ResearchAdapter()
            ref = simulate(adapter.observe(obs))
            while env.minute < 135:
                obs, _, _ = adapter.execute(env, adapter.decide(obs, action_id=0))
            for identity, timing in ref.by_id.items():
                self.assertEqual((env.ledger[identity]['finish_minute'], env.ledger[identity]['release_minute']),
                                 (timing.finish, timing.release))

    def test_counterfactual_calls_leave_real_environment_unchanged(self):
        env = environment([task('a', 60, (1, 0, 0)), task('b', 30, (1, 0, 0))])
        obs = env.reset(); adapter = ResearchAdapter()
        before = copy.deepcopy((env.minute, env.pending, env.running, env.ledger, env.history, env.steps, env.inlet))
        for action in range(7): adapter.decide(obs, action_id=action)
        self.assertEqual(before, (env.minute, env.pending, env.running, env.ledger, env.history, env.steps, env.inlet))

    def test_stale_decision_fails_loudly(self):
        env = environment([task('a', 60)]); obs = env.reset(); adapter = ResearchAdapter()
        d = adapter.decide(obs)
        obs, _, _ = adapter.execute(env, d)
        with self.assertRaises(AssertionError): adapter.execute(env, d)

    def test_calendar_and_v2_full_known_suffix_agree(self):
        env = environment([task('a', 16, (10000, 0, 0)), task('b', 31, (10000, 0, 0)),
                           task('c', 44, (0, 200, 16000))])
        obs = env.reset(); adapter = ResearchAdapter()
        s = adapter.observe(obs); plan = simulate(s)
        while env.minute < 90:
            decision = adapter.decide(obs, action_id=0)
            obs, done, row = adapter.execute(env, decision)
        for identity, expected in plan.by_id.items():
            actual = env.ledger[identity]
            self.assertEqual((actual['start_minute'], actual['finish_minute'], actual['release_minute']),
                             (expected.start, expected.finish, expected.release))

    def test_preview_vs_real_capacity_before_defer(self):
        env = environment([task('a', 15, (5000, 0, 0)), task('c', 30, (10000, 0, 0)),
                           task('b', 45, (5000, 0, 0))], sla=4)
        obs = env.reset(); adapter = ResearchAdapter()
        decision = adapter.decide(obs, proposed=('b', 'a'))
        obs, _, _ = adapter.execute(env, decision)
        self.assertEqual(adapter.records[-1]['approved'], ('b', 'a'))
        self.assertEqual(adapter.records[-1]['effective_policy_wait'], ('a',))
        self.assertEqual(adapter.records[-1]['capacity_wait'], ('b',))
        self.assertEqual(adapter.records[-1]['actual_started'], ('c',))

    def test_unknown_future_counterexample(self):
        outcomes = []
        terminal_temperatures = []
        for defer in (False, True):
            env = environment([task('a', 30, (10000, 0, 0)),
                               task('b', 10, (10000, 0, 0), 30)], sla=1.5)
            adapter = ResearchAdapter(); obs = env.reset()
            s = adapter.observe(obs)
            self.assertEqual(tuple(t.task_id for t in s.pending), ('a',))
            d = adapter.decide(obs, proposed=('a',) if defer else ())
            self.assertEqual(d.filter.approved, ('a',) if defer else ())
            obs, _, _ = adapter.execute(env, d)
            while env.minute < env.horizon:
                obs, _, _ = adapter.execute(env, adapter.decide(obs, action_id=0))
            result = env.result()
            self.assertEqual(adapter.summary(env)['irreversible_miss_ids'],
                             tuple(r['task_id'] for r in result['tasks'] if r['deadline_missed']))
            outcomes.append(result['summary']['deadline_missed'])
            terminal_temperatures.append(result['summary']['inlet_final_c'])
        self.assertEqual(outcomes, [0, 1])
        delta_t = terminal_temperatures[1]-terminal_temperatures[0]
        self.assertLessEqual(abs(delta_t), .05)
        self.assertTrue(math.isfinite(env.model.c_kj_k*delta_t/3600))

    def test_guard_failure_latches_until_reset_and_keeps_72h(self):
        env = environment([task('long', 4000, (1, 1, 1))])
        adapter = ResearchAdapter(); obs = env.reset()
        while env.minute < env.horizon:
            decision = adapter.decide(obs, action_id=3)
            self.assertTrue(adapter.guard.failure_latched)
            self.assertEqual(decision.action.defer, frozenset())
            obs, _, _ = adapter.execute(env, decision)
        s = adapter.summary(env)
        self.assertEqual(s['failure_code'], FAILURE)
        self.assertEqual(s['actual_last_release_minute'], 4005)
        self.assertEqual(s['actual_unfinished_count'], 0)
        self.assertFalse(s['actual_common_idle_tail_valid'])
        self.assertTrue(s['failure_latched'])
        self.assertTrue(adapter.records[-1]['reference_cg_pass'])
        self.assertEqual(len(env.steps), 288)
        self.assertTrue(all(r['supply_requested_c'] == 20 for r in env.steps[264:]))
        adapter.reset(); self.assertFalse(adapter.guard.failure_latched)

    def test_unfinished_and_censored_are_retained(self):
        env = environment([task('long', 5000, (1, 1, 1))]); obs = env.reset()
        adapter = ResearchAdapter()
        while env.minute < env.horizon:
            obs, _, _ = adapter.execute(env, adapter.decide(obs, action_id=0))
        result = env.result(); s = adapter.summary(env)
        self.assertEqual(result['summary']['deadline_censored'], 1)
        self.assertEqual(s['actual_unfinished_count'], 1)
        self.assertIsNone(s['actual_last_release_minute'])
        self.assertFalse(s['actual_common_idle_tail_valid'])

    def test_adapter_does_not_access_environment_for_policy(self):
        env = environment([task('a', 15, (1, 0, 0)), task('future', 10, (1, 0, 0), 300)])
        obs = env.reset(); adapter = ResearchAdapter()
        class Restricted(dict):
            def __getitem__(self, key):
                if key not in {'minute', 'capacity', 'sla_multiplier', 'pending', 'running'}:
                    raise AssertionError('Non-task observation access: ' + key)
                return super().__getitem__(key)
        d = adapter.decide(Restricted(obs), action_id=0)
        self.assertEqual(d.reference.first_slot.started, ('a',))
        self.assertEqual(set(adapter.known), {'a'})


if __name__ == '__main__': unittest.main()
