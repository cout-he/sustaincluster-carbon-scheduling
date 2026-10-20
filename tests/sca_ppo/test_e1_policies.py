"""Synthetic E1 fairness checks; no workload loader or training is called."""
import copy
import unittest

from experiments.platform_v2.inventory import load_config
from experiments.sca_ppo.adapter import Decision, ResearchAdapter
from experiments.sca_ppo.deadline_filter import CandidateCheck, FilterResult
from experiments.sca_ppo.edf_calendar import preview_now, simulate
from experiments.sca_ppo.e1_policies import MatchedRuleAdapter, filter_completion_only
from tests.sca_ppo.test_guards import environment, snapshot, task


def observation(jobs=(), minute=240, capacity=(20, 2, 20), sla=2,
                ci=500, history=((0, 100), (225, 300))):
    return dict(minute=minute, capacity=capacity, sla_multiplier=sla,
                pending=tuple(jobs), running=(), ci_g_per_kwh=ci,
                ci_history=tuple(history))


class MatchedRuleTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config()

    def adapter(self, arm):
        return MatchedRuleAdapter(arm, self.cfg)

    def test_completion_only_approves_self_late_without_personal_slack(self):
        obs = observation([task('tight', 15, arrival=240)], sla=1)
        b = self.adapter('B').decide(obs)
        c = self.adapter('C').decide(obs)
        self.assertEqual(b.proposed, ('tight',))
        self.assertEqual(c.proposed, b.proposed)
        self.assertEqual(b.filter.approved, ('tight',))
        self.assertEqual(b.filter.plan.by_id['tight'].finish, 270)
        self.assertEqual(b.filter.checks[0].lost_on_time_ids, ('tight',))
        self.assertEqual(b.filter.checks[0].reasons, ())
        self.assertEqual(c.filter.approved, ())
        self.assertIn('REFERENCE_ON_TIME_SET_LOST', c.filter.checks[0].reasons)

    def test_completion_only_lateness_diagnostics_do_not_reject(self):
        obs = observation([task('old', 15, arrival=15)], minute=60, sla=1.5)
        b = self.adapter('B').decide(obs, proposed=('old',))
        c = self.adapter('C').decide(obs, proposed=('old',))
        check = b.filter.checks[0]
        self.assertEqual((check.reference_late_total, check.trial_late_total), (37.5, 52.5))
        self.assertEqual(b.filter.approved, ('old',))
        self.assertEqual(check.reasons, ())
        self.assertEqual(c.filter.approved, ())
        self.assertIn('ORIGINAL_LATE_TOTAL_INCREASED', c.filter.checks[0].reasons)

    def test_completion_only_allows_damage_to_another_known_job(self):
        obs = observation([task('c', 20, (2, 0, 0)), task('a', 30, (4, 0, 0)),
                           task('b', 30, (2, 0, 0))], minute=0, capacity=(4, 1, 1))
        b = self.adapter('B').decide(obs, proposed=('b',))
        c = self.adapter('C').decide(obs, proposed=('b',))
        self.assertEqual(b.filter.approved, ('b',))
        self.assertIn('b', b.filter.plan.on_time_ids)
        self.assertEqual(b.filter.checks[0].lost_on_time_ids, ('a',))
        self.assertEqual(c.filter.approved, ())

    def test_arms_share_power_request_and_reference_from_identical_snapshot(self):
        obs = observation([task('cpu', 60, (2, 0, 0), 240),
                           task('gpu', 15, (0, 1, 0), 240),
                           task('long', 90, (1, 0, 0), 240)], capacity=(3, 1, 1))
        before = copy.deepcopy(obs)
        b, c = (self.adapter(arm).decide(obs) for arm in ('B', 'C'))
        self.assertEqual(b.proposed, ('gpu', 'cpu', 'long'))
        self.assertEqual(c.proposed, ('gpu', 'cpu', 'long'))
        self.assertEqual(b.snapshot, c.snapshot)
        self.assertEqual(b.reference.records, c.reference.records)
        self.assertEqual(obs, before)

    def test_trigger_uses_strict_median_and_four_hour_warmup(self):
        cases = ((225, 501, ((0, 100), (210, 300)), False),
                 (240, 200, ((0, 100), (225, 300)), False),
                 (240, 201, ((0, 100), (225, 300)), True),
                 (240, 999, (), False))
        for arm in ('B', 'C'):
            for minute, ci, history, triggered in cases:
                with self.subTest(arm=arm, minute=minute, ci=ci, history=history):
                    adapter = self.adapter(arm)
                    d = adapter.decide(observation([task('a', 60, arrival=minute)],
                                                  minute=minute, ci=ci, history=history))
                    self.assertEqual(d.proposed, ('a',) if triggered else ())
                    self.assertEqual(adapter.last_trigger['triggered'], triggered)
                    self.assertGreater(adapter.last_decision_ns, 0)

    def test_history_includes_24_hour_boundary_and_excludes_current_and_future(self):
        adapter = self.adapter('B')
        d = adapter.decide(observation([task('a', 60, arrival=1500)], minute=1500,
            ci=201, history=((45, 9999), (60, 100), (1485, 300), (1500, 0), (1515, 0))))
        self.assertEqual(d.proposed, ('a',))
        self.assertEqual(adapter.last_trigger['threshold'], 200)
        self.assertEqual(adapter.last_trigger['history_count'], 2)
        self.assertEqual(adapter.last_trigger['history'], ((60, 100), (1485, 300)))

    def test_a_never_requests_or_approves_push(self):
        adapter = self.adapter('A')
        obs = observation([task('tight', 15, arrival=240)], sla=1)
        for arguments in ({}, {'action_id': 3}, {'proposed': ('tight',)}):
            d = adapter.decide(obs, **arguments)
            self.assertEqual((d.proposed, d.filter.approved, d.action.defer), ((), (), frozenset()))
            self.assertEqual(d.action.order, ('tight',))
            self.assertEqual(d.action.supply_c, 20)
            self.assertTrue(d.reference_cg_pass)
        self.assertIsInstance(adapter, ResearchAdapter)
        self.assertIsInstance(d, Decision)

    def test_production_prefix_is_at_most_sixteen_without_refilling(self):
        jobs = [task(f'a{i:02}', 15, arrival=240) for i in range(20)]
        for arm in ('B', 'C'):
            with self.subTest(arm=arm):
                d = self.adapter(arm).decide(observation(jobs, sla=1))
                self.assertEqual(d.proposed, tuple(f'a{i:02}' for i in range(16)))
                self.assertEqual(tuple(check.task_id for check in d.filter.checks), d.proposed)
                self.assertEqual(d.filter.approved, d.proposed if arm == 'B' else ())
                self.assertTrue(set(d.filter.preview.started).issuperset({'a16', 'a17', 'a18', 'a19'}))

    def test_completion_only_trials_are_cumulative_and_preview_matches(self):
        s = snapshot([task('a', 15), task('b', 15)], minute=3930)
        ref = simulate(s)
        d = filter_completion_only(s, ref, ('a', 'b'))
        self.assertEqual(d.approved, ('a', 'b'))
        self.assertEqual(tuple(check.trial_defer for check in d.checks), (('a',), ('a', 'b')))
        self.assertEqual(d.plan.max_release, 3960)
        self.assertEqual(d.plan.forced_defer, frozenset({'a', 'b'}))
        self.assertEqual(d.preview, preview_now(s, frozenset({'a', 'b'})))
        self.assertEqual(d.preview, d.plan.first_slot)
        self.assertIsInstance(d, FilterResult)
        for check in d.checks:
            self.assertIsInstance(check, CandidateCheck)
            self.assertGreater(check.elapsed_ns, 0)
            self.assertGreater(check.calendar_elapsed_ns, 0)
            self.assertGreater(check.dispatch_slots, 0)
            self.assertGreater(check.task_checks, 0)

    def test_completion_guard_rejection_does_not_replace_prefix_candidate(self):
        obs = observation([task('a', 30, (2, 0, 0)), task('b', 15)],
                          minute=3930, capacity=(3, 1, 1), sla=400)
        d = self.adapter('B').decide(obs, action_id=1)
        self.assertEqual(d.proposed, ('a',))
        self.assertEqual(d.filter.approved, ())
        self.assertEqual(tuple(check.task_id for check in d.filter.checks), ('a',))
        self.assertEqual(d.filter.checks[0].reasons, ('COMPLETION_GUARD_REJECTED',))
        self.assertEqual(d.filter.preview.started, ('b', 'a'))

    def test_reference_guard_failure_latches_all_arms_until_reset(self):
        for arm in ('A', 'B', 'C'):
            with self.subTest(arm=arm):
                adapter = self.adapter(arm)
                first = adapter.decide(observation([task('long', 4000)]))
                self.assertFalse(first.reference_cg_pass)
                self.assertTrue(first.failure_latched)
                self.assertEqual(first.filter.approved, ())
                second = adapter.decide(observation([task('safe', 60, arrival=240)]))
                self.assertTrue(second.reference_cg_pass)
                self.assertTrue(second.failure_latched)
                self.assertEqual(second.proposed, () if arm == 'A' else ('safe',))
                self.assertEqual(second.filter.approved, ())
                self.assertEqual(adapter.guard.reference_guard_failure_count, 1)
                self.assertEqual(adapter.guard.first_failure_minute, 240)
                adapter.reset()
                self.assertFalse(adapter.guard.failure_latched)
                self.assertEqual(adapter.last_decision_ns, 0)
                self.assertIsNone(adapter.last_trigger)

    def test_request_validation_matches_e0_reference_contract(self):
        s = snapshot([task('a', 15), task('b', 30)], capacity=(1, 1, 1))
        ref = simulate(s)
        for request in (('a', 'a'), ('b',), ('missing',)):
            with self.subTest(request=request), self.assertRaises(ValueError):
                filter_completion_only(s, ref, request)
        other = snapshot([task('a', 15)], minute=15)
        with self.assertRaises(ValueError):
            filter_completion_only(other, ref, ())
        with self.assertRaises(ValueError):
            filter_completion_only(s, simulate(s, frozenset({'a'})), ())

    def test_frozen_configuration_is_required_and_not_mutated(self):
        before = copy.deepcopy(self.cfg)
        for key, value in (('threshold_warmup_hours', 3),
                           ('threshold_history_hours', 12), ('fixed_supply_c', 21)):
            cfg = copy.deepcopy(self.cfg)
            cfg['evaluation'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                MatchedRuleAdapter('B', cfg)
        with self.assertRaises(ValueError):
            MatchedRuleAdapter('unknown', self.cfg)
        self.adapter('B').decide(observation())
        self.assertEqual(self.cfg, before)

    def test_matched_previews_execute_in_frozen_v2_with_capacity_before_defer(self):
        jobs = [task('a', 15, (5000, 0, 0)), task('c', 30, (10000, 0, 0)),
                task('b', 45, (5000, 0, 0))]
        for arm in ('B', 'C'):
            with self.subTest(arm=arm):
                env = environment(jobs, sla=4)
                obs = env.reset()
                adapter = self.adapter(arm)
                d = adapter.decide(obs, proposed=('b', 'a'))
                self.assertEqual(d.filter.approved, ('b', 'a'))
                adapter.execute(env, d)
                self.assertEqual(adapter.records[-1]['approved'], ('b', 'a'))
                self.assertEqual(adapter.records[-1]['effective_policy_wait'], ('a',))
                self.assertEqual(adapter.records[-1]['capacity_wait'], ('b',))
                self.assertEqual(adapter.records[-1]['actual_started'], ('c',))


if __name__ == '__main__':
    unittest.main()
