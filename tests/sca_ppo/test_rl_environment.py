"""Synthetic-only contracts for the E2 state and actual-step reward wrapper."""
import copy
import math
import unittest
from dataclasses import replace

from experiments.platform_v2.environment import run as run_v2
from experiments.sca_ppo.edf_calendar import Running, Snapshot, simulate
from experiments.sca_ppo.rl_environment import (
    RLEpisode, RewardLedger, STATE_NAMES, STATE_SCHEMA, state_vector,
)
from tests.sca_ppo.test_guards import environment, task


def episode(tasks):
    base = environment(tasks, sla=1.5)
    idle = run_v2(replace(base.scenario, tasks=()), base.inv, base.cfg,
                  'edf', 'fixed', 1.5)
    return RLEpisode(base.scenario, base.inv, base.cfg, idle, 100., .01)


def complete(ep, first_action=0):
    ep.reset()
    done = False
    while not done:
        _, eligible = ep.prepare()
        action = first_action if not ep.records else 0
        _, done, _ = ep.step(action if eligible else 0, policy_sampled=eligible)
    return ep.finish()


class StateTests(unittest.TestCase):
    def fixture(self):
        running = task('r', 75, (50, 1, 100))
        pending = (task('a', 16, (50, 1, 100)), task('b', 30, (100, 0, 100)))
        snapshot = Snapshot(60, (100, 10, 1000), 2, pending, (Running(running, 75),))
        obs = dict(minute=60, capacity=(100, 10, 1000), available=(50, 9, 900),
                   pending=pending, running=(('r', (50, 1, 100), 75),),
                   ci_g_per_kwh=600, ci_history=((0, 200), (15, 300), (45, 500)),
                   inlet_c=22, outdoor_c=30, sla_multiplier=2)
        return obs, simulate(snapshot)

    def test_hand_calculated_27_features_use_grid_work_and_reference_finish(self):
        obs, ref = self.fixture()
        values = state_vector(obs, ref)
        expected = [math.sin(math.pi/12), math.cos(math.pi/12), 1/72, 65/66, 0,
                    .6, 1/3, .3, 1., .4, .5, .1, .1, .5, .1,
                    .002, .001, .03125, 1/480, .475/2.75,
                    -60/1440, -52/1440, 1., 2/3, .55, .75, 0.]
        self.assertEqual(len(values), 27)
        self.assertEqual(len(STATE_NAMES), len(set(STATE_NAMES)))
        self.assertEqual(len(STATE_NAMES), 27)
        self.assertTrue(STATE_SCHEMA)
        for actual, want in zip(values, expected):
            self.assertAlmostEqual(actual, want, places=12)

    def test_equal_time_and_future_ci_history_never_enter_the_state(self):
        obs, ref = self.fixture()
        before = copy.deepcopy(obs)
        baseline = state_vector(obs, ref)
        obs['ci_history'] += ((60, -100000), (75, math.nan), (4320, math.inf))
        self.assertEqual(state_vector(obs, ref), baseline)
        self.assertEqual(before['ci_history'], ((0, 200), (15, 300), (45, 500)))

    def test_history_windows_and_last_observation_at_least_one_hour_old(self):
        obs, ref = self.fixture()
        obs.update(minute=1500, pending=(), running=(), available=obs['capacity'],
                   ci_history=((0, 10000), (59, 10000), (60, 100), (1200, 200),
                               (1260, 300), (1440, 400), (1485, 900)))
        ref = simulate(Snapshot(1500, obs['capacity'], 2, ()))
        values = state_vector(obs, ref)
        self.assertAlmostEqual(values[6], 1.6/3)
        self.assertAlmostEqual(values[7], .3)
        self.assertAlmostEqual(values[9], .2)
        self.assertEqual(values[4], .5)

    def test_empty_queue_history_and_tail_are_finite_without_extra_clipping(self):
        env = environment([])
        obs = env.reset()
        obs.update(minute=4200, ci_g_per_kwh=1500, inlet_c=60)
        ref = simulate(Snapshot(4200, obs['capacity'], 2, ()))
        values = state_vector(obs, ref, failure_latched=True)
        self.assertTrue(all(math.isfinite(x) for x in values))
        self.assertEqual(values[3:5], [0., 1.])
        self.assertEqual(values[5:10], [1.5, 1.5, 1.5, 0., 0.])
        self.assertEqual(values[15:24], [0.]*9)
        self.assertEqual(values[24], 1.5)
        self.assertEqual(values[26], 1.)

    def test_nonfinite_current_observation_is_rejected(self):
        obs, ref = self.fixture()
        obs['ci_g_per_kwh'] = math.nan
        with self.assertRaises(ValueError):
            state_vector(obs, ref)


class RewardTests(unittest.TestCase):
    def test_incremental_carbon_and_each_actual_new_miss_count_once(self):
        ledger = RewardLedger(100., .01)
        first = ledger.add(25., 5., ('late',))
        self.assertAlmostEqual(first['reward'], -.21)
        self.assertEqual(first['new_miss_count'], 1)
        second = ledger.add(2., 5., ())
        self.assertAlmostEqual(second['carbon_reward'], .03)
        self.assertAlmostEqual(ledger.total_reward, -.18)
        self.assertEqual(ledger.seen_miss_ids, frozenset({'late'}))
        with self.assertRaises(ValueError):
            ledger.add(1., 0., ('late',))
        with self.assertRaises(ValueError):
            ledger.add(1., 0., ('new', 'new'))
        self.assertAlmostEqual(ledger.total_reward, -.18)
        self.assertEqual(ledger.seen_miss_ids, frozenset({'late'}))

    def test_invalid_scales_and_nonfinite_step_do_not_mutate_ledger(self):
        for scale, penalty in ((0, .01), (math.nan, .01), (100, -.01)):
            with self.assertRaises(ValueError):
                RewardLedger(scale, penalty)
        ledger = RewardLedger(100., .01)
        with self.assertRaises(ValueError):
            ledger.add(math.inf, 1., ('x',))
        self.assertEqual(ledger.total_reward, 0.)
        self.assertFalse(ledger.seen_miss_ids)


class EpisodeTests(unittest.TestCase):
    def test_all_seven_policy_actions_preserve_e0_prefix_mapping(self):
        for action, want_count in enumerate((0, 1, 4, 16, 1, 4, 16)):
            with self.subTest(action=action):
                ep = episode([task(f't{i:02}', 120, (1, 0, 0)) for i in range(20)])
                ep.reset()
                state, eligible = ep.prepare()
                self.assertTrue(eligible)
                _, done, info = ep.step(action, policy_sampled=True)
                self.assertFalse(done)
                self.assertEqual(len(info['proposed']), want_count)
                self.assertEqual(info['action_id'], action)
                self.assertTrue(info['policy_sampled'])
                self.assertEqual(info['state'], state)
                self.assertEqual(info['row']['minute'], 0.)

    def test_policy_sample_flag_and_forced_noop_are_validated_before_step(self):
        ep = episode([task('a', 30)])
        ep.reset()
        for action, sampled in ((0, False), (7, True), (-1, True)):
            with self.assertRaises(ValueError):
                ep.step(action, policy_sampled=sampled)
            self.assertEqual(ep.env.minute, 0.)
        ep.step(0, policy_sampled=True)
        self.assertFalse(ep.prepare()[1])
        for action, sampled in ((1, False), (0, True)):
            with self.assertRaises(ValueError):
                ep.step(action, policy_sampled=sampled)
            self.assertEqual(ep.env.minute, 15.)

    def test_full_72_hour_actual_reward_conserves_carbon_and_miss_ids(self):
        ep = episode([task('a', 30, (10000, 0, 0)),
                      task('b', 10, (10000, 0, 0), 30)])
        result = complete(ep, first_action=1)
        summary = result['summary']
        self.assertTrue(summary['protocol_valid'])
        self.assertEqual(summary['failure_reasons'], ())
        self.assertEqual(len(ep.records), 288)
        self.assertEqual(sum(r['new_miss_count'] for r in ep.records), 1)
        self.assertEqual(ep.ledger.seen_miss_ids, frozenset({'b'}))
        self.assertEqual({r['task_id'] for r in result['tasks'] if r['deadline_missed']}, {'b'})
        expected = -summary['common_incremental_carbon_kg']/100 - .01
        self.assertAlmostEqual(summary['reward_sum'], expected, places=11)
        self.assertAlmostEqual(summary['reward_conservation_error'], 0., places=11)
        self.assertAlmostEqual(sum(r['reward'] for r in ep.records), expected, places=11)
        self.assertTrue(any(not r['policy_sampled'] for r in ep.records))
        self.assertTrue(all(r['row']['minute'] == n*15 for n, r in enumerate(ep.records)))
        with self.assertRaises(RuntimeError):
            ep.step(0, policy_sampled=False)

    def test_grid_release_after_raw_deadline_does_not_create_a_miss(self):
        ep = episode([task('a', 16)])
        result = complete(ep)
        self.assertEqual(result['summary']['deadline_missed'], 0)
        self.assertFalse(ep.ledger.seen_miss_ids)

    def test_prepare_is_repeatable_and_guard_inspection_occurs_once_per_step(self):
        ep = episode([task('long', 4000)])
        ep.reset()
        before = copy.deepcopy((ep.env.minute, ep.env.pending, ep.env.ledger, ep.env.steps))
        first = ep.prepare()
        self.assertEqual(first, ep.prepare())
        self.assertEqual(first[0][-1], 1.)
        self.assertFalse(first[1])
        self.assertEqual(ep.adapter.guard.reference_guard_failure_count, 0)
        self.assertEqual(before, (ep.env.minute, ep.env.pending, ep.env.ledger, ep.env.steps))
        ep.step(0, policy_sampled=False)
        self.assertEqual(ep.adapter.guard.reference_guard_failure_count, 1)
        ep.prepare()
        self.assertEqual(ep.adapter.guard.reference_guard_failure_count, 1)
        while ep.env.minute < ep.env.horizon:
            ep.step(0, policy_sampled=False)
        result = ep.finish()
        self.assertEqual(len(ep.records), 288)
        self.assertTrue(result['summary']['failure_latched'])
        self.assertFalse(result['summary']['protocol_valid'])
        self.assertEqual(result['summary']['actual_last_release_minute'], 4005)
        self.assertFalse(result['summary']['actual_common_idle_tail_valid'])
        self.assertIn('EDF_REFERENCE_COMPLETION_GUARD_FAILED', result['summary']['failure_reasons'])

    def test_unfinished_censored_failure_is_preserved_without_forcing_miss_equality(self):
        ep = episode([task('long', 5000)])
        with self.assertRaises(RuntimeError):
            ep.finish()
        result = complete(ep)
        self.assertEqual(len(ep.records), 288)
        self.assertEqual(result['summary']['unfinished'], 1)
        self.assertEqual(result['summary']['deadline_censored'], 1)
        self.assertEqual(result['summary']['deadline_missed'], 0)
        self.assertFalse(result['summary']['protocol_valid'])
        self.assertIn('UNFINISHED_TASKS', result['summary']['failure_reasons'])
        self.assertIn('CENSORED_DEADLINES', result['summary']['failure_reasons'])

    def test_reset_clears_episode_records_and_miss_penalties(self):
        ep = episode([task('a', 30)])
        ep.reset()
        ep.step(0, policy_sampled=True)
        ep.reset()
        self.assertEqual(ep.env.minute, 0.)
        self.assertEqual(ep.records, [])
        self.assertEqual(ep.ledger.total_reward, 0.)
        self.assertFalse(ep.adapter.records)
        self.assertFalse(ep.adapter.guard.failure_latched)


if __name__ == '__main__':
    unittest.main()
