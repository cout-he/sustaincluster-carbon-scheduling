"""Small synthetic traces only; no E1 real-input CLI invocation."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from dataclasses import replace

from tests.sca_ppo.test_guards import environment, task
from experiments.sca_ppo.e1_policies import MatchedRuleAdapter


class E1RunnerTests(unittest.TestCase):
    def test_mismatch_failure_preserves_serializable_partial_ledger(self):
        module = self.runner(); env = environment([task('a',60)])
        original = MatchedRuleAdapter.execute
        def corrupted(adapter, real_env, decision):
            preview = replace(decision.filter.preview, started=())
            return original(adapter, real_env, replace(decision, filter=replace(decision.filter, preview=preview)))
        root = Path('research_notes/sca-ppo/unit-tmp'); root.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as tmp:
            directory = Path(tmp)/'bad'
            with patch.object(MatchedRuleAdapter,'execute',corrupted), self.assertRaises(AssertionError):
                module.run_episode(env.scenario,env.inv,env.cfg,'C',directory,
                    dict(fragment='ARTIFICIAL',capacity='OM',source_sha='ARTIFICIAL'),
                    dict(action_probe_minutes=[],stop_decision_seconds=30))
            partial = json.loads((directory/'error.json').read_text())
            self.assertEqual(partial['minute'],15)
            self.assertEqual(partial['task_ledger']['a']['state'],'running')
            self.assertEqual(partial['adapter_records'][-1]['error_code'],'PREVIEW_EXECUTION_MISMATCH')
            self.assertIsInstance(partial['adapter_records'][-1]['expected_preview'],dict)

    def runner(self):
        from experiments.sca_ppo import e1_runner
        return e1_runner

    def test_action_probes_leave_actual_state_untouched(self):
        env = environment([task('a', 60, (100, 0, 0)), task('b', 45, (0, 1, 0))])
        obs = env.reset(); adapter = MatchedRuleAdapter('C', env.cfg)
        adapter.observe(obs)
        before = (env.minute, len(adapter.records), adapter.guard.failure_latched)
        rows = self.runner().probe_actions(adapter, obs)
        self.assertEqual(len(rows), 7)
        self.assertEqual([r['action_id'] for r in rows], list(range(7)))
        self.assertEqual(before, (env.minute, len(adapter.records), adapter.guard.failure_latched))
        self.assertGreater(len({r['signature'] for r in rows}), 1)

    def test_complete_episode_saved_and_independently_verified(self):
        module = self.runner()
        env = environment([task('a', 16, (1, 1, 1)), task('b', 15, (1, 1, 1), 15)])
        root = Path('research_notes/sca-ppo/unit-tmp'); root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as tmp:
            row = module.run_episode(env.scenario, env.inv, env.cfg, 'C', Path(tmp)/'C',
                dict(fragment='ARTIFICIAL', capacity='OM', source_sha='ARTIFICIAL'),
                dict(action_probe_minutes=[0,60], stop_decision_seconds=30))
            self.assertTrue(row['verification_pass'])
            self.assertEqual(row['completed'], 2)
            self.assertTrue(row['actual_common_idle_tail_valid'])
            result = json.loads((Path(tmp)/'C/result.json').read_text())
            self.assertEqual(len(result['steps']), 288)
            self.assertEqual(result['tasks'][0]['arrival_minute'], 0)
            self.assertEqual(len((Path(tmp)/'C/action-probes.jsonl').read_text().splitlines()), 14)

    def test_malformed_protocol_matrix_rejected_before_data_read(self):
        module = self.runner()
        with self.assertRaises(ValueError):
            module.validate_protocol(dict(schema='BAD', cases=[]))


if __name__ == '__main__': unittest.main()
