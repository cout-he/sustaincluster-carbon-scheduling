"""Synthetic-only pilot integration; never invoke the real experiment entry."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from experiments.sca_ppo.ppo import PPOTrainer
from experiments.platform_v2.environment import run as run_v2
from tests.sca_ppo.test_guards import environment, task


class PilotTests(unittest.TestCase):
    def module(self):
        from experiments.sca_ppo import e2_pilot
        return e2_pilot

    def test_full_synthetic_rollout_changes_real_network(self):
        module = self.module(); torch.set_num_threads(1)
        env = environment([task('a',30,(100,1,1)),task('b',16,(100,1,1),15)],sla=1.5)
        idle = run_v2(env.scenario,env.inv,env.cfg,'edf','fixed',1.5,idle=True)
        trainer = PPOTrainer(dict(seed=11)); before = module.model_fingerprint(trainer.model)
        root=Path('research_notes/sca-ppo/unit-tmp');root.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as tmp:
            batch, summary = module.rollout(trainer,env.scenario,env.inv,env.cfg,idle,Path(tmp)/'episode',
                dict(reward=dict(c_scale_kg=100,lambda_miss=.01)),False,'ARTIFICIAL')
            self.assertEqual(len(batch['states']),288)
            self.assertTrue(batch['terminated'][-1])
            self.assertTrue(summary['protocol_valid'])
            self.assertTrue(summary['verification_pass'])
            self.assertEqual(summary['completed'],2)
            self.assertEqual(sum(summary['selected_action_counts']),sum(batch['policy_sampled']))
            metric=trainer.update(batch)
            self.assertGreater(metric['actor_parameter_delta_l2'],0)
            self.assertNotEqual(before,module.model_fingerprint(trainer.model))

    def test_budget_and_frozen_contract_guard(self):
        import json
        module = self.module()
        plan=json.loads(Path('docs/research/sca-ppo/E2-pilot-config.json').read_text())
        module.validate_plan(plan)
        bad=copy.deepcopy(plan);bad['training_episodes']=33
        with self.assertRaises(ValueError):module.validate_plan(bad)
        bad=copy.deepcopy(plan);bad['ppo']['gamma']=.99
        with self.assertRaises(ValueError):module.validate_plan(bad)

    def test_admission_split_cannot_be_swapped(self):
        import json
        module=self.module()
        plan=json.loads(Path('docs/research/sca-ppo/E2-pilot-config.json').read_text())
        left,right=plan['train_ids'][0],plan['holdout_ids'][0]
        plan['train_ids'][0],plan['holdout_ids'][0]=right,left
        plan['training_sequence']=[right if value==left else value for value in plan['training_sequence']]
        with self.assertRaises(ValueError):module.validate_plan(plan)

    def test_saved_verifier_failure_retains_original_reason(self):
        import json
        module=self.module();torch.set_num_threads(1)
        env=environment([task('a',30,(100,1,1))],sla=1.5)
        idle=run_v2(env.scenario,env.inv,env.cfg,'edf','fixed',1.5,idle=True)
        root=Path('research_notes/sca-ppo/unit-tmp');root.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as tmp:
            directory=Path(tmp)/'episode'
            with patch.object(module,'verify_result',return_value={'errors':['injected synthetic verification failure']}):
                _,summary=module.rollout(PPOTrainer(dict(seed=11)),env.scenario,env.inv,env.cfg,idle,directory,
                    dict(reward=dict(c_scale_kg=100,lambda_miss=.01)),True,'ARTIFICIAL')
            self.assertFalse(summary['protocol_valid'])
            self.assertFalse(summary['verification_pass'])
            self.assertIn('SAVED_V2_VERIFICATION_FAILED',summary['failure_reasons'])
            self.assertEqual(json.loads((directory/'verification.json').read_text())['errors'],
                             ['injected synthetic verification failure'])


if __name__ == '__main__':unittest.main()
