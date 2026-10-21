"""Synthetic task-joint pilot integration only; never start the real pilot."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import torch
from tests.sca_ppo.test_guards import environment, task
from experiments.platform_v2.environment import run as run_v2


class PilotTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('experiments.task_ppo.pilot'))
        from experiments.task_ppo import pilot
        return pilot

    def test_preregistered_balance_and_identity(self):
        module=self.module()
        plan=json.loads(module.PLAN_PATH.read_text())
        module.validate_plan(plan)
        for mutation in ('gamma','split','budget','entropy'):
            altered=copy.deepcopy(plan)
            if mutation=='gamma':altered['ppo']['gamma']=.99
            elif mutation=='entropy':altered['ppo']['entropy_coef']=.01
            elif mutation=='budget':altered['training_episodes']=33
            else:altered['holdout_ids'][0]=altered['train_ids'][0]
            with self.assertRaises(ValueError):module.validate_plan(altered)

    def test_full_physical_episode_and_real_backprop(self):
        module=self.module();torch.set_num_threads(1)
        from experiments.task_ppo.joint_ppo import JointPPOTrainer
        env=environment([task('a',30,(100,1,1)),task('b',61,(100,1,1)),task('c',16,(100,1,1),15)],sla=1.5)
        idle=run_v2(env.scenario,env.inv,env.cfg,'edf','fixed',1.5,idle=True)
        trainer=JointPPOTrainer(dict(seed=19))
        before=module.model_fingerprint(trainer.model)
        base=Path('research_notes/task-ppo-v1/unit-tmp');base.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=base) as tmp:
            batch,row=module.rollout(trainer,env.scenario,env.inv,env.cfg,idle,Path(tmp)/'episode',
                dict(reward=dict(c_scale_kg=100,lambda_miss=.01)),False,'ARTIFICIAL','OM')
            self.assertEqual(len(batch['rewards']),288)
            self.assertEqual(len(batch['task_states']),288)
            self.assertEqual(row['completed'],3)
            self.assertTrue(row['protocol_valid']);self.assertTrue(row['verification_pass'])
            self.assertEqual(sum(row['binary_action_counts']),sum(len(row) for row in batch['task_actions']))
            metrics=trainer.update(batch)
            self.assertEqual(metrics['physical_slots'],288)
            self.assertGreater(metrics['actor_parameter_delta_l2'],0)
            self.assertNotEqual(before,module.model_fingerprint(trainer.model))

    def test_engineering_continuation_compares_all_saved_semantics(self):
        module=self.module()
        row=dict(minute=0,choices=[dict(task_id='a',state=[0]*48,action=1,log_prob=-.5)],
                 reward=-.2,approved=['a'],slot_seconds=55,internal_candidate_checks=100)
        changed=copy.deepcopy(row);changed['slot_seconds']=.1;changed['internal_candidate_checks']=1
        module.compare_saved_prefix([row],[changed])
        changed['choices'][0]['action']=0
        with self.assertRaises(AssertionError):module.compare_saved_prefix([row],[changed])


if __name__=='__main__':unittest.main()
