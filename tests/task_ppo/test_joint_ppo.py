"""Artificial numeric inputs only: physical slots and ragged joint proposals."""
import copy
import itertools
import math
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np
import torch


def actor_state(value=0.):
    return [value]*27+[.2]*20+[1.]


def critic_state(value=0.):
    return [value]*27+[0.]*21


class JointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def module(self):
        from experiments.task_ppo import joint_ppo
        return joint_ppo

    def trainer(self, **kwargs):
        return self.module().JointPPOTrainer(dict(seed=17,update_epochs=2,minibatch_size=2,**kwargs))

    def batch(self,trainer,lengths=(2,0,1,3)):
        task_states=[[actor_state((t+j)/10) for j in range(n)] for t,n in enumerate(lengths)]
        actions=[[j%2 for j in range(n)] for n in lengths]
        with torch.no_grad():
            logs,_=self.module().evaluate_joint(trainer.model,task_states,actions)
            _,values=trainer.model([critic_state(t/10) for t in range(len(lengths))])
        return dict(states=[critic_state(t/10) for t in range(len(lengths))],task_states=task_states,
                    task_actions=actions,log_probs=[float(logs[t]) if n else None for t,n in enumerate(lengths)],
                    values=values.tolist(),rewards=[-.1*(t+1) for t in range(len(lengths))],
                    terminated=[False]*(len(lengths)-1)+[True])

    def test_binary_joint_four_paths_normalize_and_sum_logs(self):
        trainer=self.trainer();model=trainer.model
        with torch.no_grad():
            for p in model.parameters():p.zero_()
            model.policy_head.bias.copy_(torch.tensor([math.log(.25),math.log(.75)]))
        paths=list(itertools.product((0,1),repeat=2))
        states=[[actor_state(),actor_state(.3)] for _ in paths]
        logs,entropy=self.module().evaluate_joint(model,states,paths)
        self.assertEqual(logs.dtype,torch.float64)
        self.assertAlmostEqual(float(logs.exp().sum().detach()),1.,places=6)
        for lp,actions in zip(logs,paths):
            expected=sum(math.log(.25 if a==0 else .75) for a in actions)
            self.assertAlmostEqual(float(lp.detach()),expected,places=6)
        self.assertTrue(torch.allclose(entropy,entropy[0].expand(4)))

    def test_autoregressive_prefix_dependent_paths_normalize(self):
        trainer=self.trainer();paths=list(itertools.product((0,1),repeat=2))
        states=[[actor_state(.1),actor_state(float(prefix[0]))] for prefix in paths]
        with torch.no_grad():
            logs,_=self.module().evaluate_joint(trainer.model,states,paths)
            self.assertAlmostEqual(float(logs.exp().sum()),1.,places=6)
            for index,choices in enumerate(paths):
                conditional,_,_=trainer.model.evaluate_actions(states[index],choices)
                # Float32 MLP kernels may differ by one ULP between ragged
                # flattened evaluation and a two-row call; accumulation is64bit.
                self.assertLess(abs(float(logs[index])-float(conditional.double().sum())),1e-6)

    def test_identical_network_has_joint_ratio_one(self):
        trainer=self.trainer();batch=self.batch(trainer)
        new,_=self.module().evaluate_joint(trainer.model,batch['task_states'],batch['task_actions'])
        for index,old in enumerate(batch['log_probs']):
            if old is not None:self.assertAlmostEqual(float(torch.exp(new[index]-old).detach()),1.,places=10)

    def test_joint_clip_is_once_per_physical_slot(self):
        module=self.module()
        new=torch.log(torch.tensor([1.5,.5],dtype=torch.float64))
        old=torch.zeros(2,dtype=torch.float64);advantages=torch.tensor([2.,-3.])
        # min(1.5*2,1.2*2)=2.4; min(.5*-3,.8*-3)=-2.4 -> loss0.
        loss,kl,clipped,lo,hi=module.clipped_joint_objective(new,old,advantages,.2)
        self.assertAlmostEqual(float(loss),0.,places=7)
        self.assertEqual(float(clipped),1.)
        self.assertAlmostEqual(float(lo),math.log(.5),places=12)
        self.assertAlmostEqual(float(hi),math.log(1.5),places=12)

    def test_physical_gae_not_repeated_by_candidate_count(self):
        trainer=self.trainer();batch=self.batch(trainer,(1,0,5,2))
        prepared=trainer._prepare_joint(batch)
        self.assertEqual(len(prepared['returns']),4)
        self.assertEqual(int(prepared['sampled'].sum()),3)
        from experiments.sca_ppo.ppo import compute_gae
        _,expected=compute_gae(batch['rewards'],batch['values'],batch['terminated'],0.,1.,.95)
        self.assertTrue(torch.equal(prepared['returns'],expected))

    def test_update_changes_exclusive_actor_and_counts_slots(self):
        trainer=self.trainer();batch=self.batch(trainer)
        before=copy.deepcopy(trainer.model.state_dict());result=trainer.update(batch)
        self.assertGreater(result['actor_parameter_delta_l2'],0)
        self.assertGreater(result['parameter_delta_l2'],0)
        self.assertEqual(result['physical_slots'],4)
        self.assertEqual(result['sampled_slots'],3)
        self.assertEqual(result['conditioned_decisions'],6)
        self.assertEqual(result['policy_samples'],6)
        self.assertEqual(result['critic_samples'],8)
        self.assertEqual(result['conditioned_sample_uses'],12)
        self.assertFalse(torch.equal(before['policy_head.weight'],trainer.model.policy_head.weight))

    def test_all_zero_original_proposals_are_still_actor_samples(self):
        trainer=self.trainer();batch=self.batch(trainer,(2,1))
        batch['task_actions']=[[0,0],[0]]
        with torch.no_grad():logs,_=self.module().evaluate_joint(trainer.model,batch['task_states'],batch['task_actions'])
        batch['log_probs']=logs.tolist()
        result=trainer.update(batch)
        self.assertEqual(result['sampled_slots'],2)
        self.assertGreater(result['actor_parameter_delta_l2'],0)

    def test_rejected_ones_remain_original_actor_samples(self):
        trainer=self.trainer();batch=self.batch(trainer,(1,2))
        batch['task_actions']=[[1],[1,1]]
        with torch.no_grad():logs,_=self.module().evaluate_joint(trainer.model,batch['task_states'],batch['task_actions'])
        batch['log_probs']=logs.tolist();batch['approved']=[[],[]]
        result=trainer.update(batch)
        self.assertEqual(result['conditioned_decisions'],3)
        self.assertGreater(result['actor_parameter_delta_l2'],0)

    def test_no_candidates_still_update_critic_not_actor_head(self):
        trainer=self.trainer();result=trainer.update(self.batch(trainer,(0,0)))
        self.assertEqual(result['sampled_slots'],0)
        self.assertEqual(result['actor_parameter_delta_l2'],0)
        self.assertGreater(result['value_parameter_delta_l2'],0)
        self.assertEqual(result['critic_samples'],4)

    def test_teacher_forced_inputs_are_not_mutated_or_sampled(self):
        trainer=self.trainer();batch=self.batch(trainer);saved=copy.deepcopy(batch)
        from unittest.mock import patch
        with patch.object(trainer.model,'act',side_effect=AssertionError('No update sampling')):
            trainer.update(batch)
        self.assertEqual(batch,saved)

    def test_missing_nonterminal_bootstrap_rejected(self):
        trainer=self.trainer();batch=self.batch(trainer);batch['terminated'][-1]=False
        with self.assertRaises(ValueError):trainer.update(batch)
        batch['last_value']=.2;trainer.update(batch)

    def test_invalid_batch_and_nonfinite_values_rejected(self):
        trainer=self.trainer();valid=self.batch(trainer)
        for transform in (
            lambda b:b['states'][0].__setitem__(47,1.),
            lambda b:b['task_states'][0][0].__setitem__(47,0.),
            lambda b:b['task_actions'][0].__setitem__(0,2),
            lambda b:b['log_probs'].__setitem__(1,0.),
            lambda b:b['log_probs'].__setitem__(0,None),
            lambda b:b['rewards'].__setitem__(0,float('nan')),
            lambda b:b['log_probs'].__setitem__(0,float('-inf')),
            lambda b:b['task_actions'][0].pop(),
        ):
            broken=copy.deepcopy(valid);transform(broken)
            with self.assertRaises(ValueError):trainer.update(broken)

    def test_nonzero_entropy_or_changed_contract_rejected(self):
        for cfg in ({'entropy_coef':.01},{'gamma':.99},{'input_dim':27},{'action_dim':7}):
            with self.assertRaises(ValueError):self.module().JointPPOTrainer(cfg)

    def test_joint_ratio_overflow_stops_instead_of_clamping(self):
        with self.assertRaises(ValueError):
            self.module().clipped_joint_objective(torch.tensor([1000.],dtype=torch.float64),
                torch.tensor([0.],dtype=torch.float64),torch.tensor([1.]),.2)

    def test_complete_checkpoint_restores_rng_adam_and_next_update(self):
        trainer=self.trainer();batch=self.batch(trainer);trainer.update(batch)
        root=Path('research_notes/task-ppo-v1/unit-tmp');root.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as tmp:
            path=Path(tmp)/'state.pt';trainer.save(path,{'test':True})
            expected_rng=(random.random(),np.random.rand(),torch.rand(1))
            expected=trainer.update(batch)
            weights=copy.deepcopy(trainer.model.state_dict())
            restored=self.module().JointPPOTrainer.load(path)
            self.assertEqual(restored.metadata,{'test':True})
            self.assertEqual(random.random(),expected_rng[0]);self.assertEqual(np.random.rand(),expected_rng[1])
            self.assertTrue(torch.equal(torch.rand(1),expected_rng[2]))
            actual=restored.update(batch)
            for key in weights:self.assertTrue(torch.equal(weights[key],restored.model.state_dict()[key]))
            self.assertEqual(expected,actual)
            self.assertEqual(restored.optimizer_steps,8)


if __name__=='__main__':unittest.main()
