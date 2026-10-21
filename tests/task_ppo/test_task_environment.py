"""Synthetic task/prefix contracts, never real workload or sealed data."""
import copy
from dataclasses import replace
import importlib
import importlib.util
import itertools
import math
import unittest
from unittest.mock import patch

from experiments.platform_v2.environment import run as run_v2
from experiments.sca_ppo.rl_environment import RLEpisode, state_vector
from experiments.sca_ppo.deadline_filter import filter_requests
from experiments.sca_ppo.edf_calendar import simulate
from tests.sca_ppo.test_guards import environment, task


class TaskEpisodeTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('experiments.task_ppo.task_environment'),
                             'TaskEpisode implementation is missing')
        return importlib.import_module('experiments.task_ppo.task_environment')

    def episode(self,jobs,cls=None):
        cls=cls or self.api().TaskEpisode
        base=environment(jobs,sla=1.5)
        idle=run_v2(base.scenario,base.inv,base.cfg,'edf','fixed',1.5,idle=True)
        ep=cls(base.scenario,base.inv,base.cfg,idle,100.,.01);ep.reset()
        return ep

    def finish(self,ep,first=0):
        done=False
        while not done:
            _,ids=ep.prepare_slot()
            for _ in ids:ep.propose(first if not ep.records else 0)
            _,done,_=ep.execute_slot()
        return ep.finish()

    def test_schema_and_hand_calculated_current_task_prefix(self):
        module=self.api();ep=self.episode([task('a',30,(100,1,100)),task('b',60,(50,2,200))])
        value,ids=ep.prepare_slot();self.assertEqual(ids,('a','b'))
        self.assertEqual(len(module.STATE_NAMES),48);self.assertEqual(len(set(module.STATE_NAMES)),48)
        self.assertEqual(value[27:],[0.]*21)
        actor=ep.next_state();self.assertEqual(actor[:27],value[:27]);self.assertEqual(actor[-1],1.)
        expected=[.01,.005,.00625,30/1440,0,45/1440,15/1440,
                  (.005*100+.225)/(50+45),30/1440,30/1440,0,45/4320]
        for actual,want in zip(actor[27:39],expected):self.assertAlmostEqual(actual,want,places=12)
        self.assertEqual(actor[39:47],[.002,0.,0.,0.,0.,0.,0.,0.])
        decision=ep.propose(1);self.assertTrue(decision['approved'])
        second=ep.next_state();self.assertEqual(second[39:42],[.002,.5,.5])
        for actual,want in zip(second[42:47],[.01,.005,.00625,(.5)/(200*24),0.]):
            self.assertAlmostEqual(actual,want,places=12)

    def test_all_reference_candidates_are_processed_in_edf_order_without_sixteen_cap(self):
        jobs=[task(f't{i:02}',120,(1,0,0)) for i in reversed(range(20))]
        ep=self.episode(jobs);_,ids=ep.prepare_slot()
        self.assertEqual(ids,tuple(f't{i:02}' for i in range(20)))
        with self.assertRaises(RuntimeError):ep.execute_slot()
        for identity in ids:
            row=ep.propose(0);self.assertEqual(row['task_id'],identity)
        _,_,record=ep.execute_slot()
        self.assertEqual(len(record['task_decisions']),20)
        self.assertEqual(record['proposed'],())
        self.assertEqual(record['actual_started'],ids)

    def test_no_environment_step_until_all_task_decisions_finish_then_exactly_one(self):
        ep=self.episode([task('a',60),task('b',90)])
        with patch.object(ep.env,'step',wraps=ep.env.step) as step:
            value,ids=ep.prepare_slot()
            before=copy.deepcopy(ep.env.ledger)
            for _ in ids:
                ep.next_state();ep.propose(1)
                self.assertEqual(ep.env.minute,0.);self.assertEqual(ep.env.ledger,before)
                self.assertEqual(len(ep.ledger.seen_miss_ids),0)
                self.assertEqual(len(ep.records),0);step.assert_not_called()
            self.assertEqual(ep.prepare_slot()[0],value)
            reward,done,record=ep.execute_slot()
            self.assertEqual(ep.env.minute,15.);self.assertEqual(step.call_count,1)
            self.assertFalse(done);self.assertEqual(reward,record['reward'])
            self.assertEqual(len(ep.records),1)

    def test_full_all_zero_path_matches_original_edf_executor_and_reward(self):
        jobs=[task('a',30,(10000,0,0)),task('b',16,(10000,0,0),15)]
        ep=self.episode(jobs);new=self.finish(ep)
        old=self.episode(jobs,cls=RLEpisode);done=False
        while not done:
            _,sampled=old.prepare();_,done,_=old.step(0,policy_sampled=sampled)
        legacy=old.finish()
        self.assertEqual(new,legacy)
        self.assertEqual(len(ep.records),288)

    def test_joint_defer_conflict_is_rejected_without_losing_original_proposals(self):
        jobs=[task('a',90,(2500,0,0)),task('b',30,(2500,0,0)),
              task('c',20,(5000,0,0)),task('d',90,(5000,0,0))]
        ep=self.episode(jobs);_,ids=ep.prepare_slot();self.assertEqual(ids,('c','b','a'))
        ep.propose(0)
        b=ep.propose(1);a=ep.propose(1)
        self.assertTrue(b['approved']);self.assertFalse(a['approved'])
        self.assertIn('REFERENCE_ON_TIME_SET_LOST',a['reasons'])
        _,_,record=ep.execute_slot()
        self.assertEqual(record['proposed'],('b','a'));self.assertEqual(record['approved'],('b',))
        self.assertEqual(len(record['checks']),2)
        self.assertEqual([row['action'] for row in record['task_decisions']],[0,1,1])
        self.assertEqual([row['approved'] for row in record['task_decisions']],[False,True,False])
        self.assertEqual(record['internal_candidate_checks'],len(record['checks']))

    def test_incremental_prefix_uses_one_calendar_per_original_defer_request(self):
        module=self.api();ep=self.episode([task(f't{i:02}',120) for i in range(20)])
        calls=[]
        def counted(*args,**kwargs):
            calls.append(args[1] if len(args)>1 else frozenset())
            return simulate(*args,**kwargs)
        with patch.object(module,'simulate',side_effect=counted), \
                patch('experiments.sca_ppo.deadline_filter.simulate',side_effect=counted):
            _,ids=ep.prepare_slot()
            for _ in ids:ep.propose(1)
            self.assertEqual(len(calls),1+len(ids))
            self.assertEqual(ep.env.minute,0.)
            _,_,record=ep.execute_slot()
        self.assertEqual(len(record['checks']),20)
        self.assertEqual(record['internal_candidate_checks'],20)
        self.assertEqual(record['internal_calendar_plans'],43)

    def test_every_binary_vector_matches_original_filter_request_prefixes_and_final_plan(self):
        cases=[
            [task('a',15),task('b',30),task('c',45)],
            [task('a',45,(5000,0,0)),task('b',60,(2500,0,0)),
             task('c',120,(2500,0,0)),task('d',120,(5000,0,0)),task('e',75,(10000,0,0))],
            [task('a',90,(2500,0,0)),task('b',30,(2500,0,0)),
             task('c',20,(5000,0,0)),task('d',90,(5000,0,0))],
        ]
        for jobs in cases:
            base=self.episode(jobs);_,ids=base.prepare_slot()
            for bits in itertools.product((0,1),repeat=len(ids)):
                with self.subTest(tasks=len(jobs),bits=bits):
                    ep=self.episode(jobs);ep.prepare_slot();requested=[]
                    for identity,bit in zip(ids,bits):
                        outcome=ep.propose(bit)
                        if bit:requested.append(identity)
                        exact=filter_requests(ep._slot['snapshot'],ep._slot['reference'],tuple(requested))
                        self.assertEqual(tuple(ep._slot['approved']),exact.approved)
                        if bit:
                            check=exact.checks[-1]
                            self.assertEqual((outcome['approved'],outcome['reasons'],outcome['lost_on_time_ids']),
                                             (check.approved,check.reasons,check.lost_on_time_ids))
                    _,_,record=ep.execute_slot()
                    self.assertEqual(record['approved'],exact.approved)
                    self.assertEqual(record['actual_started'],exact.preview.started)
                    self.assertEqual(record['effective_policy_wait'],exact.preview.policy_wait)
                    self.assertEqual(record['capacity_wait'],exact.preview.capacity_wait)

    def test_incremental_candidate_guard_rejects_only_extension_past_3960(self):
        ep=self.episode([task('long',3945)])
        ep.prepare_slot();first=ep.propose(1);self.assertTrue(first['approved'])
        ep.execute_slot()
        ep.prepare_slot();second=ep.propose(1)
        self.assertFalse(second['approved'])
        self.assertEqual(second['reasons'],('COMPLETION_GUARD_REJECTED',))
        _,_,record=ep.execute_slot()
        self.assertEqual(record['proposed'],('long',));self.assertEqual(record['approved'],())
        self.assertEqual(record['actual_started'],('long',))
        self.assertFalse(record['failure_latched'])

    def test_rejected_request_is_not_replaced_and_all_rejected_is_normal_sampling(self):
        ep=self.episode([task('a',15),task('b',30)])
        _,ids=ep.prepare_slot();self.assertEqual(ids,('a','b'))
        a=ep.propose(1);self.assertFalse(a['approved'])
        ep.propose(0);_,_,record=ep.execute_slot()
        self.assertTrue(record['policy_sampled'])
        self.assertEqual(record['proposed'],('a',));self.assertEqual(record['approved'],())
        self.assertEqual(tuple(row.task_id for row in record['checks']),('a',))
        self.assertEqual(record['actual_started'],('a','b'))

    def test_reject_approve_reject_prefix_matches_one_complete_original_request_replay(self):
        ep=self.episode([task('a',45,(5000,0,0)),task('b',60,(2500,0,0)),
                         task('c',120,(2500,0,0)),task('d',120,(5000,0,0)),
                         task('e',75,(10000,0,0))])
        _,ids=ep.prepare_slot();self.assertEqual(ids,('a','b','c'))
        outcomes=[ep.propose(1) for _ in ids]
        self.assertEqual([row['approved'] for row in outcomes],[False,True,False])
        self.assertEqual([row['approved_prefix_before'] for row in outcomes],[(),(),('b',)])
        _,_,record=ep.execute_slot()
        self.assertEqual(record['proposed'],('a','b','c'));self.assertEqual(record['approved'],('b',))
        self.assertEqual(tuple(check.approved for check in record['checks']),(False,True,False))

    def test_zero_means_no_voluntary_defer_not_guaranteed_start_after_backfill(self):
        specs=[('b',60,2500,0),('c',30,2500,45),('d',15,5000,0),('f',60,5000,15),
               ('g',45,2500,0),('h',45,2500,45),('m',30,5000,45),('p',30,2500,15)]
        ep=self.episode([task(identity,duration,(cpu,0,0),arrival) for identity,duration,cpu,arrival in specs])
        for bits in ((0,1,0),(1,1),(),(1,0)):
            _,ids=ep.prepare_slot();self.assertEqual(len(ids),len(bits))
            for bit in bits:ep.propose(bit)
            _,_,record=ep.execute_slot()
        self.assertEqual(record['candidate_ids'],('c','h'))
        self.assertEqual(record['approved'],('c',))
        self.assertEqual(record['actual_started'],('m',))
        self.assertIn('h',record['capacity_wait'])
        self.assertEqual(record['task_decisions'][1]['action'],0)
        self.assertEqual(ep.env.ledger['h']['policy_wait_minutes'],0)

    def test_each_resource_and_backfill_obey_reference_and_real_capacity(self):
        for axis,cap in enumerate((10000,200,16000)):
            with self.subTest(axis=axis):
                demand=[0,0,0];demand[axis]=cap
                other=[0,0,0];other[(axis+1)%3]=1
                ep=self.episode([task('a',15,tuple(demand)),task('b',30,tuple(demand)),
                                 task('c',45,tuple(other))])
                _,ids=ep.prepare_slot();self.assertEqual(ids,('a','c'))
                for _ in ids:ep.propose(0)
                _,_,record=ep.execute_slot()
                self.assertEqual(record['actual_started'],('a','c'))
                self.assertEqual(record['capacity_wait'],('b',))
                self.assertLessEqual(record['row']['used_'+('cpu','gpu','mem')[axis]],cap)

    def test_future_arrivals_do_not_change_current_features_or_candidates(self):
        current=[task('a',60,(100,1,20)),task('b',90,(200,2,50))]
        one=self.episode(current);two=self.episode(current+[task('future',1,(10000,200,16000),15)])
        self.assertEqual(one.prepare_slot(),two.prepare_slot())
        self.assertEqual(one.next_state(),two.next_state())
        self.assertEqual(one.propose(1),two.propose(1))
        self.assertEqual(one.next_state(),two.next_state())

    def test_future_ci_history_does_not_enter_state_and_background_never_enters_features(self):
        one=self.episode([task('a',60)]);two=self.episode([task('a',60)])
        two.obs['ci_history']=tuple(two.obs['ci_history'])+((0,math.nan),(15,math.inf),(4320,-1e20))
        two.idle_result=copy.deepcopy(two.idle_result)
        two.idle_result['steps'][0]['carbon_kg']=1e20
        self.assertEqual(one.prepare_slot(),two.prepare_slot())
        self.assertEqual(one.next_state(),two.next_state())

    def test_invalid_actions_and_incomplete_slot_do_not_advance_state(self):
        ep=self.episode([task('a',30)])
        with self.assertRaises(RuntimeError):ep.execute_slot()
        for action in (-1,2,True,.5,None):
            with self.assertRaises(ValueError):ep.propose(action)
        self.assertEqual(ep.env.minute,0)
        ep.propose(0)
        with self.assertRaises(RuntimeError):ep.next_state()
        with self.assertRaises(RuntimeError):ep.propose(0)
        ep.execute_slot();self.assertEqual(ep.env.minute,15)

    def test_cg_failure_forces_no_candidates_and_stays_latched_through_full_horizon(self):
        ep=self.episode([task('long',4000)])
        done=False
        while not done:
            state,ids=ep.prepare_slot();self.assertEqual(ids,())
            self.assertEqual(state[26],1.)
            _,done,row=ep.execute_slot();self.assertFalse(row['policy_sampled'])
            if row['minute']==0:self.assertEqual(row['internal_calendar_plans'],2)
        result=ep.finish();self.assertEqual(len(ep.records),288)
        self.assertTrue(result['summary']['failure_latched'])
        self.assertFalse(result['summary']['actual_common_idle_tail_valid'])
        self.assertEqual(result['summary']['completed'],1)
        self.assertFalse(result['summary']['protocol_valid'])

    def test_miss_events_and_reward_conserve_actual_full_episode(self):
        ep=self.episode([task('a',30,(10000,0,0)),task('future',10,(10000,0,0),30)])
        result=self.finish(ep,first=1)
        self.assertEqual(len(ep.records),288)
        self.assertEqual(ep.ledger.seen_miss_ids,frozenset({'future'}))
        summary=result['summary'];self.assertTrue(summary['protocol_valid'])
        self.assertEqual(summary['deadline_missed'],1)
        self.assertAlmostEqual(summary['reward_sum'],-summary['common_incremental_carbon_kg']/100-.01,places=11)
        self.assertEqual(sum(record['new_miss_count'] for record in ep.records),1)

    def test_unfinished_and_censored_failure_is_preserved_not_dropped(self):
        ep=self.episode([task('never_done',5000)])
        result=self.finish(ep)
        self.assertEqual(len(ep.records),288)
        self.assertEqual(result['summary']['unfinished'],1)
        self.assertEqual(result['summary']['deadline_censored'],1)
        self.assertIn('UNFINISHED_TASKS',result['summary']['failure_reasons'])

    def test_reset_discards_task_prefix_and_keeps_input_immutable(self):
        ep=self.episode([task('a',60),task('b',90)])
        obs=copy.deepcopy(ep.obs);original=ep.prepare_slot();ep.propose(1)
        self.assertEqual(ep.obs,obs)
        ep.reset();self.assertEqual(ep.prepare_slot(),original)
        self.assertEqual(len(ep.records),0);self.assertEqual(ep.propose(0)['task_id'],'a')


if __name__=='__main__':unittest.main()
