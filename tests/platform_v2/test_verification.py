"""Audit must detect independent corruption, including idle-background rows."""
import copy
import unittest
from experiments.platform_v1.data import Task,Observation,Scenario
from experiments.platform_v2.inventory import load_config,build_inventory
from experiments.platform_v2.environment import run
from experiments.platform_v2.verification import verify_result


class VerificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg=load_config();cls.inv=build_inventory(cls.cfg,'OM')
        cls.scenario=Scenario('synthetic',
            (Task('a',0,16,(1,1,1),'source',1,1),Task('b',15,15,(1,1,1),'source',2,2)),
            tuple(Observation(15*n,400,25,15*n//60*60) for n in range(288)),{},1440)
        cls.result=run(cls.scenario,cls.inv,cls.cfg,'edf','fixed',1.5)
        cls.idle=run(cls.scenario,cls.inv,cls.cfg,'edf','fixed',1.5,idle=True)

    def audit(self,result,idle=False):
        return verify_result(result,self.scenario,self.inv,self.cfg,1.5,idle)

    def test_valid_workload_and_idle(self):
        self.assertEqual(self.audit(self.result)['errors'],[])
        self.assertEqual(self.audit(self.idle,True)['errors'],[])

    def test_power_carbon_and_summary_corruption_including_idle_rejected(self):
        for field in ['cpu_kw','total_kw','carbon_kg','storage_delta_kj','used_gpu']:
            result=copy.deepcopy(self.result);result['steps'][1][field]+=1
            self.assertTrue(self.audit(result)['errors'],field)
        result=copy.deepcopy(self.idle);result['summary']['total_kwh']+=1
        self.assertTrue(self.audit(result,True)['errors'])

    def test_task_identity_duration_and_missing_slots_rejected(self):
        result=copy.deepcopy(self.result);result['tasks'][0]['finish_minute']+=1
        self.assertTrue(self.audit(result)['errors'])
        result=copy.deepcopy(self.result);result['steps'].pop()
        self.assertTrue(self.audit(result)['errors'])


if __name__=='__main__':unittest.main()
