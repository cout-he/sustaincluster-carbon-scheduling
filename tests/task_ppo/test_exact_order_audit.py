"""Raw EDF float ordering must not be reconstructed from normalized features."""
import importlib.util
import unittest
from tests.sca_ppo.test_guards import task


class OrderTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('experiments.task_ppo.exact_order_audit'))
        from experiments.task_ppo import exact_order_audit
        return exact_order_audit

    def test_normalized_deadline_tie_does_not_invert_raw_edf(self):
        module=self.module()
        left=task('later',98.36666666666666,arrival=315)
        right=task('earlier',138.36666666666667,arrival=255)
        jobs={t.task_id:t for t in [left,right]}
        result=module.check_order(['later','earlier'],jobs,315,1.5)
        self.assertTrue(result['passed'])
        self.assertEqual(len(result['normalized_tie_pairs']),1)

    def test_actual_raw_inversion_is_rejected(self):
        module=self.module();a=task('a',10);b=task('b',20)
        result=module.check_order(['b','a'],{'a':a,'b':b},0,1.5)
        self.assertFalse(result['passed'])

    def test_unknown_or_not_yet_arrived_task_is_rejected(self):
        module=self.module();a=task('future',10,arrival=15)
        self.assertFalse(module.check_order(['future'],{'future':a},0,1.5)['passed'])
        self.assertFalse(module.check_order(['unknown'],{},0,1.5)['passed'])


if __name__=='__main__':unittest.main()
