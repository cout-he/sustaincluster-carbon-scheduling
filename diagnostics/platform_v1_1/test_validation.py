"""Harness regressions: immutable baseline, no plant resizing, honest failures."""
import copy
import tempfile
import unittest
from pathlib import Path

from experiments.platform_v1.inventory import load_config
from .run import variants, make_model, memory_rows, equilibrium, simulate, claim_once


class ValidationTests(unittest.TestCase):
    def test_oat_is_independent_not_cumulative(self):
        cfg=load_config(); original=copy.deepcopy(cfg)
        variants_list=list(variants(cfg,{'power.cpu_idle_w':[30],'thermal.ua_w_per_k_host':[0]}))
        self.assertEqual(variants_list[2][1]['power']['cpu_idle_w'],45)
        self.assertEqual(variants_list[1][1]['thermal']['ua_w_per_k_host'],2)
        self.assertEqual(cfg,original)

    def test_power_sensitivity_retains_same_cooling_plant(self):
        cfg=load_config(); base=make_model(cfg,cfg,'baseline')
        other=copy.deepcopy(cfg);other['power']['cpu_reference_full_w']=240
        model=make_model(other,cfg,'power.cpu_reference_full_w')
        self.assertGreater(model.it_design_kw,base.it_design_kw)
        self.assertEqual(model.coil_rating_kw,base.coil_rating_kw)
        self.assertEqual(model.supply_fan_kw,base.supply_fan_kw)

    def test_memory_candidate_satisfies_quota_but_not_same_idle_savings(self):
        cfg=load_config()
        a=memory_rows(cfg,[{'name':'fewer','dimm_count':8,'dimm_gib':16},
                           {'name':'smaller','dimm_count':16,'dimm_gib':8}],[0])
        self.assertAlmostEqual(a[0]['installed_mem_gb'],22952.305229824)
        self.assertAlmostEqual(a[0]['available_after_os_gb'],20083.267076096)
        self.assertEqual(a[0]['capacity_ok'],True)
        self.assertAlmostEqual(a[1]['it_kw']-a[0]['it_kw'],4.26382978723404)

    def test_steady_state_matches_independent_hand_balance(self):
        cfg=load_config();m=make_model(cfg,cfg,'baseline')
        # Q = 50 kW, fan separate, UA=0: T = Ts + Q/K.
        m.ua_kw_k=0;m.supply_fan_kw=0;m.air_conductance_kw_k=10;m.coil_rating_kw=100
        r=equilibrium(m,50,35,20)
        self.assertAlmostEqual(r['steady_inlet_c'],25)
        self.assertAlmostEqual(r['steady_coil_kw'],50)
        self.assertAlmostEqual(r['time_constant_minutes'],139.16666666666666)

    def test_unsafe_case_is_recorded_not_silently_dropped(self):
        cfg=load_config();m=make_model(cfg,cfg,'baseline')
        r=simulate(m,(0,0,0),35,26,'v1_safety_filter',1,28,400)
        self.assertEqual(r['status'],'no_safe_action')
        self.assertEqual(r['completed_steps'],0)
        self.assertIsNone(r['total_kwh'])

    def test_integral_and_storage_are_reported_separately(self):
        cfg=load_config();m=make_model(cfg,cfg,'baseline')
        r=simulate(m,(0,0,0),25,20,'unfiltered',1,24,400)
        self.assertAlmostEqual(r['it_kwh'],9.904787234042553)
        self.assertAlmostEqual(r['synthetic_carbon_kg'],r['total_kwh']*.4)
        self.assertAlmostEqual(r['storage_kwh_th'],m.c_kj_k*(r['inlet_end_c']-24)/3600)
        self.assertLess(r['independent_balance_residual_kj'],1e-5)

    def test_once_marker_cannot_be_reused(self):
        temp_root=Path(__file__).resolve().parents[2]/'research_notes/platform-v1-1/unit-tmp'
        temp_root.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temp_root) as tmp:
            self.assertTrue(Path(tmp).resolve().is_relative_to(temp_root.resolve()))
            p=Path(tmp)/'once.json'
            claim_once(p,{'status':'started'})
            with self.assertRaises(FileExistsError):claim_once(p,{'status':'started'})


if __name__=='__main__':unittest.main()
