"""Independent cases for inventory, reference EIR, capacity and reused ledger."""
import copy
import math
import unittest
from experiments.platform_v1.data import Task,Observation,Scenario
from experiments.platform_v1.environment import Action
from experiments.platform_v2.inventory import load_config,build_inventory
from experiments.platform_v2.chiller import Chiller
from experiments.platform_v2.physics import PhysicalModel
from experiments.platform_v2.environment import PlatformEnv


class FinalPlatformTests(unittest.TestCase):
    def setUp(self):self.cfg=load_config()

    def test_inventory_memory_counts_and_psu_cover_all_quotas(self):
        for name,hosts,gpus,dimm in [('OM',167,200,8),('OH',99,132,16),('OL',194,312,8)]:
            i=build_inventory(self.cfg,name)
            self.assertEqual((i.hosts,i.gpus,i.dimm_gib),(hosts,gpus,dimm))
            self.assertEqual(i.cpu_objects,2*hosts);self.assertEqual(i.dimm_objects,16*hosts)
            self.assertGreaterEqual(i.quota_excluded_mem_gb,0);self.assertEqual(i.psu_objects,2*hosts)

    def test_memory_occupation_is_not_access_activity_or_power(self):
        m=PhysicalModel(build_inventory(self.cfg,'OM'),self.cfg)
        self.assertEqual(m.it_power((0,0,0)),m.it_power((0,0,16000)))
        self.assertAlmostEqual(m.it_power((0,0,0))['dimm_kw'],8.016)
        for axis in [0,1]:
            first=[0,0,0];last=[0,0,0];last[axis]=m.inv.capacity[axis]
            self.assertGreater(m.it_power(last)['it_kw'],m.it_power(first)['it_kw'])

    def test_curve_domain_water_reset_and_no_hot_extrapolation(self):
        m=PhysicalModel(build_inventory(self.cfg,'OM'),self.cfg);c=m.chiller
        self.assertEqual(c.water_temperature(18),6);self.assertEqual(c.water_temperature(26),10)
        cold=c.performance(6.67,5,50);ref=c.performance(6.67,23.89,50)
        self.assertEqual(cold['available_capacity_kw'],ref['available_capacity_kw'])
        self.assertTrue(cold['low_ambient_boundary'])
        hot=c.performance(6.67,55,50)
        self.assertEqual(hot['available_capacity_kw'],0);self.assertEqual(hot['unmet_evaporator_kw'],50)
        self.assertFalse(hot['curve_domain_valid'])
        with self.assertRaises(ValueError):c.performance(13,35,10)

    def test_reference_capacity_eir_and_part_load_cycling(self):
        c=Chiller(1,self.cfg)
        full=c.performance(6.67,35,1000)
        self.assertAlmostEqual(full['available_capacity_kw'],105.52622757829755,places=7)
        half=c.performance(6.67,35,20);low=c.performance(6.67,35,1)
        self.assertAlmostEqual(low['operating_plr'],0.1)
        self.assertLess(low['cycling_fraction'],1)
        self.assertAlmostEqual(low['compressor_kw']/half['compressor_kw'],0.05)
        self.assertAlmostEqual(half['package_kw'],half['compressor_kw']+half['condenser_fan_kw'])
        self.assertEqual(c.performance(6.67,35,0)['compressor_kw'],0)
        self.assertGreater(c.performance(6.67,25,20)['available_capacity_kw'],c.performance(6.67,50,20)['available_capacity_kw'])

    def test_capacity_shortfall_drives_heat_and_safe_infeasibility(self):
        m=PhysicalModel(build_inventory(self.cfg,'OM'),self.cfg)
        r=m.transition(26,500,50,18)
        self.assertGreater(r['unmet_cooling_kw'],0);self.assertGreater(r['inlet_end_c'],28)
        self.assertLessEqual(r['evaporator_kw'],r['available_capacity_kw']+1e-8)
        action,filtered,_=m.safe_transition(26,500,50,20)
        self.assertFalse(filtered['cooling_feasible']);self.assertGreater(filtered['inlet_peak_c'],27)

    def test_first_law_and_pump_heat_not_double_counted(self):
        m=PhysicalModel(build_inventory(self.cfg,'OM'),self.cfg)
        r=m.transition(24,90,35,20)
        self.assertAlmostEqual(r['evaporator_kw'],r['coil_kw']+r['pump_kw'],places=9)
        self.assertAlmostEqual(r['heat_rejection_kw'],r['evaporator_kw']+r['compressor_kw']+r['condenser_fan_kw'],places=9)
        self.assertLess(abs(r['balance_residual_kj']),1e-5)
        self.assertLess(abs(r['chiller_balance_residual_kj']),1e-5)
        self.assertGreater(r['water_flow_kg_s'],0)

    def test_current_observation_authority_and_fixed_window_periods(self):
        tasks=(Task('a',0,16,(1,1,1),'source',1,1),Task('b',15,15,(1,1,1),'source',2,2))
        scenario=Scenario('synthetic',tasks,tuple(Observation(15*n,400,25,15*n//60*60) for n in range(288)),{},1440)
        env=PlatformEnv(scenario,build_inventory(self.cfg,'OM'),self.cfg,1.5,'fixed')
        obs=env.reset();self.assertEqual(len(obs['pending']),1)
        self.assertNotIn('future',obs)
        with self.assertRaises(ValueError):env.step(Action(('a',),frozenset(),26))
        while env.minute<env.horizon:
            obs=env.observation();env.step(Action(tuple(t.task_id for t in obs['pending']),frozenset(),20))
        result=env.result();s=result['summary']
        self.assertEqual(s['completed'],2)
        self.assertAlmostEqual(s['arrival_total_kwh']+s['drain_total_kwh'],s['total_kwh'])
        self.assertAlmostEqual(s['arrival_carbon_kg']+s['drain_carbon_kg'],s['carbon_kg'])
        self.assertEqual(s['horizon_hours'],72)


if __name__=='__main__':unittest.main()
