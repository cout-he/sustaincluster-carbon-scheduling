"""Synthetic engineering checks; never run the real-data acceptance matrix."""
import importlib
import json
import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path


def module(name):
    try:
        return importlib.import_module('experiments.platform_v1.' + name)
    except ModuleNotFoundError as exc:
        raise AssertionError('Missing V1 implementation: ' + name) from exc


def temporary_directory():
    root=Path('research_notes/platform-v1/test-temp')
    root.mkdir(parents=True,exist_ok=True)
    return tempfile.TemporaryDirectory(dir=root)


class InventoryPhysicsTests(unittest.TestCase):
    def setup_model(self, name='OM'):
        inv = module('inventory')
        cfg = inv.load_config()
        equipment = inv.build_inventory(cfg, name)
        return cfg, equipment, module('physics').PhysicalModel(equipment, cfg)

    def test_inventory_and_reserves(self):
        expected = {'OM': (167,50,200,10688), 'OH': (99,33,132,6336), 'OL': (194,78,312,12416)}
        for name, counts in expected.items():
            cfg, inv, model = self.setup_model(name)
            self.assertEqual((inv.hosts,inv.gpu_hosts,inv.gpus,inv.installed_cores), counts)
            self.assertGreaterEqual(inv.installed_cores-inv.reserved_cores,inv.capacity[0])
            self.assertGreaterEqual(inv.installed_mem_gb-inv.reserved_mem_gb,inv.capacity[2])
            self.assertEqual(inv.gpus, inv.capacity[1])
            self.assertEqual(inv.cpu_objects, 2*inv.hosts)
            self.assertEqual(inv.dimm_objects,16*inv.hosts)
            self.assertGreater(model.it_design_kw,model.it_power((0,0,0))['it_kw'])

    def test_component_endpoints_and_bounds(self):
        _,inv,m=self.setup_model()
        lo=m.it_power((0,0,0))
        hi=m.it_power((inv.installed_cores,inv.gpus,inv.installed_mem_gb))
        self.assertAlmostEqual(lo['cpu_kw'],inv.cpu_objects*.045)
        self.assertAlmostEqual(hi['cpu_kw'],inv.cpu_objects*.205)
        self.assertAlmostEqual(lo['gpu_kw'],inv.gpus*.025)
        self.assertAlmostEqual(hi['gpu_kw'],inv.gpus*.250)
        for demand in [(-1,0,0),(0,float('nan'),0),(inv.installed_cores+1,0,0)]:
            with self.assertRaises(ValueError):m.it_power(demand)

    def test_first_law_and_monotone_load(self):
        _,inv,m=self.setup_model()
        for ambient in [-5,15,35,45]:
            for supply in [18,22,26]:
                totals=[]
                for u in [0,.25,.5,.75,1]:
                    it=m.it_power(tuple(u*c for c in inv.capacity))['it_kw']
                    r=m.transition(24,it,ambient,supply)
                    self.assertLess(abs(r['balance_residual_kj']),1e-5)
                    self.assertTrue(all(math.isfinite(r[k]) and r[k]>=0 for k in ['coil_kw','pump_kw','compressor_kw','condenser_fan_kw','hvac_kw']))
                    totals.append(it+r['hvac_kw'])
                self.assertEqual(totals, sorted(totals))

    def test_cooling_weather_and_load_response(self):
        _,inv,m=self.setup_model()
        cold=m.transition(24,60,5,20)
        hot=m.transition(24,60,35,20)
        high=m.transition(24,100,35,20)
        warm=m.transition(24,60,35,24)
        self.assertGreater(hot['compressor_kw'],cold['compressor_kw'])
        self.assertGreater(high['inlet_end_c'],hot['inlet_end_c'])
        self.assertGreater(warm['inlet_end_c'],hot['inlet_end_c'])
        self.assertGreater(warm['cop'],hot['cop'])

    def test_unfiltered_can_be_unsafe_and_filter_is_safe(self):
        _,inv,m=self.setup_model()
        it=m.it_power(inv.capacity)['it_kw']
        t=26
        for _ in range(16):t=m.transition(t,it,40,26)['inlet_end_c']
        self.assertGreater(t,28)
        t=24
        overridden=0
        for _ in range(16):
            applied,r,override=m.safe_transition(t,it,40,26)
            self.assertLessEqual(r['inlet_peak_c'],27+1e-8)
            self.assertGreaterEqual(applied,18)
            overridden+=override;t=r['inlet_end_c']
        self.assertGreater(overridden,0)
        with self.assertRaisesRegex(ValueError,'safe'):
            m.safe_transition(35,it,45,26)

    def test_legacy_inventory_is_scaled_not_old_gpu_list(self):
        _,inv,_=self.setup_model('OM-legacy')
        self.assertEqual(inv.gpus,1000)
        self.assertEqual(inv.capacity,(50000,1000,80000))


class DataExecutionTests(unittest.TestCase):
    def toy(self, tasks=None, horizon=90):
        data=module('data'); inv=module('inventory'); cfg=inv.load_config()
        equipment=replace(inv.build_inventory(cfg,'OM'),capacity=(2,4,100))
        if tasks is None:
            tasks=(data.Task('a',0,20,(1,3,1),'same-job',1,0),
                   data.Task('b',0,15,(1,2,1),'b',1,1),
                   data.Task('c',0,15,(1,1,1),'c',1,2))
        observations=tuple(data.Observation(float(i),200.,25.,float(i//60*60)) for i in range(0,horizon,15))
        scenario=data.Scenario('train-synthetic',tuple(tasks),observations,{'workload_sha256':'synthetic'},24*60)
        return data,cfg,equipment,scenario

    def test_release_backfill_and_wait_attribution(self):
        _,cfg,inv,s=self.toy()
        e=module('environment'); result=e.run(s,inv,cfg,'fifo','fixed',1.5)
        rows={r['task_id']:r for r in result['tasks']}
        self.assertEqual(rows['a']['start_minute'],0)
        self.assertEqual(rows['c']['start_minute'],0)
        self.assertEqual(rows['b']['start_minute'],30)
        self.assertEqual(rows['b']['capacity_wait_minutes'],30)
        self.assertEqual(rows['b']['policy_wait_minutes'],0)
        self.assertEqual(rows['a']['finish_minute'],20)
        self.assertTrue(rows['b']['deadline_missed'])
        self.assertEqual(result['summary']['completed'],3)

    def test_duplicate_positions_preserved(self):
        d,cfg,inv,s=self.toy()
        a=d.Task('same/0',0,15,(1,1,1),'job',2,0)
        b=replace(a,task_id='same/1',source_index=1)
        result=module('environment').run(replace(s,tasks=(a,b)),inv,cfg,'fifo','fixed',1.5)
        self.assertEqual(result['summary']['arrived'],2)
        self.assertEqual(result['steps'][0]['used_cpu'],2)

    def test_action_contract_and_capacity_has_priority(self):
        _,cfg,inv,s=self.toy()
        e=module('environment'); env=e.PlatformEnv(s,inv,cfg,1.5)
        obs=env.reset()
        with self.assertRaises(ValueError):env.step(e.Action(('a',),frozenset(),20))
        obs,done,info=env.step(e.Action(('a','b','c'),frozenset({'b'}),20))
        self.assertEqual(env.ledger['b']['capacity_wait_minutes'],15)
        self.assertEqual(env.ledger['b']['policy_wait_minutes'],0)
        self.assertNotIn('future',obs)
        self.assertEqual(obs['minute'],15)

    def test_policy_wait_separate_and_no_future_threshold(self):
        d,cfg,inv,s=self.toy(tasks=())
        task=d.Task('a',240,30,(1,1,1),'a',1,0)
        samples=tuple(d.Observation(float(i),100 if i<240 else 250,25,float(i//60*60)) for i in range(0,360,15))
        result=module('environment').run(replace(s,tasks=(task,),observations=samples),inv,cfg,'threshold_fifo','fixed',2)
        row=result['tasks'][0]
        self.assertEqual(row['policy_wait_minutes'],30)
        self.assertEqual(row['capacity_wait_minutes'],0)
        self.assertFalse(row['deadline_missed'])

    def test_fixed_horizon_and_integrals(self):
        _,cfg,inv,s=self.toy()
        r=module('environment').run(s,inv,cfg,'edf','fixed',2)
        self.assertEqual(len(r['steps']),6)
        self.assertAlmostEqual(r['summary']['carbon_kg'],sum(x['total_kw']*.25*x['ci_g_per_kwh']/1000 for x in r['steps']))
        self.assertAlmostEqual(r['summary']['total_kwh'],r['summary']['it_kwh']+r['summary']['hvac_kwh'])
        self.assertLess(r['summary']['max_balance_residual_kj'],1e-5)
        r2=module('environment').run(s,inv,cfg,'threshold_edf','variable',2)
        self.assertAlmostEqual(r['summary']['it_kwh'],r2['summary']['it_kwh'])

    def test_unfinished_stays_in_denominator(self):
        d,cfg,inv,s=self.toy(horizon=30)
        s=replace(s,tasks=(d.Task('long',0,75,(1,1,1),'long',1,0),))
        r=module('environment').run(s,inv,cfg,'fifo','fixed',1.5)
        self.assertEqual(r['summary']['arrived'],1)
        self.assertEqual(r['summary']['completed'],0)
        self.assertEqual(r['summary']['unfinished'],1)

    def test_role_rejected_before_payload_read(self):
        with temporary_directory() as temp:
            p=Path(temp); (p/'manifest.json').write_text(json.dumps({'role':'validation','test_access':False}))
            with self.assertRaisesRegex(ValueError,'Train'):module('data').load_train(p,72)

    def test_hourly_hold_never_uses_next_hour(self):
        data=module('data')
        samples=[{'time':f'2023-01-01T0{h}:{m:02}:00+00:00','ci_g_per_kwh':100+h,'temperature_c':v}
                 for h,values in [(0,[10,15,20,25]),(1,[30,35,40,45])] for m,v in zip([0,15,30,45],values)]
        out=data.causal_observations(samples,'2023-01-01T00:00:00+00:00',2)
        self.assertEqual([x.outdoor_c for x in out],[10]*4+[30]*4)
        self.assertEqual([x.observed_minute for x in out],[0]*4+[60]*4)

    def test_missing_and_duplicate_hour_input_rejected(self):
        data=module('data')
        with self.assertRaises(ValueError):data.causal_observations([], '2023-01-01T00:00:00+00:00',1)
        a={'time':'2023-01-01T00:00:00+00:00','ci_g_per_kwh':100,'temperature_c':20}
        with self.assertRaises(ValueError):data.causal_observations([a,a],a['time'],1)

    def test_nonfinite_task_and_wrong_resource_shape_rejected(self):
        d,cfg,inv,s=self.toy()
        for bad in [d.Task('bad',0,float('nan'),(1,1,1),'bad',0,0),
                    d.Task('bad',0,15,(1,),'bad',0,0),
                    d.Task('bad',float('nan'),15,(1,1,1),'bad',0,0)]:
            with self.assertRaises(ValueError):module('environment').PlatformEnv(replace(s,tasks=(bad,)),inv,cfg,1.5)


class AcceptanceGuardTests(unittest.TestCase):
    def test_terminal_debt_guard_applies_to_both_pair_types(self):
        a=module('acceptance')
        self.assertFalse(a.terminal_pairs_ok([{'delta_inlet_final_c':.2}],[{'delta_inlet_final_c':.01}],.05))
        self.assertFalse(a.terminal_pairs_ok([{'delta_inlet_final_c':.01}],[{'delta_inlet_final_c':.2}],.05))
        self.assertTrue(a.terminal_pairs_ok([{'delta_inlet_final_c':.01}],[{'delta_inlet_final_c':.01}],.05))

    def test_once_guard_and_matrix_size(self):
        a=module('acceptance')
        with temporary_directory() as temp:
            p=Path(temp)/'once.json'
            a.claim_once(p,{'protocol':'synthetic'})
            with self.assertRaises(FileExistsError):a.claim_once(p,{'protocol':'synthetic'})
        self.assertEqual(len(a.case_matrix()),144)
        self.assertEqual(len(set(a.case_matrix())),144)


if __name__=='__main__':unittest.main()
