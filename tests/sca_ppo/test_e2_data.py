"""Synthetic snapshots only: protect E2 authorization and input isolation."""
import copy
from datetime import datetime, timedelta, timezone
import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.platform_v1.data import COLUMNS, canonical_sha, file_sha

ROOT = Path(__file__).resolve().parents[2]
DATES = ('20230102', '20230103', '20230104', '20230105',
         '20230201', '20230202', '20230203', '20230204',
         '20230301', '20230302', '20230303', '20230304',
         '20230403', '20230404', '20230501', '20230503')
IDS = tuple('train-'+day+'T0500-r1-s1.5' for day in DATES)


def write(path, value):
    path.write_text(json.dumps(value, sort_keys=True), encoding='utf-8')


def refresh(folder, name, payload):
    write(folder/(name+'.json'), dict(payload=payload, sha256=canonical_sha(payload)))
    manifest = json.loads((folder/'manifest.json').read_text())
    manifest['files'][name+'.json'] = file_sha(folder/(name+'.json'))
    write(folder/'manifest.json', manifest)


def fixture(pool, *, share=None, duplicate=False):
    """Sixteen small real-loader inputs, with no sealed role payload present."""
    source = 'a'*64
    for index, (day, scenario_id) in enumerate(zip(DATES, IDS)):
        origin = datetime.strptime(day+'0500', '%Y%m%d%H%M').replace(tzinfo=timezone.utc)
        start = origin.isoformat()
        raw_index = share[0] if share and index == share[1] else index
        raw = [f'SYNTHETIC-{raw_index}', 1000+raw_index, 1060+raw_index,
               '1970-01-01T00:01:00+00:00', 1, 100, 0, 1, 0, 1, 'Thursday', 3]
        rows = [raw, raw] if duplicate and index == 0 else [raw]
        tasks = [dict(task_id=f'{source[:16]}/{start}/0/{i}', arrival_time=start,
                      source_job_name=r[0], source_row=0, source_task_index=i,
                      raw_fields=r, duration=1, cores_req=5, gpu_req=0, mem_req=5,
                      bandwidth_gb=1, sla_multiplier=1.5, dc_id=1)
                 for i, r in enumerate(rows)]
        workload = dict(columns=list(COLUMNS), source_sha256=source, start=start,
                        end=(origin+timedelta(hours=24)).isoformat(), tasks=tasks,
                        resource_multiplier=5, sla_multiplier=1.5, trace_year=2020)
        environment = dict(start=start, end=(origin+timedelta(hours=72)).isoformat(),
            carbon_source_sha256='b'*64, weather_source_sha256='c'*64,
            weather_source_timezone='UTC', weather_source_utc_offset_seconds=0,
            weather_time_basis='UTC', interpolation='ci_previous_observation_weather_linear_utc',
            wet_bulb_transform='upstream_temperature_times_0.9', samples=[
                dict(time=(origin+timedelta(minutes=15*i)).isoformat(),
                     ci_g_per_kwh=100, temperature_c=20, wet_bulb_c=18)
                for i in range(288)])
        folder = pool/'scenarios'/scenario_id
        folder.mkdir(parents=True)
        for name, payload in [('workload', workload), ('environment', environment)]:
            write(folder/(name+'.json'), dict(payload=payload, sha256=canonical_sha(payload)))
        write(folder/'manifest.json', dict(role='train', test_access=False,
              scenario_id=scenario_id, schema='joint-v2-training-scenario-v1', files={
                  name+'.json': file_sha(folder/(name+'.json')) for name in ('workload', 'environment')}))
    # Merely seeing an unselected variant or sealed role directory must not read it.
    (pool/'scenarios'/'dev-sealed').mkdir()
    (pool/'scenarios'/'train-20230102T0500-r0.75-s1.5').mkdir()


class E2DataTests(unittest.TestCase):
    def module(self):
        try:
            return importlib.import_module('experiments.sca_ppo.e2_data')
        except ModuleNotFoundError:
            self.fail('E2 admission module is missing')

    def setUp(self):
        scratch = ROOT/'research_notes/sca-ppo/unit-tmp'
        scratch.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(self.temporary.cleanup)
        self.pool = Path(self.temporary.name)

    def audit(self, **kwargs):
        module = self.module()
        with patch.object(module, 'AUTHORIZED_SCENARIOS', self.pool/'scenarios'):
            return module.audit_pool(self.pool, **kwargs)

    def test_chronological_components_split_without_dropping_duplicate_rows(self):
        fixture(self.pool, duplicate=True)
        report = self.audit()
        self.assertTrue(report['passed'])
        self.assertEqual(report['total_tasks'], 17)
        self.assertEqual(report['train_ids'], list(IDS[:12]))
        self.assertEqual(report['holdout_ids'], list(IDS[12:]))
        self.assertEqual([len(c['fragment_ids']) for c in report['time_components']], [4, 4, 4, 2, 2])
        self.assertEqual(report['aggregate']['raw_content']['duplicate_excess'], 1)
        self.assertEqual(report['aggregate']['original_job_start_end']['duplicate_excess'], 1)
        self.assertEqual(report['cross_split']['shared_original_job_start_end'], 0)
        self.assertEqual(report['cross_split']['overlapping_environment_minutes'], 0)
        self.assertEqual(report['inputs'][IDS[0]]['tasks'], 2)
        self.assertEqual(len(report['inputs'][IDS[0]]['task_identity_sha256']), 64)
        self.assertEqual(report['fragments'][0]['environment']['samples_in_72_hours'], 288)
        self.assertEqual(report['read_boundary']['payload_fragment_ids'], list(IDS))
        self.assertEqual(report, self.audit())

    def test_shared_original_job_across_boundary_moves_entire_contiguous_component_to_holdout(self):
        fixture(self.pool, share=(11, 12))
        report = self.audit()
        self.assertTrue(report['passed'])
        self.assertEqual(report['train_ids'], list(IDS[:8]))
        self.assertEqual(report['holdout_ids'], list(IDS[8:]))
        self.assertEqual(report['cross_split']['shared_original_job_start_end'], 0)
        self.assertEqual(report['aggregate']['original_job_start_end']['duplicate_excess'], 1)

    def test_identity_bridge_covering_all_months_reports_blocker_instead_of_discarding_windows(self):
        fixture(self.pool, share=(0, 15))
        report = self.audit()
        self.assertFalse(report['passed'])
        self.assertEqual(report['train_ids'], [])
        self.assertEqual(report['holdout_ids'], list(IDS))
        self.assertEqual(report['usable_window_count'], 16)
        self.assertTrue(report['blockers'])

    def test_wrong_role_or_test_access_is_rejected_before_missing_payload_is_opened(self):
        fixture(self.pool)
        folder = self.pool/'scenarios'/IDS[0]
        (folder/'workload.json').unlink()
        for updates in [dict(role='dev'), dict(test_access=True), dict(scenario_id=IDS[1])]:
            manifest = json.loads((folder/'manifest.json').read_text())
            manifest.update(role='train', test_access=False, scenario_id=IDS[0])
            manifest.update(updates)
            write(folder/'manifest.json', manifest)
            with self.subTest(updates=updates), self.assertRaisesRegex(ValueError, 'authorized|identity'):
                module = self.module()
                with patch.object(module, 'AUTHORIZED_SCENARIOS', self.pool/'scenarios'):
                    module.load_authorized(folder)

    def test_unlisted_id_or_wrong_parent_is_rejected_without_manifest(self):
        module = self.module()
        with patch.object(module, 'AUTHORIZED_SCENARIOS', self.pool/'scenarios'):
            for path in [self.pool/'scenarios'/'dev-sealed', self.pool/'outside'/IDS[0],
                         self.pool/'scenarios'/'train-20230102T0500-r1-s2']:
                with self.subTest(path=path), self.assertRaisesRegex(ValueError, 'authorized'):
                    module.load_authorized(path)

    def test_missing_window_retains_invalid_entry_and_blocks_admission(self):
        fixture(self.pool)
        folder = self.pool/'scenarios'/IDS[0]
        (folder/'workload.json').unlink()
        report = self.audit()
        self.assertFalse(report['passed'])
        self.assertEqual(report['usable_window_count'], 15)
        self.assertEqual(report['invalid_windows'][0]['fragment_id'], IDS[0])
        self.assertEqual(report['train_ids'], [])

    def test_wrong_legacy_mapping_is_reported_as_invalid(self):
        fixture(self.pool)
        folder = self.pool/'scenarios'/IDS[0]
        workload = json.loads((folder/'workload.json').read_text())['payload']
        workload['tasks'][0]['cores_req'] = 6
        refresh(folder, 'workload', workload)
        report = self.audit()
        self.assertFalse(report['passed'])
        self.assertIn('mapping', report['invalid_windows'][0]['error'])

    def test_noncanonical_stable_id_and_missing_quarter_hour_are_rejected(self):
        fixture(self.pool)
        folder = self.pool/'scenarios'/IDS[0]
        workload = json.loads((folder/'workload.json').read_text())['payload']
        workload['tasks'][0]['task_id'] = 'wrong-but-unique'
        refresh(folder, 'workload', workload)
        other = self.pool/'scenarios'/IDS[1]
        environment = json.loads((other/'environment.json').read_text())['payload']
        del environment['samples'][1]
        refresh(other, 'environment', environment)
        report = self.audit()
        self.assertEqual(report['usable_window_count'], 14)
        self.assertEqual([w['fragment_id'] for w in report['invalid_windows']], list(IDS[:2]))

    def test_expected_inputs_freeze_provenance_count_and_task_identity(self):
        fixture(self.pool)
        report = self.audit()
        module = self.module()
        with patch.object(module, 'AUTHORIZED_SCENARIOS', self.pool/'scenarios'):
            loaded = module.load_authorized(self.pool/'scenarios'/IDS[0], report['inputs'][IDS[0]])
            self.assertEqual(len(loaded.tasks), 1)
            for key, value in [('tasks', 2), ('task_identity_sha256', '0'*64), ('start', 'bad')]:
                expected = copy.deepcopy(report['inputs'][IDS[0]])
                expected[key] = value
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'identity|provenance'):
                    module.load_authorized(self.pool/'scenarios'/IDS[0], expected)

    def test_window_id_cannot_hide_a_different_arrival_date(self):
        fixture(self.pool)
        folder = self.pool/'scenarios'/IDS[0]
        workload = json.loads((folder/'workload.json').read_text())['payload']
        workload['start'] = '2023-01-03T05:00:00+00:00'
        workload['end'] = '2023-01-04T05:00:00+00:00'
        workload['tasks'][0]['arrival_time'] = workload['start']
        workload['tasks'][0]['task_id'] = 'a'*16+'/'+workload['start']+'/0/0'
        refresh(folder, 'workload', workload)
        environment = json.loads((folder/'environment.json').read_text())['payload']
        for key in ('start', 'end'):
            environment[key] = (datetime.fromisoformat(environment[key])+timedelta(days=1)).isoformat()
        for sample in environment['samples']:
            sample['time'] = (datetime.fromisoformat(sample['time'])+timedelta(days=1)).isoformat()
        refresh(folder, 'environment', environment)
        module = self.module()
        with patch.object(module, 'AUTHORIZED_SCENARIOS', self.pool/'scenarios'):
            with self.assertRaisesRegex(ValueError, 'date|start'):
                module.load_authorized(folder)


if __name__ == '__main__':
    unittest.main()
