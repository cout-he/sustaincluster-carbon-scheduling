"""Synthetic frozen snapshots only; no external workload is accessed."""
import copy
from datetime import datetime, timedelta
import importlib
import json
from pathlib import Path
import tempfile
import unittest

from experiments.platform_v1.data import COLUMNS, canonical_sha, file_sha, load_train


FRAGMENTS = ('train-20230302T0500-r1-s1.5', 'train-20230404T0500-r1-s1.5',
             'train-20230501T0500-r1-s1.5')
ROOT = Path(__file__).resolve().parents[2]


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True), encoding='utf-8')


def fixture(root, bad_id=False, overflow=False):
    protocol = json.loads((ROOT/'docs/research/platform-v2-final/acceptance-protocol.json').read_text())
    source = 'a'*64
    raw = ['SYNTHETIC-A', 60, 120, '1970-01-01T00:01:00+00:00',
           1, 100, 0, 1, 0, 1, 'Thursday', 3]
    if overflow:
        raw[7] = 20000
    other_bandwidth = list(raw); other_bandwidth[9] = 2
    other_job = list(raw); other_job[0] = 'SYNTHETIC-B'
    for number, fragment in enumerate(FRAGMENTS):
        expected = protocol['inputs'][fragment]
        start = expected['start']; origin = datetime.fromisoformat(start)
        raw_rows = ([raw, raw, other_bandwidth], [raw], [other_job])[number]
        tasks = []
        for index, fields in enumerate(raw_rows):
            tasks.append(dict(task_id=f'{source[:16]}/{start}/0/{index}', arrival_time=start,
                source_job_name=fields[0], source_row=0, source_task_index=index,
                raw_fields=fields, duration=1, cores_req=5, gpu_req=0, mem_req=5*fields[7],
                bandwidth_gb=fields[9], sla_multiplier=1.5, dc_id=1))
        if bad_id and number == 0:
            tasks[0]['task_id'] = 'wrong-but-unique'
        workload = dict(columns=list(COLUMNS), source_sha256=source, start=start,
            end=expected['arrival_end'], tasks=tasks, resource_multiplier=5,
            sla_multiplier=1.5, trace_year=2020)
        environment = dict(start=start, end=(origin+timedelta(hours=72)).isoformat(),
            carbon_source_sha256=expected['carbon_source_sha256'],
            weather_source_sha256=expected['weather_source_sha256'],
            weather_source_timezone=expected['weather_source_timezone'],
            weather_source_utc_offset_seconds=expected['weather_source_utc_offset_seconds'],
            weather_time_basis=expected['weather_source_time_basis'],
            interpolation='ci_previous_observation_weather_linear_utc',
            wet_bulb_transform='upstream_temperature_times_0.9', samples=[
                dict(time=(origin+timedelta(minutes=15*i)).isoformat(),
                     ci_g_per_kwh=100+i, temperature_c=20+i/100, wet_bulb_c=18)
                for i in range(288)])
        folder = root/'scenarios'/fragment; folder.mkdir(parents=True)
        for name, payload in [('workload', workload), ('environment', environment)]:
            write_json(folder/(name+'.json'), dict(payload=payload, sha256=canonical_sha(payload)))
            expected[name+'_sha256'] = file_sha(folder/(name+'.json'))
        manifest = dict(role='train', test_access=False, scenario_id=fragment,
            schema='joint-v2-training-scenario-v1', files={
                name+'.json': expected[name+'_sha256'] for name in ('workload', 'environment')})
        write_json(folder/'manifest.json', manifest)
        expected['manifest_sha256'] = file_sha(folder/'manifest.json')
        expected['annual_workload_sha256'] = source
        expected['tasks'] = len(tasks)
        # Use the actual role-checked loader to build a consistent frozen fixture.
        loaded = load_train(folder, expected=expected)
        expected['task_identity_sha256'] = canonical_sha([
            (t.task_id, t.arrival_minute, t.duration_minutes, t.demand) for t in loaded.tasks])
    return protocol


class AuditTests(unittest.TestCase):
    def audit_module(self):
        try:
            return importlib.import_module('experiments.sca_ppo.e1_audit')
        except ModuleNotFoundError:
            self.fail('The frozen Train audit implementation is missing')

    def test_reports_raw_and_original_identity_repeats_without_dropping_tasks(self):
        with tempfile.TemporaryDirectory(dir=ROOT, prefix='.e1-audit-test-') as tmp:
            pool = Path(tmp); protocol = fixture(pool)
            audit = self.audit_module().audit_pool(pool, protocol=protocol)
            self.assertEqual(audit['total_tasks'], 5)
            self.assertEqual(audit['fragments'][0]['raw_content']['duplicate_excess'], 1)
            self.assertEqual(audit['fragments'][0]['original_job_start_end']['duplicate_excess'], 2)
            self.assertEqual(audit['fragments'][0]['original_identity_repeats']['differing_fields'],
                             {'bandwidth_gb': 1})
            self.assertEqual(audit['cross_fragment'][0]['shared_raw_content'], 1)
            self.assertEqual(audit['fragments'][0]['environment']['hourly_anchors'], 72)
            self.assertTrue(audit['fragments'][0]['environment']['causal_previous_hour_hold'])
            self.assertTrue(audit['passed'])
            self.assertEqual(audit, self.audit_module().audit_pool(pool, protocol=protocol))

    def test_rejects_wrong_role_before_missing_payload_access(self):
        with tempfile.TemporaryDirectory(dir=ROOT, prefix='.e1-audit-test-') as tmp:
            pool = Path(tmp); protocol = fixture(pool)
            folder = pool/'scenarios'/FRAGMENTS[0]
            manifest = json.loads((folder/'manifest.json').read_text()); manifest['role'] = 'dev'
            write_json(folder/'manifest.json', manifest); (folder/'workload.json').unlink()
            with self.assertRaisesRegex(ValueError, 'Train scenarios'):
                self.audit_module().audit_pool(pool, protocol=protocol)

    def test_rejects_changed_frozen_identity_and_provenance(self):
        with tempfile.TemporaryDirectory(dir=ROOT, prefix='.e1-audit-test-') as tmp:
            pool = Path(tmp); protocol = fixture(pool)
            for key, wrong in [('task_identity_sha256', '0'*64), ('start', '2023-01-01T00:00:00+00:00')]:
                changed = copy.deepcopy(protocol); changed['inputs'][FRAGMENTS[0]][key] = wrong
                with self.assertRaisesRegex(ValueError, 'identity|provenance'):
                    self.audit_module().audit_pool(pool, protocol=changed)

    def test_rejects_noncanonical_source_identity(self):
        with tempfile.TemporaryDirectory(dir=ROOT, prefix='.e1-audit-test-') as tmp:
            pool = Path(tmp); protocol = fixture(pool, bad_id=True)
            with self.assertRaisesRegex(ValueError, 'source identity'):
                self.audit_module().audit_pool(pool, protocol=protocol)

    def test_rejects_changed_configuration_before_payload_reads(self):
        with tempfile.TemporaryDirectory(dir=ROOT, prefix='.e1-audit-test-') as tmp:
            pool = Path(tmp); protocol = fixture(pool)
            protocol['source_hashes']['experiments/platform_v2/config.json'] = '0'*64
            with self.assertRaisesRegex(ValueError, 'configuration'):
                self.audit_module().audit_pool(pool, protocol=protocol)

    def test_rejects_unlisted_protocol_fragment(self):
        with tempfile.TemporaryDirectory(dir=ROOT, prefix='.e1-audit-test-') as tmp:
            pool = Path(tmp); protocol = fixture(pool)
            protocol['inputs']['train-unapproved'] = {}
            with self.assertRaisesRegex(ValueError, 'whitelist'):
                self.audit_module().audit_pool(pool, protocol=protocol)

    def test_rejects_intrinsically_oversized_task(self):
        with tempfile.TemporaryDirectory(dir=ROOT, prefix='.e1-audit-test-') as tmp:
            pool = Path(tmp); protocol = fixture(pool, overflow=True)
            with self.assertRaisesRegex(ValueError, 'capacity'):
                self.audit_module().audit_pool(pool, protocol=protocol)

    def test_rejects_missing_past_hourly_anchor_in_otherwise_consistent_freeze(self):
        with tempfile.TemporaryDirectory(dir=ROOT, prefix='.e1-audit-test-') as tmp:
            pool = Path(tmp); protocol = fixture(pool)
            folder = pool/'scenarios'/FRAGMENTS[0]
            environment = json.loads((folder/'environment.json').read_text())
            del environment['payload']['samples'][4]
            environment['sha256'] = canonical_sha(environment['payload'])
            write_json(folder/'environment.json', environment)
            expected = protocol['inputs'][FRAGMENTS[0]]
            expected['environment_sha256'] = file_sha(folder/'environment.json')
            manifest = json.loads((folder/'manifest.json').read_text())
            manifest['files']['environment.json'] = expected['environment_sha256']
            write_json(folder/'manifest.json', manifest)
            expected['manifest_sha256'] = file_sha(folder/'manifest.json')
            with self.assertRaisesRegex(ValueError, 'Missing past hourly'):
                self.audit_module().audit_pool(pool, protocol=protocol)


if __name__ == '__main__':
    unittest.main()
