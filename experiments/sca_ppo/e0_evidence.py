"""Reproduce synthetic-only E0 tests and small timing probes; no workload I/O.

Run: python -m experiments.sca_ppo.e0_evidence
Outputs are small engineering evidence, never algorithm performance results.
"""
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import platform
import statistics
import unittest
from datetime import datetime, timezone
from time import perf_counter_ns

from experiments.platform_v1.data import Task
from .adapter import ResearchAdapter

BASE = 'e372f502fb58aaa5a742e8efb4ced254507ab298'
DESIGN_SHA256 = '27d8c948ef71652abb1295049e368351ff6340510c8244c53324dfc8058f8bd9'
ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT/'docs/research/sca-ppo/e0-results'


def percentile(values, q):
    # Nearest rank; explicit for tiny samples. No interpolation/extrapolation.
    v = sorted(values)
    return v[max(0, math.ceil(q*len(v))-1)]


def stats(values):
    return dict(count=len(values), mean_ms=statistics.mean(values),
                p50_ms=percentile(values, .5), p95_ms=percentile(values, .95), max_ms=max(values))


def write_csv(path, rows):
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


class RecordedTests(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs); self.rows = []

    def addSuccess(self, test):
        super().addSuccess(test); self.rows.append(dict(test=test.id(), status='PASS'))

    def addFailure(self, test, err):
        super().addFailure(test, err); self.rows.append(dict(test=test.id(), status='FAIL'))

    def addError(self, test, err):
        super().addError(test, err); self.rows.append(dict(test=test.id(), status='ERROR'))


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    suites = unittest.TestSuite()
    for name in ('tests.sca_ppo.test_guards', 'tests.platform_v1.test_platform',
                 'tests.platform_v1.test_saved_verification', 'tests.platform_v2.test_platform',
                 'tests.platform_v2.test_verification', 'tests.platform_v2.test_acceptance'):
        suites.addTests(unittest.defaultTestLoader.loadTestsFromName(name))
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2, resultclass=RecordedTests).run(suites)
    write_csv(OUTPUT/'tests.csv', result.rows)
    if not result.wasSuccessful():
        (OUTPUT/'failed-test-output.txt').write_text(stream.getvalue(), encoding='utf-8')
        raise SystemExit(stream.getvalue())

    decisions, candidates = [], []
    for n in (8, 32, 128):
        jobs = tuple(Task(f'SYNTHETIC-{i:03}', 0, 30+15*(i % 7),
                          (1+i % 4, (i % 3)/2, 1+i % 5), 'ARTIFICIAL', i, i) for i in range(n))
        observation = dict(minute=0, capacity=(16, 8, 64), sla_multiplier=3, pending=jobs, running=())
        for repeat in range(5):
            for action in range(7):
                adapter = ResearchAdapter()
                start = perf_counter_ns()
                adapted = adapter.decide(observation, action_id=action)
                elapsed_ms = (perf_counter_ns()-start)/1e6
                ref = adapted.reference; req = adapted.proposed; decision = adapted.filter
                common = dict(case_id=f'ARTIFICIAL-N{n}', task_count=n, repeat=repeat,
                              action_id=action, frozen_base_sha=BASE, units='ms/minute/abstract-resources')
                decisions.append(dict(**common, reference_ms=ref.elapsed_ns/1e6,
                    filter_ms=decision.elapsed_ns/1e6,
                    decision_ms=elapsed_ms,
                    proposed=len(req), approved=len(decision.approved),
                    effective_policy_wait=len(decision.preview.policy_wait),
                    capacity_wait=len(decision.preview.capacity_wait),
                    actual_started_preview=len(decision.preview.started),
                    reference_dispatch_slots=ref.dispatch_slots, reference_task_checks=ref.task_checks,
                    reference_max_release_minute=ref.max_release))
                for c in decision.checks:
                    candidates.append(dict(**common, candidate_id=c.task_id,
                        trial_defer='|'.join(c.trial_defer), approved=c.approved,
                        reasons='|'.join(c.reasons), elapsed_ms=c.elapsed_ns/1e6,
                        calendar_ms=c.calendar_elapsed_ns/1e6, dispatch_slots=c.dispatch_slots,
                        task_checks=c.task_checks, max_release_minute=c.max_release))
    write_csv(OUTPUT/'timing-decisions.csv', decisions)
    write_csv(OUTPUT/'timing-candidates.csv', candidates)
    by_size = {}
    for n in (8, 32, 128):
        d = [r for r in decisions if r['task_count'] == n]
        c = [r for r in candidates if r['task_count'] == n]
        by_size[str(n)] = dict(reference=stats([r['reference_ms'] for r in d]),
                              candidate=stats([r['elapsed_ms'] for r in c]),
                              decision=stats([r['decision_ms'] for r in d]))
    paths = sorted((ROOT/'experiments/sca_ppo').glob('*.py')) + sorted((ROOT/'tests/sca_ppo').glob('*.py'))
    manifest = dict(schema='sca-ppo-e0-synthetic-evidence-v1', frozen_base_sha=BASE,
        design_sha256=DESIGN_SHA256, generated_utc=datetime.now(timezone.utc).isoformat(),
        python=platform.python_version(), os=platform.platform(),
        no_real_workload=True, no_training=True, synthetic_timing_sizes=[8,32,128], repeats=5,
        percentile_method='nearest-rank ceil(q*n), per-size, all 7 actions pooled',
        test_count=result.testsRun, passed=len(result.rows),
        decision_count=len(decisions), candidate_check_count=len(candidates),
        file_sha256={str(p.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        timings=by_size)
    (OUTPUT/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k: manifest[k] for k in ('test_count', 'decision_count', 'candidate_check_count', 'timings')}, indent=2))


if __name__ == '__main__': main()
