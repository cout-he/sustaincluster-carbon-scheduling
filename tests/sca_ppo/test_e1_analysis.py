"""Generated local results only; these tests never load input datasets."""
import csv
import importlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
FRAGMENT = 'train-20230302T0500-r1-s1.5'


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def write_lines(path, rows):
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows), encoding='utf-8')


def read_csv(path):
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def fixture(output, *, proof=True, group_pass=True):
    group = output/(FRAGMENT+'__OM'); summaries = []
    for arm in 'ABC':
        folder = group/arm; folder.mkdir(parents=True)
        starts = [0, 15, 90] if arm != 'B' else [30, 45, 120]
        tasks = [dict(task_id=identity, state='completed', start_minute=start,
            finish_minute=start+10, deadline_minute=deadline,
            deadline_missed=start+10 > deadline, deadline_censored=False,
            policy_wait_minutes=30 if arm == 'B' and identity == 'own' else 0,
            capacity_wait_minutes=45 if arm == 'B' and identity == 'indirect' else 0,
            jct_minutes=start+10-arrival)
            for identity, start, deadline, arrival in zip(
                ('own', 'indirect', 'future'), starts, (20, 50, 125), (0, 0, 90))]
        summary = dict(fragment=FRAGMENT, capacity='OM', arm=arm,
            case_id=FRAGMENT+'__OM__'+arm, source_sha='SYNTHETIC',
            arrived=3, completed=3, unfinished=0, deadline_missed=3 if arm == 'B' else 0,
            dmr=1 if arm == 'B' else 0, mean_jct_minutes=40 if arm == 'B' else 15,
            carbon_kg={'A': 10, 'B': 9, 'C': 9.5}[arm],
            common_idle_carbon_kg=4,
            common_incremental_carbon_kg={'A': 6, 'B': 5, 'C': 5.5}[arm],
            policy_wait_minutes=30 if arm == 'B' else 0,
            capacity_wait_minutes=45 if arm == 'B' else 0,
            inlet_final_c={'A': 24, 'B': 24.02, 'C': 24.01}[arm],
            thermal_capacity_kj_k=3600, verification_pass=True,
            actual_unfinished_count=0, actual_common_idle_tail_valid=True,
            reference_guard_failure_count=0, failure_latched=False,
            decision_p50_ms=1, decision_p95_ms=2, decision_max_ms=3,
            attempts=1 if arm in 'BC' else 0,
            approved=1 if arm == 'B' else 0, rejected=1 if arm == 'C' else 0,
            rejection_reason_counts={'REFERENCE_ON_TIME_SET_LOST': 1} if arm == 'C' else {})
        summaries.append(summary)
        steps = [dict(minute=15*i, started=sum(start == 15*i for start in starts))
                 for i in range(288)]
        write_json(folder/'result.json', dict(summary=summary, tasks=tasks, steps=steps))
        decisions = [dict(minute=15*i,
            actual_started=[task['task_id'] for task in tasks if task['start_minute'] == 15*i],
            effective_policy_wait=['own'] if arm == 'B' and i == 0 else [],
            capacity_wait=['indirect'] if arm == 'B' and i == 0 else [],
            proposed=['own'] if arm == 'B' and i == 0 else [],
            approved=['own'] if arm == 'B' and i == 0 else [],
            final_lost_on_time_ids=['own', 'indirect'] if proof and arm == 'B' and i == 0 else [],
            failure_latched=False, carbon_trigger=i == 0) for i in range(288)]
        write_lines(folder/'decisions.jsonl', decisions)
        checks = []
        if arm in 'BC':
            checks = [dict(minute=0, task_id='own', approved=arm == 'B',
                trial_defer=['own'], lost_on_time_ids=['own', 'indirect'] if proof else [],
                reasons=[] if arm == 'B' else ['REFERENCE_ON_TIME_SET_LOST'],
                reference_late_total=0, trial_late_total=0, max_release=150,
                elapsed_ns=1, calendar_elapsed_ns=1, dispatch_slots=1, task_checks=3)]
        write_lines(folder/'candidates.jsonl', checks)
        if arm == 'C':
            probes = []
            for minute, count in ((0, 3), (60, 0)):
                for action, signature in enumerate(('noop', 'noop', 'x', 'x', 'y', 'y', 'y')):
                    probes.append(dict(minute=minute, action_id=action, candidate_count=count,
                        proposed_count=action if count else 0, approved_count=0,
                        effective_policy_wait_count=0, capacity_wait_count=0,
                        started_count=0, signature=signature if count else 'noop',
                        approved_signature='none', power_coverage=action/6 if count else 0,
                        cpu_coverage=action/6 if count else 0, gpu_coverage=0, mem_coverage=0,
                        carbon_trigger=minute == 0, failure_latched=False, elapsed_ms=1))
            write_lines(folder/'action-probes.jsonl', probes)
    write_json(group/'group.json', dict(fragment=FRAGMENT, capacity='OM',
        passed=group_pass, errors=[] if group_pass else ['SYNTHETIC FAILURE'], arms=summaries))
    write_json(output/'execution.json', dict(status='completed', synthetic=True))
    return group


class AnalysisTests(unittest.TestCase):
    def analysis_module(self):
        try:
            return importlib.import_module('experiments.sca_ppo.e1_analysis')
        except ModuleNotFoundError:
            self.fail('The generated-result analysis implementation is missing')

    def test_pairs_deltas_and_task_weighted_dmr(self):
        with tempfile.TemporaryDirectory(dir=ROOT, prefix='.e1-analysis-test-') as tmp:
            output = Path(tmp)/'local'; publish = Path(tmp)/'publish'; fixture(output)
            report = self.analysis_module().analyze(output, publish)
            cases = read_csv(publish/'case-summary.csv')
            self.assertEqual(len(cases), 3)
            pairs = read_csv(publish/'paired-results.csv')
            pair = next(row for row in pairs if row['comparison'] == 'B-A')
            self.assertEqual(float(pair['delta_carbon_kg']), -1)
            self.assertEqual(float(pair['delta_common_incremental_carbon_kg']), -1)
            self.assertAlmostEqual(float(pair['delta_thermal_storage_kwh']), .02)
            self.assertEqual(pair['valid_pair'], 'True')
            self.assertEqual(report['dmr_by_arm']['B']['task_weighted_dmr'], 1)
            self.assertEqual(report['actual_case_count'], 3)
            self.assertFalse(report['matrix_complete'])

    def test_task_proof_and_future_flags_do_not_conflate_own_wait_with_causality(self):
        with tempfile.TemporaryDirectory(dir=ROOT, prefix='.e1-analysis-test-') as tmp:
            output = Path(tmp)/'local'; publish = Path(tmp)/'publish'; fixture(output)
            report = self.analysis_module().analyze(output, publish)
            rows = {r['task_id']: r for r in read_csv(output/'task-diagnosis-all.csv')}
            self.assertEqual(rows['own']['B_own_policy_wait'], 'True')
            self.assertEqual(rows['indirect']['B_own_policy_wait'], 'False')
            self.assertEqual(rows['indirect']['B_known_reference_harm'], 'True')
            self.assertEqual(rows['future']['future_at_first_AB_divergence'], 'True')
            self.assertEqual(rows['future']['causal_link_uncertain'], 'True')
            self.assertEqual(rows['indirect']['C_actual_reject_related'], 'True')
            self.assertEqual(report['victims']['B_new_misses_vs_A'], 3)
            self.assertEqual(report['victims']['B_new_misses_with_local_reference_harm'], 2)
            self.assertEqual(report['victims']['C_rescues_B_new_misses'], 3)
            self.assertEqual(report['groups'][0]['earliest_AB_actual_divergence_minute'], 0)
            self.assertLessEqual(len(read_csv(publish/'task-diagnosis-cases.csv')), 30)

    def test_absence_of_local_proof_remains_uncertain_despite_policy_wait(self):
        with tempfile.TemporaryDirectory(dir=ROOT, prefix='.e1-analysis-test-') as tmp:
            output = Path(tmp)/'local'; publish = Path(tmp)/'publish'; fixture(output, proof=False)
            self.analysis_module().analyze(output, publish)
            own = next(r for r in read_csv(output/'task-diagnosis-all.csv') if r['task_id'] == 'own')
            self.assertEqual(own['B_known_reference_harm'], 'False')
            self.assertEqual(own['causal_link_uncertain'], 'True')

    def test_failed_group_remains_in_case_pairs_and_aggregate_counts(self):
        with tempfile.TemporaryDirectory(dir=ROOT, prefix='.e1-analysis-test-') as tmp:
            output = Path(tmp)/'local'; publish = Path(tmp)/'publish'; fixture(output, group_pass=False)
            report = self.analysis_module().analyze(output, publish)
            self.assertEqual(len(read_csv(publish/'case-summary.csv')), 3)
            self.assertTrue(all(row['valid_pair'] == 'False' for row in read_csv(publish/'paired-results.csv')))
            self.assertEqual(report['invalid_case_count'], 3)
            self.assertEqual(report['dmr_by_arm']['B']['tasks'], 3)

    def test_action_alias_denominators_include_empty_states_and_condition_on_opportunity(self):
        with tempfile.TemporaryDirectory(dir=ROOT, prefix='.e1-analysis-test-') as tmp:
            output = Path(tmp)/'local'; publish = Path(tmp)/'publish'; fixture(output)
            self.analysis_module().analyze(output, publish)
            rows = read_csv(publish/'action-alias-summary.csv')
            all_states = next(r for r in rows if r['scope'] == 'ALL' and r['condition'] == 'all_states')
            opportunity = next(r for r in rows if r['scope'] == 'ALL' and r['condition'] == 'candidate_opportunity')
            self.assertEqual(int(all_states['states']), 2)
            self.assertEqual(int(opportunity['states']), 1)
            self.assertAlmostEqual(float(all_states['mean_alias_ratio']), 5/6)
            self.assertAlmostEqual(float(opportunity['mean_alias_ratio']), 2/3)
            self.assertAlmostEqual(float(opportunity['mean_noop_equivalence_fraction']), 1/6)

    def test_preserves_completed_arms_and_episode_error_without_group_marker(self):
        with tempfile.TemporaryDirectory(dir=ROOT, prefix='.e1-analysis-test-') as tmp:
            output = Path(tmp)/'local'; publish = Path(tmp)/'publish'; group = fixture(output)
            (group/'group.json').unlink()
            (group/'C/result.json').unlink()
            write_json(group/'C/error.json', dict(error='SYNTHETIC stop', minute=90))
            report = self.analysis_module().analyze(output, publish)
            cases = read_csv(publish/'case-summary.csv')
            self.assertEqual(len(cases), 3)
            self.assertEqual(report['error_episode_count'], 1)
            self.assertEqual(report['invalid_case_count'], 3)
            self.assertEqual(report['groups'][0]['missing_arms'], ['C'])
            self.assertEqual(read_csv(publish/'failed-episodes.csv')[0]['error'], 'SYNTHETIC stop')

    def test_dmr_weights_unequal_task_counts_instead_of_averaging_case_rates(self):
        with tempfile.TemporaryDirectory(dir=ROOT, prefix='.e1-analysis-test-') as tmp:
            output = Path(tmp)/'local'; publish = Path(tmp)/'publish'; group = fixture(output)
            fragment = 'train-20230404T0500-r1-s1.5'
            second = output/(fragment+'__OM'); shutil.copytree(group, second)
            metadata = json.loads((second/'group.json').read_text()); metadata['fragment'] = fragment
            write_json(second/'group.json', metadata)
            for arm in 'ABC':
                result = json.loads((second/arm/'result.json').read_text())
                result['tasks'] = result['tasks'][:1]
                result['tasks'][0].update(start_minute=0, finish_minute=10, deadline_missed=False, jct_minutes=10)
                result['summary'].update(fragment=fragment, arrived=1, deadline_missed=0, dmr=0)
                write_json(second/arm/'result.json', result)
            report = self.analysis_module().analyze(output, publish)
            self.assertEqual(report['dmr_by_arm']['B']['tasks'], 4)
            self.assertEqual(report['dmr_by_arm']['B']['task_weighted_dmr'], .75)

    def test_publication_replaces_irreversible_task_id_list_with_count_and_preserves_source(self):
        with tempfile.TemporaryDirectory(dir=ROOT, prefix='.e1-analysis-test-') as tmp:
            output = Path(tmp)/'local'; publish = Path(tmp)/'publish'; group = fixture(output)
            source = group/'B/result.json'; result = json.loads(source.read_text())
            result['summary']['irreversible_miss_ids'] = ['own', 'indirect', 'future']
            write_json(source, result); before = source.read_bytes()
            self.analysis_module().analyze(output, publish)
            rows = read_csv(publish/'case-summary.csv')
            self.assertNotIn('irreversible_miss_ids', rows[0])
            self.assertEqual(next(r['irreversible_miss_count'] for r in rows if r['arm'] == 'B'), '3')
            self.assertEqual(source.read_bytes(), before)

    def test_ddf_rejection_diagnostics_partition_self_other_and_lateness_overlap(self):
        with tempfile.TemporaryDirectory(dir=ROOT, prefix='.e1-analysis-test-') as tmp:
            output = Path(tmp)/'local'; publish = Path(tmp)/'publish'; group = fixture(output)
            checks = []
            for identity, victims, reasons, approved in [
                ('self', ['self'], ['REFERENCE_ON_TIME_SET_LOST'], False),
                ('other', ['victim'], ['REFERENCE_ON_TIME_SET_LOST'], False),
                ('both', ['both', 'victim'], ['REFERENCE_ON_TIME_SET_LOST', 'ORIGINAL_LATE_TOTAL_INCREASED'], False),
                ('late', [], ['ORIGINAL_LATE_TOTAL_INCREASED'], False),
                ('cg', [], ['COMPLETION_GUARD_REJECTED'], False),
                ('pass', [], [], True)]:
                checks.append(dict(minute=0, task_id=identity, lost_on_time_ids=victims,
                                   reasons=reasons, approved=approved))
            write_lines(group/'C/candidates.jsonl', checks)
            report = self.analysis_module().analyze(output, publish)
            self.assertIn('ddf_rejection_diagnostics', report)
            ddf = report['ddf_rejection_diagnostics']
            self.assertEqual(ddf['checks'], 6)
            self.assertEqual(ddf['rejected'], 5)
            self.assertEqual(ddf['approved'], 1)
            self.assertEqual(ddf['self_on_time_loss'], 2)
            self.assertEqual(ddf['other_on_time_loss'], 2)
            self.assertEqual(ddf['both'], 1)
            self.assertEqual(ddf['known_other_only'], 1)
            self.assertEqual(ddf['original_late_total_increase'], 2)
            self.assertEqual(ddf['on_time_loss_and_original_late_total_increase'], 1)
            self.assertEqual(ddf['lateness_without_on_time_loss'], 1)
            self.assertEqual(sum(ddf['primary_reason_partition'].values()), 5)

    def test_candidate_prefix_harm_restored_by_final_joint_set_remains_uncertain(self):
        with tempfile.TemporaryDirectory(dir=ROOT, prefix='.e1-analysis-test-') as tmp:
            output = Path(tmp)/'local'; publish = Path(tmp)/'publish'; group = fixture(output)
            path = group/'B/decisions.jsonl'
            decisions = [json.loads(line) for line in path.read_text().splitlines()]
            decisions[0]['final_lost_on_time_ids'] = ['own']
            write_lines(path, decisions)
            report = self.analysis_module().analyze(output, publish)
            rows = {row['task_id']: row for row in read_csv(output/'task-diagnosis-all.csv')}
            indirect = rows['indirect']
            self.assertIn('B_final_joint_reference_harm', indirect)
            self.assertEqual(indirect['B_known_reference_harm'], 'True')
            self.assertEqual(indirect['B_final_joint_reference_harm'], 'False')
            self.assertEqual(indirect['B_prefix_only_reference_harm'], 'True')
            self.assertEqual(indirect['causal_link_uncertain'], 'True')
            self.assertEqual(rows['own']['B_final_joint_reference_harm'], 'True')
            self.assertNotEqual(indirect['evidence_level'], rows['own']['evidence_level'])
            self.assertEqual(report['victims']['B_new_misses_with_local_reference_harm'], 2)
            self.assertEqual(report['victims']['B_new_misses_with_final_joint_reference_harm'], 1)
            self.assertEqual(report['victims']['B_new_misses_with_prefix_only_reference_harm'], 1)
            self.assertEqual(report['victims']['B_new_misses_causal_link_uncertain'], 2)


if __name__ == '__main__':
    unittest.main()
