"""Predeclared E1 Train run: smoke three episodes, then remaining 24, no RL.

All large traces stay in ignored research_notes. Frozen V2 is only imported.
Every actual slot uses E0's exact execution verifier. No automatic retries.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import itertools
import json
import math
from dataclasses import asdict, is_dataclass
from pathlib import Path
import subprocess
from time import perf_counter_ns, monotonic

from experiments.platform_v1.data import canonical_sha, file_sha, load_train
from experiments.platform_v2.environment import PlatformEnv, run as run_v2
from experiments.platform_v2.inventory import load_config, build_inventory
from experiments.platform_v2.verification import verify_result
from .adapter import ResearchAdapter
from .e1_policies import MatchedRuleAdapter
from .e0_evidence import percentile

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT/'docs/research/sca-ppo/E1-protocol.json'
ARMS = {'A':'EDF-CG', 'B':'CarbonRule-CG', 'C':'CarbonRule-DDF-CG'}


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    def normalize(obj):
        if is_dataclass(obj): return asdict(obj)
        if isinstance(obj,(set,frozenset)): return sorted(obj)
        if isinstance(obj,Path): return str(obj)
        raise TypeError('Unsupported evidence type: '+type(obj).__name__)
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2,
                               default=normalize)+'\n', encoding='utf-8')


def jsonl_line(stream, value):
    stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':'))+'\n')
    stream.flush()


def validate_protocol(plan):
    expected_fragments = ['train-20230501T0500-r1-s1.5','train-20230302T0500-r1-s1.5','train-20230404T0500-r1-s1.5']
    if (plan.get('schema') != 'SCA-PPO-E1-predeclared-v1' or plan.get('fragments') != expected_fragments
            or plan.get('capacities') != ['OM','OH','OL'] or plan.get('arms') != ARMS
            or plan.get('cases') != [list(x) for x in itertools.product(expected_fragments, ['OM','OH','OL'], ARMS)]
            or (plan.get('sla'), plan.get('arrival_minutes'), plan.get('horizon_minutes'),
                plan.get('clear_minutes'), plan.get('supply_requested_c')) != (1.5,1440,4320,3960,20)
            or plan.get('rule',{}).get('budget') != 16):
        raise ValueError('E1 protocol/matrix differs from user-approved frozen inputs')
    for relative, expected in {**plan['e0_source_sha256'], **plan['frozen_platform_source_sha256']}.items():
        if file_sha(ROOT/relative) != expected:
            raise ValueError('E0/frozen platform source changed: '+relative)


def probe_actions(adapter, obs):
    rows = []
    for action_id in range(7):
        probe = ResearchAdapter()
        probe.known = dict(adapter.known)
        probe.guard = copy.deepcopy(adapter.guard)
        start = perf_counter_ns(); d = probe.decide(obs, action_id=action_id)
        elapsed_ms = (perf_counter_ns()-start)/1e6
        tasks = {t.task_id:t for t in d.snapshot.pending}
        available = d.reference.first_slot.started
        proposed = d.proposed
        def power(t): return (0.005*t.demand[0]+0.225*t.demand[1])/0.94
        def coverage(score):
            total = math.fsum(score(tasks[i]) for i in available)
            return math.fsum(score(tasks[i]) for i in proposed)/total if total else 0.
        p = d.filter.preview
        rows.append(dict(minute=obs['minute'], action_id=action_id, candidate_count=len(available),
            proposed_count=len(proposed), approved_count=len(d.filter.approved),
            effective_policy_wait_count=len(p.policy_wait), capacity_wait_count=len(p.capacity_wait),
            started_count=len(p.started), signature=canonical_sha([p.started,p.policy_wait,p.capacity_wait]),
            approved_signature=canonical_sha(sorted(d.filter.approved)),
            power_coverage=coverage(power), cpu_coverage=coverage(lambda t:t.demand[0]),
            gpu_coverage=coverage(lambda t:t.demand[1]), mem_coverage=coverage(lambda t:t.demand[2]),
            carbon_trigger=bool(adapter.last_trigger and adapter.last_trigger['triggered']),
            failure_latched=adapter.guard.failure_latched, elapsed_ms=elapsed_ms))
    return rows


def run_episode(scenario, inv, cfg, arm, directory, identity, plan):
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=False)
    env = PlatformEnv(scenario, inv, cfg, 1.5, 'fixed')
    obs = env.reset(); adapter = MatchedRuleAdapter(arm, cfg)
    decisions = []; reasons = {}; attempts = approved = 0; probes = 0
    start = monotonic(); decision = None
    try:
        with (directory/'decisions.jsonl').open('w', encoding='utf-8') as df, \
             (directory/'candidates.jsonl').open('w', encoding='utf-8') as cf, \
             (directory/'action-probes.jsonl').open('w', encoding='utf-8') as pf:
            done = False
            while not done:
                decision = adapter.decide(obs)
                elapsed_ms = adapter.last_decision_ns/1e6
                if elapsed_ms > plan['stop_decision_seconds']*1000:
                    write_json(directory/'decision-budget-failure.json', dict(minute=obs['minute'],
                        elapsed_ms=elapsed_ms, known_tasks=len(decision.reference.records),
                        proposed=decision.proposed))
                    raise RuntimeError('E1 decision exceeded predeclared severe performance limit')
                if arm == 'C' and obs['minute'] in plan['action_probe_minutes']:
                    for row in probe_actions(adapter, obs): jsonl_line(pf, row); probes += 1
                if monotonic() > plan.get('_smoke_deadline', float('inf')):
                    raise RuntimeError('Minimum integration exceeded predeclared 900-second budget')
                minute = obs['minute']
                obs, done, _ = adapter.execute(env, decision)
                rec = dict(adapter.records[-1]); rec.pop('checks', None)
                rec.update(decision_ms=elapsed_ms, carbon_trigger=adapter.last_trigger['triggered'],
                    threshold=adapter.last_trigger['threshold'], history_count=adapter.last_trigger['history_count'],
                    reference_start_ids=decision.reference.first_slot.started,
                    reference_on_time_count=len(decision.reference.on_time_ids),
                    final_lost_on_time_ids=sorted(decision.reference.on_time_ids-decision.filter.plan.on_time_ids),
                    reference_late_total=decision.reference.total_lateness(decision.reference.late_ids),
                    final_late_total=decision.filter.plan.total_lateness(decision.reference.late_ids))
                jsonl_line(df, rec); decisions.append(elapsed_ms)
                for check in decision.filter.checks:
                    attempts += 1; approved += int(check.approved)
                    for reason in check.reasons: reasons[reason] = reasons.get(reason,0)+1
                    jsonl_line(cf, dict(minute=minute, **asdict(check)))
        result = env.result(); result['summary'].update(adapter.summary(env))
        source_tasks = {t.task_id:t for t in scenario.tasks}
        for row in result['tasks']:
            t = source_tasks[row['task_id']]
            row.update(arrival_minute=t.arrival_minute, duration_minutes=t.duration_minutes,
                       cpu=t.demand[0], gpu=t.demand[1], mem_gb=t.demand[2], source_row=t.source_row,
                       source_task_index=t.source_index)
        summary = result['summary']
        summary.update(**identity, arm=arm, method=ARMS[arm], case_id=f"{identity['fragment']}__{identity['capacity']}__{arm}",
            attempts=attempts, approved=approved, rejected=attempts-approved,
            rejection_reason_counts=reasons, decision_p50_ms=percentile(decisions,.5),
            decision_p95_ms=percentile(decisions,.95), decision_max_ms=max(decisions),
            elapsed_seconds=monotonic()-start, action_probe_count=probes,
            schedule_sha256=canonical_sha([(r['task_id'],r['start_minute'],r['finish_minute']) for r in result['tasks']]))
        # Write before verification, so incomplete/censored/failed traces survive.
        write_json(directory/'result.json', result)
        saved = json.loads((directory/'result.json').read_text(encoding='utf-8'))
        try: audit = verify_result(saved, scenario, inv, cfg, 1.5)
        except Exception as exc: audit = dict(errors=['Verifier raised on preserved trace: '+repr(exc)])
        summary['verification_pass'] = not audit['errors']
        write_json(directory/'verification.json', audit)
        write_json(directory/'result.json', result)
        return dict(summary)
    except Exception as exc:
        write_json(directory/'error.json', dict(error=repr(exc), minute=env.minute,
            inlet_c=env.inlet, past_steps=env.steps, task_ledger=env.ledger,
            pending_ids=[t.task_id for t in env.pending], running_ids=[t.task_id for t in env.running],
            last_decision=None if decision is None else dict(proposed=decision.proposed,
                approved=decision.filter.approved, checks=decision.filter.checks,
                reference_records=decision.reference.records, final_preview=decision.filter.preview),
            adapter_records=adapter.records))
        raise


def validate_group(results, idle, scenario, inv, cfg):
    errors = []
    if set(results) != set(ARMS): errors.append('Missing arms')
    for arm, row in results.items():
        if (row['arrived'] != len(scenario.tasks) or row['completed'] != len(scenario.tasks)
                or row['unfinished'] or row['deadline_censored']): errors.append(arm+': incomplete/censored task responsibility')
        if not row['verification_pass']: errors.append(arm+': saved V2 independent verification failed')
        if row['failure_latched'] or not row['actual_common_idle_tail_valid']:
            errors.append(arm+': CG failure/idle-tail invalid')
    values = list(results.values())
    for a,b in itertools.combinations(values,2):
        if abs(a['inlet_final_c']-b['inlet_final_c']) > .05: errors.append('Paired terminal temperature mismatch')
        if abs(a['it_kwh']-b['it_kwh']) > 1e-7: errors.append('Paired IT energy not conserved')
    if not idle['verification_pass']: errors.append('Common idle saved verification failed')
    return dict(passed=not errors, errors=errors)


def run_group(pool, fragment, capacity, output, source_sha, plan):
    started = monotonic(); group = Path(output)/(fragment+'__'+capacity)
    group.mkdir(parents=True, exist_ok=False)
    scenario = load_train(Path(pool)/'scenarios'/fragment, 72, plan['inputs'][fragment])
    cfg = load_config(); inv = build_inventory(cfg, capacity)
    idle_result = run_v2(scenario, inv, cfg, 'edf', 'fixed', 1.5, idle=True)
    write_json(group/'idle.json', idle_result)
    idle_audit = verify_result(json.loads((group/'idle.json').read_text(encoding='utf-8')),
                               scenario, inv, cfg, 1.5, idle=True)
    idle_result['summary']['verification_pass'] = not idle_audit['errors']
    write_json(group/'idle-verification.json', idle_audit); write_json(group/'idle.json', idle_result)
    summaries = {}
    episode_plan = dict(plan)
    if [fragment,capacity] == plan.get('smoke'):
        episode_plan['_smoke_deadline'] = started+plan['stop_smoke_seconds']
    for arm in ARMS:
        row = run_episode(scenario, inv, cfg, arm, group/arm,
            dict(fragment=fragment, capacity=capacity, source_sha=source_sha,
                 workload_sha256=scenario.provenance['workload_sha256']), episode_plan)
        row.update(common_idle_carbon_kg=idle_result['summary']['carbon_kg'],
                   common_incremental_carbon_kg=row['carbon_kg']-idle_result['summary']['carbon_kg'])
        # Attach the same explicit common background to every saved arm.
        path = group/arm/'result.json'; saved = json.loads(path.read_text(encoding='utf-8'))
        saved['summary'].update(row); write_json(path, saved)
        summaries[arm] = row
        print(f"E1 {fragment} {capacity} {arm} complete ({row['elapsed_seconds']:.2f}s, DMR={row['dmr']:.4f})", flush=True)
    checked = validate_group(summaries, idle_result['summary'], scenario, inv, cfg)
    result = dict(fragment=fragment, capacity=capacity, arms=list(summaries.values()),
                  elapsed_seconds=monotonic()-started, **checked)
    write_json(group/'group.json', result)
    return result


def execute(output):
    plan = json.loads(PROTOCOL.read_text(encoding='utf-8')); validate_protocol(plan)
    output = Path(output).resolve()
    if not output.is_relative_to((ROOT/'research_notes/sca-ppo').resolve()):
        raise ValueError('Large E1 logs must stay under ignored research_notes/sca-ppo')
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():
        raise ValueError('Commit protocol and executable source before real performance run')
    head = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    # Prove preregistration is in history, not merely an uncommitted file.
    protocol_commit = subprocess.check_output(['git','log','-1','--format=%H','--',str(PROTOCOL.relative_to(ROOT))],cwd=ROOT,text=True).strip()
    if not protocol_commit: raise ValueError('No protocol commit in history')
    output.parent.mkdir(parents=True,exist_ok=True)
    marker = output.parent/'E1-execution-once.json'
    metadata = dict(source_sha=head, protocol_commit=protocol_commit, protocol_sha256=file_sha(PROTOCOL),
        output=str(output), status='running', no_training=True,
        source_sha256={str(p.relative_to(ROOT)).replace('\\','/'):file_sha(p)
                       for p in sorted((ROOT/'experiments/sca_ppo').glob('*.py'))},
        groups=[], errors=[], matrix_max_workers=plan['matrix_max_workers'])
    with marker.open('x',encoding='utf-8') as stream: json.dump(metadata,stream,indent=2)
    output.mkdir(exist_ok=False); start = monotonic()
    try:
        smoke_fragment, smoke_capacity = plan['smoke']
        smoke = run_group(plan['pool'],smoke_fragment,smoke_capacity,output,head,plan)
        metadata['groups'].append(smoke)
        if not smoke['passed'] or smoke['elapsed_seconds'] > plan['stop_smoke_seconds']:
            raise RuntimeError('Minimum integration gate failed; remaining matrix not started')
        metadata['smoke_passed'] = True
        write_json(output/'execution.json',metadata)
        remaining = [x for x in itertools.product(plan['fragments'],plan['capacities'])
                     if list(x) != plan['smoke']]
        with ProcessPoolExecutor(max_workers=plan['matrix_max_workers']) as pool:
            pending = {pool.submit(run_group,plan['pool'],f,c,output,head,plan):(f,c) for f,c in remaining}
            for future in as_completed(pending):
                f,c = pending[future]
                try:
                    group = future.result(); metadata['groups'].append(group)
                    if not group['passed']: metadata['errors'].append(dict(fragment=f,capacity=c,errors=group['errors']))
                except Exception as exc:
                    metadata['errors'].append(dict(fragment=f,capacity=c,error=repr(exc)))
                    for other in pending: other.cancel()
                    raise
                write_json(output/'execution.json',metadata)
        metadata['status'] = 'completed' if len(metadata['groups']) == 9 and not metadata['errors'] else 'failed'
    except Exception as exc:
        metadata['status'] = 'failed'; metadata['errors'].append(dict(error=repr(exc)))
    metadata['elapsed_seconds'] = monotonic()-start
    metadata['workload_episode_count'] = sum(len(g['arms']) for g in metadata['groups'])
    metadata['idle_background_count'] = len(metadata['groups'])
    metadata['attempted_episode_directories'] = [str(p.relative_to(output)).replace('\\','/')
        for p in sorted(output.glob('*/*')) if p.is_dir() and p.name in ARMS]
    metadata['persisted_complete_episode_count'] = len(list(output.glob('*/*/result.json')))
    metadata['persisted_error_episode_count'] = len(list(output.glob('*/*/error.json')))
    metadata['verification_scope'] = 'All 27 workload + 9 common idle traces, frozen verify_result per saved trace; no V2 verify_saved matrix call'
    metadata['evidence_sha256'] = {str(p.relative_to(output)).replace('\\','/'):file_sha(p)
                                 for p in sorted(output.rglob('*')) if p.is_file() and p.name != 'execution.json'}
    write_json(output/'execution.json',metadata)
    write_json(marker,{k:v for k,v in metadata.items() if k not in ('groups','evidence_sha256')})
    print(json.dumps({k:metadata[k] for k in ('status','workload_episode_count','idle_background_count','elapsed_seconds','errors')},indent=2),flush=True)
    return metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default=str(ROOT/'research_notes/sca-ppo/e1-001'))
    args = parser.parse_args()
    if execute(args.output)['status'] != 'completed': raise SystemExit(2)
