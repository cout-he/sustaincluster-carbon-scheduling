"""Source-field order/arrival audit of saved task queries; no dispatch or RL.

Only the same16 explicitly authorized Train snapshots may be loaded. Unlike
policy normalized floats, original fields preserve the exact frozen EDF key.
"""
import argparse
import json
from pathlib import Path

from experiments.sca_ppo.e2_data import load_authorized
from experiments.sca_ppo.edf_calendar import edf_key
from experiments.sca_ppo.e1_runner import write_json
from experiments.platform_v1.data import file_sha

ROOT=Path(__file__).resolve().parents[2]


def check_order(identities,tasks,minute,sla):
    unknown=[identity for identity in identities if identity not in tasks]
    if unknown:return dict(passed=False,errors=['Unknown task identity'],normalized_tie_pairs=[])
    jobs=[tasks[identity] for identity in identities]
    exact=[edf_key(task,sla) for task in jobs]
    errors=[]
    if any(task.arrival_minute>minute for task in jobs):errors.append('Task query before observed arrival')
    if exact!=sorted(exact):errors.append('Actual raw-field EDF inversion')
    if len(set(identities))!=len(identities):errors.append('Repeated candidate in physical slot')
    normalized=[(key[0]/4320,key[1]/4320,key[2]) for key in exact]
    ties=[]
    for i in range(len(exact)-1):
        if exact[i]<=exact[i+1] and normalized[i]>normalized[i+1]:
            ties.append(dict(task_ids=[exact[i][2],exact[i+1][2]],
                raw_deadlines=[exact[i][0],exact[i+1][0]],raw_arrivals=[exact[i][1],exact[i+1][1]],
                normalized_deadlines=[normalized[i][0],normalized[i+1][0]],
                explanation='Division merges distinct raw primary keys; normalized secondary sort loses original order.'))
    return dict(passed=not errors,errors=errors,normalized_tie_pairs=ties)


def generate(source,output):
    source=Path(source);plan_path=ROOT/'docs/research/task-ppo-v1/pilot-config.json'
    plan=json.loads(plan_path.read_text(encoding='utf-8'));cache={};errors=[];ties=[]
    cases=slots=decisions=0
    for path in sorted(source.rglob('result.json')):
        result=json.loads(path.read_text(encoding='utf-8'));summary=result['summary']
        if not summary.get('method','').startswith('TaskPPO-'):continue
        fragment=summary['fragment']
        allowed=plan['train_ids'] if 'train' in path.relative_to(source).parts else plan['holdout_ids']
        if fragment not in allowed:raise ValueError('Saved case has an unauthorized Train grouping')
        if fragment not in cache:
            scenario=load_authorized(Path(plan['pool'])/fragment,plan['inputs'][fragment])
            cache[fragment]={task.task_id:task for task in scenario.tasks}
        cases+=1
        for line in path.with_name('decisions.jsonl').open(encoding='utf-8'):
            row=json.loads(line);ids=[choice['task_id'] for choice in row['choices']]
            checked=check_order(ids,cache[fragment],row['minute'],plan['sla']);slots+=1;decisions+=len(ids)
            for error in checked['errors']:errors.append(dict(path=str(path.relative_to(source)),minute=row['minute'],error=error))
            for tie in checked['normalized_tie_pairs']:
                ties.append(dict(path=str(path.relative_to(source)),minute=row['minute'],**tie))
    audit=dict(schema='task-ppo-exact-source-edf-audit-v1',passed=not errors,errors=errors,
        source_sha=json.loads((source/'execution.json').read_text(encoding='utf-8'))['source_sha'],
        protocol_sha256=file_sha(plan_path),data_admission_sha256=plan['data_admission_sha256'],
        task_accounts=cases,physical_slots=slots,conditional_task_decisions=decisions,
        authorized_train_fragment_ids=sorted(cache),explained_normalization_tie_cases=ties,
        scope='Read-only original field/identity/arrival check. No simulator, rescheduling, network evaluation or training.')
    write_json(output,audit)
    print(json.dumps({key:audit[key] for key in ('passed','task_accounts','physical_slots','conditional_task_decisions')}))
    return audit


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',default=str(ROOT/'research_notes/task-ppo-v1/pilot-002'))
    parser.add_argument('--output',default=str(ROOT/'docs/research/task-ppo-v1/results/exact-order-audit.json'))
    args=parser.parse_args()
    if not generate(args.input,args.output)['passed']:raise SystemExit(2)
