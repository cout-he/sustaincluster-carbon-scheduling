"""Independent file/canonical identity check; no simulator or raw input reads."""
import csv
import hashlib
import json
from pathlib import Path

root=Path(__file__).resolve().parent
def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
def load(folder):
    path=root/folder
    receipt=json.loads((path/'receipt.json').read_text())
    assert receipt['passed'] and receipt['split']=='train' and receipt['test_access'] is False
    assert json.loads((path/'run_status.json').read_text())['state']=='complete'
    for name,value in receipt['output_hashes'].items():
        assert digest(path/name)==value,(folder,name)
    with (path/'C-paired-outcomes.csv').open(newline='') as stream:
        rows={r['case_id']:r for r in csv.DictReader(stream)}
    return receipt,rows
original,first=load('C-campaign-001')
replay,second=load('C-replay-002')
assert original['selection_sha256']==replay['selection_sha256']
assert set(first)==set(second) and len(first)==12
rows=[]
for name in first:
    assert int(second[name]['historical_execute_control_files_verified'])==5
    for key in ('delta_misses','delta_carbon','execute_core_sha256','defer_core_sha256',
                'execute_full_sha256','defer_full_sha256','prefix_equal','common_end'):
        assert first[name][key]==second[name][key],(name,key)
    for branch in ('force_execute','force_defer_once'):
        for file in ('ledger.jsonl','events.jsonl','intervals.csv','summary.json',
                     'policy_ledger.jsonl','policy_events.jsonl','policy_intervals.jsonl','policy_summary.json'):
            assert digest(root/'C-campaign-001'/name/branch/file)==digest(root/'C-replay-002'/name/branch/file)
    for file in ('audit.json','online-features.json','prefixes.json','target.json','task-impacts.csv','interventions.json'):
        assert digest(root/'C-campaign-001'/name/file)==digest(root/'C-replay-002'/name/file),(name,file)
    rows.append(dict(case_id=name,scientific_branch_files_exact=16,paired_feature_files_exact=6,
                     canonical_core_and_full_exact=True,historical_control_files_verified=5))
receipt=dict(passed=True,split='train',test_access=False,case_count=12,
             byte_exact_scientific_branch_files=192,byte_exact_pair_evidence_files=72,
             historical_control_files_verified=60,all_24_core_and_full_hashes_equal=True,
             original_source_sha256=original['source_sha256'],replay_source_sha256=replay['source_sha256'],
             selection_sha256=original['selection_sha256'],rows=rows,
             runtime_excluded_from_determinism=True,script_sha256=digest(Path(__file__)))
with (root/'C-replay-verification-001.json').open('x',encoding='utf8') as stream:
    json.dump(receipt,stream,indent=2)
print(json.dumps({k:v for k,v in receipt.items() if k!='rows'}))
