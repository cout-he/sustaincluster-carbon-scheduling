"""Read-only baseline/source/Train-metadata identity check; no payload loader."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

from . import BASE_COMMIT


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(pool, output):
    root = Path(__file__).resolve().parents[2]
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=root)
    manifest_path = Path(pool)/'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf8'))
    if manifest.get('test_access') is not False:
        raise ValueError('Train pool must preserve Test protection')
    entries = [e for e in manifest['scenarios'] if e.get('role') == 'train']
    if len(entries) != 96:
        raise ValueError('Expected the frozen 96-Train scenario pool')
    selected = ('experiments/e1/', 'experiments/joint/', 'experiments/joint_v2/',
                'utils/', 'envs/', 'configs/', 'rl_components/')
    protected = {}
    for name in git('ls-tree', '-r', '--name-only', BASE_COMMIT).decode().splitlines():
        if name.startswith(selected) and Path(name).suffix in ('.py', '.yaml', '.json'):
            original = git('show', BASE_COMMIT+':'+name)
            actual = (root/name).read_bytes()
            protected[name] = dict(sha256=hashlib.sha256(actual).hexdigest(),
                                   equals_base_bytes=actual == original)
    mismatches = [p for p,v in protected.items() if not v['equals_base_bytes']]
    if mismatches:
        raise ValueError('Existing production files changed: '+str(mismatches))
    cfg = json.loads((root/'experiments/e1/configs/e1_5b_ppo.json').read_text())
    result = dict(schema='A0-provenance-v1', checked_utc=datetime.now(timezone.utc).isoformat(),
        base_commit=BASE_COMMIT, audit_branch=git('branch','--show-current').decode().strip(),
        production_files_checked=len(protected), production_files=protected,
        production_changed=False, pool_manifest_sha256=sha(manifest_path),
        existing_raw_source_sha256=cfg['source_sha256'],
        raw_source_hashes_recomputed=False,
        raw_source_hash_note='Inherited frozen identities only; annual raw containers were not scanned.',
        selected_role='train', selected_scenarios=entries,
        accessed_payloads_in_this_check=[], train_payload_validation='Performed by existing role-checked load_pool in diagnostic modules',
        development_payload_access=False, july_payload_access=False, september_payload_access=False,
        trained_models=False, data_or_existing_config_changed=False)
    path = Path(output); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(production_files_checked=len(protected), mismatch_count=len(mismatches),
                         train_scenarios=len(entries), pool_manifest_sha256=result['pool_manifest_sha256'])))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pool', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    audit(args.pool, args.output)
