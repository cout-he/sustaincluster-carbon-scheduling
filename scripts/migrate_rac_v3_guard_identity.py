"""One audited source refresh for evaluation/Test guards; never change training.

The coordinator must be stopped. This tool preserves immutable original
generations, all scientific state, configuration, data and consumed budgets.
Only evaluate.py/test_once.py may differ. It is not a general resume bypass.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from experiments.e1.contracts import content_hash, file_hash
from experiments.joint_v3.checkpoint import Store, atomic_json, code_identity

ALLOWED = {'experiments/joint_v3/evaluate.py', 'experiments/joint_v3/test_once.py'}
VOLATILE = {'elapsed_seconds', 'best_feasible_pointer', 'best_deadline_pointer', 'source_migrations'}


def validate_previous(protocol, status):
    unsigned = dict(protocol)
    digest = unsigned.pop('protocol_sha256', None)
    if (digest != content_hash(unsigned)
            or content_hash(protocol['source_files']) != protocol['source_identity']
            or status.get('protocol_sha256') != digest):
        raise ValueError('Original protocol/source/status identity differs')


def validate_diff(old, new):
    if set(old) != set(new):
        raise ValueError('Training source inventory differs; migration forbidden')
    changed = sorted(k for k in old if old[k] != new[k])
    if not set(changed) <= ALLOWED:
        raise ValueError('A training/configuration source changed; migration forbidden')
    return changed


def protected_digest(state):
    h = hashlib.sha256()
    def update(value):
        if isinstance(value, torch.Tensor):
            a = value.detach().cpu().contiguous()
            h.update(b'tensor'); h.update(str(a.dtype).encode()); h.update(str(tuple(a.shape)).encode())
            h.update(a.numpy().tobytes())
        elif isinstance(value, np.ndarray):
            h.update(b'array'); h.update(value.dtype.str.encode()); h.update(str(value.shape).encode())
            h.update(value.tobytes())
        elif isinstance(value, dict):
            h.update(b'dict')
            for k in sorted(value, key=lambda k: (type(k).__name__, repr(k))):
                update(k); update(value[k])
        elif isinstance(value, (list, tuple)):
            h.update(type(value).__name__.encode()); h.update(str(len(value)).encode())
            for item in value: update(item)
        elif value is None or isinstance(value, (str, bytes, bool, int, float)):
            h.update(type(value).__name__.encode()); h.update(repr(value).encode()); h.update(b'\0')
        else:
            raise TypeError('Unexpected checkpoint state type: '+type(value).__name__)
    update({k:v for k,v in state.items() if k not in VOLATILE})
    return h.hexdigest()


def migrate(campaign):
    root = Path(campaign).resolve()
    protocol_path, status_path = root/'protocol.json', root/'pipeline-status.json'
    old_protocol = json.loads(protocol_path.read_text(encoding='utf8'))
    status = json.loads(status_path.read_text(encoding='utf8'))
    validate_previous(old_protocol, status)
    if status['stage'] not in ('interrupted', 'failed'):
        raise ValueError('Stop the coordinator before an explicit source refresh')
    if any(v.get('stage') in ('training', 'evaluating') or v.get('pid')
           for v in status['jobs'].values()):
        raise ValueError('Training workers must be stopped')
    current = code_identity()
    changed = validate_diff(old_protocol['source_files'], current['files'])
    if not changed: raise ValueError('No approved security guard change is present')
    journal = root/'guard-migration-001'
    journal.mkdir(exist_ok=True)
    certificate_path = journal/'certificate.json'
    if certificate_path.exists(): raise FileExistsError('Source refresh was already applied')
    started_path = journal/'migration-started.json'
    if started_path.exists(): raise FileExistsError('A prior migration attempt needs explicit audit')
    certificate = dict(kind='evaluation_and_test_identity_guards_only',
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        old_source_identity=old_protocol['source_identity'], new_source_identity=current['sha256'],
        changed_files={k:dict(before=old_protocol['source_files'][k], after=current['files'][k]) for k in changed},
        migration_tool_sha256=file_hash(__file__), algorithm_or_config_changed=False,
        test_payload_accessed=False, jobs=[])
    # All checks and reads precede the first pointer/config mutation.
    prepared = []
    for job in old_protocol['jobs']:
        training = root/job['training']
        if not training.exists(): continue
        meta = json.loads((training/'identity.json').read_text(encoding='utf8'))
        if meta['code_identity'] != old_protocol['source_identity']:
            raise ValueError('Training source identity differs from old frozen protocol')
        config = json.loads((training/'config.json').read_text(encoding='utf8'))
        if content_hash(config) != meta['config_identity'] or meta['config_identity'] != job['config_identity']:
            raise ValueError('Training configuration changed')
        old_store = Store(training/'checkpoints', meta['config_identity'], meta['data_identity'], meta['code_identity'])
        pointers, states = {}, {}
        for alias in ('latest', 'best_feasible', 'best_deadline', 'final'):
            p = training/'checkpoints'/(alias+'.json')
            if p.exists():
                pointers[alias] = json.loads(p.read_text(encoding='utf8'))
                states[alias] = old_store.load(alias)
                if states[alias]['config'] != config:
                    raise ValueError('Saved scientific configuration changed')
        if 'final' in states and protected_digest(states['final']) != protected_digest(states['latest']):
            raise ValueError('Final and latest differ; this narrow migration cannot rebase them')
        consumed = max(states['latest']['elapsed_seconds'],
                       status['jobs'][job['job_id']].get('training_wall_seconds', 0.))
        record = dict(job=job['job_id'], charged_elapsed_seconds=consumed,
            physical_steps=states['latest']['physical_steps'], episodes=states['latest']['completed_episodes'],
            original_pointers=pointers, protected_sha256={k:protected_digest(v) for k,v in states.items()})
        prepared.append((job, training, meta, pointers, states, record))
        certificate['jobs'].append(record)
    atomic_json(journal/'protocol-before.json', old_protocol)
    atomic_json(journal/'status-before.json', status)
    atomic_json(started_path, dict(status='started', old_source_identity=old_protocol['source_identity'],
                                  new_source_identity=current['sha256']))
    try:
        for job, training, meta, pointers, states, record in prepared:
            store = Store(training/'checkpoints', meta['config_identity'], meta['data_identity'], current['sha256'])
            migrated = {}
            for alias in ('best_feasible', 'best_deadline'):
                if alias not in states: continue
                state = deepcopy(states[alias])
                state.setdefault('source_migrations', []).append(dict(
                    old=old_protocol['source_identity'], new=current['sha256'], kind=certificate['kind'],
                    protected_state_sha256=record['protected_sha256'][alias]))
                if protected_digest(state) != record['protected_sha256'][alias]:
                    raise ValueError('Scientific state changed during guard refresh')
                migrated[alias] = store.save(state, (alias,))
            latest = deepcopy(states['latest'])
            latest['elapsed_seconds'] = record['charged_elapsed_seconds']
            latest.setdefault('source_migrations', []).append(dict(
                old=old_protocol['source_identity'], new=current['sha256'], kind=certificate['kind'],
                protected_state_sha256=record['protected_sha256']['latest'],
                charged_elapsed_seconds=record['charged_elapsed_seconds']))
            for alias, pointer in migrated.items(): latest[alias+'_pointer'] = pointer
            if protected_digest(latest) != record['protected_sha256']['latest']:
                raise ValueError('Scientific state changed during guard refresh')
            migrated['latest'] = store.save(latest, ('latest', 'final') if 'final' in states else ('latest',))
            loaded = store.load()
            if protected_digest(loaded) != record['protected_sha256']['latest']:
                raise ValueError('New serialized scientific state differs')
            record['new_pointers'] = migrated
            new_meta = deepcopy(meta)
            new_meta.update(code_identity=current['sha256'], code_manifest=current['files'])
            new_meta.setdefault('source_migrations', []).append(dict(old=old_protocol['source_identity'],
                new=current['sha256'], certificate='guard-migration-001/certificate.json'))
            atomic_json(training/'identity.json', new_meta)
        protocol = deepcopy(old_protocol)
        protocol.update(source_identity=current['sha256'], source_files=current['files'])
        protocol.setdefault('source_migrations', []).append(dict(kind=certificate['kind'],
            old_source=old_protocol['source_identity'], new_source=current['sha256'],
            previous_protocol_sha256=old_protocol['protocol_sha256'],
            certificate='guard-migration-001/certificate.json'))
        protocol.pop('protocol_sha256')
        protocol['protocol_sha256'] = content_hash(protocol)
        status['protocol_sha256'] = protocol['protocol_sha256']
        status['source_migration'] = certificate['kind']
        atomic_json(protocol_path, protocol); atomic_json(status_path, status)
        from experiments.joint_v3.pipeline import validate_protocol
        validate_protocol(protocol_path)
        atomic_json(certificate_path, certificate)
        atomic_json(started_path, dict(status='completed', certificate=certificate_path.name))
        return certificate
    except BaseException:
        # Immutable original weight generations were never overwritten/deleted.
        for job, training, meta, pointers, states, record in prepared:
            atomic_json(training/'identity.json', meta)
            for alias, pointer in pointers.items(): atomic_json(training/'checkpoints'/(alias+'.json'), pointer)
        atomic_json(protocol_path, old_protocol)
        atomic_json(status_path, json.loads((journal/'status-before.json').read_text(encoding='utf8')))
        atomic_json(started_path, dict(status='rolled_back_requires_audit'))
        raise


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--campaign', required=True)
    a = p.parse_args(argv); result = migrate(a.campaign)
    print(json.dumps(dict(changed_files=result['changed_files'], jobs=len(result['jobs']),
                          new_source_identity=result['new_source_identity'])))


if __name__ == '__main__': main()
