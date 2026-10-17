"""A security-only source refresh must preserve the scientific training state."""
from copy import deepcopy
import pytest
import torch


def test_protected_digest_detects_weight_rng_and_cost_changes_but_not_charged_wall():
    from scripts.migrate_rac_v3_guard_identity import protected_digest
    original = dict(system={'weights': torch.tensor([1., 2.])}, global_rng={'torch': torch.tensor([5, 6])},
        physical_steps=123, constraints={'1|1.5': 10.}, elapsed_seconds=10.,
        best_feasible_pointer={'sha256': 'old'})
    changed = deepcopy(original); changed['elapsed_seconds'] = 20.
    changed['best_feasible_pointer'] = {'sha256': 'new'}
    assert protected_digest(changed) == protected_digest(original)
    for field in ('system', 'global_rng', 'physical_steps', 'constraints'):
        changed = deepcopy(original); changed[field] = None
        assert protected_digest(changed) != protected_digest(original)


def test_migration_rejects_any_training_or_config_source_change():
    from scripts.migrate_rac_v3_guard_identity import validate_diff
    old = {'experiments/joint_v3/evaluate.py': 'a', 'experiments/joint_v3/test_once.py': 'b',
           'experiments/joint_v3/mappo.py': 'c'}
    new = dict(old); new['experiments/joint_v3/evaluate.py'] = 'securityfix'
    assert validate_diff(old, new) == ['experiments/joint_v3/evaluate.py']
    new['experiments/joint_v3/mappo.py'] = 'changed'
    with pytest.raises(ValueError, match='training'):
        validate_diff(old, new)


def test_migration_requires_original_protocol_and_status_self_identity():
    from scripts.migrate_rac_v3_guard_identity import validate_previous
    from experiments.e1.contracts import content_hash
    protocol = dict(source_files={'experiments/joint_v3/evaluate.py': 'a'})
    protocol['source_identity'] = content_hash(protocol['source_files'])
    protocol['protocol_sha256'] = content_hash(protocol)
    status = dict(protocol_sha256=protocol['protocol_sha256'])
    validate_previous(protocol, status)
    for field in ('source_identity', 'protocol_sha256'):
        changed = deepcopy(protocol); changed[field] = 'tampered'
        with pytest.raises(ValueError, match='identity'):
            validate_previous(changed, status)
    with pytest.raises(ValueError, match='identity'):
        validate_previous(protocol, dict(protocol_sha256='different'))
