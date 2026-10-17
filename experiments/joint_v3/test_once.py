"""One authorized September evaluation. Default invocation rejects before payload access.

This entry is intentionally separate from every Train/Development loader. No
authorization is bundled. A completed campaign freeze and explicit human file
are mandatory, followed by a durable study-wide one-attempt marker.
"""
import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace
from experiments.e1.contracts import content_hash, file_hash
from experiments.joint_v2.checkpoint import atomic_json
from .checkpoint import code_identity
from .config import load_config
from .evaluate import load_model, load_v2, run_method, METHODS
from . import VERSION


TEST_WINDOW = ['2023-09-09T05:00:00Z', '2023-09-11T05:00:00Z', '2023-10-11T05:00:00Z']


def _checkpoint_bytes(folder, alias, identity=None):
    """Bind an existing pointer and its generation without opening any data source."""
    folder = Path(folder).resolve()
    pointer_path = folder/(alias+'.json')
    pointer = json.loads(pointer_path.read_text(encoding='utf8'))
    if identity is not None and any(pointer.get(key) != value for key, value in identity.items()):
        raise ValueError('Selected checkpoint pointer identity differs')
    target = (folder/pointer['file']).resolve()
    if (target.parent != folder or target.suffix != '.pt'
            or file_hash(target) != pointer['sha256']):
        raise ValueError('Selected checkpoint byte identity differs')
    return dict(pointer_sha256=file_hash(pointer_path), checkpoint_sha256=pointer['sha256'])


def _campaign_snapshot(campaign):
    """Rebuild every prerequisite from canonical Train/Development artifacts."""
    from .pipeline import validate_protocol, _finished, _evaluation_complete
    folder = Path(campaign).resolve()
    protocol_path = folder/'protocol.json'
    protocol = validate_protocol(protocol_path)
    metadata = Path(__file__).resolve().parents[2]/'experiments/e1/configs/e1_5b_ppo.json'
    if json.loads(metadata.read_text(encoding='utf8'))['windows']['test'] != TEST_WINDOW:
        raise ValueError('Frozen September Test window metadata changed')
    models, artifacts = [], []
    for job in protocol['jobs']:
        training = (folder/job['training']).resolve()
        evaluation = (folder/'jobs'/job['job_id']/'evaluation').resolve()
        if not training.is_relative_to(folder) or not evaluation.is_relative_to(folder):
            raise ValueError('Campaign artifact path escapes its frozen directory')
        meta_path, result_path = training/'identity.json', training/'run-result.json'
        meta = json.loads(meta_path.read_text(encoding='utf8'))
        cfg = load_config(training/'config.json')
        if (meta.get('test_access') is not False or meta.get('config_identity') != job['config_identity']
                or meta.get('code_identity') != protocol['source_identity']
                or not isinstance(meta.get('data_identity'), str)
                or content_hash(cfg) != job['config_identity']):
            raise ValueError('Campaign training config/source/data identity differs')
        result = json.loads(result_path.read_text(encoding='utf8'))
        if (result.get('test_access') is not False or result.get('seed') != job['seed']
                or result.get('variant') != job['variant']
                or result.get('completed_episodes', 0) <= 0 or not _finished(result, cfg)):
            raise ValueError('Frozen training budget is incomplete')
        if not _evaluation_complete(evaluation, job, protocol):
            raise ValueError('Frozen Development evaluation is incomplete')
        alias = 'best_feasible' if (training/'checkpoints/best_feasible.json').exists() else 'best_deadline'
        identity = {key: meta[key] for key in ('config_identity', 'data_identity', 'code_identity')}
        bound = _checkpoint_bytes(training/'checkpoints', alias, dict(schema=VERSION, **identity))
        selected = dict(training=str(training), alias=alias, seed=job['seed'], variant=job['variant'],
                        **identity, **bound)
        artifacts.append(dict(job_id=job['job_id'], identity_sha256=file_hash(meta_path),
            config_sha256=file_hash(training/'config.json'), run_result_sha256=file_hash(result_path),
            evaluation_sha256=file_hash(evaluation/'results.json'), selected_checkpoint=selected))
        if job['variant'] == 'full':
            # Decode only the three policies which the eventual Test run uses.
            # Store.load validates the generation's internal identity as well.
            load_model(training, alias)
            models.append(selected)
    if len(models) != 3:
        raise ValueError('All three frozen seed models are required')
    v2 = Path(protocol['evaluation']['v2_checkpoint']).resolve()
    v2_pointer = json.loads((v2/'best_feasible.json').read_text(encoding='utf8'))
    if v2_pointer.get('completed_episodes') != 52:
        raise ValueError('B3 must use frozen V2 episode52 best')
    v2_bound = _checkpoint_bytes(v2, 'best_feasible')
    load_v2(v2)
    return dict(campaign_complete=True, campaign_root=str(folder), source_identity=protocol['source_identity'],
        protocol_sha256=file_hash(protocol_path), models=models, campaign_artifacts=artifacts,
        v2_checkpoint=str(v2), v2_pointer_sha256=v2_bound['pointer_sha256'],
        v2_checkpoint_sha256=v2_bound['checkpoint_sha256'],
        test_window_metadata=list(TEST_WINDOW), test_payload_accessed=False)


def authorize(freeze_path, authorization_path, marker_path):
    # Missing authorization must not even open the freeze, source or model payload.
    if authorization_path is None:
        raise ValueError('September Test is sealed; explicit user authorization is required')
    authorization = json.loads(Path(authorization_path).read_text(encoding='utf8'))
    if authorization.get('authorize_september_test') is not True:
        raise ValueError('September Test is sealed; authorization does not permit access')
    if authorization.get('freeze_sha256') != file_hash(freeze_path):
        raise ValueError('User authorization must identify the exact completed experiment freeze')
    freeze = json.loads(Path(freeze_path).read_text(encoding='utf8'))
    if (not freeze.get('campaign_complete') or freeze.get('source_identity') != code_identity()['sha256']
            or freeze.get('test_window_metadata') != TEST_WINDOW):
        raise ValueError('Method, source, training and Development experiments must be frozen first')
    try:
        current = _campaign_snapshot(freeze['campaign_root'])
    except (OSError, KeyError, TypeError) as exc:
        raise ValueError('Completed campaign identity cannot be revalidated') from exc
    if freeze != current:
        raise ValueError('Completed campaign freeze identity changed')
    marker = Path(marker_path)
    marker.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w', encoding='utf8') as stream:
        json.dump(dict(status='attempted', freeze_sha256=file_hash(freeze_path)), stream)
        stream.flush(); os.fsync(stream.fileno())
    return freeze


def run(freeze_path, authorization_path, source_root, output, *, marker_path=None):
    root = Path(__file__).resolve().parents[2]
    marker = Path(marker_path) if marker_path else root/'research_notes/joint_v3/september-test-attempt.json'
    freeze = authorize(freeze_path, authorization_path, marker)
    # All model/config/source identities have already passed before authorize
    # publishes the durable attempt marker. Load execution objects before data.
    v2 = load_v2(freeze['v2_checkpoint'])
    selected_models = [(selected, load_model(selected['training'], selected['alias']))
                       for selected in freeze['models']]
    out = Path(output)
    if out.exists(): raise FileExistsError('One-shot output already exists; attempt remains consumed')
    out.mkdir(parents=True)
    # Everything below this boundary is unreachable without explicit authorization.
    from experiments.e1.workload import load_workload
    from experiments.e1.selection import load_temporal_environment
    raw = json.loads((root/'experiments/e1/configs/e1_5b_ppo.json').read_text(encoding='utf8'))
    if raw['windows']['test'] != TEST_WINDOW:
        raise ValueError('Frozen September Test window metadata changed')
    paths = {k: Path(source_root)/p for k, p in raw['paths'].items()}
    for name, expected in raw['source_sha256'].items():
        if file_hash(paths[name]) != expected: raise ValueError('Test raw-source identity differs: '+name)
    start, end, cap = raw['windows']['test']
    workload = load_workload(paths['workload'], start, end, 5., 1.5, 1, 2020)
    environment = load_temporal_environment(paths['carbon'], paths['weather'], start, cap)
    dc = raw['datacenter']
    # Historical V2 loader stays permanently sealed. This isolated authorized
    # adapter reuses its execution mechanics under an internal permitted role;
    # provenance and every exported result retain the true Test role.
    scenario = SimpleNamespace(role='validation', scenario_id='authorized-september-20230909-base',
        workload=workload, environment=environment, capacity=(50000, 1000, 80000), datacenter=dc,
        model_path=paths['physical_model'], parameters=dict(threshold_g_per_kwh=161.075),
        identity=dict(role='test', arrival_start=start, arrival_end=end, operational_end=cap,
                      raw_source_sha256=raw['source_sha256']), statistics=dict(synthetic=False))
    results = []
    for selected, (system, forecast, cfg, _) in selected_models:
        for method in METHODS:
            summary = run_method(method, scenario, cfg, system, forecast, v2)
            summary.update(scenario_role='test', authorized_one_shot=True)
            results.append(summary)
            atomic_json(out/'results.json', dict(results=results, complete=False, true_role='test'))
    atomic_json(out/'results.json', dict(results=results, complete=True, true_role='test',
        freeze_sha256=file_hash(freeze_path), test_identity=scenario.identity))
    atomic_json(marker, dict(status='completed', freeze_sha256=file_hash(freeze_path), output=str(out.resolve())))
    return results


def freeze_campaign(campaign, output):
    """Only completed predeclared full/ablation comparisons can unlock a future request."""
    try:
        frozen = _campaign_snapshot(campaign)
    except (OSError, KeyError, TypeError) as exc:
        raise ValueError('All frozen training and Development artifacts must be complete') from exc
    atomic_json(output, frozen)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--freeze', required=True); p.add_argument('--authorization')
    p.add_argument('--source-root'); p.add_argument('--output', required=True)
    p.add_argument('--freeze-campaign')
    a = p.parse_args(argv)
    if a.freeze_campaign: freeze_campaign(a.freeze_campaign, a.freeze)
    else: run(a.freeze, a.authorization, a.source_root, a.output)


if __name__ == '__main__': main()
