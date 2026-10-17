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
from experiments.e1.contracts import content_hash, file_hash, utc_time
from experiments.joint_v2.checkpoint import atomic_json
from .checkpoint import code_identity
from .evaluate import load_model, load_v2, run_method, METHODS


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
    if not freeze.get('campaign_complete') or freeze.get('source_identity') != code_identity()['sha256']:
        raise ValueError('Method, source, training and Development experiments must be frozen first')
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
    out = Path(output)
    if out.exists(): raise FileExistsError('One-shot output already exists; attempt remains consumed')
    out.mkdir(parents=True)
    # Everything below this boundary is unreachable without explicit authorization.
    from experiments.e1.workload import load_workload
    from experiments.e1.selection import load_temporal_environment
    raw = json.loads((root/'experiments/e1/configs/e1_5b_ppo.json').read_text(encoding='utf8'))
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
    v2 = load_v2(freeze['v2_checkpoint'])
    results = []
    for selected in freeze['models']:
        pointer = Path(selected['training'])/'checkpoints'/(selected['alias']+'.json')
        if file_hash(pointer) != selected['pointer_sha256']: raise ValueError('Selected checkpoint changed after freeze')
        system, forecast, cfg, _ = load_model(selected['training'], selected['alias'])
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
    folder = Path(campaign)
    protocol = json.loads((folder/'protocol.json').read_text(encoding='utf8'))
    completed = []
    for job in protocol['jobs']:
        # Job layout is part of the V3 campaign protocol.
        name = job.get('name', job.get('id', f"{job['variant']}-{job['seed']}"))
        training = folder/'jobs'/name/'training'
        evaluations = folder/'jobs'/name/'evaluation'/'results.json'
        if not (training/'run-result.json').exists() or not evaluations.exists():
            raise ValueError('All predeclared training and Development comparisons must finish')
        if not json.loads(evaluations.read_text())['complete']:
            raise ValueError('Development comparison is incomplete')
        if job['variant'] == 'full':
            alias = 'best_feasible' if (training/'checkpoints/best_feasible.json').exists() else 'best_deadline'
            pointer = training/'checkpoints'/(alias+'.json')
            completed.append(dict(training=str(training.resolve()), alias=alias, pointer_sha256=file_hash(pointer)))
    if len(completed) != 3: raise ValueError('All three frozen seed models are required')
    atomic_json(output, dict(campaign_complete=True, source_identity=code_identity()['sha256'],
        protocol_sha256=file_hash(folder/'protocol.json'), models=completed,
        v2_checkpoint=str((Path(__file__).resolve().parents[2]/'research_notes/joint_v2/formal-001/training/checkpoints').resolve()),
        test_window_metadata=['2023-09-09T05:00:00Z', '2023-09-11T05:00:00Z', '2023-10-11T05:00:00Z'],
        test_payload_accessed=False))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--freeze', required=True); p.add_argument('--authorization')
    p.add_argument('--source-root'); p.add_argument('--output', required=True)
    p.add_argument('--freeze-campaign')
    a = p.parse_args(argv)
    if a.freeze_campaign: freeze_campaign(a.freeze_campaign, a.freeze)
    else: run(a.freeze, a.authorization, a.source_root, a.output)


if __name__ == '__main__': main()
