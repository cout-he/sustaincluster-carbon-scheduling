"""End-to-end artifact/replay checks, including the actual physical adapter."""
import json
import shutil
import subprocess
import sys

import pytest

from .test_inputs import END, START, environment_files, trace_file


def configuration(tmp_path):
    from experiments.e1.config import ROOT, load_config
    cfg = load_config(ROOT / 'experiments/e1/configs/single_dc.yaml')
    trace = trace_file(tmp_path)
    ci, weather = environment_files(tmp_path)
    cfg['start'], cfg['end'] = START.isoformat(), END.isoformat()
    cfg['workload']['path'] = str(trace)
    cfg['environment']['carbon_path'] = str(ci)
    cfg['environment']['weather_path'] = str(weather)
    return cfg, (trace, ci, weather)


SCIENTIFIC_FILES = ('resolved_config.json', 'workload.json', 'environment.json', 'dc_config.json',
                    'ledger.jsonl', 'events.jsonl', 'intervals.csv', 'summary.json')


def test_real_run_complete_artifacts_reproducible_and_replay_without_raw_inputs(tmp_path):
    from experiments.e1.run import run_experiment, replay_experiment
    cfg, inputs = configuration(tmp_path)
    first, second, replay = (tmp_path / name for name in ('first', 'second', 'replay'))
    run_experiment(cfg, first)
    run_experiment(cfg, second)
    assert json.loads((first / 'run_status.json').read_text())['state'] == 'complete'
    provenance = json.loads((first / 'provenance.json').read_text())
    assert provenance['git']['head'] and provenance['sources'] and provenance['dependencies']['numpy']
    assert provenance['input_hashes']['workload_snapshot']
    for file in SCIENTIFIC_FILES:
        assert (first / file).read_bytes() == (second / file).read_bytes(), file
        assert file in provenance['output_hashes']
    ledger = [json.loads(line) for line in (first / 'ledger.jsonl').read_text().splitlines()]
    assert len(ledger) == 1 and ledger[0]['state'] == 'completed'
    for file in inputs:
        file.unlink()
    replay_experiment(first, replay)
    for file in SCIENTIFIC_FILES:
        assert (first / file).read_bytes() == (replay / file).read_bytes(), file


def test_existing_output_is_never_overwritten_even_if_empty(tmp_path):
    from experiments.e1.run import run_experiment
    cfg, _ = configuration(tmp_path)
    out = tmp_path / 'existing'; out.mkdir()
    marker = out / 'important.txt'; marker.write_text('keep')
    with pytest.raises(FileExistsError): run_experiment(cfg, out)
    assert marker.read_text() == 'keep'


@pytest.mark.parametrize('folder', ['artifacts/run', 'custom-run'])
def test_source_inventory_excludes_generated_artifact_copies(tmp_path, monkeypatch, folder):
    from experiments.e1 import run
    source = tmp_path / 'experiments/e1'; source.mkdir(parents=True)
    (source / 'engine.py').write_text('source')
    (tmp_path / 'upstream.py').write_text('upstream')
    monkeypatch.setattr(run, 'ROOT', tmp_path)
    monkeypatch.setattr(run, '_git', lambda *args: 'upstream.py')
    before = run.source_hashes()
    output = source / folder; output.mkdir(parents=True)
    (output / 'run_status.json').write_text('{"state":"running"}')
    archived = output / 'source/experiments/e1'
    archived.mkdir(parents=True)
    (archived / 'engine.py').write_text('generated copy')
    assert run.source_hashes() == before
    assert set(before) == {'experiments/e1/engine.py', 'upstream.py'}


@pytest.mark.parametrize('folder', ['artifacts/run', 'custom-run'])
def test_pytest_does_not_collect_archived_source_tests(tmp_path, folder):
    from experiments.e1.config import ROOT
    project = tmp_path / 'project'
    experimental = project / 'experiments/e1'; experimental.mkdir(parents=True)
    hook = ROOT / 'experiments/e1/conftest.py'
    if hook.exists(): shutil.copyfile(hook, experimental / 'conftest.py')
    output = experimental / folder; output.mkdir(parents=True)
    (output / 'run_status.json').write_text('{"state":"complete"}')
    archived = output / 'source/tests'; archived.mkdir(parents=True)
    (archived / 'test_generated.py').write_text('raise RuntimeError("archive must not be collected")')
    stray = experimental / 'artifacts'; stray.mkdir(exist_ok=True)
    (stray / 'test_unmarked_copy.py').write_text('raise RuntimeError("collect_ignore must still apply")')
    tests = project / 'tests'; tests.mkdir()
    (tests / 'test_actual.py').write_text('def test_actual(): pass')
    result = subprocess.run([sys.executable, '-m', 'pytest', '--collect-only', '-q'],
                            cwd=project, text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert '1 test collected' in result.stdout


@pytest.mark.parametrize('target', ['workload.json', 'dc_config.json', 'resolved_config.json', 'source'])
def test_replay_fails_on_changed_inputs_model_config_or_source(tmp_path, target):
    from experiments.e1.run import run_experiment, replay_experiment
    cfg, _ = configuration(tmp_path)
    out = tmp_path / 'first'; run_experiment(cfg, out)
    if target == 'source':
        path = out / 'provenance.json'
        value = json.loads(path.read_text())
        value['sources']['experiments/e1/engine.py'] = 'bad-hash'
        path.write_text(json.dumps(value))
    else:
        path = out / target
        value = json.loads(path.read_text())
        if target == 'workload.json': value['payload']['tasks'][0]['duration'] = 99
        elif target == 'resolved_config.json': value['workload']['sla_multiplier'] = 99
        else: value['tamper'] = True
        path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='hash|source'):
        replay_experiment(out, tmp_path / 'failed')
    assert json.loads((tmp_path / 'failed/run_status.json').read_text())['state'] == 'failed'
    assert not (tmp_path / 'failed/summary.json').exists()


def test_model_exception_leaves_failed_status_without_success_summary(tmp_path, monkeypatch):
    from experiments.e1.run import run_experiment
    from experiments.e1.power import UpstreamPowerModel
    cfg, _ = configuration(tmp_path)
    def fail(*args): raise RuntimeError('physical failure')
    monkeypatch.setattr(UpstreamPowerModel, 'measure', fail)
    with pytest.raises(RuntimeError, match='physical failure'):
        run_experiment(cfg, tmp_path / 'failed')
    status = json.loads((tmp_path / 'failed/run_status.json').read_text())
    assert status['state'] == 'failed' and status['error_type'] == 'RuntimeError'
    assert not (tmp_path / 'failed/summary.json').exists()


@pytest.mark.parametrize('contents', [None, 'policy: edf', 'start: [unclosed'])
def test_cli_config_failure_still_leaves_failed_run_status(tmp_path, monkeypatch, contents):
    from experiments.e1.run import main
    out = tmp_path / 'failed'
    config_path = tmp_path / 'config.yaml'
    if contents is not None: config_path.write_text(contents)
    monkeypatch.setattr('sys.argv', ['e1', '--config', str(config_path), '--output', str(out)])
    import yaml
    with pytest.raises((FileNotFoundError, ValueError, yaml.YAMLError)): main()
    assert json.loads((out / 'run_status.json').read_text())['state'] == 'failed'
