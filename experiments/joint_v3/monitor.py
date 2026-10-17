"""Lightweight evidence exported only from committed real episode records."""
import argparse
import csv
import json
from pathlib import Path
from experiments.joint_v2.checkpoint import atomic_json


def flush(output, state):
    root = Path(output); root.mkdir(parents=True, exist_ok=True)
    records = []
    for e in state['episodes']:
        s, u = e['summary'], e['updates']
        row = dict(episode=e['episode'], physical_steps=e['physical_steps'],
            scenario_id=e['scenario_id'], group=e['group'], elapsed_seconds=e['elapsed_seconds'],
            dmr=s['dmr'], arrived=s['arrived'], completed=s['completed'],
            carbon_kg=s['carbon_kg'], incremental_carbon_kg=s['incremental_carbon_kg'],
            multiplier=e['constraint']['value'], temperature_violations=s['temperature_violations'],
            carbon_critic_loss=u['carbon_critic_loss']['mean'], deadline_critic_loss=u['deadline_critic_loss']['mean'])
        for role in ('task', 'cooling'):
            for key in ('actor_loss', 'entropy', 'kl', 'clip_fraction', 'gradient_norm', 'teacher_loss'):
                row[role+'_'+key] = u[role][key]['mean']
            row[role+'_ppo_updates'] = u[role]['actor_optimizer_steps']
        records.append(row)
    path = root/'episodes.csv'
    if records:
        with path.open('w', encoding='utf8', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(records[0])); w.writeheader(); w.writerows(records)
    atomic_json(root/'progress.json', dict(physical_steps=state['physical_steps'],
        completed_episodes=state['completed_episodes'], elapsed_seconds=state['elapsed_seconds'],
        constraints=state['constraints'], best_feasible=state['best_feasible'],
        last_episode=records[-1] if records else None, test_access=False))
    atomic_json(root/'episode-evidence.json', dict(episodes=state['episodes'], validations=state['validations'],
        pretraining=state['pretraining'], forecaster_audit=state['forecaster'].get('audit'), test_access=False))


def plot(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import pandas as pd
    root = Path(output)
    path = root/'episodes.csv'
    if not path.exists(): return []
    frame = pd.read_csv(path)
    if frame.empty: return []
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'], 'font.size': 7, 'axes.spines.top': False,
        'axes.spines.right': False, 'pdf.fonttype': 42, 'svg.fonttype': 'none', 'axes.linewidth': .6})
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 4.5), layout='constrained')
    specs = [('incremental_carbon_kg', 'Train incremental carbon (kg)'), ('dmr', 'Train DMR'),
             ('task_kl', 'Task joint-action KL'), ('multiplier', 'Scenario constraint coefficient')]
    for ax, (key, label) in zip(axes.flat, specs):
        ax.plot(frame.physical_steps, frame[key], color='#26648e', lw=.8)
        ax.set(xlabel='Committed physical training slots', ylabel=label)
        if key == 'dmr': ax.axhline(.05, color='#b54a36', lw=.7, ls='--')
    fig.suptitle('Observed training progress; varying Train scenarios', fontsize=8)
    target = root/'figures'; target.mkdir(exist_ok=True)
    outputs = [str(target/('training-progress.'+s)) for s in ('pdf', 'svg', 'png')]
    fig.savefig(target/'training-progress.pdf')
    fig.savefig(target/'training-progress.svg')
    fig.savefig(target/'training-progress.png', dpi=600)
    plt.close(fig)
    return outputs


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--output', required=True)
    args = p.parse_args(argv); print(json.dumps(plot(args.output)))


if __name__ == '__main__': main()
