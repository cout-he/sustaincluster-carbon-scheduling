"""Read-only descriptive EDF evidence from the six sealed E1-5C cohorts."""
import csv
import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
NOTES = Path(__file__).resolve().parent
RUN = NOTES/'diagnostics-001'


def path(value):
    if os.name == 'nt' and value.startswith('/mnt/d/'):
        return Path('D:/'+value[len('/mnt/d/'):])
    if Path(value).is_absolute():
        return Path(value)
    return ROOT/value


def document(name):
    return json.loads(Path(name).read_text(encoding='utf-8'))


def lines(name):
    with Path(name).open(encoding='utf-8') as stream:
        yield from (json.loads(line) for line in stream if line.strip())


def digest(name):
    h = hashlib.sha256()
    with Path(name).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def csv_write(name, rows):
    with Path(name).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main(root=ROOT, diagnostics=RUN, output=NOTES):
    global ROOT, RUN
    ROOT, RUN = Path(root).resolve(), Path(diagnostics).resolve()
    output = Path(output)
    outputs = [output/n for n in ('pressure-timing-evidence.csv',
        'capacity-mask-evidence.csv','pressure-evidence-audit.json')]
    if any(p.exists() for p in outputs):
        raise FileExistsError('Evidence outputs already exist')
    output.mkdir(parents=True, exist_ok=True)
    manifest = document(RUN/'input-manifest.json')
    assert manifest['test_evaluation'] is False
    cohort_rows, peaks, analyses = [], [], {}
    inputs_hashes = {str((RUN/'input-manifest.json').relative_to(ROOT)):digest(RUN/'input-manifest.json')}
    for cohort in sorted(manifest['cohorts'], key=lambda c:(c['split'], c['factor'])):
        split, factor = cohort['split'], cohort['factor']
        assert split in ('train','validation')
        label = f'{split}-{factor:g}'
        folder, group = RUN/label, path(cohort['group_path'])/'edf_backfill'
        summary = document(folder/'pressure-summary.json')
        audit = document(folder/'edf-pressure-audit.json')
        with (folder/'edf-pressure-timeline.csv').open(encoding='utf-8',newline='') as stream:
            series = list(csv.DictReader(stream))
        ledger = {t['task_id']:t for t in lines(group/'ledger.jsonl')}
        masks = Counter(); ids = defaultdict(set); totals = Counter(); bytime = defaultdict(Counter)
        names, fields = ('cores','gpus','mem'), ('cores_req','gpu_req','mem_req')
        tiny_negative = 0
        for event in lines(group/'policy_events.jsonl'):
            totals[event['event']] += 1
            if event['event'] != 'capacity_waited':
                continue
            task = ledger[event['task_id']]
            deficit = [task[f]-event['available_before'][r] for f,r in zip(fields,names)]
            mask = '+'.join(r for r,d in zip(names,deficit) if d>0) or 'none'
            tolerance_mask = '+'.join(r for r,d in zip(names,deficit) if d>1e-9) or 'none'
            if mask != tolerance_mask:
                tiny_negative += 1
            masks[mask] += 1; ids[mask].add(task['task_id']); bytime[event['time']][mask] += 1
        assert totals['capacity_waited'] == audit['summary']['capacity_wait_intervals']
        assert totals['policy_deferred'] == 0
        assert len(ledger) == summary['tasks']
        assert sum(masks.values()) == totals['capacity_waited']
        row = dict(cohort=label, split=split, factor=factor, tasks=len(ledger),
            completed_misses=audit['summary']['completed_misses'],
            dmr=audit['summary']['completed_misses']/len(ledger),
            capacity_wait_events=totals['capacity_waited'],
            capacity_wait_unique_tasks=len(set().union(*ids.values())) if ids else 0,
            cores_only_events=masks['cores'], gpus_only_events=masks['gpus'],
            mem_only_events=masks['mem'], cores_gpus_events=masks['cores+gpus'],
            cores_mem_events=masks['cores+mem'], gpus_mem_events=masks['gpus+mem'],
            all_three_events=masks['cores+gpus+mem'], no_positive_deficit_events=masks['none'],
            mask_changes_with_1e_9_tolerance=tiny_negative,
            gpus_pressure_window=summary['gpus']['arrival_window_load'],
            cores_pressure_window=summary['cores']['arrival_window_load'],
            mem_pressure_window=summary['mem']['arrival_window_load'],
            urgent_arrival_fraction=summary['urgent_on_arrival_fraction'],
            mean_duration_minutes=summary['duration_minutes']['mean'],
            median_grid_slack_minutes=summary['usable_grid_slack_minutes']['p50'],
            gpus_99pct_intervals=audit['summary']['gpus_near_saturated_intervals'],
            cores_99pct_intervals=audit['summary']['cores_near_saturated_intervals'],
            mem_99pct_intervals=audit['summary']['mem_near_saturated_intervals'])
        cohort_rows.append(row)
        for field, scope in (('gpus_pressure_120m','arrival_window'),
                             ('pending','common_observation_window'),
                             ('new_latest_start_misses_known','common_observation_window')):
            candidates = [r for r in series if scope!='arrival_window'
                          or r['interval_start']<summary['arrival_end']]
            maximum = max(float(r[field]) for r in candidates)
            at_peak = min((r for r in candidates if float(r[field])==maximum),
                          key=lambda r:r['interval_start'])
            context = dict(cohort=label, split=split, factor=factor, peak_metric=field,
                peak_scope=scope, tie_break='earliest UTC maximum', time_utc=at_peak['interval_start'],
                peak_value=maximum, elapsed_hours=(
                    datetime.fromisoformat(at_peak['interval_start'])-
                    datetime.fromisoformat(summary['arrival_start'])).total_seconds()/3600)
            for name in ('arriving_tasks','gpus_work_minutes_occupied','gpus_pressure_15m',
                         'gpus_pressure_120m','cores_pressure_120m','mem_pressure_120m',
                         'pending','running','capacity_waiting_tasks',
                         'new_latest_start_misses_known','new_completed_misses_observed',
                         'cores_fraction','gpus_fraction','mem_fraction'):
                context[name] = float(at_peak[name])
            masks_at_peak = bytime[at_peak['interval_start']]
            context.update(cores_only_waits_at_peak=masks_at_peak['cores'],
                gpus_only_waits_at_peak=masks_at_peak['gpus'],
                multiple_resource_waits_at_peak=sum(v for k,v in masks_at_peak.items() if '+' in k))
            peaks.append(context)
        analyses[label] = dict(summary=summary, edf_summary=audit['summary'],
            capacity_masks=dict(masks), missed_tasks_with_any_capacity_wait=len(
                {t for group_ids in ids.values() for t in group_ids if ledger[t]['deadline_outcome']=='missed'}))
        for name in ('pressure-summary.json','edf-pressure-audit.json','edf-pressure-timeline.csv'):
            p=folder/name; inputs_hashes[str(p.relative_to(ROOT))]=digest(p)
        for name in ('ledger.jsonl','policy_events.jsonl'):
            p=group/name; inputs_hashes[str(p.relative_to(ROOT))]=digest(p)
    csv_write(outputs[0], peaks); csv_write(outputs[1],cohort_rows)
    result=dict(schema='e1-5c-pressure-evidence-v1',test_evaluation=False,training=False,
        rules=dict(mask='strict positive task demand - availability at candidate turn; mutually exclusive masks',
            rounding_sensitivity='also test absolute 1e-9 positive deficit; report differences',
            peak_selection='all six cohorts, first UTC maximum for each predeclared metric',
            scopes='GPU history peak restricted to arrival cohort; queue/known-miss peaks full common window',
            causal_claim=False), cohorts=analyses, input_files_sha256=inputs_hashes,
        outputs_sha256={p.name:digest(p) for p in outputs[:2]})
    outputs[2].write_text(json.dumps(result,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    print(json.dumps(cohort_rows,indent=2))
    print(json.dumps([r for r in peaks if r['factor']==2],indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--diagnostics', type=Path, default=RUN)
    parser.add_argument('--output', type=Path, default=NOTES)
    args=parser.parse_args()
    main(args.root, args.diagnostics, args.output)
