"""Copy explicitly selected small scientific summaries; never raw artifacts."""
import csv
import hashlib
import json
from pathlib import Path

root=Path(__file__).resolve().parent
output=root/'tables'
output.mkdir(exist_ok=False)
files={
    'baseline-comparison.csv':'baseline-replay-001/baseline-comparison.csv',
    'historical-regression.csv':'baseline-replay-001/historical-regression.csv',
    'A-cases.csv':'A-campaign-001/A-cases.csv',
    'A-edf-vs-oracle.csv':'A-campaign-001/A-edf-vs-oracle.csv',
    'A-solver-status.csv':'A-campaign-001/A-solver-status.csv',
    'A-review-summary.csv':'A-review-summary-001.csv',
    'B-energy-decomposition.csv':'B-analysis-001/B-energy-decomposition.csv',
    'B-ci-bin-energy-carbon.csv':'B-analysis-001/B-ci-bin-energy-carbon.csv',
    'B-active-idle-energy-carbon.csv':'B-analysis-001/B-active-idle-energy-carbon.csv',
    'B-carbon-opportunity.csv':'B-analysis-001/B-carbon-opportunity.csv',
    'B-duration-slack.csv':'B-zero-workload-001/B-duration-slack.csv',
    'C-paired-outcomes.csv':'C-campaign-001/C-paired-outcomes.csv',
    'C-risk-vs-outcome.csv':'C-campaign-001/C-risk-vs-outcome.csv',
    'C-task-impact-summary.csv':'C-campaign-001/C-task-impact-summary.csv',
    'C-carbon-components.csv':'C-carbon-components.csv',
    'C-proxy-vs-physical.csv':'C-proxy-vs-physical.csv'}
records={}
for name,source in files.items():
    data=(root/source).read_bytes()
    assert len(data)<256000,name
    with (output/name).open('xb') as stream:stream.write(data)
    records[name]=dict(source=source,sha256=hashlib.sha256(data).hexdigest(),bytes=len(data),source_equal=True)
stats=[]
for label in ('1','1.5','2'):
    value=json.loads((root/'B-analysis-001'/('load-'+label)/'ci-statistics.json').read_text())
    row=dict(factor=float(label),**{k:v for k,v in value.items() if isinstance(v,(str,int,float,bool)) or v is None})
    row.update(high_run_lengths_minutes=';'.join(map(str,value['high_run_lengths_minutes'])),
               low_run_lengths_minutes=';'.join(map(str,value['low_run_lengths_minutes'])))
    stats.append(row)
with (output/'B-ci-temporal-statistics.csv').open('x',newline='',encoding='utf8') as stream:
    writer=csv.DictWriter(stream,list(stats[0]),lineterminator='\n')
    writer.writeheader();writer.writerows(stats)
replay=json.loads((root/'C-replay-verification-001.json').read_text())
with (output/'C-replay-audit.csv').open('x',newline='',encoding='utf8') as stream:
    writer=csv.DictWriter(stream,list(replay['rows'][0]),lineterminator='\n')
    writer.writeheader();writer.writerows(replay['rows'])
for name in ('B-ci-temporal-statistics.csv','C-replay-audit.csv'):
    records[name]=dict(derived=True,sha256=hashlib.sha256((output/name).read_bytes()).hexdigest(),bytes=(output/name).stat().st_size)
with (root/'tables-manifest.json').open('x',encoding='utf8') as stream:json.dump(records,stream,indent=2)
print(json.dumps(dict(files=len(records),bytes=sum(v['bytes'] for v in records.values()))))
