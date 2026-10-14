"""Join registered C events with offline B opportunities; not prediction accuracy."""
import csv
from pathlib import Path
root=Path(__file__).resolve().parent
def rows(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))
c=rows(root/'C-campaign-001/C-paired-outcomes.csv')
features={r['case_id']:r for r in rows(root/'C-campaign-001/C-risk-vs-outcome.csv')}
lookup={label:{(r['task_id'],r['time']):r for r in rows(root/'B-analysis-001'/('load-'+label)/'task-carbon-opportunities.csv')}
        for label in ('1.5','2')}
joined=[]
def sign(value):
    return 1 if value>1e-9 else -1 if value< -1e-9 else 0
for row in c:
    feature=features[row['case_id']]
    proxy=lookup[format(float(row['factor']),'g')][(feature['task_id'],feature['time'])]
    actual_delta=float(row['delta_carbon'])
    proxy_delta=float(proxy['delta_proxy_carbon_kg'])
    joined.append(dict(case_id=row['case_id'],proxy_delta_kg=proxy_delta,physical_delta_kg=actual_delta,
        proxy_delay_minutes=15,sign_relation='actual_zero' if sign(actual_delta)==0 else
        'same_nonzero_sign' if sign(proxy_delta)==sign(actual_delta) else 'opposite_nonzero_sign',
        proxy_scope='isolated_task_IT_only_clairvoyant',actual_scope='full_system_after_one_defer_then_EDF',
        is_prediction_accuracy=False))
with (root/'C-proxy-vs-physical.csv').open('x',newline='',encoding='utf8') as stream:
    writer=csv.DictWriter(stream,list(joined[0]),lineterminator='\n')
    writer.writeheader();writer.writerows(joined)
print({name:sum(r['sign_relation']==name for r in joined) for name in
       ('same_nonzero_sign','opposite_nonzero_sign','actual_zero')})
