import json

import pandas as pd


def test_stress_plot_keeps_all_levels_policies_and_zero_baselines(tmp_path):
    from experiments.e1.plot_contention import plot_stress
    policies=['immediate_fifo','edf_backfill','carbon_threshold_fifo']
    root=tmp_path/'campaign';root.mkdir()
    rows=[]
    for factor in (1,1.25,1.5,2):
        for p in policies:
            rows.append(dict(factor=factor,policy=p,phase='common',carbon_kg=100*factor,
                deadline_violation_rate_observed=.1*(factor-1),mean_jct_minutes_completed=30*factor,
                equal_workload_carbon_comparison_valid=True))
    pd.DataFrame(rows).to_csv(root/'comparison.csv',index=False)
    folder=root/'load_2p00';folder.mkdir()
    series=[]
    for p in policies:
        for time in pd.date_range('2023-07-01T05:00:00Z',periods=4,freq='15min'):
            series.append(dict(policy=p,interval_start=time.isoformat(),ci_g_per_kwh=200,running=2,pending=1))
    pd.DataFrame(series).to_csv(folder/'timeseries.csv',index=False)
    (folder/'design.json').write_text(json.dumps(dict(arrival_end='2023-07-01T05:30:00Z',threshold_g_per_kwh=137.65)))
    (root/'diagnostics.json').write_text('{}')
    out=tmp_path/'figures'
    info=plot_stress(root,out)
    assert info['summary_rows']==12 and info['time_series_rows']==12
    assert info['zero_y_axes'] and not info['inferential_statistics']
    assert all((out/(name+ext)).exists() for name in ('01_carbon','02_deadline','03_jct','04_contention') for ext in ('.png','.pdf','.csv'))
