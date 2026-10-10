import numpy as np
import pandas as pd


def test_plotting_keeps_incomplete_endpoint_states_and_aligns_cumulative_interval_end():
    from experiments.e1.plot_temporal import series_xy
    frame=pd.DataFrame({'hours_from_start':[0,.25],'running':[1,1],
                        'pending':[2,2],'cumulative_carbon_kg':[.2,.5]})
    x,y=series_xy(frame,'running',.5,terminal=1)
    assert list(x)==[0,.25,.5] and list(y)==[1,1,1]
    x,y=series_xy(frame,'pending',.5,terminal=2)
    assert list(y)==[2,2,2]
    x,y=series_xy(frame,'cumulative_carbon_kg',.5)
    assert list(x)==[0,.25,.5] and list(y)==[0,.2,.5]


def test_coincident_policy_annotation_depends_on_both_coordinates():
    from experiments.e1.plot_temporal import coincidence_note
    data={'immediate_fifo':([0,1],[0,2]),'edf_backfill':([0,1],[0,2])}
    assert coincidence_note(data)=='FIFO and EDF coincide'
    data['edf_backfill']=([0,1],[0,3])
    assert coincidence_note(data)==''
    data['edf_backfill']=([0,2],[0,2])
    assert coincidence_note(data)==''
