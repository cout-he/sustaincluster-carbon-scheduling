"""Causal CI predictions. No future calendar is held by the online predictor."""
from dataclasses import dataclass
import csv
import math
from pathlib import Path
import numpy as np
import pandas as pd
from .contracts import STEP, finite, utc_time, file_hash

VERSION = 'e1-7b-online-ci-v1'

@dataclass(frozen=True)
class Forecast:
    times: tuple
    ci: tuple
    methods: tuple
    sources: tuple
    cold: bool
    fallback_count: int
    version: str = VERSION

class OnlineCIForecast:
    """Inclusive interval starts now+k*15min, up to 96 samples (<24h).

    Seasonal input for each future sample is tau-24h, now and now-24h.
    Identical repeated observation is idempotent; conflicting/backward input
    fails. At most 48h of already-observed CI is retained. No source for a
    prediction can be later than the current observed decision boundary.
    """
    def __init__(self, method='seasonal', prehistory=()):
        if method not in ('seasonal', 'persistence'): raise ValueError('Unknown forecast method')
        self.method, self.history, self.last = method, {}, None
        for row in prehistory:
            self.observe(row['time'],row['ci'])

    def observe(self, timestamp, ci):
        time=utc_time(timestamp,True);ci=finite(ci,'observed CI')
        if self.last is not None and time<self.last: raise ValueError('CI history cannot move backward')
        if time in self.history:
            if self.history[time]!=ci: raise ValueError('Conflicting duplicate CI observation')
            return
        self.history[time]=ci;self.last=time
        cutoff=time-pd.Timedelta(hours=48)
        self.history={t:v for t,v in self.history.items() if t>=cutoff}

    def forecast(self, now, horizon_steps):
        now=utc_time(now,True)
        if isinstance(horizon_steps,bool) or not isinstance(horizon_steps,int) or not 1<=horizon_steps<=96:
            raise ValueError('Forecast horizon must contain 1..96 inclusive grid samples')
        if now!=self.last or now not in self.history: raise ValueError('Forecast requires observed current CI')
        times=tuple(now+STEP*k for k in range(horizon_steps))
        current=self.history[now];lag=now-pd.Timedelta(hours=24)
        values,methods,sources=[],[],[]
        for k,time in enumerate(times):
            old=time-pd.Timedelta(hours=24)
            if k==0:
                value,method,source=current,'observed_current',(now,)
            elif self.method=='seasonal' and lag in self.history and old in self.history and old<=now:
                value=max(0.,self.history[old]+current-self.history[lag])
                method,source='seasonal',(old,now,lag)
            else:
                value,method,source=current,('persistence_missing_history' if self.method=='seasonal' else 'persistence'),(now,)
            values.append(float(value));methods.append(method);sources.append(source)
        missing=sum(m=='persistence_missing_history' for m in methods)
        return Forecast(times,tuple(values),tuple(methods),tuple(sources),bool(missing),missing)

def extract_prehistory(path, start, hours=24, expected_sha256=None):
    """Read only rows strictly before start; previous-hour observation basis.

    The source contract is strictly increasing UTC timestamps (verified for the
    registered file). Reject disorder in the consumed prefix and stop at start;
    later rows, including Test dates, are not parsed. A container SHA can be
    checked separately without extracting its future observations.
    """
    from .environment import CI_COLUMN
    start=utc_time(start,True)
    if hours!=24: raise ValueError('V1 freezes exactly 24h prehistory')
    if expected_sha256 is not None and file_hash(path)!=expected_sha256:raise ValueError('Historical source SHA mismatch')
    lower=start-pd.Timedelta(hours=24);rows={}
    previous=None
    with Path(path).open(encoding='utf-8-sig',newline='') as stream:
        for row in csv.DictReader(stream):
            # The declared Datetime (UTC) source column uses naive UTC strings.
            time=pd.Timestamp(row['Datetime (UTC)'])
            time=utc_time(time.tz_localize('UTC') if time.tzinfo is None else time)
            if previous is not None and time<=previous:raise ValueError('Historical source timestamp order differs')
            previous=time
            if time>=start:break
            if lower.floor('h')<=time<start:
                if time in rows: raise ValueError('Duplicate historical CI row')
                rows[time]=finite(row[CI_COLUMN],'historical CI')
    expected=pd.date_range(lower.floor('h'),start,freq='h',inclusive='left')
    if any(t not in rows for t in expected): raise ValueError('Historical CI coverage gap')
    return tuple(dict(time=t,ci=rows[t.floor('h')],source_time=t.floor('h'))
                 for t in pd.date_range(lower,start,freq='15min',inclusive='left'))

def evaluate_forecasts(samples, prehistory, horizons=(4,8,16)):
    """Offline scoring only: future labels are accessed AFTER emitting predictions."""
    rows=tuple(dict(time=utc_time(s['time'],True),ci=finite(s['ci'],'CI')) for s in samples)
    truth={r['time']:r['ci'] for r in rows};points=[]
    for method in ('persistence','seasonal'):
        predictor=OnlineCIForecast(method,prehistory)
        for row in rows:
            predictor.observe(row['time'],row['ci'])
            for h in horizons:
                # h means h steps ahead, distinct from inclusive prediction length.
                f=predictor.forecast(row['time'],int(h)+1);target=f.times[-1]
                if target not in truth: continue
                actual=truth[target];pred=f.ci[-1]
                points.append(dict(method=method,time=row['time'].isoformat(),target=target.isoformat(),
                    horizon_steps=h,predicted_ci=pred,actual_ci=actual,error=pred-actual,
                    direction_correct=int(np.sign(pred-row['ci'])==np.sign(actual-row['ci'])),
                    cold=f.methods[-1]=='persistence_missing_history'))
    summary=[]
    for method in ('persistence','seasonal'):
        for h in horizons:
            group=[p for p in points if p['method']==method and p['horizon_steps']==h]
            if not group: continue
            summary.append(dict(method=method,horizon_steps=h,samples=len(group),
                mae=math.fsum(abs(p['error']) for p in group)/len(group),
                rmse=math.sqrt(math.fsum(p['error']**2 for p in group)/len(group)),
                direction_accuracy=sum(p['direction_correct'] for p in group)/len(group),
                cold_samples=sum(p['cold'] for p in group)))
    return dict(version=VERSION,label_access='offline_after_prediction_only',summary=summary,points=points)
