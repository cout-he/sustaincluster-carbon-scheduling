"""Predicted full-occupancy gains, never actual system counterfactual labels."""
import math
import time
import numpy as np
import pandas as pd
from .contracts import STEP, finite
from .cars_state import demand

VERSION='e1-7b-carbon-gain-v1'

class CarbonGainEstimator:
    def __init__(self, model, forecaster, max_steps=96, tail_steps=0):
        if max_steps!=96 or tail_steps!=0:raise ValueError('V1 freezes 96-step cap and zero static HVAC tail')
        self.model,self.forecaster=model,forecaster
        self.max_steps,self.tail_steps=max_steps,tail_steps
        self._forecast_cache=None;self._background_cache={}

    def estimate(self,spec,snapshot,temperature_c,previous_hvac_kw=0.):
        clock=time.perf_counter();now=snapshot.now
        if spec.arrival_time>now:raise ValueError('Cannot estimate a future task')
        if not math.isfinite(float(temperature_c)):raise ValueError('Current weather required')
        finish=now+pd.Timedelta(minutes=spec.duration)
        # Match real nanosecond finish/ceil release, including float epsilon.
        n=max(1,int((finish.ceil('15min')-now)/STEP));h=n+1+self.tail_steps
        out=dict(version=VERSION,status='ok',occupied_steps=n,horizon_steps=h,
            gain_it_kg=None,gain_hvac_kg=None,gain_total_kg=None,
            forecast_cold=False,forecast_fallback_count=0,outside_fit=False,
            background_assumption='known running releases only; other pending and future arrivals omitted')
        if h>self.max_steps:
            out['status']='unknown_horizon';out['elapsed_seconds']=time.perf_counter()-clock;return out
        running=tuple(snapshot.running);target=demand(spec)
        # A just-started task quantized to zero duration still occupies this
        # interval: the kernel only releases tasks before the next EDF pass.
        releases=tuple((r.task_id,r.demand,max(r.release_time.value,now.value+STEP.value)) for r in running)
        key=(now,releases)
        if key not in self._background_cache:
            # Precompute ceil releases once, retaining identical fsum ordering.
            stamps=tuple(now.value+STEP.value*j for j in range(self.max_steps))
            bg=tuple(tuple(math.fsum(q[k] for _,q,release in releases if release>stamp)
                for k in range(3)) for stamp in stamps)
            self._background_cache={key:bg}
        bg=self._background_cache[key][:h]
        if any(math.fsum((q,b))>c for q,b,c in zip(target,bg[1],snapshot.capacity)):
            out['status']='capacity_opportunity_unavailable';out['elapsed_seconds']=time.perf_counter()-clock;return out
        if any(math.fsum((q,b))>c for q,b,c in zip(target,bg[0],snapshot.capacity)):
            out['status']='capacity_opportunity_unavailable';out['elapsed_seconds']=time.perf_counter()-clock;return out
        if self._forecast_cache is None or self._forecast_cache.times[0]!=now:
            self._forecast_cache=self.forecaster.forecast(now,self.max_steps)
        f=self._forecast_cache;ci=f.ci[:h]
        out.update(forecast_cold=any(m=='persistence_missing_history' for m in f.methods[:h]),
            forecast_fallback_count=sum(m=='persistence_missing_history' for m in f.methods[:h]))
        base_it=[self.model.it(q) for q in bg]
        increment=math.fsum(q*b for q,b in zip(target,self.model.it_slopes))
        it_now=[base_it[k]+(increment if k<n else 0) for k in range(h)]
        it_later=[base_it[k]+(increment if 1<=k<n+1 else 0) for k in range(h)]
        hn,hd,hb=[],[],[];state_n=state_d=state_b=finite(previous_hvac_kw,'past HVAC')
        for k in range(h):
            state_n=self.model.hvac(it_now[k],temperature_c,state_n)
            state_d=self.model.hvac(it_later[k],temperature_c,state_d)
            state_b=self.model.hvac(base_it[k],temperature_c,state_b)
            hn.append(state_n);hd.append(state_d);hb.append(state_b)
        # Marginal IT cancels the shared base exactly; no attribution of idle IT.
        it_gain=increment*.25*math.fsum((ci[0],-ci[n]))/1000
        hvac_gain=math.fsum((a-b)*.25*c/1000 for a,b,c in zip(hn,hd,ci))
        out.update(gain_it_kg=it_gain,gain_hvac_kg=hvac_gain,gain_total_kg=math.fsum((it_gain,hvac_gain)),
            carbon_it_now_kg=increment*.25*math.fsum(ci[:n])/1000,
            carbon_it_defer_kg=increment*.25*math.fsum(ci[1:n+1])/1000,
            carbon_hvac_now_kg=math.fsum((a-b)*.25*c/1000 for a,b,c in zip(hn,hb,ci)),
            carbon_hvac_defer_kg=math.fsum((a-b)*.25*c/1000 for a,b,c in zip(hd,hb,ci)),
            predicted_start_now=now.isoformat(),predicted_start_defer=(now+STEP).isoformat(),
            predicted_end=(now+STEP*h).isoformat(),
            outside_fit=not (self.model.it_fit_range[0]<=min(it_now+it_later) and
                max(it_now+it_later)<=self.model.it_fit_range[1] and
                self.model.temperature_fit_range[0]<=temperature_c<=self.model.temperature_fit_range[1]),
            elapsed_seconds=time.perf_counter()-clock)
        return out
