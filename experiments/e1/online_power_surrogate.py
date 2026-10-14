"""Train-fitted descriptive surrogates, independent of live physical state."""
from dataclasses import asdict, dataclass
import math
import numpy as np
from .contracts import content_hash, finite

VERSION='e1-7b-power-surrogate-v1'
RESOURCES=('used_cores','used_gpus','used_mem')

@dataclass(frozen=True)
class PowerSurrogate:
    it_intercept_kw: float
    it_slopes: tuple
    hvac_coefficients: tuple
    capacity: tuple
    hvac_max_kw: float
    it_fit_range: tuple=(0.,10000.)
    temperature_fit_range: tuple=(-30.,50.)
    alpha: float=0.

    def __post_init__(self):
        for key in ('it_slopes','hvac_coefficients','capacity','it_fit_range','temperature_fit_range'):
            object.__setattr__(self,key,tuple(getattr(self,key)))
        finite(self.it_intercept_kw,'IT intercept');finite(self.hvac_max_kw,'HVAC cap',True)
        if len(self.it_slopes)!=3 or len(self.capacity)!=3 or len(self.hvac_coefficients)!=5:
            raise ValueError('Surrogate dimensions differ')
        for v in self.it_slopes:finite(v,'IT slope')
        for v in self.capacity:finite(v,'capacity',True)
        for v in (*self.hvac_coefficients,*self.it_fit_range,*self.temperature_fit_range,self.alpha):
            if not math.isfinite(float(v)):raise ValueError('Surrogate parameters must be finite')
        if not 0<=self.alpha<1:raise ValueError('Stable HVAC alpha required')
        if any(len(x)!=2 or x[0]>x[1] for x in (self.it_fit_range,self.temperature_fit_range)):
            raise ValueError('Invalid calibration domain')

    def it(self, demand):
        demand=tuple(finite(v,'resource demand') for v in demand)
        if len(demand)!=3 or any(q>c for q,c in zip(demand,self.capacity)):
            raise ValueError('Predicted occupancy outside physical capacity')
        return self.it_intercept_kw+math.fsum(q*b for q,b in zip(demand,self.it_slopes))

    def hvac(self, it_kw, temperature_c, previous_kw=0.):
        finite(it_kw,'predicted IT');finite(previous_kw,'past HVAC')
        if not math.isfinite(float(temperature_c)):raise ValueError('Weather must be finite')
        i=it_kw/1000.;t=temperature_c/40.
        value=math.fsum(c*x for c,x in zip(self.hvac_coefficients,(1.,i,i*i,t,i*t)))
        response=min(self.hvac_max_kw,max(0.,value))
        return min(self.hvac_max_kw,self.alpha*previous_kw+(1-self.alpha)*response)

    def to_dict(self):return dict(version=VERSION,**asdict(self))

    @classmethod
    def from_dict(cls,value):
        value=dict(value)
        if value.pop('version')!=VERSION:raise ValueError('Power surrogate version differs')
        return cls(**value)

def fit_surrogate(it_rows,hvac_rows,*,capacity,role):
    if role!='train':raise ValueError('Power fitting is Train-only')
    it_rows,hvac_rows=list(it_rows),list(hvac_rows)
    if not it_rows or not hvac_rows:raise ValueError('Nonempty Train calibration required')
    if any(r.get(key,'train')!='train' for r in it_rows+hvac_rows for key in ('split','role','data_role')):
        raise ValueError('Forbidden non-Train model labels')
    q=np.asarray([[r[k] for k in RESOURCES] for r in it_rows],dtype=float)
    target=np.asarray([r['it_power_kw'] for r in it_rows],dtype=float)
    if not len(q) or not np.isfinite(q).all() or not np.isfinite(target).all():raise ValueError('Invalid Train rows')
    scales=np.maximum(1.,np.max(np.abs(q),axis=0));x=np.column_stack((np.ones(len(q)),q/scales))
    coeff,_,rank,sing=np.linalg.lstsq(x,target,rcond=None)
    coeff=coeff/np.r_[1.,scales]
    if rank!=4 or np.any(coeff<0):raise ValueError('Unidentifiable or negative IT OLS; require explicit new model')
    i=np.asarray([r['it_power_kw']/1000 for r in hvac_rows]);t=np.asarray([r['temperature_c']/40 for r in hvac_rows])
    y=np.asarray([r['hvac_power_kw'] for r in hvac_rows]);hx=np.column_stack((np.ones(len(i)),i,i*i,t,i*t))
    if not np.isfinite(hx).all() or not np.isfinite(y).all() or np.any(y<0):raise ValueError('Invalid HVAC Train rows')
    hc,_,hrank,hs=np.linalg.lstsq(hx,y,rcond=None)
    if hrank!=5:raise ValueError('HVAC coefficients not identifiable')
    m=PowerSurrogate(float(coeff[0]),tuple(map(float,coeff[1:])),tuple(map(float,hc)),tuple(capacity),
        float(max(1.,4*y.max())),(float(i.min()*1000),float(i.max()*1000)),(float(t.min()*40),float(t.max()*40)))
    def errors(actual,predicted):
        error=np.asarray(predicted)-actual
        return dict(mae=float(np.mean(abs(error))),rmse=float(np.sqrt(np.mean(error**2))))
    it_pred=x@(coeff*np.r_[1.,scales]);hvac_pred=np.clip(hx@hc,0,m.hvac_max_kw)
    propagated=[m.hvac(m.it(tuple(r[k] for k in RESOURCES)),r['temperature_c']) for r in hvac_rows]
    report=dict(version=VERSION,fit_role='train',model_kind='static quadratic HVAC, alpha=0; not physical inertia',
        it=dict(samples=len(q),rank=int(rank),condition=float(sing[0]/sing[-1]),**errors(target,it_pred)),
        hvac=dict(samples=len(y),rank=int(hrank),condition=float(hs[0]/hs[-1]),
            clipped=int(np.sum((hx@hc<0)|(hx@hc>m.hvac_max_kw))),**errors(y,hvac_pred)),
        propagated_hvac=errors(y,propagated),calibration_rows_sha256=content_hash(dict(it=it_rows,hvac=hvac_rows)))
    return m,report

def evaluate_surrogate(model,rows):
    rows=list(rows);records=[]
    for r in rows:
        it=model.it(tuple(r[k] for k in RESOURCES));h=model.hvac(it,r['temperature_c'])
        records.append(dict(actual_it_kw=r['it_power_kw'],predicted_it_kw=it,
            actual_hvac_kw=r['hvac_power_kw'],predicted_hvac_kw=h,
            it_error=it-r['it_power_kw'],hvac_error=h-r['hvac_power_kw'],
            outside_fit=int(not model.it_fit_range[0]<=it<=model.it_fit_range[1] or
                            not model.temperature_fit_range[0]<=r['temperature_c']<=model.temperature_fit_range[1])))
    summary={}
    for name in ('it','hvac'):
        e=np.asarray([r[name+'_error'] for r in records])
        summary[name]=dict(mae=float(np.mean(abs(e))),rmse=float(np.sqrt(np.mean(e**2))))
    summary['outside_fit']=sum(r['outside_fit'] for r in records)
    return dict(summary=summary,records=records)
