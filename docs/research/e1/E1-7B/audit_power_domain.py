"""Read-only power-domain/clipping audit; does not refit or schedule tasks."""
import argparse,json,math
from pathlib import Path
import pandas as pd
from experiments.e1.online_power_surrogate import PowerSurrogate
from experiments.e1.contracts import file_hash,write_json
from experiments.e1.carbon_gain_experiment import load_frozen
from experiments.e1.run import _fresh_run
from experiments.e1.temporal import write_csv

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path(__file__).parent)
    ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();root=a.root
    freeze=root/'calibration-003/freeze.json';model=PowerSurrogate.from_dict(load_frozen(freeze)['model'])
    def raw(it,temp):
        i=it/1000;t=temp/40
        return math.fsum(c*x for c,x in zip(model.hvac_coefficients,(1,i,i*i,t,i*t)))
    def action(out):
        rows=[];identities={}
        for split,folder in (('train','train-002'),('validation','validation-001')):
            for factor in ('1','1.5','2'):
                for path in sorted((root/folder/('load-'+factor)).glob('*/intervals.csv')):
                    identities[str(path.relative_to(root))]=file_hash(path)
                    data=pd.read_csv(path).to_dict('records');values=[];idle=[];pred_it=[];temps=[]
                    for r in data:
                        it=model.it((r['used_cores'],r['used_gpus'],r['used_mem']));temp=r['temperature_c']
                        h=model.hvac(it,temp);assert math.isfinite(h) and 0<=h<=model.hvac_max_kw
                        values.append(raw(it,temp));idle.append(raw(model.it_intercept_kw,temp));pred_it.append(it);temps.append(temp)
                    rows.append(dict(split=split,factor=float(factor),policy=path.parent.name,intervals=len(values),
                        temperature_min_c=min(temps),temperature_max_c=max(temps),predicted_it_min_kw=min(pred_it),predicted_it_max_kw=max(pred_it),
                        hvac_raw_min_kw=min(values),hvac_raw_max_kw=max(values),
                        actual_occupancy_negative_clips=sum(v<0 for v in values),actual_occupancy_upper_clips=sum(v>model.hvac_max_kw for v in values),
                        idle_background_negative_probes=sum(v<0 for v in idle),
                        scope='observed interval occupancies; idle probes hypothetical, not per-candidate branch clip counts'))
        write_csv(out/'power-domain-clipping.csv',rows)
        write_json(out/'provenance.json',dict(freeze_sha256=file_hash(freeze),inputs=identities,
            script_sha256=file_hash(Path(__file__)),test_access=False,model_refit=False))
    _fresh_run(a.output,action)
if __name__=='__main__':main()
