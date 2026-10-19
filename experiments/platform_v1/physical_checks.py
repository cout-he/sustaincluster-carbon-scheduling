"""Non-data-dependent engineering probes, never a workload integration run."""
import argparse
import itertools
from pathlib import Path

from .acceptance import csv_write,write_json
from .inventory import build_inventory,load_config
from .physics import PhysicalModel


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='research_notes/platform-v1/engineering')
    args=parser.parse_args();out=Path(args.output);cfg=load_config();rows=[];failures=[]
    for name in ['OM','OH','OL']:
        inv=build_inventory(cfg,name);model=PhysicalModel(inv,cfg)
        for fractions,ambient,supply in itertools.product(itertools.product([0.,.5,1.],repeat=3),[-5,15,35,45],[18,22,26]):
            demand=tuple(u*c for u,c in zip(fractions,inv.capacity));it=model.it_power(demand)
            r=model.transition(24,it['it_kw'],ambient,supply)
            if abs(r['balance_residual_kj'])>1e-5:failures.append('heat balance')
            rows.append(dict(capacity=name,cpu_fraction=fractions[0],gpu_fraction=fractions[1],mem_fraction=fractions[2],
                             outdoor_c=ambient,supply_c=supply,**it,**r,total_kw=it['it_kw']+r['hvac_kw']))
        # Coordinate-wise increasing load, all other coordinates fixed.
        for axis,other,ambient,supply in itertools.product(range(3),itertools.product([0.,.5,1.],repeat=2),[-5,15,35,45],[18,22,26]):
            values=[]
            for fraction in [0.,.25,.5,.75,1.]:
                fractions=list(other);fractions.insert(axis,fraction)
                it=model.it_power(tuple(u*c for u,c in zip(fractions,inv.capacity)))['it_kw']
                values.append(it+model.transition(24,it,ambient,supply)['hvac_kw'])
            if any(b<a-1e-10 for a,b in zip(values,values[1:])):failures.append('nonmonotone individual resource')
    csv_write(out/'physical-probes.csv',rows)
    write_json(out/'checks.json',dict(manual_points=len(rows),coordinate_checks=3*3*9*4*3,
                                     maximum_residual_kj=max(abs(r['balance_residual_kj']) for r in rows),
                                     failures=failures,scope='engineering only, no task/environment data, no acceptance matrix'))
    print(dict(points=len(rows),failures=failures))
    if failures:raise SystemExit(1)


if __name__=='__main__':main()
