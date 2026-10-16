"""Serializable stratified Train-only sampling with bounded heavy-load mass."""
from collections import Counter, defaultdict
import copy
import json
from pathlib import Path

import numpy as np

from experiments.e1.contracts import content_hash


class ScenarioSampler:
    def __init__(self, scenarios, seed, config=None):
        self.scenarios = sorted(list(scenarios),key=lambda item:item.scenario_id)
        if not self.scenarios or any(item.role != 'train' for item in self.scenarios):
            raise ValueError('ScenarioSampler requires a nonempty Train-only scenario pool')
        if len({item.scenario_id for item in self.scenarios}) != len(self.scenarios):
            raise ValueError('Duplicate Train scenario IDs')
        defaults=json.loads(Path(__file__).with_name('scenario_config.yaml').read_text(encoding='utf8'))['sampler']
        provided=config or {}
        self.config=dict(defaults,**provided.get('sampler',provided))
        cap=float(self.config['max_high_pressure_probability'])
        if not 0 <= cap <= 1 or float(self.config['high_pressure_threshold']) <= 0:
            raise ValueError('Invalid high-pressure sampling bound')
        self.seed=int(seed)
        self.rng=np.random.default_rng(self.seed)
        self.pool_identity=content_hash([dict(scenario_id=item.scenario_id,identity=item.identity,
                                             statistics=item.statistics) for item in self.scenarios])
        self.last_fragment=None
        self.draws=0
        self.counts=Counter()
        self.stage_counts=Counter()
        self.last_sampling_audit={}

    def _weights(self, stage):
        if stage not in (0,1,2):
            raise ValueError('Curriculum stage must be 0, 1 or 2')
        groups=defaultdict(list)
        for i,current in enumerate(self.scenarios):
            stats=current.statistics
            group=tuple(stats.get(key,'unclassified') for key in
                        ('load_category','deadline_category','ci_category','temperature_category'))
            groups[group].append(i)
        weights=np.zeros(len(self.scenarios),dtype=float)
        for group,indices in groups.items():
            weight=1.
            if stage == 0:
                if group[0] == 'high': weight*=float(self.config['stage0_high_load_weight'])
                if group[1] == 'tight': weight*=float(self.config['stage0_tight_deadline_weight'])
            elif stage == 1 and group[0] == 'high':
                weight*=.5
            extra=self.config.get('weights',{})
            if isinstance(extra,dict):
                weight*=float(extra.get('|'.join(group),1.))
            if not np.isfinite(weight) or weight < 0:
                raise ValueError('Scenario stratum weights must be finite nonnegative')
            for i in indices:
                weights[i]=weight/len(indices)
        if not weights.any():
            raise ValueError('Scenario sampling configuration assigns no positive mass')
        # Avoid immediate repeats of the same real fragment, including another
        # synthetic augmentation, whenever an alternative has nonzero mass.
        alternatives=np.asarray([s.identity.get('source_fragment_id',s.scenario_id) != self.last_fragment
                                 for s in self.scenarios])
        if self.last_fragment is not None and weights[alternatives].sum() > 0:
            weights[~alternatives]=0.
        heavy=np.asarray([float(s.statistics.get('capacity_pressure',s.statistics.get('pressure_proxy',0.)))
                          >= float(self.config['high_pressure_threshold']) for s in self.scenarios])
        cap=float(self.config['max_high_pressure_probability'])
        heavy_mass,normal_mass=weights[heavy].sum(),weights[~heavy].sum()
        fallback=False
        if heavy_mass and normal_mass:
            probability=heavy_mass/(heavy_mass+normal_mass)
            if probability > cap:
                weights[heavy]*=(cap*normal_mass/((1-cap)*heavy_mass)) if cap < 1 else 1.
        elif heavy_mass and not normal_mass:
            # Repeat avoidance must not force a high-pressure draw that breaks
            # the declared cap: reinstate the previous light fragment instead.
            normal_indices=np.flatnonzero(~heavy)
            if len(normal_indices):
                weights[normal_indices]=1./len(normal_indices)
                if cap < 1:
                    weights[heavy]*=cap/((1-cap)*heavy_mass)
                fallback=True
            else:
                fallback=True  # All-heavy user pools cannot satisfy a mass cap.
        weights/=weights.sum()
        self.last_sampling_audit=dict(stage=stage,eligible_scenarios=int(np.count_nonzero(weights)),
            high_pressure_probability=float(weights[heavy].sum()),high_pressure_cap=cap,
            finite_eligible_fallback=fallback,avoided_previous_fragment=not fallback,
            pressure_is_proxy_not_feasibility_claim=True)
        return weights

    def sample(self, stage=0):
        weights=self._weights(stage)
        index=int(self.rng.choice(len(self.scenarios),p=weights))
        result=self.scenarios[index]
        self.last_fragment=result.identity.get('source_fragment_id',result.scenario_id)
        self.draws+=1
        self.counts[result.scenario_id]+=1
        self.stage_counts[str(stage)]+=1
        return result

    def state_dict(self):
        return dict(schema='joint-v2-scenario-sampler-v1',pool_identity=self.pool_identity,
            config=copy.deepcopy(self.config),seed=self.seed,rng=copy.deepcopy(self.rng.bit_generator.state),
            last_fragment=self.last_fragment,draws=self.draws,counts=dict(self.counts),
            stage_counts=dict(self.stage_counts),last_sampling_audit=copy.deepcopy(self.last_sampling_audit))

    def load_state_dict(self, state):
        if state.get('schema') != 'joint-v2-scenario-sampler-v1' or state.get('pool_identity') != self.pool_identity:
            raise ValueError('Sampler pool identity differs from saved training state')
        if state.get('config') != self.config:
            raise ValueError('Sampler configuration differs from saved training state')
        if not isinstance(state.get('draws'),int) or state['draws'] < 0:
            raise ValueError('Saved sampler draw count is invalid')
        self.rng.bit_generator.state=copy.deepcopy(state['rng'])
        self.seed=state['seed']
        self.last_fragment=state['last_fragment']
        self.draws=state['draws']
        self.counts=Counter(state['counts'])
        self.stage_counts=Counter(state['stage_counts'])
        self.last_sampling_audit=copy.deepcopy(state.get('last_sampling_audit',{}))
