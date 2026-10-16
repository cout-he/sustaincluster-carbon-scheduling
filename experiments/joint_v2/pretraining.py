"""Transactional pretraining stages, resumed from the last committed state.

A partial stage is diagnostic work only: it cannot publish pretrained weights
or consume the sampler/model/global RNG streams of the eventual resumed run.
Each complete stage advances latest atomically before publishing its compatible
standalone checkpoint. A one-class EDF stage that deliberately skips BC still
completes when its demonstrations and explicit skip decision are complete.
"""
from copy import deepcopy
from pathlib import Path

from .checkpoint import atomic_json, global_rng_state, restore_global_rng
from .ppo import IPPOAgent
from .pretrain_task import _deadline_expired, pretrain_task
from .pretrain_cooling import pretrain_cooling


def _restore_boundary(boundary, agents, sampler):
    # Update the caller's mapping, rather than replacing its local reference.
    agents.clear()
    agents.update({role: IPPOAgent.from_state(payload) for role, payload in boundary['agents'].items()})
    sampler.load_state_dict(boundary['sampler'])
    restore_global_rng(boundary['global_rng'])


def _snapshot(state, agents, sampler):
    state['agents'] = {role: agent.export_state() for role, agent in agents.items()}
    state['sampler'] = sampler.state_dict()
    state['global_rng'] = global_rng_state()


def _stage_complete(summary):
    if summary.get('budget_exhausted', False):
        return False
    for key in ('complete', 'all_selected_scenarios_complete'):
        if key in summary and not summary[key]:
            return False
    ppo = summary.get('ppo', {})
    return not (ppo.get('budget_exhausted', False)
                or ('collection_complete' in ppo and not ppo['collection_complete']))


def run_pretraining(state, agents, sampler, store, train, cfg, out, deadline):
    """Return True only when both configured stages are committed or skipped.

    The caller must first save its initialized snapshot as latest. Interrupted
    exceptions propagate so its existing latest-generation recovery can run.
    Normal budget/prefix returns roll back in memory and return False.
    """
    if state.get('pretraining_skipped', False):
        return True
    demos = sorted(list(train), key=lambda scenario: scenario.scenario_id)
    if not demos or any(scenario.role != 'train' for scenario in demos):
        raise ValueError('Transactional pretraining requires Train role scenarios')
    output = Path(out) / 'pretrain'
    output.mkdir(parents=True, exist_ok=True)
    for role, procedure in (('task', pretrain_task), ('cooling', pretrain_cooling)):
        markers = state.get('pretraining_stages', {})
        if markers.get(role, False):
            # A crash can occur after latest commits but before its auxiliary
            # compatible file is published. Retain the original stage state so
            # repairing this file after later joint training cannot relabel a
            # fine-tuned model as the original pretrained initialization.
            saved = state.get('pretraining_agent_states', {}).get(role)
            if saved is not None and not (output / (role + '.pt')).exists():
                IPPOAgent.from_state(saved).save(output / (role + '.pt'))
            continue
        boundary = {key: deepcopy(state[key]) for key in ('agents', 'sampler', 'global_rng')}
        _restore_boundary(boundary, agents, sampler)
        if _deadline_expired(deadline):
            return False
        summary = procedure(agents[role], demos, cfg, output, deadline=deadline)
        if not isinstance(summary, dict):
            raise TypeError('Pretraining stage must return a JSON-safe summary mapping')
        if not _stage_complete(summary):
            atomic_json(output / (role + '_uncommitted_attempt.json'), summary)
            _restore_boundary(boundary, agents, sampler)
            return False
        agents[role].provenance[role + '_pretrain'] = deepcopy(summary)
        state.setdefault('pretrain', {})[role] = deepcopy(summary)
        state.setdefault('pretraining_stages', {})[role] = True
        state.setdefault('pretraining_agent_states', {})[role] = agents[role].export_state()
        _snapshot(state, agents, sampler)
        store.save(state, ('latest',))
        # Only a committed stage may produce a compatible initialization file.
        agents[role].save(output / (role + '.pt'))
    return True
