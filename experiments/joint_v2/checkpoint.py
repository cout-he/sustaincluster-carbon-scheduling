"""Immutable checkpoint generations and atomic named pointers, with byte guards."""
import json
import os
from pathlib import Path
import random
import tempfile

import numpy as np

from experiments.e1.contracts import content_hash, file_hash
from . import VERSION


def global_rng_state():
    import torch
    kind, keys, position, gaussian, cached = np.random.get_state()
    return dict(python=random.getstate(), numpy=dict(kind=kind, keys=keys.tolist(),
        position=position, gaussian=gaussian, cached=cached), torch=torch.get_rng_state())


def restore_global_rng(state):
    import torch
    random.setstate(state['python'])
    value = state['numpy']
    np.random.set_state((value['kind'], np.asarray(value['keys'], dtype=np.uint32),
                         value['position'], value['gaussian'], value['cached']))
    torch.set_rng_state(state['torch'])


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name+'.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf8', newline='\n') as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class CheckpointStore:
    def __init__(self, root, *, config_identity, data_identity):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.config_identity, self.data_identity = config_identity, data_identity

    def save(self, state, names=('latest',)):
        import torch
        for name in names:
            if name not in ('latest', 'best_feasible', 'best_deadline', 'final'):
                raise ValueError('Unsupported checkpoint alias')
        descriptor, temporary = tempfile.mkstemp(prefix='generation-', suffix='.pt', dir=self.root)
        path = Path(temporary)
        try:
            with os.fdopen(descriptor, 'wb') as stream:
                torch.save(dict(schema=VERSION, config_identity=self.config_identity,
                                data_identity=self.data_identity, state=state), stream)
                stream.flush(); os.fsync(stream.fileno())
            sha = file_hash(path)
            destination = self.root / (sha+'.pt')
            if destination.exists():
                path.unlink()
            else:
                os.replace(path, destination)
            pointer = dict(schema=VERSION, file=destination.name, sha256=sha,
                config_identity=self.config_identity, data_identity=self.data_identity,
                completed_episodes=state.get('completed_episodes', 0))
            for name in names:
                atomic_json(self.root/(name+'.json'), pointer)
            return pointer
        finally:
            if path.exists():
                path.unlink()

    def load(self, name='latest'):
        import torch
        pointer_path = Path(name) if str(name).endswith('.json') else self.root/(str(name)+'.json')
        pointer = json.loads(pointer_path.read_text(encoding='utf8'))
        for key, expected in (('schema', VERSION), ('config_identity', self.config_identity),
                              ('data_identity', self.data_identity)):
            if pointer.get(key) != expected:
                raise ValueError('Checkpoint pointer identity differs: '+key)
        path = (self.root/pointer['file']).resolve()
        if path.parent != self.root.resolve() or path.suffix != '.pt':
            raise ValueError('Checkpoint file escapes the checkpoint directory')
        if file_hash(path) != pointer['sha256']:
            raise ValueError('Checkpoint byte hash differs')
        payload = torch.load(path, map_location='cpu', weights_only=True)
        for key in ('schema', 'config_identity', 'data_identity'):
            if payload.get(key) != pointer[key]:
                raise ValueError('Checkpoint payload identity differs: '+key)
        return payload['state']

    def alias(self, name, existing='latest'):
        pointer = json.loads((self.root/(existing+'.json')).read_text(encoding='utf8'))
        # Revalidate the generation before publishing another name.
        self.load(existing)
        atomic_json(self.root/(name+'.json'), pointer)


def config_identity(config):
    return content_hash(config)
