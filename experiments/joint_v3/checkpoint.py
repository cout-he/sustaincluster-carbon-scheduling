"""Versioned byte-bound V3 checkpoint generations and atomic episode pointers."""
import json
import os
from pathlib import Path
import tempfile
from experiments.e1.contracts import content_hash, file_hash
from experiments.joint_v2.checkpoint import atomic_json, global_rng_state, restore_global_rng
from . import VERSION


def code_identity():
    from experiments.joint_v2.train_v2 import _code_provenance
    root = Path(__file__).resolve().parents[2]
    files = _code_provenance()['code_sha256']
    for p in sorted(Path(__file__).parent.glob('*')):
        if p.suffix in ('.py', '.yaml'):
            files[p.relative_to(root).as_posix()] = file_hash(p)
    return dict(files=files, sha256=content_hash(files))


class Store:
    def __init__(self, root, config, data, code):
        self.root = Path(root)
        self.identity = dict(schema=VERSION, config_identity=config, data_identity=data, code_identity=code)

    def save(self, state, names=('latest',)):
        import torch
        if any(n not in ('latest', 'final', 'best_feasible', 'best_deadline') for n in names):
            raise ValueError('Unknown checkpoint pointer')
        self.root.mkdir(parents=True, exist_ok=True)
        fd, temp = tempfile.mkstemp(prefix='generation-', suffix='.tmp', dir=self.root)
        try:
            with os.fdopen(fd, 'wb') as stream:
                torch.save(dict(self.identity, state=state), stream)
                stream.flush(); os.fsync(stream.fileno())
            digest = file_hash(temp)
            target = self.root/(digest+'.pt')
            os.replace(temp, target)
            pointer = dict(self.identity, file=target.name, sha256=digest,
                           completed_episodes=state.get('completed_episodes', 0))
            for name in names: atomic_json(self.root/(name+'.json'), pointer)
            return pointer
        finally:
            if os.path.exists(temp): os.unlink(temp)

    def load(self, name='latest'):
        import torch
        pointer = json.loads((self.root/(str(name)+'.json')).read_text(encoding='utf8'))
        for k, v in self.identity.items():
            if pointer.get(k) != v: raise ValueError('Checkpoint '+k+' differs')
        target = (self.root/pointer['file']).resolve()
        if target.parent != self.root.resolve() or target.suffix != '.pt' or file_hash(target) != pointer['sha256']:
            raise ValueError('Checkpoint file/byte identity differs')
        payload = torch.load(target, map_location='cpu', weights_only=True)
        if any(payload.get(k) != v for k, v in self.identity.items()):
            raise ValueError('Checkpoint payload identity differs')
        return payload['state']

    def restore_aliases(self, state):
        for alias in ('best_feasible', 'best_deadline'):
            pointer = state.get(alias+'_pointer')
            target = self.root/(alias+'.json')
            if pointer: atomic_json(target, pointer)
            elif target.exists(): target.unlink()
