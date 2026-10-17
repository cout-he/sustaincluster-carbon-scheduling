"""Centralized queue pooling with two distinct cost outputs at each boundary."""
import numpy as np
import torch
from torch import nn

GLOBAL_CONTEXT_DIM, QUEUE_DIM = 26, 5


class DualCostCritic(nn.Module):
    def __init__(self, context_dim=GLOBAL_CONTEXT_DIM, hidden=64):
        super().__init__()
        self.context_dim = context_dim
        self.phi = nn.Sequential(nn.Linear(QUEUE_DIM, 16), nn.Tanh(), nn.Linear(16, 16), nn.Tanh())
        self.value = nn.Sequential(nn.Linear(context_dim+32, hidden), nn.Tanh(),
                                   nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 2))

    def forward(self, states):
        if not states:
            return torch.empty((0, 2))
        contexts = np.asarray([s['context'] for s in states], dtype=np.float32)
        if contexts.shape != (len(states), self.context_dim) or not np.isfinite(contexts).all():
            raise ValueError('Invalid centralized context')
        lengths = [len(s['queue']) for s in states]
        width = max(1, max(lengths))
        queue = np.zeros((len(states), width, QUEUE_DIM), dtype=np.float32)
        mask = torch.zeros((len(states), width), dtype=torch.bool)
        for i, s in enumerate(states):
            q = np.asarray(s['queue'], dtype=np.float32)
            if q.shape != (lengths[i], QUEUE_DIM) or not np.isfinite(q).all():
                raise ValueError('Invalid observed queue')
            queue[i, :len(q)] = q
            mask[i, :len(q)] = True
        encoded = self.phi(torch.from_numpy(queue))
        mean = (encoded*mask[..., None]).sum(1)/mask.sum(1).clamp(min=1)[:, None]
        maximum = encoded.masked_fill(~mask[..., None], -torch.inf).amax(1)
        maximum = torch.where(mask.any(1)[:, None], maximum, torch.zeros_like(maximum))
        return self.value(torch.cat((torch.from_numpy(contexts), mean, maximum), 1))


class LocalDualCritic(nn.Module):
    """Separately trained IPPO-style ablation; no other controller observation."""
    def __init__(self, observation_dim, hidden=64):
        super().__init__()
        self.value = nn.Sequential(nn.Linear(observation_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 2))

    def forward(self, states):
        return self.value(torch.as_tensor(np.asarray(states), dtype=torch.float32))
