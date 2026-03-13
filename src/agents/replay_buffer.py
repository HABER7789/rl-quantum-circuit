# replay_buffer.py
# Experience replay for DDQN. Store transitions, sample random minibatches.
# Pre-allocated arrays so we don't keep resizing.

import numpy as np
import torch
from typing import NamedTuple


class Batch(NamedTuple):
    states:      torch.Tensor
    actions:     torch.Tensor
    rewards:     torch.Tensor
    next_states: torch.Tensor
    dones:       torch.Tensor


class ReplayBuffer:
    """Circular buffer for (s, a, r, s', done) transitions."""

    def __init__(self, capacity, obs_dim, device="cpu"):
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.device = torch.device(device)

        self._states      = np.zeros((capacity, obs_dim), dtype=np.float32)
        self._actions     = np.zeros(capacity, dtype=np.int64)
        self._rewards     = np.zeros(capacity, dtype=np.float32)
        self._next_states = np.zeros((capacity, obs_dim), dtype=np.float32)
        self._dones       = np.zeros(capacity, dtype=np.float32)

        self._write_ptr = 0
        self._size = 0

    def push(self, state, action, reward, next_state, done):
        self._states[self._write_ptr]      = state
        self._actions[self._write_ptr]     = action
        self._rewards[self._write_ptr]     = reward
        self._next_states[self._write_ptr] = next_state
        self._dones[self._write_ptr]       = float(done)

        self._write_ptr = (self._write_ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size, rng):
        if self._size < batch_size:
            raise ValueError(f"Only {self._size} transitions stored, need {batch_size}.")

        idx = rng.integers(0, self._size, size=batch_size)

        def t(arr):
            return torch.from_numpy(arr).to(self.device)

        return Batch(
            states      = t(self._states[idx]),
            actions     = t(self._actions[idx]),
            rewards     = t(self._rewards[idx]),
            next_states = t(self._next_states[idx]),
            dones       = t(self._dones[idx]),
        )

    def __len__(self):
        return self._size

    def ready(self, min_size=256):
        return self._size >= min_size
