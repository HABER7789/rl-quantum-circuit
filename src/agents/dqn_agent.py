# Double DQN agent. Online net picks actions, target net evaluates them.
# Target syncs every N steps to reduce overestimation bias.

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .replay_buffer import ReplayBuffer, Batch


class QNetwork(nn.Module):
    """Simple MLP: obs → hidden → hidden → Q-values per action."""

    def __init__(self, obs_dim, n_actions, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class DDQNAgent:

    def __init__(
        self,
        obs_dim,
        n_actions,
        gamma=0.99,
        lr=1e-3,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=10_000,
        batch_size=64,
        buffer_size=50_000,
        target_update=200,
        hidden_dim=256,
        min_buffer=512,
        device="cpu",
        seed=42,
    ):
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update = target_update
        self.min_buffer = min_buffer
        self.device = torch.device(device)

        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.online_net = QNetwork(obs_dim, n_actions, hidden_dim).to(self.device)
        self.target_net = copy.deepcopy(self.online_net).to(self.device)
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        self.rng = np.random.default_rng(seed)
        self.buffer = ReplayBuffer(buffer_size, obs_dim, device=device)

        self.total_steps = 0
        self.n_updates = 0

    def select_action(self, state, greedy=False):
        if not greedy and self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, self.n_actions))

        with torch.no_grad():
            state_t = torch.from_numpy(state).unsqueeze(0).to(self.device)
            return int(self.online_net(state_t).argmax(dim=1).item())

    def store_transition(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)
        self.total_steps += 1
        self._decay_epsilon()

    def update(self):
        """
        DDQN gradient step. No-ops until buffer has enough samples.

        Target: a* = argmax Q_online(s', ·)
                y  = r + gamma * Q_target(s', a*) * (1 - done)
        """
        if not self.buffer.ready(self.min_buffer):
            return None

        batch = self.buffer.sample(self.batch_size, self.rng)

        with torch.no_grad():
            next_actions = self.online_net(batch.next_states).argmax(dim=1, keepdim=True)
            next_q = self.target_net(batch.next_states).gather(1, next_actions).squeeze(1)
            targets = batch.rewards + self.gamma * next_q * (1.0 - batch.dones)

        current_q = self.online_net(batch.states).gather(1, batch.actions.unsqueeze(1)).squeeze(1)
        loss = self.loss_fn(current_q, targets)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.n_updates += 1

        if self.n_updates % self.target_update == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        return float(loss.item())

    def _decay_epsilon(self):
        progress = min(1.0, self.total_steps / self.epsilon_decay)
        self.epsilon = self.epsilon_start + progress * (self.epsilon_end - self.epsilon_start)

    def save(self, path):
        torch.save({
            "online_net":  self.online_net.state_dict(),
            "target_net":  self.target_net.state_dict(),
            "optimizer":   self.optimizer.state_dict(),
            "total_steps": self.total_steps,
            "epsilon":     self.epsilon,
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(ckpt["online_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.total_steps = ckpt["total_steps"]
        self.epsilon = ckpt["epsilon"]
