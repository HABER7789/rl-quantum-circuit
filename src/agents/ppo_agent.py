# ppo_agent.py
# PPO agent for gate selection.
#
# On-policy: collect one full episode, do a few gradient updates, throw it away.
# Key trick: clip the policy ratio so we don't make huge updates that
# destabilize training. Advantage estimation via GAE (lambda=0.95 worked well).
#
# Shared actor-critic trunk — both heads see the same features,
# which seemed to converge faster than separate networks.

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from dataclasses import dataclass, field


class ActorCritic(nn.Module):

    def __init__(self, obs_dim, n_actions, hidden_dim=256):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.actor_head = nn.Linear(hidden_dim, n_actions)
        self.critic_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        features = self.trunk(x)
        return self.actor_head(features), self.critic_head(features)

    def get_action_and_value(self, x, action=None):
        logits, value = self.forward(x)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value


@dataclass
class RolloutBuffer:
    """On-policy buffer — cleared after each update."""
    states:    list = field(default_factory=list)
    actions:   list = field(default_factory=list)
    log_probs: list = field(default_factory=list)
    rewards:   list = field(default_factory=list)
    values:    list = field(default_factory=list)
    dones:     list = field(default_factory=list)

    def clear(self):
        self.states.clear(); self.actions.clear(); self.log_probs.clear()
        self.rewards.clear(); self.values.clear(); self.dones.clear()

    def __len__(self):
        return len(self.rewards)


class PPOAgent:

    def __init__(
        self,
        obs_dim,
        n_actions,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        n_epochs=4,
        batch_size=64,
        vf_coef=0.5,
        ent_coef=0.01,
        max_grad_norm=0.5,
        hidden_dim=256,
        device="cpu",
        seed=42,
    ):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.device = torch.device(device)

        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)

        self.ac = ActorCritic(obs_dim, n_actions, hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.ac.parameters(), lr=lr)

        self.rollout = RolloutBuffer()
        self.total_steps = 0

    def select_action(self, state):
        state_t = torch.from_numpy(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_t, log_prob_t, _, value_t = self.ac.get_action_and_value(state_t)
        return int(action_t.item()), float(log_prob_t.item()), float(value_t.item())

    def store_transition(self, state, action, log_prob, reward, value, done):
        self.rollout.states.append(state.copy())
        self.rollout.actions.append(action)
        self.rollout.log_probs.append(log_prob)
        self.rollout.rewards.append(reward)
        self.rollout.values.append(value)
        self.rollout.dones.append(done)
        self.total_steps += 1

    def update(self, last_value=0.0):
        """
        Compute GAE advantages then run n_epochs of PPO updates.

        GAE: delta_t = r_t + gamma*V(s_{t+1})*(1-done) - V(s_t)
             A_t = delta_t + gamma*lambda*(1-done)*A_{t+1}
        PPO: ratio = pi_new/pi_old, clip to [1-eps, 1+eps], take min
        """
        if len(self.rollout) == 0:
            return {}

        rewards = np.array(self.rollout.rewards, dtype=np.float32)
        values  = np.array(self.rollout.values,  dtype=np.float32)
        dones   = np.array(self.rollout.dones,   dtype=np.float32)
        T = len(rewards)

        # GAE — work backwards
        advantages = np.zeros(T, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(T)):
            next_val = last_value if t == T - 1 else values[t + 1]
            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae

        returns = advantages + values

        states_t  = torch.from_numpy(np.stack(self.rollout.states)).to(self.device)
        actions_t = torch.tensor(self.rollout.actions, dtype=torch.int64).to(self.device)
        old_lp_t  = torch.tensor(self.rollout.log_probs, dtype=torch.float32).to(self.device)
        adv_t     = torch.from_numpy(advantages).to(self.device)
        returns_t = torch.from_numpy(returns).to(self.device)

        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        metric_sums = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "total_loss": 0.0}
        n_updates = 0

        for _ in range(self.n_epochs):
            perm = self.rng.permutation(T)
            for start in range(0, T, self.batch_size):
                idx = perm[start : start + self.batch_size]
                idx_t = torch.from_numpy(idx).long().to(self.device)

                _, new_lp, entropy, new_val = self.ac.get_action_and_value(
                    states_t[idx_t], actions_t[idx_t]
                )

                ratio = torch.exp(new_lp - old_lp_t[idx_t])
                adv_batch = adv_t[idx_t]

                surr1 = ratio * adv_batch
                surr2 = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * adv_batch
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = nn.functional.mse_loss(new_val.squeeze(), returns_t[idx_t])
                entropy_loss = -entropy.mean()

                total_loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss

                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.ac.parameters(), self.max_grad_norm)
                self.optimizer.step()

                metric_sums["policy_loss"] += float(policy_loss.item())
                metric_sums["value_loss"]  += float(value_loss.item())
                metric_sums["entropy"]     += float(-entropy_loss.item())
                metric_sums["total_loss"]  += float(total_loss.item())
                n_updates += 1

        if n_updates > 0:
            for k in metric_sums:
                metric_sums[k] /= n_updates

        self.rollout.clear()
        return metric_sums

    def save(self, path):
        torch.save({
            "ac":          self.ac.state_dict(),
            "optimizer":   self.optimizer.state_dict(),
            "total_steps": self.total_steps,
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.ac.load_state_dict(ckpt["ac"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.total_steps = ckpt["total_steps"]
