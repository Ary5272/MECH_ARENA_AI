"""
Fixed-size rollout buffer for on-policy PPO collection.

Stores one rollout of N steps, then yields mini-batches for PPO updates.
Cleared after each update cycle.
"""

import torch
import numpy as np


class RolloutBuffer:
    """
    Stores transitions collected during gameplay for PPO training.

    Each "step" corresponds to one 16-frame NitroGen action chunk (~1 second).
    """

    def __init__(self, buffer_size: int, obs_shape: tuple, action_dim: int,
                 device: str = "cpu"):
        self.buffer_size = buffer_size
        self.device = device
        self.ptr = 0
        self.full = False

        # Pre-allocate tensors
        self.obs = torch.zeros((buffer_size, *obs_shape), dtype=torch.float32)
        self.base_actions = torch.zeros((buffer_size, action_dim), dtype=torch.float32)
        self.actions = torch.zeros((buffer_size, action_dim), dtype=torch.float32)
        self.log_probs = torch.zeros(buffer_size, dtype=torch.float32)
        self.rewards = torch.zeros(buffer_size, dtype=torch.float32)
        self.values = torch.zeros(buffer_size, dtype=torch.float32)
        self.dones = torch.zeros(buffer_size, dtype=torch.float32)

        # Computed after rollout collection
        self.advantages = torch.zeros(buffer_size, dtype=torch.float32)
        self.returns = torch.zeros(buffer_size, dtype=torch.float32)

    # ------------------------------------------------------------------
    @property
    def size(self) -> int:
        return self.buffer_size if self.full else self.ptr

    # ------------------------------------------------------------------
    def add(self, obs: np.ndarray, base_action: np.ndarray, action: np.ndarray,
            log_prob: float, reward: float, value: float, done: bool):
        """Store one transition."""
        self.obs[self.ptr] = torch.from_numpy(obs)
        self.base_actions[self.ptr] = torch.from_numpy(base_action)
        self.actions[self.ptr] = torch.from_numpy(action)
        self.log_probs[self.ptr] = log_prob
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.dones[self.ptr] = float(done)

        self.ptr += 1
        if self.ptr >= self.buffer_size:
            self.full = True
            self.ptr = 0

    # ------------------------------------------------------------------
    def compute_gae(self, last_value: float, gamma: float = 0.99,
                    gae_lambda: float = 0.95):
        """
        Compute Generalised Advantage Estimation after a rollout.

        Args
            last_value : V(s_{T+1}) – the value of the state after the
                         last collected step (bootstrap).
        """
        n = self.size
        last_gae = 0.0
        for t in reversed(range(n)):
            if t == n - 1:
                next_value = last_value
                next_non_terminal = 1.0 - self.dones[t]
            else:
                next_value = self.values[t + 1]
                next_non_terminal = 1.0 - self.dones[t]

            delta = (self.rewards[t]
                     + gamma * next_value * next_non_terminal
                     - self.values[t])
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            self.advantages[t] = last_gae

        self.returns[:n] = self.advantages[:n] + self.values[:n]

    # ------------------------------------------------------------------
    def get_batches(self, batch_size: int):
        """
        Yield random mini-batches of indices for PPO epochs.

        Each call reshuffles.  Yields tuples of tensors moved to self.device.
        """
        n = self.size
        indices = np.random.permutation(n)

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            idx = indices[start:end]
            idx_t = torch.tensor(idx, dtype=torch.long)

            yield (
                self.obs[idx_t].to(self.device),
                self.base_actions[idx_t].to(self.device),
                self.actions[idx_t].to(self.device),
                self.log_probs[idx_t].to(self.device),
                self.returns[idx_t].to(self.device),
                self.advantages[idx_t].to(self.device),
            )

    # ------------------------------------------------------------------
    def reset(self):
        """Clear the buffer for the next rollout."""
        self.ptr = 0
        self.full = False
