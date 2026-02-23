"""
Proximal Policy Optimisation (PPO) trainer.

Runs multiple epochs of mini-batch updates on a collected rollout buffer.
Logs training metrics to TensorBoard.
"""

import os
import torch
import torch.nn as nn

from rl.policy_network import ActorCritic
from rl.rollout_buffer import RolloutBuffer


class PPOTrainer:
    """
    Standard PPO-Clip trainer.

    Parameters match common defaults from Schulman et al. and CleanRL.
    """

    def __init__(
        self,
        policy: ActorCritic,
        device: str = "cpu",
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        n_epochs: int = 4,
        batch_size: int = 64,
        target_kl: float = 0.02,
    ):
        self.policy = policy
        self.device = device
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.target_kl = target_kl

        # Optional TensorBoard writer (created lazily)
        self._writer = None
        self._update_step = 0

    # ------------------------------------------------------------------
    @property
    def writer(self):
        if self._writer is None:
            try:
                from torch.utils.tensorboard import SummaryWriter
                log_dir = os.path.join(
                    os.path.dirname(__file__), "..", "runs", "ppo")
                self._writer = SummaryWriter(log_dir=log_dir)
            except ImportError:
                pass  # tensorboard not installed — skip logging
        return self._writer

    # ------------------------------------------------------------------
    def update(self, buffer: RolloutBuffer, last_value: float) -> dict:
        """
        Run PPO update on the collected rollout.

        Args
            buffer     : filled RolloutBuffer
            last_value : V(s_{T+1}) for GAE bootstrap

        Returns
        -------
        dict of training metrics (losses, KL, entropy, etc.)
        """
        buffer.compute_gae(last_value, self.gamma, self.gae_lambda)

        n = buffer.size
        if n < self.batch_size:
            return {"skipped": True, "reason": f"buffer too small ({n})"}

        # Normalise advantages
        adv = buffer.advantages[:n]
        adv_mean = adv.mean()
        adv_std = adv.std() + 1e-8
        buffer.advantages[:n] = (adv - adv_mean) / adv_std

        metrics = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
            "clip_fraction": 0.0,
            "n_updates": 0,
        }

        for epoch in range(self.n_epochs):
            early_stop = False
            for batch in buffer.get_batches(self.batch_size):
                obs, base_act, actions, old_log_probs, returns, advantages = batch

                new_log_probs, values, entropy = self.policy.evaluate_actions(
                    obs, base_act, actions)

                # ── Policy (actor) loss ──────────────────────────────
                ratio = (new_log_probs - old_log_probs).exp()
                surr1 = ratio * advantages
                surr2 = ratio.clamp(1 - self.clip_eps,
                                    1 + self.clip_eps) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # ── Value (critic) loss ──────────────────────────────
                value_loss = nn.functional.mse_loss(values, returns)

                # ── Entropy bonus ────────────────────────────────────
                entropy_loss = -entropy.mean()

                # ── Total loss ───────────────────────────────────────
                loss = (policy_loss
                        + self.value_coef * value_loss
                        + self.entropy_coef * entropy_loss)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # ── Tracking ─────────────────────────────────────────
                with torch.no_grad():
                    approx_kl = (old_log_probs - new_log_probs).mean().item()
                    clip_frac = ((ratio - 1.0).abs() > self.clip_eps).float().mean().item()

                metrics["policy_loss"] += policy_loss.item()
                metrics["value_loss"] += value_loss.item()
                metrics["entropy"] += entropy.mean().item()
                metrics["approx_kl"] += approx_kl
                metrics["clip_fraction"] += clip_frac
                metrics["n_updates"] += 1

                if abs(approx_kl) > self.target_kl:
                    early_stop = True
                    break

            if early_stop:
                break

        # Average metrics
        nu = max(1, metrics["n_updates"])
        for k in ["policy_loss", "value_loss", "entropy", "approx_kl",
                   "clip_fraction"]:
            metrics[k] /= nu

        # TensorBoard logging
        self._log_metrics(metrics)
        self._update_step += 1

        return metrics

    # ------------------------------------------------------------------
    def _log_metrics(self, metrics: dict):
        w = self.writer
        if w is None:
            return
        step = self._update_step
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                w.add_scalar(f"ppo/{k}", v, step)
        w.flush()

    # ------------------------------------------------------------------
    def log_reward(self, episode_reward: float, episode_len: int):
        """Log per-episode reward to TensorBoard."""
        w = self.writer
        if w is None:
            return
        w.add_scalar("episode/reward", episode_reward, self._update_step)
        w.add_scalar("episode/length", episode_len, self._update_step)
        w.flush()

    # ------------------------------------------------------------------
    def save(self, path: str):
        """Save policy weights + optimiser state."""
        torch.save({
            "policy_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "update_step": self._update_step,
        }, path)
        print(f"[PPO] Checkpoint saved -> {path}")

    def load(self, path: str):
        """Load policy weights + optimiser state."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.policy.load_state_dict(ckpt["policy_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self._update_step = ckpt.get("update_step", 0)
        print(f"[PPO] Checkpoint loaded <- {path}  (step {self._update_step})")
