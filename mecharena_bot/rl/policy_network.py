"""
Lightweight actor-critic policy network for RL on top of NitroGen.

Architecture
------------
The policy takes the same 256x256 game frame that NitroGen sees, encodes it
with a small CNN, concatenates NitroGen's base action (mean over the 16-step
chunk), and outputs:

  Actor  – residual adjustments added to NitroGen's actions
           Continuous dims (joysticks):  Gaussian  (mean + learned log_std)
           Discrete  dims (buttons):     Bernoulli (logit adjustment)

  Critic – scalar state-value estimate V(s) for GAE / PPO

Action layout  (25 dims)
------------------------
  [0:2]   j_left   (x, y)     continuous  [-1, 1]
  [2:4]   j_right  (x, y)     continuous  [-1, 1]
  [4:25]  buttons  (21)       binary      {0, 1}
"""

import torch
import torch.nn as nn
from torch.distributions import Normal, Bernoulli

# Number of continuous joystick dims and discrete button dims
N_CONTINUOUS = 4   # j_left(2) + j_right(2)
N_DISCRETE = 21    # buttons
ACTION_DIM = N_CONTINUOUS + N_DISCRETE  # 25


class ActorCritic(nn.Module):
    """Small CNN actor-critic that outputs residual corrections to NitroGen."""

    def __init__(self, obs_size: int = 256, hidden: int = 256):
        super().__init__()

        # ── Visual encoder (256x256x3 → feature vector) ──────────────
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4, padding=0),  # → 63x63
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),  # → 30x30
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=0),  # → 14x14
            nn.ReLU(),
            nn.Flatten(),
        )

        # Calculate flattened size
        with torch.no_grad():
            dummy = torch.zeros(1, 3, obs_size, obs_size)
            enc_size = self.encoder(dummy).shape[1]  # 64*14*14 = 12544

        # ── Base-action context encoder ──────────────────────────────
        self.action_enc = nn.Sequential(
            nn.Linear(ACTION_DIM, 64),
            nn.ReLU(),
        )

        feat_size = enc_size + 64  # CNN features + action context

        # ── Shared trunk ──────────────────────────────────────────────
        self.trunk = nn.Sequential(
            nn.Linear(feat_size, hidden),
            nn.ReLU(),
        )

        # ── Actor head: continuous (joystick residuals) ───────────────
        self.cont_mean = nn.Linear(hidden, N_CONTINUOUS)
        self.cont_log_std = nn.Parameter(torch.zeros(N_CONTINUOUS))

        # ── Actor head: discrete (button logit adjustments) ───────────
        self.disc_logits = nn.Linear(hidden, N_DISCRETE)

        # ── Critic head ──────────────────────────────────────────────
        self.critic = nn.Linear(hidden, 1)

        # Initialise final layers near zero so initial residuals are small
        nn.init.orthogonal_(self.cont_mean.weight, gain=0.01)
        nn.init.zeros_(self.cont_mean.bias)
        nn.init.orthogonal_(self.disc_logits.weight, gain=0.01)
        nn.init.zeros_(self.disc_logits.bias)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.zeros_(self.critic.bias)

    # ------------------------------------------------------------------
    def _features(self, obs: torch.Tensor, base_action: torch.Tensor) -> torch.Tensor:
        """
        Args
            obs         : (B, 3, 256, 256) float32 normalised to [0, 1]
            base_action : (B, 25) NitroGen's mean action for the chunk
        """
        vis = self.encoder(obs)
        act = self.action_enc(base_action)
        return self.trunk(torch.cat([vis, act], dim=-1))

    # ------------------------------------------------------------------
    def forward(self, obs: torch.Tensor, base_action: torch.Tensor):
        """
        Returns
        -------
        cont_mean   : (B, 4)   Gaussian mean for joystick residuals
        cont_std    : (B, 4)   Gaussian std
        disc_logits : (B, 21)  Bernoulli logits for button adjustments
        value       : (B, 1)   state-value estimate
        """
        h = self._features(obs, base_action)
        cont_mean = self.cont_mean(h)
        cont_std = self.cont_log_std.exp().expand_as(cont_mean)
        disc_logits = self.disc_logits(h)
        value = self.critic(h)
        return cont_mean, cont_std, disc_logits, value

    # ------------------------------------------------------------------
    def get_action(self, obs: torch.Tensor, base_action: torch.Tensor,
                   deterministic: bool = False):
        """
        Sample an action and return everything PPO needs.

        Returns
        -------
        action     : (B, 25)  combined [cont_residual, button_probs]
        log_prob   : (B,)     log probability of the sampled action
        value      : (B,)     V(s)
        """
        cont_mean, cont_std, disc_logits, value = self.forward(obs, base_action)

        # ── Continuous (joystick residuals) ───────────────────────────
        cont_dist = Normal(cont_mean, cont_std)
        if deterministic:
            cont_action = cont_mean
        else:
            cont_action = cont_dist.rsample()

        # Clamp residual so final joystick stays in [-1, 1]
        cont_action = cont_action.clamp(-1.0, 1.0)

        # ── Discrete (button toggles) ────────────────────────────────
        disc_dist = Bernoulli(logits=disc_logits)
        if deterministic:
            disc_action = (disc_logits > 0).float()
        else:
            disc_action = disc_dist.sample()

        # ── Log probabilities ─────────────────────────────────────────
        cont_log_prob = cont_dist.log_prob(cont_action).sum(dim=-1)
        disc_log_prob = disc_dist.log_prob(disc_action).sum(dim=-1)
        total_log_prob = cont_log_prob + disc_log_prob

        action = torch.cat([cont_action, disc_action], dim=-1)
        return action.detach(), total_log_prob.detach(), value.squeeze(-1).detach()

    # ------------------------------------------------------------------
    def evaluate_actions(self, obs: torch.Tensor, base_action: torch.Tensor,
                         action: torch.Tensor):
        """
        Re-evaluate stored actions for PPO update.

        Args
            action : (B, 25)  previously sampled action

        Returns
        -------
        log_prob : (B,)
        value    : (B,)
        entropy  : (B,)
        """
        cont_mean, cont_std, disc_logits, value = self.forward(obs, base_action)

        cont_act = action[:, :N_CONTINUOUS]
        disc_act = action[:, N_CONTINUOUS:].round().clamp(0, 1)

        cont_dist = Normal(cont_mean, cont_std)
        disc_dist = Bernoulli(logits=disc_logits)

        cont_log_prob = cont_dist.log_prob(cont_act).sum(dim=-1)
        disc_log_prob = disc_dist.log_prob(disc_act).sum(dim=-1)

        cont_entropy = cont_dist.entropy().sum(dim=-1)
        disc_entropy = disc_dist.entropy().sum(dim=-1)

        return (
            cont_log_prob + disc_log_prob,
            value.squeeze(-1),
            cont_entropy + disc_entropy,
        )
