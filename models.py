from __future__ import annotations

import copy

import torch
import torch.nn as nn


def mlp(sizes: list[int], activation: type[nn.Module] = nn.ReLU, out_activation: type[nn.Module] | None = None) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else out_activation
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if act is not None:
            layers.append(act())
    return nn.Sequential(*layers)


class PolicyNet(nn.Module):
    """
    Deterministic policy for DDPG-style training.
    - Input: state (12)
    - Output: action (7) with tanh squash -> [-1, 1]

    Convention:
      - First 6 outputs are torques in [-1, 1]
      - Last output is release signal mapped to [0, 1] via (tanh + 1)/2 at action() time
    """

    def __init__(self, state_dim: int, action_dim: int, hidden: int = 256) -> None:
        super().__init__()
        self.net = mlp([state_dim, hidden, hidden, action_dim], activation=nn.ReLU, out_activation=nn.Tanh)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class CriticNet(nn.Module):
    """
    Q(s,a) critic.
    - Input: concat(state, action) = 12 + 7
    - Output: scalar Q
    """

    def __init__(self, state_dim: int, action_dim: int, hidden: int = 256) -> None:
        super().__init__()
        self.net = mlp([state_dim + action_dim, hidden, hidden, 1], activation=nn.ReLU, out_activation=None)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        return self.net(x).squeeze(-1)


def make_target(module: nn.Module) -> nn.Module:
    target = copy.deepcopy(module)
    for p in target.parameters():
        p.requires_grad_(False)
    return target


class GaussianPolicy(nn.Module):
    """
    Tanh-squashed Gaussian policy for SAC.
    Pre-tanh u ~ N(mu, sigma); raw = tanh(u) in (-1, 1)^action_dim.
    Map raw to env torques + release via env_action_torch in trainSAC (same convention as DDPG).
    """

    def __init__(self, state_dim: int, action_dim: int, hidden: int = 256) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.net = mlp([state_dim, hidden, hidden, 2 * action_dim], activation=nn.ReLU, out_activation=None)

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.net(state)
        mean, log_std = x.chunk(2, dim=-1)
        log_std = torch.clamp(log_std, -20.0, 2.0)
        return mean, log_std

    def sample(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Reparameterized sample; returns (raw_tanh, log_pi) with log_pi shape (B,)."""
        mean, log_std = self.forward(state)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        u = dist.rsample()
        raw = torch.tanh(u)
        log_pi = dist.log_prob(u).sum(dim=-1) - torch.log(1.0 - raw.pow(2) + 1e-6).sum(dim=-1)
        return raw, log_pi

    def mean_action(self, state: torch.Tensor) -> torch.Tensor:
        """Deterministic tanh(mean) for evaluation."""
        mean, _ = self.forward(state)
        return torch.tanh(mean)

