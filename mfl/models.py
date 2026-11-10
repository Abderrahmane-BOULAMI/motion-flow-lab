"""Three policies on one backbone: behaviour cloning (regression), diffusion (DDPM / DDIM) and conditional flow matching.
All are conditioned on the start configuration, the goal and the observed obstacle (point cloud, or the true centre and
radius for the oracle ablation)."""
from __future__ import annotations

import math

import torch
from torch import nn


class PointNet(nn.Module):
    def __init__(self, out: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(3, 32), nn.GELU(), nn.Linear(32, 64), nn.GELU(), nn.Linear(64, out))

    def forward(self, pts: torch.Tensor) -> torch.Tensor:
        centred = pts - pts.mean(1, keepdim=True)
        feat = self.mlp(centred).max(dim=1).values
        return torch.cat([feat, pts.mean(1)], dim=-1)


class Encoder(nn.Module):
    """Context vector from (q0, goal, obstacle)."""

    def __init__(self, cond: int = 128, oracle: bool = False):
        super().__init__()
        self.oracle = oracle
        self.points = None if oracle else PointNet(64)
        obs = 4 if oracle else 64 + 3
        self.mlp = nn.Sequential(nn.Linear(4 + 3 + obs, cond), nn.GELU(), nn.Linear(cond, cond))

    def forward(self, b: dict) -> torch.Tensor:
        if self.oracle:
            obs = torch.cat([b["center"], b["radius"][:, None]], dim=-1)
        else:
            obs = self.points(b["cloud"])
        return self.mlp(torch.cat([b["q0"], b["goal"], obs], dim=-1))


def time_embedding(t: torch.Tensor, dim: int = 64) -> torch.Tensor:
    freqs = torch.exp(torch.linspace(0, math.log(1000), dim // 2, device=t.device))
    a = t[:, None] * freqs[None] * 2 * math.pi
    return torch.cat([a.sin(), a.cos()], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, width: int, cond: int):
        super().__init__()
        self.norm = nn.LayerNorm(width, elementwise_affine=False)
        self.film = nn.Linear(cond, 2 * width)
        self.mlp = nn.Sequential(nn.Linear(width, 2 * width), nn.GELU(), nn.Linear(2 * width, width))

    def forward(self, h, c):
        scale, shift = self.film(c).chunk(2, dim=-1)
        return h + self.mlp(self.norm(h) * (1 + scale) + shift)


class Backbone(nn.Module):
    """Residual MLP with FiLM conditioning. With timed=True it also takes the noise level / flow time."""

    def __init__(self, x_dim: int, cond: int = 128, width: int = 256, depth: int = 4, timed: bool = True):
        super().__init__()
        self.timed = timed
        self.t_mlp = nn.Sequential(nn.Linear(64, cond), nn.GELU(), nn.Linear(cond, cond)) if timed else None
        self.inp = nn.Linear(x_dim, width)
        self.blocks = nn.ModuleList([ResBlock(width, cond) for _ in range(depth)])
        self.out = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, x_dim))

    def forward(self, x, c, t=None):
        if self.timed:
            c = c + self.t_mlp(time_embedding(t))
        h = self.inp(x)
        for blk in self.blocks:
            h = blk(h, c)
        return self.out(h)


class Policy(nn.Module):
    def __init__(self, x_dim: int, oracle: bool = False, timed: bool = True):
        super().__init__()
        self.x_dim = x_dim
        self.enc = Encoder(oracle=oracle)
        self.net = Backbone(x_dim, timed=timed)

    def context(self, b: dict, n: int) -> torch.Tensor:
        return self.enc(b).repeat_interleave(n, dim=0)


class BC(Policy):
    """Regression on the demonstrations: one output per context, the mean of the modes when there are several."""

    def __init__(self, x_dim: int, oracle: bool = False):
        super().__init__(x_dim, oracle, timed=False)

    def predict(self, c):
        return self.net(torch.zeros(len(c), self.x_dim, device=c.device), c)

    def loss(self, b):
        return (self.predict(self.enc(b)) - b["x"]).pow(2).mean()

    @torch.no_grad()
    def sample(self, b, n: int = 8, steps=None):
        c = self.context(b, n)
        return self.predict(c).view(-1, n, self.x_dim)


class FlowMatching(Policy):
    """Conditional flow matching on the straight path x_t = (1 - t) noise + t data; velocity target data - noise."""

    def loss(self, b):
        x1 = b["x"]
        x0 = torch.randn_like(x1)
        t = torch.rand(len(x1), device=x1.device)
        xt = (1 - t[:, None]) * x0 + t[:, None] * x1
        return (self.net(xt, self.enc(b), t) - (x1 - x0)).pow(2).mean()

    @torch.no_grad()
    def sample(self, b, n: int = 8, steps=None, gen: torch.Generator | None = None):
        steps = steps or 10
        c = self.context(b, n)
        x = torch.randn(len(c), self.x_dim, generator=gen)
        dt = 1.0 / steps
        for k in range(steps):
            t = torch.full((len(c),), k * dt)
            x = x + dt * self.net(x, c, t)
        return x.view(-1, n, self.x_dim)


class Diffusion(Policy):
    """DDPM with a cosine schedule and v-prediction (stable at the near-zero signal-to-noise end of the schedule, where
    epsilon prediction amplifies the network error into x0). steps=None: ancestral sampling over all K levels;
    otherwise deterministic DDIM over `steps` evenly spaced levels."""

    def __init__(self, x_dim: int, oracle: bool = False, K: int = 100):
        super().__init__(x_dim, oracle)
        self.K = K
        s = 0.008
        t = torch.linspace(0, K, K + 1, dtype=torch.float64)
        f = torch.cos((t / K + s) / (1 + s) * math.pi / 2) ** 2
        abar = (f / f[0]).clamp(1e-5, 1.0)
        beta = (1 - abar[1:] / abar[:-1]).clamp(max=0.999)
        self.register_buffer("abar", abar[1:].float())       # abar for k = 1..K at index k-1
        self.register_buffer("beta", beta.float())

    def predict(self, x, c, k: int | torch.Tensor):
        """(x0, eps) estimates at level k, with x0 clamped to the range of the normalised data."""
        k = torch.as_tensor(k).expand(len(x))
        a = self.abar[k - 1][:, None]
        v = self.net(x, c, k.float() / self.K)
        x0 = (a.sqrt() * x - (1 - a).sqrt() * v).clamp(-5, 5)
        eps = (x - a.sqrt() * x0) / (1 - a).sqrt()
        return x0, eps

    def loss(self, b):
        x0 = b["x"]
        k = torch.randint(1, self.K + 1, (len(x0),), device=x0.device)
        a = self.abar[k - 1][:, None]
        e = torch.randn_like(x0)
        xt = a.sqrt() * x0 + (1 - a).sqrt() * e
        v = a.sqrt() * e - (1 - a).sqrt() * x0
        return (self.net(xt, self.enc(b), k.float() / self.K) - v).pow(2).mean()

    @torch.no_grad()
    def sample(self, b, n: int = 8, steps=None, gen: torch.Generator | None = None):
        c = self.context(b, n)
        x = torch.randn(len(c), self.x_dim, generator=gen)
        if steps is None:
            for k in range(self.K, 0, -1):
                x0, _ = self.predict(x, c, k)
                if k == 1:
                    x = x0
                    break
                a, ap, beta = self.abar[k - 1], self.abar[k - 2], self.beta[k - 1]
                mean = (ap.sqrt() * beta * x0 + (1 - beta).sqrt() * (1 - ap) * x) / (1 - a)
                var = beta * (1 - ap) / (1 - a)
                x = mean + var.sqrt() * torch.randn(x.shape, generator=gen)
            return x.view(-1, n, self.x_dim)
        levels = torch.linspace(self.K, 0, steps + 1).round().long().tolist()
        for k, k_next in zip(levels[:-1], levels[1:], strict=True):
            x0, eps = self.predict(x, c, k)
            if k_next == 0:
                x = x0
            else:
                an = self.abar[k_next - 1]
                x = an.sqrt() * x0 + (1 - an).sqrt() * eps
        return x.view(-1, n, self.x_dim)


KINDS = {"bc": BC, "ddpm": Diffusion, "flow": FlowMatching}
