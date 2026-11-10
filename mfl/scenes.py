"""Random reaching scenes: start and goal on either side of a spherical obstacle placed on the way, plus the point
cloud a fixed depth camera would see of that obstacle."""
from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np
import torch

from mfl import robot as R

CAMERA = [1.1, 0.0, 0.9]
N_POINTS = 48


@dataclass
class Scenes:
    start_ee: torch.Tensor   # (B, 3)
    goal: torch.Tensor       # (B, 3)
    center: torch.Tensor     # (B, 3)
    radius: torch.Tensor     # (B,)
    q0: torch.Tensor         # (B, 4) start configuration
    cloud: torch.Tensor      # (B, N_POINTS, 3) observed obstacle points

    def __len__(self) -> int:
        return len(self.radius)

    def subset(self, idx) -> "Scenes":
        return Scenes(**{f.name: getattr(self, f.name)[idx] for f in fields(self)})


def ik(target: torch.Tensor, iters: int = 400, seeds: int = 6, gen: torch.Generator | None = None) -> torch.Tensor:
    """Batched inverse kinematics by gradient descent from several seeds; keeps the best, elbow-up preferred."""
    b = len(target)
    low, high = torch.tensor(R.Q_LOW), torch.tensor(R.Q_HIGH)
    home = torch.tensor(R.HOME)
    yaw = torch.atan2(target[:, 1], target[:, 0])
    q = home + 0.4 * (torch.rand(b, seeds, R.DOF, generator=gen, dtype=torch.float64) - 0.5)
    q[..., 0] = yaw[:, None]
    q[:, 0] = home
    q[:, 0, 0] = yaw
    q.requires_grad_(True)
    opt = torch.optim.Adam([q], lr=0.03)
    for _ in range(iters):
        err = (R.ee(q) - target[:, None]).pow(2).sum(-1)
        prior = 1e-3 * (q[..., 1:] - home[1:]).pow(2).sum(-1)
        lim = (torch.relu(q - high) + torch.relu(low - q)).pow(2).sum(-1)
        loss = (err + prior + 10 * lim).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        q = torch.maximum(torch.minimum(q, high), low)
        err = (R.ee(q) - target[:, None]).norm(dim=-1)
        best = err.argmin(dim=1)
    return q[torch.arange(b), best].detach()


def observe(center: torch.Tensor, radius: torch.Tensor, noise: float = 0.004,
            gen: torch.Generator | None = None) -> torch.Tensor:
    """N_POINTS surface points on the hemisphere facing the camera, with depth noise along the viewing ray."""
    b = len(radius)
    cam = torch.tensor(CAMERA, dtype=center.dtype)
    d = torch.randn(b, N_POINTS, 3, generator=gen, dtype=center.dtype)
    d = d / d.norm(dim=-1, keepdim=True)
    view = cam - center
    view = view / view.norm(dim=-1, keepdim=True)
    flip = ((d * view[:, None]).sum(-1, keepdim=True) < 0).to(d.dtype)
    d = d - 2 * flip * (d * view[:, None]).sum(-1, keepdim=True) * view[:, None]
    pts = center[:, None] + radius[:, None, None] * d
    if noise > 0:
        ray = pts - cam
        ray = ray / ray.norm(dim=-1, keepdim=True)
        pts = pts + noise * torch.randn(b, N_POINTS, 1, generator=gen, dtype=center.dtype) * ray
    return pts


def _arc_point(yaw, rad, z):
    return torch.stack([rad * torch.cos(yaw), rad * torch.sin(yaw), z], dim=-1)


def sample(n: int, seed: int) -> Scenes:
    gen = torch.Generator().manual_seed(seed)

    def u(lo, hi):
        return lo + (hi - lo) * torch.rand(n, generator=gen, dtype=torch.float64)

    ys, yg = u(0.55, 1.05), u(-1.05, -0.55)
    rs, rg = u(0.38, 0.55), u(0.38, 0.55)
    zs, zg = u(0.15, 0.35), u(0.15, 0.35)
    start_ee, goal = _arc_point(ys, rs, zs), _arc_point(yg, rg, zg)
    yc = (ys + yg) / 2 + u(-0.08, 0.08)
    center = _arc_point(yc, (rs + rg) / 2 + u(-0.02, 0.02), (zs + zg) / 2 + u(-0.02, 0.02))
    radius = u(0.07, 0.10)
    q0 = ik(start_ee, gen=gen)
    start_ee = R.ee(q0)  # exact start reached by the arm
    cloud = observe(center, radius, gen=gen)
    return Scenes(start_ee, goal, center, radius, q0, cloud)
