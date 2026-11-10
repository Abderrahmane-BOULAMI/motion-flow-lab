"""Expert demonstrations by trajectory optimisation. Each scene gets two valid solutions, one passing inside the obstacle
(between it and the robot base) and one passing outside: the multimodal data a regression policy cannot represent."""
from __future__ import annotations

import math

import numpy as np
import torch

from mfl import robot as R
from mfl import scenes as S

T = 32
MARGIN = 0.03
MODES = ("inside", "outside")
WEIGHTS = {"goal": 300.0, "smooth": 3.0, "vel": 0.05, "collision": 3000.0, "limits": 100.0, "floor": 300.0}


def via_points(s: S.Scenes, mode: str, offset: float = 0.10) -> torch.Tensor:
    """End-effector via point beside the obstacle, on the inner or outer side."""
    c = s.center
    yaw = torch.atan2(c[:, 1], c[:, 0])
    rc = c[:, :2].norm(dim=-1)
    sign = -1.0 if mode == "inside" else 1.0
    rad = (rc + sign * (s.radius + offset)).clamp(0.18, 0.68)
    return torch.stack([rad * torch.cos(yaw), rad * torch.sin(yaw), c[:, 2]], dim=-1)


def interpolate(waypoints: list[torch.Tensor], t: int = T) -> torch.Tensor:
    """Piecewise linear joint-space path through the given configurations, shape (B, t, 4)."""
    k = len(waypoints) - 1
    s = torch.linspace(0, k, t, dtype=torch.float64)
    seg = s.floor().clamp(max=k - 1).long()
    a = (s - seg)[None, :, None]
    w = torch.stack(waypoints, dim=1)
    return w[:, seg] * (1 - a) + w[:, seg + 1] * a


def upsample(q: torch.Tensor, k: int = 3) -> torch.Tensor:
    a = torch.linspace(0, 1, k + 1, dtype=q.dtype)[:-1]
    mid = q[:, :-1, None] + a[None, None, :, None] * (q[:, 1:, None] - q[:, :-1, None])
    return torch.cat([mid.flatten(1, 2), q[:, -1:]], dim=1)


def costs(q: torch.Tensor, s: S.Scenes) -> dict[str, torch.Tensor]:
    low, high = torch.tensor(R.Q_LOW), torch.tensor(R.Q_HIGH)
    dense = upsample(q)
    vel = q[:, 1:] - q[:, :-1]
    acc = vel[:, 1:] - vel[:, :-1]
    return {"goal": (R.ee(q[:, -1]) - s.goal).pow(2).sum(-1),
            "smooth": acc.pow(2).sum((-1, -2)),
            "vel": vel.pow(2).sum((-1, -2)),
            "collision": torch.relu(MARGIN - R.clearance(dense, s.center, s.radius)).pow(2).sum(-1),
            "limits": (torch.relu(q - high) + torch.relu(low - q)).pow(2).sum((-1, -2)),
            "floor": torch.relu(0.02 - R.floor_margin(dense)).pow(2).sum(-1)}


def optimise(s: S.Scenes, mode: str, iters: int = 700, gen: torch.Generator | None = None) -> torch.Tensor:
    """Optimises a T-waypoint trajectory from q0 to the goal through the side given by `mode`. Returns (B, T, 4)."""
    q_via = S.ik(via_points(s, mode), gen=gen)
    q_goal = S.ik(s.goal, gen=gen)
    init = interpolate([s.q0, q_via, q_goal])
    free = init[:, 1:].clone().requires_grad_(True)
    opt = torch.optim.Adam([free], lr=0.02)
    for i in range(iters):
        for g in opt.param_groups:
            g["lr"] = 0.002 + 0.018 * 0.5 * (1 + math.cos(math.pi * i / iters))
        q = torch.cat([s.q0[:, None], free], dim=1)
        c = costs(q, s)
        loss = sum(WEIGHTS[k] * v for k, v in c.items()).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return torch.cat([s.q0[:, None], free.detach()], dim=1)


def mode_of(traj, center) -> str:
    """Side of the obstacle the end effector passes on, measured where its yaw is closest to the obstacle's."""
    p = R.ee(torch.as_tensor(np.asarray(traj), dtype=torch.float64)).numpy()
    c = np.asarray(center)
    yaw_c = math.atan2(c[1], c[0])
    yaw = np.arctan2(p[:, 1], p[:, 0])
    k = int(np.argmin(np.abs(np.angle(np.exp(1j * (yaw - yaw_c))))))
    return "inside" if np.linalg.norm(p[k, :2]) < np.linalg.norm(c[:2]) else "outside"


def check(traj, center, radius: float, goal, checker: R.Checker, tol: float = 0.02) -> dict:
    """Ground-truth verdict in MuJoCo: collision-free on substeps, within joint limits, goal reached."""
    traj = np.asarray(traj, dtype=float)
    checker.set_obstacle(center, radius)
    free, worst = checker.trajectory_ok(traj)
    in_limits = bool(np.all(traj >= R.Q_LOW - 1e-3) and np.all(traj <= R.Q_HIGH + 1e-3))
    err = float(np.linalg.norm(checker.ee(traj[-1]) - np.asarray(goal)))
    return {"success": bool(free and in_limits and err < tol), "collision": not free, "in_limits": in_limits,
            "goal_error": err, "clearance": worst}


def jerk(traj) -> float:
    traj = np.asarray(traj)
    return float(np.mean(np.abs(np.diff(traj, n=3, axis=0)).sum(-1)))
