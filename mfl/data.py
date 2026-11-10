"""Demonstration dataset: scenes kept only when both expert solutions pass the MuJoCo check, and the normalised
trajectory representation the policies learn (standardised offsets of every waypoint from the start configuration)."""
from __future__ import annotations

from dataclasses import fields

import numpy as np
import torch

from mfl import demos as Dm
from mfl import robot as R
from mfl import scenes as S


class Normaliser:
    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        self.mean, self.std = mean, std.clamp_min(1e-3)

    def encode(self, traj: torch.Tensor) -> torch.Tensor:
        """(B, T, 4) -> (B, (T-1) * 4)."""
        delta = traj[:, 1:] - traj[:, :1]
        return ((delta - self.mean) / self.std).flatten(1)

    def decode(self, x: torch.Tensor, q0: torch.Tensor) -> torch.Tensor:
        """(B, n, (T-1) * 4) and q0 (B, 4) -> (B, n, T, 4)."""
        b, n = x.shape[:2]
        delta = x.view(b, n, Dm.T - 1, R.DOF) * self.std + self.mean
        q0 = q0[:, None, None, :].expand(b, n, 1, R.DOF)
        return torch.cat([q0, q0 + delta], dim=2)


def build(n_scenes: int, seed: int, checker: R.Checker) -> dict:
    s = S.sample(n_scenes, seed)
    gen = torch.Generator().manual_seed(seed + 1)
    demos = {m: Dm.optimise(s, m, gen=gen) for m in Dm.MODES}
    keep = []
    for i in range(len(s)):
        ok = True
        for m in Dm.MODES:
            tr = demos[m][i].numpy()
            res = Dm.check(tr, s.center[i].numpy(), s.radius[i].item(), s.goal[i].numpy(), checker)
            ok = ok and res["success"] and Dm.mode_of(tr, s.center[i].numpy()) == m
        keep.append(ok)
    idx = torch.tensor(np.flatnonzero(keep))
    return {"scenes": s.subset(idx), "demos": {m: v[idx] for m, v in demos.items()}, "kept": len(idx),
            "requested": n_scenes}


def to_batch(s: S.Scenes) -> dict:
    return {"q0": s.q0.float(), "goal": s.goal.float(), "cloud": s.cloud.float(), "center": s.center.float(),
            "radius": s.radius.float()}


def training_set(tr: dict) -> tuple[dict, Normaliser]:
    """Both demonstrations of every scene, with the same context: the target distribution is bimodal."""
    traj = torch.cat([tr["demos"][m] for m in Dm.MODES]).float()
    delta = traj[:, 1:] - traj[:, :1]
    norm = Normaliser(delta.mean(0), delta.std(0))
    ctx = to_batch(tr["scenes"])
    batch = {k: torch.cat([v] * len(Dm.MODES)) for k, v in ctx.items()}
    batch["x"] = norm.encode(traj)
    return batch, norm


def _pack(part: dict) -> dict:
    out = dict(part)
    out["scenes"] = {f.name: getattr(part["scenes"], f.name) for f in fields(part["scenes"])}
    return out


def save(bundle: dict, path) -> None:
    torch.save({k: _pack(v) for k, v in bundle.items()}, path)


def load(path) -> dict:
    raw = torch.load(path, weights_only=False)
    for v in raw.values():
        v["scenes"] = S.Scenes(**v["scenes"])
    return raw
