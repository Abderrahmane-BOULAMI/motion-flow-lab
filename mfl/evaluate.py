"""Sampling and MuJoCo-checked scoring of trajectories on unseen scenes."""
from __future__ import annotations

import time

import numpy as np
import torch

from mfl import data as D
from mfl import demos as Dm
from mfl import robot as R
from mfl import scenes as S


def generate(model, norm: D.Normaliser, s: S.Scenes, n: int = 8, steps=None, seed: int = 0) -> torch.Tensor:
    """(B, n, T, 4) joint trajectories, float64."""
    torch.manual_seed(seed)
    b = D.to_batch(s)
    with torch.no_grad():
        x = model.sample(b, n=n, steps=steps)
    return norm.decode(x, s.q0.float()).double()


def direct(s: S.Scenes) -> torch.Tensor:
    """Baseline: straight line in joint space from the start to an IK solution of the goal."""
    q_goal = S.ik(s.goal, gen=torch.Generator().manual_seed(0))
    return Dm.interpolate([s.q0, q_goal])[:, None]


def score(trajs: torch.Tensor, s: S.Scenes, checker: R.Checker) -> dict:
    trajs = trajs.numpy()
    b, n = trajs.shape[:2]
    ok = np.zeros((b, n), bool)
    coll, err, jerks, both, inside_share = [], [], [], [], []
    for i in range(b):
        c, r, g = s.center[i].numpy(), s.radius[i].item(), s.goal[i].numpy()
        modes = []
        for j in range(n):
            res = Dm.check(trajs[i, j], c, r, g, checker)
            ok[i, j] = res["success"]
            coll.append(res["collision"])
            err.append(res["goal_error"])
            jerks.append(Dm.jerk(trajs[i, j]))
            if res["success"]:
                modes.append(Dm.mode_of(trajs[i, j], c))
        both.append(len(set(modes)) == 2)
        inside_share.append(np.mean([m == "inside" for m in modes]) if modes else np.nan)
    return {"success_rate": float(ok.mean()), "collision_rate": float(np.mean(coll)),
            "both_modes": float(np.mean(both)),
            "inside_share": float(np.nanmean(inside_share)) if n > 1 and not np.all(np.isnan(inside_share))
            else float("nan"),
            "goal_error_mm": float(1000 * np.median(err)), "jerk": float(np.mean(jerks)), "per_sample_ok": ok}


def latency(model, norm: D.Normaliser, s: S.Scenes, n: int = 8, steps=None, repeats: int = 10) -> float:
    """Milliseconds to produce n trajectories for one scene on CPU."""
    one = s.subset(torch.arange(1))
    generate(model, norm, one, n, steps)
    t0 = time.perf_counter()
    for _ in range(repeats):
        generate(model, norm, one, n, steps)
    return 1000 * (time.perf_counter() - t0) / repeats
