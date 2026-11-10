"""Unit tests: kinematics and collision checks against MuJoCo, demonstrations, normalisation, and the core claim on a
toy problem (a generative policy keeps both modes, regression averages them)."""
from __future__ import annotations

import numpy as np
import torch

from mfl import data as D
from mfl import demos as Dm
from mfl import models as M
from mfl import robot as R
from mfl import scenes as S


def test_forward_kinematics_matches_mujoco():
    ck = R.Checker.make()
    rng = np.random.default_rng(0)
    for _ in range(100):
        q = rng.uniform(R.Q_LOW, R.Q_HIGH)
        assert np.allclose(ck.ee(q), R.ee(torch.tensor(q)).numpy(), atol=1e-9)


def test_clearance_matches_mujoco_distance():
    ck = R.Checker.make()
    center, radius = np.array([0.4, 0.05, 0.3]), 0.09
    ck.set_obstacle(center, radius)
    rng = np.random.default_rng(1)
    for _ in range(100):
        q = rng.uniform(R.Q_LOW, R.Q_HIGH)
        approx = R.clearance(torch.tensor(q)[None, None], torch.tensor(center)[None], torch.tensor([radius])).item()
        assert abs(approx - ck.min_distance(q)) < 1e-6        # capsule-sphere distance is exact


def test_demonstrations_are_valid_and_cover_both_modes():
    ck = R.Checker.make()
    s = S.sample(12, seed=3)
    for mode in Dm.MODES:
        traj = Dm.optimise(s, mode, iters=500, gen=torch.Generator().manual_seed(0))
        res = [Dm.check(traj[i].numpy(), s.center[i].numpy(), s.radius[i].item(), s.goal[i].numpy(), ck)
               for i in range(len(s))]
        assert np.mean([r["success"] for r in res]) >= 0.9
        assert all(Dm.mode_of(traj[i].numpy(), s.center[i].numpy()) == mode for i in range(len(s)))


def test_direct_path_collides():
    """The task is not trivial: the straight joint-space path hits the obstacle."""
    from mfl.evaluate import direct, score
    s = S.sample(10, seed=4)
    assert score(direct(s), s, R.Checker.make())["collision_rate"] > 0.8


def test_normaliser_round_trip():
    traj = torch.randn(5, Dm.T, R.DOF, dtype=torch.float64)
    delta = traj[:, 1:] - traj[:, :1]
    norm = D.Normaliser(delta.mean(0).float(), delta.std(0).float())
    back = norm.decode(norm.encode(traj.float())[:, None], traj[:, 0].float())[:, 0]
    assert torch.allclose(back, traj.float(), atol=1e-5)


def test_point_cloud_lies_on_visible_hemisphere():
    c, r = torch.tensor([[0.4, 0.0, 0.3]], dtype=torch.float64), torch.tensor([0.08], dtype=torch.float64)
    pts = S.observe(c, r, noise=0.0)
    assert torch.allclose((pts - c).norm(dim=-1), r.expand(1, S.N_POINTS), atol=1e-9)
    view = torch.tensor(S.CAMERA) - c
    assert ((pts - c) @ view.T > 0).all()


def toy_batch(n=2048):
    """Context-free bimodal target: half the samples at +1, half at -1."""
    x = torch.cat([torch.ones(n // 2, 2), -torch.ones(n // 2, 2)]) + 0.05 * torch.randn(n, 2)
    z = torch.zeros(n, 4)
    return {"q0": z, "goal": z[:, :3], "cloud": torch.zeros(n, 4, 3), "center": z[:, :3], "radius": z[:, 0], "x": x}


def test_flow_matching_keeps_both_modes_where_regression_averages():
    torch.manual_seed(0)
    b = toy_batch()
    out = {}
    for kind in ("bc", "flow"):
        m = M.KINDS[kind](2, oracle=True)
        opt = torch.optim.Adam(m.parameters(), lr=2e-3)
        for _ in range(400):
            idx = torch.randint(0, 2048, (256,))
            loss = m.loss({k: v[idx] for k, v in b.items()})
            opt.zero_grad()
            loss.backward()
            opt.step()
        one = {k: v[:1] for k, v in b.items()}
        out[kind] = m.sample(one, n=200, **({"steps": 10} if kind == "flow" else {}))[0, :, 0]
    assert out["bc"].abs().max() < 0.3                       # regression collapses to the mean, between the modes
    assert (out["flow"] > 0.5).float().mean() > 0.25 and (out["flow"] < -0.5).float().mean() > 0.25


def test_samplers_shapes():
    b = {k: v[:3] for k, v in toy_batch(8).items()}
    for kind in ("ddpm", "flow"):
        m = M.KINDS[kind](2)
        assert m.sample(b, n=4, steps=3).shape == (3, 4, 2)
    assert M.Diffusion(2).sample(b, n=2).shape == (3, 2, 2)
