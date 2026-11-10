"""A 4-DoF arm (base yaw + 3 pitch joints): differentiable kinematics and clearance in PyTorch, ground-truth checks in
MuJoCo. The PyTorch side is what the optimiser and the policies see; MuJoCo is the referee."""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
import torch

DOF = 4
BASE_H = 0.30
LINKS = (0.35, 0.30, 0.12)
RADII = (0.035, 0.030, 0.025)
Q_LOW = np.array([-2.6, -0.3, -2.6, -2.0])
Q_HIGH = np.array([2.6, 2.2, 0.3, 2.0])
HOME = np.array([0.0, 1.25, -1.9, 0.35])
GEOMS = ("g_upper", "g_fore", "g_hand")

MJCF = f"""
<mujoco model="arm4">
  <compiler angle="radian"/>
  <option gravity="0 0 0"/>
  <worldbody>
    <geom name="floor" type="plane" size="1 1 0.01" contype="0" conaffinity="0"/>
    <geom name="obstacle" type="sphere" size="0.08" pos="0.4 0 0.3" contype="0" conaffinity="0"/>
    <body name="yaw" pos="0 0 {BASE_H}">
      <inertial pos="0 0 0" mass="0.5" diaginertia="0.001 0.001 0.001"/>
      <joint name="j0" type="hinge" axis="0 0 1" range="{Q_LOW[0]} {Q_HIGH[0]}"/>
      <body name="upper">
        <joint name="j1" type="hinge" axis="0 -1 0" range="{Q_LOW[1]} {Q_HIGH[1]}"/>
        <geom name="g_upper" type="capsule" fromto="0 0 0 {LINKS[0]} 0 0" size="{RADII[0]}" contype="0" conaffinity="0"/>
        <body name="fore" pos="{LINKS[0]} 0 0">
          <joint name="j2" type="hinge" axis="0 -1 0" range="{Q_LOW[2]} {Q_HIGH[2]}"/>
          <geom name="g_fore" type="capsule" fromto="0 0 0 {LINKS[1]} 0 0" size="{RADII[1]}" contype="0" conaffinity="0"/>
          <body name="hand" pos="{LINKS[1]} 0 0">
            <joint name="j3" type="hinge" axis="0 -1 0" range="{Q_LOW[3]} {Q_HIGH[3]}"/>
            <geom name="g_hand" type="capsule" fromto="0 0 0 {LINKS[2]} 0 0" size="{RADII[2]}" contype="0" conaffinity="0"/>
            <site name="ee" pos="{LINKS[2]} 0 0"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def joint_points(q: torch.Tensor) -> torch.Tensor:
    """Shoulder, elbow, wrist and end-effector positions, shape (..., 4, 3)."""
    yaw = q[..., 0]
    phi = torch.cumsum(q[..., 1:], dim=-1)
    lengths = torch.tensor(LINKS, dtype=q.dtype, device=q.device)
    r = torch.cumsum(lengths * torch.cos(phi), dim=-1)
    z = BASE_H + torch.cumsum(lengths * torch.sin(phi), dim=-1)
    zero = torch.zeros_like(yaw)[..., None]
    r = torch.cat([zero, r], dim=-1)
    z = torch.cat([zero + BASE_H, z], dim=-1)
    return torch.stack([r * torch.cos(yaw)[..., None], r * torch.sin(yaw)[..., None], z], dim=-1)


def ee(q: torch.Tensor) -> torch.Tensor:
    return joint_points(q)[..., -1, :]


def body_samples(q: torch.Tensor, n: int = 7) -> torch.Tensor:
    """Points along the three links, shape (..., 3 * n, 3). Used for plotting and floor checks."""
    p = joint_points(q)
    s = torch.linspace(0, 1, n, dtype=q.dtype, device=q.device)[:, None]
    seg = [p[..., k:k + 1, :] + s * (p[..., k + 1:k + 2, :] - p[..., k:k + 1, :]) for k in range(3)]
    return torch.cat(seg, dim=-2)


def link_clearances(q: torch.Tensor, center: torch.Tensor, radius: torch.Tensor) -> torch.Tensor:
    """Exact signed distance between each capsule link and the sphere, shape (B, T, 3).
    q: (B, T, 4), center: (B, 3), radius: (B,)."""
    p = joint_points(q)                                     # (B, T, 4, 3)
    a, b = p[..., :-1, :], p[..., 1:, :]                    # (B, T, 3, 3)
    c = center[:, None, None, :]
    ab = b - a
    t = (((c - a) * ab).sum(-1) / (ab * ab).sum(-1).clamp_min(1e-12)).clamp(0, 1)
    closest = a + t[..., None] * ab
    d = (c - closest).norm(dim=-1)
    rad = torch.tensor(RADII, dtype=q.dtype, device=q.device)
    return d - rad - radius[:, None, None]


def clearance(q: torch.Tensor, center: torch.Tensor, radius: torch.Tensor) -> torch.Tensor:
    """Smallest arm-obstacle distance per configuration, shape (B, T)."""
    return link_clearances(q, center, radius).min(dim=-1).values


def floor_margin(q: torch.Tensor) -> torch.Tensor:
    """Lowest point of the link surfaces above the table, shape (...)."""
    p = joint_points(q)[..., 1:, 2]
    return (p - torch.tensor(RADII, dtype=q.dtype, device=q.device)).min(dim=-1).values


@dataclass
class Checker:
    """MuJoCo referee: forward kinematics and true geometric distances (mj_geomDistance)."""
    model: mujoco.MjModel
    data: mujoco.MjData

    @classmethod
    def make(cls) -> "Checker":
        m = mujoco.MjModel.from_xml_string(MJCF)
        ck = cls(m, mujoco.MjData(m))
        ck._obs = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "obstacle")
        ck._links = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, g) for g in GEOMS]
        ck._site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "ee")
        return ck

    def set_obstacle(self, center, radius: float) -> None:
        self.model.geom_pos[self._obs] = np.asarray(center, dtype=float)
        self.model.geom_size[self._obs, 0] = float(radius)
        self.model.geom_rbound[self._obs] = float(radius)

    def _fk(self, q) -> None:
        self.data.qpos[:] = q
        mujoco.mj_kinematics(self.model, self.data)

    def ee(self, q) -> np.ndarray:
        self._fk(q)
        return self.data.site_xpos[self._site].copy()

    def min_distance(self, q) -> float:
        self._fk(q)
        return min(mujoco.mj_geomDistance(self.model, self.data, g, self._obs, 10.0, None) for g in self._links)

    def floor_ok(self, q) -> bool:
        self._fk(q)
        for g, r in zip(self._links, RADII, strict=True):
            a = self.data.geom_xpos[g]
            axis = self.data.geom_xmat[g].reshape(3, 3)[:, 2]
            half = self.model.geom_size[g, 1]
            if min(a[2] + half * axis[2], a[2] - half * axis[2]) - r < 0:
                return False
        return True

    def trajectory_ok(self, traj, substeps: int = 4) -> tuple[bool, float]:
        """Checks the trajectory on linearly interpolated substeps. Returns (collision-free, worst distance)."""
        traj = np.asarray(traj, dtype=float)
        worst, ok = np.inf, True
        for k in range(len(traj) - 1):
            for s in range(substeps):
                q = traj[k] + (traj[k + 1] - traj[k]) * s / substeps
                d = self.min_distance(q)
                worst = min(worst, d)
                if d < 0 or not self.floor_ok(q):
                    ok = False
        d = self.min_distance(traj[-1])
        worst = min(worst, d)
        ok = ok and d >= 0 and self.floor_ok(traj[-1])
        return ok, float(worst)
