"""Figures of the README from results/summary.json and results/examples.json.

    python -m tools.figures
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from mfl import demos as Dm  # noqa: E402
from mfl import robot as R  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures"
NAVY, BLUE, AMBER, CORAL, TEAL, GREY = "#14213D", "#2B59C3", "#F2A541", "#E4572E", "#2A9D8F", "#9AA5B1"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#C8C2B6", "axes.titleweight": "bold", "axes.titlesize": 12})
LABELS = {"direct path": "Direct path", "behaviour cloning": "Behaviour\ncloning",
          "diffusion, DDPM 100 steps": "Diffusion\nDDPM 100", "diffusion, DDIM 10 steps": "Diffusion\nDDIM 10",
          "flow matching, 10 steps": "Flow matching\n10 steps"}


def top_view(ex: dict, i: int, ax, label: str, title: str):
    sc = {k: np.asarray(v[i]) for k, v in ex["scenes"].items()}
    ck = R.Checker.make()
    th = np.linspace(0, 2 * np.pi, 100)
    ax.fill(sc["center"][0] + sc["radius"] * np.cos(th), sc["center"][1] + sc["radius"] * np.sin(th),
            color=CORAL, alpha=.35, lw=0)
    ax.scatter(sc["cloud"][:, 0], sc["cloud"][:, 1], s=4, color="#7A1E05")
    for tr in np.asarray(ex["trajectories"][label][i]):
        ok = Dm.check(tr, sc["center"], float(sc["radius"]), sc["goal"], ck)["success"]
        mode = Dm.mode_of(tr, sc["center"])
        p = R.ee(torch.as_tensor(tr)).numpy()
        ax.plot(p[:, 0], p[:, 1], color=(BLUE if mode == "inside" else AMBER) if ok else GREY, lw=2 if ok else 1.4,
                ls="-" if ok else "--", alpha=.9)
    ax.scatter(*sc["start_ee"][:2], s=60, color=BLUE, zorder=5, marker="o")
    ax.scatter(*sc["goal"][:2], s=80, color=TEAL, zorder=5, marker="*")
    ax.scatter(0, 0, s=120, color=NAVY, marker="s")
    ax.set_aspect("equal")
    ax.set_xlim(-0.05, 0.75)
    ax.set_ylim(-0.6, 0.6)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])


def fig_modes(ex, scene: int = 0):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.6))
    for ax, (label, title) in zip(axes, (("behaviour cloning", "Behaviour cloning"),
                                         ("diffusion, DDIM 10 steps", "Diffusion, DDIM 10 steps"),
                                         ("flow matching, 10 steps", "Flow matching, 10 steps")), strict=True):
        top_view(ex, scene, ax, label, title)
    axes[0].text(0.02, 0.02, "robot base (square), start (dot), goal (star)\nobstacle and its observed point cloud (red)",
                 transform=axes[0].transAxes, fontsize=8.5, color="#4A5568")
    axes[2].text(0.02, 0.02, "blue: inside, amber: outside\ngrey dashed: rejected by MuJoCo", transform=axes[2].transAxes,
                 fontsize=8.5, color="#4A5568")
    fig.suptitle("Same scene, 8 samples per policy, seen from above", fontweight="bold", color=NAVY)
    fig.tight_layout()
    fig.savefig(OUT / "modes_top_view.png", dpi=160, facecolor="white")
    plt.close(fig)


def fig_main(s):
    rows = s["main"]
    keys = list(LABELS)
    x = np.arange(len(keys))
    fig, ax = plt.subplots(figsize=(10, 4.8))
    w = .27
    for k, (metric, color, name) in enumerate((("success_rate", TEAL, "valid samples"),
                                               ("collision_rate", CORAL, "collisions"),
                                               ("both_modes", AMBER, "scenes with both routes"))):
        vals = [rows[key][metric] for key in keys]
        bars = ax.bar(x + (k - 1) * w, vals, w, color=color, label=name)
        for b, v in zip(bars, vals, strict=True):
            ax.text(b.get_x() + b.get_width() / 2, v + .015, f"{100 * v:.0f}%", ha="center", fontsize=8.5)
    ax.set_xticks(x, [LABELS[k] for k in keys])
    ax.set_ylim(0, 1.1)
    ax.set_yticks([0, .25, .5, .75, 1], ["0", "25%", "50%", "75%", "100%"])
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.17))
    ax.set_title(f"{s['test_scenes']} unseen scenes, 8 samples each, every sample checked in MuJoCo")
    fig.tight_layout()
    fig.savefig(OUT / "results.png", dpi=160, facecolor="white")
    plt.close(fig)


def fig_steps(s):
    fig, ax = plt.subplots(figsize=(6.8, 4))
    for key, color, name in (("flow", BLUE, "flow matching (Euler)"), ("ddim", AMBER, "diffusion (DDIM)")):
        st = sorted(int(k) for k in s["sweep"][key])
        ax.plot(st, [s["sweep"][key][str(k)]["success_rate"] for k in st], "-o", color=color, lw=2, label=name)
    d = s["main"]["diffusion, DDPM 100 steps"]["success_rate"]
    ax.axhline(d, color=GREY, ls="--", lw=1.2)
    ax.text(1.05, d + .01, "diffusion, DDPM 100 steps", color=GREY, fontsize=9)
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8, 16, 32], ["1", "2", "4", "8", "16", "32"])
    ax.set_xlabel("network evaluations per sample")
    ax.set_ylabel("valid samples")
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0, .25, .5, .75, 1], ["0", "25%", "50%", "75%", "100%"])
    ax.legend(frameon=False, loc="lower right")
    ax.set_title("Quality against inference budget")
    fig.tight_layout()
    fig.savefig(OUT / "steps.png", dpi=160, facecolor="white")
    plt.close(fig)


def fig_perception(s):
    p = s["perception"]
    keys = ["ground-truth obstacle (oracle)", "point cloud (4 mm noise)", "point cloud (10 mm noise)",
            "point cloud (20 mm noise)", "point cloud (40 mm noise)", "point cloud of another scene"]
    names = ["oracle\n(centre, radius)", "4 mm", "10 mm", "20 mm", "40 mm", "cloud of\nanother scene"]
    fig, ax = plt.subplots(figsize=(7.6, 4))
    vals = [p[k]["success_rate"] for k in keys]
    bars = ax.bar(names, vals, color=[GREY] + [BLUE] * 4 + [CORAL])
    for b, v in zip(bars, vals, strict=True):
        ax.text(b.get_x() + b.get_width() / 2, v + .015, f"{100 * v:.0f}%", ha="center", fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.set_yticks([0, .25, .5, .75, 1], ["0", "25%", "50%", "75%", "100%"])
    ax.set_ylabel("valid samples")
    ax.set_xlabel("depth noise on the observed points (trained at 4 mm), and a shuffled-cloud control")
    ax.set_title("Flow matching from a point cloud")
    fig.tight_layout()
    fig.savefig(OUT / "perception.png", dpi=160, facecolor="white")
    plt.close(fig)


def fig_training(s):
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    for name, color in (("bc", GREY), ("ddpm", AMBER), ("flow", BLUE), ("flow_oracle", TEAL)):
        lg = s["train_logs"][name]
        ax.plot([r["step"] for r in lg], [r["loss"] for r in lg], color=color, lw=2, label=name)
    ax.set_yscale("log")
    ax.set_xlabel("training step")
    ax.set_ylabel("loss (own objective)")
    ax.legend(frameon=False)
    ax.set_title("Training curves")
    fig.tight_layout()
    fig.savefig(OUT / "training.png", dpi=160, facecolor="white")
    plt.close(fig)


def main(scene: int = 0):
    OUT.mkdir(parents=True, exist_ok=True)
    s = json.loads((ROOT / "results" / "summary.json").read_text())
    ex = json.loads((ROOT / "results" / "examples.json").read_text())
    fig_modes(ex, scene)
    fig_main(s)
    fig_steps(s)
    fig_perception(s)
    fig_training(s)


if __name__ == "__main__":
    main()
