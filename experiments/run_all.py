"""Full experiment: demonstrations, training of the three policies, evaluation on unseen scenes, ablations.

    python -m experiments.run_all            # about 30 min on a laptop CPU
    python -m experiments.run_all --quick    # smoke run, a few minutes
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from mfl import data as D
from mfl import evaluate as E
from mfl import robot as R
from mfl import scenes as S
from mfl.train import train

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"


def main(quick: bool = False) -> None:
    (RES / "model").mkdir(parents=True, exist_ok=True)
    n_train, n_test, steps = (300, 40, 1500) if quick else (2500, 200, 8000)
    tag = "_quick" if quick else ""
    ck = R.Checker.make()
    t0 = time.time()
    data_path = RES / f"data{tag}.pt"
    if data_path.exists():
        bundle = D.load(data_path)
    else:
        bundle = {"train": D.build(n_train, 11, ck), "test": D.build(n_test, 99, ck)}
        D.save(bundle, data_path)
    print(f"data ready: {bundle['train']['kept']} train scenes, {bundle['test']['kept']} test scenes", flush=True)
    tr, te = bundle["train"], bundle["test"]["scenes"]
    batch, norm = D.training_set(tr)
    x_dim = batch["x"].shape[1]
    summary = {"train_scenes": tr["kept"], "train_requested": tr["requested"],
               "train_trajectories": int(batch["x"].shape[0]), "test_scenes": len(te), "steps": steps,
               "waypoints": 32, "dof": R.DOF}

    models, logs = {}, {}
    for name, kind, oracle in (("bc", "bc", False), ("flow", "flow", False), ("ddpm", "ddpm", False),
                               ("flow_oracle", "flow", True)):
        path = RES / "model" / f"{name}{tag}.pt"
        if path.exists():
            m = torch.load(path, weights_only=False)
            logs[name] = json.loads(path.with_suffix(".json").read_text())
        else:
            m, logs[name] = train(kind, batch, x_dim, steps=steps, oracle=oracle)
            torch.save(m, path)
            path.with_suffix(".json").write_text(json.dumps(logs[name]))
        models[name] = m.eval()

    def run(name, steps_=None, n=8, scenes=te):
        trajs = E.generate(models[name], norm, scenes, n, steps_)
        r = E.score(trajs, scenes, ck)
        r.pop("per_sample_ok")
        r["latency_ms"] = E.latency(models[name], norm, scenes, n, steps_)
        print(f"  {name} steps={steps_}: success {r['success_rate']:.3f}", flush=True)
        return r, trajs

    main_rows = {"direct path": E.score(E.direct(te), te, ck)}
    main_rows["direct path"].pop("per_sample_ok")
    main_rows["direct path"]["latency_ms"] = 0.0
    examples = {}
    for label, name, st in (("behaviour cloning", "bc", None), ("diffusion, DDPM 100 steps", "ddpm", None),
                            ("diffusion, DDIM 10 steps", "ddpm", 10), ("flow matching, 10 steps", "flow", 10)):
        main_rows[label], trajs = run(name, st)
        examples[label] = trajs[:12].tolist()
    sweep = {"flow": {}, "ddim": {}}
    for st in (1, 2, 4, 8, 16, 32):
        sweep["flow"][st], _ = run("flow", st)
    for st in (2, 4, 8, 16, 32):
        sweep["ddim"][st], _ = run("ddpm", st)
    perception = {"point cloud (4 mm noise)": main_rows["flow matching, 10 steps"]}
    perception["ground-truth obstacle (oracle)"], _ = run("flow_oracle", 10)
    for noise in (0.01, 0.02, 0.04):
        noisy = te.subset(torch.arange(len(te)))
        noisy.cloud = S.observe(te.center, te.radius, noise=noise, gen=torch.Generator().manual_seed(5))
        perception[f"point cloud ({int(noise * 1000)} mm noise)"], _ = run("flow", 10, scenes=noisy)
    shuffled = te.subset(torch.arange(len(te)))  # sanity check: is the observation used at all?
    shuffled.cloud = te.cloud[torch.randperm(len(te), generator=torch.Generator().manual_seed(0))]
    perception["point cloud of another scene"], _ = run("flow", 10, scenes=shuffled)

    summary.update({"main": main_rows, "sweep": sweep, "perception": perception, "train_logs": logs,
                    "minutes": round((time.time() - t0) / 60, 1)})
    (RES / f"summary{tag}.json").write_text(json.dumps(summary, indent=2, default=float))
    ex = {"scenes": {f: getattr(te, f)[:12].tolist() for f in te.__dataclass_fields__}, "trajectories": examples}
    (RES / f"examples{tag}.json").write_text(json.dumps(ex))
    for k, v in main_rows.items():
        print(f"{k:28s} success {v['success_rate']:.3f}  collisions {v['collision_rate']:.3f}  "
              f"both modes {v['both_modes']:.2f}  goal {v['goal_error_mm']:.1f} mm  {v['latency_ms']:.1f} ms")
    print("minutes", summary["minutes"])


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    main(p.parse_args().quick)
