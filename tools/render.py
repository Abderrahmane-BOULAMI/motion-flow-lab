"""Exports example scenes to JSON for the three.js viewer, then renders comparison images and an animated GIF with a
headless browser (Playwright). Needs node + playwright and three.js under tools/node_modules (npm i three@0.160.0).

    python -m tools.render
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from urllib.parse import urlencode

import numpy as np
import torch

from mfl import demos as Dm
from mfl import robot as R

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
OUT = ROOT / "docs" / "figures"
CACHE = TOOLS / "render_cache"


def scene_json(ex: dict, i: int, label: str, highlight: int | None = None) -> dict:
    sc = {k: np.asarray(v[i]) for k, v in ex["scenes"].items()}
    trajs = np.asarray(ex["trajectories"][label][i])
    ck = R.Checker.make()
    paths, ok_idx = [], []
    for j, tr in enumerate(trajs):
        res = Dm.check(tr, sc["center"], float(sc["radius"]), sc["goal"], ck)
        mode = Dm.mode_of(tr, sc["center"]) if res["success"] else "fail"
        paths.append({"ee": R.ee(torch.as_tensor(tr)).numpy().round(4).tolist(), "mode": mode})
        if res["success"]:
            ok_idx.append(j)
    h = highlight if highlight is not None else (ok_idx[0] if ok_idx else 0)
    arm = R.joint_points(torch.as_tensor(trajs[h])).numpy().round(4).tolist()
    return {"scene": {"center": sc["center"].tolist(), "radius": float(sc["radius"]), "cloud": sc["cloud"].tolist(),
                      "start_ee": sc["start_ee"].tolist(), "goal": sc["goal"].tolist()},
            "paths": paths, "arm": arm, "n_ok": len(ok_idx), "n": len(trajs)}


SHOOT = r"""
const { chromium } = require('playwright');
const jobs = JSON.parse(require('fs').readFileSync(process.argv[2], 'utf8'));
(async () => {
  const b = await chromium.launch({ args: ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'] });
  const p = await b.newPage({ viewport: { width: 900, height: 700 } });
  for (const j of jobs) {
    await p.goto('http://localhost:8766/viewer.html?' + j.q);
    await p.waitForFunction(() => document.title === 'ready', { timeout: 60000 });
    await p.screenshot({ path: j.out });
  }
  await b.close();
})();
"""


def shoot(jobs: list[dict]) -> None:
    CACHE.mkdir(exist_ok=True)
    (CACHE / "jobs.json").write_text(json.dumps(jobs))
    (CACHE / "shoot.js").write_text(SHOOT)
    server = subprocess.Popen(["python3", "-m", "http.server", "8766"], cwd=TOOLS, stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL)
    try:
        time.sleep(1.0)
        subprocess.run(["node", str(CACHE / "shoot.js"), str(CACHE / "jobs.json")], check=True)
    finally:
        server.terminate()
        server.wait()


def main(scene: int = 0) -> None:
    from PIL import Image
    ex = json.loads((ROOT / "results" / "examples.json").read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(exist_ok=True)
    labels = {"behaviour cloning": "Behaviour cloning", "diffusion, DDIM 10 steps": "Diffusion (DDIM, 10 steps)",
              "flow matching, 10 steps": "Flow matching (10 steps)"}
    jobs = []
    for k, (label, title) in enumerate(labels.items()):
        js = scene_json(ex, scene, label)
        (CACHE / f"s{scene}_{k}.json").write_text(json.dumps(js))
        sub = f"{js['n_ok']} of {js['n']} samples valid"
        jobs.append({"q": urlencode({"data": f"render_cache/s{scene}_{k}.json", "model": title, "sub": sub}),
                     "out": str(CACHE / f"cmp_{k}.png")})
    js = scene_json(ex, scene, "flow matching, 10 steps")
    (CACHE / "anim.json").write_text(json.dumps(js))
    for f in range(Dm.T):
        jobs.append({"q": urlencode({"data": "render_cache/anim.json", "model": "Flow matching, 10 steps",
                                     "sub": "blue: inside, amber: outside, grey: rejected", "frame": f}),
                     "out": str(CACHE / f"anim_{f:02d}.png")})
    shoot(jobs)
    ims = [Image.open(CACHE / f"cmp_{k}.png").convert("RGB") for k in range(3)]
    w, h = ims[0].size
    strip = Image.new("RGB", (3 * w, h), "white")
    for k, im in enumerate(ims):
        strip.paste(im, (k * w, 0))
    strip.save(OUT / "comparison.png")
    frames = [Image.open(CACHE / f"anim_{f:02d}.png").convert("RGB").resize((630, 490)) for f in range(Dm.T)]
    frames = frames + frames[::-1][:6]
    frames[0].save(OUT / "flow_rollout.gif", save_all=True, append_images=frames[1:], duration=90, loop=0)
    frames[-7].save(OUT / "hero.png")


if __name__ == "__main__":
    import sys
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
