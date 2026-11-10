"""Training loop shared by the three policies: AdamW, one-cycle schedule, gradient clipping, EMA of the weights."""
from __future__ import annotations

import copy
import time

import torch

from mfl import models as M


def train(kind: str, batch: dict, x_dim: int, steps: int = 12000, bs: int = 256, lr: float = 1e-3,
          oracle: bool = False, seed: int = 0, log_every: int = 500):
    torch.manual_seed(seed)
    model = M.KINDS[kind](x_dim, oracle=oracle)
    ema = copy.deepcopy(model).requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05)
    n = len(batch["x"])
    log, running, t0 = [], 0.0, time.time()
    for step in range(1, steps + 1):
        idx = torch.randint(0, n, (bs,))
        loss = model.loss({k: v[idx] for k, v in batch.items()})
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        with torch.no_grad():
            decay = min(0.999, (1 + step) / (10 + step))
            for pe, pm in zip(ema.parameters(), model.parameters(), strict=True):
                pe.mul_(decay).add_(pm.detach(), alpha=1 - decay)
        running += loss.item()
        if step % log_every == 0:
            log.append({"step": step, "loss": running / log_every, "seconds": round(time.time() - t0, 1)})
            print(f"[{kind}{'/oracle' if oracle else ''}] step {step} loss {running / log_every:.4f}", flush=True)
            running = 0.0
    return ema.eval(), log
