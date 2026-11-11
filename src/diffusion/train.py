"""Standard DDPM training: predict the added noise (no masks are used). Resumable + EMA."""
import copy
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .schedule import Diffusion


@torch.no_grad()
def _ema_update(ema, model, decay):
    for pe, p in zip(ema.parameters(), model.parameters()):
        pe.mul_(decay).add_(p.detach(), alpha=1 - decay)


def train_ddpm(model, loader, ckpt_path, max_steps=100_000, lr=1e-4, ema_decay=0.999, save_every=2000,
               device="cuda", amp=False, T=1000, log_every=100):
    diff = Diffusion(T=T, device=device)
    model.to(device).train()
    ema = copy.deepcopy(model).eval()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    step, ckpt_path = 0, Path(ckpt_path)
    if ckpt_path.exists():                                  # resume after a job timeout
        ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck["model"]); ema.load_state_dict(ck["ema"]); opt.load_state_dict(ck["opt"])
        step = ck["step"]; print(f"resumed from step {step}")
    scaler = torch.amp.GradScaler(enabled=amp and device == "cuda")
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    def save():
        torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(),
                    "step": step, "cfg": model.cfg}, ckpt_path)

    t0 = time.time()
    while step < max_steps:
        for batch in loader:
            x = (batch[0] if isinstance(batch, (list, tuple)) else batch).to(device)
            t = torch.randint(0, T, (x.shape[0],), device=device)
            xt, noise = diff.q_sample(x, t)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp and device == "cuda"):
                loss = F.mse_loss(model(xt, t), noise)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            _ema_update(ema, model, ema_decay)
            step += 1
            if step % log_every == 0:
                print(f"step {step} loss {loss.item():.4f} ({(time.time()-t0)/60:.1f} min)", flush=True)
            if step % save_every == 0:
                save()
            if step >= max_steps:
                break
    save()
    return model
