"""MAE fine-tuning loop (self-supervised: random 75% patch masking, MSE on hidden patches). Resumable."""
import math
import time
from pathlib import Path

import torch

from .mae import save_mae, to_mae_input


def train_mae(model, loader, ckpt_path, epochs=40, lr=1.5e-4, weight_decay=0.05, warmup_epochs=2,
              mask_ratio=0.75, device="cuda", amp=True, log_every=50):
    model.to(device).train()
    model.config.mask_ratio = mask_ratio
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=weight_decay)
    steps_per_epoch = len(loader)
    total, warm = epochs * steps_per_epoch, warmup_epochs * steps_per_epoch
    ckpt_path, start_epoch = Path(ckpt_path), 0
    if ckpt_path.exists():                                   # resume after a job time limit
        ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); start_epoch = ck["epoch"]
        print(f"resumed from epoch {start_epoch}")
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    use_amp = amp and device == "cuda"
    t0 = time.time()
    for epoch in range(start_epoch, epochs):
        for i, batch in enumerate(loader):
            step = epoch * steps_per_epoch + i
            cur = lr * (step / warm if step < warm else 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm))))
            for g in opt.param_groups:
                g["lr"] = cur
            x = (batch[0] if isinstance(batch, (list, tuple)) else batch).to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                loss = model(pixel_values=to_mae_input(x)).loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if i % log_every == 0:
                print(f"epoch {epoch + 1}/{epochs} step {i}/{steps_per_epoch} loss {loss.item():.4f} lr {cur:.2e} ({(time.time() - t0) / 60:.1f} min)", flush=True)
        save_mae(model, ckpt_path, {"opt": opt.state_dict(), "epoch": epoch + 1})
    return model
