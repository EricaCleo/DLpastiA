"""GAN training (Methods 2, 3, 4). Budget is wall-clock hours; resumable from the checkpoint.

mode 'uncond' : unconditional StyleGAN2-style generator (Method 2, used for inversion). Loss: non-saturating + R1.
mode 'comod'  : supervised inpainting generator, input = masked image + mask (Method 3).
mode 'mirror' : same, plus the flipped masked image and mask (Method 4); sym_w > 0 adds the symmetry loss (04b).
Supervised modes: discriminator sees (composited image, mask); G loss = adversarial + L1 (hole) + LPIPS (+ symmetry).
"""
import copy
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .inputs import INPUTS
from .networks import CoModGenerator, Discriminator, UncondGenerator


def build_models(cfg):
    if cfg["mode"] == "uncond":
        G = UncondGenerator(cfg["size"], cfg["z_dim"], cfg["w_dim"], cfg["base"], cfg["cap"], cfg["map_layers"])
        D = Discriminator(cfg["size"], 3, cfg["base"], cfg["cap"])
    else:
        G = CoModGenerator(cfg["size"], INPUTS[cfg["mode"]][1], cfg["z_dim"], cfg["w_dim"], cfg["base"], cfg["cap"], cfg["map_layers"])
        D = Discriminator(cfg["size"], 4, cfg["base"], cfg["cap"])
    return G, D


def load_generator(path, device="cpu"):
    """Returns (G_ema in eval mode, cfg)."""
    ck = torch.load(path, map_location=device)
    G, _ = build_models(ck["cfg"])
    G.load_state_dict(ck["G_ema"])
    return G.to(device).eval(), ck["cfg"]


def _random_flip(x, m, xm):
    f = torch.rand(x.shape[0], device=x.device) < 0.5
    sel = lambda t: torch.where(f.view(-1, 1, 1, 1), torch.flip(t, [3]), t)
    return sel(x), sel(m), sel(xm)


def train_gan(cfg, loader, ckpt_path, max_hours=22.0, device="cuda", lr=2e-3, r1_gamma=1.0, adv_w=1.0, l1_w=1.0,
              lpips_w=1.0, sym_w=0.0, flip=True, ema_kimg=10.0, save_every=1000, sample_every=2000, log_every=100,
              sample_dir=None, r1_every=16, max_steps=None):
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    mode, supervised = cfg["mode"], cfg["mode"] != "uncond"
    build_input = INPUTS[mode][0] if supervised else None
    G, D = build_models(cfg)
    G, D = G.to(device), D.to(device)
    G_ema = copy.deepcopy(G).eval().requires_grad_(False)
    optG = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.0, 0.99), eps=1e-8)
    optD = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.0, 0.99), eps=1e-8)
    lp = None
    if supervised and lpips_w > 0:
        import lpips
        lp = lpips.LPIPS(net="alex", verbose=False).to(device).eval().requires_grad_(False)

    step, elapsed, ckpt_path = 0, 0.0, Path(ckpt_path)
    if ckpt_path.exists():                                   # resume after a job time limit
        ck = torch.load(ckpt_path, map_location=device)
        G.load_state_dict(ck["G"]); D.load_state_dict(ck["D"]); G_ema.load_state_dict(ck["G_ema"])
        optG.load_state_dict(ck["optG"]); optD.load_state_dict(ck["optD"]); step, elapsed = ck["step"], ck["elapsed"]
        print(f"resumed from step {step} ({elapsed:.2f} h already trained)", flush=True)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    if sample_dir:
        Path(sample_dir).mkdir(parents=True, exist_ok=True)

    def save():
        torch.save({"G": G.state_dict(), "D": D.state_dict(), "G_ema": G_ema.state_dict(), "optG": optG.state_dict(),
                    "optD": optD.state_dict(), "step": step, "elapsed": elapsed, "cfg": cfg}, ckpt_path)

    def make_fake(x, m, xm):
        z = torch.randn(x.shape[0], cfg["z_dim"], device=device)
        if supervised:
            y = G(build_input(xm, m), z)
            comp = m * y + (1 - m) * x
            return y, comp, torch.cat([comp, m], 1)
        y = G(z)
        return y, y, y

    t_start, t_seen = time.time(), elapsed
    it = iter(loader)
    log = {}
    while True:
        try:
            x, m, xm = next(it)
        except StopIteration:
            it = iter(loader); x, m, xm = next(it)
        x, m, xm = x.to(device), m.to(device), xm.to(device)
        if flip:
            x, m, xm = _random_flip(x, m, xm)
        real_in = torch.cat([x, m], 1) if supervised else x

        # ---- discriminator
        D.requires_grad_(True)
        with torch.no_grad():
            _, _, fake_in = make_fake(x, m, xm)
        lossD = F.softplus(D(fake_in)).mean() + F.softplus(-D(real_in)).mean()
        optD.zero_grad(set_to_none=True)
        lossD.backward()
        if step % r1_every == 0:                              # lazy R1 regularisation on real images
            r = real_in.detach().requires_grad_(True)
            grad, = torch.autograd.grad(D(r).sum(), r, create_graph=True)
            r1 = grad.square().sum([1, 2, 3]).mean()
            (r1 * (r1_gamma / 2) * r1_every).backward()
            log["r1"] = r1.item()
        optD.step()

        # ---- generator
        D.requires_grad_(False)
        y, comp, fake_in = make_fake(x, m, xm)
        adv = F.softplus(-D(fake_in)).mean()
        lossG = adv_w * adv
        parts = {"adv": adv.item()}
        if supervised:
            if l1_w > 0:
                l1 = ((y - x).abs() * m).sum() / (3 * m.sum() + 1e-6)
                lossG = lossG + l1_w * l1; parts["l1"] = l1.item()
            if lp is not None:
                pl = lp(comp, x).mean()
                lossG = lossG + lpips_w * pl; parts["lpips"] = pl.item()
            if sym_w > 0:
                sym = (comp - torch.flip(comp, [3])).abs().mean()
                lossG = lossG + sym_w * sym; parts["sym"] = sym.item()
        optG.zero_grad(set_to_none=True)
        lossG.backward()
        optG.step()

        # ---- EMA of G
        beta = min(0.5 ** (x.shape[0] / (ema_kimg * 1000)), (1 + step) / (10 + step))
        with torch.no_grad():
            for pe, p in zip(G_ema.parameters(), G.parameters()):
                pe.lerp_(p, 1 - beta)
        step += 1
        elapsed = t_seen + (time.time() - t_start) / 3600

        if step % log_every == 0:
            print(f"step {step} D {lossD.item():.3f} " + " ".join(f"{k} {v:.3f}" for k, v in {**parts, **log}.items())
                  + f" | {elapsed:.2f} h | {step * x.shape[0] / 1000:.0f} kimg", flush=True)
        if sample_dir and step % sample_every == 0:
            from torchvision.utils import save_image
            with torch.no_grad():
                z = torch.randn(x.shape[0], cfg["z_dim"], device=device)
                out = G_ema(build_input(xm, m), z) if supervised else G_ema(z)
                n = min(8, x.shape[0])
                rows = [xm[:n], (m * out + (1 - m) * x)[:n], x[:n]] if supervised else [out[:n]]
                save_image((torch.cat(rows) + 1) / 2, Path(sample_dir) / f"step_{step:07d}.png", nrow=n)
        if step % save_every == 0:
            save()
        if max_steps is not None and step >= max_steps:
            break
        if elapsed >= max_hours:
            print(f"time budget of {max_hours} h reached at step {step}", flush=True)
            break
    save()
    return G_ema
