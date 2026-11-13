"""Method 2: StyleGAN inversion (unsupervised). Slow (an optimisation per image), stochastic.

An unconditional generator G (trained only on clean AFHQ, never on holes: scripts/m2_stylegan_inversion.py train).
For each image, optimise the style vector w so that G(w) matches the VISIBLE pixels:
    L(w) = mean_visible (G(w) - x)^2  +  lambda_p * LPIPS((1-m) G(w), (1-m) x)
then composite the hole from G(w). Different random starting styles z -> different fills (K samples).
`w_init` hook: start from a "pet code" for the optional personalisation extension.
"""
import math

import torch

from src.gan import load_generator
from .base import Inpainter


class StyleGANInversion(Inpainter):
    name = "stylegan_inversion"
    stochastic = True

    def __init__(self, G, steps=500, lr=0.05, lpips_weight=1.0, use_lpips=True, device="cuda"):
        self.G, self.device = G.to(device).eval().requires_grad_(False), device
        self.steps, self.lr, self.lpips_weight, self.use_lpips = steps, lr, lpips_weight, use_lpips
        self._lp = None

    @classmethod
    def from_ckpt(cls, path, device="cuda", **kw):
        G, cfg = load_generator(path, device)
        assert cfg["mode"] == "uncond", f"checkpoint is mode '{cfg['mode']}': inversion needs the unconditional generator"
        return cls(G, device=device, **kw)

    def _lpips(self):
        if self._lp is None:
            import lpips
            self._lp = lpips.LPIPS(net="alex", verbose=False).to(self.device).eval().requires_grad_(False)
        return self._lp

    def invert(self, masked, mask, w_init=None):
        """Optimise w on the visible pixels; returns the final image and w. `masked` has the hole zeroed."""
        B = masked.shape[0]
        vis = 1 - mask
        if w_init is None:
            w_init = self.G.mapping(torch.randn(B, self.G.z_dim, device=self.device))
        w = w_init.detach().clone().requires_grad_(True)
        opt = torch.optim.Adam([w], lr=self.lr)
        with torch.enable_grad():                       # the evaluation loop runs under no_grad; inversion needs gradients
            for i in range(self.steps):
                for g in opt.param_groups:
                    g["lr"] = self.lr * 0.5 * (1 + math.cos(math.pi * i / self.steps))
                img = self.G.synth(w)
                loss = ((img - masked) * vis).square().sum() / (3 * vis.sum() + 1e-6)
                if self.use_lpips and self.lpips_weight > 0:
                    loss = loss + self.lpips_weight * self._lpips()(img * vis, masked).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
        with torch.no_grad():
            return self.G.synth(w).clamp(-1, 1), w.detach()

    def fill(self, masked, mask, K=1):
        masked, mask = masked.to(self.device), mask.to(self.device)
        self.G.synth.set_noise(False)                     # keep the generator deterministic while fitting w
        try:
            return [self.invert(masked, mask)[0] for _ in range(K)]
        finally:
            self.G.synth.set_noise(True)
