"""Method 5: RePaint (unsupervised, zero-shot inpainting with an unconditional DDPM). Slow, stochastic.

Training: standard DDPM noise-prediction loss on clean AFHQ (no masks): see scripts/train_ddpm.py.
Sampling (Lugmayr et al., 2022). At each reverse step from level t:
  unknown part <- one reverse DDPM step of the model          (invented content)
  known part   <- real pixels noised to level t-1             (correct content)
  x_{t-1}      = hole * unknown + (1 - hole) * known
plus resampling: every `jump_length` levels, jump back (add noise) and denoise again,
`resamples` times, so hole and surroundings harmonise.
Mask convention here: mask = 1 in the hole (the paper's mask is the opposite).
"""
import torch

from src.diffusion import Diffusion, repaint_schedule
from .base import Inpainter


class RePaint(Inpainter):
    name = "repaint"
    stochastic = True

    def __init__(self, model, T_train=1000, steps=250, jump_length=10, resamples=10, clip=True, device="cuda"):
        self.model = model.to(device).eval()
        self.diff = Diffusion(T=T_train, steps=steps, device=device)
        self.schedule = repaint_schedule(self.diff.num_steps, jump_length, resamples)
        self.clip, self.device = clip, device

    def guide(self, x0_hat, gt, keep, t=None):
        """Hook to modify the model's clean-image estimate each step (identity for plain RePaint). t = current level."""
        return x0_hat

    def _reverse_step(self, x, t, gt, keep):
        d = self.diff
        tt = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
        eps = self.model(x, d.timestep_map[tt])
        ab = d.alphas_cumprod[t]
        x0 = (x - (1 - ab).sqrt() * eps) / ab.sqrt()
        if self.clip:
            x0 = x0.clamp(-1, 1)
        x0 = self.guide(x0, gt, keep, t)
        mean = d.post_c1[t] * x0 + d.post_c2[t] * x
        noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
        unknown = mean + (0.5 * d.post_logvar[t]).exp() * noise
        abp = d.alphas_cumprod_prev[t]                       # known pixels noised to level t-1
        known = abp.sqrt() * gt + (1 - abp).sqrt() * torch.randn_like(gt)
        return keep * known + (1 - keep) * unknown

    @torch.no_grad()
    def sample(self, gt, mask):
        keep = 1 - mask
        x = torch.randn_like(gt)
        times = self.schedule
        for t_last, t_cur in zip(times[:-1], times[1:]):
            if t_cur < t_last:
                x = self._reverse_step(x, t_last, gt, keep)
            else:                                            # resampling jump: add noise, go to level t_cur
                x = self.diff.undo(x, t_cur)
        return x

    @torch.no_grad()
    def fill(self, masked, mask, K=1):
        masked, mask = masked.to(self.device), mask.to(self.device)
        return [self.sample(masked, mask) for _ in range(K)]
