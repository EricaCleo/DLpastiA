"""DDPM noise schedule with optional timestep re-spacing (RePaint uses e.g. 250 of 1000 steps).

Index convention: respaced "level" t = 0..N-1; abar[t] is the signal level at level t.
Level -1 means the clean image (abar_prev[0] = 1).  `timestep_map[t]` gives the original DDPM
timestep to feed the network, so a model trained with T=1000 works with fewer sampling steps.
"""
import numpy as np
import torch


def linear_betas(T=1000):
    scale = 1000 / T
    return np.linspace(scale * 1e-4, scale * 0.02, T, dtype=np.float64)


class Diffusion:
    def __init__(self, T=1000, steps=None, device="cpu"):
        betas = linear_betas(T)
        abar = np.cumprod(1.0 - betas)
        use = list(range(T)) if steps is None else sorted(
            {int(round(x)) for x in np.linspace(0, T - 1, steps)})
        last, nb = 1.0, []
        for i in use:
            nb.append(1 - abar[i] / last)
            last = abar[i]
        b = np.array(nb)
        a = 1.0 - b
        ab = np.cumprod(a)
        ab_prev = np.append(1.0, ab[:-1])
        pv = b * (1 - ab_prev) / (1 - ab)
        t = lambda x: torch.tensor(x, dtype=torch.float32, device=device)
        self.T, self.num_steps, self.device = T, len(use), device
        self.timestep_map = torch.tensor(use, dtype=torch.long, device=device)
        self.betas, self.alphas_cumprod, self.alphas_cumprod_prev = t(b), t(ab), t(ab_prev)
        self.post_logvar = t(np.log(np.append(pv[1], pv[1:])))
        self.post_c1 = t(b * np.sqrt(ab_prev) / (1 - ab))
        self.post_c2 = t((1 - ab_prev) * np.sqrt(a) / (1 - ab))

    def q_sample(self, x0, t, noise=None):
        """Forward noising at original DDPM timesteps t (used for training; steps=None)."""
        noise = torch.randn_like(x0) if noise is None else noise
        ab = self.alphas_cumprod[t].view(-1, 1, 1, 1)
        return ab.sqrt() * x0 + (1 - ab).sqrt() * noise, noise

    def undo(self, x, t):
        """RePaint forward jump: x_{t-1} -> x_t  (here t is the NEW, larger level)."""
        b = self.betas[t]
        return (1 - b).sqrt() * x + b.sqrt() * torch.randn_like(x)


def repaint_schedule(n_steps, jump_length=10, jump_n_sample=10):
    """Sequence of levels visited by RePaint (resampling jumps), ending with -1 (= clean image).
    Same logic as the official RePaint code. Total network calls ~ n_steps + jumps."""
    jumps = {j: jump_n_sample - 1 for j in range(0, n_steps - jump_length, jump_length)}
    t, ts = n_steps, []
    while t >= 1:
        t -= 1
        ts.append(t)
        if jumps.get(t, 0) > 0:
            jumps[t] -= 1
            for _ in range(jump_length):
                t += 1
                ts.append(t)
    ts.append(-1)
    return ts
