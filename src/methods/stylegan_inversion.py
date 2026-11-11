"""Method 2: StyleGAN2 inversion (unsupervised). Slow (optimisation per image), stochastic.

Plan: pretrained StyleGAN2-ADA AFHQ generator G (never trained on holes). For each image optimise
w to minimise  ||(1-m)*(G(w)-x)||^2 + lambda_p * LPIPS((1-m)*G(w), (1-m)*x)  for ~500-1000 steps,
then composite the hole region. Different initial w -> different fills.
"""
from .base import Inpainter


class StyleGANInversion(Inpainter):
    name = "stylegan_inversion"
    stochastic = True

    def __init__(self, generator_pkl=None, steps=800, lr=0.05, lpips_weight=1.0, device="cuda"):
        self.steps, self.lr, self.lpips_weight = steps, lr, lpips_weight
        # TODO: load StyleGAN2-ADA AFHQ pkl (needs the official repo's torch_utils)

    def invert(self, masked, mask, w_init=None):
        """Return the optimised latent w. `w_init` = pet code for the personalisation extension."""
        raise NotImplementedError

    def fill(self, masked, mask, K=1):
        raise NotImplementedError
