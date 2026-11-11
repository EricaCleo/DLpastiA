"""Trivial baselines: sanity-check the eval pipeline and give lower bounds."""
import torch

from .base import Inpainter


class GreyFill(Inpainter):
    name = "grey_fill"

    def fill(self, masked, mask, K=1):
        return [torch.zeros_like(masked)]


class MirrorPaste(Inpainter):
    """Copy the horizontally flipped visible image into the hole (no learning).
    Shows how much raw left-right symmetry is worth, a useful reference for the mirror methods."""
    name = "mirror_paste"

    def fill(self, masked, mask, K=1):
        flipped = torch.flip(masked, dims=[3])
        use = mask * torch.flip(1 - mask, dims=[3])  # hole pixel whose mirror is visible
        return [use * flipped]
