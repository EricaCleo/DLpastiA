"""Common interface so every method is scored by the same evaluation code."""
from abc import ABC, abstractmethod


class Inpainter(ABC):
    name = "base"
    stochastic = False  # True -> evaluation draws K fills per image

    @abstractmethod
    def fill(self, masked, mask, K=1):
        """masked: (B,3,H,W) in [-1,1] with hole zeroed; mask: (B,1,H,W), 1 = hole.
        Returns a list of K tensors (B,3,H,W) in [-1,1]. Only the hole region is scored;
        the visible region is replaced by the original during evaluation."""
