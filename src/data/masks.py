"""Hole masks (1 = missing, 0 = visible), shared by all methods."""
import numpy as np


def random_square_mask(size, ratio_range=(0.10, 0.40), rng=None):
    """Random square covering ratio_range of the image area."""
    rng = rng or np.random.default_rng()
    area = rng.uniform(*ratio_range) * size * size
    side = int(round(area ** 0.5))
    y = rng.integers(0, size - side + 1)
    x = rng.integers(0, size - side + 1)
    m = np.zeros((size, size), dtype=np.float32)
    m[y:y + side, x:x + side] = 1.0
    return m


def one_sided_mask(size, side="left", ratio_range=(0.10, 0.25), rng=None):
    """Square placed entirely in the left or right half (one eye / one ear).
    Used to test the mirror methods, where the other half stays visible."""
    rng = rng or np.random.default_rng()
    area = rng.uniform(*ratio_range) * size * size
    s = min(int(round(area ** 0.5)), size // 2)
    y = rng.integers(0, size - s + 1)
    x0, x1 = (0, size // 2 - s) if side == "left" else (size // 2, size - s)
    x = rng.integers(x0, x1 + 1)
    m = np.zeros((size, size), dtype=np.float32)
    m[y:y + s, x:x + s] = 1.0
    return m


def centre_mask(size, ratio=0.2):
    """Square in the centre (nose/mouth): control case where the mirror adds little."""
    s = int(round((ratio * size * size) ** 0.5))
    o = (size - s) // 2
    m = np.zeros((size, size), dtype=np.float32)
    m[o:o + s, o:o + s] = 1.0
    return m


def composite(pred, original, mask):
    """result = m * pred + (1 - m) * original"""
    return mask * pred + (1 - mask) * original
