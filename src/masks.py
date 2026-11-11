"""Square-hole masks (1 = missing, 0 = visible), shared by all methods."""
import numpy as np


def random_square_mask(size, ratio_range=(0.10, 0.40), rng=None):
    rng = rng or np.random.default_rng()
    area = rng.uniform(*ratio_range) * size * size
    side = int(round(area ** 0.5))
    y = rng.integers(0, size - side + 1)
    x = rng.integers(0, size - side + 1)
    m = np.zeros((size, size), dtype=np.float32)
    m[y:y + side, x:x + side] = 1.0
    return m


def composite(pred, original, mask):
    """result = m * pred + (1 - m) * original"""
    return mask * pred + (1 - mask) * original
